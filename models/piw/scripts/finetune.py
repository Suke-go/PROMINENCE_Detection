"""
PIW Fine-tuning Script

Fine-tuning pretrained model for prominence detection.

Usage:
    python scripts/finetune.py --config configs/finetune_prominence.yaml --pretrained outputs/pretrain/final_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import PIWModel
from dataset import HelsinkiProsodyDataset, create_dataloader
from heads.prominence import ProminenceHead

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Merge with base
    if "defaults" in config:
        base_names = config.pop("defaults")
        config_dir = Path(config_path).parent
        
        base_config = {}
        for base_name in base_names:
            base_path = config_dir / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path, "r") as f:
                    base = yaml.safe_load(f)
                    for k, v in base.items():
                        if k not in base_config:
                            base_config[k] = v
                        elif isinstance(v, dict):
                            base_config[k].update(v)
        
        for k, v in config.items():
            if k in base_config and isinstance(v, dict):
                base_config[k].update(v)
            else:
                base_config[k] = v
        
        config = base_config
    
    return config


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                waveform=waveform,
                labels=labels,
                apply_mask=False,
            )
            
            total_loss += outputs["prominence_loss"].item()
            n_batches += 1
            
            all_preds.append(outputs["prominence_preds"].cpu())
            all_labels.append(labels.cpu())
    
    # Compute metrics
    preds = torch.cat(all_preds, dim=0).flatten()
    labels = torch.cat(all_labels, dim=0).flatten()
    
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "loss": total_loss / n_batches,
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def train(config: dict, args):
    """Main fine-tuning loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        model = PIWModel.from_pretrained(
            args.pretrained,
            pretraining=False,
            finetuning=True,
        )
    else:
        # Train from scratch
        model_config = config.get("model", {})
        model = PIWModel(
            frontend_dim=model_config.get("frontend_dim", 768),
            encoder_layers=model_config.get("encoder_layers", 6),
            encoder_heads=model_config.get("encoder_heads", 8),
            pretraining=False,
            finetuning=True,
        )
    
    model = model.to(device)
    
    # Dataset
    data_config = config.get("data", {})
    data_root = args.data_root or os.environ.get("HELSINKI_PROSODY_ROOT", "./data/helsinki")
    
    train_dataset = HelsinkiProsodyDataset(
        root=data_root,
        split="train",
        sample_rate=data_config.get("sample_rate", 16000),
    )
    
    val_dataset = HelsinkiProsodyDataset(
        root=data_root,
        split="val",
        sample_rate=data_config.get("sample_rate", 16000),
    )
    
    train_config = config.get("training", {})
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=False,
    )
    
    # Optimizer (potentially freeze frontend)
    freeze_steps = train_config.get("freeze_frontend_steps", 0)
    
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
    )
    
    # Mixed precision
    use_fp16 = train_config.get("fp16", True) and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None
    
    # Logging
    log_config = config.get("logging", {})
    
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=log_config.get("wandb_project", "piw"),
            name=f"finetune_{datetime.now():%Y%m%d_%H%M}",
            config=config,
        )
    
    output_dir = Path(args.output_dir or "./outputs/finetune")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    max_epochs = train_config.get("max_epochs", 50)
    best_f1 = 0
    global_step = 0
    
    print(f"\nStarting fine-tuning for {max_epochs} epochs...")
    
    for epoch in range(max_epochs):
        model.train()
        
        # Freeze/unfreeze frontend
        if global_step < freeze_steps:
            for param in model.frontend.parameters():
                param.requires_grad = False
        else:
            for param in model.frontend.parameters():
                param.requires_grad = True
        
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=use_fp16):
                outputs = model(
                    waveform=waveform,
                    labels=labels,
                    apply_mask=False,
                )
                loss = outputs["prominence_loss"]
            
            if use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate
        metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{max_epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"P: {metrics['precision']:.4f} | "
              f"R: {metrics['recall']:.4f}")
        
        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": metrics["loss"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
            })
        
        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_path = output_dir / "best_model.pt"
            model.save_pretrained(str(best_path), config=config)
            print(f"  New best F1: {best_f1:.4f} - Saved to {best_path}")
    
    print(f"\nFine-tuning complete! Best F1: {best_f1:.4f}")
    
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="PIW Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
