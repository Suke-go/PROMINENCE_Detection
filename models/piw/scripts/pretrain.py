"""
PIW Pretraining Script

Self-supervised pretraining on LibriSpeech using masked feature prediction.

Usage:
    python scripts/pretrain.py --config configs/pretrain_1h.yaml
    python scripts/pretrain.py --config configs/pretrain_10h.yaml
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import PIWModel
from dataset import LibriSpeechDataset, create_dataloader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load and merge config files."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load base config if specified
    if "defaults" in config:
        base_names = config.pop("defaults")
        base_config = {}
        config_dir = Path(config_path).parent
        
        for base_name in base_names:
            base_path = config_dir / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path, "r") as f:
                    base = yaml.safe_load(f)
                    base_config = deep_merge(base_config, base)
        
        config = deep_merge(base_config, config)
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create warmup + cosine scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    return LambdaLR(optimizer, lr_lambda)


def train(config: dict, args):
    """Main training loop."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model_config = config.get("model", {})
    model = PIWModel(
        frontend_dim=model_config.get("frontend_dim", 768),
        encoder_layers=model_config.get("encoder_layers", 6),
        encoder_heads=model_config.get("encoder_heads", 8),
        encoder_ff_dim=model_config.get("encoder_ff_dim", 3072),
        pretraining=True,
        finetuning=False,
        dropout=model_config.get("dropout", 0.1),
    )
    model = model.to(device)
    
    # Print model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Create dataset
    data_config = config.get("data", {})
    
    # Try to find LibriSpeech data
    data_root = args.data_root or os.environ.get("LIBRISPEECH_ROOT", "./data/librispeech")
    
    dataset = LibriSpeechDataset(
        root=data_root,
        subset=data_config.get("subset", "train-clean-100"),
        max_hours=data_config.get("max_hours", None),
        sample_rate=data_config.get("sample_rate", 16000),
        max_length=data_config.get("max_audio_length", 10.0),
        min_length=data_config.get("min_audio_length", 1.0),
    )
    
    train_config = config.get("training", {})
    dataloader = create_dataloader(
        dataset,
        batch_size=train_config.get("batch_size", 8),
        shuffle=True,
        num_workers=4,
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 5e-4),
        betas=(
            train_config.get("adam_beta1", 0.9),
            train_config.get("adam_beta2", 0.98),
        ),
        eps=train_config.get("adam_epsilon", 1e-6),
        weight_decay=train_config.get("weight_decay", 0.01),
    )
    
    # Scheduler
    max_steps = train_config.get("max_steps", 50000)
    warmup_steps = train_config.get("warmup_steps", 1000)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, max_steps)
    
    # Mixed precision
    use_fp16 = train_config.get("fp16", True) and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None
    
    # Gradient accumulation
    grad_accum = train_config.get("gradient_accumulation", 1)
    
    # Logging
    log_config = config.get("logging", {})
    log_every = log_config.get("log_every_n_steps", 100)
    save_every = log_config.get("save_every_n_steps", 5000)
    
    # W&B
    if WANDB_AVAILABLE and not args.no_wandb:
        exp_config = config.get("experiment", {})
        wandb.init(
            project=log_config.get("wandb_project", "piw"),
            name=exp_config.get("name", f"pretrain_{datetime.now():%Y%m%d_%H%M}"),
            config=config,
        )
    
    # Output directory
    output_dir = Path(args.output_dir or "./outputs/pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    global_step = 0
    model.train()
    
    print(f"\nStarting training for {max_steps} steps...")
    print(f"Batch size: {train_config.get('batch_size')} x {grad_accum} = {train_config.get('batch_size', 8) * grad_accum}")
    
    while global_step < max_steps:
        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            with autocast(enabled=use_fp16):
                outputs = model(
                    waveform=waveform,
                    attention_mask=None,  # Handle after frontend
                    apply_mask=True,
                )
                loss = outputs["pretrain_loss"] / grad_accum
            
            # Backward pass
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient step
            if (global_step + 1) % grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if global_step % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {lr:.2e}")
                
                if WANDB_AVAILABLE and not args.no_wandb:
                    wandb.log({
                        "loss": loss.item() * grad_accum,
                        "learning_rate": lr,
                        "step": global_step,
                    })
            
            # Save checkpoint
            if global_step % save_every == 0:
                ckpt_path = output_dir / f"checkpoint_{global_step}.pt"
                model.save_pretrained(str(ckpt_path), config=config)
                print(f"Saved checkpoint: {ckpt_path}")
            
            if global_step >= max_steps:
                break
    
    # Final save
    final_path = output_dir / "final_model.pt"
    model.save_pretrained(str(final_path), config=config)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Log learned parameters
    params = model.get_learned_parameters()
    print("\nLearned Physical Parameters:")
    print(f"  F0 range: {params['f0_hz'].min():.1f} - {params['f0_hz'].max():.1f} Hz")
    print(f"  Envelope τ: {params['envelope_tau_ms']:.2f} ms")
    
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="PIW Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data_root", type=str, default=None, help="LibriSpeech root directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small steps)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.debug:
        config["training"]["max_steps"] = 100
        config["logging"]["log_every_n_steps"] = 10
        config["logging"]["save_every_n_steps"] = 50
    
    train(config, args)


if __name__ == "__main__":
    main()
