"""
PIW Evaluation Script

Evaluate trained model on test set.

Usage:
    python scripts/evaluate.py --model outputs/finetune/best_model.pt --data_root ./data/helsinki
"""

import sys
import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import PIWModel
from dataset import HelsinkiProsodyDataset, create_dataloader


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    model = PIWModel.from_pretrained(args.model, finetuning=True, pretraining=False)
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = HelsinkiProsodyDataset(
        root=args.data_root,
        split="test",
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )
    
    # Evaluate
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            waveform = batch["waveform"].to(device)
            labels = batch["labels"]
            
            outputs = model(waveform=waveform, apply_mask=False)
            
            all_probs.append(outputs["prominence_probs"].cpu())
            all_preds.append(outputs["prominence_preds"].cpu())
            all_labels.append(labels)
    
    probs = torch.cat(all_probs, dim=0).numpy().flatten()
    preds = torch.cat(all_preds, dim=0).numpy().flatten()
    labels = torch.cat(all_labels, dim=0).numpy().flatten()
    
    # Metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Non-Prominent", "Prominent"]))
    
    auc = roc_auc_score(labels, probs)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Print learned parameters
    params = model.get_learned_parameters()
    print("\nLearned Physical Parameters:")
    print(f"  F0 filters: {params['f0_hz'].min():.1f} - {params['f0_hz'].max():.1f} Hz")
    print(f"  Spectral filters: {params['spectral_fc_hz'].min():.1f} - {params['spectral_fc_hz'].max():.1f} Hz")
    print(f"  Modulation filters: {params['modulation_fm_hz'].min():.2f} - {params['modulation_fm_hz'].max():.2f} Hz")
    print(f"  Envelope τ: {params['envelope_tau_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="PIW Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_root", type=str, required=True, help="Test data root")
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
