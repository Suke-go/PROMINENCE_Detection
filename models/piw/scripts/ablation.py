"""
PIW Ablation Study Script

Evaluate contribution of each frontend branch.

Usage:
    python scripts/ablation.py --model outputs/finetune/best_model.pt --data_root ./data/helsinki
"""

import sys
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import PIWModel
from dataset import HelsinkiProsodyDataset, create_dataloader


def evaluate_model(model, dataloader, device):
    """Evaluate and return F1 score."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            labels = batch["labels"]
            
            outputs = model(waveform=waveform, apply_mask=False)
            
            all_preds.append(outputs["prominence_preds"].cpu())
            all_labels.append(labels)
    
    preds = torch.cat(all_preds, dim=0).flatten()
    labels = torch.cat(all_labels, dim=0).flatten()
    
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1.item()


class ZeroedBranch(nn.Module):
    """Wrapper that zeros out a branch's output."""
    
    def __init__(self, original, n_filters):
        super().__init__()
        self.original = original
        self.n_filters = n_filters
    
    def forward(self, x):
        out = self.original(x)
        return torch.zeros_like(out)


def run_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    print(f"Loading model from {args.model}")
    base_model = PIWModel.from_pretrained(args.model, finetuning=True, pretraining=False)
    
    # Test data
    test_dataset = HelsinkiProsodyDataset(root=args.data_root, split="test")
    test_loader = create_dataloader(test_dataset, batch_size=16, shuffle=False)
    
    results = {}
    
    # 1. Full model
    print("\nEvaluating configurations...")
    model = deepcopy(base_model).to(device)
    results["full"] = evaluate_model(model, test_loader, device)
    print(f"  Full model:      F1 = {results['full']:.4f}")
    
    # 2. Without Source Branch
    model = deepcopy(base_model)
    model.frontend.source_branch = ZeroedBranch(
        model.frontend.source_branch, 32
    )
    model = model.to(device)
    results["no_source"] = evaluate_model(model, test_loader, device)
    print(f"  w/o Source:      F1 = {results['no_source']:.4f} (Δ = {results['no_source'] - results['full']:+.4f})")
    
    # 3. Without Filter Branch
    model = deepcopy(base_model)
    model.frontend.filter_branch = ZeroedBranch(
        model.frontend.filter_branch, 64
    )
    model = model.to(device)
    results["no_filter"] = evaluate_model(model, test_loader, device)
    print(f"  w/o Filter:      F1 = {results['no_filter']:.4f} (Δ = {results['no_filter'] - results['full']:+.4f})")
    
    # 4. Without Rhythm Branch
    model = deepcopy(base_model)
    model.frontend.rhythm_branch = ZeroedBranch(
        model.frontend.rhythm_branch, 8
    )
    model = model.to(device)
    results["no_rhythm"] = evaluate_model(model, test_loader, device)
    print(f"  w/o Rhythm:      F1 = {results['no_rhythm']:.4f} (Δ = {results['no_rhythm'] - results['full']:+.4f})")
    
    # Summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"\nBranch Contributions (F1 drop when removed):")
    print(f"  Source (Harmonic):  {results['full'] - results['no_source']:.4f}")
    print(f"  Filter (Spectral):  {results['full'] - results['no_filter']:.4f}")
    print(f"  Rhythm (AM):        {results['full'] - results['no_rhythm']:.4f}")
    
    most_important = max(
        [("Source", results['full'] - results['no_source']),
         ("Filter", results['full'] - results['no_filter']),
         ("Rhythm", results['full'] - results['no_rhythm'])],
        key=lambda x: x[1]
    )
    print(f"\nMost important branch: {most_important[0]} (Δ = {most_important[1]:.4f})")


def main():
    parser = argparse.ArgumentParser(description="PIW Ablation Study")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_root", type=str, required=True, help="Test data root")
    
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
