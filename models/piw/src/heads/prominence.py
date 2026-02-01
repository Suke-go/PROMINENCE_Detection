"""
Task Heads: Prominence Detection

Implements the downstream task head for prominence classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProminenceHead(nn.Module):
    """
    Downstream task head for prominence detection.
    
    Frame-level binary classification for prosodic prominence.
    
    Args:
        input_dim (int): Transformer output dimension (default: 768)
        hidden_dim (int): Hidden layer dimension (default: 256)
        dropout (float): Dropout probability (default: 0.1)
        label_smoothing (float): Label smoothing for BCE loss (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.label_smoothing = label_smoothing
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        labels: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Compute prominence predictions and optionally loss.
        
        Args:
            encoder_output: Transformer output (batch, seq_len, input_dim)
            labels: Optional ground truth labels (batch, seq_len)
            padding_mask: Optional mask for padded positions (batch, seq_len)
            
        Returns:
            Dictionary with 'logits', 'probabilities', and optionally 'loss'
        """
        logits = self.classifier(encoder_output).squeeze(-1)  # (B, T)
        probs = torch.sigmoid(logits)
        
        result = {
            "logits": logits,
            "probabilities": probs,
            "predictions": (probs > 0.5).long(),
        }
        
        if labels is not None:
            # Apply label smoothing
            if self.label_smoothing > 0:
                smooth_labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                smooth_labels = labels
            
            loss = F.binary_cross_entropy_with_logits(
                logits, smooth_labels.float(), reduction="none"
            )
            
            # Mask padding
            if padding_mask is not None:
                loss = loss * padding_mask.float()
                loss = loss.sum() / padding_mask.float().sum()
            else:
                loss = loss.mean()
            
            result["loss"] = loss
        
        return result
    
    def get_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Binary predictions (batch, seq_len)
            labels: Ground truth labels (batch, seq_len)
            padding_mask: Optional padding mask (batch, seq_len)
            
        Returns:
            Dictionary with precision, recall, f1, accuracy
        """
        if padding_mask is not None:
            predictions = predictions[padding_mask]
            labels = labels[padding_mask]
        else:
            predictions = predictions.flatten()
            labels = labels.flatten()
        
        tp = ((predictions == 1) & (labels == 1)).sum().float()
        fp = ((predictions == 1) & (labels == 0)).sum().float()
        fn = ((predictions == 0) & (labels == 1)).sum().float()
        tn = ((predictions == 0) & (labels == 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "accuracy": accuracy.item(),
        }
