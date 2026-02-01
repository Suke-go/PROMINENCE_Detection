"""
Task Heads: Masked Prediction

Implements the pretext task head for self-supervised pretraining
using L2 reconstruction loss on masked frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPredictionHead(nn.Module):
    """
    Pretext task head for masked feature prediction.
    
    Predicts the original frontend features at masked positions
    using L2 reconstruction loss.
    
    Args:
        input_dim (int): Transformer output dimension (default: 768)
        target_dim (int): Frontend feature dimension to predict (default: 104)
        hidden_dim (int): Hidden layer dimension (default: 256)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        target_dim: int = 104,  # 32 + 64 + 8 from frontend
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Compute masked prediction loss.
        
        Args:
            encoder_output: Transformer output (batch, seq_len, input_dim)
            target_features: Original frontend features (batch, seq_len, target_dim)
            mask: Boolean mask indicating masked positions (batch, seq_len)
            
        Returns:
            Dictionary with 'loss' and 'predictions' keys
        """
        # Predict features
        predictions = self.predictor(encoder_output)  # (B, T, target_dim)
        
        # Compute loss only on masked positions
        masked_pred = predictions[mask]
        masked_target = target_features[mask]
        
        loss = F.mse_loss(masked_pred, masked_target)
        
        return {
            "loss": loss,
            "predictions": predictions,
            "masked_predictions": masked_pred,
            "masked_targets": masked_target,
        }


def generate_mask(
    batch_size: int,
    seq_len: int,
    mask_prob: float = 0.065,
    mask_length: int = 10,
    min_masks: int = 2,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate span mask for pretraining.
    
    Creates contiguous masked regions similar to Wav2Vec 2.0.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_prob: Probability of masking a frame
        mask_length: Length of each masked span
        min_masks: Minimum number of masked spans
        device: Target device
        
    Returns:
        Boolean mask (batch_size, seq_len) where True = masked
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Number of masks per sample
    num_masks = max(min_masks, int(mask_prob * seq_len / mask_length))
    
    for b in range(batch_size):
        # Random starting positions
        if seq_len <= mask_length:
            continue
            
        max_start = seq_len - mask_length
        starts = torch.randint(0, max_start, (num_masks,), device=device)
        
        for start in starts:
            end = min(start + mask_length, seq_len)
            mask[b, start:end] = True
    
    return mask
