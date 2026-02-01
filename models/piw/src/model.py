"""
PIW Model: Physically-Inductive Wav2Vec

Main model class integrating frontend, encoder, and task heads.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .frontend import InductiveFrontend
from .encoder import TransformerEncoder
from .heads import MaskedPredictionHead, ProminenceHead
from .heads.masked_prediction import generate_mask


class PIWModel(nn.Module):
    """
    Physically-Inductive Wav2Vec model.
    
    Combines physics-informed frontend with Transformer encoder
    for self-supervised pretraining and prominence detection.
    
    Args:
        # Frontend config
        frontend_dim (int): Frontend output dimension (default: 768)
        sample_rate (int): Audio sample rate (default: 16000)
        stride (int): Frontend stride in samples (default: 160)
        
        # Encoder config  
        encoder_layers (int): Number of Transformer layers (default: 6)
        encoder_heads (int): Number of attention heads (default: 8)
        encoder_ff_dim (int): Feedforward dimension (default: 3072)
        
        # Task config
        pretraining (bool): If True, create pretraining head (default: True)
        finetuning (bool): If True, create prominence head (default: False)
        
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        # Frontend
        frontend_dim: int = 768,
        sample_rate: int = 16000,
        stride: int = 160,
        # Encoder
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        encoder_ff_dim: int = 3072,
        # Tasks
        pretraining: bool = True,
        finetuning: bool = False,
        # Common
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.stride = stride
        self.pretraining = pretraining
        self.finetuning = finetuning
        
        # Frontend: Physics-informed feature extraction
        self.frontend = InductiveFrontend(
            output_dim=frontend_dim,
            sample_rate=sample_rate,
            stride=stride,
            dropout=dropout,
        )
        
        # Encoder: Transformer with relative position encoding
        self.encoder = TransformerEncoder(
            d_model=frontend_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            d_ff=encoder_ff_dim,
            dropout=dropout,
        )
        
        # Task heads
        # Frontend total dim: 32 (source) + 64 (filter) + 8 (rhythm) = 104
        frontend_feature_dim = 32 + 64 + 8
        
        if pretraining:
            self.pretrain_head = MaskedPredictionHead(
                input_dim=frontend_dim,
                target_dim=frontend_feature_dim,
            )
        
        if finetuning:
            self.prominence_head = ProminenceHead(
                input_dim=frontend_dim,
                dropout=dropout,
            )
    
    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            waveform: Raw audio (batch, time) or (batch, 1, time)
            attention_mask: Optional padding mask (batch, seq_len)
            labels: Optional prominence labels for finetuning (batch, seq_len)
            apply_mask: Whether to apply masking for pretraining
            
        Returns:
            Dictionary with model outputs and losses
        """
        # Frontend feature extraction
        features = self.frontend(waveform)  # (B, T, D)
        
        # For pretraining: get raw frontend features before projection
        # We'll use the projected features as targets for simplicity
        B, T, D = features.shape
        
        # Apply masking for pretraining
        if self.pretraining and apply_mask:
            mask = generate_mask(
                batch_size=B,
                seq_len=T,
                device=features.device,
            )
            # Replace masked positions with learnable mask embedding
            # For simplicity, we use zeros + noise
            masked_features = features.clone()
            masked_features[mask] = 0.0
        else:
            masked_features = features
            mask = None
        
        # Encode
        encoded = self.encoder(masked_features, attention_mask=attention_mask)
        
        result = {
            "features": features,
            "encoded": encoded,
        }
        
        # Pretraining loss
        if self.pretraining and mask is not None:
            # Use features as prediction target (simplified)
            pretrain_out = self.pretrain_head(
                encoded,
                target_features=features,
                mask=mask,
            )
            result["pretrain_loss"] = pretrain_out["loss"]
            result["mask"] = mask
        
        # Finetuning loss
        if self.finetuning and labels is not None:
            prominence_out = self.prominence_head(
                encoded,
                labels=labels,
                padding_mask=attention_mask,
            )
            result["prominence_loss"] = prominence_out["loss"]
            result["prominence_probs"] = prominence_out["probabilities"]
            result["prominence_preds"] = prominence_out["predictions"]
        
        return result
    
    def get_learned_parameters(self) -> Dict[str, Any]:
        """Return learned physical parameters for interpretation."""
        return self.frontend.get_learned_parameters()
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "PIWModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        
        config = checkpoint.get("config", {})
        config.update(kwargs)
        
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def save_pretrained(self, path: str, config: Dict = None):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }, path)
