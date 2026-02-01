"""
Feature Projection Layer

Concatenates outputs from all frontend branches and projects
to Transformer input dimension.
"""

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Concatenate and project frontend branch outputs.
    
    Combines Source, Filter, and Rhythm branch outputs into a single
    feature representation suitable for Transformer encoding.
    
    Args:
        source_dim (int): Source branch output channels (default: 32)
        filter_dim (int): Filter branch output channels (default: 64)
        rhythm_dim (int): Rhythm branch output channels (default: 8)
        output_dim (int): Transformer input dimension (default: 768)
        dropout (float): Dropout probability (default: 0.1)
    
    Input Shape: 
        source: (batch, source_dim, time)
        filter: (batch, filter_dim, time)
        rhythm: (batch, rhythm_dim, time)
    Output Shape: (batch, time, output_dim)
    """
    
    def __init__(
        self,
        source_dim: int = 32,
        filter_dim: int = 64,
        rhythm_dim: int = 8,
        output_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.source_dim = source_dim
        self.filter_dim = filter_dim
        self.rhythm_dim = rhythm_dim
        self.output_dim = output_dim
        
        input_dim = source_dim + filter_dim + rhythm_dim
        
        # Two-layer projection with GELU activation
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        source: torch.Tensor,
        filter: torch.Tensor,
        rhythm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate and project features.
        
        Args:
            source: Source branch output (B, C_s, T)
            filter: Filter branch output (B, C_f, T)
            rhythm: Rhythm branch output (B, C_r, T)
            
        Returns:
            Projected features (B, T, D)
        """
        # Concatenate along channel dimension
        concat = torch.cat([source, filter, rhythm], dim=1)  # (B, C_total, T)
        
        # Transpose to (B, T, C_total)
        concat = concat.transpose(1, 2)
        
        # Project
        out = self.projection(concat)
        out = self.layer_norm(out)
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"input_dims=[{self.source_dim}, {self.filter_dim}, {self.rhythm_dim}], "
            f"output_dim={self.output_dim}"
        )
