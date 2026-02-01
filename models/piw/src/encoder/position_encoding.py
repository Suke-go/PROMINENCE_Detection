"""
Relative Positional Encoding

Implements relative position bias for Transformer attention,
essential for capturing prosodic structure.
"""

import math
import torch
import torch.nn as nn


class RelativePositionalEncoding(nn.Module):
    """
    Learnable relative positional encoding for Transformer.
    
    Adds position-dependent bias to attention scores based on
    relative distance between positions.
    
    Args:
        max_position (int): Maximum relative distance (default: 512)
        num_heads (int): Number of attention heads (default: 8)
    """
    
    def __init__(
        self,
        max_position: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.max_position = max_position
        self.num_heads = num_heads
        
        # Learnable bias table: [-max_pos, max_pos]
        self.relative_bias = nn.Parameter(
            torch.zeros(2 * max_position + 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias tensor (1, num_heads, seq_len, seq_len)
        """
        # Create relative position indices
        positions = torch.arange(seq_len, device=self.relative_bias.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clamp to valid range and shift to positive indices
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_position,
            self.max_position
        )
        relative_positions = relative_positions + self.max_position
        
        # Look up bias values
        bias = self.relative_bias[relative_positions]  # (T, T, H)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)
        
        return bias
