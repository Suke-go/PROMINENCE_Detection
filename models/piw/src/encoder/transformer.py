"""
Transformer Encoder

Implements Transformer encoder with relative positional encoding
for context modeling in the PIW architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import RelativePositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with relative position bias.
    
    Args:
        d_model (int): Model dimension (default: 768)
        num_heads (int): Number of attention heads (default: 8)
        d_ff (int): Feedforward dimension (default: 3072)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Multi-head self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, d_model)
            attention_mask: Optional mask (batch, seq_len)
            position_bias: Relative position bias (1, num_heads, seq_len, seq_len)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        
        # QKV projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        
        # Add relative position bias
        if position_bias is not None:
            attn = attn + position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask to (B, 1, 1, T)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        x = residual + out
        
        # Pre-norm feedforward
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack with relative positional encoding.
    
    Args:
        d_model (int): Model dimension (default: 768)
        num_layers (int): Number of layers (default: 6)
        num_heads (int): Number of attention heads (default: 8)
        d_ff (int): Feedforward dimension (default: 3072)
        dropout (float): Dropout probability (default: 0.1)
        max_position (int): Maximum relative position (default: 512)
    
    Input Shape: (batch, seq_len, d_model)
    Output Shape: (batch, seq_len, d_model)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_position: int = 512,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(
            max_position=max_position,
            num_heads=num_heads,
        )
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder stack.
        
        Args:
            x: Input features (batch, seq_len, d_model)
            attention_mask: Optional padding mask (batch, seq_len)
            
        Returns:
            Encoded features (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Get relative position bias
        position_bias = self.pos_encoding(seq_len)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_bias=position_bias)
        
        x = self.final_norm(x)
        
        return x
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_layers={self.num_layers}"
