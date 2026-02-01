"""
Harmonic SincNet: Source Branch for Inductive Front-end

Implements a learnable harmonic comb filter based on SincNet architecture.
Captures fundamental frequency (F0) and harmonic structure of voice source.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HarmonicSincNet(nn.Module):
    """
    Harmonic Comb Filter based on SincNet.
    
    Generates learnable bandpass filters centered on fundamental frequency f0
    and its integer harmonics (2f0, 3f0, ...). This directly models the 
    quasi-periodic nature of voice source excitation.
    
    Args:
        n_filters (int): Number of parallel F0 filters (default: 32)
        n_harmonics (int): Number of harmonics per filter (default: 8)
        kernel_size (int): Filter kernel size in samples (default: 251, ~15.7ms@16kHz)
        stride (int): Convolution stride (default: 160, 10ms@16kHz)
        sample_rate (int): Audio sample rate (default: 16000)
        f0_min (float): Minimum F0 in Hz (default: 50.0)
        f0_max (float): Maximum F0 in Hz (default: 500.0)
        learnable_harmonic_weights (bool): If True, harmonic weights are learnable
    
    Input Shape: (batch, 1, time)
    Output Shape: (batch, n_filters, time // stride)
    """
    
    def __init__(
        self,
        n_filters: int = 32,
        n_harmonics: int = 8,
        kernel_size: int = 251,
        stride: int = 160,
        sample_rate: int = 16000,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        learnable_harmonic_weights: bool = True,
    ):
        super().__init__()
        
        self.n_filters = n_filters
        self.n_harmonics = n_harmonics
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        # Ensure odd kernel size for symmetric filters
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        
        # F0 parameters (raw values, transformed via sigmoid)
        # Initialize on log scale from f0_min to f0_max
        f0_init = torch.logspace(
            math.log10(f0_min), 
            math.log10(f0_max), 
            n_filters
        )
        # Inverse sigmoid to get raw initialization
        f0_normalized = (f0_init - f0_min) / (f0_max - f0_min)
        f0_raw_init = torch.log(f0_normalized / (1 - f0_normalized + 1e-8) + 1e-8)
        self.f0_raw = nn.Parameter(f0_raw_init)
        
        # Harmonic weights (optional learnable)
        if learnable_harmonic_weights:
            # Initialize with decreasing weights for higher harmonics
            harmonic_weights_init = 1.0 / torch.arange(1, n_harmonics + 1).float()
            self.harmonic_weights = nn.Parameter(
                harmonic_weights_init.unsqueeze(0).expand(n_filters, -1).clone()
            )
        else:
            self.register_buffer(
                "harmonic_weights",
                (1.0 / torch.arange(1, n_harmonics + 1).float())
                .unsqueeze(0).expand(n_filters, -1)
            )
        
        # Pre-compute time axis (fixed)
        t = torch.linspace(
            -self.kernel_size // 2,
            self.kernel_size // 2,
            self.kernel_size
        ) / sample_rate
        self.register_buffer("t", t)
        
        # Hann window for smooth filter edges
        window = torch.hann_window(self.kernel_size)
        self.register_buffer("window", window)
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(n_filters)
    
    @property
    def f0(self) -> torch.Tensor:
        """Get actual F0 values in Hz with range constraint."""
        return self.f0_min + torch.sigmoid(self.f0_raw) * (self.f0_max - self.f0_min)
    
    def _generate_kernels(self) -> torch.Tensor:
        """
        Generate harmonic comb filter kernels.
        
        Returns:
            Tensor of shape (n_filters, 1, kernel_size)
        """
        f0 = self.f0  # (n_filters,)
        
        # Reshape for broadcasting: (n_filters, n_harmonics, kernel_size)
        f0_expanded = f0.view(-1, 1, 1)  # (F, 1, 1)
        t_expanded = self.t.view(1, 1, -1)  # (1, 1, K)
        n_harm = torch.arange(1, self.n_harmonics + 1, device=f0.device).float()
        n_harm = n_harm.view(1, -1, 1)  # (1, H, 1)
        weights = self.harmonic_weights.unsqueeze(-1)  # (F, H, 1)
        
        # Generate sinusoidal components for each harmonic
        # kernel[f, h, t] = w[f,h] * sin(2π * h * f0[f] * t)
        kernels = weights * torch.sin(2 * math.pi * n_harm * f0_expanded * t_expanded)
        
        # Sum over harmonics
        kernels = kernels.sum(dim=1)  # (F, K)
        
        # Apply window
        kernels = kernels * self.window
        
        # Normalize each filter
        kernels = kernels / (kernels.norm(dim=1, keepdim=True) + 1e-8)
        
        return kernels.unsqueeze(1)  # (F, 1, K)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply harmonic filters to input waveform.
        
        Args:
            x: Input waveform of shape (batch, 1, time) or (batch, time)
            
        Returns:
            Filtered output of shape (batch, n_filters, time // stride)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Generate kernels dynamically (for gradient flow)
        kernels = self._generate_kernels()  # (F, 1, K)
        
        # Causal convolution with left padding
        padding = self.kernel_size - 1
        x_padded = F.pad(x, (padding, 0))
        
        # Apply convolution
        out = F.conv1d(x_padded, kernels, stride=self.stride)
        
        # Take absolute value (energy) and apply layer norm
        out = torch.abs(out)
        out = out.transpose(1, 2)  # (B, T, F)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # (B, F, T)
        
        return out
    
    def get_f0_hz(self) -> torch.Tensor:
        """Return learned F0 values in Hz for interpretation."""
        return self.f0.detach().cpu()
    
    def extra_repr(self) -> str:
        return (
            f"n_filters={self.n_filters}, n_harmonics={self.n_harmonics}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"f0_range=[{self.f0_min}, {self.f0_max}]Hz"
        )
