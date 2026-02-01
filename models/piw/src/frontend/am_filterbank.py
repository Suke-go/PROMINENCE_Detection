"""
AM Filterbank: Rhythm Branch for Inductive Front-end

Implements amplitude modulation filterbank to capture syllabic rhythm
and energy modulation patterns in the 1-16Hz range.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_envelope import CausalEnvelope


class AMFilterbank(nn.Module):
    """
    Amplitude Modulation filterbank for rhythm extraction.
    
    Processes the amplitude envelope through a bank of bandpass filters
    tuned to speech rhythm frequencies (1-16Hz). This captures:
    - Syllabic rate (~2-8Hz)
    - Stress patterns
    - Phrase-level prosody
    
    Architecture:
        Raw Audio → Causal Envelope → AM Filterbank → Output
    
    Args:
        n_filters (int): Number of AM filters (default: 8)
        kernel_size (int): Filter length in envelope samples (default: 801)
        stride (int): Convolution stride (default: 160)
        sample_rate (int): Audio sample rate (default: 16000)
        hop_size (int): Envelope downsampling hop (default: 160)
        f_min (float): Minimum modulation frequency (default: 1.0)
        f_max (float): Maximum modulation frequency (default: 16.0)
        envelope_tau_ms (float): Envelope time constant (default: 3.0)
    
    Input Shape: (batch, 1, time)
    Output Shape: (batch, n_filters, time // (hop_size * stride))
    """
    
    def __init__(
        self,
        n_filters: int = 8,
        kernel_size: int = 801,
        stride: int = 1,  # After envelope downsampling
        sample_rate: int = 16000,
        hop_size: int = 160,  # 10ms hop for envelope
        f_min: float = 1.0,
        f_max: float = 16.0,
        envelope_tau_ms: float = 3.0,
    ):
        super().__init__()
        
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f_min = f_min
        self.f_max = f_max
        
        # Envelope sample rate after downsampling
        self.envelope_sr = sample_rate / hop_size  # 100 Hz typically
        
        # Ensure odd kernel
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        
        # Causal envelope extractor
        self.envelope = CausalEnvelope(
            tau_ms=envelope_tau_ms,
            sample_rate=sample_rate,
            learnable=True
        )
        
        # Initialize modulation frequencies on log scale
        fm_init = torch.logspace(
            math.log10(f_min),
            math.log10(f_max),
            n_filters
        )
        
        # Learnable parameters
        fm_norm = (fm_init - f_min) / (f_max - f_min)
        fm_norm = torch.clamp(fm_norm, 0.001, 0.999)
        self.fm_raw = nn.Parameter(torch.log(fm_norm / (1 - fm_norm)))
        
        # Bandwidth (Q factor around 2)
        bw_init = fm_init / 2.0  # Q = 2
        max_bw = f_max / 2
        min_bw = 0.5
        bw_norm = (bw_init - min_bw) / (max_bw - min_bw)
        bw_norm = torch.clamp(bw_norm, 0.001, 0.999)
        self.bw_raw = nn.Parameter(torch.log(bw_norm / (1 - bw_norm)))
        self.min_bw = min_bw
        self.max_bw = max_bw
        
        # Time axis for kernel generation (in seconds, at envelope sample rate)
        t = torch.linspace(
            -self.kernel_size // 2,
            self.kernel_size // 2,
            self.kernel_size
        ) / self.envelope_sr
        self.register_buffer("t", t)
        
        # Hann window
        window = torch.hann_window(self.kernel_size)
        self.register_buffer("window", window)
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(n_filters)
    
    @property
    def fm(self) -> torch.Tensor:
        """Modulation center frequencies in Hz."""
        return self.f_min + torch.sigmoid(self.fm_raw) * (self.f_max - self.f_min)
    
    @property
    def bw(self) -> torch.Tensor:
        """Modulation bandwidths in Hz."""
        return self.min_bw + torch.sigmoid(self.bw_raw) * (self.max_bw - self.min_bw)
    
    def _generate_kernels(self) -> torch.Tensor:
        """
        Generate AM bandpass filter kernels.
        
        Returns:
            Tensor of shape (n_filters, 1, kernel_size)
        """
        fm = self.fm.view(-1, 1)  # (F, 1)
        bw = self.bw.view(-1, 1)  # (F, 1)
        t = self.t.view(1, -1)    # (1, K)
        
        # Gammatone-like modulation filter
        # Simplified as windowed cosine modulated by Gaussian envelope
        gaussian_env = torch.exp(-0.5 * (t * bw * 2) ** 2)
        carrier = torch.cos(2 * math.pi * fm * t)
        
        kernels = gaussian_env * carrier * self.window  # (F, K)
        
        # Normalize
        kernels = kernels / (kernels.norm(dim=1, keepdim=True) + 1e-8)
        
        return kernels.unsqueeze(1)  # (F, 1, K)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract amplitude modulation features.
        
        Args:
            x: Input waveform (batch, 1, time) or (batch, time)
            
        Returns:
            AM features (batch, n_filters, time // hop_size)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Extract envelope
        env = self.envelope(x)  # (B, T)
        
        # Downsample envelope
        env_ds = env[:, ::self.hop_size]  # (B, T // hop_size)
        env_ds = env_ds.unsqueeze(1)  # (B, 1, T')
        
        # Generate kernels
        kernels = self._generate_kernels()  # (F, 1, K)
        
        # Causal convolution
        padding = self.kernel_size - 1
        env_padded = F.pad(env_ds, (padding, 0))
        
        out = F.conv1d(env_padded, kernels, stride=self.stride)
        
        # Rectify and normalize
        out = torch.abs(out)
        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"n_filters={self.n_filters}, kernel_size={self.kernel_size}, "
            f"fm_range=[{self.f_min}, {self.f_max}]Hz, hop_size={self.hop_size}"
        )
