"""
Causal Filterbank: Filter Branch for Inductive Front-end

Implements learnable causal bandpass filters for spectral analysis.
Replaces non-causal STFT with differentiable causal convolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalFilterbank(nn.Module):
    """
    Learnable causal bandpass filterbank using SincNet-style filters.
    
    Captures spectral variations (formant movements, spectral flux) without
    requiring STFT. Each filter is parameterized by center frequency and bandwidth.
    
    Args:
        n_filters (int): Number of bandpass filters (default: 64)
        kernel_size (int): Filter length in samples (default: 401, ~25ms@16kHz)
        stride (int): Convolution stride (default: 160, 10ms@16kHz)
        sample_rate (int): Audio sample rate (default: 16000)
        f_min (float): Minimum center frequency in Hz (default: 100.0)
        f_max (float): Maximum center frequency in Hz (default: 4000.0)
        min_bandwidth (float): Minimum bandwidth in Hz (default: 50.0)
    
    Input Shape: (batch, 1, time)
    Output Shape: (batch, n_filters, time // stride)
    """
    
    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 401,
        stride: int = 160,
        sample_rate: int = 16000,
        f_min: float = 100.0,
        f_max: float = 4000.0,
        min_bandwidth: float = 50.0,
    ):
        super().__init__()
        
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.min_bandwidth = min_bandwidth
        
        # Ensure odd kernel for symmetric filters
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        
        # Initialize center frequencies on mel scale
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        
        # Filter low freq, high freq, and center freq
        self.register_buffer("_hz_points_init", hz_points)
        
        # Learnable parameters: center frequency and bandwidth
        # Use raw parameters with sigmoid transform for constraints
        fc_init = hz_points[1:-1]  # Center frequencies
        bw_init = hz_points[2:] - hz_points[:-2]  # Bandwidths
        
        # Inverse sigmoid transform
        fc_norm = (fc_init - f_min) / (f_max - f_min)
        fc_norm = torch.clamp(fc_norm, 0.001, 0.999)
        self.fc_raw = nn.Parameter(torch.log(fc_norm / (1 - fc_norm)))
        
        # Bandwidth (relative to nyquist/2)
        max_bw = sample_rate / 4
        bw_norm = (bw_init - min_bandwidth) / (max_bw - min_bandwidth)
        bw_norm = torch.clamp(bw_norm, 0.001, 0.999)
        self.bw_raw = nn.Parameter(torch.log(bw_norm / (1 - bw_norm)))
        
        # Pre-compute time axis
        t = torch.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size)
        t = t / sample_rate
        self.register_buffer("t", t)
        
        # Hamming window
        window = torch.hamming_window(self.kernel_size)
        self.register_buffer("window", window)
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(n_filters)
    
    @staticmethod
    def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)
    
    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    @property
    def fc(self) -> torch.Tensor:
        """Center frequencies in Hz."""
        return self.f_min + torch.sigmoid(self.fc_raw) * (self.f_max - self.f_min)
    
    @property
    def bw(self) -> torch.Tensor:
        """Bandwidths in Hz."""
        max_bw = self.sample_rate / 4
        return self.min_bandwidth + torch.sigmoid(self.bw_raw) * (max_bw - self.min_bandwidth)
    
    def _sinc(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized sinc function."""
        x_safe = torch.where(
            x == 0,
            torch.ones_like(x),
            x
        )
        return torch.where(
            x == 0,
            torch.ones_like(x),
            torch.sin(math.pi * x_safe) / (math.pi * x_safe)
        )
    
    def _generate_kernels(self) -> torch.Tensor:
        """
        Generate bandpass filter kernels.
        
        Bandpass = Lowpass(f_high) - Lowpass(f_low)
        where f_high = fc + bw/2, f_low = fc - bw/2
        
        Returns:
            Tensor of shape (n_filters, 1, kernel_size)
        """
        fc = self.fc.view(-1, 1)  # (F, 1)
        bw = self.bw.view(-1, 1)  # (F, 1)
        t = self.t.view(1, -1)     # (1, K)
        
        f_low = fc - bw / 2
        f_high = fc + bw / 2
        
        # Lowpass filters
        lowpass_high = 2 * f_high * self._sinc(2 * f_high * t)
        lowpass_low = 2 * f_low * self._sinc(2 * f_low * t)
        
        # Bandpass = difference
        bandpass = lowpass_high - lowpass_low  # (F, K)
        
        # Apply window
        bandpass = bandpass * self.window
        
        # Normalize
        bandpass = bandpass / (bandpass.norm(dim=1, keepdim=True) + 1e-8)
        
        return bandpass.unsqueeze(1)  # (F, 1, K)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply filterbank to input waveform.
        
        Args:
            x: Input waveform (batch, 1, time) or (batch, time)
            
        Returns:
            Filtered output (batch, n_filters, time // stride)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        kernels = self._generate_kernels()  # (F, 1, K)
        
        # Causal convolution
        padding = self.kernel_size - 1
        x_padded = F.pad(x, (padding, 0))
        
        out = F.conv1d(x_padded, kernels, stride=self.stride)
        
        # Energy (squared magnitude)
        out = out ** 2
        
        # Layer norm
        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)
        
        return out
    
    def compute_flux(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral flux from filterbank output.
        
        Args:
            x: Input waveform (batch, 1, time)
            
        Returns:
            Spectral flux (batch, n_filters, time // stride - 1)
        """
        energy = self.forward(x)
        flux = F.relu(energy[:, :, 1:] - energy[:, :, :-1])
        return flux
    
    def extra_repr(self) -> str:
        return (
            f"n_filters={self.n_filters}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, f_range=[{self.f_min}, {self.f_max}]Hz"
        )
