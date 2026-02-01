"""
Inductive Frontend: Unified module combining all branches

Main entry point for the physics-informed feature extraction.
"""

import torch
import torch.nn as nn

from .harmonic_sincnet import HarmonicSincNet
from .causal_filterbank import CausalFilterbank
from .am_filterbank import AMFilterbank
from .projection import FeatureProjection


class InductiveFrontend(nn.Module):
    """
    Physically-Inductive Frontend combining Source, Filter, and Rhythm branches.
    
    Replaces the CNN feature extractor in Wav2Vec2 with physics-informed
    differentiable signal processing layers.
    
    Args:
        output_dim (int): Output feature dimension (default: 768)
        sample_rate (int): Audio sample rate (default: 16000)
        stride (int): Overall stride in samples (default: 160, 10ms@16kHz)
        
        # Source Branch (Harmonic SincNet)
        n_f0_filters (int): Number of F0 filters (default: 32)
        n_harmonics (int): Number of harmonics (default: 8)
        f0_kernel_size (int): F0 filter kernel size (default: 251)
        f0_min (float): Minimum F0 (default: 50.0)
        f0_max (float): Maximum F0 (default: 500.0)
        
        # Filter Branch (Causal Filterbank)
        n_spectral_filters (int): Number of spectral filters (default: 64)
        spectral_kernel_size (int): Spectral filter kernel size (default: 401)
        f_min (float): Minimum frequency (default: 100.0)
        f_max (float): Maximum frequency (default: 4000.0)
        
        # Rhythm Branch (AM Filterbank)
        n_am_filters (int): Number of AM filters (default: 8)
        am_kernel_size (int): AM filter kernel size (default: 801)
        fm_min (float): Minimum modulation frequency (default: 1.0)
        fm_max (float): Maximum modulation frequency (default: 16.0)
        
        dropout (float): Projection dropout (default: 0.1)
    
    Input Shape: (batch, time) or (batch, 1, time)
    Output Shape: (batch, time // stride, output_dim)
    """
    
    def __init__(
        self,
        output_dim: int = 768,
        sample_rate: int = 16000,
        stride: int = 160,
        # Source Branch
        n_f0_filters: int = 32,
        n_harmonics: int = 8,
        f0_kernel_size: int = 251,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        # Filter Branch
        n_spectral_filters: int = 64,
        spectral_kernel_size: int = 401,
        f_min: float = 100.0,
        f_max: float = 4000.0,
        # Rhythm Branch
        n_am_filters: int = 8,
        am_kernel_size: int = 801,
        fm_min: float = 1.0,
        fm_max: float = 16.0,
        # Common
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.stride = stride
        
        # Source Branch: Harmonic structure
        self.source_branch = HarmonicSincNet(
            n_filters=n_f0_filters,
            n_harmonics=n_harmonics,
            kernel_size=f0_kernel_size,
            stride=stride,
            sample_rate=sample_rate,
            f0_min=f0_min,
            f0_max=f0_max,
            learnable_harmonic_weights=True,
        )
        
        # Filter Branch: Spectral structure
        self.filter_branch = CausalFilterbank(
            n_filters=n_spectral_filters,
            kernel_size=spectral_kernel_size,
            stride=stride,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
        )
        
        # Rhythm Branch: Amplitude modulation
        self.rhythm_branch = AMFilterbank(
            n_filters=n_am_filters,
            kernel_size=am_kernel_size,
            stride=1,  # Already downsampled by hop_size
            sample_rate=sample_rate,
            hop_size=stride,
            f_min=fm_min,
            f_max=fm_max,
        )
        
        # Feature Projection
        self.projection = FeatureProjection(
            source_dim=n_f0_filters,
            filter_dim=n_spectral_filters,
            rhythm_dim=n_am_filters,
            output_dim=output_dim,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract physics-informed features from raw waveform.
        
        Args:
            x: Raw waveform (batch, time) or (batch, 1, time)
            
        Returns:
            Feature sequence (batch, time // stride, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        
        # Apply branches
        source_out = self.source_branch(x)  # (B, C_s, T')
        filter_out = self.filter_branch(x)  # (B, C_f, T')
        rhythm_out = self.rhythm_branch(x)  # (B, C_r, T')
        
        # Ensure same time dimension (take minimum)
        min_len = min(source_out.size(2), filter_out.size(2), rhythm_out.size(2))
        source_out = source_out[:, :, :min_len]
        filter_out = filter_out[:, :, :min_len]
        rhythm_out = rhythm_out[:, :, :min_len]
        
        # Project to output dimension
        out = self.projection(source_out, filter_out, rhythm_out)  # (B, T', D)
        
        return out
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return input_length // self.stride
    
    def get_learned_parameters(self) -> dict:
        """Return learned physical parameters for interpretation."""
        return {
            "f0_hz": self.source_branch.get_f0_hz(),
            "harmonic_weights": self.source_branch.harmonic_weights.detach().cpu(),
            "spectral_fc_hz": self.filter_branch.fc.detach().cpu(),
            "spectral_bw_hz": self.filter_branch.bw.detach().cpu(),
            "modulation_fm_hz": self.rhythm_branch.fm.detach().cpu(),
            "envelope_tau_ms": self.rhythm_branch.envelope.tau_ms,
        }
