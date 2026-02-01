"""
Causal Envelope Extraction

Implements causal amplitude envelope extraction using IIR exponential
moving average filter. Replaces non-causal Hilbert transform.
"""

import torch
import torch.nn as nn


@torch.jit.script
def iir_envelope_forward(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled IIR envelope extraction.
    
    y[t] = α * y[t-1] + (1-α) * |x[t]|
    
    Args:
        x: Input signal (batch, time)
        alpha: Smoothing coefficient (scalar or per-batch)
        
    Returns:
        Envelope signal (batch, time)
    """
    B, T = x.shape
    y = torch.zeros_like(x)
    abs_x = torch.abs(x)
    
    y[:, 0] = abs_x[:, 0]
    one_minus_alpha = 1.0 - alpha
    
    for t in range(1, T):
        y[:, t] = alpha * y[:, t - 1] + one_minus_alpha * abs_x[:, t]
    
    return y


class CausalEnvelope(nn.Module):
    """
    Causal envelope extraction using IIR exponential moving average.
    
    Implements: y[t] = α * y[t-1] + (1-α) * |x[t]|
    
    The time constant τ determines the smoothness of the envelope:
    - Small τ (~1ms): Tracks rapid amplitude changes, noisy
    - Large τ (~10ms): Smooth envelope, may miss transients
    
    Args:
        tau_ms (float): Time constant in milliseconds (default: 3.0)
        sample_rate (int): Audio sample rate (default: 16000)
        learnable (bool): If True, tau is a learnable parameter (default: True)
        min_tau_ms (float): Minimum tau constraint (default: 0.5)
        max_tau_ms (float): Maximum tau constraint (default: 20.0)
    
    Input Shape: (batch, time) or (batch, 1, time)
    Output Shape: (batch, time) or (batch, 1, time)
    """
    
    def __init__(
        self,
        tau_ms: float = 3.0,
        sample_rate: int = 16000,
        learnable: bool = True,
        min_tau_ms: float = 0.5,
        max_tau_ms: float = 20.0,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.min_tau_ms = min_tau_ms
        self.max_tau_ms = max_tau_ms
        
        # Convert tau to alpha
        # α = exp(-1 / (τ * sr))
        # For learnable version, we store raw parameter and transform
        tau_samples = tau_ms * sample_rate / 1000.0
        alpha_init = 1.0 - 1.0 / tau_samples
        
        if learnable:
            # Store as raw parameter (inverse sigmoid)
            # Map alpha to [0.005, 0.995] for stability
            alpha_normalized = (alpha_init - 0.005) / 0.99
            alpha_normalized = max(0.001, min(0.999, alpha_normalized))
            alpha_raw = torch.log(torch.tensor(alpha_normalized / (1 - alpha_normalized)))
            self.alpha_raw = nn.Parameter(alpha_raw)
        else:
            self.register_buffer("alpha_raw", torch.tensor(alpha_init))
        
        self.learnable = learnable
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get alpha with stability constraint [0.005, 0.995]."""
        if self.learnable:
            return 0.005 + torch.sigmoid(self.alpha_raw) * 0.99
        else:
            return self.alpha_raw
    
    @property
    def tau_ms(self) -> float:
        """Get current time constant in milliseconds."""
        alpha = self.alpha.item()
        tau_samples = -1.0 / (torch.log(torch.tensor(alpha)) + 1e-8)
        return float(tau_samples * 1000.0 / self.sample_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract causal envelope from input signal.
        
        Args:
            x: Input waveform (batch, time) or (batch, 1, time)
            
        Returns:
            Envelope with same shape as input
        """
        squeeze_output = False
        if x.dim() == 3:
            if x.size(1) == 1:
                x = x.squeeze(1)
                squeeze_output = True
            else:
                raise ValueError(f"Expected (B, 1, T) or (B, T), got {x.shape}")
        
        # Apply IIR filter
        envelope = iir_envelope_forward(x, self.alpha)
        
        if squeeze_output:
            envelope = envelope.unsqueeze(1)
        
        return envelope
    
    def extra_repr(self) -> str:
        return (
            f"tau_ms={self.tau_ms:.2f}, sample_rate={self.sample_rate}, "
            f"learnable={self.learnable}"
        )


class MultiScaleCausalEnvelope(nn.Module):
    """
    Multi-scale envelope extraction with multiple time constants.
    
    Useful for capturing both fast transients and slow modulations.
    
    Args:
        tau_ms_list (list): List of time constants in ms
        sample_rate (int): Audio sample rate
        learnable (bool): If True, all tau values are learnable
    
    Input Shape: (batch, time)
    Output Shape: (batch, n_scales, time)
    """
    
    def __init__(
        self,
        tau_ms_list: list = [1.0, 3.0, 10.0],
        sample_rate: int = 16000,
        learnable: bool = True,
    ):
        super().__init__()
        
        self.envelopes = nn.ModuleList([
            CausalEnvelope(tau_ms=tau, sample_rate=sample_rate, learnable=learnable)
            for tau in tau_ms_list
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale envelopes.
        
        Args:
            x: Input waveform (batch, time)
            
        Returns:
            Stacked envelopes (batch, n_scales, time)
        """
        envelopes = [env(x) for env in self.envelopes]
        return torch.stack(envelopes, dim=1)
