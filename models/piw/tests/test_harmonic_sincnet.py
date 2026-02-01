"""
Unit tests for HarmonicSincNet module.
"""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).replace("tests/test_harmonic_sincnet.py", "src"))

from frontend import HarmonicSincNet


class TestHarmonicSincNet:
    """Test suite for HarmonicSincNet."""
    
    @pytest.fixture
    def model(self):
        return HarmonicSincNet(
            n_filters=16,
            n_harmonics=4,
            kernel_size=101,
            stride=80,
            sample_rate=16000,
        )
    
    def test_output_shape(self, model):
        """Test output shape is correct."""
        batch_size = 2
        audio_len = 16000  # 1 second
        
        x = torch.randn(batch_size, 1, audio_len)
        out = model(x)
        
        expected_time = audio_len // model.stride
        assert out.shape == (batch_size, model.n_filters, expected_time)
    
    def test_f0_range_constraint(self, model):
        """Test F0 values stay within valid range."""
        f0 = model.f0
        
        assert (f0 >= model.f0_min).all()
        assert (f0 <= model.f0_max).all()
    
    def test_gradient_flow(self, model):
        """Test gradients flow through model."""
        x = torch.randn(2, 1, 8000, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.f0_raw.grad is not None
        assert model.harmonic_weights.grad is not None
    
    def test_kernel_generation_differentiable(self, model):
        """Test kernel generation is differentiable."""
        # Force gradient computation
        model.f0_raw.requires_grad_(True)
        kernels = model._generate_kernels()
        
        loss = kernels.sum()
        loss.backward()
        
        assert model.f0_raw.grad is not None
    
    def test_2d_input(self, model):
        """Test model accepts 2D input (batch, time)."""
        x = torch.randn(2, 8000)
        out = model(x)
        
        assert out.dim() == 3
    
    def test_f0_initialization(self, model):
        """Test F0 is initialized on log scale."""
        f0 = model.f0.detach()
        
        # Check roughly log-spaced
        ratios = f0[1:] / f0[:-1]
        # Ratios should be similar for log-spaced values
        assert ratios.std() < ratios.mean() * 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
