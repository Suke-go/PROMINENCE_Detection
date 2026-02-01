"""
Unit tests for CausalEnvelope module.
"""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).replace("tests/test_causal_envelope.py", "src"))

from frontend import CausalEnvelope


class TestCausalEnvelope:
    """Test suite for CausalEnvelope."""
    
    @pytest.fixture
    def model(self):
        return CausalEnvelope(
            tau_ms=3.0,
            sample_rate=16000,
            learnable=True,
        )
    
    def test_output_shape(self, model):
        """Test output shape matches input."""
        x = torch.randn(2, 8000)
        out = model(x)
        
        assert out.shape == x.shape
    
    def test_alpha_constraint(self, model):
        """Test alpha stays in valid range."""
        alpha = model.alpha
        
        assert alpha >= 0.005
        assert alpha <= 0.995
    
    def test_output_non_negative(self, model):
        """Test envelope is always non-negative."""
        x = torch.randn(2, 8000)
        out = model(x)
        
        assert (out >= 0).all()
    
    def test_gradient_flow(self, model):
        """Test gradients flow through IIR filter."""
        x = torch.randn(2, 4000, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert model.alpha_raw.grad is not None
    
    def test_3d_input(self, model):
        """Test model accepts 3D input (batch, 1, time)."""
        x = torch.randn(2, 1, 8000)
        out = model(x)
        
        assert out.shape == x.shape
    
    def test_smoothing_effect(self, model):
        """Test envelope is smoother than input."""
        # High frequency noise
        t = torch.linspace(0, 1, 16000)
        x = torch.sin(2 * 3.14159 * 1000 * t).unsqueeze(0)  # 1kHz sine
        
        out = model(x)
        
        # Envelope should have lower variance than rectified input
        rect_var = torch.abs(x).var()
        env_var = out.var()
        
        assert env_var < rect_var


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
