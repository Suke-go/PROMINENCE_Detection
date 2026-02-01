"""
Unit tests for InductiveFrontend (integrated module).
"""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).replace("tests/test_frontend.py", "src"))

from frontend import InductiveFrontend


class TestInductiveFrontend:
    """Test suite for InductiveFrontend."""
    
    @pytest.fixture
    def model(self):
        return InductiveFrontend(
            output_dim=256,  # Smaller for testing
            sample_rate=16000,
            stride=160,
            n_f0_filters=8,
            n_harmonics=4,
            n_spectral_filters=16,
            n_am_filters=4,
        )
    
    def test_output_shape(self, model):
        """Test output shape is correct."""
        batch_size = 2
        audio_len = 16000  # 1 second
        
        x = torch.randn(batch_size, audio_len)
        out = model(x)
        
        expected_time = audio_len // model.stride
        assert out.shape[0] == batch_size
        assert out.shape[2] == model.output_dim
        # Time dimension might be slightly shorter due to branch alignment
        assert out.shape[1] <= expected_time
    
    def test_gradient_flow(self, model):
        """Test gradients flow through all branches."""
        x = torch.randn(2, 8000, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        
        # Check each branch has gradients
        assert model.source_branch.f0_raw.grad is not None
        assert model.filter_branch.fc_raw.grad is not None
        assert model.rhythm_branch.fm_raw.grad is not None
    
    def test_learned_parameters(self, model):
        """Test get_learned_parameters returns expected keys."""
        params = model.get_learned_parameters()
        
        expected_keys = [
            "f0_hz",
            "harmonic_weights",
            "spectral_fc_hz",
            "spectral_bw_hz",
            "modulation_fm_hz",
            "envelope_tau_ms",
        ]
        
        for key in expected_keys:
            assert key in params
    
    def test_3d_input(self, model):
        """Test model accepts 3D input."""
        x = torch.randn(2, 1, 8000)
        out = model(x)
        
        assert out.dim() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
