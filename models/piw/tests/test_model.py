"""
Unit tests for PIWModel (full model).
"""

import pytest
import torch
import sys
sys.path.insert(0, str(__file__).replace("tests/test_model.py", "src"))

from model import PIWModel


class TestPIWModel:
    """Test suite for PIWModel."""
    
    @pytest.fixture
    def pretrain_model(self):
        return PIWModel(
            frontend_dim=256,
            encoder_layers=2,
            encoder_heads=4,
            encoder_ff_dim=512,
            pretraining=True,
            finetuning=False,
        )
    
    @pytest.fixture
    def finetune_model(self):
        return PIWModel(
            frontend_dim=256,
            encoder_layers=2,
            encoder_heads=4,
            encoder_ff_dim=512,
            pretraining=False,
            finetuning=True,
        )
    
    def test_pretrain_output(self, pretrain_model):
        """Test pretraining forward pass."""
        x = torch.randn(2, 16000)
        out = pretrain_model(x, apply_mask=True)
        
        assert "features" in out
        assert "encoded" in out
        assert "pretrain_loss" in out
        assert "mask" in out
    
    def test_finetune_output(self, finetune_model):
        """Test finetuning forward pass."""
        x = torch.randn(2, 16000)
        
        # Get sequence length first
        out = finetune_model(x, apply_mask=False)
        seq_len = out["encoded"].shape[1]
        
        # Create labels
        labels = torch.randint(0, 2, (2, seq_len))
        
        out = finetune_model(x, labels=labels, apply_mask=False)
        
        assert "prominence_loss" in out
        assert "prominence_probs" in out
        assert "prominence_preds" in out
    
    def test_gradient_flow(self, pretrain_model):
        """Test gradients flow through full model."""
        x = torch.randn(2, 8000, requires_grad=True)
        out = pretrain_model(x)
        
        loss = out["pretrain_loss"]
        loss.backward()
        
        assert x.grad is not None
    
    def test_learned_parameters(self, pretrain_model):
        """Test model exposes learned physical parameters."""
        params = pretrain_model.get_learned_parameters()
        
        assert "f0_hz" in params
        assert params["f0_hz"].shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
