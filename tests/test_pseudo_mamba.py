import unittest
import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cyborg_rl.models.pseudo_mamba import PseudoMambaBlock, PseudoMambaEncoder

class TestPseudoMamba(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.input_dim = 8
        self.hidden_dim = 16
        self.latent_dim = 8
        
    def test_block_forward(self):
        """Test PseudoMambaBlock forward pass."""
        block = PseudoMambaBlock(d_model=self.hidden_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        out = block(x)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertFalse(torch.isnan(out).any())
        
    def test_encoder_forward(self):
        """Test PseudoMambaEncoder forward pass."""
        encoder = PseudoMambaEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2
        )
        
        # Sequential input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        latent, hidden = encoder(x)
        
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))
        self.assertIsNone(hidden)  # PseudoMamba is currently stateless/internal state only
        
        # Non-sequential input
        x_flat = torch.randn(self.batch_size, self.input_dim)
        latent, hidden = encoder(x_flat)
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))

if __name__ == '__main__':
    unittest.main()
