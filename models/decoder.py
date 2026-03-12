"""
Decoder component for Latent Forecasting Network.

Maps latent representations to vocabulary logits for token prediction.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder that maps latent representations to vocabulary logits.

    Uses a simple linear projection without activation function,
    producing raw logits suitable for cross-entropy loss computation.
    """

    def __init__(self, latent_dim: int, vocab_size: int):
        """
        Initialize decoder with linear projection layer.

        Args:
            latent_dim: Dimension of latent representations
            vocab_size: Size of vocabulary

        Preconditions:
            - latent_dim > 0
            - vocab_size > 0
        """
        super().__init__()

        # Validate inputs
        assert latent_dim > 0, "latent_dim must be positive"
        assert vocab_size > 0, "vocab_size must be positive"

        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Linear projection from latent space to vocabulary space
        # No activation function - produces raw logits
        self.projection = nn.Linear(latent_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Map latent representations to vocabulary logits.

        Args:
            latents: Latent states [batch_size, seq_len, latent_dim]

        Returns:
            logits: Vocabulary logits [batch_size, seq_len, vocab_size]

        Preconditions:
            - latents.shape == [batch_size, seq_len, latent_dim]
            - All latent values are finite

        Postconditions:
            - Output shape == [batch_size, seq_len, vocab_size]
            - All logit values are finite
            - No softmax applied (raw logits for cross-entropy)
        """
        # Validate preconditions
        assert latents.dim() == 3, f"Expected 3D tensor, got {latents.dim()}D"
        assert (
            latents.size(-1) == self.latent_dim
        ), f"Expected latent_dim={self.latent_dim}, got {latents.size(-1)}"
        assert torch.isfinite(latents).all(), "Input contains NaN or Inf values"

        batch_size, seq_len, _ = latents.shape

        # Linear projection: [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, vocab_size]
        logits = self.projection(latents)

        # Validate postconditions
        assert logits.shape == (
            batch_size,
            seq_len,
            self.vocab_size,
        ), f"Output shape mismatch: expected {(batch_size, seq_len, self.vocab_size)}, got {logits.shape}"
        assert torch.isfinite(logits).all(), "Output contains NaN or Inf values"

        return logits
