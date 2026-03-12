"""
Baseline models for comparison with Latent Forecasting Network.

Provides two baseline implementations:
1. StandardTransformer: Standard transformer without latent forecasting
2. TransformerWithoutLatentLoss: Uses LFN architecture but disables latent loss (λ=0)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import Decoder
from models.encoder import Encoder


@dataclass
class BaselineModelOutput:
    """Output from baseline model forward pass."""

    logits: torch.Tensor  # [batch_size, seq_len, vocab_size]
    latents: torch.Tensor  # [batch_size, seq_len, latent_dim]
    token_loss: torch.Tensor  # Scalar
    total_loss: torch.Tensor  # Scalar (same as token_loss for baselines)


class StandardTransformer(nn.Module):
    """
    Standard Transformer baseline without latent forecasting.

    This model uses the same encoder-decoder architecture as LFN but
    without the latent forecasting network. It only performs token
    prediction using cross-entropy loss.
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        max_context_length: int = 512,
    ):
        """
        Initialize standard transformer baseline.

        Args:
            vocab_size: Size of vocabulary
            latent_dim: Dimension of latent representations
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_context_length: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_context_length = max_context_length

        # Initialize encoder (same as LFN)
        self.encoder = Encoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context_length,
        )

        # Initialize decoder (same as LFN)
        self.decoder = Decoder(latent_dim=latent_dim, vocab_size=vocab_size)

    def forward(
        self, tokens: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> BaselineModelOutput:
        """
        Forward pass with token prediction only.

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len] (optional)

        Returns:
            BaselineModelOutput containing logits, latents, and loss
        """
        batch_size, seq_len = tokens.shape

        # Create labels if not provided
        if labels is None:
            labels = tokens.clone()

        # Encode tokens to latent representations
        latents = self.encoder(tokens)

        # Decode latents to vocabulary logits
        logits = self.decoder(latents)

        # Compute token prediction loss
        token_loss = self._compute_token_loss(logits, labels)

        # For baseline, total loss is just token loss
        total_loss = token_loss

        return BaselineModelOutput(
            logits=logits, latents=latents, token_loss=token_loss, total_loss=total_loss
        )

    def _compute_token_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for token prediction.

        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(
            logits_flat, labels_flat, reduction="mean", ignore_index=-100
        )

        return loss

    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            tokens: Initial token sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = self.forward(tokens)
                logits = output.logits

                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                tokens = torch.cat([tokens, next_token], dim=1)

                if tokens.size(1) > self.max_context_length:
                    tokens = tokens[:, -self.max_context_length :]

        return tokens


class TransformerWithoutLatentLoss(nn.Module):
    """
    Transformer using LFN architecture but without latent forecasting loss.

    This baseline uses the complete LFN architecture (encoder, forecasting
    network, decoder) but sets λ=0, effectively disabling the latent
    forecasting loss. This helps isolate the effect of the latent loss
    on model performance.
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        max_context_length: int = 512,
    ):
        """
        Initialize transformer without latent loss.

        Args:
            vocab_size: Size of vocabulary
            latent_dim: Dimension of latent representations
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_context_length: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_context_length = max_context_length

        # Initialize encoder
        self.encoder = Encoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context_length,
        )

        # Initialize decoder
        self.decoder = Decoder(latent_dim=latent_dim, vocab_size=vocab_size)

        # Note: No forecasting network needed since we don't compute latent loss

    def forward(
        self, tokens: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> BaselineModelOutput:
        """
        Forward pass with token prediction only (no latent forecasting).

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len] (optional)

        Returns:
            BaselineModelOutput containing logits, latents, and loss
        """
        batch_size, seq_len = tokens.shape

        # Create labels if not provided
        if labels is None:
            labels = tokens.clone()

        # Encode tokens to latent representations
        latents = self.encoder(tokens)

        # Decode latents to vocabulary logits
        logits = self.decoder(latents)

        # Compute token prediction loss only
        token_loss = self._compute_token_loss(logits, labels)

        # Total loss is just token loss (λ=0)
        total_loss = token_loss

        return BaselineModelOutput(
            logits=logits, latents=latents, token_loss=token_loss, total_loss=total_loss
        )

    def _compute_token_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for token prediction.

        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(
            logits_flat, labels_flat, reduction="mean", ignore_index=-100
        )

        return loss

    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            tokens: Initial token sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = self.forward(tokens)
                logits = output.logits

                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                tokens = torch.cat([tokens, next_token], dim=1)

                if tokens.size(1) > self.max_context_length:
                    tokens = tokens[:, -self.max_context_length :]

        return tokens


def build_baseline_model(
    model_type: str,
    vocab_size: int,
    latent_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 2048,
    dropout: float = 0.1,
    max_context_length: int = 512,
    device: str = "cuda",
) -> nn.Module:
    """
    Factory function to build baseline models.

    Args:
        model_type: Type of baseline ('standard' or 'no_latent_loss')
        vocab_size: Size of vocabulary
        latent_dim: Dimension of latent representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Dimension of feedforward network
        dropout: Dropout probability
        max_context_length: Maximum sequence length
        device: Target device ('cuda' or 'cpu')

    Returns:
        Initialized baseline model on specified device

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "standard":
        model = StandardTransformer(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context_length,
        )
    elif model_type == "no_latent_loss":
        model = TransformerWithoutLatentLoss(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context_length,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Must be 'standard' or 'no_latent_loss'"
        )

    # Move to device
    model = model.to(device)

    return model
