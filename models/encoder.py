"""
Encoder component for Latent Forecasting Network.

Transforms input token sequences into latent representations using
a Transformer-based architecture with causal masking.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model (latent_dim)
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    """
    Transformer-based encoder that converts token sequences into latent representations.

    The encoder applies causal masking to prevent future information leakage,
    making it suitable for autoregressive language modeling tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        max_context_length: int = 512,
    ):
        """
        Initialize encoder with embedding and transformer layers.

        Args:
            vocab_size: Size of vocabulary
            latent_dim: Dimension of latent representations
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_context_length: Maximum sequence length

        Preconditions:
            - vocab_size > 0
            - latent_dim > 0 and latent_dim % num_heads == 0
            - num_layers > 0
            - num_heads > 0
            - hidden_dim > 0
            - 0 <= dropout < 1
        """
        super().__init__()

        # Validate inputs
        assert vocab_size > 0, "vocab_size must be positive"
        assert latent_dim > 0, "latent_dim must be positive"
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        assert num_layers > 0, "num_layers must be positive"
        assert num_heads > 0, "num_heads must be positive"
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert 0 <= dropout < 1, "dropout must be in [0, 1)"

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_context_length = max_context_length

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, latent_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=latent_dim, max_len=max_context_length, dropout=dropout
        )

        # Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(latent_dim)
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.token_embedding.weight)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask to prevent attention to future positions.

        Args:
            seq_len: Sequence length
            device: Device for tensor
        Returns:
            Causal mask [seq_len, seq_len] with True for masked positions
        """
        # Create upper triangular matrix (True for positions to mask)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Transform input tokens into latent representations.

        Args:
            tokens: Input token IDs [batch_size, seq_len]

        Returns:
            latents: Latent representations [batch_size, seq_len, latent_dim]

        Preconditions:
            - tokens.shape == [batch_size, seq_len]
            - All token IDs in range [0, vocab_size)
            - seq_len <= max_context_length

        Postconditions:
            - Output shape == [batch_size, seq_len, latent_dim]
            - All output values are finite (no NaN or Inf)
            - Latent representations preserve sequence ordering
        """
        batch_size, seq_len = tokens.shape

        # Validate preconditions
        assert tokens.dim() == 2, f"Expected 2D tensor, got {tokens.dim()}D"
        assert (
            seq_len <= self.max_context_length
        ), f"Sequence length {seq_len} exceeds max_context_length {self.max_context_length}"
        assert (
            tokens.min() >= 0 and tokens.max() < self.vocab_size
        ), f"Token IDs must be in range [0, {self.vocab_size})"

        # Token embedding: [batch_size, seq_len] -> [batch_size, seq_len, latent_dim]
        embeddings = self.token_embedding(tokens)

        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)

        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, tokens.device)

        # Apply transformer encoder with causal masking
        # mask=True means "do not attend to this position"
        latents = self.transformer_encoder(embeddings, mask=causal_mask, is_causal=True)

        # Apply dropout
        latents = self.dropout(latents)

        # Validate postconditions
        assert latents.shape == (
            batch_size,
            seq_len,
            self.latent_dim,
        ), f"Output shape mismatch: expected {(batch_size, seq_len, self.latent_dim)}, got {latents.shape}"
        assert torch.isfinite(latents).all(), "Output contains NaN or Inf values"

        return latents
