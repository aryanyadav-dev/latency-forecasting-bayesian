"""
Complete Latent Forecasting Network Model.

Integrates encoder, forecasting network, and decoder into a unified model
with support for both token prediction and latent forecasting losses.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import Decoder
from models.encoder import Encoder
from models.forecasting_network import LatentForecastingNetwork


@dataclass
class ModelConfig:
    """Configuration for Latent Forecasting Model."""

    vocab_size: int
    latent_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    hidden_dim: int = 2048
    dropout: float = 0.1
    forecast_horizons: List[int] = None
    max_context_length: int = 512
    lambda_latent: float = 0.1  # Weight for latent forecasting loss

    def __post_init__(self):
        """Set default forecast horizons if not provided."""
        if self.forecast_horizons is None:
            self.forecast_horizons = [1, 2, 5, 10]

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")
        if not all(h > 0 for h in self.forecast_horizons):
            raise ValueError("All forecast_horizons must be positive")
        if self.max_context_length <= max(self.forecast_horizons):
            raise ValueError(
                "max_context_length must be greater than max forecast horizon"
            )
        if self.lambda_latent < 0:
            raise ValueError("lambda_latent must be non-negative")

        return True


@dataclass
class ModelOutput:
    """Output from Latent Forecasting Model forward pass."""

    logits: torch.Tensor  # [batch_size, seq_len, vocab_size]
    latents: torch.Tensor  # [batch_size, seq_len, latent_dim]
    predicted_latents: Dict[
        int, torch.Tensor
    ]  # {horizon: [batch_size, seq_len-k, latent_dim]}
    token_loss: torch.Tensor  # Scalar
    latent_loss: Optional[torch.Tensor]  # Scalar or None
    total_loss: torch.Tensor  # Scalar


class LatentForecastingModel(nn.Module):
    """
    Complete Latent Forecasting Network model.

    Integrates encoder, forecasting network, and decoder to perform both
    token prediction and latent state forecasting. Supports baseline mode
    where latent forecasting loss is disabled.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize complete model from configuration.

        Args:
            config: Model configuration

        Preconditions:
            - config is valid (passes config.validate())
        """
        super().__init__()

        # Validate configuration
        config.validate()

        self.config = config

        # Initialize encoder
        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            max_context_length=config.max_context_length,
        )

        # Initialize latent forecasting network
        self.forecasting_network = LatentForecastingNetwork(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            forecast_horizons=config.forecast_horizons,
            dropout=config.dropout,
        )

        # Initialize decoder
        self.decoder = Decoder(
            latent_dim=config.latent_dim, vocab_size=config.vocab_size
        )

        # Store lambda for loss weighting
        self.lambda_latent = config.lambda_latent

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_latent_loss: bool = True,
    ) -> ModelOutput:
        """
        Forward pass with loss computation.

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len] (optional, defaults to shifted tokens)
            compute_latent_loss: Whether to compute latent forecasting loss

        Returns:
            ModelOutput containing logits, latents, predictions, and losses

        Preconditions:
            - tokens.shape == [batch_size, seq_len]
            - All token IDs in valid range [0, vocab_size)
            - seq_len <= max_context_length
            - If compute_latent_loss=True: seq_len > max(forecast_horizons)
            - If labels provided: labels.shape == tokens.shape

        Postconditions:
            - All output tensors have correct shapes
            - Losses are non-negative scalars
            - total_loss = token_loss + lambda_latent * latent_loss
            - Gradients flow through all components
        """
        batch_size, seq_len = tokens.shape

        # Validate preconditions
        assert tokens.dim() == 2, f"Expected 2D tensor, got {tokens.dim()}D"
        assert (
            seq_len <= self.config.max_context_length
        ), f"Sequence length {seq_len} exceeds max_context_length {self.config.max_context_length}"

        if compute_latent_loss:
            assert seq_len > max(
                self.config.forecast_horizons
            ), f"Sequence length {seq_len} must be greater than max horizon {max(self.config.forecast_horizons)}"

        # Create labels if not provided (shift tokens by 1)
        if labels is None:
            labels = tokens.clone()

        # 1. Encode tokens to latent representations
        latents = self.encoder(tokens)  # [batch_size, seq_len, latent_dim]

        # 2. Predict future latent states (if enabled)
        predicted_latents = {}
        latent_loss = None

        if compute_latent_loss:
            predicted_latents = self.forecasting_network(latents)
            latent_loss = self._compute_latent_forecasting_loss(
                latents, predicted_latents
            )

        # 3. Decode latents to vocabulary logits
        logits = self.decoder(latents)  # [batch_size, seq_len, vocab_size]

        # 4. Compute token prediction loss
        token_loss = self._compute_token_loss(logits, labels)

        # 5. Compute total loss
        if compute_latent_loss and latent_loss is not None:
            total_loss = token_loss + self.lambda_latent * latent_loss
        else:
            total_loss = token_loss

        # Validate postconditions
        assert logits.shape == (
            batch_size,
            seq_len,
            self.config.vocab_size,
        ), f"Logits shape mismatch: expected {(batch_size, seq_len, self.config.vocab_size)}, got {logits.shape}"
        assert latents.shape == (
            batch_size,
            seq_len,
            self.config.latent_dim,
        ), f"Latents shape mismatch: expected {(batch_size, seq_len, self.config.latent_dim)}, got {latents.shape}"
        assert token_loss >= 0, "Token loss must be non-negative"
        if latent_loss is not None:
            assert latent_loss >= 0, "Latent loss must be non-negative"
        assert total_loss >= 0, "Total loss must be non-negative"
        assert torch.isfinite(total_loss), "Total loss must be finite"

        return ModelOutput(
            logits=logits,
            latents=latents,
            predicted_latents=predicted_latents,
            token_loss=token_loss,
            latent_loss=latent_loss,
            total_loss=total_loss,
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
        # Reshape for cross-entropy: [batch_size * seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction="mean",
            ignore_index=-100,  # Ignore padding tokens if any
        )

        return loss

    def _compute_latent_forecasting_loss(
        self, latents: torch.Tensor, predicted_latents: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-horizon latent forecasting loss.

        Args:
            latents: Ground truth latent states [batch_size, seq_len, latent_dim]
            predicted_latents: Predicted latents for each horizon
                              {horizon: [batch_size, seq_len-k, latent_dim]}

        Returns:
            Average MSE loss across all horizons

        Preconditions:
            - latents.shape == [batch_size, seq_len, latent_dim]
            - For each k in forecast_horizons:
                predicted_latents[k].shape == [batch_size, seq_len-k, latent_dim]
            - All tensors contain finite values
            - seq_len > max(forecast_horizons)

        Postconditions:
            - Returns non-negative scalar
            - Loss is finite
            - Loss represents average prediction error across all horizons
        """
        total_loss = 0.0
        num_horizons = len(self.config.forecast_horizons)

        for horizon in self.config.forecast_horizons:
            # Get target latents at time t+k
            target_latents = latents[
                :, horizon:, :
            ]  # [batch_size, seq_len-k, latent_dim]

            # Get predicted latents for this horizon
            pred_latents = predicted_latents[
                horizon
            ]  # [batch_size, seq_len-k, latent_dim]

            # Compute MSE for this horizon
            mse = F.mse_loss(pred_latents, target_latents, reduction="mean")
            total_loss += mse

        # Average across horizons
        avg_loss = total_loss / num_horizons

        # Validate postconditions
        assert avg_loss >= 0, "Latent loss must be non-negative"
        assert torch.isfinite(avg_loss), "Latent loss must be finite"

        return avg_loss

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
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for current sequence
                output = self.forward(tokens, compute_latent_loss=False)
                logits = output.logits

                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)

                # Truncate if exceeds max context length
                if tokens.size(1) > self.config.max_context_length:
                    tokens = tokens[:, -self.config.max_context_length :]

        return tokens


def build_model(config: ModelConfig, device: str = "cuda") -> LatentForecastingModel:
    """
    Construct complete LFN model from configuration.

    Args:
        config: Model configuration
        device: Target device ('cuda' or 'cpu')

    Returns:
        Initialized model on specified device

    Preconditions:
        - config is valid (passes config.validate())
        - device is either 'cuda' or 'cpu'
        - If device='cuda', CUDA is available

    Postconditions:
        - Returns LatentForecastingModel instance
        - Model is on specified device
        - All parameters are initialized (not None)
        - All parameters are finite (no NaN or Inf)
        - Model is in training mode by default
    """
    # Validate preconditions
    config.validate()
    assert device in ["cuda", "cpu"], f"Invalid device: {device}"
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"

    # Build model
    model = LatentForecastingModel(config)

    # Move to device
    model = model.to(device)

    # Validate postconditions
    assert all(p is not None for p in model.parameters()), "Some parameters are None"
    assert all(
        torch.isfinite(p).all() for p in model.parameters()
    ), "Some parameters are not finite"
    assert model.training, "Model should be in training mode by default"

    return model
