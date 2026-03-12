"""
Latent Forecasting Network component.

Predicts future latent states at multiple time horizons using separate
prediction heads for each forecasting horizon.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class LatentForecastingNetwork(nn.Module):
    """
    Neural network that predicts future latent states at multiple time horizons.

    The network maintains separate prediction heads for each forecasting horizon
    to enable multi-step ahead prediction while maintaining causal ordering.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        forecast_horizons: List[int],
        dropout: float = 0.1,
    ):
        """
        Initialize forecasting network with separate heads per horizon.

        Args:
            latent_dim: Dimension of latent representations
            hidden_dim: Dimension of hidden layer in prediction heads
            forecast_horizons: List of forecasting horizons (e.g., [1, 2, 5, 10])
            dropout: Dropout probability

        Preconditions:
            - latent_dim > 0
            - hidden_dim > 0
            - All forecast_horizons > 0
            - 0 <= dropout < 1
        """
        super().__init__()

        # Validate inputs
        assert latent_dim > 0, "latent_dim must be positive"
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert all(
            h > 0 for h in forecast_horizons
        ), "All forecast_horizons must be positive"
        assert 0 <= dropout < 1, "dropout must be in [0, 1)"

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizons = sorted(forecast_horizons)  # Sort for consistency
        self.dropout = dropout

        # Create separate prediction head for each horizon
        self.prediction_heads = nn.ModuleDict()

        for horizon in self.forecast_horizons:
            # Each head is a small MLP: latent_dim -> hidden_dim -> latent_dim
            head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.Dropout(dropout),
            )
            self.prediction_heads[str(horizon)] = head

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for head in self.prediction_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, latents: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Predict future latent states at multiple horizons.

        Args:
            latents: Current latent states [batch_size, seq_len, latent_dim]

        Returns:
            predictions: Dict mapping horizon k to predicted latents
                        {k: [batch_size, seq_len-k, latent_dim]}

        Preconditions:
            - latents.shape == [batch_size, seq_len, latent_dim]
            - seq_len > max(forecast_horizons)
            - All latent values are finite

        Postconditions:
            - Returns dict with keys matching forecast_horizons
            - For horizon k: output shape is [batch_size, seq_len-k, latent_dim]
            - All predictions are finite
            - Predictions maintain causal ordering (z_hat_{t+k} depends only on z_<=t)
        """
        batch_size, seq_len, latent_dim = latents.shape

        # Validate preconditions
        assert latents.dim() == 3, f"Expected 3D tensor, got {latents.dim()}D"
        assert (
            latent_dim == self.latent_dim
        ), f"Latent dimension mismatch: expected {self.latent_dim}, got {latent_dim}"
        assert seq_len > max(
            self.forecast_horizons
        ), f"Sequence length {seq_len} must be greater than max horizon {max(self.forecast_horizons)}"
        assert torch.isfinite(latents).all(), "Input latents contain NaN or Inf values"

        predictions = {}

        # Generate predictions for each horizon
        for horizon in self.forecast_horizons:
            # For horizon k, we predict z_{t+k} from z_t
            # Input: z_t for t in [0, seq_len-k)
            # Output: z_hat_{t+k} for t in [0, seq_len-k)

            # Extract latents up to position seq_len-k (to maintain causality)
            input_latents = latents[
                :, :-horizon, :
            ]  # [batch_size, seq_len-k, latent_dim]

            # Apply prediction head for this horizon
            head = self.prediction_heads[str(horizon)]
            predicted_latents = head(
                input_latents
            )  # [batch_size, seq_len-k, latent_dim]

            # Store predictions
            predictions[horizon] = predicted_latents

            # Validate output shape
            expected_shape = (batch_size, seq_len - horizon, latent_dim)
            assert predicted_latents.shape == expected_shape, (
                f"Prediction shape mismatch for horizon {horizon}: "
                f"expected {expected_shape}, got {predicted_latents.shape}"
            )

        # Validate postconditions
        assert set(predictions.keys()) == set(
            self.forecast_horizons
        ), "Predictions must be generated for all forecast horizons"
        assert all(
            torch.isfinite(pred).all() for pred in predictions.values()
        ), "Predictions contain NaN or Inf values"

        return predictions
