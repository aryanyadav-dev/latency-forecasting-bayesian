"""Loss functions for Latent Forecasting Network.

This module implements the loss computation functions for training the LFN model:
- Token prediction loss (cross-entropy)
- Latent forecasting loss (MSE)
- Total loss with configurable weighting
- Loss validation utilities
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_token_loss(
    logits: Tensor, labels: Tensor, ignore_index: int = -100
) -> Tensor:
    """
    Compute cross-entropy loss for token prediction.

    Args:
        logits: Model output logits with shape [batch_size, seq_len, vocab_size]
        labels: Target token IDs with shape [batch_size, seq_len]
        ignore_index: Token ID to ignore in loss computation (e.g., padding)

    Returns:
        Scalar tensor containing the average cross-entropy loss

    Preconditions:
        - logits has shape [batch_size, seq_len, vocab_size]
        - labels has shape [batch_size, seq_len]
        - All logit values are finite
        - All label values are valid token IDs or ignore_index

    Postconditions:
        - Returns non-negative scalar tensor
        - Loss is finite (no NaN or Inf)
    """
    # Reshape for cross-entropy: [batch_size * seq_len, vocab_size]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits_flat, labels_flat, ignore_index=ignore_index, reduction="mean"
    )

    return loss


def compute_latent_forecasting_loss(
    latents: Tensor, predicted_latents: Dict[int, Tensor], forecast_horizons: list
) -> Tensor:
    """
    Compute multi-horizon latent forecasting loss using MSE.

    This function computes the mean squared error between predicted and actual
    latent states across multiple forecasting horizons, then averages across
    all horizons.

    Args:
        latents: Ground truth latent states with shape [batch_size, seq_len, latent_dim]
        predicted_latents: Dict mapping horizon k to predicted latents
                          {k: [batch_size, seq_len-k, latent_dim]}
        forecast_horizons: List of forecasting horizons to compute loss for

    Returns:
        Scalar tensor containing the average MSE loss across all horizons

    Preconditions:
        - latents has shape [batch_size, seq_len, latent_dim]
        - For each k in forecast_horizons:
            predicted_latents[k] has shape [batch_size, seq_len-k, latent_dim]
        - All tensor values are finite
        - seq_len > max(forecast_horizons)
        - len(forecast_horizons) > 0

    Postconditions:
        - Returns non-negative scalar tensor
        - Loss is finite (no NaN or Inf)
        - Loss represents average prediction error across all horizons
    """
    if len(forecast_horizons) == 0:
        raise ValueError("forecast_horizons must contain at least one horizon")

    total_loss = 0.0
    num_horizons = len(forecast_horizons)

    for k in forecast_horizons:
        # Validate horizon
        if k <= 0:
            raise ValueError(f"Forecast horizon must be positive, got {k}")
        if k >= latents.size(1):
            raise ValueError(
                f"Forecast horizon {k} must be less than sequence length {latents.size(1)}"
            )

        # Get target latents at time t+k
        target_latents = latents[:, k:, :]  # [batch_size, seq_len-k, latent_dim]

        # Get predicted latents for horizon k
        if k not in predicted_latents:
            raise KeyError(f"Horizon {k} not found in predicted_latents")

        pred_latents = predicted_latents[k]  # [batch_size, seq_len-k, latent_dim]

        # Validate shapes match
        if target_latents.shape != pred_latents.shape:
            raise ValueError(
                f"Shape mismatch for horizon {k}: "
                f"target {target_latents.shape} vs pred {pred_latents.shape}"
            )

        # Compute MSE for this horizon
        mse = F.mse_loss(pred_latents, target_latents, reduction="mean")
        total_loss += mse

    # Average across horizons
    avg_loss = total_loss / num_horizons

    return avg_loss


def compute_total_loss(
    token_loss: Tensor, latent_loss: Optional[Tensor], lambda_latent: float = 0.1
) -> Tensor:
    """
    Compute total loss as weighted combination of token and latent losses.

    Total loss = token_loss + λ * latent_loss

    Args:
        token_loss: Cross-entropy loss for token prediction (scalar)
        latent_loss: MSE loss for latent forecasting (scalar or None)
        lambda_latent: Weight for latent loss (λ parameter)

    Returns:
        Scalar tensor containing the total weighted loss

    Preconditions:
        - token_loss is non-negative scalar tensor
        - latent_loss is non-negative scalar tensor or None
        - lambda_latent >= 0
        - All losses are finite

    Postconditions:
        - Returns non-negative scalar tensor
        - If latent_loss is None: total_loss == token_loss
        - If latent_loss is not None: total_loss == token_loss + λ * latent_loss
        - Total loss is finite
    """
    if lambda_latent < 0:
        raise ValueError(f"lambda_latent must be non-negative, got {lambda_latent}")

    # Start with token loss
    total = token_loss

    # Add weighted latent loss if provided
    if latent_loss is not None:
        total = total + lambda_latent * latent_loss

    return total


def validate_loss(loss: Tensor, loss_name: str = "loss") -> None:
    """
    Validate that a loss value is finite and non-negative.

    Args:
        loss: Loss tensor to validate
        loss_name: Name of the loss for error messages

    Raises:
        ValueError: If loss is NaN, Inf, or negative

    Preconditions:
        - loss is a tensor

    Postconditions:
        - If function returns, loss is finite and non-negative
        - If loss is invalid, ValueError is raised
    """
    if not torch.isfinite(loss):
        if torch.isnan(loss):
            raise ValueError(f"{loss_name} is NaN")
        else:
            raise ValueError(f"{loss_name} is Inf")

    if loss < 0:
        raise ValueError(f"{loss_name} is negative: {loss.item()}")


def validate_all_losses(
    token_loss: Tensor, latent_loss: Optional[Tensor], total_loss: Tensor
) -> None:
    """
    Validate all loss components are finite and non-negative.

    Args:
        token_loss: Token prediction loss
        latent_loss: Latent forecasting loss (or None)
        total_loss: Total combined loss

    Raises:
        ValueError: If any loss is NaN, Inf, or negative

    Preconditions:
        - All losses are tensors (or latent_loss is None)

    Postconditions:
        - If function returns, all losses are valid
        - If any loss is invalid, ValueError is raised with descriptive message
    """
    validate_loss(token_loss, "token_loss")

    if latent_loss is not None:
        validate_loss(latent_loss, "latent_loss")

    validate_loss(total_loss, "total_loss")


class LossComputer:
    """
    Utility class for computing and tracking losses during training.

    This class encapsulates loss computation logic and provides a convenient
    interface for computing all losses at once with automatic validation.
    """

    def __init__(
        self,
        lambda_latent: float = 0.1,
        ignore_index: int = -100,
        validate: bool = True,
    ):
        """
        Initialize loss computer.

        Args:
            lambda_latent: Weight for latent forecasting loss
            ignore_index: Token ID to ignore in token loss computation
            validate: Whether to validate losses for NaN/Inf
        """
        self.lambda_latent = lambda_latent
        self.ignore_index = ignore_index
        self.validate = validate

    def compute_losses(
        self,
        logits: Tensor,
        labels: Tensor,
        latents: Optional[Tensor] = None,
        predicted_latents: Optional[Dict[int, Tensor]] = None,
        forecast_horizons: Optional[list] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all losses for a batch.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            latents: Ground truth latents [batch_size, seq_len, latent_dim] (optional)
            predicted_latents: Predicted latents dict (optional)
            forecast_horizons: List of horizons (optional)

        Returns:
            Dictionary containing:
                - 'token_loss': Token prediction loss
                - 'latent_loss': Latent forecasting loss (if computed)
                - 'total_loss': Total weighted loss

        Raises:
            ValueError: If losses are invalid and validate=True
        """
        # Compute token loss
        token_loss = compute_token_loss(logits, labels, self.ignore_index)

        # Compute latent loss if components provided
        latent_loss = None
        if (
            latents is not None
            and predicted_latents is not None
            and forecast_horizons is not None
        ):
            latent_loss = compute_latent_forecasting_loss(
                latents, predicted_latents, forecast_horizons
            )

        # Compute total loss
        total_loss = compute_total_loss(token_loss, latent_loss, self.lambda_latent)

        # Validate if requested
        if self.validate:
            validate_all_losses(token_loss, latent_loss, total_loss)

        # Return all losses
        result = {"token_loss": token_loss, "total_loss": total_loss}

        if latent_loss is not None:
            result["latent_loss"] = latent_loss

        return result
