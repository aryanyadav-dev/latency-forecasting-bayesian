"""Tests for loss functions module."""

import pytest
import torch
import torch.nn.functional as F

from training.loss_functions import (LossComputer,
                                     compute_latent_forecasting_loss,
                                     compute_token_loss, compute_total_loss,
                                     validate_all_losses, validate_loss)


class TestComputeTokenLoss:
    """Tests for compute_token_loss function."""

    def test_basic_token_loss(self):
        """Test basic token loss computation."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_token_loss(logits, labels)

        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # Non-negative
        assert torch.isfinite(loss)  # Finite

    def test_token_loss_with_ignore_index(self):
        """Test token loss with padding tokens ignored."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Set some labels to ignore_index
        labels[:, -2:] = -100

        loss = compute_token_loss(logits, labels, ignore_index=-100)

        assert loss >= 0
        assert torch.isfinite(loss)

    def test_token_loss_matches_cross_entropy(self):
        """Test that token loss matches PyTorch cross_entropy."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_token_loss(logits, labels)

        # Compute expected loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        expected_loss = F.cross_entropy(logits_flat, labels_flat)

        assert torch.allclose(loss, expected_loss, atol=1e-6)

    def test_token_loss_shape_validation(self):
        """Test that token loss handles various batch sizes."""
        for batch_size in [1, 4, 16]:
            for seq_len in [5, 20, 50]:
                vocab_size = 100
                logits = torch.randn(batch_size, seq_len, vocab_size)
                labels = torch.randint(0, vocab_size, (batch_size, seq_len))

                loss = compute_token_loss(logits, labels)

                assert loss.dim() == 0
                assert loss >= 0


class TestComputeLatentForecastingLoss:
    """Tests for compute_latent_forecasting_loss function."""

    def test_basic_latent_loss(self):
        """Test basic latent forecasting loss computation."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        forecast_horizons = [1, 2, 5]

        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {
            k: torch.randn(batch_size, seq_len - k, latent_dim)
            for k in forecast_horizons
        }

        loss = compute_latent_forecasting_loss(
            latents, predicted_latents, forecast_horizons
        )

        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # Non-negative
        assert torch.isfinite(loss)  # Finite

    def test_latent_loss_single_horizon(self):
        """Test latent loss with single forecasting horizon."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        forecast_horizons = [1]

        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {1: torch.randn(batch_size, seq_len - 1, latent_dim)}

        loss = compute_latent_forecasting_loss(
            latents, predicted_latents, forecast_horizons
        )

        # Should equal MSE between latents[:, 1:] and predicted_latents[1]
        expected_loss = F.mse_loss(predicted_latents[1], latents[:, 1:])

        assert torch.allclose(loss, expected_loss, atol=1e-6)

    def test_latent_loss_multiple_horizons(self):
        """Test latent loss averages across multiple horizons."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        forecast_horizons = [1, 2, 5]

        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {
            k: torch.randn(batch_size, seq_len - k, latent_dim)
            for k in forecast_horizons
        }

        loss = compute_latent_forecasting_loss(
            latents, predicted_latents, forecast_horizons
        )

        # Compute expected average
        total = 0.0
        for k in forecast_horizons:
            mse = F.mse_loss(predicted_latents[k], latents[:, k:])
            total += mse
        expected_loss = total / len(forecast_horizons)

        assert torch.allclose(loss, expected_loss, atol=1e-6)

    def test_latent_loss_perfect_prediction(self):
        """Test that perfect predictions give zero loss."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        forecast_horizons = [1, 2]

        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {k: latents[:, k:, :].clone() for k in forecast_horizons}

        loss = compute_latent_forecasting_loss(
            latents, predicted_latents, forecast_horizons
        )

        assert loss < 1e-6  # Should be very close to zero

    def test_latent_loss_empty_horizons_raises(self):
        """Test that empty forecast_horizons raises ValueError."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {}

        with pytest.raises(ValueError, match="must contain at least one horizon"):
            compute_latent_forecasting_loss(latents, predicted_latents, [])

    def test_latent_loss_invalid_horizon_raises(self):
        """Test that invalid horizons raise ValueError."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        latents = torch.randn(batch_size, seq_len, latent_dim)

        # Horizon too large
        predicted_latents = {100: torch.randn(batch_size, 1, latent_dim)}
        with pytest.raises(ValueError, match="must be less than sequence length"):
            compute_latent_forecasting_loss(latents, predicted_latents, [100])

        # Negative horizon
        predicted_latents = {-1: torch.randn(batch_size, seq_len, latent_dim)}
        with pytest.raises(ValueError, match="must be positive"):
            compute_latent_forecasting_loss(latents, predicted_latents, [-1])

    def test_latent_loss_missing_horizon_raises(self):
        """Test that missing horizon in dict raises KeyError."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {1: torch.randn(batch_size, seq_len - 1, latent_dim)}

        with pytest.raises(KeyError, match="Horizon 2 not found"):
            compute_latent_forecasting_loss(latents, predicted_latents, [1, 2])

    def test_latent_loss_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        batch_size, seq_len, latent_dim = 2, 20, 64
        latents = torch.randn(batch_size, seq_len, latent_dim)

        # Wrong shape for predicted latents
        predicted_latents = {1: torch.randn(batch_size, seq_len - 2, latent_dim)}

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_latent_forecasting_loss(latents, predicted_latents, [1])


class TestComputeTotalLoss:
    """Tests for compute_total_loss function."""

    def test_total_loss_with_latent(self):
        """Test total loss computation with latent loss."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(0.5)
        lambda_latent = 0.1

        total = compute_total_loss(token_loss, latent_loss, lambda_latent)

        expected = token_loss + lambda_latent * latent_loss
        assert torch.allclose(total, expected)

    def test_total_loss_without_latent(self):
        """Test total loss when latent loss is None."""
        token_loss = torch.tensor(2.0)

        total = compute_total_loss(token_loss, None, lambda_latent=0.1)

        assert torch.allclose(total, token_loss)

    def test_total_loss_zero_lambda(self):
        """Test total loss with lambda=0 ignores latent loss."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(0.5)

        total = compute_total_loss(token_loss, latent_loss, lambda_latent=0.0)

        assert torch.allclose(total, token_loss)

    def test_total_loss_various_lambdas(self):
        """Test total loss with various lambda values."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(0.5)

        for lambda_val in [0.01, 0.1, 0.5, 1.0, 2.0]:
            total = compute_total_loss(token_loss, latent_loss, lambda_val)
            expected = token_loss + lambda_val * latent_loss
            assert torch.allclose(total, expected)

    def test_total_loss_negative_lambda_raises(self):
        """Test that negative lambda raises ValueError."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(0.5)

        with pytest.raises(ValueError, match="must be non-negative"):
            compute_total_loss(token_loss, latent_loss, lambda_latent=-0.1)


class TestValidateLoss:
    """Tests for validate_loss function."""

    def test_validate_finite_loss(self):
        """Test that finite loss passes validation."""
        loss = torch.tensor(1.5)
        validate_loss(loss)  # Should not raise

    def test_validate_zero_loss(self):
        """Test that zero loss passes validation."""
        loss = torch.tensor(0.0)
        validate_loss(loss)  # Should not raise

    def test_validate_nan_raises(self):
        """Test that NaN loss raises ValueError."""
        loss = torch.tensor(float("nan"))

        with pytest.raises(ValueError, match="is NaN"):
            validate_loss(loss)

    def test_validate_inf_raises(self):
        """Test that Inf loss raises ValueError."""
        loss = torch.tensor(float("inf"))

        with pytest.raises(ValueError, match="is Inf"):
            validate_loss(loss)

    def test_validate_negative_raises(self):
        """Test that negative loss raises ValueError."""
        loss = torch.tensor(-1.0)

        with pytest.raises(ValueError, match="is negative"):
            validate_loss(loss)

    def test_validate_custom_name(self):
        """Test that custom loss name appears in error message."""
        loss = torch.tensor(float("nan"))

        with pytest.raises(ValueError, match="custom_loss is NaN"):
            validate_loss(loss, "custom_loss")


class TestValidateAllLosses:
    """Tests for validate_all_losses function."""

    def test_validate_all_valid_losses(self):
        """Test that all valid losses pass validation."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(0.5)
        total_loss = torch.tensor(2.05)

        validate_all_losses(token_loss, latent_loss, total_loss)  # Should not raise

    def test_validate_all_without_latent(self):
        """Test validation when latent loss is None."""
        token_loss = torch.tensor(2.0)
        total_loss = torch.tensor(2.0)

        validate_all_losses(token_loss, None, total_loss)  # Should not raise

    def test_validate_all_invalid_token_loss(self):
        """Test that invalid token loss raises."""
        token_loss = torch.tensor(float("nan"))
        latent_loss = torch.tensor(0.5)
        total_loss = torch.tensor(2.05)

        with pytest.raises(ValueError, match="token_loss"):
            validate_all_losses(token_loss, latent_loss, total_loss)

    def test_validate_all_invalid_latent_loss(self):
        """Test that invalid latent loss raises."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(float("inf"))
        total_loss = torch.tensor(2.05)

        with pytest.raises(ValueError, match="latent_loss"):
            validate_all_losses(token_loss, latent_loss, total_loss)

    def test_validate_all_invalid_total_loss(self):
        """Test that invalid total loss raises."""
        token_loss = torch.tensor(2.0)
        latent_loss = torch.tensor(0.5)
        total_loss = torch.tensor(-1.0)

        with pytest.raises(ValueError, match="total_loss"):
            validate_all_losses(token_loss, latent_loss, total_loss)


class TestLossComputer:
    """Tests for LossComputer class."""

    def test_loss_computer_initialization(self):
        """Test LossComputer initialization."""
        computer = LossComputer(lambda_latent=0.2, ignore_index=-100)

        assert computer.lambda_latent == 0.2
        assert computer.ignore_index == -100
        assert computer.validate is True

    def test_compute_losses_token_only(self):
        """Test computing token loss only."""
        computer = LossComputer(lambda_latent=0.1)

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        losses = computer.compute_losses(logits, labels)

        assert "token_loss" in losses
        assert "total_loss" in losses
        assert "latent_loss" not in losses
        assert losses["token_loss"] >= 0
        assert losses["total_loss"] >= 0
        assert torch.allclose(losses["token_loss"], losses["total_loss"])

    def test_compute_losses_with_latent(self):
        """Test computing both token and latent losses."""
        computer = LossComputer(lambda_latent=0.1)

        batch_size, seq_len, vocab_size, latent_dim = 2, 20, 100, 64
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        latents = torch.randn(batch_size, seq_len, latent_dim)
        predicted_latents = {
            1: torch.randn(batch_size, seq_len - 1, latent_dim),
            2: torch.randn(batch_size, seq_len - 2, latent_dim),
        }
        forecast_horizons = [1, 2]

        losses = computer.compute_losses(
            logits, labels, latents, predicted_latents, forecast_horizons
        )

        assert "token_loss" in losses
        assert "latent_loss" in losses
        assert "total_loss" in losses
        assert losses["token_loss"] >= 0
        assert losses["latent_loss"] >= 0
        assert losses["total_loss"] >= 0

        # Verify composition
        expected_total = losses["token_loss"] + 0.1 * losses["latent_loss"]
        assert torch.allclose(losses["total_loss"], expected_total, atol=1e-6)

    def test_compute_losses_validation_disabled(self):
        """Test that validation can be disabled."""
        computer = LossComputer(validate=False)

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Should not raise even with invalid losses (though we can't easily create them)
        losses = computer.compute_losses(logits, labels)

        assert "token_loss" in losses
        assert "total_loss" in losses

    def test_compute_losses_with_ignore_index(self):
        """Test that ignore_index is used in token loss."""
        computer = LossComputer(ignore_index=-100)

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, -2:] = -100  # Mask last 2 positions

        losses = computer.compute_losses(logits, labels)

        # Loss should be computed only on non-masked positions
        assert losses["token_loss"] >= 0
        assert torch.isfinite(losses["token_loss"])


class TestLossProperties:
    """Property-based tests for loss functions."""

    def test_loss_non_negativity_property(self):
        """Test that all losses are always non-negative."""
        for _ in range(10):
            batch_size = torch.randint(1, 8, (1,)).item()
            seq_len = torch.randint(10, 50, (1,)).item()
            vocab_size = torch.randint(50, 200, (1,)).item()
            latent_dim = torch.randint(32, 128, (1,)).item()

            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            latents = torch.randn(batch_size, seq_len, latent_dim)
            predicted_latents = {1: torch.randn(batch_size, seq_len - 1, latent_dim)}

            token_loss = compute_token_loss(logits, labels)
            latent_loss = compute_latent_forecasting_loss(
                latents, predicted_latents, [1]
            )
            total_loss = compute_total_loss(token_loss, latent_loss, 0.1)

            assert token_loss >= 0
            assert latent_loss >= 0
            assert total_loss >= 0

    def test_loss_composition_property(self):
        """Test that total loss composition is always correct."""
        for _ in range(10):
            token_loss = torch.rand(1).item() * 5
            latent_loss = torch.rand(1).item() * 2
            lambda_val = torch.rand(1).item()

            token_loss_tensor = torch.tensor(token_loss)
            latent_loss_tensor = torch.tensor(latent_loss)

            total = compute_total_loss(
                token_loss_tensor, latent_loss_tensor, lambda_val
            )
            expected = token_loss + lambda_val * latent_loss

            assert abs(total.item() - expected) < 1e-5

    def test_loss_non_negativity_comprehensive(self):
        """Comprehensive test for loss non-negativity across various configurations."""
        # Test with different batch sizes, sequence lengths, and dimensions
        test_configs = [
            (1, 5, 50, 32),
            (2, 10, 100, 64),
            (4, 20, 200, 128),
            (8, 50, 500, 256),
        ]

        for batch_size, seq_len, vocab_size, latent_dim in test_configs:
            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            latents = torch.randn(batch_size, seq_len, latent_dim)

            # Test with multiple horizons
            horizons = [1, 2, 5] if seq_len > 5 else [1]
            predicted_latents = {
                k: torch.randn(batch_size, seq_len - k, latent_dim) for k in horizons
            }

            token_loss = compute_token_loss(logits, labels)
            latent_loss = compute_latent_forecasting_loss(
                latents, predicted_latents, horizons
            )

            # Test with various lambda values
            for lambda_val in [0.0, 0.01, 0.1, 0.5, 1.0]:
                total_loss = compute_total_loss(token_loss, latent_loss, lambda_val)

                assert token_loss >= 0, f"Token loss negative for config {test_configs}"
                assert (
                    latent_loss >= 0
                ), f"Latent loss negative for config {test_configs}"
                assert total_loss >= 0, f"Total loss negative for config {test_configs}"
                assert torch.isfinite(token_loss), "Token loss not finite"
                assert torch.isfinite(latent_loss), "Latent loss not finite"
                assert torch.isfinite(total_loss), "Total loss not finite"

    def test_loss_composition_exact(self):
        """Test exact loss composition: total = token + λ*latent."""
        for _ in range(20):
            token_loss = torch.rand(1) * 10
            latent_loss = torch.rand(1) * 5
            lambda_val = torch.rand(1).item() * 2

            total_loss = compute_total_loss(token_loss, latent_loss, lambda_val)
            expected = token_loss + lambda_val * latent_loss

            # Test exact equality
            assert torch.allclose(
                total_loss, expected, atol=1e-7
            ), f"Loss composition failed: {total_loss} != {expected}"

    def test_loss_composition_without_latent(self):
        """Test loss composition when latent loss is None."""
        for _ in range(10):
            token_loss = torch.rand(1) * 10
            lambda_val = torch.rand(1).item()

            total_loss = compute_total_loss(token_loss, None, lambda_val)

            # Should equal token loss exactly
            assert torch.allclose(
                total_loss, token_loss, atol=1e-7
            ), "Total loss should equal token loss when latent is None"

    def test_loss_composition_zero_lambda(self):
        """Test that lambda=0 ignores latent loss."""
        for _ in range(10):
            token_loss = torch.rand(1) * 10
            latent_loss = torch.rand(1) * 5

            total_loss = compute_total_loss(token_loss, latent_loss, lambda_latent=0.0)

            # Should equal token loss exactly
            assert torch.allclose(
                total_loss, token_loss, atol=1e-7
            ), "Total loss should equal token loss when lambda=0"
