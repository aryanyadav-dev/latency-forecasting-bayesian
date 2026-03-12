"""Integration tests for loss functions with the complete model."""

import pytest
import torch

from models.complete_model import (LatentForecastingModel, ModelConfig,
                                   build_model)
from training.loss_functions import (LossComputer,
                                     compute_latent_forecasting_loss,
                                     compute_token_loss, compute_total_loss)


class TestLossIntegration:
    """Test loss functions integrate correctly with the model."""

    @pytest.fixture
    def model_config(self):
        """Create a small model configuration for testing."""
        return ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.1,
            forecast_horizons=[1, 2, 5],
            max_context_length=50,
        )

    @pytest.fixture
    def model(self, model_config):
        """Create a model instance."""
        return build_model(model_config, device="cpu")

    def test_loss_functions_match_model_methods(self, model):
        """Test that standalone loss functions match model's internal methods."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output
        output = model(tokens, compute_latent_loss=True)

        # Compute losses using standalone functions
        token_loss = compute_token_loss(output.logits, tokens)
        latent_loss = compute_latent_forecasting_loss(
            output.latents, output.predicted_latents, model.config.forecast_horizons
        )

        # Compare with model's computed losses
        assert torch.allclose(token_loss, output.token_loss, atol=1e-5)
        assert torch.allclose(latent_loss, output.latent_loss, atol=1e-5)

    def test_loss_computer_with_model(self, model):
        """Test LossComputer works with model outputs."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output
        output = model(tokens, compute_latent_loss=True)

        # Use LossComputer
        computer = LossComputer(lambda_latent=model.config.lambda_latent)
        losses = computer.compute_losses(
            output.logits,
            tokens,
            output.latents,
            output.predicted_latents,
            model.config.forecast_horizons,
        )

        # Verify losses are computed correctly
        assert "token_loss" in losses
        assert "latent_loss" in losses
        assert "total_loss" in losses

        # Check composition
        expected_total = (
            losses["token_loss"] + model.config.lambda_latent * losses["latent_loss"]
        )
        assert torch.allclose(losses["total_loss"], expected_total, atol=1e-5)

    def test_loss_backward_pass(self, model):
        """Test that losses support backward pass."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output
        output = model(tokens, compute_latent_loss=True)

        # Compute total loss using standalone functions
        token_loss = compute_token_loss(output.logits, tokens)
        latent_loss = compute_latent_forecasting_loss(
            output.latents, output.predicted_latents, model.config.forecast_horizons
        )
        total_loss = compute_total_loss(
            token_loss, latent_loss, model.config.lambda_latent
        )

        # Backward pass
        total_loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_loss_with_different_lambda_values(self, model):
        """Test loss computation with various lambda values."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output
        output = model(tokens, compute_latent_loss=True)

        # Test different lambda values
        for lambda_val in [0.0, 0.01, 0.1, 0.5, 1.0]:
            computer = LossComputer(lambda_latent=lambda_val)
            losses = computer.compute_losses(
                output.logits,
                tokens,
                output.latents,
                output.predicted_latents,
                model.config.forecast_horizons,
            )

            # Verify composition
            expected_total = losses["token_loss"] + lambda_val * losses["latent_loss"]
            assert torch.allclose(losses["total_loss"], expected_total, atol=1e-5)

    def test_loss_without_latent_forecasting(self, model):
        """Test loss computation when latent forecasting is disabled."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output without latent loss
        output = model(tokens, compute_latent_loss=False)

        # Compute losses
        computer = LossComputer(lambda_latent=0.1)
        losses = computer.compute_losses(output.logits, tokens)

        # Should only have token loss
        assert "token_loss" in losses
        assert "total_loss" in losses
        assert "latent_loss" not in losses
        assert torch.allclose(losses["token_loss"], losses["total_loss"])

    def test_loss_validation_catches_issues(self, model):
        """Test that loss validation catches invalid values."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output
        output = model(tokens, compute_latent_loss=True)

        # Create invalid loss (NaN)
        invalid_loss = torch.tensor(float("nan"))

        # Should raise when validating
        computer = LossComputer(validate=True)

        with pytest.raises(ValueError, match="NaN"):
            computer.compute_losses(
                output.logits * 0 + float("nan"), tokens  # Create NaN logits
            )

    def test_loss_with_padding_tokens(self, model):
        """Test loss computation with padding tokens in labels."""
        batch_size, seq_len = 2, 20
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Get model output with valid tokens
        output = model(tokens, compute_latent_loss=True)

        # Create labels with padding tokens
        labels = tokens.clone()
        labels[:, -5:] = -100  # Last 5 tokens are padding in labels

        # Compute loss with ignore_index
        computer = LossComputer(ignore_index=-100)
        losses = computer.compute_losses(
            output.logits,
            labels,  # Use labels with padding
            output.latents,
            output.predicted_latents,
            model.config.forecast_horizons,
        )

        # Loss should be computed only on non-padding tokens
        assert losses["token_loss"] >= 0
        assert torch.isfinite(losses["token_loss"])
        assert torch.isfinite(losses["total_loss"])


class TestLossGradientFlow:
    """Test gradient flow through loss functions."""

    @pytest.fixture
    def model_config(self):
        """Create a small model configuration."""
        return ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.0,  # Disable dropout for deterministic gradients
            forecast_horizons=[1, 2],
            max_context_length=30,
        )

    @pytest.fixture
    def model(self, model_config):
        """Create a model instance."""
        model = build_model(model_config, device="cpu")
        model.train()
        return model

    def test_token_loss_gradient_flow(self, model):
        """Test gradients flow through token loss."""
        batch_size, seq_len = 2, 15
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        output = model(tokens, compute_latent_loss=False)
        loss = compute_token_loss(output.logits, tokens)
        loss.backward()

        # Check encoder gradients
        assert model.encoder.token_embedding.weight.grad is not None
        assert torch.isfinite(model.encoder.token_embedding.weight.grad).all()

        # Check decoder gradients
        assert model.decoder.projection.weight.grad is not None
        assert torch.isfinite(model.decoder.projection.weight.grad).all()

    def test_latent_loss_gradient_flow(self, model):
        """Test gradients flow through latent forecasting loss."""
        batch_size, seq_len = 2, 15
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        output = model(tokens, compute_latent_loss=True)
        loss = compute_latent_forecasting_loss(
            output.latents, output.predicted_latents, model.config.forecast_horizons
        )
        loss.backward()

        # Check forecasting network gradients
        for name, param in model.forecasting_network.named_parameters():
            assert param.grad is not None, f"No gradient for forecasting_network.{name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_total_loss_gradient_flow(self, model):
        """Test gradients flow through total loss to all components."""
        batch_size, seq_len = 2, 15
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        output = model(tokens, compute_latent_loss=True)

        token_loss = compute_token_loss(output.logits, tokens)
        latent_loss = compute_latent_forecasting_loss(
            output.latents, output.predicted_latents, model.config.forecast_horizons
        )
        total_loss = compute_total_loss(token_loss, latent_loss, 0.1)
        total_loss.backward()

        # Check all components have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
