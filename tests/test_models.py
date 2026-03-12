"""
Comprehensive tests for all model components.

Tests cover:
- Encoder output shapes and causal masking
- Forecasting network multi-horizon predictions
- Decoder output shapes
- Complete model forward pass
- Gradient flow through all components
- Baseline models

All tests validate preconditions and postconditions, and check for finite values.
"""

from typing import Dict

import pytest
import torch
import torch.nn as nn

from models.baseline_models import (BaselineModelOutput, StandardTransformer,
                                    TransformerWithoutLatentLoss,
                                    build_baseline_model)
from models.complete_model import (LatentForecastingModel, ModelConfig,
                                   ModelOutput, build_model)
from models.decoder import Decoder
from models.encoder import Encoder, PositionalEncoding
from models.forecasting_network import LatentForecastingNetwork


# Test configuration fixtures
@pytest.fixture
def small_config():
    """Small model configuration for fast testing."""
    return ModelConfig(
        vocab_size=100,
        latent_dim=64,
        num_layers=2,
        num_heads=4,
        hidden_dim=256,
        dropout=0.1,
        forecast_horizons=[1, 2, 5],
        max_context_length=128,
        lambda_latent=0.1,
    )


@pytest.fixture
def device():
    """Get available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_tokens(device):
    """Generate sample token batch for testing."""
    batch_size = 4
    seq_len = 32
    vocab_size = 100
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return tokens


# ============================================================================
# Task 3.7.2: Test encoder output shapes and causal masking
# ============================================================================


class TestEncoder:
    """Tests for Encoder component."""

    def test_encoder_output_shape(self, small_config, device):
        """Test that encoder outputs correct shape [batch_size, seq_len, latent_dim]."""
        encoder = Encoder(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        batch_size = 4
        seq_len = 32
        tokens = torch.randint(
            0, small_config.vocab_size, (batch_size, seq_len), device=device
        )

        latents = encoder(tokens)

        # Validate output shape
        assert latents.shape == (
            batch_size,
            seq_len,
            small_config.latent_dim,
        ), f"Expected shape {(batch_size, seq_len, small_config.latent_dim)}, got {latents.shape}"

        # Validate all values are finite
        assert torch.isfinite(latents).all(), "Encoder output contains NaN or Inf"

    def test_encoder_causal_masking(self, small_config, device):
        """Test that causal masking prevents future information leakage."""
        encoder = Encoder(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=0.0,  # Disable dropout for deterministic test
            max_context_length=small_config.max_context_length,
        ).to(device)

        encoder.eval()  # Set to eval mode for deterministic behavior

        batch_size = 2
        seq_len = 16

        # Create two token sequences that differ only in future positions
        tokens1 = torch.randint(
            0, small_config.vocab_size, (batch_size, seq_len), device=device
        )
        tokens2 = tokens1.clone()

        # Change tokens at positions > 8 (future positions relative to position 8)
        tokens2[:, 9:] = torch.randint(
            0, small_config.vocab_size, (batch_size, seq_len - 9), device=device
        )

        with torch.no_grad():
            latents1 = encoder(tokens1)
            latents2 = encoder(tokens2)

        # Latents at position 8 should be identical (no future information leakage)
        # Allow small numerical tolerance
        position = 8
        diff = torch.abs(latents1[:, position, :] - latents2[:, position, :])
        max_diff = diff.max().item()

        assert max_diff < 1e-5, (
            f"Causal masking violated: latents at position {position} differ by {max_diff} "
            f"when future tokens change"
        )

    def test_encoder_different_batch_sizes(self, small_config, device):
        """Test encoder with various batch sizes."""
        encoder = Encoder(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        seq_len = 32

        for batch_size in [1, 2, 8, 16]:
            tokens = torch.randint(
                0, small_config.vocab_size, (batch_size, seq_len), device=device
            )
            latents = encoder(tokens)

            assert latents.shape == (batch_size, seq_len, small_config.latent_dim)
            assert torch.isfinite(latents).all()

    def test_encoder_different_sequence_lengths(self, small_config, device):
        """Test encoder with various sequence lengths."""
        encoder = Encoder(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        batch_size = 4

        for seq_len in [8, 16, 32, 64]:
            tokens = torch.randint(
                0, small_config.vocab_size, (batch_size, seq_len), device=device
            )
            latents = encoder(tokens)

            assert latents.shape == (batch_size, seq_len, small_config.latent_dim)
            assert torch.isfinite(latents).all()

    def test_positional_encoding(self, small_config, device):
        """Test positional encoding component."""
        pos_enc = PositionalEncoding(
            d_model=small_config.latent_dim,
            max_len=small_config.max_context_length,
            dropout=0.0,
        ).to(device)

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, small_config.latent_dim, device=device)

        output = pos_enc(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()


# ============================================================================
# Task 3.7.3: Test forecasting network multi-horizon predictions
# ============================================================================


class TestForecastingNetwork:
    """Tests for Latent Forecasting Network component."""

    def test_forecasting_network_output_shapes(self, small_config, device):
        """Test that forecasting network produces correct shapes for each horizon."""
        forecasting_net = LatentForecastingNetwork(
            latent_dim=small_config.latent_dim,
            hidden_dim=small_config.hidden_dim,
            forecast_horizons=small_config.forecast_horizons,
            dropout=small_config.dropout,
        ).to(device)

        batch_size = 4
        seq_len = 32
        latents = torch.randn(
            batch_size, seq_len, small_config.latent_dim, device=device
        )

        predictions = forecasting_net(latents)

        # Check that predictions exist for all horizons
        assert set(predictions.keys()) == set(
            small_config.forecast_horizons
        ), f"Expected horizons {small_config.forecast_horizons}, got {list(predictions.keys())}"

        # Check shape for each horizon
        for horizon in small_config.forecast_horizons:
            expected_shape = (batch_size, seq_len - horizon, small_config.latent_dim)
            assert (
                predictions[horizon].shape == expected_shape
            ), f"Horizon {horizon}: expected shape {expected_shape}, got {predictions[horizon].shape}"

            # Check all values are finite
            assert torch.isfinite(
                predictions[horizon]
            ).all(), f"Horizon {horizon}: predictions contain NaN or Inf"

    def test_forecasting_network_causal_ordering(self, small_config, device):
        """Test that predictions maintain causal ordering (no future information leakage)."""
        forecasting_net = LatentForecastingNetwork(
            latent_dim=small_config.latent_dim,
            hidden_dim=small_config.hidden_dim,
            forecast_horizons=small_config.forecast_horizons,
            dropout=0.0,  # Disable dropout for deterministic test
        ).to(device)

        forecasting_net.eval()

        batch_size = 2
        seq_len = 32

        # Create two latent sequences that differ only in future positions
        latents1 = torch.randn(
            batch_size, seq_len, small_config.latent_dim, device=device
        )
        latents2 = latents1.clone()

        # Change latents at positions > 16 (future positions relative to position 16)
        latents2[:, 17:, :] = torch.randn(
            batch_size, seq_len - 17, small_config.latent_dim, device=device
        )

        with torch.no_grad():
            predictions1 = forecasting_net(latents1)
            predictions2 = forecasting_net(latents2)

        # For horizon k=1, prediction at position 15 should be identical
        # (predicting position 16 from positions 0-15)
        position = 15
        horizon = 1

        diff = torch.abs(
            predictions1[horizon][:, position, :]
            - predictions2[horizon][:, position, :]
        )
        max_diff = diff.max().item()

        assert max_diff < 1e-5, (
            f"Causal ordering violated: predictions at position {position} differ by {max_diff} "
            f"when future latents change"
        )

    def test_forecasting_network_different_horizons(self, device):
        """Test forecasting network with different horizon configurations."""
        latent_dim = 64
        hidden_dim = 256
        batch_size = 4
        seq_len = 32

        # Test with different horizon sets
        horizon_configs = [[1], [1, 2], [1, 2, 5], [1, 2, 5, 10]]

        for horizons in horizon_configs:
            forecasting_net = LatentForecastingNetwork(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                forecast_horizons=horizons,
                dropout=0.1,
            ).to(device)

            latents = torch.randn(batch_size, seq_len, latent_dim, device=device)
            predictions = forecasting_net(latents)

            # Check all horizons are present
            assert set(predictions.keys()) == set(horizons)

            # Check shapes
            for horizon in horizons:
                expected_shape = (batch_size, seq_len - horizon, latent_dim)
                assert predictions[horizon].shape == expected_shape
                assert torch.isfinite(predictions[horizon]).all()


# ============================================================================
# Task 3.7.4: Test decoder output shapes
# ============================================================================


class TestDecoder:
    """Tests for Decoder component."""

    def test_decoder_output_shape(self, small_config, device):
        """Test that decoder outputs correct shape [batch_size, seq_len, vocab_size]."""
        decoder = Decoder(
            latent_dim=small_config.latent_dim, vocab_size=small_config.vocab_size
        ).to(device)

        batch_size = 4
        seq_len = 32
        latents = torch.randn(
            batch_size, seq_len, small_config.latent_dim, device=device
        )

        logits = decoder(latents)

        # Validate output shape
        expected_shape = (batch_size, seq_len, small_config.vocab_size)
        assert (
            logits.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {logits.shape}"

        # Validate all values are finite
        assert torch.isfinite(logits).all(), "Decoder output contains NaN or Inf"

    def test_decoder_no_activation(self, small_config, device):
        """Test that decoder produces raw logits without activation."""
        decoder = Decoder(
            latent_dim=small_config.latent_dim, vocab_size=small_config.vocab_size
        ).to(device)

        batch_size = 4
        seq_len = 32
        latents = torch.randn(
            batch_size, seq_len, small_config.latent_dim, device=device
        )

        logits = decoder(latents)

        # Raw logits should have both positive and negative values
        # (not all positive like after softmax or sigmoid)
        assert (logits > 0).any() and (
            logits < 0
        ).any(), (
            "Decoder should produce raw logits with both positive and negative values"
        )

    def test_decoder_different_batch_sizes(self, small_config, device):
        """Test decoder with various batch sizes."""
        decoder = Decoder(
            latent_dim=small_config.latent_dim, vocab_size=small_config.vocab_size
        ).to(device)

        seq_len = 32

        for batch_size in [1, 2, 8, 16]:
            latents = torch.randn(
                batch_size, seq_len, small_config.latent_dim, device=device
            )
            logits = decoder(latents)

            assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
            assert torch.isfinite(logits).all()


# ============================================================================
# Task 3.7.5: Test complete model forward pass
# ============================================================================


class TestCompleteModel:
    """Tests for complete Latent Forecasting Model."""

    def test_model_forward_pass(self, small_config, device, sample_tokens):
        """Test complete model forward pass with all components."""
        model = build_model(small_config, device)

        output = model(sample_tokens, compute_latent_loss=True)

        batch_size, seq_len = sample_tokens.shape

        # Validate output types
        assert isinstance(output, ModelOutput), "Output should be ModelOutput instance"

        # Validate shapes
        assert output.logits.shape == (
            batch_size,
            seq_len,
            small_config.vocab_size,
        ), f"Logits shape mismatch"
        assert output.latents.shape == (
            batch_size,
            seq_len,
            small_config.latent_dim,
        ), f"Latents shape mismatch"

        # Validate predicted latents
        assert set(output.predicted_latents.keys()) == set(
            small_config.forecast_horizons
        ), "Predicted latents should exist for all horizons"

        for horizon in small_config.forecast_horizons:
            expected_shape = (batch_size, seq_len - horizon, small_config.latent_dim)
            assert (
                output.predicted_latents[horizon].shape == expected_shape
            ), f"Predicted latents shape mismatch for horizon {horizon}"

        # Validate losses
        assert output.token_loss is not None, "Token loss should not be None"
        assert (
            output.latent_loss is not None
        ), "Latent loss should not be None when compute_latent_loss=True"
        assert output.total_loss is not None, "Total loss should not be None"

        # Validate loss properties
        assert output.token_loss >= 0, "Token loss must be non-negative"
        assert output.latent_loss >= 0, "Latent loss must be non-negative"
        assert output.total_loss >= 0, "Total loss must be non-negative"

        assert torch.isfinite(output.token_loss), "Token loss must be finite"
        assert torch.isfinite(output.latent_loss), "Latent loss must be finite"
        assert torch.isfinite(output.total_loss), "Total loss must be finite"

        # Validate loss composition: total = token + λ * latent
        expected_total = (
            output.token_loss + small_config.lambda_latent * output.latent_loss
        )
        assert torch.allclose(
            output.total_loss, expected_total, rtol=1e-5
        ), f"Total loss should equal token_loss + λ * latent_loss"

    def test_model_without_latent_loss(self, small_config, device, sample_tokens):
        """Test model forward pass with latent loss disabled."""
        model = build_model(small_config, device)

        output = model(sample_tokens, compute_latent_loss=False)

        # Latent loss should be None when disabled
        assert (
            output.latent_loss is None
        ), "Latent loss should be None when compute_latent_loss=False"

        # Total loss should equal token loss
        assert torch.allclose(
            output.total_loss, output.token_loss
        ), "Total loss should equal token loss when latent loss is disabled"

        # Predicted latents should be empty
        assert (
            len(output.predicted_latents) == 0
        ), "Predicted latents should be empty when compute_latent_loss=False"

    def test_model_with_custom_labels(self, small_config, device, sample_tokens):
        """Test model with custom labels (not shifted tokens)."""
        model = build_model(small_config, device)

        batch_size, seq_len = sample_tokens.shape
        custom_labels = torch.randint(
            0, small_config.vocab_size, (batch_size, seq_len), device=device
        )

        output = model(sample_tokens, labels=custom_labels, compute_latent_loss=True)

        # Should work with custom labels
        assert output.token_loss >= 0
        assert torch.isfinite(output.token_loss)

    def test_model_config_validation(self):
        """Test that model configuration validation works correctly."""
        # Valid config
        valid_config = ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=256,
            dropout=0.1,
            forecast_horizons=[1, 2, 5],
            max_context_length=128,
        )
        assert valid_config.validate()

        # Invalid: latent_dim not divisible by num_heads
        with pytest.raises(
            ValueError, match="latent_dim must be divisible by num_heads"
        ):
            invalid_config = ModelConfig(
                vocab_size=100, latent_dim=65, num_heads=4  # Not divisible by 4
            )
            invalid_config.validate()

        # Invalid: max_context_length too small
        with pytest.raises(
            ValueError,
            match="max_context_length must be greater than max forecast horizon",
        ):
            invalid_config = ModelConfig(
                vocab_size=100,
                latent_dim=64,
                forecast_horizons=[1, 2, 5, 10],
                max_context_length=8,  # Less than max horizon (10)
            )
            invalid_config.validate()

    def test_model_different_sequence_lengths(self, small_config, device):
        """Test model with various sequence lengths."""
        model = build_model(small_config, device)

        batch_size = 4

        # Test with different sequence lengths
        for seq_len in [16, 32, 64]:
            tokens = torch.randint(
                0, small_config.vocab_size, (batch_size, seq_len), device=device
            )
            output = model(tokens, compute_latent_loss=True)

            assert output.logits.shape == (batch_size, seq_len, small_config.vocab_size)
            assert output.latents.shape == (
                batch_size,
                seq_len,
                small_config.latent_dim,
            )
            assert torch.isfinite(output.total_loss)


# ============================================================================
# Task 3.7.6: Test gradient flow through all components
# ============================================================================


class TestGradientFlow:
    """Tests for gradient flow through model components."""

    def test_gradient_flow_complete_model(self, small_config, device, sample_tokens):
        """Test that gradients flow through all model components."""
        model = build_model(small_config, device)
        model.train()

        # Forward pass
        output = model(sample_tokens, compute_latent_loss=True)

        # Backward pass
        output.total_loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradients"
            assert param.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"

    def test_gradient_flow_encoder(self, small_config, device, sample_tokens):
        """Test gradient flow through encoder."""
        model = build_model(small_config, device)
        model.train()

        output = model(sample_tokens, compute_latent_loss=True)
        output.total_loss.backward()

        # Check encoder parameters have gradients
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"Encoder parameter {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Encoder parameter {name} has non-finite gradients"

    def test_gradient_flow_forecasting_network(
        self, small_config, device, sample_tokens
    ):
        """Test gradient flow through forecasting network."""
        model = build_model(small_config, device)
        model.train()

        output = model(sample_tokens, compute_latent_loss=True)
        output.total_loss.backward()

        # Check forecasting network parameters have gradients
        for name, param in model.forecasting_network.named_parameters():
            assert (
                param.grad is not None
            ), f"Forecasting network parameter {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Forecasting network parameter {name} has non-finite gradients"

    def test_gradient_flow_decoder(self, small_config, device, sample_tokens):
        """Test gradient flow through decoder."""
        model = build_model(small_config, device)
        model.train()

        output = model(sample_tokens, compute_latent_loss=True)
        output.total_loss.backward()

        # Check decoder parameters have gradients
        for name, param in model.decoder.named_parameters():
            assert param.grad is not None, f"Decoder parameter {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Decoder parameter {name} has non-finite gradients"

    def test_gradient_accumulation(self, small_config, device):
        """Test gradient accumulation across multiple batches."""
        model = build_model(small_config, device)
        model.train()

        batch_size = 4
        seq_len = 32

        # Accumulate gradients over 3 batches
        for i in range(3):
            tokens = torch.randint(
                0, small_config.vocab_size, (batch_size, seq_len), device=device
            )
            output = model(tokens, compute_latent_loss=True)
            loss = output.total_loss / 3  # Scale loss for accumulation
            loss.backward()

        # Check that gradients are accumulated (non-zero)
        for name, param in model.named_parameters():
            assert (
                param.grad is not None
            ), f"Parameter {name} has no gradient after accumulation"
            assert (
                param.grad.abs().sum() > 0
            ), f"Parameter {name} has zero gradient after accumulation"

    def test_gradient_clipping(self, small_config, device, sample_tokens):
        """Test gradient clipping functionality."""
        model = build_model(small_config, device)
        model.train()

        output = model(sample_tokens, compute_latent_loss=True)
        output.total_loss.backward()

        # Clip gradients and get the norm before clipping
        max_norm = 1.0
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm
        )

        # Compute actual gradient norm after clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm_after = total_norm**0.5

        # After clipping, norm should be <= max_norm (or close to it if it was already smaller)
        assert (
            grad_norm_after <= max_norm + 1e-5
        ), f"Gradient norm after clipping {grad_norm_after} exceeds max_norm {max_norm}"


# ============================================================================
# Task 3.7.7: Test baseline models
# ============================================================================


class TestBaselineModels:
    """Tests for baseline model implementations."""

    def test_standard_transformer(self, small_config, device, sample_tokens):
        """Test StandardTransformer baseline model."""
        model = StandardTransformer(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        output = model(sample_tokens)

        batch_size, seq_len = sample_tokens.shape

        # Validate output types
        assert isinstance(output, BaselineModelOutput)

        # Validate shapes
        assert output.logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert output.latents.shape == (batch_size, seq_len, small_config.latent_dim)

        # Validate losses
        assert output.token_loss >= 0
        assert output.total_loss >= 0
        assert torch.isfinite(output.token_loss)
        assert torch.isfinite(output.total_loss)

        # For baseline, total loss should equal token loss
        assert torch.allclose(
            output.total_loss, output.token_loss
        ), "Baseline total loss should equal token loss"

    def test_transformer_without_latent_loss(self, small_config, device, sample_tokens):
        """Test TransformerWithoutLatentLoss baseline model."""
        model = TransformerWithoutLatentLoss(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        output = model(sample_tokens)

        batch_size, seq_len = sample_tokens.shape

        # Validate output
        assert isinstance(output, BaselineModelOutput)
        assert output.logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert output.latents.shape == (batch_size, seq_len, small_config.latent_dim)

        # Validate losses
        assert output.token_loss >= 0
        assert output.total_loss >= 0
        assert torch.allclose(output.total_loss, output.token_loss)

    def test_baseline_model_factory(self, small_config, device):
        """Test build_baseline_model factory function."""
        # Test standard transformer
        model_standard = build_baseline_model(
            model_type="standard",
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
            device=device,
        )
        assert isinstance(model_standard, StandardTransformer)

        # Test transformer without latent loss
        model_no_latent = build_baseline_model(
            model_type="no_latent_loss",
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
            device=device,
        )
        assert isinstance(model_no_latent, TransformerWithoutLatentLoss)

        # Test invalid model type
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_baseline_model(
                model_type="invalid", vocab_size=small_config.vocab_size, device=device
            )

    def test_baseline_gradient_flow(self, small_config, device, sample_tokens):
        """Test gradient flow through baseline models."""
        # Test StandardTransformer
        model = StandardTransformer(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        model.train()
        output = model(sample_tokens)
        output.total_loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Baseline parameter {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Baseline parameter {name} has non-finite gradients"

    def test_baseline_generation(self, small_config, device):
        """Test text generation with baseline models."""
        model = StandardTransformer(
            vocab_size=small_config.vocab_size,
            latent_dim=small_config.latent_dim,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            hidden_dim=small_config.hidden_dim,
            dropout=small_config.dropout,
            max_context_length=small_config.max_context_length,
        ).to(device)

        batch_size = 2
        initial_seq_len = 10
        max_new_tokens = 5

        initial_tokens = torch.randint(
            0, small_config.vocab_size, (batch_size, initial_seq_len), device=device
        )

        generated = model.generate(
            initial_tokens, max_new_tokens=max_new_tokens, temperature=1.0
        )

        # Check output shape
        expected_len = initial_seq_len + max_new_tokens
        assert generated.shape == (
            batch_size,
            expected_len,
        ), f"Expected shape {(batch_size, expected_len)}, got {generated.shape}"

        # Check all tokens are valid
        assert (generated >= 0).all() and (
            generated < small_config.vocab_size
        ).all(), "Generated tokens should be in valid range"


# ============================================================================
# Additional integration tests
# ============================================================================


class TestModelIntegration:
    """Integration tests for complete model pipeline."""

    def test_training_step(self, small_config, device, sample_tokens):
        """Test a complete training step (forward + backward + optimizer step)."""
        model = build_model(small_config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()

        # Forward pass
        output = model(sample_tokens, compute_latent_loss=True)
        loss = output.total_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Check that parameters were updated
        # (This is a simple check - in practice, we'd compare before/after values)
        for param in model.parameters():
            assert param.grad is not None

    def test_model_save_load(self, small_config, device, sample_tokens, tmp_path):
        """Test model checkpoint saving and loading."""
        model = build_model(small_config, device)

        # Get initial output
        model.eval()
        with torch.no_grad():
            output_before = model(sample_tokens, compute_latent_loss=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "model_checkpoint.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": small_config},
            checkpoint_path,
        )

        # Create new model and load checkpoint
        model_loaded = build_model(small_config, device)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model_loaded.load_state_dict(checkpoint["model_state_dict"])

        # Get output from loaded model
        model_loaded.eval()
        with torch.no_grad():
            output_after = model_loaded(sample_tokens, compute_latent_loss=False)

        # Outputs should be identical
        assert torch.allclose(
            output_before.logits, output_after.logits, rtol=1e-5
        ), "Loaded model should produce identical outputs"

    def test_model_eval_mode(self, small_config, device, sample_tokens):
        """Test model behavior in eval mode (deterministic with dropout disabled)."""
        model = build_model(small_config, device)
        model.eval()

        # Run forward pass twice
        with torch.no_grad():
            output1 = model(sample_tokens, compute_latent_loss=False)
            output2 = model(sample_tokens, compute_latent_loss=False)

        # Outputs should be identical in eval mode
        assert torch.allclose(
            output1.logits, output2.logits, rtol=1e-5
        ), "Model should be deterministic in eval mode"
