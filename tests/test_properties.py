"""
Property-based tests for Latent Forecasting Network.

Tests model invariants across random inputs:
- Shape invariants for various input sizes
- Loss non-negativity and composition rules
- Determinism with fixed seeds
- Gradient flow through all components
- Checkpoint save/load reversibility
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.complete_model import (LatentForecastingModel, ModelConfig,
                                   build_model)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    defaults = dict(
        vocab_size=100,
        latent_dim=64,
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        dropout=0.0,
        forecast_horizons=[1, 2, 5],
        max_context_length=64,
        lambda_latent=0.1,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@pytest.fixture
def small_model():
    config = _make_config()
    return build_model(config, device="cpu")


# ---------------------------------------------------------------------------
# 9.1.2  Shape invariant tests
# ---------------------------------------------------------------------------


class TestShapeInvariants:
    """Output shapes must be correct for any valid input size."""

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 16),
            (2, 32),
            (4, 64),
            (8, 12),
        ],
    )
    def test_output_shapes(self, batch_size, seq_len):
        config = _make_config()
        model = build_model(config, device="cpu")
        model.eval()

        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            out = model(
                tokens, compute_latent_loss=(seq_len > max(config.forecast_horizons))
            )

        assert out.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert out.latents.shape == (batch_size, seq_len, config.latent_dim)

    def test_single_sample(self):
        config = _make_config()
        model = build_model(config, device="cpu")
        model.eval()
        tokens = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            out = model(tokens, compute_latent_loss=True)
        assert out.logits.dim() == 3

    def test_forecasting_output_shapes(self):
        config = _make_config()
        model = build_model(config, device="cpu")
        model.eval()
        tokens = torch.randint(0, config.vocab_size, (2, 32))
        with torch.no_grad():
            out = model(tokens, compute_latent_loss=True)
        for k in config.forecast_horizons:
            assert k in out.predicted_latents
            assert out.predicted_latents[k].shape == (2, 32 - k, config.latent_dim)


# ---------------------------------------------------------------------------
# 9.1.3  Loss property tests
# ---------------------------------------------------------------------------


class TestLossProperties:
    """Losses must satisfy algebraic properties."""

    def test_loss_non_negative(self, small_model):
        tokens = torch.randint(0, 100, (2, 16))
        out = small_model(tokens, compute_latent_loss=True)
        assert out.token_loss.item() >= 0
        assert out.latent_loss.item() >= 0
        assert out.total_loss.item() >= 0

    def test_loss_composition(self, small_model):
        tokens = torch.randint(0, 100, (2, 16))
        out = small_model(tokens, compute_latent_loss=True)
        expected = out.token_loss + small_model.lambda_latent * out.latent_loss
        assert torch.isclose(out.total_loss, expected, atol=1e-5)

    def test_loss_finite(self, small_model):
        tokens = torch.randint(0, 100, (2, 16))
        out = small_model(tokens, compute_latent_loss=True)
        assert torch.isfinite(out.token_loss)
        assert torch.isfinite(out.latent_loss)
        assert torch.isfinite(out.total_loss)

    def test_baseline_mode_no_latent_loss(self):
        config = _make_config(lambda_latent=0.0)
        model = build_model(config, device="cpu")
        tokens = torch.randint(0, 100, (2, 16))
        out = model(tokens, compute_latent_loss=False)
        assert out.latent_loss is None
        assert torch.isclose(out.total_loss, out.token_loss, atol=1e-6)


# ---------------------------------------------------------------------------
# 9.1.4  Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed + input → identical output."""

    def test_deterministic_forward(self):
        config = _make_config()
        tokens = torch.randint(0, 100, (2, 16))

        torch.manual_seed(42)
        m1 = LatentForecastingModel(config)
        m1.eval()
        with torch.no_grad():
            o1 = m1(tokens, compute_latent_loss=True)

        torch.manual_seed(42)
        m2 = LatentForecastingModel(config)
        m2.eval()
        with torch.no_grad():
            o2 = m2(tokens, compute_latent_loss=True)

        assert torch.allclose(o1.logits, o2.logits, atol=1e-6)
        assert torch.allclose(o1.latents, o2.latents, atol=1e-6)

    def test_repeated_forward_same_result(self, small_model):
        small_model.eval()
        tokens = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            o1 = small_model(tokens, compute_latent_loss=True)
            o2 = small_model(tokens, compute_latent_loss=True)
        assert torch.allclose(o1.logits, o2.logits, atol=1e-6)


# ---------------------------------------------------------------------------
# 9.1.5  Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients must flow through every trainable parameter."""

    def test_all_params_get_gradients(self, small_model):
        small_model.train()
        tokens = torch.randint(0, 100, (2, 16))
        out = small_model(tokens, compute_latent_loss=True)
        out.total_loss.backward()

        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # At least some gradient should be non-zero
                assert param.grad.abs().sum() > 0, f"Gradient all zeros for {name}"

    def test_gradient_finite(self, small_model):
        small_model.train()
        tokens = torch.randint(0, 100, (2, 16))
        out = small_model(tokens, compute_latent_loss=True)
        out.total_loss.backward()

        for name, param in small_model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# 9.1.6  Checkpoint reversibility tests
# ---------------------------------------------------------------------------


class TestCheckpointReversibility:
    """Save then reload must restore identical state."""

    def test_model_state_roundtrip(self, small_model):
        tokens = torch.randint(0, 100, (2, 16))
        small_model.eval()
        with torch.no_grad():
            o1 = small_model(tokens, compute_latent_loss=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
            torch.save({"model_state_dict": small_model.state_dict()}, ckpt_path)

        # Build a fresh model and load
        config = _make_config()
        fresh = build_model(config, device="cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        fresh.load_state_dict(ckpt["model_state_dict"])
        fresh.eval()

        with torch.no_grad():
            o2 = fresh(tokens, compute_latent_loss=True)

        assert torch.allclose(o1.logits, o2.logits, atol=1e-6)
        assert torch.allclose(o1.latents, o2.latents, atol=1e-6)
        assert torch.isclose(o1.total_loss, o2.total_loss, atol=1e-6)

        os.unlink(ckpt_path)

    def test_optimizer_state_roundtrip(self, small_model):
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)

        # Do one step
        tokens = torch.randint(0, 100, (2, 16))
        out = small_model(tokens, compute_latent_loss=True)
        out.total_loss.backward()
        optimizer.step()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
            torch.save(
                {
                    "model_state_dict": small_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )

        config = _make_config()
        fresh = build_model(config, device="cpu")
        opt2 = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        fresh.load_state_dict(ckpt["model_state_dict"])
        opt2.load_state_dict(ckpt["optimizer_state_dict"])

        # Verify optimizer state keys match
        for key in optimizer.state_dict()["state"]:
            assert key in opt2.state_dict()["state"]

        os.unlink(ckpt_path)
