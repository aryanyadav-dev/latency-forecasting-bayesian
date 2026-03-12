"""
Tests for TrainingConfig dataclass.

This module tests the TrainingConfig dataclass including:
- Default values
- Configuration validation
- Invalid parameter detection
"""

import pytest

from training import TrainingConfig


class TestTrainingConfig:
    """Test suite for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that TrainingConfig has correct default values."""
        config = TrainingConfig()

        # Verify all default values match design specification
        assert config.num_epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0
        assert config.warmup_steps == 1000
        assert config.lambda_latent == 0.1
        assert config.use_mixed_precision is True
        assert config.checkpoint_every == 1000
        assert config.log_every == 100
        assert config.seed == 42
        assert config.device == "cuda"

    def test_custom_values(self):
        """Test that TrainingConfig accepts custom values."""
        config = TrainingConfig(
            num_epochs=20,
            batch_size=64,
            learning_rate=5e-4,
            weight_decay=0.02,
            gradient_accumulation_steps=4,
            max_grad_norm=2.0,
            warmup_steps=2000,
            lambda_latent=0.2,
            use_mixed_precision=False,
            checkpoint_every=500,
            log_every=50,
            seed=123,
            device="cpu",
        )

        assert config.num_epochs == 20
        assert config.batch_size == 64
        assert config.learning_rate == 5e-4
        assert config.weight_decay == 0.02
        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 2.0
        assert config.warmup_steps == 2000
        assert config.lambda_latent == 0.2
        assert config.use_mixed_precision is False
        assert config.checkpoint_every == 500
        assert config.log_every == 50
        assert config.seed == 123
        assert config.device == "cpu"

    def test_validation_success(self):
        """Test that valid configuration passes validation."""
        config = TrainingConfig()
        assert config.validate() is True

    def test_validation_invalid_num_epochs(self):
        """Test that invalid num_epochs raises ValueError."""
        config = TrainingConfig(num_epochs=0)
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            config.validate()

        config = TrainingConfig(num_epochs=-1)
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            config.validate()

    def test_validation_invalid_batch_size(self):
        """Test that invalid batch_size raises ValueError."""
        config = TrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

        config = TrainingConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_validation_invalid_learning_rate(self):
        """Test that invalid learning_rate raises ValueError."""
        config = TrainingConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

        config = TrainingConfig(learning_rate=-1e-4)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

    def test_validation_invalid_weight_decay(self):
        """Test that invalid weight_decay raises ValueError."""
        config = TrainingConfig(weight_decay=-0.01)
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            config.validate()

    def test_validation_invalid_gradient_accumulation_steps(self):
        """Test that invalid gradient_accumulation_steps raises ValueError."""
        config = TrainingConfig(gradient_accumulation_steps=0)
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be positive"
        ):
            config.validate()

        config = TrainingConfig(gradient_accumulation_steps=-1)
        with pytest.raises(
            ValueError, match="gradient_accumulation_steps must be positive"
        ):
            config.validate()

    def test_validation_invalid_max_grad_norm(self):
        """Test that invalid max_grad_norm raises ValueError."""
        config = TrainingConfig(max_grad_norm=0.0)
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            config.validate()

        config = TrainingConfig(max_grad_norm=-1.0)
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            config.validate()

    def test_validation_invalid_warmup_steps(self):
        """Test that invalid warmup_steps raises ValueError."""
        config = TrainingConfig(warmup_steps=-1)
        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            config.validate()

    def test_validation_invalid_lambda_latent(self):
        """Test that invalid lambda_latent raises ValueError."""
        config = TrainingConfig(lambda_latent=-0.1)
        with pytest.raises(ValueError, match="lambda_latent must be non-negative"):
            config.validate()

    def test_validation_invalid_checkpoint_every(self):
        """Test that invalid checkpoint_every raises ValueError."""
        config = TrainingConfig(checkpoint_every=0)
        with pytest.raises(ValueError, match="checkpoint_every must be positive"):
            config.validate()

        config = TrainingConfig(checkpoint_every=-1)
        with pytest.raises(ValueError, match="checkpoint_every must be positive"):
            config.validate()

    def test_validation_invalid_log_every(self):
        """Test that invalid log_every raises ValueError."""
        config = TrainingConfig(log_every=0)
        with pytest.raises(ValueError, match="log_every must be positive"):
            config.validate()

        config = TrainingConfig(log_every=-1)
        with pytest.raises(ValueError, match="log_every must be positive"):
            config.validate()

    def test_validation_edge_cases(self):
        """Test validation with edge case values."""
        # Zero weight_decay should be valid
        config = TrainingConfig(weight_decay=0.0)
        assert config.validate() is True

        # Zero warmup_steps should be valid
        config = TrainingConfig(warmup_steps=0)
        assert config.validate() is True

        # Zero lambda_latent should be valid (baseline mode)
        config = TrainingConfig(lambda_latent=0.0)
        assert config.validate() is True

    def test_config_immutability(self):
        """Test that config can be modified after creation."""
        config = TrainingConfig()

        # Dataclasses are mutable by default
        config.num_epochs = 20
        assert config.num_epochs == 20

        # But validation should catch invalid values
        config.num_epochs = -1
        with pytest.raises(ValueError):
            config.validate()

    def test_config_serialization(self):
        """Test that config can be converted to dict."""
        config = TrainingConfig(num_epochs=15, batch_size=48, learning_rate=2e-4)

        # Dataclasses can be converted to dict using __dict__
        config_dict = config.__dict__

        assert config_dict["num_epochs"] == 15
        assert config_dict["batch_size"] == 48
        assert config_dict["learning_rate"] == 2e-4
        assert "weight_decay" in config_dict
        assert "lambda_latent" in config_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
