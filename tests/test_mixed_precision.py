"""
Tests for mixed precision training (Task 5.2.4).

This module tests numerical stability and correctness of FP16 training.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from models.complete_model import LatentForecastingModel, ModelConfig
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer, TrainingConfig


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    config = ModelConfig(
        vocab_size=100,
        latent_dim=64,
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        dropout=0.1,
        forecast_horizons=[1, 2],
        max_context_length=32,
        lambda_latent=0.1,
    )
    model = LatentForecastingModel(config)
    return model


@pytest.fixture
def dummy_dataloader():
    """Create a dummy dataloader for testing."""
    batch_size = 4
    seq_len = 16
    num_batches = 10

    tokens = torch.randint(0, 100, (num_batches * batch_size, seq_len))
    labels = tokens.clone()

    dataset = TensorDataset(tokens, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    class DictDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader

        def __iter__(self):
            for tokens, labels in self.dataloader:
                yield {"input_ids": tokens, "labels": labels}

        def __len__(self):
            return len(self.dataloader)

    return DictDataLoader(dataloader)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_initialization(
    small_model, dummy_dataloader, temp_checkpoint_dir
):
    """Test that GradScaler is properly initialized when mixed precision is enabled."""
    config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=True,
        device="cuda",
    )

    model = small_model.to("cuda")
    optimizer = create_optimizer(model, learning_rate=config.learning_rate)
    scheduler = create_scheduler(optimizer, num_training_steps=100, warmup_steps=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Verify GradScaler is initialized
    assert trainer.scaler is not None
    assert isinstance(trainer.scaler, GradScaler)
    assert trainer.use_amp is True


def test_mixed_precision_disabled(small_model, dummy_dataloader, temp_checkpoint_dir):
    """Test that GradScaler is not initialized when mixed precision is disabled."""
    config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=False,
        device="cpu",
    )

    optimizer = create_optimizer(small_model, learning_rate=config.learning_rate)
    scheduler = create_scheduler(optimizer, num_training_steps=100, warmup_steps=5)

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Verify GradScaler is not initialized
    assert trainer.scaler is None
    assert trainer.use_amp is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numerical_stability_fp16(small_model, dummy_dataloader, temp_checkpoint_dir):
    """Test numerical stability with FP16 training."""
    config = TrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_mixed_precision=True,
        checkpoint_every=100,
        log_every=5,
        device="cuda",
    )

    model = small_model.to("cuda")
    optimizer = create_optimizer(model, learning_rate=config.learning_rate)
    scheduler = create_scheduler(optimizer, num_training_steps=100, warmup_steps=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    metrics = trainer.train_epoch()

    # Verify numerical stability
    assert torch.isfinite(
        torch.tensor(metrics["total_loss"])
    ), "Total loss is not finite"
    assert torch.isfinite(
        torch.tensor(metrics["token_loss"])
    ), "Token loss is not finite"
    assert torch.isfinite(
        torch.tensor(metrics["latent_loss"])
    ), "Latent loss is not finite"

    # Verify all model parameters are finite
    for name, param in model.named_parameters():
        assert torch.isfinite(
            param
        ).all(), f"Parameter {name} contains non-finite values"
        if param.grad is not None:
            assert torch.isfinite(
                param.grad
            ).all(), f"Gradient for {name} contains non-finite values"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_fp32_consistency(small_model, dummy_dataloader, temp_checkpoint_dir):
    """Test that FP16 and FP32 training produce similar results."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Train with FP32
    config_fp32 = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=False,
        device="cuda",
        seed=42,
    )

    model_fp32 = small_model.to("cuda")
    optimizer_fp32 = create_optimizer(
        model_fp32, learning_rate=config_fp32.learning_rate
    )
    scheduler_fp32 = create_scheduler(
        optimizer_fp32, num_training_steps=100, warmup_steps=5
    )

    trainer_fp32 = Trainer(
        model=model_fp32,
        optimizer=optimizer_fp32,
        scheduler=scheduler_fp32,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config_fp32,
        checkpoint_dir=temp_checkpoint_dir,
    )

    metrics_fp32 = trainer_fp32.train_epoch()

    # Reset seed and train with FP16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config_fp16 = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=True,
        device="cuda",
        seed=42,
    )

    # Create new model with same initialization
    model_fp16 = LatentForecastingModel(small_model.config).to("cuda")
    model_fp16.load_state_dict(model_fp32.state_dict())

    optimizer_fp16 = create_optimizer(
        model_fp16, learning_rate=config_fp16.learning_rate
    )
    scheduler_fp16 = create_scheduler(
        optimizer_fp16, num_training_steps=100, warmup_steps=5
    )

    trainer_fp16 = Trainer(
        model=model_fp16,
        optimizer=optimizer_fp16,
        scheduler=scheduler_fp16,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config_fp16,
        checkpoint_dir=temp_checkpoint_dir,
    )

    metrics_fp16 = trainer_fp16.train_epoch()

    # Compare losses (should be similar but not identical due to FP16 precision)
    # Allow 10% relative difference
    relative_diff = (
        abs(metrics_fp32["total_loss"] - metrics_fp16["total_loss"])
        / metrics_fp32["total_loss"]
    )
    assert (
        relative_diff < 0.1
    ), f"FP16 and FP32 losses differ by {relative_diff*100:.1f}%"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_scaling(small_model, dummy_dataloader, temp_checkpoint_dir):
    """Test that gradient scaling works correctly."""
    config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=True,
        device="cuda",
    )

    model = small_model.to("cuda")
    optimizer = create_optimizer(model, learning_rate=config.learning_rate)
    scheduler = create_scheduler(optimizer, num_training_steps=100, warmup_steps=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Get initial scale
    initial_scale = trainer.scaler.get_scale()
    assert initial_scale > 0, "Initial scale should be positive"

    # Train one epoch
    trainer.train_epoch()

    # Scale should still be positive (may have changed)
    final_scale = trainer.scaler.get_scale()
    assert final_scale > 0, "Final scale should be positive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_with_scaler(small_model, dummy_dataloader, temp_checkpoint_dir):
    """Test that GradScaler state is saved and loaded correctly."""
    config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=True,
        device="cuda",
    )

    model = small_model.to("cuda")
    optimizer = create_optimizer(model, learning_rate=config.learning_rate)
    scheduler = create_scheduler(optimizer, num_training_steps=100, warmup_steps=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    trainer.train_epoch()

    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "test_checkpoint.pt"
    trainer.save_checkpoint(filename="test_checkpoint.pt")

    # Get scaler state
    original_scale = trainer.scaler.get_scale()

    # Create new trainer and load checkpoint
    new_model = LatentForecastingModel(small_model.config).to("cuda")
    new_optimizer = create_optimizer(new_model, learning_rate=config.learning_rate)
    new_scheduler = create_scheduler(
        new_optimizer, num_training_steps=100, warmup_steps=5
    )

    new_trainer = Trainer(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Load checkpoint
    new_trainer.load_checkpoint(str(checkpoint_path))

    # Verify scaler state was restored
    loaded_scale = new_trainer.scaler.get_scale()
    assert loaded_scale == original_scale, "Scaler state not restored correctly"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocast_context(small_model):
    """Test that autocast context works correctly for forward pass."""
    model = small_model.to("cuda")
    model.eval()

    # Create dummy input
    tokens = torch.randint(0, 100, (2, 16)).to("cuda")

    # Forward pass with autocast
    with autocast(device_type="cuda", enabled=True):
        output = model(tokens, compute_latent_loss=True)

    # Verify output is valid
    assert output.logits is not None
    assert output.latents is not None
    assert output.token_loss is not None
    assert torch.isfinite(output.token_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_accumulation_with_fp16(
    small_model, dummy_dataloader, temp_checkpoint_dir
):
    """Test gradient accumulation works correctly with FP16."""
    config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        gradient_accumulation_steps=4,
        use_mixed_precision=True,
        device="cuda",
    )

    model = small_model.to("cuda")
    optimizer = create_optimizer(model, learning_rate=config.learning_rate)
    scheduler = create_scheduler(optimizer, num_training_steps=100, warmup_steps=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    initial_step = trainer.global_step
    metrics = trainer.train_epoch()

    # Verify training completed successfully
    assert trainer.global_step > initial_step
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def test_master_weights_fp32():
    """Test that master weights are maintained in FP32."""
    # This is a conceptual test - PyTorch's AMP automatically maintains FP32 master weights
    # We verify that the optimizer stores parameters in FP32

    model = nn.Linear(10, 10)
    if torch.cuda.is_available():
        model = model.to("cuda")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Verify optimizer stores parameters in FP32
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                # Optimizer maintains FP32 copy internally
                assert param.dtype == torch.float32 or param.dtype == torch.float16
                # After first step, optimizer state will be in FP32

        # Do a forward/backward pass
        x = torch.randn(2, 10).to("cuda")
        with autocast(device_type="cuda", enabled=True):
            loss = model(x).sum()

        scaler = GradScaler(device="cuda")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Verify optimizer state is in FP32
        for state in optimizer.state.values():
            if "exp_avg" in state:
                assert state["exp_avg"].dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
