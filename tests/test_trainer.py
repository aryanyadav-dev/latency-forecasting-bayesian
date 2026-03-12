"""
Tests for Trainer class.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.complete_model import LatentForecastingModel, ModelConfig
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer, TrainingConfig, TrainingHistory


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
    # Create dummy data
    batch_size = 4
    seq_len = 16
    num_batches = 10

    tokens = torch.randint(0, 100, (num_batches * batch_size, seq_len))
    labels = tokens.clone()

    dataset = TensorDataset(tokens, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Convert to dict format expected by trainer
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
def training_config():
    """Create a training configuration for testing."""
    return TrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        warmup_steps=5,
        lambda_latent=0.1,
        use_mixed_precision=False,  # Disable for CPU testing
        checkpoint_every=20,
        log_every=5,
        seed=42,
        device="cpu",
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_training_config_validation():
    """Test TrainingConfig validation."""
    # Valid config
    config = TrainingConfig()
    assert config.validate()

    # Invalid num_epochs
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        config = TrainingConfig(num_epochs=0)
        config.validate()

    # Invalid learning_rate
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        config = TrainingConfig(learning_rate=-0.1)
        config.validate()

    # Invalid gradient_accumulation_steps
    with pytest.raises(
        ValueError, match="gradient_accumulation_steps must be positive"
    ):
        config = TrainingConfig(gradient_accumulation_steps=0)
        config.validate()


def test_trainer_initialization(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test Trainer initialization."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    assert trainer.global_step == 0
    assert trainer.current_epoch == 0
    assert trainer.best_val_loss == float("inf")
    assert isinstance(trainer.history, TrainingHistory)
    assert Path(temp_checkpoint_dir).exists()


def test_train_epoch(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test single epoch training."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    metrics = trainer.train_epoch()

    # Check metrics
    assert "total_loss" in metrics
    assert "token_loss" in metrics
    assert "latent_loss" in metrics
    assert metrics["total_loss"] > 0
    assert metrics["token_loss"] > 0
    assert trainer.global_step > 0


def test_validate(small_model, dummy_dataloader, training_config, temp_checkpoint_dir):
    """Test validation."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Run validation
    metrics = trainer.validate()

    # Check metrics
    assert "total_loss" in metrics
    assert "token_loss" in metrics
    assert "latent_loss" in metrics
    assert metrics["total_loss"] > 0
    assert metrics["token_loss"] > 0


def test_complete_training(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test complete training loop."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train for configured epochs
    history = trainer.train()

    # Check history
    assert isinstance(history, TrainingHistory)
    assert len(history.epochs) == training_config.num_epochs
    assert len(history.train_losses) == training_config.num_epochs
    assert len(history.val_losses) == training_config.num_epochs

    # Check that best model was saved
    best_model_path = Path(temp_checkpoint_dir) / "best_model.pt"
    assert best_model_path.exists()


def test_checkpoint_save_load(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test checkpoint saving and loading."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    trainer.train_epoch()

    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "test_checkpoint.pt"
    trainer.save_checkpoint(filename="test_checkpoint.pt")
    assert checkpoint_path.exists()

    # Store current state
    original_step = trainer.global_step
    original_epoch = trainer.current_epoch

    # Create new trainer and load checkpoint
    new_model = LatentForecastingModel(small_model.config)
    new_optimizer = create_optimizer(
        new_model, learning_rate=training_config.learning_rate
    )
    new_scheduler = create_scheduler(
        new_optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    new_trainer = Trainer(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Load checkpoint
    metadata = new_trainer.load_checkpoint(str(checkpoint_path))

    # Verify state was restored
    assert new_trainer.global_step == original_step
    assert new_trainer.current_epoch == original_epoch
    assert metadata["global_step"] == original_step
    assert metadata["epoch"] == original_epoch


def test_gradient_accumulation(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test gradient accumulation."""
    # Set gradient accumulation steps
    training_config.gradient_accumulation_steps = 4

    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    initial_step = trainer.global_step
    trainer.train_epoch()

    # Check that steps were accumulated correctly
    # With 10 batches and accumulation of 4, we should have 2-3 optimizer steps
    assert trainer.global_step > initial_step
    assert trainer.global_step < len(dummy_dataloader)


def test_gradient_clipping(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test gradient clipping."""
    # Set a small max_grad_norm to ensure clipping happens
    training_config.max_grad_norm = 0.1

    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch (should clip gradients)
    metrics = trainer.train_epoch()

    # Training should complete without errors
    assert metrics["total_loss"] > 0


def test_training_history():
    """Test TrainingHistory tracking."""
    history = TrainingHistory()

    # Add epoch metrics
    history.add_epoch(
        epoch=0,
        train_metrics={"total_loss": 1.5, "token_loss": 1.2, "latent_loss": 0.3},
        val_metrics={"total_loss": 1.6, "token_loss": 1.3, "latent_loss": 0.3},
        learning_rate=1e-4,
    )

    assert len(history.epochs) == 1
    assert history.train_losses[0] == 1.5
    assert history.val_losses[0] == 1.6
    assert history.learning_rates[0] == 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_checkpoint_metadata(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test checkpoint includes comprehensive metadata."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    trainer.train_epoch()

    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "test_checkpoint.pt"
    trainer.save_checkpoint(filename="test_checkpoint.pt")

    # Load checkpoint and verify metadata
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    assert "metadata" in checkpoint
    metadata = checkpoint["metadata"]

    # Check required metadata fields
    assert "timestamp" in metadata
    assert "global_step" in metadata
    assert "epoch" in metadata
    assert "best_val_loss" in metadata
    assert "config" in metadata

    # Check config fields
    config_dict = metadata["config"]
    assert "learning_rate" in config_dict
    assert "batch_size" in config_dict
    assert "seed" in config_dict

    # Check JSON metadata file was created
    json_path = checkpoint_path.with_suffix(".json")
    assert json_path.exists()

    with open(json_path, "r") as f:
        json_metadata = json.load(f)

    assert json_metadata["global_step"] == metadata["global_step"]


def test_emergency_checkpoint_on_nan_loss(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test emergency checkpoint is saved when NaN loss is detected."""

    # Create a model that will produce NaN
    class NaNModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.call_count = 0

        def forward(self, tokens, labels=None, compute_latent_loss=True):
            self.call_count += 1
            # Return NaN after a few normal iterations
            if self.call_count > 3:
                output = self.base_model(
                    tokens, labels=labels, compute_latent_loss=compute_latent_loss
                )
                # Force NaN in loss
                output.total_loss = torch.tensor(float("nan"))
                return output
            return self.base_model(
                tokens, labels=labels, compute_latent_loss=compute_latent_loss
            )

    nan_model = NaNModel(small_model)

    optimizer = create_optimizer(nan_model, learning_rate=training_config.learning_rate)
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=nan_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Training should fail with RuntimeError
    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train_epoch()

    # Check that emergency checkpoint was saved
    emergency_files = list(Path(temp_checkpoint_dir).glob("emergency_checkpoint_*.pt"))
    assert len(emergency_files) > 0

    # Load emergency checkpoint and verify it's marked as emergency
    emergency_checkpoint = torch.load(emergency_files[0], weights_only=False)
    assert emergency_checkpoint.get("emergency", False) == True
    assert "error_message" in emergency_checkpoint


def test_emergency_checkpoint_on_exception(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test emergency checkpoint is saved when training fails with exception."""

    # Create a model that will raise an exception
    class FailingModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.call_count = 0

        def forward(self, tokens, labels=None, compute_latent_loss=True):
            self.call_count += 1
            if self.call_count > 3:
                raise ValueError("Simulated training failure")
            return self.base_model(
                tokens, labels=labels, compute_latent_loss=compute_latent_loss
            )

    failing_model = FailingModel(small_model)

    optimizer = create_optimizer(
        failing_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=failing_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Training should fail with ValueError
    with pytest.raises(ValueError, match="Simulated training failure"):
        trainer.train_epoch()

    # Check that emergency checkpoint was saved
    emergency_files = list(Path(temp_checkpoint_dir).glob("emergency_checkpoint_*.pt"))
    assert len(emergency_files) > 0


def test_checkpoint_validation(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test checkpoint validation detects invalid checkpoints."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Test missing required keys
    invalid_checkpoint = {"model_state_dict": {}}
    with pytest.raises(ValueError, match="missing required key"):
        trainer._validate_checkpoint(invalid_checkpoint)

    # Test invalid global_step
    invalid_checkpoint = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "global_step": -1,
        "epoch": 0,
    }
    with pytest.raises(ValueError, match="Invalid global_step"):
        trainer._validate_checkpoint(invalid_checkpoint)


def test_checkpoint_architecture_mismatch(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test checkpoint loading detects architecture mismatches."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train and save checkpoint
    trainer.train_epoch()
    checkpoint_path = Path(temp_checkpoint_dir) / "test_checkpoint.pt"
    trainer.save_checkpoint(filename="test_checkpoint.pt")

    # Create a different model with different architecture
    different_config = ModelConfig(
        vocab_size=100,
        latent_dim=128,  # Different from original
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        dropout=0.1,
        forecast_horizons=[1, 2],
        max_context_length=32,
        lambda_latent=0.1,
    )
    different_model = LatentForecastingModel(different_config)

    new_optimizer = create_optimizer(
        different_model, learning_rate=training_config.learning_rate
    )
    new_scheduler = create_scheduler(
        new_optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    new_trainer = Trainer(
        model=different_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Loading should fail with architecture mismatch
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        new_trainer.load_checkpoint(str(checkpoint_path))


def test_resume_training(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test resume training from checkpoint."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    # Set to train for 4 epochs total
    training_config.num_epochs = 4

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train for 2 epochs
    training_config.num_epochs = 2
    trainer.config.num_epochs = 2
    history1 = trainer.train()

    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "resume_checkpoint.pt"
    trainer.save_checkpoint(filename="resume_checkpoint.pt")

    # Create new trainer and resume
    new_model = LatentForecastingModel(small_model.config)
    new_optimizer = create_optimizer(
        new_model, learning_rate=training_config.learning_rate
    )
    new_scheduler = create_scheduler(
        new_optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    training_config.num_epochs = 4  # Total epochs
    new_trainer = Trainer(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Resume training
    history2 = new_trainer.resume_training(str(checkpoint_path))

    # Verify training resumed from correct epoch
    assert new_trainer.current_epoch >= 2
    assert len(history2.epochs) >= 2


def test_checkpoint_reversibility(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test checkpoint save/load reversibility."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    trainer.train_epoch()

    # Get model state before saving
    state_before = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "reversibility_test.pt"
    trainer.save_checkpoint(filename="reversibility_test.pt")

    # Modify model state
    for param in trainer.model.parameters():
        param.data.fill_(0.0)

    # Load checkpoint
    trainer.load_checkpoint(str(checkpoint_path))

    # Get model state after loading
    state_after = trainer.model.state_dict()

    # Verify states match
    for key in state_before.keys():
        assert torch.allclose(
            state_before[key], state_after[key], rtol=1e-5, atol=1e-7
        ), f"State mismatch for {key}"


def test_best_checkpoint_tracking(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test best checkpoint is saved based on validation loss."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train for configured epochs
    history = trainer.train()

    # Check that best model was saved
    best_model_path = Path(temp_checkpoint_dir) / "best_model.pt"
    assert best_model_path.exists()

    # Load best checkpoint
    best_checkpoint = torch.load(best_model_path, weights_only=False)

    # Verify it has the best validation loss
    assert "best_val_loss" in best_checkpoint
    assert best_checkpoint["best_val_loss"] == trainer.best_val_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_tensorboard_logging(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test TensorBoard logging is created and populated."""
    import tempfile

    temp_log_dir = tempfile.mkdtemp()

    try:
        optimizer = create_optimizer(
            small_model, learning_rate=training_config.learning_rate
        )
        scheduler = create_scheduler(
            optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=training_config,
            checkpoint_dir=temp_checkpoint_dir,
            log_dir=temp_log_dir,
        )

        # Train one epoch
        trainer.train_epoch()

        # Check TensorBoard directory was created
        tensorboard_dir = Path(temp_log_dir) / "tensorboard"
        assert tensorboard_dir.exists()

        # Check that event files were created
        event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

        # Close writer
        trainer.tensorboard_writer.close()
    finally:
        shutil.rmtree(temp_log_dir)


def test_csv_logging(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test CSV logging is created and populated."""
    import csv
    import tempfile

    temp_log_dir = tempfile.mkdtemp()

    try:
        optimizer = create_optimizer(
            small_model, learning_rate=training_config.learning_rate
        )
        scheduler = create_scheduler(
            optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=training_config,
            checkpoint_dir=temp_checkpoint_dir,
            log_dir=temp_log_dir,
        )

        # Train one epoch
        trainer.train_epoch()

        # Check CSV file was created
        csv_path = Path(temp_log_dir) / "training_log.csv"
        assert csv_path.exists()

        # Read CSV and verify headers
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Check required headers
            assert "step" in headers
            assert "epoch" in headers
            assert "total_loss" in headers
            assert "token_loss" in headers
            assert "latent_loss" in headers
            assert "learning_rate" in headers
            assert "grad_norm" in headers
            assert "tokens_per_sec" in headers
            assert "timestamp" in headers
            assert "phase" in headers

            # Check that rows were written
            rows = list(reader)
            assert len(rows) > 0

            # Verify data types
            for row in rows:
                assert row["phase"] in ["train", "val"]
                assert float(row["total_loss"]) >= 0
                assert float(row["token_loss"]) >= 0

        # Close writer
        trainer.tensorboard_writer.close()
    finally:
        shutil.rmtree(temp_log_dir)


def test_timestamped_log_directories(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test timestamped log directories are created."""
    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    # Create trainer without specifying log_dir (should create timestamped dir)
    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Check log directory was created with timestamp
    assert trainer.log_dir.exists()
    assert "run_" in str(trainer.log_dir)

    # Check subdirectories
    tensorboard_dir = trainer.log_dir / "tensorboard"
    assert tensorboard_dir.exists()

    # Close writer
    trainer.tensorboard_writer.close()

    # Clean up
    shutil.rmtree(trainer.log_dir)


def test_learning_rate_logging(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test learning rate is logged correctly."""
    import csv
    import tempfile

    temp_log_dir = tempfile.mkdtemp()

    try:
        optimizer = create_optimizer(
            small_model, learning_rate=training_config.learning_rate
        )
        scheduler = create_scheduler(
            optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=training_config,
            checkpoint_dir=temp_checkpoint_dir,
            log_dir=temp_log_dir,
        )

        # Train one epoch
        trainer.train_epoch()

        # Read CSV and verify learning rate is logged
        csv_path = Path(temp_log_dir) / "training_log.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Check that learning rate is present and valid
            for row in rows:
                if row["phase"] == "train":
                    lr = float(row["learning_rate"])
                    assert lr > 0
                    assert lr <= training_config.learning_rate

        # Close writer
        trainer.tensorboard_writer.close()
    finally:
        shutil.rmtree(temp_log_dir)


def test_gradient_norm_logging(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test gradient norm is logged correctly."""
    import csv
    import tempfile

    temp_log_dir = tempfile.mkdtemp()

    try:
        optimizer = create_optimizer(
            small_model, learning_rate=training_config.learning_rate
        )
        scheduler = create_scheduler(
            optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=training_config,
            checkpoint_dir=temp_checkpoint_dir,
            log_dir=temp_log_dir,
        )

        # Train one epoch
        trainer.train_epoch()

        # Read CSV and verify gradient norm is logged
        csv_path = Path(temp_log_dir) / "training_log.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Check that gradient norm is present and valid
            train_rows_with_grad = []
            for row in rows:
                if row["phase"] == "train":
                    grad_norm = float(row["grad_norm"])
                    assert grad_norm >= 0
                    train_rows_with_grad.append(grad_norm)

            # Should have logged gradient norms
            assert len(train_rows_with_grad) > 0

        # Close writer
        trainer.tensorboard_writer.close()
    finally:
        shutil.rmtree(temp_log_dir)


def test_throughput_logging(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test throughput (tokens/second) is logged correctly."""
    import csv
    import tempfile

    temp_log_dir = tempfile.mkdtemp()

    try:
        optimizer = create_optimizer(
            small_model, learning_rate=training_config.learning_rate
        )
        scheduler = create_scheduler(
            optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=training_config,
            checkpoint_dir=temp_checkpoint_dir,
            log_dir=temp_log_dir,
        )

        # Train one epoch
        trainer.train_epoch()

        # Read CSV and verify throughput is logged
        csv_path = Path(temp_log_dir) / "training_log.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Check that throughput is present and valid
            for row in rows:
                if row["phase"] == "train":
                    tokens_per_sec = float(row["tokens_per_sec"])
                    assert tokens_per_sec >= 0

        # Verify total tokens processed is tracked
        assert trainer.total_tokens_processed > 0

        # Close writer
        trainer.tensorboard_writer.close()
    finally:
        shutil.rmtree(temp_log_dir)


def test_logging_during_complete_training(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test logging works correctly during complete training loop."""
    import csv
    import tempfile

    temp_log_dir = tempfile.mkdtemp()

    try:
        optimizer = create_optimizer(
            small_model, learning_rate=training_config.learning_rate
        )
        scheduler = create_scheduler(
            optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
        )

        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=training_config,
            checkpoint_dir=temp_checkpoint_dir,
            log_dir=temp_log_dir,
        )

        # Train for configured epochs
        history = trainer.train()

        # Check CSV file has entries for both train and val
        csv_path = Path(temp_log_dir) / "training_log.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Should have both train and val entries
            train_rows = [r for r in rows if r["phase"] == "train"]
            val_rows = [r for r in rows if r["phase"] == "val"]

            assert len(train_rows) > 0
            assert len(val_rows) > 0

            # Validation rows should have one per epoch
            assert len(val_rows) == training_config.num_epochs

        # Check TensorBoard files exist
        tensorboard_dir = Path(temp_log_dir) / "tensorboard"
        event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    finally:
        shutil.rmtree(temp_log_dir)


def test_mixed_precision_training(small_model, dummy_dataloader, temp_checkpoint_dir):
    """Test mixed precision training with GradScaler."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping mixed precision test")

    # Create config with mixed precision enabled
    training_config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        warmup_steps=5,
        lambda_latent=0.1,
        use_mixed_precision=True,  # Enable mixed precision
        checkpoint_every=20,
        log_every=5,
        seed=42,
        device="cuda",
    )

    # Move model to CUDA
    small_model = small_model.to("cuda")

    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Verify GradScaler is initialized
    assert trainer.scaler is not None
    assert trainer.use_amp is True

    # Train one epoch
    metrics = trainer.train_epoch()

    # Check metrics
    assert "total_loss" in metrics
    assert "token_loss" in metrics
    assert "latent_loss" in metrics
    assert metrics["total_loss"] > 0
    assert metrics["token_loss"] > 0
    assert trainer.global_step > 0

    # Verify training completed without errors
    assert not torch.isnan(torch.tensor(metrics["total_loss"]))


def test_mixed_precision_disabled(
    small_model, dummy_dataloader, training_config, temp_checkpoint_dir
):
    """Test that mixed precision can be disabled."""
    # Ensure mixed precision is disabled
    training_config.use_mixed_precision = False

    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Verify GradScaler is not initialized
    assert trainer.scaler is None
    assert trainer.use_amp is False

    # Train one epoch
    metrics = trainer.train_epoch()

    # Check metrics
    assert "total_loss" in metrics
    assert metrics["total_loss"] > 0


def test_mixed_precision_checkpoint_compatibility(
    small_model, dummy_dataloader, temp_checkpoint_dir
):
    """Test that checkpoints work correctly with mixed precision."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping mixed precision test")

    # Create config with mixed precision enabled
    training_config = TrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_mixed_precision=True,
        device="cuda",
    )

    # Move model to CUDA
    small_model = small_model.to("cuda")

    optimizer = create_optimizer(
        small_model, learning_rate=training_config.learning_rate
    )
    scheduler = create_scheduler(
        optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    trainer = Trainer(
        model=small_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Train one epoch
    trainer.train_epoch()

    # Save checkpoint
    checkpoint_path = Path(temp_checkpoint_dir) / "mixed_precision_checkpoint.pt"
    trainer.save_checkpoint(filename="mixed_precision_checkpoint.pt")
    assert checkpoint_path.exists()

    # Create new trainer and load checkpoint
    new_model = LatentForecastingModel(small_model.config).to("cuda")
    new_optimizer = create_optimizer(
        new_model, learning_rate=training_config.learning_rate
    )
    new_scheduler = create_scheduler(
        new_optimizer, num_training_steps=100, warmup_steps=training_config.warmup_steps
    )

    new_trainer = Trainer(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        config=training_config,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Load checkpoint
    metadata = new_trainer.load_checkpoint(str(checkpoint_path))

    # Verify checkpoint loaded successfully
    assert metadata["global_step"] == trainer.global_step
    assert new_trainer.scaler is not None

    # Continue training to verify it works
    metrics = new_trainer.train_epoch()
    assert metrics["total_loss"] > 0
