"""
Trainer class for Latent Forecasting Network.

This module implements the complete training loop with support for:
- Gradient accumulation
- Gradient clipping
- Mixed precision training (AMP)
- Automatic checkpointing
- Learning rate scheduling
- Validation and logging
"""

import csv
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    lambda_latent: float = 0.1
    use_mixed_precision: bool = True
    checkpoint_every: int = 1000
    log_every: int = 100
    seed: int = 42
    device: str = "cuda"

    def validate(self) -> bool:
        """
        Validate training configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.lambda_latent < 0:
            raise ValueError("lambda_latent must be non-negative")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")

        return True


@dataclass
class TrainingHistory:
    """Training history tracking."""

    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    train_token_losses: list = field(default_factory=list)
    train_latent_losses: list = field(default_factory=list)
    val_token_losses: list = field(default_factory=list)
    val_latent_losses: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    epochs: list = field(default_factory=list)

    def add_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ):
        """Add metrics for an epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics.get("total_loss", 0.0))
        self.val_losses.append(val_metrics.get("total_loss", 0.0))
        self.train_token_losses.append(train_metrics.get("token_loss", 0.0))
        self.train_latent_losses.append(train_metrics.get("latent_loss", 0.0))
        self.val_token_losses.append(val_metrics.get("token_loss", 0.0))
        self.val_latent_losses.append(val_metrics.get("latent_loss", 0.0))
        self.learning_rates.append(learning_rate)


class Trainer:
    """
    Trainer for Latent Forecasting Network.

    Manages the complete training loop including:
    - Forward and backward passes
    - Gradient accumulation and clipping
    - Mixed precision training
    - Learning rate scheduling
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for saving logs (TensorBoard and CSV)

        Preconditions:
            - model is on correct device
            - optimizer is configured for model parameters
            - scheduler is configured for optimizer
            - config is valid
        """
        # Validate config
        config.validate()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup log directory with timestamp (5.4.5: Create timestamped log directories)
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"experiments/logs/run_{timestamp}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer (5.4.1: Add TensorBoard logging for losses and metrics)
        self.tensorboard_writer = SummaryWriter(
            log_dir=str(self.log_dir / "tensorboard")
        )

        # Initialize CSV logger (5.4.2: Add CSV logging for offline analysis)
        self.csv_log_path = self.log_dir / "training_log.csv"
        self._initialize_csv_logger()

        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Setup mixed precision training (5.2.1: Add GradScaler for automatic mixed precision)
        self.scaler = (
            GradScaler(device=config.device) if config.use_mixed_precision else None
        )
        self.use_amp = config.use_mixed_precision

        # Training history
        self.history = TrainingHistory()

        # Throughput tracking (5.4.4: Log throughput)
        self.total_tokens_processed = 0
        self.training_start_time = None

        logger.info(f"Initialized Trainer with config: {config}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        logger.info(f"Logging to: {self.log_dir}")

    def _initialize_csv_logger(self) -> None:
        """
        Initialize CSV logger with headers.

        Creates CSV file with headers for training metrics.
        """
        headers = [
            "step",
            "epoch",
            "total_loss",
            "token_loss",
            "latent_loss",
            "learning_rate",
            "grad_norm",
            "tokens_per_sec",
            "timestamp",
            "phase",  # 'train' or 'val'
        ]

        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

        logger.info(f"Initialized CSV logger at {self.csv_log_path}")

    def _log_to_csv(
        self, step: int, epoch: int, metrics: Dict[str, float], phase: str = "train"
    ) -> None:
        """
        Log metrics to CSV file.

        Args:
            step: Global training step
            epoch: Current epoch
            metrics: Dictionary of metrics to log
            phase: Training phase ('train' or 'val')
        """
        row = {
            "step": step,
            "epoch": epoch,
            "total_loss": metrics.get("total_loss", 0.0),
            "token_loss": metrics.get("token_loss", 0.0),
            "latent_loss": metrics.get("latent_loss", 0.0),
            "learning_rate": metrics.get("learning_rate", 0.0),
            "grad_norm": metrics.get("grad_norm", 0.0),
            "tokens_per_sec": metrics.get("tokens_per_sec", 0.0),
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
        }

        with open(self.csv_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def _log_to_tensorboard(
        self, step: int, metrics: Dict[str, float], phase: str = "train"
    ) -> None:
        """
        Log metrics to TensorBoard.

        Args:
            step: Global training step
            metrics: Dictionary of metrics to log
            phase: Training phase ('train' or 'val')
        """
        # Log losses
        if "total_loss" in metrics:
            self.tensorboard_writer.add_scalar(
                f"{phase}/total_loss", metrics["total_loss"], step
            )
        if "token_loss" in metrics:
            self.tensorboard_writer.add_scalar(
                f"{phase}/token_loss", metrics["token_loss"], step
            )
        if "latent_loss" in metrics:
            self.tensorboard_writer.add_scalar(
                f"{phase}/latent_loss", metrics["latent_loss"], step
            )

        # Log learning rate (5.4.3: Log learning rate and gradient norms)
        if "learning_rate" in metrics:
            self.tensorboard_writer.add_scalar(
                "training/learning_rate", metrics["learning_rate"], step
            )

        # Log gradient norm (5.4.3: Log learning rate and gradient norms)
        if "grad_norm" in metrics:
            self.tensorboard_writer.add_scalar(
                "training/grad_norm", metrics["grad_norm"], step
            )

        # Log throughput (5.4.4: Log throughput)
        if "tokens_per_sec" in metrics:
            self.tensorboard_writer.add_scalar(
                "training/tokens_per_sec", metrics["tokens_per_sec"], step
            )

        # Flush to ensure data is written
        self.tensorboard_writer.flush()

    def _get_git_commit_hash(self) -> Optional[str]:
        """
        Get current git commit hash if available.

        Returns:
            Git commit hash or None if not in a git repository
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return result.stdout.strip()
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            return None

    def _create_checkpoint_metadata(self) -> Dict[str, Any]:
        """
        Create comprehensive checkpoint metadata.

        Returns:
            Dictionary with metadata including timestamp, git hash, config
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "max_grad_norm": self.config.max_grad_norm,
                "warmup_steps": self.config.warmup_steps,
                "lambda_latent": self.config.lambda_latent,
                "use_mixed_precision": self.config.use_mixed_precision,
                "seed": self.config.seed,
                "device": self.config.device,
            },
        }

        # Add git commit hash if available
        git_hash = self._get_git_commit_hash()
        if git_hash:
            metadata["git_commit"] = git_hash

        return metadata

    def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            True if checkpoint is valid

        Raises:
            ValueError: If checkpoint is invalid or corrupted
        """
        required_keys = [
            "model_state_dict",
            "optimizer_state_dict",
            "global_step",
            "epoch",
        ]

        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required key: {key}")

        # Validate global_step and epoch are non-negative
        if checkpoint["global_step"] < 0:
            raise ValueError(f"Invalid global_step: {checkpoint['global_step']}")
        if checkpoint["epoch"] < 0:
            raise ValueError(f"Invalid epoch: {checkpoint['epoch']}")

        return True

    def train(self) -> TrainingHistory:
        """
        Execute complete training loop with emergency checkpoint on failure.

        Returns:
            TrainingHistory with metrics per epoch

        Postconditions:
            - Model has been trained for num_epochs
            - Best checkpoint has been saved
            - Training history contains metrics for all epochs
            - Emergency checkpoint saved if training fails
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.training_start_time = time.time()

        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch

                # Train for one epoch
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                # Update history
                self.history.add_epoch(epoch, train_metrics, val_metrics, current_lr)

                # Log epoch summary to TensorBoard and CSV
                epoch_train_metrics = {**train_metrics, "learning_rate": current_lr}
                self._log_to_tensorboard(
                    self.global_step, epoch_train_metrics, phase="train"
                )
                self._log_to_tensorboard(self.global_step, val_metrics, phase="val")
                self._log_to_csv(
                    self.global_step, epoch, epoch_train_metrics, phase="train"
                )
                self._log_to_csv(self.global_step, epoch, val_metrics, phase="val")

                # Log epoch summary
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"LR: {current_lr:.2e}"
                )

                # Save best checkpoint
                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")

            # Log final training statistics
            total_training_time = time.time() - self.training_start_time
            avg_tokens_per_sec = self.total_tokens_processed / total_training_time
            logger.info("Training complete")
            logger.info(f"Total training time: {total_training_time:.2f} seconds")
            logger.info(f"Average throughput: {avg_tokens_per_sec:.0f} tokens/sec")

            # Close TensorBoard writer
            self.tensorboard_writer.close()

            return self.history

        except Exception as e:
            # Save emergency checkpoint on any failure
            logger.error(f"Training failed with error: {e}")
            self.save_emergency_checkpoint(error_message=str(e))
            # Close TensorBoard writer
            self.tensorboard_writer.close()
            raise

    def train_epoch(self) -> Dict[str, float]:
        """
        Execute single training epoch with NaN/Inf detection.

        Returns:
            Dictionary with average metrics for the epoch

        Postconditions:
            - Model parameters have been updated
            - global_step has been incremented
            - Metrics are averaged across all batches
            - Emergency checkpoint saved if NaN/Inf detected
        """
        self.model.train()

        # Metrics accumulation
        total_loss = 0.0
        total_token_loss = 0.0
        total_latent_loss = 0.0
        num_batches = 0

        # Progress tracking
        start_time = time.time()
        tokens_in_interval = 0

        try:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                tokens = batch["input_ids"].to(self.config.device)
                labels = batch.get("labels", tokens)

                # Track tokens for throughput calculation
                batch_tokens = tokens.size(0) * tokens.size(1)
                tokens_in_interval += batch_tokens
                self.total_tokens_processed += batch_tokens

                # Forward pass with mixed precision (5.2.2: Implement autocast context for forward pass)
                with autocast(device_type=self.config.device, enabled=self.use_amp):
                    output = self.model(tokens, labels=labels, compute_latent_loss=True)
                    loss = output.total_loss / self.config.gradient_accumulation_steps

                # Check for NaN/Inf in loss
                if not torch.isfinite(output.total_loss):
                    logger.error(f"Non-finite loss detected at step {self.global_step}")
                    logger.error(
                        f"total_loss={output.total_loss.item()}, "
                        f"token_loss={output.token_loss.item()}, "
                        f"latent_loss={output.latent_loss.item() if output.latent_loss else 'N/A'}"
                    )
                    self.save_emergency_checkpoint(
                        error_message=f"Non-finite loss: total={output.total_loss.item()}, "
                        f"token={output.token_loss.item()}"
                    )
                    raise RuntimeError(
                        f"Training diverged: non-finite loss at step {self.global_step}"
                    )

                # Backward pass (5.2.3: Add loss scaling and unscaling)
                if self.scaler:
                    # Scale loss for FP16 training to prevent underflow
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping (5.2.3: Add loss scaling and unscaling)
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)

                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                    # Check for NaN/Inf in gradients
                    if not torch.isfinite(grad_norm):
                        logger.error(
                            f"Non-finite gradient norm detected at step {self.global_step}"
                        )
                        self.save_emergency_checkpoint(
                            error_message=f"Non-finite gradient norm: {grad_norm}"
                        )
                        raise RuntimeError(
                            f"Training diverged: non-finite gradients at step {self.global_step}"
                        )

                    # Optimizer step with scaled gradients
                    if self.scaler:
                        # Step optimizer with scaled gradients
                        self.scaler.step(self.optimizer)
                        # Update scaler for next iteration
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Update learning rate
                    self.scheduler.step()

                    # Increment global step
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            tokens_in_interval / elapsed if elapsed > 0 else 0.0
                        )
                        current_lr = self.scheduler.get_last_lr()[0]

                        # Prepare metrics for logging
                        step_metrics = {
                            "total_loss": output.total_loss.item(),
                            "token_loss": output.token_loss.item(),
                            "latent_loss": (
                                output.latent_loss.item() if output.latent_loss else 0.0
                            ),
                            "learning_rate": current_lr,
                            "grad_norm": grad_norm.item(),
                            "tokens_per_sec": tokens_per_sec,
                        }

                        # Log to TensorBoard and CSV
                        self._log_to_tensorboard(
                            self.global_step, step_metrics, phase="train"
                        )
                        self._log_to_csv(
                            self.global_step,
                            self.current_epoch,
                            step_metrics,
                            phase="train",
                        )

                        logger.info(
                            f"Step {self.global_step} - "
                            f"Loss: {output.total_loss.item():.4f}, "
                            f"Token Loss: {output.token_loss.item():.4f}, "
                            f"Latent Loss: {output.latent_loss.item() if output.latent_loss else 0.0:.4f}, "
                            f"Grad Norm: {grad_norm:.4f}, "
                            f"LR: {current_lr:.2e}, "
                            f"Tokens/sec: {tokens_per_sec:.0f}"
                        )

                        # Reset interval tracking
                        start_time = time.time()
                        tokens_in_interval = 0

                    # Checkpointing
                    if self.global_step % self.config.checkpoint_every == 0:
                        self.save_checkpoint(is_best=False)

                # Accumulate metrics
                total_loss += output.total_loss.item()
                total_token_loss += output.token_loss.item()
                if output.latent_loss is not None:
                    total_latent_loss += output.latent_loss.item()
                num_batches += 1

            # Compute average metrics
            avg_metrics = {
                "total_loss": total_loss / num_batches,
                "token_loss": total_token_loss / num_batches,
                "latent_loss": (
                    total_latent_loss / num_batches if total_latent_loss > 0 else 0.0
                ),
            }

            return avg_metrics

        except Exception as e:
            # Save emergency checkpoint on any failure during epoch
            logger.error(f"Training epoch failed: {e}")
            self.save_emergency_checkpoint(error_message=str(e))
            raise

    def validate(self) -> Dict[str, float]:
        """
        Run validation and return metrics.

        Returns:
            Dictionary with validation metrics

        Postconditions:
            - Model is in eval mode during validation
            - Model returns to train mode after validation
            - No gradients are computed
        """
        self.model.eval()

        total_loss = 0.0
        total_token_loss = 0.0
        total_latent_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                tokens = batch["input_ids"].to(self.config.device)
                labels = batch.get("labels", tokens)

                # Forward pass
                output = self.model(tokens, labels=labels, compute_latent_loss=True)

                # Accumulate metrics
                total_loss += output.total_loss.item()
                total_token_loss += output.token_loss.item()
                if output.latent_loss is not None:
                    total_latent_loss += output.latent_loss.item()
                num_batches += 1

        # Compute average metrics
        avg_metrics = {
            "total_loss": total_loss / num_batches,
            "token_loss": total_token_loss / num_batches,
            "latent_loss": (
                total_latent_loss / num_batches if total_latent_loss > 0 else 0.0
            ),
        }

        return avg_metrics

    def save_checkpoint(
        self, is_best: bool = False, filename: Optional[str] = None
    ) -> None:
        """
        Save model checkpoint with comprehensive metadata.

        Args:
            is_best: Whether this is the best checkpoint
            filename: Optional custom filename

        Postconditions:
            - Checkpoint file exists at specified path
            - Checkpoint contains model, optimizer, scheduler states and metadata
            - If is_best=True, also saves to 'best_model.pt'
        """
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Create comprehensive metadata
        metadata = self._create_checkpoint_metadata()

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "history": self.history,
            "metadata": metadata,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save metadata as JSON for easy inspection
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

            # Save best model metadata
            best_metadata_path = self.checkpoint_dir / "best_model.json"
            with open(best_metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def save_emergency_checkpoint(self, error_message: str = "") -> None:
        """
        Save emergency checkpoint when training fails.

        Args:
            error_message: Description of the error that triggered emergency save

        Postconditions:
            - Emergency checkpoint saved with error information
            - Checkpoint can be used to resume or debug training
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emergency_checkpoint_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / filename

        # Create metadata with error information
        metadata = self._create_checkpoint_metadata()
        metadata["emergency"] = True
        metadata["error_message"] = error_message

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "history": self.history,
            "metadata": metadata,
            "emergency": True,
            "error_message": error_message,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.error(f"Saved emergency checkpoint to {checkpoint_path}")
        logger.error(f"Error: {error_message}")

        # Save emergency metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(
        self, checkpoint_path: str, strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state with validation.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching

        Returns:
            Dictionary with checkpoint metadata

        Raises:
            ValueError: If checkpoint is invalid or corrupted
            RuntimeError: If architecture mismatch detected

        Postconditions:
            - Model, optimizer, scheduler states are restored
            - Training state (global_step, epoch) is restored
            - Returns metadata for inspection
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.config.device, weights_only=False
            )
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")

        # Validate checkpoint integrity
        self._validate_checkpoint(checkpoint)

        # Check for architecture mismatch
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except RuntimeError as e:
            logger.error(f"Architecture mismatch detected: {e}")
            if "config" in checkpoint:
                logger.error(f"Checkpoint config: {checkpoint['config']}")
                logger.error(f"Current config: {self.config}")
            raise RuntimeError(f"Cannot load checkpoint: architecture mismatch. {e}")

        # Restore optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Failed to restore optimizer state: {e}")
            logger.warning("Continuing with fresh optimizer state")

        # Restore scheduler state
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Failed to restore scheduler state: {e}")
            logger.warning("Continuing with fresh scheduler state")

        # Restore scaler state if present
        if self.scaler and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Failed to restore scaler state: {e}")
                logger.warning("Continuing with fresh scaler state")

        # Restore training state
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # Restore history if present
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        # Log metadata if present
        if "metadata" in checkpoint:
            metadata = checkpoint["metadata"]
            logger.info(f"Checkpoint metadata:")
            logger.info(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
            logger.info(f"  Git commit: {metadata.get('git_commit', 'N/A')}")
            if checkpoint.get("emergency", False):
                logger.warning(
                    f"  EMERGENCY CHECKPOINT - Error: {checkpoint.get('error_message', 'Unknown')}"
                )

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(
            f"Resuming from step {self.global_step}, epoch {self.current_epoch}"
        )

        return {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "metadata": checkpoint.get("metadata", {}),
            "emergency": checkpoint.get("emergency", False),
        }

    def resume_training(self, checkpoint_path: str) -> TrainingHistory:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            TrainingHistory with metrics from resumed training

        Postconditions:
            - Training state restored from checkpoint
            - Training continues from saved epoch
            - History includes both old and new metrics
        """
        # Load checkpoint
        metadata = self.load_checkpoint(checkpoint_path)

        logger.info(f"Resuming training from epoch {self.current_epoch + 1}")

        # Adjust num_epochs to continue from current epoch
        remaining_epochs = self.config.num_epochs - self.current_epoch
        if remaining_epochs <= 0:
            logger.warning(f"Training already completed {self.current_epoch} epochs")
            return self.history

        logger.info(f"Training for {remaining_epochs} more epochs")

        # Continue training
        return self.train()
