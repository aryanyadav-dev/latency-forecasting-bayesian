"""
Tests for learning rate scheduler functionality.
"""

import math

import pytest
import torch
import torch.nn as nn

from training.optimizer import create_optimizer
from training.scheduler import (compute_cosine_annealing,
                                compute_warmup_schedule, create_scheduler,
                                get_all_lrs, get_current_lr,
                                get_scheduler_state_dict,
                                load_scheduler_state_dict,
                                validate_scheduler_config)


class SimpleModel(nn.Module):
    """Simple model for testing scheduler."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestCreateScheduler:
    """Tests for create_scheduler function."""

    def test_create_cosine_scheduler(self):
        """Test creating cosine annealing scheduler."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        assert scheduler is not None
        assert len(scheduler.get_last_lr()) > 0

    def test_create_linear_scheduler(self):
        """Test creating linear decay scheduler."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="linear",
        )

        assert scheduler is not None

    def test_create_constant_scheduler(self):
        """Test creating constant scheduler."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="constant",
        )

        assert scheduler is not None

    def test_create_cosine_with_restarts_scheduler(self):
        """Test creating cosine with restarts scheduler."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine_with_restarts",
            num_cycles=2.0,
        )

        assert scheduler is not None

    def test_invalid_scheduler_type_raises(self):
        """Test that invalid scheduler type raises ValueError."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            create_scheduler(
                optimizer,
                num_training_steps=1000,
                warmup_steps=100,
                scheduler_type="invalid",
            )

    def test_scheduler_with_min_lr_ratio(self):
        """Test scheduler with minimum learning rate ratio."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
            min_lr_ratio=0.1,
        )

        assert scheduler is not None


class TestSchedulerWarmup:
    """Tests for warmup phase of schedulers."""

    def test_warmup_starts_at_zero(self):
        """Test that learning rate starts at zero during warmup."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        # Initial LR should be very close to zero (first step of warmup)
        initial_lr = get_current_lr(scheduler)
        assert initial_lr < base_lr * 0.1, "Initial LR should be near zero"

    def test_warmup_reaches_base_lr(self):
        """Test that learning rate reaches base LR after warmup."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=warmup_steps,
            scheduler_type="constant",  # Use constant to isolate warmup
        )

        # Step through warmup
        for _ in range(warmup_steps):
            scheduler.step()

        # LR should be at base_lr after warmup
        current_lr = get_current_lr(scheduler)
        assert (
            abs(current_lr - base_lr) < 1e-6
        ), f"LR should be {base_lr} after warmup, got {current_lr}"

    def test_warmup_linear_increase(self):
        """Test that warmup increases learning rate linearly."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=warmup_steps,
            scheduler_type="constant",
        )

        # Record LRs during warmup
        lrs = []
        for _ in range(warmup_steps):
            lrs.append(get_current_lr(scheduler))
            scheduler.step()

        # Check that LRs increase monotonically
        for i in range(len(lrs) - 1):
            assert lrs[i] <= lrs[i + 1], "LR should increase during warmup"

    def test_warmup_schedule_computation(self):
        """Test compute_warmup_schedule function."""
        warmup_steps = 100

        # Test at various steps
        assert compute_warmup_schedule(warmup_steps, 0) == 0.0
        assert abs(compute_warmup_schedule(warmup_steps, 50) - 0.5) < 1e-6
        assert abs(compute_warmup_schedule(warmup_steps, 100) - 1.0) < 1e-6
        assert compute_warmup_schedule(warmup_steps, 150) == 1.0


class TestCosineScheduler:
    """Tests for cosine annealing scheduler."""

    def test_cosine_decreases_after_warmup(self):
        """Test that cosine scheduler decreases LR after warmup."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=warmup_steps,
            scheduler_type="cosine",
            min_lr_ratio=0.0,
        )

        # Step through warmup
        for _ in range(warmup_steps):
            scheduler.step()

        lr_after_warmup = get_current_lr(scheduler)

        # Step through half of remaining steps
        for _ in range(450):
            scheduler.step()

        lr_mid = get_current_lr(scheduler)

        # LR should have decreased
        assert lr_mid < lr_after_warmup, "LR should decrease during cosine annealing"

    def test_cosine_reaches_min_lr(self):
        """Test that cosine scheduler reaches minimum LR at end."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100
        total_steps = 1000
        min_lr_ratio = 0.1

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=total_steps,
            warmup_steps=warmup_steps,
            scheduler_type="cosine",
            min_lr_ratio=min_lr_ratio,
        )

        # Step through all training
        for _ in range(total_steps):
            scheduler.step()

        final_lr = get_current_lr(scheduler)
        expected_min_lr = base_lr * min_lr_ratio

        # Final LR should be close to min_lr
        assert (
            abs(final_lr - expected_min_lr) < base_lr * 0.05
        ), f"Final LR should be {expected_min_lr}, got {final_lr}"

    def test_cosine_annealing_computation(self):
        """Test compute_cosine_annealing function."""
        total_steps = 100
        min_ratio = 0.1

        # At start, should be 1.0
        assert abs(compute_cosine_annealing(0, total_steps, min_ratio) - 1.0) < 1e-6

        # At end, should be min_ratio
        assert (
            abs(
                compute_cosine_annealing(total_steps, total_steps, min_ratio)
                - min_ratio
            )
            < 1e-6
        )

        # At middle, should be between min_ratio and 1.0
        mid_value = compute_cosine_annealing(total_steps // 2, total_steps, min_ratio)
        assert min_ratio < mid_value < 1.0

    def test_cosine_smooth_curve(self):
        """Test that cosine annealing produces smooth curve."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
            min_lr_ratio=0.0,
        )

        # Record LRs
        lrs = []
        for _ in range(1000):
            lrs.append(get_current_lr(scheduler))
            scheduler.step()

        # Check that changes are smooth (no sudden jumps)
        for i in range(1, len(lrs)):
            change = abs(lrs[i] - lrs[i - 1])
            assert change < base_lr * 0.1, "LR changes should be smooth"


class TestLinearScheduler:
    """Tests for linear decay scheduler."""

    def test_linear_decreases_monotonically(self):
        """Test that linear scheduler decreases monotonically."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=warmup_steps,
            scheduler_type="linear",
            min_lr_ratio=0.0,
        )

        # Step through warmup
        for _ in range(warmup_steps):
            scheduler.step()

        # Record LRs during decay
        lrs = []
        for _ in range(900):
            lrs.append(get_current_lr(scheduler))
            scheduler.step()

        # Check monotonic decrease
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], "LR should decrease monotonically"

    def test_linear_reaches_min_lr(self):
        """Test that linear scheduler reaches minimum LR."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100
        total_steps = 1000
        min_lr_ratio = 0.1

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=total_steps,
            warmup_steps=warmup_steps,
            scheduler_type="linear",
            min_lr_ratio=min_lr_ratio,
        )

        # Step through all training
        for _ in range(total_steps):
            scheduler.step()

        final_lr = get_current_lr(scheduler)
        expected_min_lr = base_lr * min_lr_ratio

        assert abs(final_lr - expected_min_lr) < base_lr * 0.01


class TestConstantScheduler:
    """Tests for constant scheduler."""

    def test_constant_maintains_lr_after_warmup(self):
        """Test that constant scheduler maintains LR after warmup."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        warmup_steps = 100

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=warmup_steps,
            scheduler_type="constant",
        )

        # Step through warmup
        for _ in range(warmup_steps):
            scheduler.step()

        lr_after_warmup = get_current_lr(scheduler)

        # Step through many more steps
        for _ in range(500):
            scheduler.step()

        lr_later = get_current_lr(scheduler)

        # LR should remain constant
        assert (
            abs(lr_after_warmup - lr_later) < 1e-7
        ), "LR should remain constant after warmup"


class TestSchedulerStateDict:
    """Tests for scheduler state saving and loading."""

    def test_save_and_load_scheduler_state(self):
        """Test saving and loading scheduler state."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        # Step scheduler
        for _ in range(50):
            scheduler.step()

        # Save state
        state_dict = get_scheduler_state_dict(scheduler)
        current_lr = get_current_lr(scheduler)

        # Create new scheduler and load state
        new_optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )
        new_scheduler = create_scheduler(
            new_optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        load_scheduler_state_dict(new_scheduler, state_dict)

        # LR should match
        new_lr = get_current_lr(new_scheduler)
        assert abs(current_lr - new_lr) < 1e-7, "LR should match after loading state"


class TestSchedulerUtilities:
    """Tests for scheduler utility functions."""

    def test_get_current_lr(self):
        """Test getting current learning rate."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=False
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        lr = get_current_lr(scheduler)
        assert isinstance(lr, float)
        assert lr >= 0

    def test_get_all_lrs(self):
        """Test getting all learning rates."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=True
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        lrs = get_all_lrs(scheduler)
        assert isinstance(lrs, list)
        assert len(lrs) > 0
        assert all(lr >= 0 for lr in lrs)

    def test_validate_scheduler_config_valid(self):
        """Test that valid config passes validation."""
        assert validate_scheduler_config(
            num_training_steps=1000,
            warmup_steps=100,
            min_lr_ratio=0.1,
            scheduler_type="cosine",
        )

    def test_validate_scheduler_config_invalid_steps(self):
        """Test that invalid steps raise ValueError."""
        with pytest.raises(ValueError, match="num_training_steps must be positive"):
            validate_scheduler_config(
                num_training_steps=0,
                warmup_steps=100,
                min_lr_ratio=0.1,
                scheduler_type="cosine",
            )

        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            validate_scheduler_config(
                num_training_steps=1000,
                warmup_steps=-10,
                min_lr_ratio=0.1,
                scheduler_type="cosine",
            )

        with pytest.raises(ValueError, match="warmup_steps.*must be less than"):
            validate_scheduler_config(
                num_training_steps=1000,
                warmup_steps=1000,
                min_lr_ratio=0.1,
                scheduler_type="cosine",
            )

    def test_validate_scheduler_config_invalid_min_lr(self):
        """Test that invalid min_lr_ratio raises ValueError."""
        with pytest.raises(ValueError, match="min_lr_ratio must be in"):
            validate_scheduler_config(
                num_training_steps=1000,
                warmup_steps=100,
                min_lr_ratio=-0.1,
                scheduler_type="cosine",
            )

        with pytest.raises(ValueError, match="min_lr_ratio must be in"):
            validate_scheduler_config(
                num_training_steps=1000,
                warmup_steps=100,
                min_lr_ratio=1.5,
                scheduler_type="cosine",
            )

    def test_validate_scheduler_config_invalid_type(self):
        """Test that invalid scheduler type raises ValueError."""
        with pytest.raises(ValueError, match="scheduler_type must be one of"):
            validate_scheduler_config(
                num_training_steps=1000,
                warmup_steps=100,
                min_lr_ratio=0.1,
                scheduler_type="invalid",
            )


class TestSchedulerIntegration:
    """Integration tests for scheduler with optimizer."""

    def test_scheduler_with_optimizer_steps(self):
        """Test scheduler updates LR correctly with optimizer steps."""
        model = SimpleModel()
        base_lr = 1e-3
        optimizer = create_optimizer(
            model, learning_rate=base_lr, use_parameter_groups=False
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=100,
            warmup_steps=10,
            scheduler_type="cosine",
            min_lr_ratio=0.1,  # Use non-zero min to ensure LR stays positive
        )

        # Simulate training loop
        for step in range(100):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Verify LR is updated
            current_lr = get_current_lr(scheduler)
            assert current_lr > 0, f"LR should be positive at step {step}"

    def test_scheduler_multiple_param_groups(self):
        """Test scheduler with multiple parameter groups."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model, learning_rate=1e-3, use_parameter_groups=True
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine",
        )

        # All parameter groups should have LRs
        lrs = get_all_lrs(scheduler)
        assert len(lrs) == len(optimizer.param_groups)

        # Step scheduler
        scheduler.step()

        # All LRs should update
        new_lrs = get_all_lrs(scheduler)
        assert len(new_lrs) == len(lrs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
