"""
Learning rate scheduler for Latent Forecasting Network training.

This module provides learning rate scheduling with warmup and cosine annealing
to improve training stability and convergence.
"""

import logging
import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def create_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_steps: int = 1000,
    scheduler_type: str = "cosine",
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """
    Create learning rate scheduler with warmup phase.

    Supports multiple scheduling strategies:
    - 'cosine': Cosine annealing after warmup
    - 'linear': Linear decay after warmup
    - 'constant': Constant LR after warmup
    - 'cosine_with_restarts': Cosine annealing with warm restarts

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps (linear warmup from 0 to base LR)
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant', 'cosine_with_restarts')
        min_lr_ratio: Minimum learning rate as ratio of base LR (default: 0.0)
        num_cycles: Number of cycles for cosine_with_restarts (default: 0.5)

    Returns:
        LambdaLR scheduler instance

    Example:
        >>> optimizer = create_optimizer(model, learning_rate=1e-4)
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     num_training_steps=10000,
        ...     warmup_steps=1000,
        ...     scheduler_type='cosine'
        ... )
        >>> for step in range(num_training_steps):
        ...     optimizer.step()
        ...     scheduler.step()

    Preconditions:
        - optimizer is valid Optimizer instance
        - num_training_steps > 0
        - warmup_steps >= 0
        - warmup_steps < num_training_steps
        - 0.0 <= min_lr_ratio <= 1.0

    Postconditions:
        - Returns LambdaLR scheduler
        - Scheduler applies warmup for first warmup_steps
        - After warmup, applies selected scheduling strategy
        - Learning rate is always >= base_lr * min_lr_ratio
    """
    # Validate inputs
    assert num_training_steps > 0, "num_training_steps must be positive"
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert (
        warmup_steps < num_training_steps
    ), "warmup_steps must be less than num_training_steps"
    assert 0.0 <= min_lr_ratio <= 1.0, "min_lr_ratio must be in [0, 1]"

    # Select scheduling function
    if scheduler_type == "cosine":
        lr_lambda = _get_cosine_schedule_with_warmup_lambda(
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "linear":
        lr_lambda = _get_linear_schedule_with_warmup_lambda(
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "constant":
        lr_lambda = _get_constant_schedule_with_warmup_lambda(warmup_steps=warmup_steps)
    elif scheduler_type == "cosine_with_restarts":
        lr_lambda = _get_cosine_with_restarts_schedule_lambda(
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            num_cycles=num_cycles,
            min_lr_ratio=min_lr_ratio,
        )
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Must be one of: 'cosine', 'linear', 'constant', 'cosine_with_restarts'"
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Created {scheduler_type} scheduler with warmup_steps={warmup_steps}, "
        f"num_training_steps={num_training_steps}, min_lr_ratio={min_lr_ratio}"
    )

    return scheduler


def _get_cosine_schedule_with_warmup_lambda(
    num_training_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0
) -> Callable[[int], float]:
    """
    Create lambda function for cosine annealing schedule with warmup.

    Schedule:
    - Steps 0 to warmup_steps: Linear warmup from 0 to 1.0
    - Steps warmup_steps to num_training_steps: Cosine annealing from 1.0 to min_lr_ratio

    Args:
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR ratio at end of training

    Returns:
        Lambda function that takes current_step and returns LR multiplier
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine annealing phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale from min_lr_ratio to 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return lr_lambda


def _get_linear_schedule_with_warmup_lambda(
    num_training_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0
) -> Callable[[int], float]:
    """
    Create lambda function for linear decay schedule with warmup.

    Schedule:
    - Steps 0 to warmup_steps: Linear warmup from 0 to 1.0
    - Steps warmup_steps to num_training_steps: Linear decay from 1.0 to min_lr_ratio

    Args:
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR ratio at end of training

    Returns:
        Lambda function that takes current_step and returns LR multiplier
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Linear decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return lr_lambda


def _get_constant_schedule_with_warmup_lambda(
    warmup_steps: int,
) -> Callable[[int], float]:
    """
    Create lambda function for constant LR schedule with warmup.

    Schedule:
    - Steps 0 to warmup_steps: Linear warmup from 0 to 1.0
    - Steps warmup_steps onwards: Constant at 1.0

    Args:
        warmup_steps: Number of warmup steps

    Returns:
        Lambda function that takes current_step and returns LR multiplier
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return lr_lambda


def _get_cosine_with_restarts_schedule_lambda(
    num_training_steps: int,
    warmup_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """
    Create lambda function for cosine annealing with warm restarts.

    Schedule:
    - Steps 0 to warmup_steps: Linear warmup from 0 to 1.0
    - Steps warmup_steps onwards: Cosine annealing with restarts

    Args:
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps
        num_cycles: Number of cosine cycles (0.5 = half cycle, 1.0 = full cycle)
        min_lr_ratio: Minimum LR ratio at trough of cycle

    Returns:
        Lambda function that takes current_step and returns LR multiplier
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine with restarts phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress * num_cycles * 2.0))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value

    return lr_lambda


def get_scheduler_state_dict(scheduler: LambdaLR) -> dict:
    """
    Get scheduler state dictionary for checkpointing.

    Args:
        scheduler: Scheduler instance

    Returns:
        State dictionary
    """
    return scheduler.state_dict()


def load_scheduler_state_dict(scheduler: LambdaLR, state_dict: dict) -> None:
    """
    Load scheduler state from checkpoint.

    Args:
        scheduler: Scheduler instance to load state into
        state_dict: State dictionary from checkpoint
    """
    scheduler.load_state_dict(state_dict)
    logger.info("Loaded scheduler state from checkpoint")


def get_current_lr(scheduler: LambdaLR) -> float:
    """
    Get current learning rate from scheduler.

    Args:
        scheduler: Scheduler instance

    Returns:
        Current learning rate (from first parameter group)
    """
    return scheduler.get_last_lr()[0]


def get_all_lrs(scheduler: LambdaLR) -> list:
    """
    Get current learning rates for all parameter groups.

    Args:
        scheduler: Scheduler instance

    Returns:
        List of learning rates for each parameter group
    """
    return scheduler.get_last_lr()


def compute_warmup_schedule(warmup_steps: int, current_step: int) -> float:
    """
    Compute warmup multiplier for given step.

    Args:
        warmup_steps: Total warmup steps
        current_step: Current training step

    Returns:
        Learning rate multiplier in [0, 1]

    Preconditions:
        - warmup_steps > 0
        - current_step >= 0

    Postconditions:
        - Returns value in [0, 1]
        - Returns 0 when current_step = 0
        - Returns 1 when current_step >= warmup_steps
        - Monotonically increasing
    """
    if current_step >= warmup_steps:
        return 1.0
    if warmup_steps == 0:
        return 1.0
    return float(current_step) / float(warmup_steps)


def compute_cosine_annealing(
    current_step: int, total_steps: int, min_ratio: float = 0.0
) -> float:
    """
    Compute cosine annealing multiplier.

    Args:
        current_step: Current step in annealing phase
        total_steps: Total steps for annealing
        min_ratio: Minimum LR ratio at end

    Returns:
        Learning rate multiplier

    Preconditions:
        - total_steps > 0
        - current_step >= 0
        - 0 <= min_ratio <= 1

    Postconditions:
        - Returns value in [min_ratio, 1.0]
        - Returns 1.0 when current_step = 0
        - Returns min_ratio when current_step = total_steps
        - Smooth cosine curve
    """
    if current_step >= total_steps:
        return min_ratio

    progress = float(current_step) / float(total_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_ratio + (1.0 - min_ratio) * cosine_decay


def validate_scheduler_config(
    num_training_steps: int, warmup_steps: int, min_lr_ratio: float, scheduler_type: str
) -> bool:
    """
    Validate scheduler configuration parameters.

    Args:
        num_training_steps: Total training steps
        warmup_steps: Warmup steps
        min_lr_ratio: Minimum LR ratio
        scheduler_type: Type of scheduler

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if num_training_steps <= 0:
        raise ValueError(
            f"num_training_steps must be positive, got {num_training_steps}"
        )

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

    if warmup_steps >= num_training_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be less than "
            f"num_training_steps ({num_training_steps})"
        )

    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    valid_types = ["cosine", "linear", "constant", "cosine_with_restarts"]
    if scheduler_type not in valid_types:
        raise ValueError(
            f"scheduler_type must be one of {valid_types}, got {scheduler_type}"
        )

    return True
