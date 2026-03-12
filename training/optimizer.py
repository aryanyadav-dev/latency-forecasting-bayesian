"""
Optimizer setup for Latent Forecasting Network training.

This module provides utilities for creating and configuring optimizers
with support for parameter grouping and differential learning rates.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    use_parameter_groups: bool = True,
    encoder_lr_multiplier: float = 1.0,
    forecasting_lr_multiplier: float = 1.0,
    decoder_lr_multiplier: float = 1.0,
) -> Optimizer:
    """
    Create AdamW optimizer with optional parameter grouping for differential learning rates.

    Args:
        model: PyTorch model to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay coefficient (L2 regularization)
        betas: Adam beta parameters (beta1, beta2)
        eps: Adam epsilon for numerical stability
        use_parameter_groups: Whether to use different learning rates for different components
        encoder_lr_multiplier: Learning rate multiplier for encoder parameters
        forecasting_lr_multiplier: Learning rate multiplier for forecasting network parameters
        decoder_lr_multiplier: Learning rate multiplier for decoder parameters

    Returns:
        Configured AdamW optimizer

    Example:
        >>> model = LatentForecastingModel(config)
        >>> optimizer = create_optimizer(
        ...     model,
        ...     learning_rate=1e-4,
        ...     weight_decay=0.01,
        ...     use_parameter_groups=True,
        ...     encoder_lr_multiplier=0.5  # Lower LR for encoder
        ... )
    """
    if not use_parameter_groups:
        # Simple optimizer without parameter grouping
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        logger.info(
            f"Created AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}"
        )
        return optimizer

    # Create parameter groups with differential learning rates
    param_groups = _create_parameter_groups(
        model=model,
        base_lr=learning_rate,
        weight_decay=weight_decay,
        encoder_lr_multiplier=encoder_lr_multiplier,
        forecasting_lr_multiplier=forecasting_lr_multiplier,
        decoder_lr_multiplier=decoder_lr_multiplier,
    )

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,  # Base LR (will be overridden by group-specific LRs)
        betas=betas,
        eps=eps,
    )

    logger.info(f"Created AdamW optimizer with {len(param_groups)} parameter groups")
    for i, group in enumerate(param_groups):
        logger.info(
            f"  Group {i}: {group['name']}, lr={group['lr']}, "
            f"weight_decay={group['weight_decay']}, "
            f"num_params={sum(p.numel() for p in group['params'])}"
        )

    return optimizer


def _create_parameter_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    encoder_lr_multiplier: float,
    forecasting_lr_multiplier: float,
    decoder_lr_multiplier: float,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with differential learning rates and weight decay.

    Strategy:
    - Separate parameters by component (encoder, forecasting network, decoder)
    - Apply different learning rate multipliers to each component
    - Exclude bias and LayerNorm parameters from weight decay

    Args:
        model: Model to extract parameters from
        base_lr: Base learning rate
        weight_decay: Weight decay coefficient
        encoder_lr_multiplier: LR multiplier for encoder
        forecasting_lr_multiplier: LR multiplier for forecasting network
        decoder_lr_multiplier: LR multiplier for decoder

    Returns:
        List of parameter group dictionaries
    """
    # Collect parameters by component
    encoder_params_decay = []
    encoder_params_no_decay = []
    forecasting_params_decay = []
    forecasting_params_no_decay = []
    decoder_params_decay = []
    decoder_params_no_decay = []
    other_params_decay = []
    other_params_no_decay = []

    # Parameters that should not have weight decay
    no_decay_names = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if parameter should have weight decay
        should_decay = not any(nd in name for nd in no_decay_names)

        # Categorize by component
        if "encoder" in name:
            if should_decay:
                encoder_params_decay.append(param)
            else:
                encoder_params_no_decay.append(param)
        elif "forecasting_network" in name or "forecasting" in name:
            if should_decay:
                forecasting_params_decay.append(param)
            else:
                forecasting_params_no_decay.append(param)
        elif "decoder" in name:
            if should_decay:
                decoder_params_decay.append(param)
            else:
                decoder_params_no_decay.append(param)
        else:
            # Other parameters (e.g., embeddings, output layers)
            if should_decay:
                other_params_decay.append(param)
            else:
                other_params_no_decay.append(param)

    # Create parameter groups
    param_groups = []

    # Encoder groups
    if encoder_params_decay:
        param_groups.append(
            {
                "name": "encoder_decay",
                "params": encoder_params_decay,
                "lr": base_lr * encoder_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )
    if encoder_params_no_decay:
        param_groups.append(
            {
                "name": "encoder_no_decay",
                "params": encoder_params_no_decay,
                "lr": base_lr * encoder_lr_multiplier,
                "weight_decay": 0.0,
            }
        )

    # Forecasting network groups
    if forecasting_params_decay:
        param_groups.append(
            {
                "name": "forecasting_decay",
                "params": forecasting_params_decay,
                "lr": base_lr * forecasting_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )
    if forecasting_params_no_decay:
        param_groups.append(
            {
                "name": "forecasting_no_decay",
                "params": forecasting_params_no_decay,
                "lr": base_lr * forecasting_lr_multiplier,
                "weight_decay": 0.0,
            }
        )

    # Decoder groups
    if decoder_params_decay:
        param_groups.append(
            {
                "name": "decoder_decay",
                "params": decoder_params_decay,
                "lr": base_lr * decoder_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )
    if decoder_params_no_decay:
        param_groups.append(
            {
                "name": "decoder_no_decay",
                "params": decoder_params_no_decay,
                "lr": base_lr * decoder_lr_multiplier,
                "weight_decay": 0.0,
            }
        )

    # Other parameters groups
    if other_params_decay:
        param_groups.append(
            {
                "name": "other_decay",
                "params": other_params_decay,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )
    if other_params_no_decay:
        param_groups.append(
            {
                "name": "other_no_decay",
                "params": other_params_no_decay,
                "lr": base_lr,
                "weight_decay": 0.0,
            }
        )

    return param_groups


def get_optimizer_state_dict(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get optimizer state dictionary for checkpointing.

    Args:
        optimizer: Optimizer instance

    Returns:
        State dictionary
    """
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer: Optimizer, state_dict: Dict[str, Any]) -> None:
    """
    Load optimizer state from checkpoint.

    Args:
        optimizer: Optimizer instance to load state into
        state_dict: State dictionary from checkpoint
    """
    optimizer.load_state_dict(state_dict)
    logger.info("Loaded optimizer state from checkpoint")


def get_learning_rates(optimizer: Optimizer) -> Dict[str, float]:
    """
    Get current learning rates for all parameter groups.

    Args:
        optimizer: Optimizer instance

    Returns:
        Dictionary mapping group names to learning rates
    """
    learning_rates = {}
    for i, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get("name", f"group_{i}")
        learning_rates[group_name] = param_group["lr"]
    return learning_rates


def set_learning_rate(
    optimizer: Optimizer, learning_rate: float, group_name: Optional[str] = None
) -> None:
    """
    Set learning rate for optimizer or specific parameter group.

    Args:
        optimizer: Optimizer instance
        learning_rate: New learning rate
        group_name: Optional name of parameter group to update.
                   If None, updates all groups.
    """
    if group_name is None:
        # Update all parameter groups
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        logger.debug(f"Set learning rate to {learning_rate} for all groups")
    else:
        # Update specific group
        for param_group in optimizer.param_groups:
            if param_group.get("name") == group_name:
                param_group["lr"] = learning_rate
                logger.debug(
                    f"Set learning rate to {learning_rate} for group '{group_name}'"
                )
                return
        logger.warning(f"Parameter group '{group_name}' not found")


def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get comprehensive information about optimizer configuration.

    Args:
        optimizer: Optimizer instance

    Returns:
        Dictionary with optimizer information
    """
    info = {
        "optimizer_type": type(optimizer).__name__,
        "num_param_groups": len(optimizer.param_groups),
        "param_groups": [],
    }

    for i, param_group in enumerate(optimizer.param_groups):
        group_info = {
            "name": param_group.get("name", f"group_{i}"),
            "lr": param_group["lr"],
            "weight_decay": param_group.get("weight_decay", 0.0),
            "num_params": len(param_group["params"]),
            "total_parameters": sum(p.numel() for p in param_group["params"]),
        }
        info["param_groups"].append(group_info)

    return info
