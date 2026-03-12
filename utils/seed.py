"""Reproducibility utilities for setting random seeds."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value (non-negative integer)
    
    Raises:
        ValueError: If seed is negative
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
