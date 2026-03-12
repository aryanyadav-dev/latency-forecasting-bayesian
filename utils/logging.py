"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_file: str = "experiment.log",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for experiments.
    
    Args:
        log_dir: Directory for log files (creates if doesn't exist)
        log_file: Name of log file
        level: Logging level (default: INFO)
        console: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("latent_forecasting")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_dir specified
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
