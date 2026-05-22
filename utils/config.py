"""Configuration management utilities."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Empty configuration file: {config_path}")
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if configuration is valid
    
    Raises:
        ValueError: If configuration is invalid with descriptive message
    """
    # Check required top-level sections
    required_sections = ['model', 'data', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model config
    model_config = config['model']
    _validate_model_config(model_config)
    
    # Validate data config
    data_config = config['data']
    _validate_data_config(data_config)
    
    # Validate training config
    training_config = config['training']
    _validate_training_config(training_config)
    
    return True


def _validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration section."""
    required_fields = ['vocab_size', 'latent_dim', 'num_layers', 'num_heads']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required model config field: {field}")
    
    # Validate positive integers
    if config['vocab_size'] <= 0:
        raise ValueError(f"vocab_size must be positive, got {config['vocab_size']}")
    
    if config['latent_dim'] <= 0:
        raise ValueError(f"latent_dim must be positive, got {config['latent_dim']}")
    
    if config['num_layers'] <= 0:
        raise ValueError(f"num_layers must be positive, got {config['num_layers']}")
    
    if config['num_heads'] <= 0:
        raise ValueError(f"num_heads must be positive, got {config['num_heads']}")
    
    # Validate latent_dim divisible by num_heads
    if config['latent_dim'] % config['num_heads'] != 0:
        raise ValueError(
            f"latent_dim ({config['latent_dim']}) must be divisible by "
            f"num_heads ({config['num_heads']})"
        )
    
    # Validate dropout if present
    if 'dropout' in config:
        dropout = config['dropout']
        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    
    # Validate forecast_horizons if present
    if 'forecast_horizons' in config:
        horizons = config['forecast_horizons']
        if not isinstance(horizons, list) or len(horizons) == 0:
            raise ValueError("forecast_horizons must be non-empty list")
        if any(h <= 0 for h in horizons):
            raise ValueError("All forecast_horizons must be positive")


def _validate_data_config(config: Dict[str, Any]) -> None:
    """Validate data configuration section."""
    required_fields = ['dataset_name', 'context_length', 'batch_size']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required data config field: {field}")
    
    # Validate dataset name
    valid_datasets = [
        'wikitext',
        'wikitext-2',
        'wikitext-103',
        'ptb',
        'ptb_text_only',
        'tinystories',
        'openwebtext',
    ]
    if config['dataset_name'] not in valid_datasets:
        raise ValueError(
            f"Invalid dataset_name: {config['dataset_name']}. "
            f"Must be one of {valid_datasets}"
        )
    
    # Validate positive integers
    if config['context_length'] <= 0:
        raise ValueError(f"context_length must be positive, got {config['context_length']}")
    
    if config['batch_size'] <= 0:
        raise ValueError(f"batch_size must be positive, got {config['batch_size']}")
    
    # Validate stride if present
    if 'stride' in config:
        stride = config['stride']
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if stride > config['context_length']:
            raise ValueError(
                f"stride ({stride}) must be <= context_length ({config['context_length']})"
            )


def _validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration section."""
    required_fields = ['num_epochs', 'learning_rate']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required training config field: {field}")
    
    # Validate positive integers
    if config['num_epochs'] <= 0:
        raise ValueError(f"num_epochs must be positive, got {config['num_epochs']}")
    
    # Validate positive floats
    if config['learning_rate'] <= 0:
        raise ValueError(f"learning_rate must be positive, got {config['learning_rate']}")
    
    # Validate optional fields
    if 'weight_decay' in config and config['weight_decay'] < 0:
        raise ValueError(f"weight_decay must be non-negative, got {config['weight_decay']}")
    
    if 'gradient_accumulation_steps' in config and config['gradient_accumulation_steps'] <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be positive, "
            f"got {config['gradient_accumulation_steps']}"
        )
    
    if 'max_grad_norm' in config and config['max_grad_norm'] <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {config['max_grad_norm']}")
    
    if 'lambda_latent' in config and config['lambda_latent'] < 0:
        raise ValueError(f"lambda_latent must be non-negative, got {config['lambda_latent']}")
