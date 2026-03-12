"""
Latent Forecasting Network - Model Components

This package contains the core model architecture components:
- Encoder: Transformer-based encoder for token-to-latent conversion
- LatentForecastingNetwork: Multi-horizon latent state predictor
- Decoder: Latent-to-vocabulary projection
- Complete LFN model integration
"""

from models.complete_model import (LatentForecastingModel, ModelConfig,
                                   ModelOutput, build_model)
from models.decoder import Decoder
from models.encoder import Encoder, PositionalEncoding
from models.forecasting_network import LatentForecastingNetwork

__version__ = "0.1.0"

__all__ = [
    "Encoder",
    "PositionalEncoding",
    "LatentForecastingNetwork",
    "Decoder",
    "LatentForecastingModel",
    "ModelConfig",
    "ModelOutput",
    "build_model",
]
