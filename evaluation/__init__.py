"""
Latent Forecasting Network - Evaluation and Analysis

This package contains evaluation components:
- Language modeling metrics (perplexity, accuracy)
- Latent forecasting metrics (MSE, LPS)
- Representation analysis
- Visualization tools
"""

__version__ = "0.1.0"

# Import representation analysis functions
from evaluation.latent_analysis import (analyze_representations,
                                        compute_cluster_separability,
                                        compute_cosine_similarity_drift,
                                        compute_latent_entropy,
                                        compute_latent_variance)
# Import main evaluation functions
from evaluation.metrics import (Evaluator, compute_latent_predictability_score,
                                compute_perplexity, compute_token_accuracy,
                                evaluate_language_modeling,
                                evaluate_latent_forecasting)
# Import visualization
from evaluation.visualization import Visualizer

__all__ = [
    # Metrics
    "compute_perplexity",
    "compute_token_accuracy",
    "evaluate_language_modeling",
    "evaluate_latent_forecasting",
    "compute_latent_predictability_score",
    "Evaluator",
    # Representation analysis
    "compute_latent_entropy",
    "compute_latent_variance",
    "compute_cosine_similarity_drift",
    "compute_cluster_separability",
    "analyze_representations",
    # Visualization
    "Visualizer",
]
