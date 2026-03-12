"""
Latent representation analysis for Latent Forecasting Network.

This module implements advanced analysis functions for evaluating the quality
and characteristics of learned latent representations, including:
- Latent entropy (representation diversity)
- Latent variance (representation spread)
- Cosine similarity drift (representation stability)
- Cluster separability (representation structure)
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader


def compute_latent_entropy(latents: np.ndarray, num_bins: int = 100) -> float:
    """
    Compute entropy of latent representations.

    Entropy measures the diversity of latent activations. Higher entropy
    indicates more diverse representations, while lower entropy suggests
    the model is using a limited subset of the latent space.

    H(Z) = -sum(p(z) * log(p(z)))

    Args:
        latents: Latent representations [num_samples, latent_dim]
        num_bins: Number of bins for histogram approximation

    Returns:
        Entropy value (non-negative float)

    Preconditions:
        - latents is 2D numpy array
        - latents contains finite values
        - num_bins > 0

    Postconditions:
        - Returns non-negative float
        - Entropy is finite
        - Higher values indicate more diverse representations

    Examples:
        >>> latents = np.random.randn(1000, 512)
        >>> entropy = compute_latent_entropy(latents)
        >>> assert entropy >= 0
        >>> assert np.isfinite(entropy)
    """
    # Validate preconditions
    if not isinstance(latents, np.ndarray):
        raise TypeError(f"latents must be numpy array, got {type(latents)}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D array, got shape {latents.shape}")
    if not np.all(np.isfinite(latents)):
        raise ValueError("latents must contain only finite values")
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    # Flatten all latent values
    flat_latents = latents.flatten()

    # Compute histogram to approximate probability distribution
    hist, _ = np.histogram(flat_latents, bins=num_bins, density=True)

    # Remove zero bins to avoid log(0)
    hist = hist[hist > 0]

    # Normalize to get probabilities
    hist = hist / hist.sum()

    # Compute entropy: H = -sum(p * log(p))
    entropy = -np.sum(hist * np.log(hist + 1e-10))

    # Validate postconditions
    assert entropy >= 0, f"Entropy must be non-negative, got {entropy}"
    assert np.isfinite(entropy), f"Entropy must be finite, got {entropy}"

    return float(entropy)


def compute_latent_variance(latents: np.ndarray) -> float:
    """
    Compute average variance of latent representations.

    Variance measures the spread of latent activations. Higher variance
    indicates the model is using a wider range of values in the latent space.

    Args:
        latents: Latent representations [num_samples, latent_dim]

    Returns:
        Average variance across latent dimensions (non-negative float)

    Preconditions:
        - latents is 2D numpy array
        - latents contains finite values
        - latents has at least 2 samples

    Postconditions:
        - Returns non-negative float
        - Variance is finite
        - Variance is averaged across all latent dimensions

    Examples:
        >>> latents = np.random.randn(1000, 512)
        >>> variance = compute_latent_variance(latents)
        >>> assert variance >= 0
        >>> assert np.isfinite(variance)
    """
    # Validate preconditions
    if not isinstance(latents, np.ndarray):
        raise TypeError(f"latents must be numpy array, got {type(latents)}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D array, got shape {latents.shape}")
    if not np.all(np.isfinite(latents)):
        raise ValueError("latents must contain only finite values")
    if latents.shape[0] < 2:
        raise ValueError(
            f"latents must have at least 2 samples, got {latents.shape[0]}"
        )

    # Compute variance along sample dimension (axis=0)
    # This gives variance for each latent dimension
    variances = np.var(latents, axis=0)  # [latent_dim]

    # Average across all latent dimensions
    avg_variance = np.mean(variances)

    # Validate postconditions
    assert avg_variance >= 0, f"Variance must be non-negative, got {avg_variance}"
    assert np.isfinite(avg_variance), f"Variance must be finite, got {avg_variance}"

    return float(avg_variance)


def compute_cosine_similarity_drift(
    latents: np.ndarray, max_samples: int = 1000
) -> float:
    """
    Compute cosine similarity drift between consecutive latent states.

    Drift measures how much latent representations change between consecutive
    time steps. Lower drift indicates more stable representations.

    Drift = 1 - mean(cosine_similarity(z_t, z_{t+1}))

    Args:
        latents: Latent representations [num_samples, latent_dim]
        max_samples: Maximum number of consecutive pairs to evaluate

    Returns:
        Drift value in range [0, 2] (float)

    Preconditions:
        - latents is 2D numpy array
        - latents contains finite values
        - latents has at least 2 samples
        - max_samples > 0

    Postconditions:
        - Returns float in range [0, 2]
        - Drift is finite
        - Lower values indicate more stable representations
        - Drift = 0 means perfect stability (identical consecutive states)
        - Drift = 2 means maximum instability (opposite consecutive states)

    Examples:
        >>> latents = np.random.randn(1000, 512)
        >>> drift = compute_cosine_similarity_drift(latents)
        >>> assert 0 <= drift <= 2
        >>> assert np.isfinite(drift)
    """
    # Validate preconditions
    if not isinstance(latents, np.ndarray):
        raise TypeError(f"latents must be numpy array, got {type(latents)}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D array, got shape {latents.shape}")
    if not np.all(np.isfinite(latents)):
        raise ValueError("latents must contain only finite values")
    if latents.shape[0] < 2:
        raise ValueError(
            f"latents must have at least 2 samples, got {latents.shape[0]}"
        )
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    # Limit number of samples to evaluate
    num_pairs = min(max_samples, latents.shape[0] - 1)

    # Compute cosine similarity between consecutive states
    similarities = []
    for i in range(num_pairs):
        # Get consecutive latent states
        z_t = latents[i : i + 1]  # [1, latent_dim]
        z_t_plus_1 = latents[i + 1 : i + 2]  # [1, latent_dim]

        # Compute cosine similarity
        sim = cosine_similarity(z_t, z_t_plus_1)[0, 0]
        similarities.append(sim)

    # Compute average similarity
    avg_similarity = np.mean(similarities)

    # Compute drift: 1 - similarity
    # Cosine similarity ranges from -1 to 1
    # So drift ranges from 0 (similarity=1) to 2 (similarity=-1)
    drift = 1.0 - avg_similarity

    # Validate postconditions
    assert 0 <= drift <= 2, f"Drift must be in [0, 2], got {drift}"
    assert np.isfinite(drift), f"Drift must be finite, got {drift}"

    return float(drift)


def compute_cluster_separability(
    latents: np.ndarray,
    n_clusters: int = 10,
    max_samples: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Compute cluster separability using k-means and silhouette score.

    Separability measures how well-structured the latent space is. Higher
    silhouette scores indicate that latent representations form distinct,
    well-separated clusters.

    Args:
        latents: Latent representations [num_samples, latent_dim]
        n_clusters: Number of clusters for k-means
        max_samples: Maximum number of samples to use (for efficiency)
        random_state: Random seed for reproducibility

    Returns:
        Silhouette score in range [-1, 1] (float)

    Preconditions:
        - latents is 2D numpy array
        - latents contains finite values
        - latents has at least n_clusters samples
        - n_clusters >= 2
        - max_samples >= n_clusters

    Postconditions:
        - Returns float in range [-1, 1]
        - Separability is finite
        - Higher values indicate better cluster separation
        - Score near 1: well-separated clusters
        - Score near 0: overlapping clusters
        - Score near -1: incorrect clustering

    Examples:
        >>> latents = np.random.randn(1000, 512)
        >>> separability = compute_cluster_separability(latents)
        >>> assert -1 <= separability <= 1
        >>> assert np.isfinite(separability)
    """
    # Validate preconditions
    if not isinstance(latents, np.ndarray):
        raise TypeError(f"latents must be numpy array, got {type(latents)}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D array, got shape {latents.shape}")
    if not np.all(np.isfinite(latents)):
        raise ValueError("latents must contain only finite values")
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
    if latents.shape[0] < n_clusters:
        raise ValueError(
            f"latents must have at least {n_clusters} samples, got {latents.shape[0]}"
        )
    if max_samples < n_clusters:
        raise ValueError(
            f"max_samples must be >= n_clusters, got {max_samples} < {n_clusters}"
        )

    # Sample latents if we have too many (for computational efficiency)
    if latents.shape[0] > max_samples:
        indices = np.random.RandomState(random_state).choice(
            latents.shape[0], size=max_samples, replace=False
        )
        sample_latents = latents[indices]
    else:
        sample_latents = latents

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(sample_latents)

    # Compute silhouette score
    # This measures how similar each point is to its own cluster compared to other clusters
    separability = silhouette_score(sample_latents, cluster_labels)

    # Validate postconditions
    assert (
        -1 <= separability <= 1
    ), f"Separability must be in [-1, 1], got {separability}"
    assert np.isfinite(separability), f"Separability must be finite, got {separability}"

    return float(separability)


def analyze_representations(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_samples: int = 10000,
    n_clusters: int = 10,
) -> Dict[str, float]:
    """
    Comprehensive analysis of latent representations.

    Computes all representation quality metrics:
    - Latent entropy (diversity)
    - Latent variance (spread)
    - Cosine similarity drift (stability)
    - Cluster separability (structure)

    Args:
        model: Model to analyze (should return ModelOutput with latents)
        dataloader: DataLoader providing batches with 'input_ids'
        device: Device for computation ('cuda' or 'cpu')
        max_samples: Maximum number of latent samples to collect
        n_clusters: Number of clusters for separability analysis

    Returns:
        Dictionary containing:
            - 'latent_entropy': Entropy value (float >= 0)
            - 'latent_variance': Average variance (float >= 0)
            - 'cosine_similarity_drift': Drift value (float in [0, 2])
            - 'cluster_separability': Silhouette score (float in [-1, 1])

    Preconditions:
        - model is in eval mode or will be set to eval mode
        - dataloader provides valid batches
        - device is valid ('cuda' or 'cpu')
        - model.forward() returns object with .latents attribute
        - max_samples > 0
        - n_clusters >= 2

    Postconditions:
        - Returns dict with all four metrics
        - All metric values are finite
        - latent_entropy >= 0
        - latent_variance >= 0
        - 0 <= cosine_similarity_drift <= 2
        - -1 <= cluster_separability <= 1
        - Model remains in eval mode
        - No gradients are computed

    Examples:
        >>> model = build_model(config)
        >>> metrics = analyze_representations(model, test_loader)
        >>> assert 'latent_entropy' in metrics
        >>> assert 'latent_variance' in metrics
        >>> assert 'cosine_similarity_drift' in metrics
        >>> assert 'cluster_separability' in metrics
    """
    # Validate preconditions
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

    # Set model to eval mode
    model.eval()

    # Collect latent representations
    all_latents = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Check if we've collected enough samples
            if total_samples >= max_samples:
                break

            # Move batch to device
            tokens = batch["input_ids"].to(device)

            # Forward pass (without latent loss for efficiency)
            output = model(tokens, compute_latent_loss=False)

            # Get latents: [batch_size, seq_len, latent_dim]
            latents = output.latents.cpu().numpy()

            # Flatten batch and sequence dimensions
            # Shape: [batch_size * seq_len, latent_dim]
            latents_flat = latents.reshape(-1, latents.shape[-1])

            # Add to collection
            all_latents.append(latents_flat)
            total_samples += latents_flat.shape[0]

    # Validate we collected some data
    if len(all_latents) == 0:
        raise ValueError("Dataloader is empty, no latents collected")

    # Concatenate all latents
    all_latents = np.concatenate(all_latents, axis=0)  # [total_samples, latent_dim]

    # Limit to max_samples if we collected too many
    if all_latents.shape[0] > max_samples:
        all_latents = all_latents[:max_samples]

    # Validate we have enough samples for clustering
    if all_latents.shape[0] < n_clusters:
        raise ValueError(
            f"Not enough samples for clustering: {all_latents.shape[0]} < {n_clusters}"
        )

    # Compute all metrics
    metrics = {}

    # 1. Latent Entropy
    metrics["latent_entropy"] = compute_latent_entropy(all_latents)

    # 2. Latent Variance
    metrics["latent_variance"] = compute_latent_variance(all_latents)

    # 3. Cosine Similarity Drift
    metrics["cosine_similarity_drift"] = compute_cosine_similarity_drift(all_latents)

    # 4. Cluster Separability
    metrics["cluster_separability"] = compute_cluster_separability(
        all_latents, n_clusters=n_clusters
    )

    # Validate postconditions
    assert all(np.isfinite(v) for v in metrics.values()), "All metrics must be finite"
    assert (
        metrics["latent_entropy"] >= 0
    ), f"Entropy must be non-negative, got {metrics['latent_entropy']}"
    assert (
        metrics["latent_variance"] >= 0
    ), f"Variance must be non-negative, got {metrics['latent_variance']}"
    assert (
        0 <= metrics["cosine_similarity_drift"] <= 2
    ), f"Drift must be in [0, 2], got {metrics['cosine_similarity_drift']}"
    assert (
        -1 <= metrics["cluster_separability"] <= 1
    ), f"Separability must be in [-1, 1], got {metrics['cluster_separability']}"

    return metrics
