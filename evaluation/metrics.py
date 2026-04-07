"""
Evaluation metrics for Latent Forecasting Network.

This module implements language modeling metrics including:
- Perplexity computation
- Token accuracy
- Complete language modeling evaluation
- Latent representation analysis
- Downstream task evaluation (linear probing)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# Import representation analysis functions
try:
    from evaluation.latent_analysis import analyze_representations

    LATENT_ANALYSIS_AVAILABLE = True
except ImportError:
    LATENT_ANALYSIS_AVAILABLE = False

# Import downstream evaluation
try:
    from evaluation.downstream_eval import (
        DownstreamEvaluator,
        compute_effective_dimensionality,
        extract_pooled_representations,
    )

    DOWNSTREAM_AVAILABLE = True
except ImportError:
    DOWNSTREAM_AVAILABLE = False


def compute_perplexity(loss: Tensor) -> Tensor:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity is defined as exp(loss), where loss is the average
    cross-entropy per token. Lower perplexity indicates better model performance.

    Args:
        loss: Cross-entropy loss (scalar tensor)

    Returns:
        Perplexity value (scalar tensor)

    Preconditions:
        - loss is a scalar tensor
        - loss is non-negative
        - loss is finite (no NaN or Inf)

    Postconditions:
        - Returns positive scalar tensor
        - perplexity >= 1.0
        - perplexity is finite
        - perplexity = exp(loss)

    Examples:
        >>> loss = torch.tensor(2.0)
        >>> ppl = compute_perplexity(loss)
        >>> assert torch.isclose(ppl, torch.tensor(7.389), atol=0.001)
    """
    # Validate preconditions
    if not torch.isfinite(loss):
        raise ValueError(f"Loss must be finite, got {loss}")
    if loss < 0:
        raise ValueError(f"Loss must be non-negative, got {loss}")

    # Compute perplexity
    perplexity = torch.exp(loss)

    # Validate postconditions
    assert torch.isfinite(perplexity), "Perplexity must be finite"
    assert perplexity >= 1.0, f"Perplexity must be >= 1.0, got {perplexity}"

    return perplexity


def compute_token_accuracy(
    logits: Tensor, labels: Tensor, ignore_index: int = -100
) -> Tensor:
    """
    Compute token-level prediction accuracy.

    Accuracy is the fraction of tokens where the model's top prediction
    matches the ground truth label.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Token ID to ignore in accuracy computation

    Returns:
        Accuracy value in range [0, 1] (scalar tensor)

    Preconditions:
        - logits has shape [batch_size, seq_len, vocab_size]
        - labels has shape [batch_size, seq_len]
        - All logit values are finite
        - All label values are valid token IDs or ignore_index

    Postconditions:
        - Returns scalar tensor in range [0, 1]
        - Accuracy is finite
        - Accuracy = (correct predictions) / (total non-ignored tokens)

    Examples:
        >>> logits = torch.randn(2, 10, 100)
        >>> labels = torch.randint(0, 100, (2, 10))
        >>> acc = compute_token_accuracy(logits, labels)
        >>> assert 0.0 <= acc <= 1.0
    """
    # Get predicted token IDs (argmax over vocabulary dimension)
    predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

    # Create mask for non-ignored tokens
    mask = labels != ignore_index

    # Count correct predictions
    correct = (predictions == labels) & mask
    num_correct = correct.sum().float()

    # Count total non-ignored tokens
    num_total = mask.sum().float()

    # Compute accuracy (handle case where all tokens are ignored)
    if num_total > 0:
        accuracy = num_correct / num_total
    else:
        accuracy = torch.tensor(0.0, device=logits.device)

    # Validate postconditions
    assert torch.isfinite(accuracy), "Accuracy must be finite"
    assert 0.0 <= accuracy <= 1.0, f"Accuracy must be in [0, 1], got {accuracy}"

    return accuracy


def evaluate_latent_forecasting(
    model: nn.Module, dataloader: DataLoader, device: str = "cuda"
) -> Dict[int, float]:
    """
    Evaluate latent forecasting performance on a dataset.

    Computes Mean Squared Error (MSE) between predicted and actual latent
    states for each forecasting horizon. Lower MSE indicates better
    forecasting accuracy.

    Args:
        model: Model to evaluate (should return ModelOutput with predicted_latents)
        dataloader: DataLoader providing batches with 'input_ids'
        device: Device for computation ('cuda' or 'cpu')

    Returns:
        Dictionary mapping horizon k to average MSE:
            {1: mse_1, 2: mse_2, 5: mse_5, 10: mse_10, ...}

    Preconditions:
        - model is in eval mode or will be set to eval mode
        - dataloader provides valid batches
        - device is valid ('cuda' or 'cpu')
        - model.forward() returns ModelOutput with predicted_latents dict
        - model has forecast_horizons configured

    Postconditions:
        - Returns dict with one entry per forecasting horizon
        - All MSE values are non-negative floats
        - All MSE values are finite
        - MSE values are averaged across all test batches
        - Model remains in eval mode
        - No gradients are computed

    Examples:
        >>> model = build_model(config)
        >>> results = evaluate_latent_forecasting(model, test_loader)
        >>> assert all(k in results for k in [1, 2, 5, 10])
        >>> assert all(mse >= 0 for mse in results.values())
    """
    # Set model to eval mode
    model.eval()

    # Initialize accumulators for each horizon
    horizon_mse_sum = {}
    num_batches = 0

    # Disable gradient computation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            tokens = batch["input_ids"].to(device)

            # Forward pass with latent loss computation
            output = model(tokens, compute_latent_loss=True)

            # Get latents and predictions
            latents = output.latents  # [batch_size, seq_len, latent_dim]
            predicted_latents = output.predicted_latents  # {horizon: predictions}

            # Compute MSE for each forecasting horizon
            for horizon, predictions in predicted_latents.items():
                # Get target latents at time t+k
                target_latents = latents[
                    :, horizon:, :
                ]  # [batch_size, seq_len-k, latent_dim]

                # Compute MSE for this horizon
                mse = F.mse_loss(predictions, target_latents, reduction="mean")

                # Accumulate MSE
                if horizon not in horizon_mse_sum:
                    horizon_mse_sum[horizon] = 0.0
                horizon_mse_sum[horizon] += mse.item()

            num_batches += 1

    # Validate we processed some data
    if num_batches == 0:
        raise ValueError("Dataloader is empty, no batches to evaluate")

    # Compute average MSE for each horizon
    horizon_mse_avg = {}
    for horizon, mse_sum in horizon_mse_sum.items():
        avg_mse = mse_sum / num_batches

        # Validate postconditions for this horizon
        assert (
            avg_mse >= 0
        ), f"MSE must be non-negative, got {avg_mse} for horizon {horizon}"
        assert torch.isfinite(
            torch.tensor(avg_mse)
        ), f"MSE must be finite, got {avg_mse} for horizon {horizon}"

        horizon_mse_avg[horizon] = avg_mse

    return horizon_mse_avg


def compute_latent_predictability_score(
    model: nn.Module, dataloader: DataLoader, device: str = "cuda"
) -> Dict[int, float]:
    """
    Compute Latent Predictability Score (LPS) for each forecasting horizon.

    LPS(k) measures the L2 distance between predicted and actual latent states
    at horizon k. Lower LPS indicates more predictable latent representations.

    LPS(k) = ||z_{t+k} - z_hat_{t+k}||_2

    Args:
        model: Model to evaluate (should return ModelOutput with predicted_latents)
        dataloader: DataLoader providing batches with 'input_ids'
        device: Device for computation ('cuda' or 'cpu')

    Returns:
        Dictionary mapping horizon k to average LPS score:
            {1: lps_1, 2: lps_2, 5: lps_5, 10: lps_10, ...}

    Preconditions:
        - model is in eval mode or will be set to eval mode
        - dataloader provides valid batches
        - device is valid ('cuda' or 'cpu')
        - model.forward() returns ModelOutput with predicted_latents dict
        - model has forecast_horizons configured

    Postconditions:
        - Returns dict with one entry per forecasting horizon
        - All LPS values are non-negative floats
        - All LPS values are finite
        - LPS values are averaged across all test batches
        - Model remains in eval mode
        - No gradients are computed

    Examples:
        >>> model = build_model(config)
        >>> lps_scores = compute_latent_predictability_score(model, test_loader)
        >>> assert all(k in lps_scores for k in [1, 2, 5, 10])
        >>> assert all(score >= 0 for score in lps_scores.values())
    """
    # Set model to eval mode
    model.eval()

    # Initialize accumulators for each horizon
    horizon_lps_sum = {}
    num_batches = 0

    # Disable gradient computation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            tokens = batch["input_ids"].to(device)

            # Forward pass with latent loss computation
            output = model(tokens, compute_latent_loss=True)

            # Get latents and predictions
            latents = output.latents  # [batch_size, seq_len, latent_dim]
            predicted_latents = output.predicted_latents  # {horizon: predictions}

            # Compute L2 norm for each forecasting horizon
            for horizon, predictions in predicted_latents.items():
                # Get target latents at time t+k
                target_latents = latents[
                    :, horizon:, :
                ]  # [batch_size, seq_len-k, latent_dim]

                # Compute difference: z_{t+k} - z_hat_{t+k}
                diff = (
                    target_latents - predictions
                )  # [batch_size, seq_len-k, latent_dim]

                # Compute L2 norm: ||diff||_2
                # Compute norm along latent dimension, then average over batch and sequence
                l2_norm = torch.norm(diff, p=2, dim=-1)  # [batch_size, seq_len-k]
                avg_l2_norm = l2_norm.mean()  # Average over batch and sequence

                # Accumulate LPS
                if horizon not in horizon_lps_sum:
                    horizon_lps_sum[horizon] = 0.0
                horizon_lps_sum[horizon] += avg_l2_norm.item()

            num_batches += 1

    # Validate we processed some data
    if num_batches == 0:
        raise ValueError("Dataloader is empty, no batches to evaluate")

    # Compute average LPS for each horizon
    horizon_lps_avg = {}
    for horizon, lps_sum in horizon_lps_sum.items():
        avg_lps = lps_sum / num_batches

        # Validate postconditions for this horizon
        assert (
            avg_lps >= 0
        ), f"LPS must be non-negative, got {avg_lps} for horizon {horizon}"
        assert torch.isfinite(
            torch.tensor(avg_lps)
        ), f"LPS must be finite, got {avg_lps} for horizon {horizon}"

        horizon_lps_avg[horizon] = avg_lps

    return horizon_lps_avg


def evaluate_language_modeling(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    compute_accuracy: bool = True,
) -> Dict[str, float]:
    """
    Evaluate language modeling performance on a dataset.

    Computes perplexity, token loss, and optionally token accuracy
    by running the model on all batches in the dataloader.

    Args:
        model: Model to evaluate (should have forward method returning ModelOutput)
        dataloader: DataLoader providing batches with 'input_ids' and 'labels'
        device: Device for computation ('cuda' or 'cpu')
        compute_accuracy: Whether to compute token accuracy (can be slow)

    Returns:
        Dictionary containing:
            - 'perplexity': Perplexity value (float)
            - 'token_loss': Average cross-entropy loss (float)
            - 'accuracy': Token accuracy (float, if compute_accuracy=True)

    Preconditions:
        - model is in eval mode or will be set to eval mode
        - dataloader provides valid batches
        - device is valid ('cuda' or 'cpu')
        - model.forward() returns object with .token_loss and .logits attributes

    Postconditions:
        - Returns dict with required keys
        - perplexity = exp(token_loss) within numerical precision
        - All values are finite floats
        - Model remains in eval mode
        - No gradients are computed

    Examples:
        >>> model = build_model(config)
        >>> results = evaluate_language_modeling(model, test_loader)
        >>> assert 'perplexity' in results
        >>> assert 'token_loss' in results
        >>> assert results['perplexity'] > 0
    """
    # Set model to eval mode
    model.eval()

    # Initialize accumulators
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    num_tokens = 0

    # Disable gradient computation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass (without latent loss for efficiency)
            output = model(tokens, labels=labels, compute_latent_loss=False)

            # Accumulate token loss
            batch_loss = output.token_loss.item()
            total_loss += batch_loss

            # Compute accuracy if requested
            if compute_accuracy:
                batch_accuracy = compute_token_accuracy(output.logits, labels)
                total_accuracy += batch_accuracy.item()

            num_batches += 1
            num_tokens += (labels != -100).sum().item()

    # Validate we processed some data
    if num_batches == 0:
        raise ValueError("Dataloader is empty, no batches to evaluate")

    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_loss_tensor = torch.tensor(avg_loss)

    # Compute perplexity
    perplexity = compute_perplexity(avg_loss_tensor)

    # Build results dictionary
    results = {
        "perplexity": perplexity.item(),
        "token_loss": avg_loss,
    }

    if compute_accuracy:
        avg_accuracy = total_accuracy / num_batches
        results["accuracy"] = avg_accuracy

    # Validate postconditions
    assert torch.isfinite(perplexity), "Perplexity must be finite"
    assert (
        abs(results["perplexity"] - torch.exp(avg_loss_tensor).item()) < 1e-5
    ), "Perplexity must equal exp(loss)"

    return results


class Evaluator:
    """
    Comprehensive evaluator for Latent Forecasting Network models.

    This class provides a unified interface for computing all evaluation
    metrics including language modeling metrics, latent forecasting metrics,
    and representation analysis.
    """

    def __init__(
        self, model: nn.Module, device: str = "cuda", compute_accuracy: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device for computation
            compute_accuracy: Whether to compute token accuracy
        """
        self.model = model
        self.device = device
        self.compute_accuracy = compute_accuracy

        # Set model to eval mode
        self.model.eval()

    def evaluate_language_modeling(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate language modeling performance.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary with perplexity, token_loss, and optionally accuracy
        """
        return evaluate_language_modeling(
            self.model, dataloader, self.device, self.compute_accuracy
        )

    def evaluate_latent_forecasting(self, dataloader: DataLoader) -> Dict[int, float]:
        """
        Evaluate latent forecasting performance.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary mapping horizon to MSE
        """
        return evaluate_latent_forecasting(self.model, dataloader, self.device)

    def compute_latent_predictability_score(
        self, dataloader: DataLoader
    ) -> Dict[int, float]:
        """
        Compute Latent Predictability Score for each horizon.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary mapping horizon to LPS score
        """
        return compute_latent_predictability_score(self.model, dataloader, self.device)

    def analyze_representations(
        self, dataloader: DataLoader, max_samples: int = 10000, n_clusters: int = 10
    ) -> Dict[str, float]:
        """
        Analyze latent representation quality.

        Computes representation metrics including entropy, variance,
        cosine similarity drift, and cluster separability.

        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum number of latent samples to collect
            n_clusters: Number of clusters for separability analysis

        Returns:
            Dictionary with representation metrics:
                - latent_entropy: Entropy value (float >= 0)
                - latent_variance: Average variance (float >= 0)
                - cosine_similarity_drift: Drift value (float in [0, 2])
                - cluster_separability: Silhouette score (float in [-1, 1])
        """
        if not LATENT_ANALYSIS_AVAILABLE:
            raise ImportError(
                "Latent analysis module not available. "
                "Please ensure evaluation.latent_analysis is installed."
            )

        return analyze_representations(
            self.model,
            dataloader,
            self.device,
            max_samples=max_samples,
            n_clusters=n_clusters,
        )

    def evaluate_model(
        self,
        dataloader: DataLoader,
        include_representation_analysis: bool = False,
        max_samples: int = 10000,
        n_clusters: int = 10,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Computes language modeling metrics, latent forecasting metrics,
        latent predictability scores, and optionally representation analysis.

        Args:
            dataloader: DataLoader for evaluation
            include_representation_analysis: Whether to compute representation metrics
            max_samples: Maximum samples for representation analysis
            n_clusters: Number of clusters for separability analysis

        Returns:
            Dictionary with all evaluation metrics:
                - Language modeling: perplexity, token_loss, accuracy
                - Latent forecasting: horizon_mse (dict mapping horizon to MSE)
                - Latent predictability: lps_scores (dict mapping horizon to LPS)
                - Representation analysis (if enabled): latent_entropy, latent_variance,
                  cosine_similarity_drift, cluster_separability
        """
        results = {}

        # Language modeling metrics
        lm_metrics = self.evaluate_language_modeling(dataloader)
        results.update(lm_metrics)

        # Latent forecasting metrics
        try:
            lf_metrics = self.evaluate_latent_forecasting(dataloader)
            results["horizon_mse"] = lf_metrics
        except Exception as e:
            # If model doesn't support latent forecasting, skip gracefully
            print(f"Warning: Could not compute latent forecasting metrics: {e}")

        # Latent predictability scores
        try:
            lps_scores = self.compute_latent_predictability_score(dataloader)
            results["lps_scores"] = lps_scores
        except Exception as e:
            # If model doesn't support latent forecasting, skip gracefully
            print(f"Warning: Could not compute latent predictability scores: {e}")

        # Representation analysis (optional, can be slow)
        if include_representation_analysis:
            try:
                rep_metrics = self.analyze_representations(
                    dataloader, max_samples=max_samples, n_clusters=n_clusters
                )
                results["representation_metrics"] = rep_metrics
            except Exception as e:
                print(f"Warning: Could not compute representation metrics: {e}")

        return results

    def compute_effective_dimensionality(
        self, dataloader: DataLoader, max_samples: int = 5000
    ) -> float:
        """
        Compute effective dimensionality of model representations.

        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum number of samples to use

        Returns:
            Effective dimensionality (participation ratio)
        """
        if not DOWNSTREAM_AVAILABLE:
            raise ImportError(
                "Downstream evaluation module not available. "
                "Please ensure evaluation.downstream_eval is installed."
            )

        # Extract representations
        reps, _ = extract_pooled_representations(
            self.model, dataloader, self.device, pooling="mean"
        )

        # Limit samples if needed
        if reps.size(0) > max_samples:
            indices = torch.randperm(reps.size(0))[:max_samples]
            reps = reps[indices]

        return compute_effective_dimensionality(reps)

    def evaluate_downstream(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        task_name: str = "classification",
        num_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate model on downstream task via linear probing.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_classes: Number of classes
            task_name: Name of the task
            num_epochs: Training epochs for linear probe

        Returns:
            Dictionary with downstream evaluation metrics
        """
        if not DOWNSTREAM_AVAILABLE:
            raise ImportError(
                "Downstream evaluation module not available. "
                "Please ensure evaluation.downstream_eval is installed."
            )

        evaluator = DownstreamEvaluator(self.model, device=self.device)
        return evaluator.evaluate_classification(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            task_name=task_name,
            num_epochs=num_epochs,
        )

    def evaluate_model_extended(
        self,
        dataloader: DataLoader,
        include_representation_analysis: bool = True,
        include_downstream: bool = False,
        downstream_loaders: Optional[Dict] = None,
        max_samples: int = 10000,
        n_clusters: int = 10,
    ) -> Dict[str, Any]:
        """
        Extended comprehensive model evaluation with downstream tasks.

        Args:
            dataloader: DataLoader for main evaluation
            include_representation_analysis: Whether to compute representation metrics
            include_downstream: Whether to compute downstream metrics
            downstream_loaders: Dict with 'train', 'val', 'test' loaders
            max_samples: Maximum samples for representation analysis
            n_clusters: Number of clusters for separability analysis

        Returns:
            Dictionary with all evaluation metrics including downstream
        """
        # Get base metrics
        results = self.evaluate_model(
            dataloader,
            include_representation_analysis=include_representation_analysis,
            max_samples=max_samples,
            n_clusters=n_clusters,
        )

        # Add effective dimensionality
        try:
            results["effective_dimensionality"] = self.compute_effective_dimensionality(
                dataloader, max_samples=max_samples
            )
        except Exception as e:
            print(f"Warning: Could not compute effective dimensionality: {e}")

        # Add downstream evaluation
        if include_downstream and downstream_loaders is not None:
            try:
                num_classes = downstream_loaders.get("num_classes", 2)
                task_name = downstream_loaders.get("task_name", "classification")

                downstream_results = self.evaluate_downstream(
                    train_loader=downstream_loaders["train"],
                    val_loader=downstream_loaders["val"],
                    test_loader=downstream_loaders["test"],
                    num_classes=num_classes,
                    task_name=task_name,
                )
                results["downstream"] = downstream_results
            except Exception as e:
                print(f"Warning: Could not compute downstream metrics: {e}")

        return results
