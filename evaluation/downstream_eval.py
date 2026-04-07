"""
Downstream evaluation tasks for validating transfer properties of representations.

This module implements:
- Linear probing for various downstream tasks
- Sentiment analysis (SST-2)
- Text classification (AG News)
- Linguistic property probing (POS tagging, syntactic depth)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class LinearProbe(nn.Module):
    """
    Simple linear probe for evaluating representation quality.

    A linear probe is a linear classifier trained on top of frozen
    representations to measure how well they capture task-relevant information.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        """
        Initialize linear probe.

        Args:
            input_dim: Dimension of input representations
            num_classes: Number of output classes
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through linear probe.

        Args:
            x: Input representations [batch_size, input_dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.dropout(x)
        return self.classifier(x)


def extract_pooled_representations(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    pooling: str = "mean",
) -> Tuple[Tensor, Tensor]:
    """
    Extract and pool latent representations from a model.

    Args:
        model: Model to extract representations from
        dataloader: DataLoader providing batches with 'input_ids'
        device: Device for computation
        pooling: Pooling strategy ('mean', 'cls', 'last')

    Returns:
        Tuple of (representations, labels) tensors
    """
    model.eval()
    all_reps = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["input_ids"].to(device)
            labels = batch.get("labels", torch.zeros(tokens.size(0)))

            # Get model output
            if hasattr(model, 'encoder'):
                # For LFN and baseline models
                latents = model.encoder(tokens)
            elif hasattr(model, 'forward'):
                # Generic model
                output = model(tokens, compute_latent_loss=False)
                latents = output.latents
            else:
                raise ValueError("Model does not have expected interface")

            # Pool representations
            if pooling == "mean":
                # Mean pool over sequence dimension
                pooled = latents.mean(dim=1)  # [batch_size, latent_dim]
            elif pooling == "cls":
                # Use first token representation
                pooled = latents[:, 0, :]  # [batch_size, latent_dim]
            elif pooling == "last":
                # Use last token representation
                # Find actual sequence lengths if mask available
                pooled = latents[:, -1, :]  # [batch_size, latent_dim]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_reps.append(pooled.cpu())
            all_labels.append(labels)

    return torch.cat(all_reps, dim=0), torch.cat(all_labels, dim=0)


def train_linear_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    pooling: str = "mean",
) -> Tuple[LinearProbe, Dict[str, float]]:
    """
    Train a linear probe on frozen representations.

    Args:
        model: Pretrained model to extract representations from
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of output classes
        device: Device for computation
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        pooling: Pooling strategy

    Returns:
        Tuple of (trained probe, training metrics)
    """
    # Extract representations
    train_reps, train_labels = extract_pooled_representations(
        model, train_loader, device, pooling
    )
    val_reps, val_labels = extract_pooled_representations(
        model, val_loader, device, pooling
    )

    # Move to device
    train_reps = train_reps.to(device)
    train_labels = train_labels.to(device)
    val_reps = val_reps.to(device)
    val_labels = val_labels.to(device)

    # Create probe
    input_dim = train_reps.size(-1)
    probe = LinearProbe(input_dim, num_classes).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # Train
        probe.train()
        logits = probe(train_reps)
        loss = criterion(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_reps)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = probe.state_dict().copy()

    # Load best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    metrics = {
        "best_val_accuracy": best_val_acc,
        "num_epochs": num_epochs,
        "final_train_loss": loss.item(),
    }

    return probe, metrics


def evaluate_linear_probe(
    model: nn.Module,
    probe: LinearProbe,
    test_loader: DataLoader,
    device: str = "cuda",
    pooling: str = "mean",
) -> Dict[str, float]:
    """
    Evaluate a trained linear probe on test data.

    Args:
        model: Model to extract representations from
        probe: Trained linear probe
        test_loader: Test data loader
        device: Device for computation
        pooling: Pooling strategy

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract representations
    test_reps, test_labels = extract_pooled_representations(
        model, test_loader, device, pooling
    )
    test_reps = test_reps.to(device)
    test_labels = test_labels.to(device)

    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(test_reps)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == test_labels).float().mean().item()

        # Compute per-class accuracy if applicable
        num_classes = logits.size(-1)
        per_class_acc = {}
        for c in range(num_classes):
            mask = test_labels == c
            if mask.sum() > 0:
                per_class_acc[f"class_{c}_accuracy"] = (
                    (preds[mask] == test_labels[mask]).float().mean().item()
                )

    results = {
        "accuracy": accuracy,
        **per_class_acc,
    }

    return results


def run_probing_evaluation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    task_name: str,
    device: str = "cuda",
    num_epochs: int = 10,
) -> Dict[str, float]:
    """
    Complete pipeline for linear probing evaluation.

    Args:
        model: Model to evaluate
        train_loader: Training data
        val_loader: Validation data
        test_loader: Test data
        num_classes: Number of classes
        task_name: Name of the task
        device: Device for computation
        num_epochs: Training epochs

    Returns:
        Dictionary with probing results
    """
    # Train probe
    probe, train_metrics = train_linear_probe(
        model, train_loader, val_loader, num_classes, device, num_epochs
    )

    # Evaluate on test set
    test_metrics = evaluate_linear_probe(model, probe, test_loader, device)

    return {
        "task": task_name,
        "linear_probe_accuracy": test_metrics["accuracy"],
        **train_metrics,
    }


def compute_representation_similarity(
    model1: nn.Module,
    model2: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    method: str = "cka",
) -> float:
    """
    Compute representation similarity between two models.

    Args:
        model1: First model
        model2: Second model
        dataloader: DataLoader providing data
        device: Device for computation
        method: Similarity method ('cka' or 'correlation')

    Returns:
        Similarity score (0 to 1, higher = more similar)
    """
    reps1, _ = extract_pooled_representations(model1, dataloader, device)
    reps2, _ = extract_pooled_representations(model2, dataloader, device)

    if method == "cka":
        return compute_cka(reps1, reps2)
    elif method == "correlation":
        return compute_correlation(reps1, reps2)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_cka(X: Tensor, Y: Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two representation matrices.

    CKA measures the similarity between two sets of representations
    without requiring them to be in the same coordinate system.

    Args:
        X: First representation matrix [n_samples, d1]
        Y: Second representation matrix [n_samples, d2]

    Returns:
        CKA score in [0, 1]
    """
    # Center the representations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute kernels
    K = X @ X.T
    L = Y @ Y.T

    # CKA formula
    hsic = (K * L).sum()
    norm_k = (K * K).sum().sqrt()
    norm_l = (L * L).sum().sqrt()

    cka = hsic / (norm_k * norm_l + 1e-8)
    return cka.item()


def compute_correlation(X: Tensor, Y: Tensor) -> float:
    """
    Compute average correlation between aligned representations.

    Args:
        X: First representation matrix [n_samples, d]
        Y: Second representation matrix [n_samples, d]

    Returns:
        Average correlation coefficient
    """
    # Normalize
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
    Y = (Y - Y.mean(dim=0)) / (Y.std(dim=0) + 1e-8)

    # Compute correlation
    n = X.size(0)
    corr_matrix = (X.T @ Y) / n

    # Average diagonal (aligned dimensions)
    return corr_matrix.diag().mean().item()


class DownstreamEvaluator:
    """
    Evaluator for downstream task performance.

    This class provides a unified interface for evaluating how well
    learned representations transfer to downstream tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        pooling: str = "mean",
    ):
        """
        Initialize downstream evaluator.

        Args:
            model: Model to evaluate
            device: Device for computation
            pooling: Pooling strategy for representations
        """
        self.model = model
        self.device = device
        self.pooling = pooling

    def evaluate_sentiment(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate on sentiment analysis task.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_epochs: Training epochs

        Returns:
            Dictionary with evaluation metrics
        """
        return run_probing_evaluation(
            self.model,
            train_loader,
            val_loader,
            test_loader,
            num_classes=2,
            task_name="sentiment_analysis",
            device=self.device,
            num_epochs=num_epochs,
        )

    def evaluate_classification(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        task_name: str = "classification",
        num_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate on text classification task.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_classes: Number of classes
            task_name: Name of classification task
            num_epochs: Training epochs

        Returns:
            Dictionary with evaluation metrics
        """
        return run_probing_evaluation(
            self.model,
            train_loader,
            val_loader,
            test_loader,
            num_classes=num_classes,
            task_name=task_name,
            device=self.device,
            num_epochs=num_epochs,
        )

    def compute_transfer_metrics(
        self,
        baseline_model: nn.Module,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Compute metrics that compare model to baseline for transfer learning.

        Args:
            baseline_model: Baseline model for comparison
            test_loader: Test data loader

        Returns:
            Dictionary with transfer metrics
        """
        # Extract representations from both models
        model_reps, _ = extract_pooled_representations(
            self.model, test_loader, self.device, self.pooling
        )
        baseline_reps, _ = extract_pooled_representations(
            baseline_model, test_loader, self.device, self.pooling
        )

        # Compute CKA similarity
        cka = compute_cka(model_reps, baseline_reps)

        # Compute representational complexity (effective dimensionality)
        model_complexity = compute_effective_dimensionality(model_reps)
        baseline_complexity = compute_effective_dimensionality(baseline_reps)

        return {
            "cka_vs_baseline": cka,
            "model_effective_dim": model_complexity,
            "baseline_effective_dim": baseline_complexity,
            "relative_complexity": model_complexity / (baseline_complexity + 1e-8),
        }


def compute_effective_dimensionality(representations: Tensor) -> float:
    """
    Compute effective dimensionality of representations using participation ratio.

    Args:
        representations: Representation matrix [n_samples, dim]

    Returns:
        Effective dimensionality
    """
    # Compute covariance
    centered = representations - representations.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / centered.size(0)

    # Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(cov)

    # Participation ratio
    sum_sq = eigenvalues.sum() ** 2
    sum_of_sq = (eigenvalues ** 2).sum()

    pr = sum_sq / (sum_of_sq + 1e-8)
    return pr.item()
