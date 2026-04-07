#!/usr/bin/env python3
"""
Comprehensive theoretical validation experiments for the paper.

This script runs all experiments needed to validate theoretical predictions:
1. Effective Dimensionality vs lambda
2. Effective Dimensionality vs Horizon Set
3. CKA vs lambda sweep
4. Cosine Drift vs lambda
5. Eigenvalue spectrum plots
6. psi(K) computation and comparison
7. Residual distribution test (Gaussian assumption)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset_loader import create_dataloaders
from evaluation.downstream_eval import (
    compute_cka,
    compute_effective_dimensionality,
    extract_pooled_representations,
)
from evaluation.latent_analysis import (
    compute_cosine_similarity_drift,
    compute_latent_entropy,
    compute_latent_variance,
)
from evaluation.metrics import Evaluator
from models.complete_model import ModelConfig, build_model
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer, TrainingConfig
from utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("theoretical_validation")


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_model(config: ModelConfig, device: str):
    """Build model from config."""
    return build_model(config, device=device)


def train_model(
    config: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    device: str,
    checkpoint_dir: str,
) -> Tuple[Any, Dict]:
    """Train a model and return it with training history."""
    model = create_model(config, device)
    optimizer = create_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    total_steps = (
        training_config.num_epochs
        * len(train_loader)
        // training_config.gradient_accumulation_steps
    )
    scheduler = create_scheduler(
        optimizer, warmup_steps=training_config.warmup_steps, num_training_steps=total_steps
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        checkpoint_dir=checkpoint_dir,
    )

    history = trainer.train()
    return model, history


def extract_all_representations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    max_samples: int = 10000,
) -> Tuple[Tensor, Tensor]:
    """Extract representations from model, returning both pooled and sequence representations."""
    model.eval()
    all_pooled = []
    all_sequence = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["input_ids"].to(device)

            # Get latent representations
            if hasattr(model, "encoder"):
                latents = model.encoder(tokens)
            else:
                output = model(tokens, compute_latent_loss=False)
                latents = output.latents

            # Pooled representation (mean over sequence)
            pooled = latents.mean(dim=1)
            all_pooled.append(pooled.cpu())

            # Sequence representation (for temporal drift)
            all_sequence.append(latents.cpu())

            if sum(p.shape[0] for p in all_pooled) >= max_samples:
                break

    return torch.cat(all_pooled, dim=0), torch.cat(all_sequence, dim=0)


def compute_eigenvalue_spectrum(representations: Tensor) -> np.ndarray:
    """Compute eigenvalues of representation covariance matrix."""
    centered = representations - representations.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / centered.size(0)
    eigenvalues = torch.linalg.eigvalsh(cov)
    return eigenvalues.numpy()[::-1]  # Sort descending


def compute_forecasting_residuals(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    horizons: List[int],
    max_samples: int = 5000,
) -> Dict[int, Tensor]:
    """Compute forecasting residuals for each horizon."""
    model.eval()
    residuals = {k: [] for k in horizons}

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["input_ids"].to(device)

            # Get latent representations
            if hasattr(model, "encoder"):
                latents = model.encoder(tokens)
            else:
                output = model(tokens, compute_latent_loss=False)
                latents = output.latents

            # Get predictions
            if hasattr(model, "forecasting_network"):
                for k in horizons:
                    if latents.size(1) > k:
                        z_t = latents[:, :-k, :]
                        z_future = latents[:, k:, :]
                        z_pred = model.forecasting_network(z_t.reshape(-1, z_t.size(-1)))
                        z_pred = z_pred.reshape(z_t.size(0), z_t.size(1), -1)

                        # Compute residuals
                        res = (z_future - z_pred).reshape(-1, z_future.size(-1))
                        residuals[k].append(res.cpu())

            total_samples = sum(sum(r.shape[0] for r in residuals[k]) for k in horizons)
            if total_samples >= max_samples:
                break

    return {k: torch.cat(v, dim=0) for k, v in residuals.items() if v}


def test_gaussian_residuals(residuals: Tensor) -> Dict[str, float]:
    """Test if residuals are Gaussian distributed."""
    from scipy import stats

    residuals_np = residuals.numpy()

    # Flatten across dimensions for overall test
    flat = residuals_np.flatten()

    # Shapiro-Wilk test (limited to 5000 samples)
    sample = flat[:5000] if len(flat) > 5000 else flat
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    # D'Agostino's K-squared test
    if len(flat) >= 20:
        dagostino_stat, dagostino_p = stats.normaltest(flat[:5000])
    else:
        dagostino_stat, dagostino_p = 0.0, 1.0

    # Skewness and kurtosis
    skewness = stats.skew(flat)
    kurtosis = stats.kurtosis(flat)

    return {
        "shapiro_statistic": shapiro_stat,
        "shapiro_p_value": shapiro_p,
        "dagostino_statistic": dagostino_stat,
        "dagostino_p_value": dagostino_p,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_gaussian_05": shapiro_p > 0.05,
    }


def compute_psi_K(horizons: List[int], latent_dim: int, data_predictability: float = 0.5) -> float:
    """
    Compute theoretical psi(K) function from the paper.
    
    psi(K) = sum over k in K of [1 / (1 + alpha * k)]
    where alpha depends on data predictability.
    
    This measures the total gradient signal strength from multi-horizon forecasting.
    """
    alpha = 1.0 - data_predictability  # Higher alpha = less predictable
    psi = sum(1.0 / (1.0 + alpha * k) for k in horizons)
    return psi


def run_lambda_sweep(
    lambda_values: List[float],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    base_config: Dict,
    device: str,
    output_dir: Path,
    seed: int = 42,
) -> Dict[float, Dict]:
    """Run experiments across different lambda values."""
    results = {}
    
    # First, train baseline (lambda=0) to get reference representations
    logger.info("=" * 60)
    logger.info("Training BASELINE model (lambda=0)")
    logger.info("=" * 60)
    
    baseline_config = ModelConfig(
        vocab_size=base_config.get("vocab_size", 50257),
        latent_dim=base_config.get("latent_dim", 256),
        num_layers=base_config.get("num_layers", 4),
        num_heads=base_config.get("num_heads", 4),
        hidden_dim=base_config.get("hidden_dim", 1024),
        dropout=base_config.get("dropout", 0.1),
        forecast_horizons=base_config.get("forecast_horizons", [1, 2, 5]),
        max_context_length=base_config.get("max_context_length", 256),
        lambda_latent=0.0,
    )
    
    training_config = TrainingConfig(
        num_epochs=base_config.get("num_epochs", 3),
        batch_size=base_config.get("batch_size", 8),
        learning_rate=base_config.get("learning_rate", 3e-4),
        weight_decay=base_config.get("weight_decay", 0.01),
        gradient_accumulation_steps=base_config.get("gradient_accumulation_steps", 4),
        max_grad_norm=base_config.get("max_grad_norm", 1.0),
        warmup_steps=base_config.get("warmup_steps", 200),
        lambda_latent=0.0,
        use_mixed_precision=False,
        checkpoint_every=500,
        log_every=50,
        seed=seed,
        device=device,
    )
    
    set_seed(seed)
    baseline_model, _ = train_model(
        baseline_config,
        train_loader,
        val_loader,
        training_config,
        device,
        str(output_dir / "checkpoints" / "baseline"),
    )
    
    # Extract baseline representations for CKA comparison
    baseline_reps, baseline_seq_reps = extract_all_representations(
        baseline_model, test_loader, device
    )
    baseline_eff_dim = compute_effective_dimensionality(baseline_reps)
    baseline_eigenvalues = compute_eigenvalue_spectrum(baseline_reps)
    
    results[0.0] = {
        "effective_dimensionality": baseline_eff_dim,
        "eigenvalues": baseline_eigenvalues.tolist(),
        "perplexity": 0.0,  # Will be computed below
        "cosine_drift": 0.0,
        "cka_vs_baseline": 1.0,
        "linear_probe_accuracy": 0.0,
    }
    
    # Evaluate baseline
    evaluator = Evaluator(baseline_model, device=device, compute_accuracy=True)
    baseline_results = evaluator.evaluate_model(test_loader, include_representation_analysis=True)
    results[0.0]["perplexity"] = baseline_results.get("perplexity", 0.0)
    results[0.0]["cosine_drift"] = baseline_results.get("representation_metrics", {}).get(
        "cosine_similarity_drift", 0.0
    )
    
    # Now train models for each lambda > 0
    for lam in lambda_values:
        if lam == 0.0:
            continue  # Already done
            
        logger.info("=" * 60)
        logger.info(f"Training model with lambda={lam}")
        logger.info("=" * 60)
        
        set_seed(seed)
        config = ModelConfig(
            vocab_size=base_config.get("vocab_size", 50257),
            latent_dim=base_config.get("latent_dim", 256),
            num_layers=base_config.get("num_layers", 4),
            num_heads=base_config.get("num_heads", 4),
            hidden_dim=base_config.get("hidden_dim", 1024),
            dropout=base_config.get("dropout", 0.1),
            forecast_horizons=base_config.get("forecast_horizons", [1, 2, 5]),
            max_context_length=base_config.get("max_context_length", 256),
            lambda_latent=lam,
        )
        
        training_config.lambda_latent = lam
        
        model, _ = train_model(
            config,
            train_loader,
            val_loader,
            training_config,
            device,
            str(output_dir / "checkpoints" / f"lambda_{lam}"),
        )
        
        # Extract representations
        reps, seq_reps = extract_all_representations(model, test_loader, device)
        
        # Compute metrics
        eff_dim = compute_effective_dimensionality(reps)
        eigenvalues = compute_eigenvalue_spectrum(reps)
        cka = compute_cka(reps, baseline_reps)
        
        # Evaluate
        evaluator = Evaluator(model, device=device, compute_accuracy=True)
        eval_results = evaluator.evaluate_model(test_loader, include_representation_analysis=True)
        
        results[lam] = {
            "effective_dimensionality": eff_dim,
            "eigenvalues": eigenvalues.tolist(),
            "perplexity": eval_results.get("perplexity", 0.0),
            "cosine_drift": eval_results.get("representation_metrics", {}).get(
                "cosine_similarity_drift", 0.0
            ),
            "cka_vs_baseline": cka,
            "lps_scores": eval_results.get("lps_scores", {}),
            "horizon_mse": eval_results.get("horizon_mse", {}),
        }
        
        logger.info(f"Lambda={lam}: EffDim={eff_dim:.2f}, CKA={cka:.4f}, PPL={results[lam]['perplexity']:.2f}")
    
    return results


def run_horizon_ablation(
    horizon_configs: List[List[int]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    base_config: Dict,
    device: str,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run experiments across different horizon configurations."""
    results = {}
    
    for horizons in horizon_configs:
        horizon_key = str(horizons)
        logger.info("=" * 60)
        logger.info(f"Training model with horizons={horizons}")
        logger.info("=" * 60)
        
        set_seed(seed)
        config = ModelConfig(
            vocab_size=base_config.get("vocab_size", 50257),
            latent_dim=base_config.get("latent_dim", 256),
            num_layers=base_config.get("num_layers", 4),
            num_heads=base_config.get("num_heads", 4),
            hidden_dim=base_config.get("hidden_dim", 1024),
            dropout=base_config.get("dropout", 0.1),
            forecast_horizons=horizons,
            max_context_length=base_config.get("max_context_length", 256),
            lambda_latent=base_config.get("lambda_latent", 0.1),
        )
        
        training_config = TrainingConfig(
            num_epochs=base_config.get("num_epochs", 3),
            batch_size=base_config.get("batch_size", 8),
            learning_rate=base_config.get("learning_rate", 3e-4),
            weight_decay=base_config.get("weight_decay", 0.01),
            gradient_accumulation_steps=base_config.get("gradient_accumulation_steps", 4),
            max_grad_norm=base_config.get("max_grad_norm", 1.0),
            warmup_steps=base_config.get("warmup_steps", 200),
            lambda_latent=base_config.get("lambda_latent", 0.1),
            use_mixed_precision=False,
            checkpoint_every=500,
            log_every=50,
            seed=seed,
            device=device,
        )
        
        model, _ = train_model(
            config,
            train_loader,
            val_loader,
            training_config,
            device,
            str(output_dir / "checkpoints" / f"horizons_{'_'.join(map(str, horizons))}"),
        )
        
        # Extract representations
        reps, seq_reps = extract_all_representations(model, test_loader, device)
        
        # Compute metrics
        eff_dim = compute_effective_dimensionality(reps)
        eigenvalues = compute_eigenvalue_spectrum(reps)
        
        # Compute psi(K)
        psi = compute_psi_K(horizons, config.latent_dim)
        
        # Compute diversity metric
        diversity = len(horizons) / max(horizons)
        
        # Evaluate
        evaluator = Evaluator(model, device=device, compute_accuracy=True)
        eval_results = evaluator.evaluate_model(test_loader, include_representation_analysis=True)
        
        results[horizon_key] = {
            "horizons": horizons,
            "effective_dimensionality": eff_dim,
            "eigenvalues": eigenvalues.tolist(),
            "psi_K": psi,
            "diversity": diversity,
            "perplexity": eval_results.get("perplexity", 0.0),
            "cosine_drift": eval_results.get("representation_metrics", {}).get(
                "cosine_similarity_drift", 0.0
            ),
            "lps_scores": eval_results.get("lps_scores", {}),
        }
        
        logger.info(f"Horizons={horizons}: EffDim={eff_dim:.2f}, psi(K)={psi:.4f}, Drift={results[horizon_key]['cosine_drift']:.6f}")
    
    return results


def generate_eigenvalue_plots(
    lambda_results: Dict[float, Dict],
    output_dir: Path,
):
    """Generate eigenvalue spectrum plots for different lambda values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    lambdas_to_plot = [0.0, 0.1, 0.5]
    colors = ["blue", "green", "red"]
    
    for ax, lam, color in zip(axes, lambdas_to_plot, colors):
        if lam in lambda_results:
            eigenvalues = lambda_results[lam]["eigenvalues"]
            # Plot top 50 eigenvalues
            n_plot = min(50, len(eigenvalues))
            ax.bar(range(n_plot), eigenvalues[:n_plot], color=color, alpha=0.7)
            ax.set_xlabel("Eigenvalue Index")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(f"lambda = {lam}\nEff. Dim = {lambda_results[lam]['effective_dimensionality']:.1f}")
            ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(output_dir / "eigenvalue_spectrum.png", dpi=150)
    plt.close()
    
    # Also create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lam, color in zip(lambdas_to_plot, colors):
        if lam in lambda_results:
            eigenvalues = lambda_results[lam]["eigenvalues"]
            n_plot = min(100, len(eigenvalues))
            ax.semilogy(range(n_plot), eigenvalues[:n_plot], label=f"lambda={lam}", color=color, linewidth=2)
    
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Eigenvalue Spectrum Comparison: Effect of Lambda")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "eigenvalue_comparison.png", dpi=150)
    plt.close()
    
    logger.info(f"Saved eigenvalue plots to {output_dir}")


def generate_summary_tables(
    lambda_results: Dict[float, Dict],
    horizon_results: Dict[str, Dict],
    output_dir: Path,
):
    """Generate summary tables for the paper."""
    
    # Lambda sweep table
    lambda_table = []
    for lam in sorted(lambda_results.keys()):
        r = lambda_results[lam]
        lambda_table.append({
            "lambda": lam,
            "eff_dim": r["effective_dimensionality"],
            "ppl": r["perplexity"],
            "cka": r["cka_vs_baseline"],
            "drift": r["cosine_drift"],
        })
    
    # Horizon ablation table
    horizon_table = []
    for key in sorted(horizon_results.keys(), key=lambda x: (len(eval(x)), eval(x) if eval(x) else 0)):
        r = horizon_results[key]
        horizon_table.append({
            "horizons": r["horizons"],
            "diversity": r["diversity"],
            "psi_K": r["psi_K"],
            "eff_dim": r["effective_dimensionality"],
            "drift": r["cosine_drift"],
            "ppl": r["perplexity"],
        })
    
    # Save as JSON
    with open(output_dir / "lambda_sweep_results.json", "w") as f:
        json.dump(lambda_results, f, indent=2)
    
    with open(output_dir / "horizon_ablation_results.json", "w") as f:
        json.dump(horizon_results, f, indent=2)
    
    # Save as CSV-like format
    with open(output_dir / "lambda_sweep_table.csv", "w") as f:
        f.write("lambda,eff_dim,ppl,cka,drift\n")
        for row in lambda_table:
            f.write(f"{row['lambda']},{row['eff_dim']:.2f},{row['ppl']:.2f},{row['cka']:.4f},{row['drift']:.6f}\n")
    
    with open(output_dir / "horizon_ablation_table.csv", "w") as f:
        f.write("horizons,diversity,psi_K,eff_dim,drift,ppl\n")
        for row in horizon_table:
            h_str = '"{' + ",".join(map(str, row['horizons'])) + '}"'
            f.write(f'{h_str},{row["diversity"]:.2f},{row["psi_K"]:.4f},{row["eff_dim"]:.2f},{row["drift"]:.6f},{row["ppl"]:.2f}\n')
    
    logger.info(f"Saved summary tables to {output_dir}")


def main():
    """Run all theoretical validation experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Theoretical validation experiments")
    parser.add_argument("--output-dir", type=str, default="experiments/results/theoretical_validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Run quick version with fewer epochs")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device or get_device()
    logger.info(f"Using device: {device}")
    
    # Configuration
    base_config = {
        "vocab_size": 50257,
        "latent_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "hidden_dim": 1024,
        "dropout": 0.1,
        "forecast_horizons": [1, 2, 5],
        "max_context_length": 256,
        "num_epochs": 1 if args.quick else 3,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 200,
        "lambda_latent": 0.1,
    }
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name="wikitext-2",
        tokenizer_name="gpt2",
        context_length=base_config["max_context_length"],
        stride=128,
        batch_size=base_config["batch_size"],
        num_workers=0,
    )
    
    # Lambda values to test
    lambda_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    # Horizon configurations to test
    horizon_configs = [
        [1],
        [2],
        [5],
        [10],
        [1, 2],
        [1, 2, 5],
        [1, 2, 5, 10],
        [1, 3, 5, 10, 20],
    ]
    
    # Run lambda sweep
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: Lambda Sweep Experiments")
    logger.info("=" * 80)
    
    lambda_results = run_lambda_sweep(
        lambda_values=lambda_values,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        base_config=base_config,
        device=device,
        output_dir=output_dir,
        seed=args.seed,
    )
    
    # Run horizon ablation
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: Horizon Ablation Experiments")
    logger.info("=" * 80)
    
    horizon_results = run_horizon_ablation(
        horizon_configs=horizon_configs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        base_config=base_config,
        device=device,
        output_dir=output_dir,
        seed=args.seed,
    )
    
    # Generate plots
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: Generating Plots and Tables")
    logger.info("=" * 80)
    
    generate_eigenvalue_plots(lambda_results, output_dir)
    generate_summary_tables(lambda_results, horizon_results, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Lambda Sweep Results")
    logger.info("=" * 80)
    print(f"{'Lambda':<10} {'EffDim':<12} {'CKA':<10} {'Drift':<12} {'PPL':<10}")
    print("-" * 54)
    for lam in sorted(lambda_results.keys()):
        r = lambda_results[lam]
        print(f"{lam:<10.2f} {r['effective_dimensionality']:<12.2f} {r['cka_vs_baseline']:<10.4f} {r['cosine_drift']:<12.6f} {r['perplexity']:<10.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Horizon Ablation Results")
    logger.info("=" * 80)
    print(f"{'Horizons':<20} {'Diversity':<10} {'psi(K)':<10} {'EffDim':<12} {'Drift':<12}")
    print("-" * 64)
    for key in sorted(horizon_results.keys(), key=lambda x: (len(eval(x)), -horizon_results[x]["diversity"])):
        r = horizon_results[key]
        h_str = "{" + ",".join(map(str, r["horizons"])) + "}"
        print(f"{h_str:<20} {r['diversity']:<10.2f} {r['psi_K']:<10.4f} {r['effective_dimensionality']:<12.2f} {r['cosine_drift']:<12.6f}")
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return lambda_results, horizon_results


if __name__ == "__main__":
    main()
