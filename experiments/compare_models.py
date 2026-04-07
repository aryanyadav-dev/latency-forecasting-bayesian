"""
Comprehensive evaluation script for comparing LFN with baselines (CPC, JEPA, Standard Transformer).

This script runs a standardized evaluation protocol that:
1. Evaluates language modeling performance (perplexity, accuracy)
2. Evaluates representation quality (LPS, downstream probing, CKA)
3. Evaluates latent predictability (forecasting MSE)
4. Generates comparison visualizations

Usage:
    python experiments/compare_models.py \
        --lfn-checkpoint path/to/lfn.pt \
        --cpc-checkpoint path/to/cpc.pt \
        --jepa-checkpoint path/to/jepa.pt \
        --baseline-checkpoint path/to/baseline.pt \
        --config experiments/configs/default.yaml \
        --output experiments/results/comparison
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from data.dataset_loader import create_dataloaders
from evaluation.downstream_eval import (
    DownstreamEvaluator,
    compute_representation_similarity,
)
from evaluation.metrics import Evaluator
from models.baseline_models import build_baseline_model
from models.complete_model import ModelConfig, build_model
from models.cpc_model import build_cpc_model
from models.jepa_model import build_jepa_model
from utils.seed import set_seed


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compare_models")


def load_model(model_type: str, checkpoint_path: str, config: Dict, device: str):
    """
    Load a model of specified type from checkpoint.

    Args:
        model_type: Type of model ('lfn', 'cpc', 'jepa', 'baseline')
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    model_cfg = config.get("model", {})
    vocab_size = model_cfg.get("vocab_size", 50257)
    latent_dim = model_cfg.get("latent_dim", 512)
    num_layers = model_cfg.get("num_layers", 6)
    num_heads = model_cfg.get("num_heads", 8)
    hidden_dim = model_cfg.get("hidden_dim", 2048)
    dropout = model_cfg.get("dropout", 0.1)
    max_context = model_cfg.get("max_context_length", 512)
    horizons = model_cfg.get("forecast_horizons", [1, 2, 5])

    # Build model
    if model_type == "lfn":
        model_config = ModelConfig(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            forecast_horizons=horizons,
            max_context_length=max_context,
            lambda_latent=config.get("training", {}).get("lambda_latent", 0.1),
        )
        model = build_model(model_config, device=device)
    elif model_type == "cpc":
        model = build_cpc_model(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context,
            prediction_horizons=tuple(horizons),
            device=device,
        )
    elif model_type == "jepa":
        model = build_jepa_model(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context,
            prediction_horizons=tuple(horizons),
            device=device,
        )
    elif model_type == "baseline":
        model = build_baseline_model(
            model_type="standard",
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    return model


def evaluate_model_comprehensive(
    model,
    model_type: str,
    test_loader,
    downstream_loaders: Optional[Dict] = None,
    baseline_model=None,
    device: str = "cuda",
) -> Dict:
    """
    Run comprehensive evaluation on a model.

    Args:
        model: Model to evaluate
        model_type: Type of model
        test_loader: Test data loader
        downstream_loaders: Dict with 'train', 'val', 'test' for downstream tasks
        baseline_model: Baseline model for CKA comparison
        device: Device for computation

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {"model_type": model_type}

    # 1. Language modeling metrics
    logger.info(f"Evaluating {model_type} language modeling...")
    evaluator = Evaluator(model, device=device, compute_accuracy=True)
    lm_metrics = evaluator.evaluate_model(
        test_loader, include_representation_analysis=True
    )
    results["language_modeling"] = {
        "perplexity": lm_metrics.get("perplexity"),
        "token_loss": lm_metrics.get("token_loss"),
        "accuracy": lm_metrics.get("accuracy"),
    }

    # 2. Representation quality metrics
    if "representation_metrics" in lm_metrics:
        results["representation_quality"] = lm_metrics["representation_metrics"]

    # 3. Latent forecasting metrics (if available)
    if "horizon_mse" in lm_metrics:
        results["latent_forecasting"] = {
            "horizon_mse": lm_metrics["horizon_mse"],
            "lps_scores": lm_metrics.get("lps_scores", {}),
        }

    # 4. Downstream evaluation
    if downstream_loaders is not None:
        logger.info(f"Evaluating {model_type} downstream tasks...")
        ds_evaluator = DownstreamEvaluator(model, device=device)

        # Check if downstream data is available
        if all(k in downstream_loaders for k in ["train", "val", "test"]):
            try:
                downstream_results = ds_evaluator.evaluate_classification(
                    downstream_loaders["train"],
                    downstream_loaders["val"],
                    downstream_loaders["test"],
                    num_classes=downstream_loaders.get("num_classes", 2),
                    task_name="downstream_classification",
                    num_epochs=10,
                )
                results["downstream"] = downstream_results
            except Exception as e:
                logger.warning(f"Downstream evaluation failed: {e}")

    # 5. Representation similarity to baseline (CKA)
    if baseline_model is not None:
        logger.info(f"Computing CKA similarity for {model_type}...")
        try:
            cka_score = compute_representation_similarity(
                model, baseline_model, test_loader, device=device, method="cka"
            )
            results["representation_similarity"] = {
                "cka_vs_baseline": cka_score,
            }
        except Exception as e:
            logger.warning(f"CKA computation failed: {e}")

    # 6. Effective dimensionality
    logger.info(f"Computing effective dimensionality for {model_type}...")
    try:
        from evaluation.downstream_eval import (
            compute_effective_dimensionality,
            extract_pooled_representations,
        )

        reps, _ = extract_pooled_representations(model, test_loader, device)
        eff_dim = compute_effective_dimensionality(reps)
        results["effective_dimensionality"] = eff_dim
    except Exception as e:
        logger.warning(f"Effective dimensionality computation failed: {e}")

    return results


def compare_all_models(
    model_paths: Dict[str, str],
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """
    Run comprehensive comparison of all models.

    Args:
        model_paths: Dict mapping model_type to checkpoint path
        config_path: Path to config file
        output_dir: Output directory for results
        device: Device for computation
        seed: Random seed

    Returns:
        Dictionary with all comparison results
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create data loaders
    logger.info("Creating data loaders...")
    data_cfg = config.get("data", {})
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=data_cfg.get("dataset_name", "wikitext-2-raw-v1"),
        tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
        context_length=data_cfg.get("context_length", 512),
        stride=data_cfg.get("stride", 256),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
    )

    # Evaluate each model
    all_results = {}
    baseline_model = None

    # Load baseline first for CKA comparison
    if "baseline" in model_paths:
        baseline_model = load_model("baseline", model_paths["baseline"], config, device)

    for model_type, checkpoint_path in model_paths.items():
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping {model_type}")
            continue

        try:
            model = load_model(model_type, checkpoint_path, config, device)

            results = evaluate_model_comprehensive(
                model=model,
                model_type=model_type,
                test_loader=test_loader,
                downstream_loaders=None,  # Can be added if downstream datasets available
                baseline_model=baseline_model,
                device=device,
            )

            all_results[model_type] = results

            # Save individual results
            with open(output_path / f"{model_type}_results.json", "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to evaluate {model_type}: {e}")
            all_results[model_type] = {"error": str(e)}

    # Generate comparison summary
    summary = generate_comparison_summary(all_results)

    with open(output_path / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate comparison table
    table = generate_comparison_table(all_results)
    with open(output_path / "comparison_table.txt", "w") as f:
        f.write(table)

    logger.info(f"Comparison complete. Results saved to {output_path}")
    return all_results


def generate_comparison_summary(results: Dict) -> Dict:
    """
    Generate summary statistics from comparison results.

    Args:
        results: Dict of model_type -> evaluation results

    Returns:
        Summary dictionary
    """
    summary = {}

    # Perplexity comparison
    perplexities = {}
    for model_type, res in results.items():
        if "language_modeling" in res and res["language_modeling"]:
            ppl = res["language_modeling"].get("perplexity")
            if ppl is not None:
                perplexities[model_type] = ppl

    if perplexities:
        summary["perplexity_comparison"] = {
            "best_model": min(perplexities, key=perplexities.get),
            "best_perplexity": min(perplexities.values()),
            "all_perplexities": perplexities,
        }

    # Downstream accuracy comparison
    accuracies = {}
    for model_type, res in results.items():
        if "downstream" in res and res["downstream"]:
            acc = res["downstream"].get("linear_probe_accuracy")
            if acc is not None:
                accuracies[model_type] = acc

    if accuracies:
        summary["downstream_comparison"] = {
            "best_model": max(accuracies, key=accuracies.get),
            "best_accuracy": max(accuracies.values()),
            "all_accuracies": accuracies,
        }

    # Representation quality comparison
    rep_metrics = {}
    for model_type, res in results.items():
        if "representation_quality" in res and res["representation_quality"]:
            rep_metrics[model_type] = res["representation_quality"]

    if rep_metrics:
        summary["representation_quality_comparison"] = rep_metrics

    # Effective dimensionality comparison
    eff_dims = {}
    for model_type, res in results.items():
        if "effective_dimensionality" in res:
            eff_dims[model_type] = res["effective_dimensionality"]

    if eff_dims:
        summary["effective_dimensionality_comparison"] = eff_dims

    return summary


def generate_comparison_table(results: Dict) -> str:
    """
    Generate formatted comparison table.

    Args:
        results: Dict of model_type -> evaluation results

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COMPARISON RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Header
    headers = ["Model", "PPL", "Accuracy", "Downstream Acc", "Eff. Dim", "CKA vs Base"]
    lines.append(" | ".join(f"{h:15}" for h in headers))
    lines.append("-" * 80)

    # Data rows
    for model_type, res in results.items():
        ppl = res.get("language_modeling", {}).get("perplexity", "N/A")
        acc = res.get("language_modeling", {}).get("accuracy", "N/A")
        ds_acc = res.get("downstream", {}).get("linear_probe_accuracy", "N/A")
        eff_dim = res.get("effective_dimensionality", "N/A")
        cka = res.get("representation_similarity", {}).get("cka_vs_baseline", "N/A")

        # Format values
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        ds_str = f"{ds_acc:.4f}" if isinstance(ds_acc, float) else str(ds_acc)
        dim_str = f"{eff_dim:.1f}" if isinstance(eff_dim, float) else str(eff_dim)
        cka_str = f"{cka:.4f}" if isinstance(cka, float) else str(cka)

        row = [model_type, ppl_str, acc_str, ds_str, dim_str, cka_str]
        lines.append(" | ".join(f"{v:15}" for v in row))

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare LFN with baseline models (CPC, JEPA, Standard Transformer)"
    )
    parser.add_argument(
        "--lfn-checkpoint",
        type=str,
        default=None,
        help="Path to LFN checkpoint",
    )
    parser.add_argument(
        "--cpc-checkpoint",
        type=str,
        default=None,
        help="Path to CPC checkpoint",
    )
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=None,
        help="Path to JEPA checkpoint",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=None,
        help="Path to baseline Transformer checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Collect model paths
    model_paths = {}
    if args.lfn_checkpoint:
        model_paths["lfn"] = args.lfn_checkpoint
    if args.cpc_checkpoint:
        model_paths["cpc"] = args.cpc_checkpoint
    if args.jepa_checkpoint:
        model_paths["jepa"] = args.jepa_checkpoint
    if args.baseline_checkpoint:
        model_paths["baseline"] = args.baseline_checkpoint

    if not model_paths:
        parser.error("At least one model checkpoint must be provided")

    # Run comparison
    results = compare_all_models(
        model_paths=model_paths,
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        seed=args.seed,
    )

    # Print summary
    with open(Path(args.output) / "comparison_table.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
