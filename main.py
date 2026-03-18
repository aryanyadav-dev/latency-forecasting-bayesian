#!/usr/bin/env python3
"""
CLI entry point for Latent Forecasting Network.

Subcommands:
    train       Train a model from a YAML config
    evaluate    Evaluate a trained model checkpoint
    analyze     Run representation analysis and generate visualizations
    ablation    Run ablation study from ablation config
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load and return a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(requested: Optional[str]) -> str:
    """Return a valid torch device string."""
    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning(
            "CUDA requested but not available, falling back to CPU"
        )
        return "cpu"
    return requested


def _build_model_from_config(cfg: Dict[str, Any], device: str):
    """Construct an LFN model from a config dict."""
    from models.complete_model import ModelConfig, build_model

    model_cfg = cfg.get("model", {})
    config = ModelConfig(
        vocab_size=model_cfg.get("vocab_size", 50257),
        latent_dim=model_cfg.get("latent_dim", 512),
        num_layers=model_cfg.get("num_layers", 6),
        num_heads=model_cfg.get("num_heads", 8),
        hidden_dim=model_cfg.get("hidden_dim", 2048),
        dropout=model_cfg.get("dropout", 0.1),
        forecast_horizons=model_cfg.get("forecast_horizons", [1, 2, 5]),
        max_context_length=model_cfg.get("max_context_length", 512),
        lambda_latent=cfg.get("training", {}).get("lambda_latent", 0.1),
    )
    return build_model(config, device=device), config


def _create_dataloaders(cfg: Dict[str, Any], device: str):
    """Create train / val / test dataloaders from config."""
    from data.dataset_loader import create_dataloaders

    data_cfg = cfg.get("data", {})
    return create_dataloaders(
        dataset_name=data_cfg.get("dataset_name", "wikitext-2-raw-v1"),
        tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
        context_length=data_cfg.get("context_length", 512),
        stride=data_cfg.get("stride", 256),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
    )


def _load_checkpoint(checkpoint_path: str, device: str):
    """Load and return a checkpoint dict."""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def train_command(args: argparse.Namespace) -> int:
    """Run model training."""
    logger = logging.getLogger("train")
    cfg = _load_config(args.config)
    device = _resolve_device(args.device)

    # Seed
    from utils.seed import set_seed

    seed = args.seed or cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    logger.info("Building model …")
    model, model_cfg = _build_model_from_config(cfg, device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    logger.info("Creating dataloaders …")
    train_loader, val_loader, test_loader = _create_dataloaders(cfg, device)

    # Build optimizer & scheduler
    from training.optimizer import create_optimizer
    from training.scheduler import create_scheduler
    from training.trainer import Trainer, TrainingConfig

    train_cfg = cfg.get("training", {})
    training_config = TrainingConfig(
        num_epochs=train_cfg.get("num_epochs", 10),
        batch_size=train_cfg.get("batch_size", 32),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        lambda_latent=train_cfg.get("lambda_latent", 0.1),
        use_mixed_precision=train_cfg.get("use_mixed_precision", True),
        checkpoint_every=train_cfg.get("checkpoint_every", 1000),
        log_every=train_cfg.get("log_every", 100),
        seed=seed,
        device=device,
    )

    optimizer = create_optimizer(
        model,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    total_steps = (
        training_config.num_epochs
        * len(train_loader)
        // training_config.gradient_accumulation_steps
    )
    scheduler = create_scheduler(
        optimizer, warmup_steps=training_config.warmup_steps, total_steps=total_steps
    )

    # Checkpoint dirs
    output = Path(
        args.output
        or cfg.get("experiment", {}).get("output_dir", "experiments/results")
    )
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training …")
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed:.1f}s")

    # Generate training visualizations
    try:
        from evaluation.visualization import Visualizer

        viz = Visualizer(str(output / "plots"))
        viz.generate_all_visualizations(training_history=history)
        logger.info("Training visualizations saved.")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")

    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Evaluate a trained model."""
    logger = logging.getLogger("evaluate")
    cfg = _load_config(args.config)
    device = _resolve_device(args.device)

    logger.info("Building model …")
    model, model_cfg = _build_model_from_config(cfg, device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state_dict"])

    logger.info("Creating test dataloader …")
    _, _, test_loader = _create_dataloaders(cfg, device)

    from evaluation.metrics import Evaluator

    evaluator = Evaluator(model, device=device, compute_accuracy=True)
    logger.info("Running evaluation …")
    results = evaluator.evaluate_model(
        test_loader,
        include_representation_analysis=args.representation_analysis,
    )

    # Print results
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2:.6f}")
        else:
            print(
                f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}"
            )
    print("=" * 50 + "\n")

    # Save results JSON
    output = Path(args.output or "experiments/results")
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "evaluation_results.json"

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {str(k2): float(v2) for k2, v2 in v.items()}
        elif isinstance(v, float):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return 0


def analyze_command(args: argparse.Namespace) -> int:
    """Run representation analysis and generate visualizations."""
    logger = logging.getLogger("analyze")
    cfg = _load_config(args.config)
    device = _resolve_device(args.device)

    logger.info("Building model …")
    model, _ = _build_model_from_config(cfg, device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = _load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, test_loader = _create_dataloaders(cfg, device)

    from evaluation.metrics import Evaluator

    evaluator = Evaluator(model, device=device)

    # Full evaluation
    logger.info("Computing all metrics …")
    results = evaluator.evaluate_model(
        test_loader,
        include_representation_analysis=True,
    )

    # Collect latents for visualization
    latents_list = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 20:  # Limit batches for viz
                break
            tokens = batch["input_ids"].to(device)
            output = model(tokens, compute_latent_loss=False)
            latents_list.append(output.latents.cpu().numpy())
    all_latents = np.concatenate(latents_list, axis=0)

    # Generate visualizations
    if args.visualize:
        from evaluation.visualization import Visualizer

        output_dir = Path(args.output or "experiments/results") / "analysis"
        viz = Visualizer(str(output_dir))
        viz.generate_all_visualizations(
            evaluation_results=results,
            latents=all_latents,
        )
        logger.info(f"Visualizations saved to {output_dir}")

    # Save analysis results
    output_base = Path(args.output or "experiments/results")
    output_base.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {str(k2): float(v2) for k2, v2 in v.items()}
        elif isinstance(v, (int, float)):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(output_base / "analysis_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Analysis complete.")
    return 0


def ablation_command(args: argparse.Namespace) -> int:
    """Run ablation study from ablation configuration."""
    logger = logging.getLogger("ablation")
    ablation_cfg = _load_config(args.config)
    device = _resolve_device(args.device)

    experiments = ablation_cfg.get("experiments", [])
    if not experiments:
        logger.error("No experiments found in ablation config.")
        return 1

    base_config_path = ablation_cfg.get("ablation_study", {}).get("base_config")
    base_cfg = _load_config(base_config_path) if base_config_path else {}

    output_base = Path(
        args.output
        or ablation_cfg.get("ablation_study", {}).get(
            "output_dir", "experiments/results/ablation"
        )
    )
    output_base.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Any]] = {}

    for exp in experiments:
        exp_name = exp.get("name", "unnamed")
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation experiment: {exp_name}")
        logger.info(f"Description: {exp.get('description', '')}")
        logger.info(f"{'='*60}")

        # Merge overrides with base config
        import copy

        merged = copy.deepcopy(base_cfg)
        overrides = exp.get("overrides", {})
        for section, values in overrides.items():
            if section not in merged:
                merged[section] = {}
            merged[section].update(values)

        # Write merged config for this experiment
        exp_dir = output_base / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        merged_path = exp_dir / "config.yaml"
        with open(merged_path, "w") as f:
            yaml.dump(merged, f, default_flow_style=False)

        try:
            from utils.seed import set_seed

            set_seed(merged.get("training", {}).get("seed", 42))

            model, _ = _build_model_from_config(merged, device)
            train_loader, val_loader, test_loader = _create_dataloaders(merged, device)

            from training.optimizer import create_optimizer
            from training.scheduler import create_scheduler
            from training.trainer import Trainer, TrainingConfig

            t = merged.get("training", {})
            tc = TrainingConfig(
                num_epochs=t.get("num_epochs", 10),
                batch_size=t.get("batch_size", 32),
                learning_rate=t.get("learning_rate", 1e-4),
                weight_decay=t.get("weight_decay", 0.01),
                gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
                max_grad_norm=t.get("max_grad_norm", 1.0),
                warmup_steps=t.get("warmup_steps", 1000),
                lambda_latent=t.get("lambda_latent", 0.1),
                use_mixed_precision=t.get("use_mixed_precision", True),
                checkpoint_every=t.get("checkpoint_every", 1000),
                log_every=t.get("log_every", 100),
                seed=t.get("seed", 42),
                device=device,
            )

            opt = create_optimizer(
                model, lr=tc.learning_rate, weight_decay=tc.weight_decay
            )
            total_steps = (
                tc.num_epochs * len(train_loader) // tc.gradient_accumulation_steps
            )
            sched = create_scheduler(
                opt, warmup_steps=tc.warmup_steps, total_steps=total_steps
            )

            trainer = Trainer(
                model=model,
                optimizer=opt,
                scheduler=sched,
                train_loader=train_loader,
                val_loader=val_loader,
                config=tc,
                checkpoint_dir=str(exp_dir / "checkpoints"),
                log_dir=str(exp_dir / "logs"),
            )

            history = trainer.train()

            # Evaluate
            from evaluation.metrics import Evaluator

            evaluator = Evaluator(model, device=device)
            exp_results = evaluator.evaluate_model(test_loader)

            all_results[exp_name] = {
                k: (v if isinstance(v, (int, float)) else str(v))
                for k, v in exp_results.items()
            }

            # Save per-experiment results
            with open(exp_dir / "results.json", "w") as f:
                json.dump(all_results[exp_name], f, indent=2)

            logger.info(
                f"Experiment {exp_name} complete: perplexity={exp_results.get('perplexity', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            all_results[exp_name] = {"error": str(e)}

    # Save summary
    with open(output_base / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate comparison plots
    try:
        from evaluation.visualization import Visualizer

        viz = Visualizer(str(output_base / "plots"))
        # Filter out errored experiments
        valid = {k: v for k, v in all_results.items() if "error" not in v}
        if valid:
            for metric in ["perplexity", "token_loss"]:
                if any(metric in v for v in valid.values()):
                    viz.plot_ablation_comparison(valid, metric=metric)
        logger.info("Ablation comparison plots saved.")
    except Exception as e:
        logger.warning(f"Could not generate comparison plots: {e}")

    logger.info(
        f"\nAblation study complete. Summary saved to {output_base / 'ablation_summary.json'}"
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfn",
        description="Latent Forecasting Network – CLI Interface",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda or cpu (auto-detect if omitted)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- train ----
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_train.add_argument("--output", "-o", default=None, help="Output directory")
    p_train.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume from"
    )

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_eval.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p_eval.add_argument(
        "--output", "-o", default=None, help="Output directory for results"
    )
    p_eval.add_argument(
        "--representation-analysis",
        action="store_true",
        help="Include representation analysis",
    )

    # ---- analyze ----
    p_analyze = sub.add_parser(
        "analyze", help="Representation analysis + visualizations"
    )
    p_analyze.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_analyze.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    p_analyze.add_argument("--output", "-o", default=None, help="Output directory")
    p_analyze.add_argument(
        "--visualize", action="store_true", default=True, help="Generate visualizations"
    )
    p_analyze.add_argument("--no-visualize", dest="visualize", action="store_false")

    # ---- ablation ----
    p_abl = sub.add_parser("ablation", help="Run ablation study")
    p_abl.add_argument(
        "--config", "-c", required=True, help="Path to ablation YAML config"
    )
    p_abl.add_argument("--output", "-o", default=None, help="Output directory")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    _setup_logging(args.verbose)

    dispatch = {
        "train": train_command,
        "evaluate": evaluate_command,
        "analyze": analyze_command,
        "ablation": ablation_command,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
