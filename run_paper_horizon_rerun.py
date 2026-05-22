#!/usr/bin/env python3
"""
Run a paper-grade 5-seed horizon-ablation sweep with a proper frozen SST-2 probe.

This script is designed to be resumable. It saves per-seed JSON outputs and
refreshes aggregate statistics after each completed seed/config pair.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "experiments" / ".cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
for path in (CACHE_DIR, HF_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from data.dataset_loader import create_dataloaders
from evaluation.downstream_eval import (
    compute_effective_dimensionality,
    evaluate_linear_probe,
    train_linear_probe,
)
from evaluation.metrics import Evaluator
from models.complete_model import ModelConfig
from run_theoretical_validation import extract_all_representations, train_model
from training.trainer import TrainingConfig
from utils.seed import set_seed


HORIZON_CONFIGS = [
    [1],
    [2],
    [5],
    [10],
    [1, 2],
    [1, 2, 5],
    [1, 2, 5, 10],
    [1, 3, 5, 10, 20],
]


DATASET_ALIASES = {
    "wikitext": "wikitext-2",
    "wikitext-2": "wikitext-2",
    "ptb": "ptb",
    "ptb_text_only": "ptb",
}


def normalize_dataset_name(dataset_name: str) -> str:
    try:
        return DATASET_ALIASES[dataset_name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}") from exc


def dataset_slug(dataset_name: str) -> str:
    return "wikitext" if dataset_name == "wikitext-2" else dataset_name.replace("-", "_")


@dataclass
class RunConfig:
    dataset_name: str = "wikitext-2"
    latent_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    hidden_dim: int = 1024
    dropout: float = 0.1
    max_context_length: int = 256
    stride: int = 128
    batch_size: int = 8
    num_workers: int = 0
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    lambda_latent: float = 0.1
    probe_task: str = "glue/sst2"
    probe_max_length: int = 128
    probe_batch_size: int = 32
    probe_num_epochs: int = 20
    probe_learning_rate: float = 1e-3
    probe_pooling: str = "mean"
    probe_train_limit: int | None = None
    probe_val_limit: int | None = None
    probe_test_limit: int | None = None


class TextClassificationDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_horizons(horizons: Iterable[int]) -> str:
    return "{" + ",".join(map(str, horizons)) + "}"


def build_probe_loaders(cfg: RunConfig, tokenizer_name: str = "gpt2") -> tuple[DataLoader, DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("glue", "sst2")
    train_ds = ds["train"]
    val_ds = ds["validation"]

    train_size = int(0.9 * len(train_ds))
    split_generator = torch.Generator().manual_seed(1234)
    train_subset, val_subset = random_split(train_ds, [train_size, len(train_ds) - train_size], generator=split_generator)

    def encode_split(examples, limit: int | None) -> TextClassificationDataset:
        sentences = [examples[i]["sentence"] for i in range(len(examples))]
        labels = [examples[i]["label"] for i in range(len(examples))]
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        encoded = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=cfg.probe_max_length,
            return_tensors="pt",
        )
        return TextClassificationDataset(encoded["input_ids"], torch.tensor(labels, dtype=torch.long))

    train_dataset = encode_split(train_subset, cfg.probe_train_limit)
    val_dataset = encode_split(val_subset, cfg.probe_val_limit)
    test_dataset = encode_split(val_ds, cfg.probe_test_limit)

    loader_kwargs = {
        "batch_size": cfg.probe_batch_size,
        "num_workers": 0,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def compute_diversity(horizons: list[int]) -> float:
    return len(horizons) / max(horizons)


def compute_psi(horizons: list[int], alpha: float = 0.5) -> float:
    return float(sum(1.0 / (1.0 + alpha * k) for k in horizons))


def compute_temporal_cka_drift(sequence_reps: torch.Tensor, max_points: int = 5000) -> float:
    """Return 1 - linear CKA between adjacent latent time steps."""
    if sequence_reps.ndim != 3 or sequence_reps.size(1) < 2:
        return 0.0
    x = sequence_reps[:, :-1, :].reshape(-1, sequence_reps.size(-1)).float()
    y = sequence_reps[:, 1:, :].reshape(-1, sequence_reps.size(-1)).float()
    if x.size(0) > max_points:
        x = x[:max_points]
        y = y[:max_points]
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(x.T @ y, ord="fro") ** 2
    denominator = (
        torch.linalg.matrix_norm(x.T @ x, ord="fro")
        * torch.linalg.matrix_norm(y.T @ y, ord="fro")
    )
    if denominator.item() <= 1e-12:
        return 0.0
    return float(1.0 - (numerator / denominator).clamp(0.0, 1.0).item())


def model_and_training_config(cfg: RunConfig, horizons: list[int], device: str, seed: int) -> tuple[ModelConfig, TrainingConfig]:
    model_cfg = ModelConfig(
        vocab_size=50257,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        forecast_horizons=horizons,
        max_context_length=cfg.max_context_length,
        lambda_latent=cfg.lambda_latent,
    )
    train_cfg = TrainingConfig(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=cfg.warmup_steps,
        lambda_latent=cfg.lambda_latent,
        use_mixed_precision=False,
        checkpoint_every=500,
        log_every=10,
        seed=seed,
        device=device,
    )
    return model_cfg, train_cfg


def compute_probe_accuracy(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    pooling: str,
    num_epochs: int,
    learning_rate: float,
) -> float:
    probe, _ = train_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=2,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        pooling=pooling,
    )
    metrics = evaluate_linear_probe(
        model=model,
        probe=probe,
        test_loader=test_loader,
        device=device,
        pooling=pooling,
    )
    return float(metrics["accuracy"])


def evaluate_horizon_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    probe_test_loader: DataLoader,
    horizons: list[int],
    cfg: RunConfig,
    device: str,
) -> dict:
    reps, sequence_reps = extract_all_representations(model, test_loader, device)
    evaluator = Evaluator(model, device=device, compute_accuracy=True)
    eval_results = evaluator.evaluate_model(test_loader, include_representation_analysis=True)
    probe_acc = compute_probe_accuracy(
        model=model,
        train_loader=probe_train_loader,
        val_loader=probe_val_loader,
        test_loader=probe_test_loader,
        device=device,
        pooling=cfg.probe_pooling,
        num_epochs=cfg.probe_num_epochs,
        learning_rate=cfg.probe_learning_rate,
    )
    cosine_drift = float(
        eval_results.get("representation_metrics", {}).get("cosine_similarity_drift", 0.0)
    )
    cka_drift = compute_temporal_cka_drift(sequence_reps)
    return {
        "dataset": dataset_slug(cfg.dataset_name),
        "horizon_set_K": horizons,
        "horizons": horizons,
        "diversity": compute_diversity(horizons),
        "psi_K": compute_psi(horizons),
        "eff_dim": float(compute_effective_dimensionality(reps)),
        "effective_dimensionality": float(compute_effective_dimensionality(reps)),
        "cka_drift": cka_drift,
        "cosine_drift": cosine_drift,
        "perplexity": float(eval_results.get("perplexity", 0.0)),
        "linear_probe_accuracy": probe_acc,
    }


def aggregate_metrics(records: list[dict]) -> dict:
    out: dict[str, dict] = {}
    metric_keys = ["effective_dimensionality", "cosine_drift", "perplexity", "linear_probe_accuracy"]
    for key in metric_keys:
        vals = np.array([r[key] for r in records], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    out["diversity"] = records[0]["diversity"]
    out["psi_K"] = records[0]["psi_K"]
    out["horizons"] = records[0]["horizons"]
    return out


def refresh_aggregate(output_dir: Path, cfg: RunConfig, seeds: list[int]) -> None:
    aggregate: dict[str, dict] = {
        "metadata": {
            "run_config": asdict(cfg),
            "completed_seeds": seeds,
        },
        "results": {},
    }
    for horizons in HORIZON_CONFIGS:
        key = format_horizons(horizons)
        records = []
        for seed in seeds:
            seed_path = output_dir / "per_seed" / key / f"seed_{seed}.json"
            if seed_path.exists():
                records.append(json.loads(seed_path.read_text()))
        if records:
            aggregate["results"][key] = aggregate_metrics(records)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))


def run_single(output_dir: Path, cfg: RunConfig, horizons: list[int], seed: int, device: str) -> dict:
    set_seed(seed)
    raise RuntimeError("run_single should not be called directly; use run_single_with_loaders")


def run_single_with_loaders(
    output_dir: Path,
    cfg: RunConfig,
    horizons: list[int],
    seed: int,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    probe_test_loader: DataLoader,
) -> dict:
    set_seed(seed)
    model_cfg, train_cfg = model_and_training_config(cfg, horizons, device, seed)
    checkpoint_dir = output_dir / "checkpoints" / format_horizons(horizons) / f"seed_{seed}"
    model, history = train_model(
        model_cfg,
        train_loader,
        val_loader,
        train_cfg,
        device,
        str(checkpoint_dir),
    )
    result = evaluate_horizon_model(
        model=model,
        test_loader=test_loader,
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader,
        probe_test_loader=probe_test_loader,
        horizons=horizons,
        cfg=cfg,
        device=device,
    )
    result["seed"] = seed
    result["training_history"] = history
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade 5-seed horizon rerun.")
    parser.add_argument("--output", type=str, default="experiments/results/paper_horizon_rerun")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-horizons", nargs="*", default=None, help="Subset like 1,2,5 1,3,5,10,20")
    parser.add_argument("--probe-train-limit", type=int, default=None)
    parser.add_argument("--probe-val-limit", type=int, default=None)
    parser.add_argument("--probe-test-limit", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "ptb"])
    return parser.parse_args()


def parse_only_horizons(values: list[str] | None) -> list[list[int]] | None:
    if not values:
        return None
    parsed = []
    for value in values:
        parsed.append([int(x) for x in value.split(",") if x])
    return parsed


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        dataset_name=normalize_dataset_name(args.dataset),
        num_epochs=args.num_epochs,
        probe_train_limit=args.probe_train_limit,
        probe_val_limit=args.probe_val_limit,
        probe_test_limit=args.probe_test_limit,
    )
    device = resolve_device(args.device)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    horizon_configs = parse_only_horizons(args.only_horizons) or HORIZON_CONFIGS

    metadata = {
        "device": device,
        "seeds": seeds,
        "run_config": asdict(cfg),
        "horizon_configs": horizon_configs,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=cfg.dataset_name,
        tokenizer_name="gpt2",
        context_length=cfg.max_context_length,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    probe_train_loader, probe_val_loader, probe_test_loader = build_probe_loaders(cfg)

    for horizons in horizon_configs:
        key = format_horizons(horizons)
        horizon_dir = output_dir / "per_seed" / key
        horizon_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            seed_path = horizon_dir / f"seed_{seed}.json"
            if args.resume and seed_path.exists():
                continue

            print(f"[run] horizons={horizons} seed={seed} device={device}", flush=True)
            result = run_single_with_loaders(
                output_dir=output_dir,
                cfg=cfg,
                horizons=horizons,
                seed=seed,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                probe_train_loader=probe_train_loader,
                probe_val_loader=probe_val_loader,
                probe_test_loader=probe_test_loader,
            )
            seed_path.write_text(json.dumps(result, indent=2))
            refresh_aggregate(output_dir, cfg, seeds)
            print(
                json.dumps(
                    {
                        "horizons": horizons,
                        "seed": seed,
                        "effective_dimensionality": result["effective_dimensionality"],
                        "cosine_drift": result["cosine_drift"],
                        "perplexity": result["perplexity"],
                        "linear_probe_accuracy": result["linear_probe_accuracy"],
                    },
                    indent=2,
                ),
                flush=True,
            )

    refresh_aggregate(output_dir, cfg, seeds)
    print(f"[done] wrote aggregate to {output_dir / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
