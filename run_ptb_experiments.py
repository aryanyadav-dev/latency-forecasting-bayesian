#!/usr/bin/env python3
"""Crash-safe PTB horizon sweep for paper experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from data.dataset_loader import create_dataloaders
from run_paper_horizon_rerun import (
    ROOT,
    RunConfig,
    aggregate_metrics,
    build_probe_loaders,
    dataset_slug,
    format_horizons,
    resolve_device,
    run_single_with_loaders,
)


PTB_HORIZON_CONFIGS = [[1], [1, 2, 5], [1, 2, 5, 10]]
PTB_SEEDS = [1, 2, 3, 4, 5]


def horizon_file_slug(horizons: list[int]) -> str:
    return "K" + "_".join(str(k) for k in horizons)


def write_aggregate(output_dir: Path, cfg: RunConfig, completed_files: list[Path]) -> None:
    aggregate = {
        "metadata": {
            "dataset": dataset_slug(cfg.dataset_name),
            "run_config": asdict(cfg),
            "horizon_configs": PTB_HORIZON_CONFIGS,
            "seeds": PTB_SEEDS,
        },
        "results": {},
    }

    for horizons in PTB_HORIZON_CONFIGS:
        records = []
        for seed in PTB_SEEDS:
            path = output_dir / f"results_ptb_{horizon_file_slug(horizons)}_seed{seed}.json"
            if path.exists():
                records.append(json.loads(path.read_text()))
        if records:
            aggregate["results"][format_horizons(horizons)] = aggregate_metrics(records)

    aggregate["metadata"]["completed_files"] = [str(path.name) for path in completed_files]
    (output_dir / "aggregate_ptb.json").write_text(json.dumps(aggregate, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PTB horizon experiments.")
    parser.add_argument("--output", type=str, default="experiments/results/ptb_horizon_rerun")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Re-run completed result files.")
    parser.add_argument("--probe-train-limit", type=int, default=None)
    parser.add_argument("--probe-val-limit", type=int, default=None)
    parser.add_argument("--probe-test-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        dataset_name="ptb",
        num_epochs=args.num_epochs,
        lambda_latent=0.1,
        probe_train_limit=args.probe_train_limit,
        probe_val_limit=args.probe_val_limit,
        probe_test_limit=args.probe_test_limit,
    )

    metadata = {
        "device": device,
        "dataset": "ptb",
        "horizon_configs": PTB_HORIZON_CONFIGS,
        "seeds": PTB_SEEDS,
        "run_config": asdict(cfg),
    }
    (output_dir / "run_metadata_ptb.json").write_text(json.dumps(metadata, indent=2))

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=cfg.dataset_name,
        tokenizer_name="gpt2",
        context_length=cfg.max_context_length,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    probe_train_loader, probe_val_loader, probe_test_loader = build_probe_loaders(cfg)

    completed_files = sorted(output_dir.glob("results_ptb_K*_seed*.json"))
    total = len(PTB_HORIZON_CONFIGS) * len(PTB_SEEDS)
    run_idx = 0

    for horizons in PTB_HORIZON_CONFIGS:
        for seed in PTB_SEEDS:
            run_idx += 1
            result_path = output_dir / f"results_ptb_{horizon_file_slug(horizons)}_seed{seed}.json"
            print(f"RUN {run_idx}/{total} | K={format_horizons(horizons)} | seed={seed}", flush=True)

            if result_path.exists() and not args.overwrite:
                print(f"  skipping existing {result_path.name}", flush=True)
                continue

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
            result["dataset"] = "ptb"
            result["horizon_set_K"] = horizons
            result["seed"] = seed
            result_path.write_text(json.dumps(result, indent=2))
            completed_files = sorted(output_dir.glob("results_ptb_K*_seed*.json"))
            write_aggregate(output_dir, cfg, completed_files)

            print(
                "  saved "
                f"{result_path.name} | eff_dim={result['eff_dim']:.2f} | "
                f"cka_drift={result['cka_drift']:.4f} | "
                f"probe={result['linear_probe_accuracy']:.4f} | "
                f"ppl={result['perplexity']:.2f}",
                flush=True,
            )

    write_aggregate(output_dir, cfg, completed_files)
    print(f"Done. Aggregate saved to {output_dir / 'aggregate_ptb.json'}", flush=True)


if __name__ == "__main__":
    main()
