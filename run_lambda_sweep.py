#!/usr/bin/env python3
"""Crash-safe lambda sweep for paper experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict


FIXED_HORIZONS = [1, 2, 5, 10]
LAMBDA_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DEFAULT_SEED_START = 42


def lambda_file_slug(lambda_value: float) -> str:
    return str(lambda_value)


def result_payload(result: dict, lambda_value: float, seed: int) -> dict:
    return {
        "lambda_latent": lambda_value,
        "horizons": FIXED_HORIZONS,
        "seed": seed,
        "effective_dimensionality": result["effective_dimensionality"],
        "cosine_drift": result["cosine_drift"],
        "perplexity": result["perplexity"],
        "linear_probe_accuracy": result["linear_probe_accuracy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-horizon lambda sweep.")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output", type=str, default="experiments/results/lambda_sweep")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "wikitext-2", "ptb"])
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run starting at 42.")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Re-run completed result files.")
    parser.add_argument("--probe-train-limit", type=int, default=None)
    parser.add_argument("--probe-val-limit", type=int, default=None)
    parser.add_argument("--probe-test-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from data.dataset_loader import create_dataloaders
    from run_paper_horizon_rerun import (
        ROOT,
        RunConfig,
        build_probe_loaders,
        dataset_slug,
        normalize_dataset_name,
        resolve_device,
        run_single_with_loaders,
    )

    device = resolve_device(args.device)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = normalize_dataset_name(args.dataset)
    seeds = list(range(DEFAULT_SEED_START, DEFAULT_SEED_START + args.seeds))
    total = len(LAMBDA_VALUES) * len(seeds)

    metadata = {
        "device": device,
        "dataset": dataset_slug(dataset_name),
        "lambda_values": LAMBDA_VALUES,
        "horizons": FIXED_HORIZONS,
        "seeds": seeds,
        "num_epochs": args.num_epochs,
    }
    (output_dir / "run_metadata_lambda_sweep.json").write_text(json.dumps(metadata, indent=2, default=str))

    base_cfg = RunConfig(
        dataset_name=dataset_name,
        num_epochs=args.num_epochs,
        probe_train_limit=args.probe_train_limit,
        probe_val_limit=args.probe_val_limit,
        probe_test_limit=args.probe_test_limit,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=base_cfg.dataset_name,
        tokenizer_name="gpt2",
        context_length=base_cfg.max_context_length,
        stride=base_cfg.stride,
        batch_size=base_cfg.batch_size,
        num_workers=base_cfg.num_workers,
    )
    probe_train_loader, probe_val_loader, probe_test_loader = build_probe_loaders(base_cfg)

    run_idx = 0
    for lambda_value in LAMBDA_VALUES:
        for seed in seeds:
            run_idx += 1
            lambda_slug = lambda_file_slug(lambda_value)
            result_path = output_dir / f"results_lambda_{lambda_slug}_seed{seed}.json"
            print(f"RUN {run_idx}/{total} | λ={lambda_value} | seed={seed} | Starting…", flush=True)

            if result_path.exists() and not args.overwrite:
                existing = json.loads(result_path.read_text())
                print(
                    f"RUN {run_idx}/{total} | λ={lambda_value} | seed={seed} | "
                    f"Done ✓ | eff_dim={existing['effective_dimensionality']:.2f} | "
                    f"ppl={existing['perplexity']:.2f} | skipped existing",
                    flush=True,
                )
                continue

            cfg = RunConfig(
                **{
                    **asdict(base_cfg),
                    "lambda_latent": lambda_value,
                }
            )
            result = run_single_with_loaders(
                output_dir=output_dir,
                cfg=cfg,
                horizons=FIXED_HORIZONS,
                seed=seed,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                probe_train_loader=probe_train_loader,
                probe_val_loader=probe_val_loader,
                probe_test_loader=probe_test_loader,
            )
            payload = result_payload(result, lambda_value, seed)
            result_path.write_text(json.dumps(payload, indent=2, default=str))

            print(
                f"RUN {run_idx}/{total} | λ={lambda_value} | seed={seed} | "
                f"Done ✓ | eff_dim={payload['effective_dimensionality']:.2f} | "
                f"ppl={payload['perplexity']:.2f}",
                flush=True,
            )

    if total == 30:
        print("ALL 30 LAMBDA SWEEP RUNS COMPLETE", flush=True)
    else:
        print(f"ALL {total} LAMBDA SWEEP RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
