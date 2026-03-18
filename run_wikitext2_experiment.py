#!/usr/bin/env python3
"""
WikiText-2 Experiment Runner for arXiv Paper.

Trains both a Baseline Transformer and a Latent Forecasting Network (LFN)
on WikiText-2 and evaluates them. Produces real metrics: perplexity, token
accuracy, LPS scores, and representation quality metrics.

Usage:
    python run_wikitext2_experiment.py [--epochs 3] [--device auto]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment")


# ── Helpers ────────────────────────────────────────────────────────────────

def resolve_device(requested: str = "auto") -> str:
    """Pick the best available device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Model builders ─────────────────────────────────────────────────────────

def build_lfn(vocab_size: int, device: str):
    """Build the Latent Forecasting Network (Small)."""
    from models.complete_model import ModelConfig, LatentForecastingModel

    cfg = ModelConfig(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        forecast_horizons=[1, 2, 5],
        max_context_length=256,
        lambda_latent=0.1,
    )
    model = LatentForecastingModel(cfg).to(device)
    return model, cfg


def build_baseline(vocab_size: int, device: str):
    """Build the Baseline Transformer (identical encoder-decoder, no forecasting)."""
    from models.baseline_models import StandardTransformer

    model = StandardTransformer(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        max_context_length=256,
    ).to(device)
    return model


# ── Data ───────────────────────────────────────────────────────────────────

def create_loaders(device: str):
    """Create WikiText-2 data loaders."""
    from data.dataset_loader import create_dataloaders

    return create_dataloaders(
        dataset_name="wikitext-2",
        tokenizer_name="gpt2",
        context_length=256,
        stride=128,
        batch_size=8,
        num_workers=0,  # MPS-safe
    )


# ── Training loop ─────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, num_epochs=3,
                lr=3e-4, grad_accum=4, is_lfn=True, max_batches=500):
    """Train a model and return per-epoch metrics."""
    from training.optimizer import create_optimizer
    from training.scheduler import create_scheduler

    optimizer = create_optimizer(model, learning_rate=lr, weight_decay=0.01)
    batches_per_epoch = min(len(train_loader), max_batches)
    total_steps = num_epochs * batches_per_epoch // grad_accum
    scheduler = create_scheduler(optimizer, warmup_steps=100, num_training_steps=max(total_steps, 101))

    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    log_every = 50  # Print progress every N batches

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if is_lfn:
                output = model(tokens, labels=labels, compute_latent_loss=True)
            else:
                output = model(tokens, labels=labels)

            loss = output.total_loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += output.total_loss.item()
            num_batches += 1

            # Progress logging
            if (batch_idx + 1) % log_every == 0:
                avg_so_far = total_loss / num_batches
                elapsed = time.time() - t0
                logger.info(
                    f"  [Epoch {epoch+1}] batch {batch_idx+1}/{batches_per_epoch}  "
                    f"loss={avg_so_far:.4f}  elapsed={elapsed:.1f}s"
                )

        avg_train_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - t0
        history["train_loss"].append(avg_train_loss)
        history["epoch_times"].append(epoch_time)

        # ── Validate ──
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 50:  # Cap validation to 50 batches for speed
                    break
                tokens = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                if is_lfn:
                    out = model(tokens, labels=labels, compute_latent_loss=True)
                else:
                    out = model(tokens, labels=labels)
                val_total += out.total_loss.item()
                val_batches += 1
        avg_val_loss = val_total / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        logger.info(
            f"  Epoch {epoch+1}/{num_epochs} DONE  "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"time={epoch_time:.1f}s"
        )

    return history


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device, is_lfn=True):
    """Evaluate a trained model and return all metrics."""
    from evaluation.metrics import (
        compute_perplexity,
        compute_token_accuracy,
    )

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    # LPS accumulators (LFN only)
    lps_sums = {}
    mse_sums = {}
    latents_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 50:  # Cap evaluation to 50 batches for speed
                break
                
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if is_lfn:
                output = model(tokens, labels=labels, compute_latent_loss=True)
            else:
                output = model(tokens, labels=labels)

            total_loss += output.token_loss.item()
            acc = compute_token_accuracy(output.logits, labels)
            total_acc += acc.item()
            num_batches += 1

            # Collect latents for representation analysis
            latents = output.latents.detach().cpu()
            latents_list.append(latents)

            # LPS / MSE per horizon (LFN only)
            if is_lfn and hasattr(output, "predicted_latents") and output.predicted_latents:
                for k, preds in output.predicted_latents.items():
                    targets = output.latents[:, k:, :]
                    diff = targets - preds
                    l2 = torch.norm(diff, p=2, dim=-1).mean().item()
                    mse = torch.nn.functional.mse_loss(preds, targets).item()
                    lps_sums[k] = lps_sums.get(k, 0.0) + l2
                    mse_sums[k] = mse_sums.get(k, 0.0) + mse

    avg_loss = total_loss / max(num_batches, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    avg_acc = total_acc / max(num_batches, 1)

    results = {
        "perplexity": round(ppl, 2),
        "token_loss": round(avg_loss, 4),
        "token_accuracy": round(avg_acc, 4),
    }

    # LPS scores
    if lps_sums:
        results["lps_scores"] = {str(k): round(v / num_batches, 4) for k, v in sorted(lps_sums.items())}
        results["horizon_mse"] = {str(k): round(v / num_batches, 6) for k, v in sorted(mse_sums.items())}

    # Representation metrics
    all_latents = torch.cat(latents_list, dim=0)  # [N, seq, dim]
    flat = all_latents.reshape(-1, all_latents.shape[-1]).numpy()
    # Take a subsample if too large
    if flat.shape[0] > 20000:
        idx = np.random.choice(flat.shape[0], 20000, replace=False)
        flat = flat[idx]

    results["representation_metrics"] = compute_repr_metrics(flat, all_latents)

    return results


def compute_repr_metrics(flat_latents: np.ndarray, seq_latents: torch.Tensor) -> dict:
    """Compute representation quality metrics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    metrics = {}

    # Latent variance (average per-dimension variance)
    var = np.var(flat_latents, axis=0).mean()
    metrics["latent_variance"] = round(float(var), 6)

    # Latent entropy (discretize activations, compute Shannon entropy)
    bins = 50
    hist_sum = 0.0
    for d in range(flat_latents.shape[1]):
        counts, _ = np.histogram(flat_latents[:, d], bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        hist_sum += -np.sum(probs * np.log(probs))
    metrics["latent_entropy"] = round(float(hist_sum / flat_latents.shape[1]), 4)

    # Cosine similarity drift (average across consecutive time steps)
    # Use first 500 sequences to keep fast
    seqs = seq_latents[:500]  # [N, T, D]
    a = seqs[:, :-1, :]
    b = seqs[:, 1:, :]
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    drift = 1.0 - cos_sim.mean().item()
    metrics["cosine_similarity_drift"] = round(drift, 6)

    # Cluster separability (silhouette score with k=10)
    n_clusters = min(10, flat_latents.shape[0] // 2)
    if n_clusters >= 2:
        try:
            km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
            labels = km.fit_predict(flat_latents[:5000])
            sil = silhouette_score(flat_latents[:5000], labels, sample_size=2000)
            metrics["cluster_separability"] = round(float(sil), 4)
        except Exception:
            metrics["cluster_separability"] = None
    else:
        metrics["cluster_separability"] = None

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WikiText-2 Experiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="experiments/results/wikitext2_real")
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    # Seed
    from utils.seed import set_seed
    set_seed(42)

    # Data
    logger.info("Loading WikiText-2 data...")
    train_loader, val_loader, test_loader = create_loaders(device)
    logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    vocab_size = 50257  # GPT-2

    all_results = {}

    # ── 1. Baseline Transformer ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training Baseline Transformer (no latent forecasting)")
    logger.info("=" * 60)

    baseline = build_baseline(vocab_size, device)
    logger.info(f"Baseline parameters: {count_parameters(baseline):,}")

    baseline_history = train_model(
        baseline, train_loader, val_loader, device,
        num_epochs=args.epochs, is_lfn=False,
    )
    baseline_results = evaluate_model(baseline, test_loader, device, is_lfn=False)
    baseline_results["num_parameters"] = count_parameters(baseline)
    baseline_results["training_history"] = baseline_history
    all_results["baseline_transformer"] = baseline_results

    logger.info(f"Baseline  PPL={baseline_results['perplexity']:.2f}  "
                f"Acc={baseline_results['token_accuracy']:.4f}")

    # Free memory
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 2. Latent Forecasting Network ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training Latent Forecasting Network (lambda=0.1)")
    logger.info("=" * 60)

    set_seed(42)
    lfn, lfn_cfg = build_lfn(vocab_size, device)
    logger.info(f"LFN parameters: {count_parameters(lfn):,}")

    lfn_history = train_model(
        lfn, train_loader, val_loader, device,
        num_epochs=args.epochs, is_lfn=True,
    )
    lfn_results = evaluate_model(lfn, test_loader, device, is_lfn=True)
    lfn_results["num_parameters"] = count_parameters(lfn)
    lfn_results["training_history"] = lfn_history
    all_results["latent_forecasting_network"] = lfn_results

    logger.info(f"LFN       PPL={lfn_results['perplexity']:.2f}  "
                f"Acc={lfn_results['token_accuracy']:.4f}")
    if "lps_scores" in lfn_results:
        logger.info(f"LPS scores: {lfn_results['lps_scores']}")

    # ── Save results ───────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Add experiment metadata
    all_results["metadata"] = {
        "device": device,
        "epochs": args.epochs,
        "dataset": "WikiText-2",
        "context_length": 256,
        "batch_size": 8,
        "grad_accum": 4,
        "effective_batch_size": 32,
        "seed": 42,
    }

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("  WikiText-2 Experiment Results (Small Model)")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Baseline':>14} {'LFN':>14}")
    print("-" * 65)
    print(f"  {'Parameters':.<30} {baseline_results['num_parameters']:>14,} {lfn_results['num_parameters']:>14,}")
    print(f"  {'Perplexity':.<30} {baseline_results['perplexity']:>14.2f} {lfn_results['perplexity']:>14.2f}")
    print(f"  {'Token Accuracy':.<30} {baseline_results['token_accuracy']:>14.4f} {lfn_results['token_accuracy']:>14.4f}")
    print(f"  {'Token Loss':.<30} {baseline_results['token_loss']:>14.4f} {lfn_results['token_loss']:>14.4f}")

    if "lps_scores" in lfn_results:
        print()
        print("  LFN Latent Predictability Scores:")
        for k, v in lfn_results["lps_scores"].items():
            print(f"    Horizon k={k}: LPS={v:.4f}")

    for name, res in [("Baseline", baseline_results), ("LFN", lfn_results)]:
        rm = res.get("representation_metrics", {})
        if rm:
            print(f"\n  {name} Representation Metrics:")
            for k, v in rm.items():
                if v is not None:
                    print(f"    {k}: {v}")

    print("=" * 65)


if __name__ == "__main__":
    main()
