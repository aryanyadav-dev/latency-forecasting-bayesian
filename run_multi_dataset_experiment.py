#!/usr/bin/env python3
"""
Multi-Dataset Experiment Runner.

Runs LFN and baseline experiments on multiple datasets (WikiText-2, TinyStories)
to demonstrate generalization across data scales and domains.

Usage:
    python run_multi_dataset_experiment.py --seeds 3 --epochs 3
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("multi_dataset_experiment")


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


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lfn(vocab_size: int, device: str, context_length: int = 256):
    """Build the Latent Forecasting Network."""
    from models.complete_model import ModelConfig, LatentForecastingModel
    
    cfg = ModelConfig(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        forecast_horizons=[1, 2, 5],
        max_context_length=context_length,
        lambda_latent=0.1,
    )
    model = LatentForecastingModel(cfg).to(device)
    return model, cfg


def build_baseline(vocab_size: int, device: str, context_length: int = 256):
    """Build the Baseline Transformer."""
    from models.baseline_models import StandardTransformer
    
    model = StandardTransformer(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        max_context_length=context_length,
    ).to(device)
    return model


def create_loaders(dataset_name: str, device: str, context_length: int = 256, batch_size: int = 8):
    """Create data loaders for specified dataset."""
    from data.dataset_loader import create_dataloaders
    
    return create_dataloaders(
        dataset_name=dataset_name,
        tokenizer_name="gpt2",
        context_length=context_length,
        stride=128,
        batch_size=batch_size,
        num_workers=0,
    )


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
    log_every = 50

    for epoch in range(num_epochs):
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

            try:
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
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM error, skipping batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                raise

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

        # Validation
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 50:
                    break
                tokens = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                try:
                    if is_lfn:
                        out = model(tokens, labels=labels, compute_latent_loss=True)
                    else:
                        out = model(tokens, labels=labels)
                    val_total += out.total_loss.item()
                    val_batches += 1
                except RuntimeError:
                    continue
        avg_val_loss = val_total / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        logger.info(
            f"  Epoch {epoch+1}/{num_epochs} DONE  "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"time={epoch_time:.1f}s"
        )

    return history


def evaluate_model(model, test_loader, device, is_lfn=True):
    """Evaluate a trained model and return all metrics."""
    from evaluation.metrics import compute_token_accuracy

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    lps_sums = {}
    mse_sums = {}
    latents_list = []
    all_hidden_states = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 50:
                break
                
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            try:
                if is_lfn:
                    output = model(tokens, labels=labels, compute_latent_loss=True)
                else:
                    output = model(tokens, labels=labels)

                total_loss += output.token_loss.item()
                acc = compute_token_accuracy(output.logits, labels)
                total_acc += acc.item()
                num_batches += 1

                latents = output.latents.detach().cpu()
                latents_list.append(latents)
                
                # Collect for linear probe
                all_hidden_states.append(latents[:, 0, :].cpu())
                all_labels.append(labels[:, 0].cpu())

                if is_lfn and hasattr(output, 'predicted_latents') and output.predicted_latents:
                    for k, preds in output.predicted_latents.items():
                        targets = output.latents[:, k:, :]
                        diff = targets - preds
                        l2 = torch.norm(diff, p=2, dim=-1).mean().item()
                        mse = torch.nn.functional.mse_loss(preds, targets).item()
                        lps_sums[k] = lps_sums.get(k, 0.0) + l2
                        mse_sums[k] = mse_sums.get(k, 0.0) + mse
            except RuntimeError:
                continue

    avg_loss = total_loss / max(num_batches, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    avg_acc = total_acc / max(num_batches, 1)

    results = {
        "perplexity": ppl,
        "token_loss": avg_loss,
        "token_accuracy": avg_acc,
    }

    if lps_sums:
        results["lps_scores"] = {str(k): v / num_batches for k, v in sorted(lps_sums.items())}
        results["horizon_mse"] = {str(k): v / num_batches for k, v in sorted(mse_sums.items())}

    # Representation metrics
    if latents_list:
        all_latents = torch.cat(latents_list, dim=0)
        flat = all_latents.reshape(-1, all_latents.shape[-1]).numpy()
        if flat.shape[0] > 20000:
            idx = np.random.choice(flat.shape[0], 20000, replace=False)
            flat = flat[idx]
        results["representation_metrics"] = compute_repr_metrics(flat, all_latents)

    # Linear probe evaluation
    if all_hidden_states and all_labels:
        hidden = torch.cat(all_hidden_states, dim=0).numpy()[:1000]
        labels_np = torch.cat(all_labels, dim=0).numpy()[:1000]
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                hidden, labels_np, test_size=0.3, random_state=42, stratify=labels_np
            )
            clf = LogisticRegression(max_iter=100, random_state=42)
            clf.fit(X_train, y_train)
            linear_probe_acc = clf.score(X_test, y_test)
            results["linear_probe_accuracy"] = linear_probe_acc
        except Exception as e:
            logger.warning(f"Linear probe failed: {e}")
            results["linear_probe_accuracy"] = 0.0
    else:
        results["linear_probe_accuracy"] = 0.0

    return results


def compute_repr_metrics(flat_latents: np.ndarray, seq_latents: torch.Tensor) -> dict:
    """Compute representation quality metrics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    metrics = {}

    # Latent variance
    var = np.var(flat_latents, axis=0).mean()
    metrics["latent_variance"] = float(var)

    # Latent entropy
    bins = 50
    hist_sum = 0.0
    for d in range(flat_latents.shape[1]):
        counts, _ = np.histogram(flat_latents[:, d], bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        hist_sum += -np.sum(probs * np.log(probs))
    metrics["latent_entropy"] = float(hist_sum / flat_latents.shape[1])

    # Cosine similarity drift
    seqs = seq_latents[:500]
    a = seqs[:, :-1, :]
    b = seqs[:, 1:, :]
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    drift = 1.0 - cos_sim.mean().item()
    metrics["cosine_similarity_drift"] = drift

    # Cluster separability
    n_clusters = min(10, flat_latents.shape[0] // 2)
    if n_clusters >= 2:
        try:
            km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
            labels = km.fit_predict(flat_latents[:5000])
            sil = silhouette_score(flat_latents[:5000], labels, sample_size=2000)
            metrics["cluster_separability"] = float(sil)
        except Exception:
            metrics["cluster_separability"] = None
    else:
        metrics["cluster_separability"] = None

    # Effective dimensionality
    try:
        cov = np.cov(flat_latents.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        eff_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        metrics["effective_dim"] = float(eff_dim)
    except Exception:
        metrics["effective_dim"] = 0.0

    return metrics


def run_single_dataset(model_type: str, dataset_name: str, seed: int, 
                       num_epochs: int, device: str) -> Dict[str, Any]:
    """Run single experiment on a dataset."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running {model_type} on {dataset_name} (seed={seed})")
    logger.info(f"{'='*70}")
    
    set_seed(seed)
    
    # Adjust context length for dataset
    context_length = 256  # Default
    batch_size = 8
    
    if dataset_name == "tinystories":
        # TinyStories has simpler language, can use same context length
        pass
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_loaders(
        dataset_name, device, context_length, batch_size
    )
    
    vocab_size = 50257
    
    # Build model
    is_lfn = model_type == "lfn"
    
    if model_type == "baseline":
        model = build_baseline(vocab_size, device, context_length)
    elif model_type == "lfn":
        model, _ = build_lfn(vocab_size, device, context_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Train
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, is_lfn=is_lfn
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, is_lfn=is_lfn)
    results["seed"] = seed
    results["num_parameters"] = count_parameters(model)
    results["training_history"] = history
    results["dataset"] = dataset_name
    results["model_type"] = model_type
    
    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return results


def compute_stats(values):
    """Compute mean, std, and 95% CI."""
    if not values:
        return 0.0, 0.0, (0.0, 0.0)
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    
    if len(arr) > 1:
        sem = std / np.sqrt(len(arr))
        ci = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=sem)
    else:
        ci = (mean, mean)
    
    return mean, std, ci


def aggregate_dataset_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across seeds for a dataset."""
    metrics = ["perplexity", "token_accuracy", "token_loss", "linear_probe_accuracy"]
    agg = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        mean, std, ci = compute_stats(values)
        agg[metric] = {"mean": mean, "std": std, "ci_95": ci}
    
    # Effective dim
    eff_dims = []
    for r in results:
        rm = r.get("representation_metrics", {})
        if rm and "effective_dim" in rm:
            eff_dims.append(rm["effective_dim"])
    mean, std, ci = compute_stats(eff_dims)
    agg["effective_dim"] = {"mean": mean, "std": std, "ci_95": ci}
    
    # LPS scores
    lps_keys = set()
    for r in results:
        if "lps_scores" in r:
            lps_keys.update(r["lps_scores"].keys())
    
    agg["lps_scores"] = {}
    for k in lps_keys:
        values = [r["lps_scores"][k] for r in results if "lps_scores" in r and k in r["lps_scores"]]
        mean, std, ci = compute_stats(values)
        agg["lps_scores"][k] = {"mean": mean, "std": std, "ci_95": ci}
    
    agg["n_seeds"] = len(results)
    agg["all_results"] = results
    
    return agg


def run_multi_dataset_experiment(datasets: List[str], seeds: List[int], 
                                 num_epochs: int, device: str) -> Dict[str, Any]:
    """Run experiments on multiple datasets."""
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"# DATASET: {dataset.upper()}")
        logger.info(f"{'#'*70}")
        
        dataset_results = {
            "baseline": [],
            "lfn": [],
        }
        
        # Run baseline
        for seed in seeds:
            try:
                r = run_single_dataset("baseline", dataset, seed, num_epochs, device)
                dataset_results["baseline"].append(r)
            except Exception as e:
                logger.error(f"Error running baseline on {dataset} with seed {seed}: {e}")
        
        # Run LFN
        for seed in seeds:
            try:
                r = run_single_dataset("lfn", dataset, seed, num_epochs, device)
                dataset_results["lfn"].append(r)
            except Exception as e:
                logger.error(f"Error running LFN on {dataset} with seed {seed}: {e}")
        
        # Aggregate
        all_results[dataset] = {
            "baseline": aggregate_dataset_results(dataset_results["baseline"]),
            "lfn": aggregate_dataset_results(dataset_results["lfn"]),
        }
        
        # Print summary
        print_dataset_summary(dataset, all_results[dataset])
    
    return all_results


def print_dataset_summary(dataset_name: str, results: Dict[str, Any]):
    """Print summary for a dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} RESULTS")
    print(f"{'='*70}")
    
    b = results["baseline"]
    l = results["lfn"]
    
    print(f"  {'Metric':<30} {'Baseline':>18} {'LFN':>18}")
    print(f"  {'-'*66}")
    
    for metric, label in [
        ("perplexity", "Perplexity"),
        ("token_accuracy", "Token Accuracy"),
        ("linear_probe_accuracy", "Linear Probe Acc"),
        ("effective_dim", "Effective Dim"),
    ]:
        if metric in b and metric in l:
            b_val = b[metric]
            l_val = l[metric]
            b_str = f"{b_val['mean']:.3f}±{b_val['std']:.3f}"
            l_str = f"{l_val['mean']:.3f}±{l_val['std']:.3f}"
            print(f"  {label:<30} {b_str:>18} {l_str:>18}")
    
    print(f"{'='*70}\n")


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table for multi-dataset results."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-Dataset Evaluation: WikiText-2 vs TinyStories (Mean $\pm$ Std across 3 seeds)}")
    lines.append(r"\label{tab:cross_dataset}")
    lines.append(r"\begin{tabular}{@{}llcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{PPL} $\downarrow$ & \textbf{Acc} $\uparrow$ & \textbf{Eff.Dim} $\uparrow$ & \textbf{Lin.Probe} $\uparrow$ & $\Delta$ \textbf{Probe} \\ \midrule")
    
    for dataset in ["wikitext-2", "tinystories"]:
        if dataset not in results:
            continue
            
        b = results[dataset]["baseline"]
        l = results[dataset]["lfn"]
        
        # Dataset name for display
        display_name = "WikiText-2" if dataset == "wikitext-2" else "TinyStories"
        
        # Baseline row
        b_ppl = f"{b['perplexity']['mean']:.1f} $\\pm$ {b['perplexity']['std']:.1f}"
        b_acc = f"{b['token_accuracy']['mean']:.4f} $\\pm$ {b['token_accuracy']['std']:.4f}"
        b_ed = f"{b['effective_dim']['mean']:.1f} $\\pm$ {b['effective_dim']['std']:.1f}"
        b_lp = f"{b['linear_probe_accuracy']['mean']:.3f} $\\pm$ {b['linear_probe_accuracy']['std']:.3f}"
        
        lines.append(f"{display_name} & Baseline & {b_ppl} & {b_acc} & {b_ed} & {b_lp} & --- \\\\")
        
        # LFN row
        l_ppl = f"{l['perplexity']['mean']:.1f} $\\pm$ {l['perplexity']['std']:.1f}"
        l_acc = f"{l['token_accuracy']['mean']:.4f} $\\pm$ {l['token_accuracy']['std']:.4f}"
        l_ed = f"{l['effective_dim']['mean']:.1f} $\\pm$ {l['effective_dim']['std']:.1f}"
        l_lp = f"{l['linear_probe_accuracy']['mean']:.3f} $\\pm$ {l['linear_probe_accuracy']['std']:.3f}"
        
        delta_lp = l['linear_probe_accuracy']['mean'] - b['linear_probe_accuracy']['mean']
        delta_str = f"+{delta_lp:.3f}"
        
        lines.append(f" & LFN & {l_ppl} & {l_acc} & {l_ed} & {l_lp} & {delta_str} \\\\")
        
        if dataset != "tinystories":
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Experiment Runner")
    parser.add_argument("--datasets", nargs="+", default=["wikitext-2", "tinystories"],
                       choices=["wikitext-2", "tinystories", "wikitext-103"],
                       help="Datasets to run experiments on")
    parser.add_argument("--seeds", type=int, default=3,
                       help="Number of random seeds")
    parser.add_argument("--seed-start", type=int, default=42,
                       help="Starting seed value")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--output", type=str, default="experiments/results/multi_dataset",
                       help="Output directory")
    args = parser.parse_args()
    
    device = resolve_device(args.device)
    logger.info(f"Device: {device}")
    
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    logger.info(f"Running on datasets: {args.datasets}")
    logger.info(f"Seeds: {seeds}")
    
    start_time = time.time()
    
    # Run experiments
    results = run_multi_dataset_experiment(args.datasets, seeds, args.epochs, device)
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal experiment time: {elapsed/60:.1f} minutes")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    print("\n\nLaTeX TABLE CODE:")
    print("="*70)
    print(latex_table)
    print("="*70)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    output_data = {
        "metadata": {
            "datasets": args.datasets,
            "seeds": seeds,
            "num_epochs": args.epochs,
            "device": device,
            "total_time_minutes": elapsed / 60,
        },
        "results": serializable_results,
        "latex_table": latex_table,
    }
    
    results_path = output_dir / "multi_dataset_results.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
