#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner for Statistical Significance Testing.

This script runs experiments with multiple random seeds (3-5) and computes
mean ± standard deviation for all metrics, enabling statistical significance
reporting in the research paper.

Usage:
    python run_multi_seed_experiment.py --experiment baseline --seeds 5 --epochs 3
    python run_multi_seed_experiment.py --experiment ablation_lambda --seeds 5
    python run_multi_seed_experiment.py --experiment ablation_horizon --seeds 5
    python run_multi_seed_experiment.py --experiment dataset_scale --seeds 3 --dataset wikitext-103
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("multi_seed_experiment")


@dataclass
class SeedResults:
    """Results from a single seed run."""
    seed: int
    perplexity: float
    token_accuracy: float
    token_loss: float
    lps_scores: Dict[str, float]
    linear_probe_accuracy: float
    effective_dim: float
    cka_vs_baseline: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateResults:
    """Aggregated results across multiple seeds with statistics."""
    mean_perplexity: float
    std_perplexity: float
    mean_token_accuracy: float
    std_token_accuracy: float
    mean_token_loss: float
    std_token_loss: float
    mean_lps_scores: Dict[str, float]
    std_lps_scores: Dict[str, float]
    mean_linear_probe: float
    std_linear_probe: float
    mean_effective_dim: float
    std_effective_dim: float
    mean_cka: float
    std_cka: float
    n_seeds: int
    confidence_interval_95: Dict[str, Tuple[float, float]]
    
    def format_with_error(self, metric_name: str, value: float, std: float) -> str:
        """Format metric with error bar for LaTeX table."""
        return f"{value:.3f} $\\pm$ {std:.3f}"


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


def build_lfn(vocab_size: int, device: str, lambda_latent: float = 0.1, 
              horizons: List[int] = None):
    """Build the Latent Forecasting Network."""
    from models.complete_model import ModelConfig, LatentForecastingModel
    
    if horizons is None:
        horizons = [1, 2, 5]
    
    cfg = ModelConfig(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        forecast_horizons=horizons,
        max_context_length=256,
        lambda_latent=lambda_latent,
    )
    model = LatentForecastingModel(cfg).to(device)
    return model, cfg


def build_baseline(vocab_size: int, device: str):
    """Build the Baseline Transformer."""
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


def build_cpc(vocab_size: int, device: str):
    """Build the CPC model."""
    from models.cpc_model import build_cpc_model
    
    model = build_cpc_model(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        max_context_length=256,
        prediction_horizons=(1, 2, 5),
        temperature=0.1,
        num_negatives=10,
        device=device,
    )
    return model


def build_jepa(vocab_size: int, device: str):
    """Build the JEPA model."""
    from models.jepa_model import build_jepa_model
    
    model = build_jepa_model(
        vocab_size=vocab_size,
        latent_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        max_context_length=256,
        target_encoder_ema=True,
        ema_decay=0.996,
        prediction_horizons=(1, 2, 5),
        lambda_jepa=1.0,
        device=device,
    )
    return model


def create_loaders(dataset_name: str, device: str, context_length: int = 256):
    """Create data loaders for specified dataset."""
    from data.dataset_loader import create_dataloaders
    
    return create_dataloaders(
        dataset_name=dataset_name,
        tokenizer_name="gpt2",
        context_length=context_length,
        stride=128,
        batch_size=8,
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
                    
                    # Update EMA for JEPA
                    if hasattr(model, 'update_target_encoder'):
                        model.update_target_encoder()

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


def evaluate_model(model, test_loader, device, is_lfn=True, model_type="lfn"):
    """Evaluate a trained model and return all metrics."""
    from evaluation.metrics import compute_perplexity, compute_token_accuracy
    from evaluation.downstream_eval import LinearProbeEvaluator

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
                    if model_type == "cpc":
                        output = model(tokens, labels=labels, compute_cpc_loss=True)
                    elif model_type == "jepa":
                        output = model(tokens, labels=labels, compute_jepa_loss=True)
                    else:
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

                # LPS per horizon
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

    # Linear probe evaluation (simplified - just use a simple classifier)
    if all_hidden_states and all_labels:
        hidden = torch.cat(all_hidden_states, dim=0).numpy()[:1000]
        labels_np = torch.cat(all_labels, dim=0).numpy()[:1000]
        
        # Simple linear classification
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


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Centered Kernel Alignment between two representation matrices."""
    def center(K):
        n = K.shape[0]
        unit = np.ones((n, n)) / n
        return K - unit @ K - K @ unit + unit @ K @ unit
    
    def linear_kernel(X):
        return X @ X.T
    
    Kx = center(linear_kernel(X))
    Ky = center(linear_kernel(Y))
    
    hsic = np.trace(Kx @ Ky)
    var_x = np.trace(Kx @ Kx)
    var_y = np.trace(Ky @ Ky)
    
    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0
    
    return float(hsic / np.sqrt(var_x * var_y))


def run_single_seed(seed: int, model_type: str, dataset_name: str, 
                    num_epochs: int, device: str, **kwargs) -> Dict[str, Any]:
    """Run experiment with a single seed."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {model_type} with seed={seed}")
    logger.info(f"{'='*60}")
    
    set_seed(seed)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_loaders(dataset_name, device)
    vocab_size = 50257
    
    # Build model
    is_lfn = model_type in ["lfn", "cpc", "jepa"]
    
    if model_type == "baseline":
        model = build_baseline(vocab_size, device)
    elif model_type == "lfn":
        lambda_val = kwargs.get("lambda_latent", 0.1)
        horizons = kwargs.get("horizons", [1, 2, 5])
        model, _ = build_lfn(vocab_size, device, lambda_val, horizons)
    elif model_type == "cpc":
        model = build_cpc(vocab_size, device)
    elif model_type == "jepa":
        model = build_jepa(vocab_size, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Train
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, is_lfn=is_lfn,
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, is_lfn=is_lfn, model_type=model_type)
    results["seed"] = seed
    results["num_parameters"] = count_parameters(model)
    results["training_history"] = history
    
    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return results


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean ± std across multiple seed runs."""
    n = len(all_results)
    
    # Extract metrics
    perplexities = [r["perplexity"] for r in all_results]
    token_accs = [r["token_accuracy"] for r in all_results]
    token_losses = [r["token_loss"] for r in all_results]
    linear_probes = [r.get("linear_probe_accuracy", 0.0) for r in all_results]
    
    # Representation metrics
    eff_dims = []
    for r in all_results:
        rm = r.get("representation_metrics", {})
        if rm and "effective_dim" in rm:
            eff_dims.append(rm["effective_dim"])
    
    def compute_stats(values):
        """Compute mean, std, and 95% CI."""
        if not values:
            return 0.0, 0.0, (0.0, 0.0)
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        
        # 95% confidence interval
        if len(arr) > 1:
            sem = std / np.sqrt(len(arr))
            ci = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=sem)
        else:
            ci = (mean, mean)
        
        return mean, std, ci
    
    agg = {}
    
    # Perplexity
    mean, std, ci = compute_stats(perplexities)
    agg["perplexity"] = {"mean": mean, "std": std, "ci_95": ci}
    
    # Token accuracy
    mean, std, ci = compute_stats(token_accs)
    agg["token_accuracy"] = {"mean": mean, "std": std, "ci_95": ci}
    
    # Token loss
    mean, std, ci = compute_stats(token_losses)
    agg["token_loss"] = {"mean": mean, "std": std, "ci_95": ci}
    
    # Linear probe
    mean, std, ci = compute_stats(linear_probes)
    agg["linear_probe_accuracy"] = {"mean": mean, "std": std, "ci_95": ci}
    
    # Effective dimensionality
    mean, std, ci = compute_stats(eff_dims)
    agg["effective_dim"] = {"mean": mean, "std": std, "ci_95": ci}
    
    # LPS scores (if available)
    lps_keys = set()
    for r in all_results:
        if "lps_scores" in r:
            lps_keys.update(r["lps_scores"].keys())
    
    agg["lps_scores"] = {}
    for k in lps_keys:
        values = [r["lps_scores"][k] for r in all_results if "lps_scores" in r and k in r["lps_scores"]]
        mean, std, ci = compute_stats(values)
        agg["lps_scores"][k] = {"mean": mean, "std": std, "ci_95": ci}
    
    agg["n_seeds"] = n
    agg["all_results"] = all_results
    
    return agg


def run_baseline_comparison(seeds: List[int], dataset_name: str, 
                            num_epochs: int, device: str) -> Dict[str, Any]:
    """Run baseline vs LFN comparison with multiple seeds."""
    results = {
        "baseline": [],
        "lfn": [],
    }
    
    # Run baseline
    for seed in seeds:
        r = run_single_seed(seed, "baseline", dataset_name, num_epochs, device)
        results["baseline"].append(r)
    
    # Run LFN
    for seed in seeds:
        r = run_single_seed(seed, "lfn", dataset_name, num_epochs, device, lambda_latent=0.1)
        results["lfn"].append(r)
    
    # Aggregate
    aggregated = {
        "baseline": aggregate_results(results["baseline"]),
        "lfn": aggregate_results(results["lfn"]),
    }
    
    return aggregated


def run_lambda_ablation(seeds: List[int], dataset_name: str, 
                        num_epochs: int, device: str) -> Dict[str, Any]:
    """Run lambda ablation with multiple seeds."""
    lambda_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    results = {}
    
    for lam in lambda_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Lambda = {lam}")
        logger.info(f"{'='*60}")
        
        lam_results = []
        for seed in seeds:
            if lam == 0.0:
                r = run_single_seed(seed, "baseline", dataset_name, num_epochs, device)
            else:
                r = run_single_seed(seed, "lfn", dataset_name, num_epochs, device, lambda_latent=lam)
            lam_results.append(r)
        
        results[f"lambda_{lam}"] = aggregate_results(lam_results)
    
    return results


def run_horizon_ablation(seeds: List[int], dataset_name: str, 
                         num_epochs: int, device: str) -> Dict[str, Any]:
    """Run horizon ablation with multiple seeds."""
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
    results = {}
    
    for horizons in horizon_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizons = {horizons}")
        logger.info(f"{'='*60}")
        
        h_results = []
        for seed in seeds:
            r = run_single_seed(seed, "lfn", dataset_name, num_epochs, device, 
                              lambda_latent=0.1, horizons=horizons)
            h_results.append(r)
        
        key = f"h_{'_'.join(map(str, horizons))}"
        results[key] = aggregate_results(h_results)
    
    return results


def run_baseline_model_comparison(seeds: List[int], dataset_name: str,
                                  num_epochs: int, device: str) -> Dict[str, Any]:
    """Run comparison of all models: Baseline, CPC, JEPA, LFN."""
    model_types = ["baseline", "cpc", "jepa", "lfn"]
    results = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_type.upper()}")
        logger.info(f"{'='*60}")
        
        model_results = []
        for seed in seeds:
            if model_type == "lfn":
                r = run_single_seed(seed, model_type, dataset_name, num_epochs, device, 
                                    lambda_latent=0.1)
            else:
                r = run_single_seed(seed, model_type, dataset_name, num_epochs, device)
            model_results.append(r)
        
        results[model_type] = aggregate_results(model_results)
    
    return results


def print_aggregate_table(aggregated: Dict[str, Any], title: str):
    """Print formatted table of aggregated results."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    # Header
    print(f"  {'Metric':<25} {'Mean':>15} {'Std':>15} {'95% CI':>20}")
    print(f"  {'-'*75}")
    
    # Main metrics
    for metric in ["perplexity", "token_accuracy", "token_loss", "linear_probe_accuracy", "effective_dim"]:
        if metric in aggregated:
            m = aggregated[metric]
            ci_low, ci_high = m["ci_95"]
            print(f"  {metric:<25} {m['mean']:>15.4f} {m['std']:>15.4f} "
                  f"[{ci_low:.4f}, {ci_high:.4f}]")
    
    # LPS scores
    if "lps_scores" in aggregated and aggregated["lps_scores"]:
        print(f"\n  LPS Scores by Horizon:")
        for k, v in sorted(aggregated["lps_scores"].items(), key=lambda x: int(x[0])):
            ci_low, ci_high = v["ci_95"]
            print(f"    k={k:<5} {v['mean']:>11.4f} ± {v['std']:<11.4f} "
                  f"[{ci_low:.4f}, {ci_high:.4f}]")
    
    print(f"{'='*80}\n")


def generate_latex_table_with_errors(aggregated_results: Dict[str, Dict[str, Any]], 
                                     table_name: str) -> str:
    """Generate LaTeX table code with error bars."""
    lines = []
    lines.append(f"% {table_name}")
    lines.append(f"% Mean ± Std across {list(aggregated_results.values())[0].get('n_seeds', 'N')} seeds")
    lines.append("")
    
    if "baseline" in aggregated_results and "lfn" in aggregated_results:
        # Baseline comparison table
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{WikiText-2 Evaluation Results with Error Bars (Mean ± Std)}")
        lines.append(r"\label{tab:results_with_errors}")
        lines.append(r"\begin{tabular*}{\columnwidth}{@{}l@{}rr@{}}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Metric} & \textbf{Baseline} & \textbf{LFN} \\ \midrule")
        
        b = aggregated_results["baseline"]
        l = aggregated_results["lfn"]
        
        # Format each row with error bars
        for metric, label in [
            ("perplexity", "Perplexity ($\\downarrow$)"),
            ("token_accuracy", "Token Accuracy ($\\uparrow$)"),
            ("token_loss", "Token Loss ($\\downarrow$)"),
        ]:
            b_mean = b[metric]["mean"]
            b_std = b[metric]["std"]
            l_mean = l[metric]["mean"]
            l_std = l[metric]["std"]
            lines.append(f"{label} & ${b_mean:.2f} \\pm {b_std:.2f}$ & ${l_mean:.2f} \\pm {l_std:.2f}$ \\\\")
        
        # Representation metrics
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{3}{l}{\textit{Representation Quality}} \\")
        
        # Effective dim
        b_ed = b.get("effective_dim", {}).get("mean", 0)
        b_ed_std = b.get("effective_dim", {}).get("std", 0)
        l_ed = l.get("effective_dim", {}).get("mean", 0)
        l_ed_std = l.get("effective_dim", {}).get("std", 0)
        lines.append(f"Eff. Dimensionality ($\\uparrow$) & ${b_ed:.1f} \\pm {b_ed_std:.1f}$ & ${l_ed:.1f} \\pm {l_ed_std:.1f}$ \\\\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")
        lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["baseline", "ablation_lambda", "ablation_horizon", 
                               "model_comparison", "dataset_scale"],
                       help="Type of experiment to run")
    parser.add_argument("--seeds", type=int, default=3,
                       help="Number of random seeds to run (default: 3)")
    parser.add_argument("--seed-start", type=int, default=42,
                       help="Starting seed value (default: 42)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda/mps)")
    parser.add_argument("--dataset", type=str, default="wikitext-2",
                       choices=["wikitext-2", "wikitext-103", "tinystories"],
                       help="Dataset to use")
    parser.add_argument("--output", type=str, default="experiments/results/multi_seed",
                       help="Output directory for results")
    parser.add_argument("--max-batches", type=int, default=500,
                       help="Max batches per epoch for faster testing")
    args = parser.parse_args()
    
    device = resolve_device(args.device)
    logger.info(f"Device: {device}")
    
    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    logger.info(f"Running with seeds: {seeds}")
    
    # Run experiment
    start_time = time.time()
    
    if args.experiment == "baseline":
        results = run_baseline_comparison(seeds, args.dataset, args.epochs, device)
        print_aggregate_table(results["baseline"], f"Baseline Transformer ({args.dataset})")
        print_aggregate_table(results["lfn"], f"LFN ({args.dataset})")
        
        # Generate LaTeX
        latex = generate_latex_table_with_errors(results, "Baseline Comparison")
        print("\n" + latex)
        
    elif args.experiment == "ablation_lambda":
        results = run_lambda_ablation(seeds, args.dataset, args.epochs, device)
        for key, agg in results.items():
            print_aggregate_table(agg, f"Lambda {key}")
            
    elif args.experiment == "ablation_horizon":
        results = run_horizon_ablation(seeds, args.dataset, args.epochs, device)
        for key, agg in results.items():
            print_aggregate_table(agg, f"Horizon {key}")
            
    elif args.experiment == "model_comparison":
        results = run_baseline_model_comparison(seeds, args.dataset, args.epochs, device)
        for model_type, agg in results.items():
            print_aggregate_table(agg, f"{model_type.upper()} ({args.dataset})")
            
    elif args.experiment == "dataset_scale":
        # Run on multiple datasets
        all_results = {}
        for dataset in ["wikitext-2", "tinystories"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset: {dataset}")
            logger.info(f"{'='*60}")
            all_results[dataset] = run_baseline_comparison(seeds, dataset, args.epochs, device)
        results = all_results
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal experiment time: {elapsed/60:.1f} minutes")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    exp_name = f"{args.experiment}_{args.dataset}_{args.seeds}seeds"
    results_path = output_dir / f"{exp_name}.json"
    
    # Convert to JSON-serializable format
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
    
    # Add metadata
    output_data = {
        "metadata": {
            "experiment": args.experiment,
            "dataset": args.dataset,
            "seeds": seeds,
            "num_epochs": args.epochs,
            "device": device,
            "total_time_minutes": elapsed / 60,
        },
        "results": serializable_results,
    }
    
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
