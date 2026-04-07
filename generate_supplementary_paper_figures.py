#!/usr/bin/env python3
"""
Generate supplementary paper figures:
  - fig4_residual_diagnostics.png
  - fig5_psi_overlay.png

Figure 4 is computed from a quick latent forecasting run on WikiText-2 so the
residual diagnostics are based on actual model outputs.
Figure 5 is generated from the existing horizon-ablation results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "experiments" / ".cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
for path in (CACHE_DIR, HF_CACHE_DIR, MPL_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats

from data.dataset_loader import create_dataloaders
from models.complete_model import LatentForecastingModel, ModelConfig


OUT_DIR = ROOT / "experiments" / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG4_PATH = OUT_DIR / "fig4_residual_diagnostics.png"
FIG4_STATS_PATH = OUT_DIR / "fig4_residual_diagnostics_stats.json"
FIG5_PATH = OUT_DIR / "fig5_psi_overlay.png"

RESULTS_DIR = ROOT / "experiments" / "results" / "theoretical_validation"
HORIZON_JSON = RESULTS_DIR / "horizon_ablation_results.json"


PALETTE = {
    "blue": "#1D4ED8",
    "teal": "#0F766E",
    "orange": "#EA580C",
    "red": "#B91C1C",
    "gold": "#B45309",
    "slate": "#334155",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "bg": "#F8FAFC",
}

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "axes.facecolor": PALETTE["bg"],
        "figure.facecolor": "white",
        "axes.edgecolor": PALETTE["grid"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
    },
)


@dataclass
class Figure4Config:
    context_length: int = 64
    stride: int = 64
    batch_size: int = 4
    latent_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    hidden_dim: int = 256
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 2, 5)
    lambda_latent: float = 0.1
    max_train_steps: int = 60
    eval_batches: int = 16
    sample_size: int = 4000
    lr: float = 3e-4
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_horizons(horizons: Iterable[int]) -> str:
    return "{" + ",".join(str(h) for h in horizons) + "}"


def standardize(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std < 1e-8:
        return values - mean
    return (values - mean) / std


def train_quick_lfn(cfg: Figure4Config, device: str) -> LatentForecastingModel:
    train_loader, _, _ = create_dataloaders(
        dataset_name="wikitext-2",
        tokenizer_name="gpt2",
        context_length=cfg.context_length,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    model_cfg = ModelConfig(
        vocab_size=50257,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        forecast_horizons=list(cfg.horizons),
        max_context_length=cfg.context_length,
        lambda_latent=cfg.lambda_latent,
    )
    model = LatentForecastingModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    model.train()
    step = 0
    while step < cfg.max_train_steps:
        for batch in train_loader:
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(tokens, labels=labels, compute_latent_loss=True)
            optimizer.zero_grad()
            output.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step >= cfg.max_train_steps:
                break

    return model


def collect_residuals(model: LatentForecastingModel, cfg: Figure4Config, device: str) -> Dict[int, np.ndarray]:
    _, _, test_loader = create_dataloaders(
        dataset_name="wikitext-2",
        tokenizer_name="gpt2",
        context_length=cfg.context_length,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    residuals: Dict[int, List[np.ndarray]] = {h: [] for h in cfg.horizons}
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= cfg.eval_batches:
                break
            tokens = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(tokens, labels=labels, compute_latent_loss=True)
            for horizon, preds in output.predicted_latents.items():
                targets = output.latents[:, horizon:, :]
                diff = (targets - preds).reshape(-1).detach().cpu().numpy()
                residuals[horizon].append(diff)

    sampled: Dict[int, np.ndarray] = {}
    for horizon, chunks in residuals.items():
        merged = np.concatenate(chunks, axis=0)
        if merged.size > cfg.sample_size:
            idx = np.random.choice(merged.size, size=cfg.sample_size, replace=False)
            merged = merged[idx]
        sampled[horizon] = standardize(merged)
    return sampled


def residual_stats(sample: np.ndarray) -> Dict[str, float]:
    shapiro_sample = sample[: min(5000, sample.size)]
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)
    return {
        "n": int(sample.size),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample, ddof=1)),
        "skewness": float(stats.skew(sample)),
        "kurtosis": float(stats.kurtosis(sample)),
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_p_value": float(shapiro_p),
    }


def plot_figure4(residuals: Dict[int, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.28)

    horizon_colors = {
        1: PALETTE["blue"],
        2: PALETTE["teal"],
        5: PALETTE["orange"],
    }
    stats_blob: Dict[str, Dict[str, float]] = {}

    for col, horizon in enumerate(sorted(residuals)):
        sample = residuals[horizon]
        color = horizon_colors.get(horizon, PALETTE["slate"])
        stats_blob[str(horizon)] = residual_stats(sample)

        qq_ax = axes[0, col]
        (osm, osr), (slope, intercept, _) = stats.probplot(sample, dist="norm")
        qq_ax.scatter(osm, osr, s=10, alpha=0.5, color=color, edgecolors="none")
        xline = np.linspace(min(osm), max(osm), 200)
        qq_ax.plot(xline, slope * xline + intercept, "--", lw=1.5, color=PALETTE["red"])
        qq_ax.set_title(f"Horizon k={horizon}", fontsize=11, weight="bold")
        qq_ax.set_xlabel("Theoretical Quantiles")
        qq_ax.set_ylabel("Standardized Residual Quantiles")
        stat_text = (
            f"Shapiro p={stats_blob[str(horizon)]['shapiro_p_value']:.3f}\n"
            f"Skew={stats_blob[str(horizon)]['skewness']:.2f}\n"
            f"Kurt={stats_blob[str(horizon)]['kurtosis']:.2f}"
        )
        qq_ax.text(
            0.04,
            0.96,
            stat_text,
            transform=qq_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["grid"]},
        )

        hist_ax = axes[1, col]
        sns.histplot(sample, bins=40, stat="density", color=color, alpha=0.35, edgecolor="none", ax=hist_ax)
        x = np.linspace(-4, 4, 400)
        hist_ax.plot(x, stats.norm.pdf(x), color=PALETTE["red"], lw=1.7, label=r"$\mathcal{N}(0,1)$")
        sns.kdeplot(sample, color=color, lw=1.8, ax=hist_ax)
        hist_ax.set_xlabel("Standardized Residual")
        hist_ax.set_ylabel("Density")
        hist_ax.set_xlim(-4, 4)
        hist_ax.legend(frameon=True, loc="upper left", fontsize=8)

    fig.suptitle(
        "Residual Diagnostics for the Gaussian Forecasting Prior",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Top row: Q-Q plots against a standard Gaussian. Bottom row: empirical standardized residual density "
        "with Gaussian reference overlay for the three forecasting horizons.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["muted"],
    )
    fig.savefig(FIG4_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    FIG4_STATS_PATH.write_text(json.dumps(stats_blob, indent=2))


def plot_figure5() -> None:
    horizon_results = json.loads(HORIZON_JSON.read_text())
    baseline_effdim = 142.3

    records = []
    for key, row in horizon_results.items():
        horizons = row["horizons"]
        records.append(
            {
                "label": format_horizons(horizons),
                "psi": float(row["psi_K"]),
                "effdim": float(row["effective_dimensionality"]),
                "delta_effdim": float(row["effective_dimensionality"]) - baseline_effdim,
                "diversity": float(row["diversity"]),
                "k_size": len(horizons),
            }
        )

    records.sort(key=lambda item: item["psi"])
    psi = np.array([r["psi"] for r in records], dtype=float)
    delta = np.array([r["delta_effdim"] for r in records], dtype=float)
    diversity = np.array([r["diversity"] for r in records], dtype=float)
    sizes = np.array([90 + 40 * r["k_size"] for r in records], dtype=float)

    slope = float((psi @ delta) / (psi @ psi))
    x_fit = np.linspace(0.0, psi.max() * 1.08, 200)
    y_fit = slope * x_fit
    pearson_r = float(np.corrcoef(psi, delta)[0, 1])
    spearman_r = float(stats.spearmanr(psi, delta).statistic)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    scatter = ax.scatter(
        psi,
        delta,
        c=diversity,
        s=sizes,
        cmap="cividis",
        alpha=0.92,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.plot(
        x_fit,
        y_fit,
        "--",
        color=PALETTE["red"],
        lw=2.0,
        label=rf"Best proportional fit: $\Delta d_{{eff}} \approx {slope:.2f}\,\psi(\mathcal{{K}})$",
        zorder=2,
    )

    for record in records:
        ax.annotate(
            record["label"],
            (record["psi"], record["delta_effdim"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            color=PALETTE["slate"],
        )

    ax.set_title("Theory Overlay: $\\psi(\\mathcal{K})$ vs. Empirical EffDim Gain", fontsize=14, weight="bold")
    ax.set_xlabel(r"Theoretical multi-horizon factor $\psi(\mathcal{K})$")
    ax.set_ylabel(r"Empirical effective-dimensionality gain $\Delta d_{\mathrm{eff}}$")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    metric_text = f"Pearson r = {pearson_r:.2f}\nSpearman $\\rho$ = {spearman_r:.2f}"
    ax.text(
        0.98,
        0.05,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["grid"]},
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Horizon diversity $|\mathcal{K}| / \max(\mathcal{K})$")

    fig.text(
        0.5,
        0.01,
        "Points are annotated by the evaluated horizon set. Marker area scales with the number of horizons "
        "and color encodes horizon diversity.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["muted"],
    )
    fig.savefig(FIG5_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate supplementary paper figures.")
    parser.add_argument("--skip-fig4", action="store_true", help="Only generate Figure 5.")
    args = parser.parse_args()

    plot_figure5()

    if args.skip_fig4:
        return

    cfg = Figure4Config()
    set_seed(cfg.seed)
    device = choose_device()
    model = train_quick_lfn(cfg, device)
    residuals = collect_residuals(model, cfg, device)
    plot_figure4(residuals)


if __name__ == "__main__":
    main()
