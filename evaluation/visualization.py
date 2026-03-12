"""
Visualization module for Latent Forecasting Network.

This module implements automated visualization tools for:
- Training curves (loss, perplexity, learning rate)
- Latent forecasting error analysis
- Latent Predictability Score curves
- Representation visualizations (PCA, t-SNE, trajectories)
"""

from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for server/CI
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Default style configuration
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}


class Visualizer:
    """
    Automated visualization generator for LFN experiments.

    Creates high-quality publication-ready plots for training curves,
    latent analysis, and representation visualizations.
    """

    def __init__(self, output_dir: str):
        """
        Initialize visualizer with output directory.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply plotting style
        plt.rcParams.update(PLOT_STYLE)

        logger.info(f"Visualizer initialized. Output directory: {self.output_dir}")

    def _save_figure(self, fig: Figure, filename: str) -> Path:
        """Save figure to output directory and close it."""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot: {filepath}")
        return filepath

    # ------------------------------------------------------------------ #
    # Training curve plots (7.2)
    # ------------------------------------------------------------------ #

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_token_losses: Optional[List[float]] = None,
        train_latent_losses: Optional[List[float]] = None,
        val_token_losses: Optional[List[float]] = None,
        val_latent_losses: Optional[List[float]] = None,
        learning_rates: Optional[List[float]] = None,
        perplexities: Optional[List[float]] = None,
    ) -> List[Path]:
        """
        Plot training curves: loss, perplexity, and learning rate.

        Args:
            train_losses: Training total loss per epoch
            val_losses: Validation total loss per epoch
            train_token_losses: Training token loss per epoch
            train_latent_losses: Training latent loss per epoch
            val_token_losses: Validation token loss per epoch
            val_latent_losses: Validation latent loss per epoch
            learning_rates: Learning rate per epoch
            perplexities: Validation perplexity per epoch

        Returns:
            List of saved file paths
        """
        saved = []
        epochs = list(range(1, len(train_losses) + 1))

        # --- Total loss curves ---
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
        ax.plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        saved.append(self._save_figure(fig, "training_loss.png"))

        # --- Token + latent loss breakdown ---
        if train_token_losses and train_latent_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(
                epochs,
                train_token_losses,
                label="Train Token Loss",
                marker="o",
                markersize=3,
            )
            if val_token_losses:
                ax1.plot(
                    epochs,
                    val_token_losses,
                    label="Val Token Loss",
                    marker="s",
                    markersize=3,
                )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Token Loss (Cross-Entropy)")
            ax1.set_title("Token Loss")
            ax1.legend()

            ax2.plot(
                epochs,
                train_latent_losses,
                label="Train Latent Loss",
                marker="o",
                markersize=3,
                color="tab:orange",
            )
            if val_latent_losses:
                ax2.plot(
                    epochs,
                    val_latent_losses,
                    label="Val Latent Loss",
                    marker="s",
                    markersize=3,
                    color="tab:red",
                )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Latent Loss (MSE)")
            ax2.set_title("Latent Forecasting Loss")
            ax2.legend()

            fig.suptitle("Loss Component Breakdown", fontsize=14)
            fig.tight_layout()
            saved.append(self._save_figure(fig, "loss_breakdown.png"))

        # --- Perplexity ---
        if perplexities:
            fig, ax = plt.subplots()
            ax.plot(
                epochs[: len(perplexities)],
                perplexities,
                label="Perplexity",
                marker="o",
                markersize=3,
                color="tab:green",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Perplexity")
            ax.set_title("Validation Perplexity")
            ax.legend()
            saved.append(self._save_figure(fig, "perplexity.png"))

        # --- Learning rate schedule ---
        if learning_rates:
            fig, ax = plt.subplots()
            ax.plot(range(len(learning_rates)), learning_rates, color="tab:purple")
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            saved.append(self._save_figure(fig, "learning_rate.png"))

        return saved

    # ------------------------------------------------------------------ #
    # Latent analysis plots (7.3)
    # ------------------------------------------------------------------ #

    def plot_latent_forecast_error(
        self,
        horizon_mse: Dict[int, float],
    ) -> Path:
        """
        Plot latent forecasting MSE error per horizon.

        Args:
            horizon_mse: Dict mapping horizon k to MSE value

        Returns:
            Path to saved plot
        """
        horizons = sorted(horizon_mse.keys())
        mse_values = [horizon_mse[h] for h in horizons]

        fig, ax = plt.subplots()
        bars = ax.bar(
            [str(h) for h in horizons],
            mse_values,
            color="tab:blue",
            alpha=0.8,
            edgecolor="black",
        )

        # Add value labels on bars
        for bar, val in zip(bars, mse_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Forecasting Horizon (k)")
        ax.set_ylabel("Mean Squared Error")
        ax.set_title("Latent Forecasting Error by Horizon")
        return self._save_figure(fig, "latent_forecast_error.png")

    def plot_latent_predictability_curve(
        self,
        lps_scores: Dict[int, float],
        error_bars: Optional[Dict[int, float]] = None,
    ) -> Path:
        """
        Plot Latent Predictability Score (LPS) vs forecasting horizon.

        Args:
            lps_scores: Dict mapping horizon k to LPS score
            error_bars: Optional dict mapping horizon k to std deviation

        Returns:
            Path to saved plot
        """
        horizons = sorted(lps_scores.keys())
        scores = [lps_scores[h] for h in horizons]
        errors = [error_bars.get(h, 0) for h in horizons] if error_bars else None

        fig, ax = plt.subplots()
        ax.errorbar(
            horizons,
            scores,
            yerr=errors,
            marker="o",
            capsize=5,
            capthick=1.5,
            color="tab:red",
            ecolor="tab:orange",
            label="LPS(k)",
        )
        ax.set_xlabel("Forecasting Horizon (k)")
        ax.set_ylabel("LPS (L2 Norm)")
        ax.set_title("Latent Predictability Score vs Horizon")
        ax.set_xticks(horizons)
        ax.legend()
        return self._save_figure(fig, "latent_predictability.png")

    # ------------------------------------------------------------------ #
    # Representation visualizations (7.4)
    # ------------------------------------------------------------------ #

    def plot_pca_projection(
        self,
        latents: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "2D PCA Projection of Latent States",
    ) -> Path:
        """
        2D PCA projection of latent representations.

        Args:
            latents: Latent representations [num_samples, latent_dim]
            labels: Optional color labels [num_samples]
            title: Plot title

        Returns:
            Path to saved plot
        """
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        projected = pca.fit_transform(latents)

        fig, ax = plt.subplots(figsize=(8, 8))

        scatter_kwargs: Dict[str, Any] = dict(alpha=0.5, s=8)
        if labels is not None:
            scatter = ax.scatter(
                projected[:, 0],
                projected[:, 1],
                c=labels,
                cmap="tab10",
                **scatter_kwargs,
            )
            fig.colorbar(scatter, ax=ax, label="Label")
        else:
            ax.scatter(projected[:, 0], projected[:, 1], **scatter_kwargs)

        explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
        ax.set_title(title)
        return self._save_figure(fig, "pca_projection.png")

    def plot_tsne_clustering(
        self,
        latents: np.ndarray,
        labels: Optional[np.ndarray] = None,
        perplexity: float = 30.0,
        title: str = "t-SNE Clustering of Latent States",
    ) -> Path:
        """
        t-SNE clustering visualization of latent representations.

        Args:
            latents: Latent representations [num_samples, latent_dim]
            labels: Optional color labels [num_samples]
            perplexity: t-SNE perplexity parameter
            title: Plot title

        Returns:
            Path to saved plot
        """
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        projected = tsne.fit_transform(latents)

        fig, ax = plt.subplots(figsize=(8, 8))

        scatter_kwargs: Dict[str, Any] = dict(alpha=0.5, s=8)
        if labels is not None:
            scatter = ax.scatter(
                projected[:, 0],
                projected[:, 1],
                c=labels,
                cmap="tab10",
                **scatter_kwargs,
            )
            fig.colorbar(scatter, ax=ax, label="Label")
        else:
            ax.scatter(projected[:, 0], projected[:, 1], **scatter_kwargs)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title)
        return self._save_figure(fig, "tsne_clustering.png")

    def plot_latent_trajectory(
        self,
        latents: np.ndarray,
        sequence_idx: int = 0,
        title: str = "Latent State Trajectory",
    ) -> Path:
        """
        Visualize how latent states evolve over a single sequence.

        Uses PCA to project a single sequence's latent states to 2D,
        then draws the trajectory with arrows showing direction.

        Args:
            latents: Latent representations [num_sequences, seq_len, latent_dim]
            sequence_idx: Index of the sequence to visualize
            title: Plot title

        Returns:
            Path to saved plot
        """
        from sklearn.decomposition import PCA

        # Select one sequence
        seq_latents = latents[sequence_idx]  # [seq_len, latent_dim]
        seq_len = seq_latents.shape[0]

        pca = PCA(n_components=2)
        projected = pca.fit_transform(seq_latents)  # [seq_len, 2]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Color by position in sequence
        colors = np.linspace(0, 1, seq_len)
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=colors,
            cmap="viridis",
            s=20,
            zorder=2,
        )
        fig.colorbar(scatter, ax=ax, label="Sequence Position (normalized)")

        # Draw trajectory lines
        ax.plot(
            projected[:, 0], projected[:, 1], "-", alpha=0.3, color="gray", zorder=1
        )

        # Mark start and end
        ax.scatter(
            *projected[0],
            marker="^",
            s=100,
            c="green",
            edgecolors="black",
            zorder=3,
            label="Start",
        )
        ax.scatter(
            *projected[-1],
            marker="v",
            s=100,
            c="red",
            edgecolors="black",
            zorder=3,
            label="End",
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend()
        return self._save_figure(fig, "latent_trajectory.png")

    # ------------------------------------------------------------------ #
    # Ablation / comparison plots
    # ------------------------------------------------------------------ #

    def plot_ablation_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "perplexity",
        title: Optional[str] = None,
    ) -> Path:
        """
        Plot comparison across ablation experiments.

        Args:
            results: Dict mapping experiment name to metrics dict
            metric: Metric name to compare
            title: Optional plot title

        Returns:
            Path to saved plot
        """
        names = list(results.keys())
        values = [results[n].get(metric, 0.0) for n in names]

        fig, ax = plt.subplots(figsize=(max(8, len(names)), 6))
        bars = ax.bar(
            range(len(names)), values, color="tab:blue", alpha=0.8, edgecolor="black"
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Ablation Comparison: {metric}")
        fig.tight_layout()
        return self._save_figure(fig, f"ablation_{metric}.png")

    # ------------------------------------------------------------------ #
    # Summary / orchestrator (7.5)
    # ------------------------------------------------------------------ #

    def generate_all_visualizations(
        self,
        training_history: Optional[Any] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        latents: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> List[Path]:
        """
        Generate all visualizations from available data.

        Args:
            training_history: TrainingHistory dataclass from trainer
            evaluation_results: Results dict from Evaluator.evaluate_model()
            latents: Collected latent representations [N, seq_len, latent_dim]
            labels: Token labels for coloring

        Returns:
            List of all saved file paths
        """
        saved: List[Path] = []

        # Training curves
        if training_history is not None:
            h = training_history
            perplexities = None
            if hasattr(h, "val_losses") and h.val_losses:
                perplexities = [float(np.exp(l)) for l in h.val_losses]
            curves = self.plot_training_curves(
                train_losses=h.train_losses,
                val_losses=h.val_losses,
                train_token_losses=getattr(h, "train_token_losses", None),
                train_latent_losses=getattr(h, "train_latent_losses", None),
                val_token_losses=getattr(h, "val_token_losses", None),
                val_latent_losses=getattr(h, "val_latent_losses", None),
                learning_rates=getattr(h, "learning_rates", None),
                perplexities=perplexities,
            )
            saved.extend(curves)

        # Evaluation metrics plots
        if evaluation_results is not None:
            if "horizon_mse" in evaluation_results:
                saved.append(
                    self.plot_latent_forecast_error(evaluation_results["horizon_mse"])
                )
            if "lps_scores" in evaluation_results:
                saved.append(
                    self.plot_latent_predictability_curve(
                        evaluation_results["lps_scores"]
                    )
                )

        # Representation visualizations
        if latents is not None:
            flat = (
                latents.reshape(-1, latents.shape[-1]) if latents.ndim == 3 else latents
            )
            # Limit samples for efficiency
            max_vis = min(5000, flat.shape[0])
            vis_latents = flat[:max_vis]
            vis_labels = labels[:max_vis] if labels is not None else None

            saved.append(self.plot_pca_projection(vis_latents, vis_labels))
            saved.append(self.plot_tsne_clustering(vis_latents, vis_labels))

            if latents.ndim == 3 and latents.shape[0] > 0:
                saved.append(self.plot_latent_trajectory(latents, sequence_idx=0))

        # Create summary figure
        if training_history is not None or evaluation_results is not None:
            saved.append(
                self._create_summary_figure(training_history, evaluation_results)
            )

        logger.info(f"Generated {len(saved)} visualizations in {self.output_dir}")
        return saved

    def _create_summary_figure(
        self,
        training_history: Optional[Any] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create a summary figure with key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Experiment Summary", fontsize=16, fontweight="bold")

        # Panel 1: Training loss
        if training_history and training_history.train_losses:
            ax = axes[0, 0]
            epochs = range(1, len(training_history.train_losses) + 1)
            ax.plot(epochs, training_history.train_losses, label="Train")
            ax.plot(epochs, training_history.val_losses, label="Val")
            ax.set_title("Total Loss")
            ax.set_xlabel("Epoch")
            ax.legend()
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "No training data",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )

        # Panel 2: Perplexity
        if training_history and training_history.val_losses:
            ax = axes[0, 1]
            perps = [float(np.exp(l)) for l in training_history.val_losses]
            ax.plot(
                range(1, len(perps) + 1),
                perps,
                color="tab:green",
                marker="o",
                markersize=3,
            )
            ax.set_title("Validation Perplexity")
            ax.set_xlabel("Epoch")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No perplexity data",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )

        # Panel 3: LPS scores
        if evaluation_results and "lps_scores" in evaluation_results:
            ax = axes[1, 0]
            lps = evaluation_results["lps_scores"]
            horizons = sorted(lps.keys())
            ax.plot(horizons, [lps[h] for h in horizons], "o-", color="tab:red")
            ax.set_title("Latent Predictability Score")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("LPS")
            ax.set_xticks(horizons)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No LPS data",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

        # Panel 4: Key metrics table
        ax = axes[1, 1]
        ax.axis("off")
        table_data = []
        if evaluation_results:
            if "perplexity" in evaluation_results:
                table_data.append(
                    ["Perplexity", f"{evaluation_results['perplexity']:.2f}"]
                )
            if "token_loss" in evaluation_results:
                table_data.append(
                    ["Token Loss", f"{evaluation_results['token_loss']:.4f}"]
                )
            if "accuracy" in evaluation_results:
                table_data.append(["Accuracy", f"{evaluation_results['accuracy']:.2%}"])
        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=["Metric", "Value"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.5)
            ax.set_title("Key Metrics")
        else:
            ax.text(
                0.5, 0.5, "No metrics", ha="center", va="center", transform=ax.transAxes
            )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_figure(fig, "experiment_summary.png")
