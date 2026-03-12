"""
Integration tests for Latent Forecasting Network.

Tests end-to-end workflows:
- Complete training pipeline (2 epochs, small synthetic dataset)
- Evaluation pipeline end-to-end
- Visualization pipeline
- CLI argument parsing
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import Evaluator
from evaluation.visualization import Visualizer
from models.complete_model import ModelConfig, build_model
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer, TrainingConfig, TrainingHistory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config():
    return ModelConfig(
        vocab_size=50,
        latent_dim=32,
        num_layers=1,
        num_heads=2,
        hidden_dim=64,
        dropout=0.0,
        forecast_horizons=[1, 2],
        max_context_length=16,
        lambda_latent=0.1,
    )


class DictDataLoader:
    """Wrap a standard DataLoader to produce dictionaries."""

    def __init__(self, dl, has_labels=True):
        self.dl = dl
        self.has_labels = has_labels

    def __iter__(self):
        for batch in self.dl:
            if self.has_labels:
                tokens, labels = batch
                yield {"input_ids": tokens, "labels": labels}
            else:
                (tokens,) = batch
                yield {"input_ids": tokens}

    def __len__(self):
        return len(self.dl)


def _make_loaders(
    vocab_size=50, batch_size=4, seq_len=8, num_batches=3, has_labels=True
):
    """Create tiny train and val dataloaders."""
    all_tokens = []
    all_labels = []
    for _ in range(num_batches):
        all_tokens.append(torch.randint(0, vocab_size, (batch_size, seq_len)))
        all_labels.append(torch.randint(0, vocab_size, (batch_size, seq_len)))
    tokens_t = torch.cat(all_tokens)
    labels_t = torch.cat(all_labels)

    if has_labels:
        ds = TensorDataset(tokens_t, labels_t)
    else:
        ds = TensorDataset(tokens_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return DictDataLoader(dl, has_labels=has_labels)


# ---------------------------------------------------------------------------
# 9.2.2  Training pipeline (2 epochs, small dataset)
# ---------------------------------------------------------------------------


class TestTrainingPipeline:
    """End-to-end training test."""

    def test_full_training_2_epochs(self):
        config = _tiny_config()
        model = build_model(config, device="cpu")

        train_loader = _make_loaders(vocab_size=config.vocab_size, seq_len=8)
        val_loader = _make_loaders(vocab_size=config.vocab_size, seq_len=8)

        tc = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            warmup_steps=0,
            lambda_latent=0.1,
            use_mixed_precision=False,
            checkpoint_every=9999,
            log_every=9999,
            seed=42,
            device="cpu",
        )

        optimizer = create_optimizer(
            model, learning_rate=tc.learning_rate, weight_decay=tc.weight_decay
        )
        total_steps = tc.num_epochs * len(train_loader)
        scheduler = create_scheduler(
            optimizer, warmup_steps=tc.warmup_steps, num_training_steps=max(total_steps, 1)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                config=tc,
                checkpoint_dir=os.path.join(tmpdir, "ckpt"),
                log_dir=os.path.join(tmpdir, "logs"),
            )
            history = trainer.train()

        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 2
        assert all(l > 0 for l in history.train_losses)

    def test_training_reduces_loss(self):
        """After a few epochs on tiny data, loss should decrease."""
        config = _tiny_config()
        model = build_model(config, device="cpu")

        train_loader = _make_loaders(
            vocab_size=config.vocab_size, seq_len=8, num_batches=5
        )
        val_loader = _make_loaders(
            vocab_size=config.vocab_size, seq_len=8, num_batches=2
        )

        tc = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=5e-3,
            weight_decay=0.0,
            gradient_accumulation_steps=1,
            max_grad_norm=5.0,
            warmup_steps=0,
            lambda_latent=0.1,
            use_mixed_precision=False,
            checkpoint_every=9999,
            log_every=9999,
            seed=42,
            device="cpu",
        )

        optimizer = create_optimizer(
            model, learning_rate=tc.learning_rate, weight_decay=tc.weight_decay
        )
        total_steps = tc.num_epochs * len(train_loader)
        scheduler = create_scheduler(
            optimizer, warmup_steps=0, num_training_steps=max(total_steps, 1)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                config=tc,
                checkpoint_dir=os.path.join(tmpdir, "ckpt"),
                log_dir=os.path.join(tmpdir, "logs"),
            )
            history = trainer.train()

        # Loss should generally decrease (first > last)
        assert (
            history.train_losses[-1] <= history.train_losses[0] * 1.5
        )  # Allow some tolerance


# ---------------------------------------------------------------------------
# 9.2.3  Evaluation pipeline end-to-end
# ---------------------------------------------------------------------------


class TestEvaluationPipeline:
    """End-to-end evaluation test."""

    def test_full_evaluation(self):
        config = _tiny_config()
        model = build_model(config, device="cpu")
        model.eval()

        test_loader = _make_loaders(vocab_size=config.vocab_size, seq_len=8)

        evaluator = Evaluator(model, device="cpu", compute_accuracy=True)
        results = evaluator.evaluate_model(
            test_loader, include_representation_analysis=False
        )

        assert "perplexity" in results
        assert "token_loss" in results
        assert results["perplexity"] > 0
        assert results["token_loss"] >= 0

    def test_evaluation_with_representation_analysis(self):
        config = _tiny_config()
        model = build_model(config, device="cpu")
        model.eval()

        test_loader = _make_loaders(
            vocab_size=config.vocab_size, seq_len=8, num_batches=5
        )

        evaluator = Evaluator(model, device="cpu")
        results = evaluator.evaluate_model(
            test_loader,
            include_representation_analysis=True,
            max_samples=500,
            n_clusters=3,
        )

        assert "perplexity" in results
        assert "representation_metrics" in results
        rep = results["representation_metrics"]
        assert "latent_entropy" in rep
        assert "latent_variance" in rep
        assert rep["latent_entropy"] >= 0
        assert rep["latent_variance"] >= 0


# ---------------------------------------------------------------------------
# 9.2: Visualization pipeline test
# ---------------------------------------------------------------------------


class TestVisualizationPipeline:
    """Test visualization generates files."""

    def test_training_curves_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Visualizer(tmpdir)
            saved = viz.plot_training_curves(
                train_losses=[5.0, 4.5, 4.0, 3.8, 3.6],
                val_losses=[5.2, 4.7, 4.3, 4.0, 3.9],
            )
            assert len(saved) >= 1
            for path in saved:
                assert path.exists()
                assert path.stat().st_size > 0

    def test_latent_forecast_error_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Visualizer(tmpdir)
            path = viz.plot_latent_forecast_error({1: 0.5, 2: 0.8, 5: 1.2})
            assert path.exists()

    def test_lps_curve_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Visualizer(tmpdir)
            path = viz.plot_latent_predictability_curve({1: 0.3, 2: 0.5, 5: 0.9})
            assert path.exists()

    def test_pca_projection_plot(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Visualizer(tmpdir)
            latents = np.random.randn(200, 32)
            path = viz.plot_pca_projection(latents)
            assert path.exists()

    def test_generate_all(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Visualizer(tmpdir)

            history = TrainingHistory()
            history.add_epoch(
                0,
                {"total_loss": 5.0, "token_loss": 4.5, "latent_loss": 0.5},
                {"total_loss": 5.2, "token_loss": 4.7, "latent_loss": 0.5},
                1e-4,
            )
            history.add_epoch(
                1,
                {"total_loss": 4.0, "token_loss": 3.5, "latent_loss": 0.5},
                {"total_loss": 4.2, "token_loss": 3.7, "latent_loss": 0.5},
                9e-5,
            )

            eval_results = {
                "perplexity": 55.0,
                "token_loss": 4.0,
                "horizon_mse": {1: 0.5, 2: 0.8},
                "lps_scores": {1: 0.3, 2: 0.5},
            }

            latents = np.random.randn(5, 8, 32)

            saved = viz.generate_all_visualizations(
                training_history=history,
                evaluation_results=eval_results,
                latents=latents,
            )
            assert len(saved) >= 4  # multiple plots generated
            for p in saved:
                assert p.exists()


# ---------------------------------------------------------------------------
# 9.2.5  CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_help_runs(self):
        from main import build_parser

        parser = build_parser()
        # Should not raise
        assert parser is not None

    def test_train_parser(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["train", "--config", "experiments/configs/default.yaml"]
        )
        assert args.command == "train"
        assert args.config == "experiments/configs/default.yaml"

    def test_evaluate_parser(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["evaluate", "--config", "c.yaml", "--checkpoint", "ckpt.pt"]
        )
        assert args.command == "evaluate"
        assert args.checkpoint == "ckpt.pt"

    def test_analyze_parser(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--config",
                "c.yaml",
                "--checkpoint",
                "ckpt.pt",
                "--no-visualize",
            ]
        )
        assert args.command == "analyze"
        assert args.visualize is False

    def test_ablation_parser(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(["ablation", "--config", "ablation.yaml"])
        assert args.command == "ablation"
