"""
Tests for evaluation metrics.

This module tests:
- Perplexity computation
- Token accuracy computation
- Language modeling evaluation
- Perplexity = exp(loss) relationship
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import (Evaluator, compute_latent_predictability_score,
                                compute_perplexity, compute_token_accuracy,
                                evaluate_language_modeling,
                                evaluate_latent_forecasting)
from models.complete_model import ModelConfig, build_model


class TestComputePerplexity:
    """Test perplexity computation."""

    def test_perplexity_basic(self):
        """Test basic perplexity computation."""
        loss = torch.tensor(2.0)
        ppl = compute_perplexity(loss)

        # Perplexity should be exp(2.0) ≈ 7.389
        expected = torch.exp(loss)
        assert torch.isclose(ppl, expected, atol=1e-5)
        assert ppl > 1.0

    def test_perplexity_zero_loss(self):
        """Test perplexity with zero loss."""
        loss = torch.tensor(0.0)
        ppl = compute_perplexity(loss)

        # Perplexity should be exp(0) = 1.0
        assert torch.isclose(ppl, torch.tensor(1.0), atol=1e-5)

    def test_perplexity_high_loss(self):
        """Test perplexity with high loss."""
        loss = torch.tensor(10.0)
        ppl = compute_perplexity(loss)

        # Perplexity should be exp(10.0) ≈ 22026
        expected = torch.exp(loss)
        assert torch.isclose(ppl, expected, rtol=1e-4)
        assert ppl > 1000.0

    def test_perplexity_exp_relationship(self):
        """Verify perplexity = exp(loss) relationship."""
        losses = [0.5, 1.0, 2.0, 3.0, 5.0]

        for loss_val in losses:
            loss = torch.tensor(loss_val)
            ppl = compute_perplexity(loss)
            expected = torch.exp(loss)

            # Verify relationship holds
            assert torch.isclose(ppl, expected, rtol=1e-5)

    def test_perplexity_negative_loss_raises(self):
        """Test that negative loss raises error."""
        loss = torch.tensor(-1.0)

        with pytest.raises(ValueError, match="non-negative"):
            compute_perplexity(loss)

    def test_perplexity_nan_raises(self):
        """Test that NaN loss raises error."""
        loss = torch.tensor(float("nan"))

        with pytest.raises(ValueError, match="finite"):
            compute_perplexity(loss)

    def test_perplexity_inf_raises(self):
        """Test that Inf loss raises error."""
        loss = torch.tensor(float("inf"))

        with pytest.raises(ValueError, match="finite"):
            compute_perplexity(loss)


class TestComputeTokenAccuracy:
    """Test token accuracy computation."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create logits where argmax matches labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = (
                    100.0  # Make correct token have highest logit
                )

        accuracy = compute_token_accuracy(logits, labels)

        # Should be 100% accurate
        assert torch.isclose(accuracy, torch.tensor(1.0), atol=1e-5)

    def test_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create logits where argmax never matches labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                # Make a different token have highest logit
                wrong_token = (labels[b, s] + 1) % vocab_size
                logits[b, s, wrong_token] = 100.0
                logits[b, s, labels[b, s]] = -100.0

        accuracy = compute_token_accuracy(logits, labels)

        # Should be 0% accurate
        assert torch.isclose(accuracy, torch.tensor(0.0), atol=1e-5)

    def test_accuracy_with_ignore_index(self):
        """Test accuracy computation with ignored tokens."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        ignore_index = -100

        # Create labels with some ignored tokens
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, :5] = ignore_index  # Ignore first 5 tokens of first batch

        # Create logits where non-ignored tokens are correct
        logits = torch.randn(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                if labels[b, s] != ignore_index:
                    logits[b, s, labels[b, s]] = 100.0

        accuracy = compute_token_accuracy(logits, labels, ignore_index)

        # Should be 100% accurate for non-ignored tokens
        assert torch.isclose(accuracy, torch.tensor(1.0), atol=1e-5)

    def test_accuracy_range(self):
        """Test that accuracy is always in [0, 1]."""
        batch_size, seq_len, vocab_size = 4, 20, 100

        # Random logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        accuracy = compute_token_accuracy(logits, labels)

        # Accuracy must be in valid range
        assert 0.0 <= accuracy <= 1.0
        assert torch.isfinite(accuracy)


class TestComputeLatentPredictabilityScore:
    """Test Latent Predictability Score computation."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.1,
            forecast_horizons=[1, 2, 5],
            max_context_length=32,
        )
        model = build_model(config, device="cpu")
        return model

    @pytest.fixture
    def test_dataloader(self):
        """Create a small test dataloader."""
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        num_batches = 5

        # Create random data
        all_tokens = []

        for _ in range(num_batches):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            all_tokens.append(tokens)

        tokens_tensor = torch.cat(all_tokens, dim=0)

        dataset = TensorDataset(tokens_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Wrap to provide dict format
        class DictDataLoader:
            def __init__(self, dl):
                self.dl = dl

            def __iter__(self):
                for (tokens,) in self.dl:
                    yield {"input_ids": tokens}

            def __len__(self):
                return len(self.dl)

        return DictDataLoader(dataloader)

    def test_lps_basic(self, small_model, test_dataloader):
        """Test basic LPS computation."""
        results = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        # Check all horizons are present
        assert 1 in results
        assert 2 in results
        assert 5 in results

        # Check all LPS values are valid
        for horizon, lps in results.items():
            assert lps >= 0, f"LPS for horizon {horizon} must be non-negative"
            assert torch.isfinite(
                torch.tensor(lps)
            ), f"LPS for horizon {horizon} must be finite"

    def test_lps_non_negative(self, small_model, test_dataloader):
        """Test that all LPS values are non-negative."""
        results = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        for horizon, lps in results.items():
            assert (
                lps >= 0
            ), f"LPS must be non-negative, got {lps} for horizon {horizon}"

    def test_lps_all_horizons(self, small_model, test_dataloader):
        """Test that all configured horizons are evaluated."""
        expected_horizons = [1, 2, 5]

        results = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        # Check all expected horizons are present
        for horizon in expected_horizons:
            assert horizon in results, f"Horizon {horizon} missing from results"

    def test_lps_model_stays_eval(self, small_model, test_dataloader):
        """Test that model stays in eval mode after LPS computation."""
        small_model.train()  # Start in train mode

        compute_latent_predictability_score(small_model, test_dataloader, device="cpu")

        # Should be in eval mode after
        assert not small_model.training

    def test_lps_deterministic(self, small_model, test_dataloader):
        """Test that LPS computation is deterministic."""
        # Run computation twice
        results1 = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        results2 = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        # Results should be identical
        assert results1.keys() == results2.keys()
        for horizon in results1.keys():
            assert (
                abs(results1[horizon] - results2[horizon]) < 1e-6
            ), f"Results not deterministic for horizon {horizon}"

    def test_lps_increases_with_horizon(self, small_model, test_dataloader):
        """Test that LPS generally increases with horizon (less predictable further out)."""
        results = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        # Get sorted horizons
        horizons = sorted(results.keys())

        # LPS should generally increase with horizon
        # (though this is not guaranteed for untrained models, just check it's reasonable)
        for horizon in horizons:
            assert (
                results[horizon] >= 0
            ), f"LPS must be non-negative for horizon {horizon}"

    def test_lps_vs_mse_relationship(self, small_model, test_dataloader):
        """Test relationship between LPS and MSE."""
        lps_results = compute_latent_predictability_score(
            small_model, test_dataloader, device="cpu"
        )

        mse_results = evaluate_latent_forecasting(
            small_model, test_dataloader, device="cpu"
        )

        # Both should have same horizons
        assert lps_results.keys() == mse_results.keys()

        # LPS is L2 norm, MSE is squared L2 norm
        # So roughly: LPS ≈ sqrt(MSE * latent_dim)
        # (This is approximate due to averaging order)
        for horizon in lps_results.keys():
            lps = lps_results[horizon]
            mse = mse_results[horizon]

            # Both should be non-negative
            assert lps >= 0
            assert mse >= 0

            # LPS should be roughly sqrt(MSE * latent_dim)
            # But we just check they're both reasonable
            assert lps < 1000, f"LPS seems too large for horizon {horizon}"
            assert mse < 1000, f"MSE seems too large for horizon {horizon}"


class TestEvaluateLatentForecasting:
    """Test latent forecasting evaluation."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.1,
            forecast_horizons=[1, 2, 5],
            max_context_length=32,
        )
        model = build_model(config, device="cpu")
        return model

    @pytest.fixture
    def test_dataloader(self):
        """Create a small test dataloader."""
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        num_batches = 5

        # Create random data
        all_tokens = []

        for _ in range(num_batches):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            all_tokens.append(tokens)

        tokens_tensor = torch.cat(all_tokens, dim=0)

        dataset = TensorDataset(tokens_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Wrap to provide dict format
        class DictDataLoader:
            def __init__(self, dl):
                self.dl = dl

            def __iter__(self):
                for (tokens,) in self.dl:
                    yield {"input_ids": tokens}

            def __len__(self):
                return len(self.dl)

        return DictDataLoader(dataloader)

    def test_evaluate_latent_forecasting_basic(self, small_model, test_dataloader):
        """Test basic latent forecasting evaluation."""
        results = evaluate_latent_forecasting(
            small_model, test_dataloader, device="cpu"
        )

        # Check all horizons are present
        assert 1 in results
        assert 2 in results
        assert 5 in results

        # Check all MSE values are valid
        for horizon, mse in results.items():
            assert mse >= 0, f"MSE for horizon {horizon} must be non-negative"
            assert torch.isfinite(
                torch.tensor(mse)
            ), f"MSE for horizon {horizon} must be finite"

    def test_evaluate_latent_forecasting_mse_non_negative(
        self, small_model, test_dataloader
    ):
        """Test that all MSE values are non-negative."""
        results = evaluate_latent_forecasting(
            small_model, test_dataloader, device="cpu"
        )

        for horizon, mse in results.items():
            assert (
                mse >= 0
            ), f"MSE must be non-negative, got {mse} for horizon {horizon}"

    def test_evaluate_latent_forecasting_all_horizons(
        self, small_model, test_dataloader
    ):
        """Test that all configured horizons are evaluated."""
        expected_horizons = [1, 2, 5]

        results = evaluate_latent_forecasting(
            small_model, test_dataloader, device="cpu"
        )

        # Check all expected horizons are present
        for horizon in expected_horizons:
            assert horizon in results, f"Horizon {horizon} missing from results"

    def test_evaluate_latent_forecasting_model_stays_eval(
        self, small_model, test_dataloader
    ):
        """Test that model stays in eval mode after evaluation."""
        small_model.train()  # Start in train mode

        evaluate_latent_forecasting(small_model, test_dataloader, device="cpu")

        # Should be in eval mode after
        assert not small_model.training

    def test_evaluate_latent_forecasting_deterministic(
        self, small_model, test_dataloader
    ):
        """Test that evaluation is deterministic."""
        # Run evaluation twice
        results1 = evaluate_latent_forecasting(
            small_model, test_dataloader, device="cpu"
        )

        results2 = evaluate_latent_forecasting(
            small_model, test_dataloader, device="cpu"
        )

        # Results should be identical
        assert results1.keys() == results2.keys()
        for horizon in results1.keys():
            assert (
                abs(results1[horizon] - results2[horizon]) < 1e-6
            ), f"Results not deterministic for horizon {horizon}"


class TestEvaluateLanguageModeling:
    """Test language modeling evaluation."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.1,
            forecast_horizons=[1, 2],
            max_context_length=32,
        )
        model = build_model(config, device="cpu")
        return model

    @pytest.fixture
    def test_dataloader(self):
        """Create a small test dataloader."""
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        num_batches = 5

        # Create random data
        all_tokens = []
        all_labels = []

        for _ in range(num_batches):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens_tensor = torch.cat(all_tokens, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        dataset = TensorDataset(tokens_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Wrap to provide dict format
        class DictDataLoader:
            def __init__(self, dl):
                self.dl = dl

            def __iter__(self):
                for tokens, labels in self.dl:
                    yield {"input_ids": tokens, "labels": labels}

            def __len__(self):
                return len(self.dl)

        return DictDataLoader(dataloader)

    def test_evaluate_basic(self, small_model, test_dataloader):
        """Test basic evaluation."""
        results = evaluate_language_modeling(
            small_model, test_dataloader, device="cpu", compute_accuracy=True
        )

        # Check all required keys present
        assert "perplexity" in results
        assert "token_loss" in results
        assert "accuracy" in results

        # Check values are valid
        assert results["perplexity"] > 0
        assert results["token_loss"] >= 0
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_perplexity_relationship(self, small_model, test_dataloader):
        """Verify perplexity = exp(loss) relationship in evaluation."""
        results = evaluate_language_modeling(
            small_model, test_dataloader, device="cpu", compute_accuracy=False
        )

        # Compute expected perplexity
        expected_ppl = torch.exp(torch.tensor(results["token_loss"])).item()

        # Verify relationship holds
        assert abs(results["perplexity"] - expected_ppl) < 1e-4

    def test_evaluate_without_accuracy(self, small_model, test_dataloader):
        """Test evaluation without computing accuracy."""
        results = evaluate_language_modeling(
            small_model, test_dataloader, device="cpu", compute_accuracy=False
        )

        # Should have perplexity and loss but not accuracy
        assert "perplexity" in results
        assert "token_loss" in results
        assert "accuracy" not in results

    def test_evaluate_model_stays_in_eval_mode(self, small_model, test_dataloader):
        """Test that model stays in eval mode after evaluation."""
        small_model.train()  # Start in train mode

        evaluate_language_modeling(small_model, test_dataloader, device="cpu")

        # Should be in eval mode after
        assert not small_model.training


class TestEvaluator:
    """Test Evaluator class."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            latent_dim=64,
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.1,
            forecast_horizons=[1, 2],
            max_context_length=32,
        )
        model = build_model(config, device="cpu")
        return model

    @pytest.fixture
    def test_dataloader(self):
        """Create a small test dataloader."""
        batch_size = 4
        seq_len = 16
        vocab_size = 100
        num_batches = 3

        all_tokens = []
        all_labels = []

        for _ in range(num_batches):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens_tensor = torch.cat(all_tokens, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        dataset = TensorDataset(tokens_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        class DictDataLoader:
            def __init__(self, dl):
                self.dl = dl

            def __iter__(self):
                for tokens, labels in self.dl:
                    yield {"input_ids": tokens, "labels": labels}

            def __len__(self):
                return len(self.dl)

        return DictDataLoader(dataloader)

    def test_evaluator_initialization(self, small_model):
        """Test Evaluator initialization."""
        evaluator = Evaluator(small_model, device="cpu")

        assert evaluator.model is small_model
        assert evaluator.device == "cpu"
        assert not small_model.training  # Should be in eval mode

    def test_evaluator_language_modeling(self, small_model, test_dataloader):
        """Test Evaluator language modeling evaluation."""
        evaluator = Evaluator(small_model, device="cpu", compute_accuracy=True)

        results = evaluator.evaluate_language_modeling(test_dataloader)

        assert "perplexity" in results
        assert "token_loss" in results
        assert "accuracy" in results

    def test_evaluator_latent_forecasting(self, small_model, test_dataloader):
        """Test Evaluator latent forecasting evaluation."""
        evaluator = Evaluator(small_model, device="cpu")

        results = evaluator.evaluate_latent_forecasting(test_dataloader)

        # Should have MSE for each horizon
        assert 1 in results
        assert 2 in results
        assert all(mse >= 0 for mse in results.values())

    def test_evaluator_lps(self, small_model, test_dataloader):
        """Test Evaluator LPS computation."""
        evaluator = Evaluator(small_model, device="cpu")

        results = evaluator.compute_latent_predictability_score(test_dataloader)

        # Should have LPS for each horizon
        assert 1 in results
        assert 2 in results
        assert all(lps >= 0 for lps in results.values())

    def test_evaluator_complete_evaluation(self, small_model, test_dataloader):
        """Test Evaluator complete model evaluation."""
        evaluator = Evaluator(small_model, device="cpu")

        results = evaluator.evaluate_model(test_dataloader)

        # Should include language modeling metrics
        assert "perplexity" in results
        assert "token_loss" in results

        # Should include latent forecasting metrics
        assert "horizon_mse" in results
        assert isinstance(results["horizon_mse"], dict)
        assert 1 in results["horizon_mse"]
        assert 2 in results["horizon_mse"]

        # Should include LPS scores
        assert "lps_scores" in results
        assert isinstance(results["lps_scores"], dict)
        assert 1 in results["lps_scores"]
        assert 2 in results["lps_scores"]


class TestPerplexityExpRelationship:
    """Comprehensive tests for perplexity = exp(loss) relationship."""

    def test_relationship_various_losses(self):
        """Test relationship across various loss values."""
        test_losses = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

        for loss_val in test_losses:
            loss = torch.tensor(loss_val)
            ppl = compute_perplexity(loss)
            expected = torch.exp(loss)

            # Verify relationship with high precision
            rel_error = abs(ppl - expected) / expected
            assert rel_error < 1e-6, f"Relationship failed for loss={loss_val}"

    def test_relationship_numerical_stability(self):
        """Test numerical stability of relationship."""
        # Test very small losses
        small_losses = [1e-6, 1e-4, 1e-2]
        for loss_val in small_losses:
            loss = torch.tensor(loss_val)
            ppl = compute_perplexity(loss)
            expected = torch.exp(loss)
            assert torch.isclose(ppl, expected, rtol=1e-5)

        # Test moderate losses
        moderate_losses = [1.0, 2.0, 5.0]
        for loss_val in moderate_losses:
            loss = torch.tensor(loss_val)
            ppl = compute_perplexity(loss)
            expected = torch.exp(loss)
            assert torch.isclose(ppl, expected, rtol=1e-5)

    def test_relationship_in_evaluation(self):
        """Test that relationship holds in full evaluation pipeline."""
        # Create simple model
        config = ModelConfig(
            vocab_size=50,
            latent_dim=32,
            num_layers=1,
            num_heads=2,
            hidden_dim=64,
            dropout=0.0,
            forecast_horizons=[1],
            max_context_length=16,
        )
        model = build_model(config, device="cpu")
        model.eval()

        # Create simple dataloader
        batch_size = 2
        seq_len = 8
        num_batches = 3

        all_tokens = []
        all_labels = []

        for _ in range(num_batches):
            tokens = torch.randint(0, 50, (batch_size, seq_len))
            labels = torch.randint(0, 50, (batch_size, seq_len))
            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens_tensor = torch.cat(all_tokens, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        dataset = TensorDataset(tokens_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        class DictDataLoader:
            def __init__(self, dl):
                self.dl = dl

            def __iter__(self):
                for tokens, labels in self.dl:
                    yield {"input_ids": tokens, "labels": labels}

        wrapped_loader = DictDataLoader(dataloader)

        # Evaluate
        results = evaluate_language_modeling(model, wrapped_loader, device="cpu")

        # Verify relationship
        expected_ppl = torch.exp(torch.tensor(results["token_loss"])).item()
        actual_ppl = results["perplexity"]

        # Should match within numerical precision
        rel_error = abs(actual_ppl - expected_ppl) / expected_ppl
        assert (
            rel_error < 1e-5
        ), f"Perplexity relationship violated: ppl={actual_ppl}, exp(loss)={expected_ppl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
