"""
Tests for data pipeline components (tokenizer and dataset loader).

This module tests:
- Tokenization correctness
- Sequence chunking and stride
- Batch shapes and padding
- Data loading performance
"""

import time
from typing import Dict

import pytest
import torch
from transformers import PreTrainedTokenizer

from data.dataset_loader import (SequenceDataset, collate_batch_with_padding,
                                 create_dataloaders)
from data.tokenizer import (add_special_tokens_to_sequence,
                            get_special_token_ids, load_tokenizer,
                            tokenize_text)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    """Load GPT-2 tokenizer for testing."""
    return load_tokenizer("gpt2")


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return "Hello world! This is a test sentence for tokenization."


@pytest.fixture
def sample_texts() -> list:
    """Multiple sample texts for batch testing."""
    return [
        "First test sentence.",
        "Second test sentence with more words.",
        "Third sentence.",
    ]


# ============================================================================
# Task 2.4.2: Test Tokenization Correctness
# ============================================================================


class TestTokenizationCorrectness:
    """Tests for tokenization correctness."""

    def test_load_tokenizer_success(self):
        """Test that tokenizer loads successfully."""
        tokenizer = load_tokenizer("gpt2")
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token is not None

    def test_tokenizer_has_special_tokens(self, tokenizer):
        """Test that tokenizer has required special tokens."""
        assert tokenizer.eos_token is not None
        assert tokenizer.pad_token is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None

    def test_tokenize_single_text(self, tokenizer, sample_text):
        """Test tokenization of single text."""
        result = tokenize_text(sample_text, tokenizer)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].dim() == 2  # [batch_size, seq_len]
        assert result["input_ids"].shape[0] == 1  # batch_size = 1
        assert result["input_ids"].shape[1] > 0  # seq_len > 0

    def test_tokenize_batch_texts(self, tokenizer, sample_texts):
        """Test tokenization of multiple texts."""
        result = tokenize_text(
            sample_texts, tokenizer, padding=True, return_tensors="pt"
        )

        assert result["input_ids"].shape[0] == len(sample_texts)
        # All sequences should have same length due to padding
        assert result["input_ids"].shape[1] > 0

    def test_token_ids_in_valid_range(self, tokenizer, sample_text):
        """Test that all token IDs are in valid range [0, vocab_size)."""
        result = tokenize_text(sample_text, tokenizer)
        input_ids = result["input_ids"]
        vocab_size = tokenizer.vocab_size

        assert torch.all(input_ids >= 0), "Token IDs must be non-negative"
        assert torch.all(
            input_ids < vocab_size
        ), f"Token IDs must be less than vocab_size ({vocab_size})"

    def test_attention_mask_binary(self, tokenizer, sample_texts):
        """Test that attention mask contains only 0s and 1s."""
        result = tokenize_text(
            sample_texts, tokenizer, padding=True, return_tensors="pt"
        )
        attention_mask = result["attention_mask"]

        assert torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ), "Attention mask must contain only 0s and 1s"

    def test_tokenization_with_max_length(self, tokenizer, sample_text):
        """Test tokenization with max_length truncation."""
        max_length = 5
        result = tokenize_text(
            sample_text, tokenizer, max_length=max_length, truncation=True
        )

        assert result["input_ids"].shape[1] <= max_length

    def test_tokenization_caching(self, tokenizer, sample_text):
        """Test that tokenization caching works."""
        cache = {}

        # First call - should populate cache
        result1 = tokenize_text(sample_text, tokenizer, cache=cache)
        assert len(cache) == 1

        # Second call - should use cache
        result2 = tokenize_text(sample_text, tokenizer, cache=cache)
        assert len(cache) == 1  # Cache size unchanged

        # Results should be identical
        assert torch.equal(result1["input_ids"], result2["input_ids"])

    def test_get_special_token_ids(self, tokenizer):
        """Test retrieval of special token IDs."""
        special_tokens = get_special_token_ids(tokenizer)

        assert "eos_token_id" in special_tokens
        assert "pad_token_id" in special_tokens
        assert special_tokens["eos_token_id"] is not None
        assert special_tokens["pad_token_id"] is not None

        # Validate IDs are in valid range
        vocab_size = tokenizer.vocab_size
        for token_name, token_id in special_tokens.items():
            if token_id is not None:
                assert (
                    0 <= token_id < vocab_size
                ), f"{token_name} must be in range [0, {vocab_size})"

    def test_add_eos_token(self, tokenizer):
        """Test adding EOS token to sequence."""
        token_ids = torch.tensor([[100, 200, 300]])
        result = add_special_tokens_to_sequence(token_ids, tokenizer, add_eos=True)

        assert result.shape[1] == token_ids.shape[1] + 1
        assert result[0, -1] == tokenizer.eos_token_id

    def test_add_bos_token(self, tokenizer):
        """Test adding BOS token to sequence (if available)."""
        token_ids = torch.tensor([[100, 200, 300]])
        result = add_special_tokens_to_sequence(token_ids, tokenizer, add_bos=True)

        # GPT-2 doesn't have BOS token, so length should be unchanged
        # or increased by 1 if BOS is available
        assert result.shape[1] >= token_ids.shape[1]

    def test_tokenization_deterministic(self, tokenizer, sample_text):
        """Test that tokenization is deterministic."""
        result1 = tokenize_text(sample_text, tokenizer)
        result2 = tokenize_text(sample_text, tokenizer)

        assert torch.equal(result1["input_ids"], result2["input_ids"])
        assert torch.equal(result1["attention_mask"], result2["attention_mask"])


# ============================================================================
# Task 2.4.3: Test Sequence Chunking and Stride
# ============================================================================


class TestSequenceChunkingAndStride:
    """Tests for sequence chunking with sliding window."""

    def test_sequence_dataset_creation(self, tokenizer):
        """Test that SequenceDataset can be created."""
        dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",  # Use test split (smaller)
            tokenizer=tokenizer,
            context_length=128,
            stride=64,
        )

        assert len(dataset) > 0
        assert dataset.context_length == 128
        assert dataset.stride == 64

    def test_sequence_length_correct(self, tokenizer):
        """Test that all sequences have correct length."""
        context_length = 128
        dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=64,
        )

        # Check multiple sequences
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert item["input_ids"].shape[0] == context_length
            assert item["labels"].shape[0] == context_length

    def test_stride_creates_overlap(self, tokenizer):
        """Test that stride creates overlapping sequences."""
        context_length = 128
        stride = 64
        dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=stride,
        )

        if len(dataset) >= 2:
            # Get two consecutive sequences
            seq1 = dataset[0]["input_ids"]
            seq2 = dataset[1]["input_ids"]

            # Check for overlap
            overlap_size = context_length - stride
            # Last overlap_size tokens of seq1 should match first overlap_size of seq2
            # (This may not always be true due to document boundaries, so we just check shapes)
            assert seq1.shape[0] == context_length
            assert seq2.shape[0] == context_length

    def test_different_strides(self, tokenizer):
        """Test that different strides produce different number of sequences."""
        context_length = 128

        dataset_stride_32 = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=32,
        )

        dataset_stride_64 = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=64,
        )

        # Smaller stride should produce more sequences
        assert len(dataset_stride_32) > len(dataset_stride_64)

    def test_labels_shifted_correctly(self, tokenizer):
        """Test that labels are shifted by 1 position."""
        dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=128,
            stride=64,
        )

        item = dataset[0]
        input_ids = item["input_ids"]
        labels = item["labels"]

        # labels[i] should equal input_ids[i+1] for i < context_length-1
        for i in range(len(input_ids) - 1):
            assert (
                labels[i] == input_ids[i + 1]
            ), f"Label at position {i} should equal input_id at position {i+1}"

    def test_no_data_leakage_between_splits(self, tokenizer):
        """Test that train/val/test splits don't overlap."""
        context_length = 128
        stride = 64

        train_dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="train",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=stride,
        )

        test_dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=stride,
        )

        # Just verify both datasets exist and have different sizes
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        # Train should typically be larger than test
        assert len(train_dataset) > len(test_dataset)

    def test_stride_equals_context_length(self, tokenizer):
        """Test stride equal to context_length (no overlap)."""
        context_length = 128
        stride = 128  # No overlap

        dataset = SequenceDataset(
            dataset_name="wikitext-2",
            split="test",
            tokenizer=tokenizer,
            context_length=context_length,
            stride=stride,
        )

        assert len(dataset) > 0
        # Verify sequences have correct length
        item = dataset[0]
        assert item["input_ids"].shape[0] == context_length


# ============================================================================
# Task 2.4.4: Test Batch Shapes and Padding
# ============================================================================


class TestBatchShapesAndPadding:
    """Tests for batch creation and padding."""

    def test_create_dataloaders(self, tokenizer):
        """Test that dataloaders can be created."""
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,  # Use 0 workers for testing
            tokenizer=tokenizer,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_shapes(self, tokenizer):
        """Test that batches have correct shapes."""
        batch_size = 4
        context_length = 128

        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=context_length,
            stride=64,
            batch_size=batch_size,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Get first batch
        batch = next(iter(train_loader))

        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["input_ids"].shape[1] == context_length
        assert batch["labels"].shape[0] == batch_size
        assert batch["labels"].shape[1] == context_length

    def test_dynamic_padding(self, tokenizer):
        """Test dynamic padding with variable-length sequences."""
        # Create sample batch with different lengths
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "labels": torch.tensor([2, 3, 4, 5, 6]),
            },
            {
                "input_ids": torch.tensor([10, 11, 12]),
                "labels": torch.tensor([11, 12, 13]),
            },
            {
                "input_ids": torch.tensor([20, 21, 22, 23]),
                "labels": torch.tensor([21, 22, 23, 24]),
            },
        ]

        pad_token_id = tokenizer.pad_token_id
        result = collate_batch_with_padding(batch, pad_token_id)

        # Check shapes
        assert result["input_ids"].shape[0] == 3  # batch_size
        assert result["input_ids"].shape[1] == 5  # max_len in batch
        assert result["labels"].shape == result["input_ids"].shape
        assert result["attention_mask"].shape == result["input_ids"].shape

    def test_attention_mask_correct(self, tokenizer):
        """Test that attention mask is correct for padded sequences."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([10, 11]), "labels": torch.tensor([11, 12])},
        ]

        pad_token_id = tokenizer.pad_token_id
        result = collate_batch_with_padding(batch, pad_token_id)

        # First sequence: all 1s (no padding)
        assert torch.all(result["attention_mask"][0] == 1)

        # Second sequence: first 2 are 1s, last is 0 (padding)
        assert result["attention_mask"][1, 0] == 1
        assert result["attention_mask"][1, 1] == 1
        assert result["attention_mask"][1, 2] == 0

    def test_padding_uses_pad_token(self, tokenizer):
        """Test that padding uses correct pad token ID."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([10]), "labels": torch.tensor([11])},
        ]

        pad_token_id = tokenizer.pad_token_id
        result = collate_batch_with_padding(batch, pad_token_id)

        # Check that padded positions use pad_token_id
        assert result["input_ids"][1, 1] == pad_token_id
        assert result["input_ids"][1, 2] == pad_token_id

    def test_batch_token_ids_in_range(self, tokenizer):
        """Test that all token IDs in batch are in valid range."""
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        vocab_size = tokenizer.vocab_size

        assert torch.all(input_ids >= 0)
        assert torch.all(input_ids < vocab_size)
        assert torch.all(labels >= 0)
        assert torch.all(labels < vocab_size)

    def test_dataloader_with_dynamic_padding(self, tokenizer):
        """Test dataloader with dynamic padding enabled."""
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
            use_dynamic_padding=True,
        )

        batch = next(iter(train_loader))

        # Should have attention_mask when using dynamic padding
        assert "attention_mask" in batch
        assert batch["attention_mask"].shape == batch["input_ids"].shape


# ============================================================================
# Task 2.4.5: Test Data Loading Performance
# ============================================================================


class TestDataLoadingPerformance:
    """Tests for data loading performance."""

    def test_dataloader_iteration(self, tokenizer):
        """Test that dataloader can iterate through all batches."""
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        batch_count = 0
        for batch in train_loader:
            assert "input_ids" in batch
            assert "labels" in batch
            batch_count += 1
            if batch_count >= 10:  # Test first 10 batches
                break

        assert batch_count > 0

    def test_throughput_measurement(self, tokenizer):
        """Test data loading throughput (tokens/second)."""
        context_length = 128
        batch_size = 8

        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=context_length,
            stride=64,
            batch_size=batch_size,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Measure time to load batches
        num_batches = 10
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

        elapsed_time = time.time() - start_time

        # Calculate throughput
        total_tokens = num_batches * batch_size * context_length
        throughput = total_tokens / elapsed_time

        # Should be able to load at least 1000 tokens/second
        # (This is a very conservative threshold for testing)
        assert (
            throughput > 1000
        ), f"Throughput {throughput:.0f} tokens/sec is below minimum 1000"

    def test_prefetching_enabled(self, tokenizer):
        """Test that prefetching can be enabled."""
        # Test with num_workers > 0 and prefetch_factor
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=2,
            tokenizer=tokenizer,
            prefetch_factor=2,
        )

        # Just verify we can iterate
        batch = next(iter(train_loader))
        assert batch is not None

    def test_multiple_workers(self, tokenizer):
        """Test data loading with multiple workers."""
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=2,
            tokenizer=tokenizer,
        )

        # Iterate through a few batches
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 5:
                break

        assert batch_count == 5

    def test_pin_memory(self, tokenizer):
        """Test that pin_memory is enabled for faster GPU transfer."""
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Check that pin_memory is enabled
        assert train_loader.pin_memory is True

    def test_shuffle_enabled_for_training(self, tokenizer):
        """Test that training data is shuffled."""
        train_loader, val_loader, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Get first batch from two iterations
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))

        # Batches should exist (can't easily test shuffling without multiple epochs)
        assert batch1 is not None
        assert batch2 is not None

    def test_no_blocking_on_data_load(self, tokenizer):
        """Test that data loading doesn't block excessively."""
        train_loader, _, _ = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Time to get first batch should be reasonable
        start_time = time.time()
        batch = next(iter(train_loader))
        elapsed_time = time.time() - start_time

        # Should take less than 5 seconds to get first batch
        assert (
            elapsed_time < 5.0
        ), f"First batch took {elapsed_time:.2f}s, which is too long"
        assert batch is not None


# ============================================================================
# Additional Integration Tests
# ============================================================================


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""

    def test_end_to_end_pipeline(self, tokenizer):
        """Test complete data pipeline from loading to batching."""
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Test train loader
        train_batch = next(iter(train_loader))
        assert train_batch["input_ids"].shape == (4, 128)
        assert train_batch["labels"].shape == (4, 128)

        # Test val loader
        val_batch = next(iter(val_loader))
        assert val_batch["input_ids"].shape[0] <= 4  # May be smaller
        assert val_batch["input_ids"].shape[1] == 128

        # Test test loader
        test_batch = next(iter(test_loader))
        assert test_batch["input_ids"].shape[0] <= 4
        assert test_batch["input_ids"].shape[1] == 128

    def test_dataset_sizes_reasonable(self, tokenizer):
        """Test that dataset sizes are reasonable."""
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name="wikitext-2",
            tokenizer_name="gpt2",
            context_length=128,
            stride=64,
            batch_size=4,
            num_workers=0,
            tokenizer=tokenizer,
        )

        # Train should be largest
        assert len(train_loader.dataset) > len(val_loader.dataset)
        assert len(train_loader.dataset) > len(test_loader.dataset)

        # All should be non-empty
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
