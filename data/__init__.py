"""
Latent Forecasting Network - Data Pipeline

This package contains data loading and preprocessing:
- Dataset loaders for HuggingFace datasets
- Tokenization utilities
- Sequence chunking and batching
- Dynamic batching with padding
- Multi-worker data loading with prefetching
"""

from data.dataset_loader import (
    SequenceDataset,
    create_dataloaders,
    collate_batch_with_padding
)
from data.tokenizer import (
    load_tokenizer,
    tokenize_text,
    get_special_token_ids,
    add_special_tokens_to_sequence
)

__version__ = "0.1.0"

__all__ = [
    'SequenceDataset',
    'create_dataloaders',
    'collate_batch_with_padding',
    'load_tokenizer',
    'tokenize_text',
    'get_special_token_ids',
    'add_special_tokens_to_sequence'
]
