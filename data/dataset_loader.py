"""
Dataset loader module for sequence modeling tasks.

This module provides the SequenceDataset class for loading and preprocessing
text datasets from HuggingFace with support for sliding window chunking,
tokenization, and label creation for next-token prediction.
"""

from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset
import logging

from data.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """
    Dataset for sequence modeling with sliding window chunking.
    
    Supports loading datasets from HuggingFace and preprocessing them into
    fixed-length sequences with proper tokenization and label creation.
    
    Attributes:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split ('train', 'validation', 'test')
        tokenizer: HuggingFace tokenizer instance
        context_length: Length of each sequence
        stride: Stride for sliding window (overlap = context_length - stride)
        sequences: List of tokenized sequences
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        stride: int
    ):
        """
        Initialize SequenceDataset with tokenization and chunking.
        
        Args:
            dataset_name: Name of HuggingFace dataset
                         ('wikitext-2-raw-v1', 'wikitext-103-raw-v1',
                          'ptb', 'roneneldan/TinyStories', 'openwebtext')
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: HuggingFace tokenizer instance
            context_length: Length of each sequence
            stride: Stride for sliding window
        
        Preconditions:
            - dataset_name is valid HuggingFace dataset
            - split is valid for the dataset
            - tokenizer is a valid PreTrainedTokenizer
            - context_length > 0
            - stride > 0 and stride <= context_length
        
        Postconditions:
            - self.sequences contains all chunked sequences
            - Each sequence has exactly context_length tokens
            - Labels are created by shifting input_ids by 1 position
            - No data leakage between splits
        """
        # Validate inputs
        assert context_length > 0, "context_length must be positive"
        assert stride > 0, "stride must be positive"
        assert stride <= context_length, "stride must be <= context_length"
        
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.sequences = []
        
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        logger.info(f"Context length: {context_length}, stride: {stride}")
        
        # Load and process dataset
        self._load_and_process_dataset()
        
        logger.info(f"Dataset loaded: {len(self.sequences)} sequences")
    
    def _load_and_process_dataset(self) -> None:
        """
        Load dataset from HuggingFace and process into sequences.
        
        Preconditions:
            - self.dataset_name is valid
            - self.split is valid for the dataset
        
        Postconditions:
            - self.sequences is populated with tokenized sequences
            - Each sequence has shape [context_length]
        """
        # Map common dataset names to HuggingFace identifiers
        dataset_mapping = {
            'wikitext-2': 'wikitext',
            'wikitext': 'wikitext',
            'wikitext-103': 'wikitext',
            'ptb': 'ptb_text_only',
            'penn-treebank': 'ptb_text_only',
            'penn_treebank': 'ptb_text_only',
            'ptb_text_only': 'ptb_text_only',
            'tinystories': 'roneneldan/TinyStories',
            'openwebtext': 'openwebtext'
        }
        
        # Get actual dataset name and config
        if self.dataset_name.lower() in dataset_mapping:
            hf_dataset_name = dataset_mapping[self.dataset_name.lower()]
            if 'wikitext' in self.dataset_name.lower():
                if '103' in self.dataset_name:
                    config_name = 'wikitext-103-raw-v1'
                else:
                    config_name = 'wikitext-2-raw-v1'
            elif hf_dataset_name == 'ptb_text_only':
                config_name = 'penn_treebank'
            else:
                config_name = None
        else:
            hf_dataset_name = self.dataset_name
            config_name = None
        
        # Map split names
        split_mapping = {
            'train': 'train',
            'validation': 'validation',
            'val': 'validation',
            'test': 'test'
        }
        hf_split = split_mapping.get(self.split.lower(), self.split)
        
        try:
            # Load dataset
            if config_name:
                dataset = load_dataset(hf_dataset_name, config_name, split=hf_split)
            else:
                dataset = load_dataset(hf_dataset_name, split=hf_split)
            
            logger.info(f"Dataset loaded: {len(dataset)} examples")
            
            # Tokenize and chunk dataset
            self._tokenize_and_chunk(dataset)
            
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            logger.info("Check dataset name, split, and internet connectivity")
            raise
    
    def _tokenize_and_chunk(self, dataset: HFDataset) -> None:
        """
        Tokenize dataset and chunk into fixed-length sequences.
        
        Args:
            dataset: HuggingFace dataset instance
        
        Preconditions:
            - dataset is valid HFDataset
            - dataset contains 'text' field
        
        Postconditions:
            - self.sequences contains all chunked sequences
            - Each sequence has exactly context_length tokens
        """
        # Concatenate all text
        logger.info("Tokenizing dataset...")
        all_text = []
        
        # Get text field name (varies by dataset)
        text_field = 'text' if 'text' in dataset.column_names else dataset.column_names[0]
        
        for example in dataset:
            text = example[text_field]
            if text and text.strip():  # Skip empty texts
                all_text.append(text)
        
        # Join with newlines to preserve document boundaries
        combined_text = '\n\n'.join(all_text)
        
        # Tokenize entire corpus
        encoded = self.tokenizer(
            combined_text,
            add_special_tokens=False,
            return_tensors='pt'
        )
        token_ids = encoded['input_ids'].squeeze(0)  # [total_tokens]
        
        logger.info(f"Total tokens: {len(token_ids)}")
        
        # Chunk into sequences with sliding window
        self._chunk_sequences(token_ids)
    
    def _chunk_sequences(self, token_ids: torch.Tensor) -> None:
        """
        Chunk token IDs into fixed-length sequences with sliding window.
        
        Args:
            token_ids: Tensor of token IDs [total_tokens]
        
        Preconditions:
            - token_ids is 1D tensor
            - len(token_ids) >= context_length
        
        Postconditions:
            - self.sequences contains chunked sequences
            - Each sequence has exactly context_length tokens
            - Sequences overlap by (context_length - stride) tokens
        """
        total_tokens = len(token_ids)
        
        if total_tokens < self.context_length:
            logger.warning(
                f"Dataset has only {total_tokens} tokens, "
                f"less than context_length {self.context_length}"
            )
            # Pad if necessary
            if total_tokens > 0:
                padding_length = self.context_length - total_tokens
                padding = torch.full(
                    (padding_length,),
                    self.tokenizer.pad_token_id,
                    dtype=token_ids.dtype
                )
                token_ids = torch.cat([token_ids, padding])
                self.sequences.append(token_ids)
            return
        
        # Sliding window chunking
        num_sequences = 0
        for start_idx in range(0, total_tokens - self.context_length + 1, self.stride):
            end_idx = start_idx + self.context_length
            sequence = token_ids[start_idx:end_idx]
            
            # Validate sequence length
            assert len(sequence) == self.context_length, \
                f"Sequence length {len(sequence)} != context_length {self.context_length}"
            
            self.sequences.append(sequence)
            num_sequences += 1
        
        logger.info(f"Created {num_sequences} sequences with stride {self.stride}")
    
    def __len__(self) -> int:
        """
        Return number of sequences in dataset.
        
        Returns:
            Number of sequences
        
        Postconditions:
            - Returns non-negative integer
        """
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sequence at index with labels.
        
        Args:
            idx: Sequence index
        
        Returns:
            Dictionary containing:
                - input_ids: Token IDs [context_length]
                - labels: Target token IDs [context_length] (shifted by 1)
        
        Preconditions:
            - 0 <= idx < len(self.sequences)
        
        Postconditions:
            - Returns dict with 'input_ids' and 'labels' keys
            - Both tensors have shape [context_length]
            - labels[i] = input_ids[i+1] for i < context_length-1
            - All token IDs are in range [0, vocab_size)
        """
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range [0, {len(self.sequences)})")
        
        input_ids = self.sequences[idx]
        
        # Create labels by shifting input_ids by 1 position
        # labels[i] = input_ids[i+1]
        labels = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.pad_token_id])])
        
        # Validate token IDs are in valid range
        vocab_size = self.tokenizer.vocab_size
        assert torch.all(input_ids >= 0), "Token IDs must be non-negative"
        assert torch.all(input_ids < vocab_size), \
            f"Token IDs must be less than vocab_size ({vocab_size})"
        assert torch.all(labels >= 0), "Label IDs must be non-negative"
        assert torch.all(labels < vocab_size), \
            f"Label IDs must be less than vocab_size ({vocab_size})"
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_dataloaders(
    dataset_name: str,
    tokenizer_name: str = 'gpt2',
    context_length: int = 512,
    stride: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    use_dynamic_padding: bool = False,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        dataset_name: Name of HuggingFace dataset
        tokenizer_name: Name of tokenizer to use
        context_length: Length of each sequence
        stride: Stride for sliding window
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        tokenizer: Optional pre-loaded tokenizer (if None, loads from tokenizer_name)
        use_dynamic_padding: If True, use dynamic padding to longest sequence in batch
        prefetch_factor: Number of batches to prefetch per worker (for data prefetching)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Preconditions:
        - dataset_name is valid HuggingFace dataset
        - tokenizer_name is valid if tokenizer is None
        - context_length > 0
        - stride > 0 and stride <= context_length
        - batch_size > 0
        - num_workers >= 0
        - prefetch_factor > 0 if num_workers > 0

    Postconditions:
        - Returns tuple of three DataLoader objects
        - Each loader yields batches with 'input_ids' and 'labels' keys
        - Batch shapes: [batch_size, seq_len] (seq_len varies if use_dynamic_padding=True)
        - No data overlap between train/val/test splits
        - Labels are shifted by 1 position from input_ids
        - Data prefetching enabled for improved throughput

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     dataset_name='wikitext-2',
        ...     context_length=512,
        ...     batch_size=32,
        ...     use_dynamic_padding=True
        ... )
        >>> batch = next(iter(train_loader))
        >>> print(batch['input_ids'].shape)
        torch.Size([32, 512])
    """
    # Validate inputs
    assert context_length > 0, "context_length must be positive"
    assert stride > 0, "stride must be positive"
    assert stride <= context_length, "stride must be <= context_length"
    assert batch_size > 0, "batch_size must be positive"
    assert num_workers >= 0, "num_workers must be non-negative"
    assert prefetch_factor > 0 or num_workers == 0, \
        "prefetch_factor must be positive when num_workers > 0"

    # Load tokenizer if not provided
    if tokenizer is None:
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = load_tokenizer(tokenizer_name)

    # Create collate function for dynamic padding if requested
    collate_fn = None
    if use_dynamic_padding:
        logger.info("Using dynamic padding for variable-length sequences")
        from functools import partial
        collate_fn = partial(
            collate_batch_with_padding,
            pad_token_id=tokenizer.pad_token_id
        )

    # Create datasets for each split
    logger.info("Creating datasets...")

    train_dataset = SequenceDataset(
        dataset_name=dataset_name,
        split='train',
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride
    )

    val_dataset = SequenceDataset(
        dataset_name=dataset_name,
        split='validation',
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride
    )

    test_dataset = SequenceDataset(
        dataset_name=dataset_name,
        split='test',
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride
    )

    # Create data loaders with prefetching support
    logger.info("Creating data loaders...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Prefetch factor: {prefetch_factor if num_workers > 0 else 'N/A'}")
    logger.info(f"  Dynamic padding: {use_dynamic_padding}")

    # Configure prefetch_factor (only used when num_workers > 0)
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,  # Faster GPU transfer
        'collate_fn': collate_fn
    }

    # Add prefetch_factor only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,  # Shuffle training data
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,  # Don't shuffle validation
        **dataloader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,  # Don't shuffle test
        **dataloader_kwargs
    )

    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} sequences, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} sequences, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} sequences, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader

def collate_batch_with_padding(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """
    Collate batch with dynamic padding to longest sequence.

    This function enables dynamic batching where sequences are padded to the
    longest sequence in the batch rather than a fixed length, improving
    efficiency for variable-length sequences.

    Args:
        batch: List of dictionaries, each containing 'input_ids' and 'labels'
        pad_token_id: Token ID to use for padding

    Returns:
        Dictionary containing:
            - input_ids: Padded token IDs [batch_size, max_seq_len]
            - labels: Padded labels [batch_size, max_seq_len]
            - attention_mask: Attention mask [batch_size, max_seq_len]

    Preconditions:
        - batch is non-empty list
        - Each item in batch has 'input_ids' and 'labels' keys
        - All tensors in batch are 1D
        - pad_token_id is valid token ID

    Postconditions:
        - All sequences padded to same length (max length in batch)
        - Attention mask is 1 for real tokens, 0 for padding
        - Padded positions use pad_token_id
        - Batch shapes: [batch_size, max_seq_len]

    Example:
        >>> batch = [
        ...     {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])},
        ...     {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([6, 7])}
        ... ]
        >>> result = collate_batch_with_padding(batch, pad_token_id=0)
        >>> print(result['input_ids'].shape)
        torch.Size([2, 3])
    """
    # Validate batch
    assert len(batch) > 0, "Batch must be non-empty"

    # Extract input_ids and labels
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # Find maximum sequence length in batch
    max_len = max(len(ids) for ids in input_ids_list)
    batch_size = len(batch)

    # Initialize padded tensors
    padded_input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=input_ids_list[0].dtype
    )
    padded_labels = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=labels_list[0].dtype
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long
    )

    # Fill in actual sequences and create attention mask
    for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = len(input_ids)
        padded_input_ids[i, :seq_len] = input_ids
        padded_labels[i, :seq_len] = labels
        attention_mask[i, :seq_len] = 1

    # Validate output shapes
    assert padded_input_ids.shape == (batch_size, max_len)
    assert padded_labels.shape == (batch_size, max_len)
    assert attention_mask.shape == (batch_size, max_len)

    # Validate attention mask contains only 0s and 1s
    assert torch.all((attention_mask == 0) | (attention_mask == 1)), \
        "Attention mask must contain only 0s and 1s"

    return {
        'input_ids': padded_input_ids,
        'labels': padded_labels,
        'attention_mask': attention_mask
    }


