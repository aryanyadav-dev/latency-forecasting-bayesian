"""
Tokenizer module for text tokenization using HuggingFace tokenizers.

This module provides utilities for loading tokenizers and tokenizing text
with proper caching and special token handling.
"""

from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def load_tokenizer(
    tokenizer_name: str = "gpt2",
    add_special_tokens: bool = True
) -> PreTrainedTokenizer:
    """
    Load a HuggingFace tokenizer with proper configuration.
    
    Args:
        tokenizer_name: Name of the tokenizer to load (default: "gpt2")
        add_special_tokens: Whether to add special tokens during tokenization
    
    Returns:
        Loaded tokenizer instance
    
    Preconditions:
        - tokenizer_name is a valid HuggingFace tokenizer identifier
    
    Postconditions:
        - Returns a PreTrainedTokenizer instance
        - Tokenizer has pad_token set (uses eos_token if not available)
        - Tokenizer is ready for use
    
    Example:
        >>> tokenizer = load_tokenizer("gpt2")
        >>> tokens = tokenizer("Hello world")
    """
    try:
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure pad_token is set (GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Log tokenizer info
        logger.info(f"Tokenizer loaded successfully")
        logger.info(f"Vocab size: {tokenizer.vocab_size}")
        logger.info(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        logger.info(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        logger.info(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        logger.info("Check tokenizer name and internet connectivity")
        raise


def tokenize_text(
    text: Union[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = False,
    truncation: bool = False,
    return_tensors: Optional[str] = "pt",
    add_special_tokens: bool = True,
    cache: Optional[Dict] = None
) -> Dict[str, torch.Tensor]:
    """
    Tokenize text with caching support and special token handling.
    
    Args:
        text: Input text or list of texts to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length (None for no limit)
        padding: Padding strategy ('max_length', 'longest', or False)
        truncation: Whether to truncate sequences exceeding max_length
        return_tensors: Format of returned tensors ('pt' for PyTorch)
        add_special_tokens: Whether to add BOS/EOS tokens
        cache: Optional dictionary for caching tokenized results
    
    Returns:
        Dictionary containing:
            - input_ids: Token IDs [batch_size, seq_len]
            - attention_mask: Attention mask [batch_size, seq_len]
    
    Preconditions:
        - text is non-empty string or list of strings
        - tokenizer is a valid PreTrainedTokenizer
        - max_length > 0 if specified
    
    Postconditions:
        - All token IDs are in range [0, vocab_size)
        - All token IDs are non-negative integers
        - Attention mask contains only 0s and 1s
        - If padding=True, all sequences have same length
        - Special tokens (BOS, EOS, PAD) are handled correctly
    
    Example:
        >>> tokenizer = load_tokenizer("gpt2")
        >>> result = tokenize_text("Hello world", tokenizer)
        >>> print(result['input_ids'].shape)
        torch.Size([1, 3])
    """
    # Check cache if provided
    if cache is not None:
        cache_key = (
            text if isinstance(text, str) else tuple(text),
            max_length,
            padding,
            truncation,
            add_special_tokens
        )
        if cache_key in cache:
            logger.debug("Returning cached tokenization result")
            return cache[cache_key]
    
    # Tokenize text
    try:
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
        
        # Validate token IDs are in valid range
        input_ids = encoded['input_ids']
        vocab_size = tokenizer.vocab_size
        
        assert torch.all(input_ids >= 0), "Token IDs must be non-negative"
        assert torch.all(input_ids < vocab_size), \
            f"Token IDs must be less than vocab_size ({vocab_size})"
        
        # Validate attention mask
        if 'attention_mask' in encoded:
            attention_mask = encoded['attention_mask']
            assert torch.all((attention_mask == 0) | (attention_mask == 1)), \
                "Attention mask must contain only 0s and 1s"
        
        # Cache result if cache provided
        if cache is not None:
            cache[cache_key] = encoded
        
        return encoded
        
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise


def get_special_token_ids(tokenizer: PreTrainedTokenizer) -> Dict[str, Optional[int]]:
    """
    Get special token IDs from tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
    
    Returns:
        Dictionary mapping special token names to their IDs:
            - bos_token_id: Beginning of sequence token ID
            - eos_token_id: End of sequence token ID
            - pad_token_id: Padding token ID
            - unk_token_id: Unknown token ID
            - sep_token_id: Separator token ID
            - cls_token_id: Classification token ID
    
    Preconditions:
        - tokenizer is a valid PreTrainedTokenizer
    
    Postconditions:
        - Returns dictionary with special token IDs
        - Values are either integers or None if token not available
        - All non-None IDs are in range [0, vocab_size)
    
    Example:
        >>> tokenizer = load_tokenizer("gpt2")
        >>> special_tokens = get_special_token_ids(tokenizer)
        >>> print(special_tokens['eos_token_id'])
        50256
    """
    special_tokens = {
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        'sep_token_id': tokenizer.sep_token_id,
        'cls_token_id': tokenizer.cls_token_id,
    }
    
    # Validate non-None IDs are in valid range
    vocab_size = tokenizer.vocab_size
    for token_name, token_id in special_tokens.items():
        if token_id is not None:
            assert 0 <= token_id < vocab_size, \
                f"{token_name} ({token_id}) must be in range [0, {vocab_size})"
    
    return special_tokens


def add_special_tokens_to_sequence(
    token_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    add_bos: bool = False,
    add_eos: bool = False
) -> torch.Tensor:
    """
    Add special tokens (BOS, EOS) to token sequences.
    
    Args:
        token_ids: Token IDs tensor [batch_size, seq_len] or [seq_len]
        tokenizer: HuggingFace tokenizer instance
        add_bos: Whether to add BOS token at the beginning
        add_eos: Whether to add EOS token at the end
    
    Returns:
        Token IDs with special tokens added
    
    Preconditions:
        - token_ids is a valid tensor
        - All token IDs are in range [0, vocab_size)
        - tokenizer has required special tokens if add_bos/add_eos is True
    
    Postconditions:
        - Returns tensor with special tokens added
        - If add_bos=True, first token is BOS token
        - If add_eos=True, last token is EOS token
        - All token IDs remain in valid range
    
    Example:
        >>> tokenizer = load_tokenizer("gpt2")
        >>> token_ids = torch.tensor([[100, 200, 300]])
        >>> result = add_special_tokens_to_sequence(token_ids, tokenizer, add_eos=True)
        >>> print(result.shape)
        torch.Size([1, 4])
    """
    # Handle 1D tensor
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    
    # Prepare special tokens
    tokens_to_add = []
    
    if add_bos:
        if tokenizer.bos_token_id is None:
            logger.warning("BOS token not available in tokenizer, skipping")
        else:
            bos_tokens = torch.full(
                (batch_size, 1),
                tokenizer.bos_token_id,
                dtype=token_ids.dtype,
                device=device
            )
            tokens_to_add.append(bos_tokens)
    
    tokens_to_add.append(token_ids)
    
    if add_eos:
        if tokenizer.eos_token_id is None:
            logger.warning("EOS token not available in tokenizer, skipping")
        else:
            eos_tokens = torch.full(
                (batch_size, 1),
                tokenizer.eos_token_id,
                dtype=token_ids.dtype,
                device=device
            )
            tokens_to_add.append(eos_tokens)
    
    # Concatenate tokens
    result = torch.cat(tokens_to_add, dim=1)
    
    # Validate result
    vocab_size = tokenizer.vocab_size
    assert torch.all(result >= 0), "Token IDs must be non-negative"
    assert torch.all(result < vocab_size), \
        f"Token IDs must be less than vocab_size ({vocab_size})"
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result
