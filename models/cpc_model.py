"""
Contrastive Predictive Coding (CPC) baseline model.

CPC learns representations by predicting future observations in a latent space
using a contrastive loss. This encourages representations that capture
temporal structure and are discriminative for future states.

Reference:
    Oord et al. (2018). "Representation Learning with Contrastive Predictive Coding"
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.encoder import Encoder
from models.decoder import Decoder


@dataclass
class CPCOutput:
    """Output from CPC model forward pass."""

    logits: Tensor  # [batch_size, seq_len, vocab_size]
    latents: Tensor  # [batch_size, seq_len, latent_dim]
    token_loss: Tensor  # Scalar
    cpc_loss: Tensor  # Scalar (contrastive loss)
    total_loss: Tensor  # Scalar (combined)


class CPCModel(nn.Module):
    """
    Contrastive Predictive Coding model for sequence modeling.

    CPC extends the standard Transformer encoder with:
    1. A context network (the encoder) that produces context representations
    2. A prediction network that predicts future context representations
    3. A contrastive loss that distinguishes true future states from negatives

    The key idea is to learn representations that are maximally informative
    about future observations while being discriminative.
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        max_context_length: int = 512,
        context_dim: Optional[int] = None,
        prediction_horizons: Tuple[int, ...] = (1, 2, 5),
        temperature: float = 0.1,
        num_negatives: int = 10,
    ):
        """
        Initialize CPC model.

        Args:
            vocab_size: Size of vocabulary
            latent_dim: Dimension of latent representations
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_context_length: Maximum sequence length
            context_dim: Dimension of context network output (default: latent_dim)
            prediction_horizons: Time steps to predict into future
            temperature: Temperature for contrastive loss
            num_negatives: Number of negative samples per positive
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_context_length = max_context_length
        self.prediction_horizons = list(prediction_horizons)
        self.temperature = temperature
        self.num_negatives = num_negatives

        # Context network (encoder)
        self.encoder = Encoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context_length,
        )

        # Context dimension
        self.context_dim = context_dim or latent_dim

        # Context projection (if needed)
        if latent_dim != self.context_dim:
            self.context_proj = nn.Linear(latent_dim, self.context_dim)
        else:
            self.context_proj = nn.Identity()

        # Prediction networks (one per horizon)
        self.predictors = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(self.context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.context_dim),
            )
            for k in prediction_horizons
        })

        # Target projection (for contrastive learning)
        # Targets are future encoder outputs projected to context space
        self.target_proj = nn.Linear(latent_dim, self.context_dim)

        # Decoder for language modeling
        self.decoder = Decoder(latent_dim=latent_dim, vocab_size=vocab_size)

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def encode(
        self, tokens: Tensor, return_context: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Encode tokens to latent representations and optionally context.

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            return_context: Whether to return context representations

        Returns:
            Tuple of (latents, contexts)
        """
        latents = self.encoder(tokens)

        if return_context:
            contexts = self.context_proj(latents)
            return latents, contexts
        return latents, None

    def predict_future(
        self, contexts: Tensor, horizon: int
    ) -> Tensor:
        """
        Predict future context representation.

        Args:
            contexts: Current context representations [batch_size, seq_len, context_dim]
            horizon: Number of steps ahead to predict

        Returns:
            Predicted future contexts [batch_size, seq_len-horizon, context_dim]
        """
        predictor = self.predictors[str(horizon)]
        # Predict from context at time t
        current_contexts = contexts[:, :-horizon, :]
        predicted = predictor(current_contexts)
        return predicted

    def contrastive_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        negatives: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute contrastive InfoNCE loss.

        Args:
            predictions: Predicted representations [batch_size, seq_len-k, context_dim]
            targets: Target representations [batch_size, seq_len-k, context_dim]
            negatives: Negative samples [num_negatives, context_dim] (optional)

        Returns:
            Scalar loss value
        """
        batch_size, seq_len_k, dim = predictions.shape

        # Normalize for cosine similarity
        predictions = F.normalize(predictions, dim=-1)
        targets = F.normalize(targets, dim=-1)

        # Positive similarity: (batch, seq_len-k)
        positive_sim = (predictions * targets).sum(dim=-1)

        # Temperature scaling
        temperature = torch.exp(self.log_temperature)
        positive_sim = positive_sim / temperature

        if negatives is not None and len(negatives) > 0:
            # Normalize negatives
            negatives = F.normalize(negatives, dim=-1)  # [num_negatives, context_dim]

            # Compute similarities to negatives
            # Expand predictions for batch matrix multiplication
            pred_expanded = predictions.view(-1, dim)  # [batch * (seq-k), dim]

            # Negative similarities: [batch * (seq-k), num_negatives]
            neg_sim = pred_expanded @ negatives.T
            neg_sim = neg_sim / temperature

            # Concatenate positive and negative similarities
            # Positive first, then negatives
            logits = torch.cat([
                positive_sim.view(-1, 1),  # [batch * (seq-k), 1]
                neg_sim  # [batch * (seq-k), num_negatives]
            ], dim=1)
        else:
            # Simple version: use other positions in batch as negatives
            # This is less efficient but works without explicit negative sampling
            all_sims = predictions @ targets.transpose(-2, -1)  # [batch, seq-k, seq-k]
            logits = all_sims / temperature

        # InfoNCE loss: maximize log probability of positive
        labels = torch.zeros(batch_size * seq_len_k, dtype=torch.long, device=predictions.device)
        loss = F.cross_entropy(logits.view(batch_size * seq_len_k, -1), labels)

        return loss

    def sample_negatives(self, targets: Tensor) -> Tensor:
        """
        Sample negative examples from target representations.

        Args:
            targets: Target representations [batch_size, seq_len, dim]

        Returns:
            Negative samples [num_negatives, dim]
        """
        batch_size, seq_len, dim = targets.shape

        # Flatten and sample
        flat_targets = targets.view(-1, dim)
        num_targets = flat_targets.size(0)

        if num_targets <= self.num_negatives:
            # Not enough targets, return all
            return flat_targets

        # Random sampling
        indices = torch.randperm(num_targets, device=targets.device)[:self.num_negatives]
        negatives = flat_targets[indices]

        return negatives

    def forward(
        self,
        tokens: Tensor,
        labels: Optional[Tensor] = None,
        compute_cpc_loss: bool = True,
    ) -> CPCOutput:
        """
        Forward pass with CPC and token prediction.

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len] (optional)
            compute_cpc_loss: Whether to compute CPC loss

        Returns:
            CPCOutput with losses and predictions
        """
        batch_size, seq_len = tokens.shape

        if labels is None:
            labels = tokens.clone()

        # Encode to latent and context representations
        latents, contexts = self.encode(tokens, return_context=True)

        # Token prediction
        logits = self.decoder(latents)
        token_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            reduction="mean",
            ignore_index=-100,
        )

        # CPC loss
        if compute_cpc_loss and contexts is not None:
            total_cpc_loss = 0.0

            # Sample negatives once per batch
            negatives = self.sample_negatives(contexts) if self.num_negatives > 0 else None

            for horizon in self.prediction_horizons:
                if seq_len <= horizon:
                    continue

                # Predict future
                predicted = self.predict_future(contexts, horizon)

                # Get target contexts
                target_latents = latents[:, horizon:, :]
                target_contexts = self.target_proj(target_latents)

                # Contrastive loss
                cpc_loss = self.contrastive_loss(predicted, target_contexts, negatives)
                total_cpc_loss += cpc_loss

            # Average over horizons
            if len(self.prediction_horizons) > 0:
                cpc_loss = total_cpc_loss / len(self.prediction_horizons)
            else:
                cpc_loss = torch.tensor(0.0, device=tokens.device)
        else:
            cpc_loss = torch.tensor(0.0, device=tokens.device)

        # Combined loss (equal weighting)
        total_loss = token_loss + cpc_loss

        return CPCOutput(
            logits=logits,
            latents=latents,
            token_loss=token_loss,
            cpc_loss=cpc_loss,
            total_loss=total_loss,
        )

    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            tokens: Initial token sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = self.forward(tokens, compute_cpc_loss=False)
                logits = output.logits

                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                tokens = torch.cat([tokens, next_token], dim=1)

                if tokens.size(1) > self.max_context_length:
                    tokens = tokens[:, -self.max_context_length:]

        return tokens


def build_cpc_model(
    vocab_size: int,
    latent_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 2048,
    dropout: float = 0.1,
    max_context_length: int = 512,
    prediction_horizons: Tuple[int, ...] = (1, 2, 5),
    temperature: float = 0.1,
    num_negatives: int = 10,
    device: str = "cuda",
) -> CPCModel:
    """
    Factory function to build CPC model.

    Args:
        vocab_size: Size of vocabulary
        latent_dim: Dimension of latent representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Dimension of feedforward network
        dropout: Dropout probability
        max_context_length: Maximum sequence length
        prediction_horizons: Time steps to predict into future
        temperature: Temperature for contrastive loss
        num_negatives: Number of negative samples
        device: Target device

    Returns:
        Initialized CPC model
    """
    model = CPCModel(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_context_length=max_context_length,
        prediction_horizons=prediction_horizons,
        temperature=temperature,
        num_negatives=num_negatives,
    )
    return model.to(device)
