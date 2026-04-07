"""
Joint Embedding Predictive Architecture (JEPA) baseline model.

JEPA learns representations by predicting target embeddings from context embeddings
in a joint embedding space. Unlike CPC which uses contrastive learning, JEPA
uses a predictive approach with a stop-gradient on the target branch.

Reference:
    LeCun (2022). "A Path Towards Autonomous Machine Intelligence"
    Assran et al. (2023). "Self-Supervised Learning from Images with a Joint
    Embedding Predictive Architecture"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.encoder import Encoder
from models.decoder import Decoder


@dataclass
class JEPAOutput:
    """Output from JEPA model forward pass."""

    logits: Tensor  # [batch_size, seq_len, vocab_size]
    latents: Tensor  # [batch_size, seq_len, latent_dim]
    token_loss: Tensor  # Scalar
    jepa_loss: Tensor  # Scalar (predictive loss)
    total_loss: Tensor  # Scalar (combined)


class JEPAModel(nn.Module):
    """
    Joint Embedding Predictive Architecture for sequence modeling.

    JEPA consists of:
    1. A context encoder (same as standard encoder with learned parameters)
    2. A target encoder (shared weights or EMA of context encoder)
    3. A predictor network that predicts target embeddings from context

    Key features:
    - Stop-gradient on target branch (target embeddings are fixed targets)
    - Predictive loss in embedding space (not input space)
    - Asymmetric architecture allows context to be partial, target to be complete

    For sequential data, we use a causal variant where:
    - Context = tokens up to position t
    - Target = tokens up to position t+k (future)
    - Predictor must predict target embedding from context embedding
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
        target_encoder_ema: bool = True,
        ema_decay: float = 0.996,
        prediction_horizons: Tuple[int, ...] = (1, 2, 5),
        predictor_hidden_dim: Optional[int] = None,
        lambda_jepa: float = 1.0,
    ):
        """
        Initialize JEPA model.

        Args:
            vocab_size: Size of vocabulary
            latent_dim: Dimension of latent representations
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_context_length: Maximum sequence length
            target_encoder_ema: Use EMA for target encoder (vs shared weights)
            ema_decay: EMA decay rate for target encoder
            prediction_horizons: Time steps to predict into future
            predictor_hidden_dim: Hidden dim for predictor (default: hidden_dim)
            lambda_jepa: Weight for JEPA loss
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_context_length = max_context_length
        self.target_encoder_ema = target_encoder_ema
        self.ema_decay = ema_decay
        self.prediction_horizons = list(prediction_horizons)
        self.lambda_jepa = lambda_jepa

        # Context encoder (learned)
        self.context_encoder = Encoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_context_length=max_context_length,
        )

        # Target encoder
        if target_encoder_ema:
            # Target encoder is EMA of context encoder
            self.target_encoder = Encoder(
                vocab_size=vocab_size,
                latent_dim=latent_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                max_context_length=max_context_length,
            )
            # Initialize with same weights, then maintain as EMA
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
            # Disable gradients for target encoder
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            # Target encoder shares weights with context encoder
            self.target_encoder = self.context_encoder

        # Predictor network
        pred_hidden = predictor_hidden_dim or hidden_dim
        self.predictors = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(latent_dim, pred_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(pred_hidden, latent_dim),
            )
            for k in prediction_horizons
        })

        # Decoder for language modeling
        self.decoder = Decoder(latent_dim=latent_dim, vocab_size=vocab_size)

        # EMA tracking
        self._ema_steps = 0

    def update_target_encoder(self):
        """
        Update target encoder with EMA of context encoder.
        Called after each training step.
        """
        if not self.target_encoder_ema:
            return

        with torch.no_grad():
            for target_param, context_param in zip(
                self.target_encoder.parameters(),
                self.context_encoder.parameters()
            ):
                target_param.data.mul_(self.ema_decay).add_(
                    context_param.data, alpha=1 - self.ema_decay
                )

        self._ema_steps += 1

    def encode_context(self, tokens: Tensor) -> Tensor:
        """
        Encode tokens using context encoder.

        Args:
            tokens: Input token IDs [batch_size, seq_len]

        Returns:
            Context representations [batch_size, seq_len, latent_dim]
        """
        return self.context_encoder(tokens)

    def encode_target(self, tokens: Tensor) -> Tensor:
        """
        Encode tokens using target encoder (with stop-gradient).

        Args:
            tokens: Input token IDs [batch_size, seq_len]

        Returns:
            Target representations [batch_size, seq_len, latent_dim] (detached)
        """
        with torch.no_grad():
            return self.target_encoder(tokens)

    def predict_target(
        self, context_repr: Tensor, horizon: int
    ) -> Tensor:
        """
        Predict target representation from context.

        Args:
            context_repr: Context representations [batch_size, seq_len, latent_dim]
            horizon: Number of steps ahead to predict

        Returns:
            Predicted target representations [batch_size, seq_len-horizon, latent_dim]
        """
        predictor = self.predictors[str(horizon)]
        # Predict from context at time t
        current_contexts = context_repr[:, :-horizon, :]
        predicted = predictor(current_contexts)
        return predicted

    def jepa_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        loss_type: str = "smooth_l1",
    ) -> Tensor:
        """
        Compute JEPA predictive loss.

        Args:
            predictions: Predicted representations [batch_size, seq_len-k, latent_dim]
            targets: Target representations (detached) [batch_size, seq_len-k, latent_dim]
            loss_type: Type of loss ('mse', 'smooth_l1', 'cosine')

        Returns:
            Scalar loss value
        """
        if loss_type == "mse":
            return F.mse_loss(predictions, targets)
        elif loss_type == "smooth_l1":
            return F.smooth_l1_loss(predictions, targets)
        elif loss_type == "cosine":
            # Cosine distance (1 - cosine similarity)
            return (1 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        tokens: Tensor,
        labels: Optional[Tensor] = None,
        compute_jepa_loss: bool = True,
    ) -> JEPAOutput:
        """
        Forward pass with JEPA and token prediction.

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len] (optional)
            compute_jepa_loss: Whether to compute JEPA loss

        Returns:
            JEPAOutput with losses and predictions
        """
        batch_size, seq_len = tokens.shape

        if labels is None:
            labels = tokens.clone()

        # Encode with context encoder for language modeling
        context_latents = self.encode_context(tokens)

        # Token prediction
        logits = self.decoder(context_latents)
        token_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            reduction="mean",
            ignore_index=-100,
        )

        # JEPA loss
        if compute_jepa_loss:
            total_jepa_loss = 0.0

            for horizon in self.prediction_horizons:
                if seq_len <= horizon:
                    continue

                # Get target embeddings (with stop-gradient)
                # Target uses full sequence
                target_latents_full = self.encode_target(tokens)
                target_latents = target_latents_full[:, horizon:, :]

                # Predict target from context
                predicted = self.predict_target(context_latents, horizon)

                # Predictive loss
                pred_loss = self.jepa_loss(predicted, target_latents)
                total_jepa_loss += pred_loss

            # Average over horizons
            if len(self.prediction_horizons) > 0:
                jepa_loss = total_jepa_loss / len(self.prediction_horizons)
            else:
                jepa_loss = torch.tensor(0.0, device=tokens.device)
        else:
            jepa_loss = torch.tensor(0.0, device=tokens.device)

        # Combined loss
        total_loss = token_loss + self.lambda_jepa * jepa_loss

        return JEPAOutput(
            logits=logits,
            latents=context_latents,
            token_loss=token_loss,
            jepa_loss=jepa_loss,
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
                output = self.forward(tokens, compute_jepa_loss=False)
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


def build_jepa_model(
    vocab_size: int,
    latent_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 2048,
    dropout: float = 0.1,
    max_context_length: int = 512,
    target_encoder_ema: bool = True,
    ema_decay: float = 0.996,
    prediction_horizons: Tuple[int, ...] = (1, 2, 5),
    lambda_jepa: float = 1.0,
    device: str = "cuda",
) -> JEPAModel:
    """
    Factory function to build JEPA model.

    Args:
        vocab_size: Size of vocabulary
        latent_dim: Dimension of latent representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Dimension of feedforward network
        dropout: Dropout probability
        max_context_length: Maximum sequence length
        target_encoder_ema: Use EMA for target encoder
        ema_decay: EMA decay rate
        prediction_horizons: Time steps to predict into future
        lambda_jepa: Weight for JEPA loss
        device: Target device

    Returns:
        Initialized JEPA model
    """
    model = JEPAModel(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_context_length=max_context_length,
        target_encoder_ema=target_encoder_ema,
        ema_decay=ema_decay,
        prediction_horizons=prediction_horizons,
        lambda_jepa=lambda_jepa,
    )
    return model.to(device)
