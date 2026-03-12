# Training Module - Loss Functions

This module implements the loss computation functions for training the Latent Forecasting Network (LFN) model.

## Overview

The LFN model uses a composite loss function that combines:
1. **Token Prediction Loss**: Standard cross-entropy loss for next-token prediction
2. **Latent Forecasting Loss**: MSE loss for predicting future latent states
3. **Total Loss**: Weighted combination of the above losses

The total loss is computed as:
```
L_total = L_token + λ * L_latent
```

where λ (lambda) is a configurable weight parameter.

## Components

### Core Functions

#### `compute_token_loss(logits, labels, ignore_index=-100)`
Computes cross-entropy loss for token prediction.

**Parameters:**
- `logits`: Model output logits [batch_size, seq_len, vocab_size]
- `labels`: Target token IDs [batch_size, seq_len]
- `ignore_index`: Token ID to ignore (e.g., padding tokens)

**Returns:**
- Scalar tensor containing the average cross-entropy loss

**Example:**
```python
from training.loss_functions import compute_token_loss

logits = model_output.logits  # [2, 20, 50257]
labels = batch['labels']       # [2, 20]

token_loss = compute_token_loss(logits, labels)
print(f"Token Loss: {token_loss.item():.4f}")
```

#### `compute_latent_forecasting_loss(latents, predicted_latents, forecast_horizons)`
Computes multi-horizon latent forecasting loss using MSE.

**Parameters:**
- `latents`: Ground truth latent states [batch_size, seq_len, latent_dim]
- `predicted_latents`: Dict mapping horizon k to predictions {k: [batch_size, seq_len-k, latent_dim]}
- `forecast_horizons`: List of forecasting horizons (e.g., [1, 2, 5, 10])

**Returns:**
- Scalar tensor containing the average MSE loss across all horizons

**Example:**
```python
from training.loss_functions import compute_latent_forecasting_loss

latents = model_output.latents                    # [2, 20, 512]
predicted_latents = model_output.predicted_latents # {1: [2, 19, 512], 2: [2, 18, 512], ...}
horizons = [1, 2, 5]

latent_loss = compute_latent_forecasting_loss(latents, predicted_latents, horizons)
print(f"Latent Loss: {latent_loss.item():.4f}")
```

#### `compute_total_loss(token_loss, latent_loss, lambda_latent=0.1)`
Computes total loss as weighted combination of token and latent losses.

**Parameters:**
- `token_loss`: Cross-entropy loss for token prediction (scalar)
- `latent_loss`: MSE loss for latent forecasting (scalar or None)
- `lambda_latent`: Weight for latent loss (λ parameter)

**Returns:**
- Scalar tensor containing the total weighted loss

**Example:**
```python
from training.loss_functions import compute_total_loss

token_loss = torch.tensor(2.5)
latent_loss = torch.tensor(0.3)
lambda_latent = 0.1

total_loss = compute_total_loss(token_loss, latent_loss, lambda_latent)
# total_loss = 2.5 + 0.1 * 0.3 = 2.53
```

### Validation Functions

#### `validate_loss(loss, loss_name="loss")`
Validates that a loss value is finite and non-negative.

**Raises:**
- `ValueError` if loss is NaN, Inf, or negative

**Example:**
```python
from training.loss_functions import validate_loss

loss = torch.tensor(2.5)
validate_loss(loss, "token_loss")  # Passes

invalid_loss = torch.tensor(float('nan'))
validate_loss(invalid_loss, "token_loss")  # Raises ValueError
```

#### `validate_all_losses(token_loss, latent_loss, total_loss)`
Validates all loss components at once.

**Example:**
```python
from training.loss_functions import validate_all_losses

validate_all_losses(token_loss, latent_loss, total_loss)
```

### LossComputer Class

A utility class that encapsulates loss computation logic and provides a convenient interface.

**Parameters:**
- `lambda_latent`: Weight for latent forecasting loss (default: 0.1)
- `ignore_index`: Token ID to ignore in token loss (default: -100)
- `validate`: Whether to validate losses for NaN/Inf (default: True)

**Example:**
```python
from training.loss_functions import LossComputer

# Create loss computer
computer = LossComputer(
    lambda_latent=0.1,
    ignore_index=-100,
    validate=True
)

# Compute all losses at once
losses = computer.compute_losses(
    logits=output.logits,
    labels=tokens,
    latents=output.latents,
    predicted_latents=output.predicted_latents,
    forecast_horizons=[1, 2, 5]
)

# Access individual losses
print(f"Token Loss: {losses['token_loss'].item():.4f}")
print(f"Latent Loss: {losses['latent_loss'].item():.4f}")
print(f"Total Loss: {losses['total_loss'].item():.4f}")
```

## Usage Examples

### Basic Training Loop

```python
from models.complete_model import build_model, ModelConfig
from training.loss_functions import LossComputer
import torch

# Setup
config = ModelConfig(vocab_size=50257, latent_dim=512, lambda_latent=0.1)
model = build_model(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
computer = LossComputer(lambda_latent=config.lambda_latent)

# Training step
for batch in dataloader:
    tokens = batch['input_ids']
    
    # Forward pass
    output = model(tokens, compute_latent_loss=True)
    
    # Compute losses
    losses = computer.compute_losses(
        output.logits,
        tokens,
        output.latents,
        output.predicted_latents,
        config.forecast_horizons
    )
    
    # Backward pass
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()
    
    # Log metrics
    print(f"Step {step}: Loss = {losses['total_loss'].item():.4f}")
```

### Computing Losses Without Latent Forecasting

```python
# For baseline models or when latent forecasting is disabled
output = model(tokens, compute_latent_loss=False)

losses = computer.compute_losses(
    output.logits,
    tokens
    # No latent components provided
)

# Only token loss is computed
assert 'latent_loss' not in losses
assert losses['total_loss'] == losses['token_loss']
```

### Handling Padding Tokens

```python
# Create labels with padding
labels = tokens.clone()
labels[:, -5:] = -100  # Last 5 tokens are padding

# Compute loss with ignore_index
computer = LossComputer(ignore_index=-100)
losses = computer.compute_losses(output.logits, labels)

# Loss is computed only on non-padding tokens
```

### Experimenting with Different Lambda Values

```python
# Test different lambda values
for lambda_val in [0.0, 0.01, 0.1, 0.5, 1.0]:
    computer = LossComputer(lambda_latent=lambda_val)
    losses = computer.compute_losses(
        output.logits,
        tokens,
        output.latents,
        output.predicted_latents,
        config.forecast_horizons
    )
    print(f"λ={lambda_val}: Total Loss = {losses['total_loss'].item():.4f}")
```

## Loss Properties

The loss functions satisfy the following properties:

### 1. Non-Negativity
All losses are always non-negative:
```
∀ inputs: L_token ≥ 0, L_latent ≥ 0, L_total ≥ 0
```

### 2. Loss Composition
Total loss is always the correct weighted sum:
```
∀ inputs: L_total = L_token + λ * L_latent
```

### 3. Finiteness
All losses are finite (no NaN or Inf) when validation is enabled.

### 4. Gradient Flow
All losses support backpropagation and gradient flow to model parameters.

## Testing

The module includes comprehensive tests:

```bash
# Run loss function tests
pytest tests/test_loss_functions.py -v

# Run integration tests with model
pytest tests/test_loss_integration.py -v

# Run all tests
pytest tests/test_loss*.py -v
```

## Demo Script

Run the demo script to see all features in action:

```bash
PYTHONPATH=. python examples/loss_functions_demo.py
```

## Implementation Details

### Token Loss
- Uses PyTorch's `F.cross_entropy` with mean reduction
- Supports `ignore_index` for padding tokens
- Reshapes logits and labels to 2D for efficient computation

### Latent Loss
- Computes MSE between predicted and actual latents for each horizon
- Averages across all forecasting horizons
- Validates horizon values and tensor shapes

### Total Loss
- Simple weighted sum of component losses
- Handles case when latent loss is None (baseline mode)
- Validates lambda parameter is non-negative

### Validation
- Checks for NaN using `torch.isnan()`
- Checks for Inf using `torch.isfinite()`
- Checks for negative values
- Provides descriptive error messages

## Performance Considerations

- All operations are vectorized for GPU efficiency
- Loss computation is O(batch_size * seq_len) for token loss
- Latent loss is O(batch_size * seq_len * num_horizons)
- Validation adds minimal overhead (~1% of total time)

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive Lambda**: Automatically adjust λ during training
2. **Horizon Weighting**: Different weights for different forecasting horizons
3. **Focal Loss**: Option to use focal loss for token prediction
4. **Label Smoothing**: Support for label smoothing in token loss
5. **Loss Scheduling**: Gradually increase/decrease λ during training

## References

- Cross-Entropy Loss: [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- MSE Loss: [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html)
- Multi-Task Learning: [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)


---

# Trainer Module

The `Trainer` class manages the complete training loop for the Latent Forecasting Network.

## Features

- Forward and backward passes with automatic loss computation
- Gradient accumulation for simulating larger batch sizes
- Gradient clipping for training stability
- Mixed precision training (FP16) for 2x speedup
- Learning rate scheduling with warmup
- Automatic validation after each epoch
- Checkpoint saving and loading
- Training history tracking
- Progress logging

## TrainingConfig

Configuration dataclass for training parameters:

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 10                      # Number of training epochs
    batch_size: int = 32                      # Batch size
    learning_rate: float = 1e-4               # Base learning rate
    weight_decay: float = 0.01                # Weight decay coefficient
    gradient_accumulation_steps: int = 1      # Steps to accumulate gradients
    max_grad_norm: float = 1.0                # Maximum gradient norm for clipping
    warmup_steps: int = 1000                  # Number of warmup steps
    lambda_latent: float = 0.1                # Weight for latent forecasting loss
    use_mixed_precision: bool = True          # Enable FP16 training
    checkpoint_every: int = 1000              # Save checkpoint every N steps
    log_every: int = 100                      # Log metrics every N steps
    seed: int = 42                            # Random seed
    device: str = 'cuda'                      # Device ('cuda' or 'cpu')
```

## Basic Usage

```python
from models.complete_model import LatentForecastingModel, ModelConfig
from training.trainer import Trainer, TrainingConfig
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler

# Create model
model_config = ModelConfig(vocab_size=50257, latent_dim=512)
model = LatentForecastingModel(model_config)

# Create optimizer and scheduler
optimizer = create_optimizer(model, learning_rate=1e-4)
num_training_steps = len(train_loader) * num_epochs
scheduler = create_scheduler(optimizer, num_training_steps=num_training_steps, warmup_steps=1000)

# Create training config
training_config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    use_mixed_precision=True,
    device='cuda'
)

# Initialize trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config,
    checkpoint_dir='checkpoints'
)

# Train
history = trainer.train()

# Access training history
print(f"Final train loss: {history.train_losses[-1]:.4f}")
print(f"Final val loss: {history.val_losses[-1]:.4f}")
print(f"Best val loss: {trainer.best_val_loss:.4f}")
```

## Advanced Features

### Gradient Accumulation

Simulate larger batch sizes with gradient accumulation:

```python
training_config = TrainingConfig(
    batch_size=8,                      # Actual batch size
    gradient_accumulation_steps=4,     # Effective batch size = 8 * 4 = 32
    learning_rate=1e-4
)
```

### Mixed Precision Training

Enable FP16 for 2x speedup on modern GPUs:

```python
training_config = TrainingConfig(
    use_mixed_precision=True,
    device='cuda'
)
```

### Checkpoint Management

```python
# Save checkpoint manually
trainer.save_checkpoint(filename='my_checkpoint.pt')

# Load checkpoint and resume training
metadata = trainer.load_checkpoint('checkpoints/checkpoint_5000.pt')
print(f"Resuming from step {metadata['global_step']}")

# Continue training
history = trainer.train()
```

### Training History

```python
# Access training metrics
history = trainer.train()

for epoch in range(len(history.epochs)):
    print(f"Epoch {history.epochs[epoch]}:")
    print(f"  Train Loss: {history.train_losses[epoch]:.4f}")
    print(f"  Val Loss: {history.val_losses[epoch]:.4f}")
    print(f"  Learning Rate: {history.learning_rates[epoch]:.2e}")
```

## Methods

### `train() -> TrainingHistory`

Execute complete training loop for configured number of epochs.

**Returns:**
- `TrainingHistory` with metrics per epoch

**Example:**
```python
history = trainer.train()
```

### `train_epoch() -> Dict[str, float]`

Execute single training epoch.

**Returns:**
- Dictionary with average metrics for the epoch

**Example:**
```python
metrics = trainer.train_epoch()
print(f"Epoch loss: {metrics['total_loss']:.4f}")
```

### `validate() -> Dict[str, float]`

Run validation and return metrics.

**Returns:**
- Dictionary with validation metrics

**Example:**
```python
val_metrics = trainer.validate()
print(f"Validation loss: {val_metrics['total_loss']:.4f}")
```

### `save_checkpoint(is_best=False, filename=None)`

Save model checkpoint with metadata.

**Parameters:**
- `is_best`: Whether this is the best checkpoint
- `filename`: Optional custom filename

**Example:**
```python
trainer.save_checkpoint(is_best=True)
trainer.save_checkpoint(filename='epoch_5.pt')
```

### `load_checkpoint(checkpoint_path) -> Dict[str, Any]`

Load checkpoint and restore training state.

**Parameters:**
- `checkpoint_path`: Path to checkpoint file

**Returns:**
- Dictionary with checkpoint metadata

**Example:**
```python
metadata = trainer.load_checkpoint('checkpoints/best_model.pt')
print(f"Loaded from step {metadata['global_step']}")
```

## Examples

### Complete Training Workflow

See `examples/trainer_demo.py` for a complete example:

```bash
PYTHONPATH=. python examples/trainer_demo.py
```

### Custom Training Loop

```python
# Initialize trainer
trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, config)

# Train for specific number of epochs
for epoch in range(num_epochs):
    # Train one epoch
    train_metrics = trainer.train_epoch()
    
    # Validate
    val_metrics = trainer.validate()
    
    # Custom logging
    print(f"Epoch {epoch}: Train={train_metrics['total_loss']:.4f}, Val={val_metrics['total_loss']:.4f}")
    
    # Custom checkpoint logic
    if val_metrics['total_loss'] < best_loss:
        best_loss = val_metrics['total_loss']
        trainer.save_checkpoint(is_best=True)
```

## Testing

Run trainer tests:

```bash
pytest tests/test_trainer.py -v
```

## Performance Tips

1. **Use Mixed Precision**: Enable `use_mixed_precision=True` for 2x speedup on modern GPUs
2. **Gradient Accumulation**: Use gradient accumulation to simulate larger batches without OOM
3. **Gradient Clipping**: Set `max_grad_norm=1.0` to prevent gradient explosion
4. **Warmup**: Use warmup steps for training stability (typically 1000-5000 steps)
5. **Checkpointing**: Save checkpoints regularly to avoid losing progress

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
training_config.batch_size = 16

# Or use gradient accumulation
training_config.batch_size = 8
training_config.gradient_accumulation_steps = 4  # Effective batch size = 32
```

### NaN or Inf Loss

```python
# Reduce learning rate
training_config.learning_rate = 1e-5

# Increase gradient clipping
training_config.max_grad_norm = 0.5

# Disable mixed precision
training_config.use_mixed_precision = False
```

### Slow Training

```python
# Enable mixed precision
training_config.use_mixed_precision = True

# Increase batch size (if memory allows)
training_config.batch_size = 64

# Use more workers for data loading
train_loader = DataLoader(dataset, num_workers=4)
```
