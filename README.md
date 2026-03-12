# Latent Forecasting Network (LFN)

A PyTorch implementation of Latent Forecasting Networks for predictive representation learning in neural sequence models.

## Overview

The Latent Forecasting Network (LFN) is a novel architecture that learns to predict the evolution of its own internal latent representations across multiple time horizons, in addition to standard next-token prediction. This dual objective encourages the model to develop more structured, interpretable, and predictable internal representations.

### Key Features

- **Multi-Horizon Latent Forecasting**: Predicts future latent states at horizons k ∈ {1, 2, 5, 10}
- **Transformer-Based Architecture**: Built on proven transformer encoder architecture
- **Comprehensive Evaluation**: Includes perplexity, Latent Predictability Score (LPS), and representation quality metrics
- **Flexible Training**: Supports mixed precision, gradient accumulation, and multi-GPU training
- **Cloud-Ready**: Docker support and deployment guides for AWS, GCP, and Azure
- **Research-Focused**: Designed for experimentation with ablation study support

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/latent-forecasting-network.git
cd latent-forecasting-network

# Run setup script
./setup.sh

# Activate environment
source venv/bin/activate

# Verify installation
python quick_test.py
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (for training)

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Training

```bash
# Train with default configuration (WikiText-2)
python main.py train experiments/configs/default.yaml

# Train with custom config
python main.py train experiments/configs/wikitext103.yaml --device cuda

# Resume from checkpoint
python main.py train experiments/configs/default.yaml --resume
```

### Evaluation

```bash
# Evaluate a trained model
python main.py evaluate checkpoints/best_model.pt

# With detailed analysis
python main.py analyze checkpoints/best_model.pt --visualize
```

### Monitoring

```bash
# Start TensorBoard
tensorboard --logdir=experiments/logs

# Access at http://localhost:6006
```

## Architecture

The LFN consists of three main components:

1. **Encoder**: Transforms input tokens into latent representations
   - Token embedding with positional encoding
   - Multi-layer transformer encoder
   - Causal masking for autoregressive modeling

2. **Latent Forecasting Network**: Predicts future latent states
   - Separate prediction heads for each horizon
   - Multi-horizon forecasting (k = 1, 2, 5, 10)
   - Maintains temporal consistency

3. **Decoder**: Maps latents to vocabulary logits
   - Linear projection layer
   - Standard cross-entropy loss

### Loss Function

```
L_total = L_token + λ * L_latent

where:
- L_token: Cross-entropy loss for next-token prediction
- L_latent: MSE loss for latent forecasting
- λ: Configurable weight (default: 0.1)
```

## Configuration

Configuration files are in YAML format. Example:

```yaml
model:
  vocab_size: 50257
  latent_dim: 512
  num_layers: 6
  num_heads: 8
  hidden_dim: 2048
  dropout: 0.1
  forecast_horizons: [1, 2, 5, 10]

training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  lambda_latent: 0.1
  use_mixed_precision: true

data:
  dataset_name: "wikitext-2"
  context_length: 512
  stride: 256
```

See `experiments/configs/` for more examples.

## Evaluation Metrics

### Language Modeling
- **Perplexity**: Standard language modeling metric
- **Token Accuracy**: Next-token prediction accuracy
- **Cross-Entropy Loss**: Average per-token loss

### Latent Forecasting
- **MSE**: Mean squared error for each horizon
- **LPS (Latent Predictability Score)**: L2 distance between predicted and actual latents

### Representation Quality
- **Latent Entropy**: Measures representation diversity
- **Latent Variance**: Measures representation spread
- **Cosine Similarity Drift**: Measures representation stability
- **Cluster Separability**: Measures representation structure (silhouette score)

## Project Structure

```
latent-forecasting-network/
├── data/                   # Data loading and tokenization
├── models/                 # Model architectures
│   ├── encoder.py
│   ├── forecasting_network.py
│   ├── decoder.py
│   └── complete_model.py
├── training/               # Training utilities
│   ├── trainer.py
│   ├── loss_functions.py
│   ├── optimizer.py
│   └── scheduler.py
├── evaluation/             # Evaluation and analysis
│   ├── metrics.py
│   └── latent_analysis.py
├── experiments/            # Configs, logs, and results
│   ├── configs/
│   ├── logs/
│   └── results/
├── deployment/             # Deployment guides
│   ├── DOCKER.md
│   ├── AWS_SETUP.md
│   ├── GCP_SETUP.md
│   └── AZURE_SETUP.md
├── tests/                  # Test suite
├── main.py                 # CLI entry point
└── quick_test.py           # Fast validation script
```

## Docker Deployment

```bash
# Build image
docker-compose build

# Run training
docker-compose up -d lfn-training
docker-compose exec lfn-training python main.py train experiments/configs/default.yaml

# Start TensorBoard
docker-compose up -d tensorboard
```

See `deployment/DOCKER.md` for detailed instructions.

## Cloud Deployment

Comprehensive deployment guides available for:

- **AWS EC2**: `deployment/AWS_SETUP.md`
  - Spot instances (up to 90% cost savings)
  - S3 integration
  - Multi-GPU support

- **Google Cloud**: `deployment/GCP_SETUP.md`
  - Preemptible instances (up to 80% savings)
  - Cloud Storage integration
  - Regional availability

- **Microsoft Azure**: `deployment/AZURE_SETUP.md`
  - Low-priority VMs (up to 80% savings)
  - Blob Storage integration
  - Managed identities

## Development

### Running Tests

```bash
# Quick validation (fast, recommended for development)
python quick_test.py

# Full test suite (comprehensive but slow, downloads datasets)
# Note: Full test suite available but not run due to computational constraints
# Tests validate: data pipeline, model training, evaluation, checkpointing
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

**Testing Status**: See [TESTING_STATUS.md](TESTING_STATUS.md) for detailed information.

- Quick validation: All modules import and basic functionality verified
- Full test suite: Available but requires GPU and dataset downloads (~10-15 min)
- Integration testing: Planned for future work with computational resources

### Code Quality

```bash
# Format code
black .

# Lint
flake8 models/ data/ training/ evaluation/

# Type checking
mypy models/ data/ training/ evaluation/
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

- Reduce `batch_size` in config
- Enable gradient checkpointing
- Use mixed precision training (enabled by default)

### Slow Training

- Increase `num_workers` in data config
- Use SSD storage for datasets
- Enable mixed precision training
- Check GPU utilization with `nvidia-smi`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lfn2024,
  title={Predictive Representation Learning via Latent Forecasting Networks},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Uses [HuggingFace Transformers](https://huggingface.co/transformers/) for tokenization
- Datasets from [HuggingFace Datasets](https://huggingface.co/datasets)

## Future Work

Due to computational resource constraints, the following experimental validation is planned for future work:

- **Baseline Comparisons**: Training standard Transformer vs LFN on WikiText-2, WikiText-103, and TinyStories
- **Ablation Studies**: Systematic evaluation of λ values, forecasting horizons, and latent dimensions
- **Performance Benchmarking**: Throughput, memory usage, and training time analysis
- **Representation Analysis**: Comprehensive evaluation of learned representations
- **Multi-Dataset Validation**: Cross-dataset generalization studies

Expected results based on similar architectures suggest that latent forecasting should improve representation quality while maintaining competitive perplexity scores.



