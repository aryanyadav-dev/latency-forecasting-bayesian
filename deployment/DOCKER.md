# Docker Deployment Guide

This guide explains how to build and run the Latent Forecasting Network using Docker containers.

## Prerequisites

- Docker Engine 20.10+ installed
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.1+ support (for training)
- At least 20GB free disk space

## Quick Start

### 1. Build the Docker Image

```bash
# Build the image
docker build -t latent-forecasting-network:latest .

# Or use docker-compose
docker-compose build
```

### 2. Run with Docker Compose (Recommended)

```bash
# Start the container
docker-compose up -d lfn-training

# Access the container
docker-compose exec lfn-training bash

# Inside the container, run training
python main.py train experiments/configs/default.yaml
```

### 3. Run with Docker CLI

```bash
# Run interactive container with GPU support
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/experiments:/workspace/experiments \
  --shm-size=8g \
  latent-forecasting-network:latest \
  bash

# Inside the container
python main.py train experiments/configs/default.yaml
```

## Volume Mounts

The Docker setup uses the following volume mounts for data persistence:

- `./data` → `/workspace/data` - Dataset storage
- `./checkpoints` → `/workspace/checkpoints` - Model checkpoints
- `./experiments/logs` → `/workspace/experiments/logs` - Training logs
- `./experiments/results` → `/workspace/experiments/results` - Evaluation results

## GPU Support

### Verify GPU Access

```bash
# Inside the container
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Select Specific GPUs

```bash
# Use specific GPU(s)
docker run --gpus '"device=0,1"' -it latent-forecasting-network:latest

# Or set environment variable
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 -it latent-forecasting-network:latest
```

## TensorBoard Monitoring

### Start TensorBoard Service

```bash
# Start TensorBoard with docker-compose
docker-compose up -d tensorboard

# Access TensorBoard at http://localhost:6006
```

### Manual TensorBoard

```bash
# Inside the training container
tensorboard --logdir=experiments/logs --host=0.0.0.0 --port=6006
```

## Common Tasks

### Training a Model

```bash
# Start container
docker-compose exec lfn-training bash

# Train with default config
python main.py train experiments/configs/default.yaml

# Train with custom config
python main.py train experiments/configs/wikitext103.yaml --device cuda
```

### Evaluating a Model

```bash
# Evaluate a checkpoint
python main.py evaluate checkpoints/best_model.pt --output experiments/results/eval

# With visualization
python main.py analyze checkpoints/best_model.pt --visualize
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Development Workflow

### Live Code Editing

The docker-compose setup mounts the current directory, allowing live code editing:

```bash
# Start container
docker-compose up -d lfn-training

# Edit code on host machine
vim models/encoder.py

# Changes are immediately available in container
docker-compose exec lfn-training python -c "from models.encoder import Encoder; print('Updated!')"
```

### Installing Additional Packages

```bash
# Inside container
pip install <package-name>

# To persist, add to requirements.txt and rebuild
echo "new-package==1.0.0" >> requirements.txt
docker-compose build
```

## Troubleshooting

### GPU Not Detected

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify NVIDIA Docker runtime is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check Docker daemon configuration (`/etc/docker/daemon.json`):
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

3. Restart Docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

### Out of Memory Errors

**Problem**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size in config file
2. Increase shared memory size:
   ```bash
   docker run --shm-size=16g ...
   ```
3. Enable gradient checkpointing in model config

### Permission Issues

**Problem**: Cannot write to mounted volumes

**Solutions**:
1. Run container with user permissions:
   ```bash
   docker run --user $(id -u):$(id -g) ...
   ```

2. Or fix permissions on host:
   ```bash
   sudo chown -R $USER:$USER checkpoints/ experiments/
   ```

### Slow Data Loading

**Problem**: DataLoader is slow with multiple workers

**Solutions**:
1. Increase shared memory:
   ```bash
   docker run --shm-size=8g ...
   ```

2. Reduce `num_workers` in data config

## Image Management

### View Images

```bash
# List images
docker images | grep latent-forecasting

# Check image size
docker images latent-forecasting-network:latest --format "{{.Size}}"
```

### Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove image
docker rmi latent-forecasting-network:latest

# Clean up all unused images
docker image prune -a
```

### Save and Load Images

```bash
# Save image to tar file
docker save latent-forecasting-network:latest | gzip > lfn-image.tar.gz

# Load image on another machine
docker load < lfn-image.tar.gz
```

## Multi-GPU Training

### DataParallel

```bash
# Use all available GPUs
docker run --gpus all -it latent-forecasting-network:latest
python main.py train experiments/configs/default.yaml --device cuda
```

### DistributedDataParallel

```bash
# Run on 2 GPUs
docker run --gpus all -it latent-forecasting-network:latest
python -m torch.distributed.launch --nproc_per_node=2 main.py train experiments/configs/default.yaml
```

## Production Deployment

### Build Optimized Image

```bash
# Build with specific PyTorch version
docker build --build-arg PYTORCH_VERSION=2.1.0 -t lfn:prod .

# Multi-stage build for smaller image (if needed)
docker build -f Dockerfile.prod -t lfn:prod .
```

### Resource Limits

```bash
# Limit CPU and memory
docker run --gpus all \
  --cpus=8 \
  --memory=32g \
  --memory-swap=32g \
  -it latent-forecasting-network:latest
```

## Security Considerations

1. **Run as non-root user** (add to Dockerfile):
   ```dockerfile
   RUN useradd -m -u 1000 lfn
   USER lfn
   ```

2. **Read-only root filesystem**:
   ```bash
   docker run --read-only --tmpfs /tmp --tmpfs /workspace/.cache ...
   ```

3. **Network isolation**:
   ```bash
   docker run --network=none ...  # No network access
   ```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
