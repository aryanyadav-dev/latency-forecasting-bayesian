# Latent Forecasting Network - Docker Image
# Base image with CUDA support for GPU training

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/checkpoints \
    /workspace/experiments/logs \
    /workspace/experiments/results \
    /workspace/data

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden)
CMD ["/bin/bash"]