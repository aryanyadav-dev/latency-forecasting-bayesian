#!/bin/bash
# Latent Forecasting Network - Environment Setup Script
# For Linux and macOS systems

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    print_section "Latent Forecasting Network - Environment Setup"
    
    # Step 1: Check Python version
    print_section "Step 1: Checking Python Version"
    check_python_version
    
    # Step 2: Check CUDA availability
    print_section "Step 2: Checking CUDA Availability"
    check_cuda
    
    # Step 3: Create virtual environment
    print_section "Step 3: Creating Virtual Environment"
    create_virtual_environment
    
    # Step 4: Install dependencies
    print_section "Step 4: Installing Dependencies"
    install_dependencies
    
    # Step 5: Verify installation
    print_section "Step 5: Verifying Installation"
    verify_installation
    
    # Step 6: Create necessary directories
    print_section "Step 6: Creating Project Directories"
    create_directories
    
    # Step 7: Download datasets (optional)
    print_section "Step 7: Dataset Setup"
    setup_datasets
    
    # Final message
    print_section "Setup Complete!"
    print_success_message
}

# Check Python version (3.9+)
check_python_version() {
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    print_info "Found Python $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python 3.9 or higher is required. Found Python $PYTHON_VERSION"
        exit 1
    fi
    
    print_info "✓ Python version check passed"
}

# Check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        print_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        
        # Check CUDA version
        if command_exists nvcc; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
            print_info "CUDA version: $CUDA_VERSION"
        else
            print_warning "nvcc not found. CUDA toolkit may not be installed."
        fi
        
        print_info "✓ GPU support available"
    else
        print_warning "No NVIDIA GPU detected. Training will use CPU (much slower)."
        print_warning "For GPU support, install NVIDIA drivers and CUDA toolkit."
    fi
}

# Create virtual environment
create_virtual_environment() {
    ENV_NAME="venv"
    
    if [ -d "$ENV_NAME" ]; then
        print_warning "Virtual environment '$ENV_NAME' already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$ENV_NAME"
        else
            print_info "Using existing virtual environment."
            return
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv "$ENV_NAME"
    
    print_info "✓ Virtual environment created: $ENV_NAME"
    print_info "To activate: source $ENV_NAME/bin/activate"
}

# Install dependencies
install_dependencies() {
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_info "✓ Dependencies installed successfully"
}

# Verify installation
verify_installation() {
    print_info "Verifying PyTorch installation..."
    source venv/bin/activate
    
    python3 << EOF
import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("✓ PyTorch installation verified")
except ImportError as e:
    print(f"✗ Error importing PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("✓ Transformers installation verified")
except ImportError as e:
    print(f"✗ Error importing Transformers: {e}")
    sys.exit(1)

try:
    import numpy as np
    import sklearn
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print("✓ Scientific libraries verified")
except ImportError as e:
    print(f"✗ Error importing scientific libraries: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_info "✓ All packages verified successfully"
    else
        print_error "Package verification failed"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    DIRS=(
        "checkpoints"
        "experiments/logs"
        "experiments/results"
        "experiments/configs"
        "data"
    )
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_info "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
    
    print_info "✓ Project directories ready"
}

# Setup datasets
setup_datasets() {
    print_info "Dataset setup options:"
    echo "  1. Download datasets now (requires internet)"
    echo "  2. Skip dataset download (download during training)"
    echo ""
    read -p "Choose option (1/2): " -n 1 -r
    echo ""
    
    if [[ $REPLY == "1" ]]; then
        print_info "Downloading datasets..."
        source venv/bin/activate
        
        python3 << EOF
from datasets import load_dataset
import os

# Set cache directory
cache_dir = os.path.join(os.getcwd(), "data", ".cache")
os.makedirs(cache_dir, exist_ok=True)

print("Downloading WikiText-2...")
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    print("✓ WikiText-2 downloaded")
except Exception as e:
    print(f"✗ Error downloading WikiText-2: {e}")

print("\nDatasets will be cached in: data/.cache")
EOF
        
        print_info "✓ Dataset download complete"
    else
        print_info "Skipping dataset download. Datasets will be downloaded during first training run."
    fi
}

# Print success message
print_success_message() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete! ✓                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     ${GREEN}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Run tests to verify everything works:"
    echo "     ${GREEN}pytest tests/ -v${NC}"
    echo ""
    echo "  3. Start training with default configuration:"
    echo "     ${GREEN}python main.py train experiments/configs/default.yaml${NC}"
    echo ""
    echo "  4. Monitor training with TensorBoard:"
    echo "     ${GREEN}tensorboard --logdir=experiments/logs${NC}"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Troubleshooting section
show_troubleshooting() {
    echo ""
    echo "=========================================="
    echo "Troubleshooting"
    echo "=========================================="
    echo ""
    echo "Common issues and solutions:"
    echo ""
    echo "1. CUDA not available:"
    echo "   - Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx"
    echo "   - Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "2. Out of memory errors:"
    echo "   - Reduce batch_size in config file"
    echo "   - Enable gradient checkpointing"
    echo "   - Use mixed precision training (enabled by default)"
    echo ""
    echo "3. Dataset download fails:"
    echo "   - Check internet connection"
    echo "   - Clear HuggingFace cache: rm -rf ~/.cache/huggingface"
    echo "   - Try manual download"
    echo ""
    echo "4. Import errors:"
    echo "   - Ensure virtual environment is activated"
    echo "   - Reinstall dependencies: pip install -r requirements.txt --force-reinstall"
    echo ""
}

# Parse command line arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Latent Forecasting Network - Setup Script"
    echo ""
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --troubleshoot      Show troubleshooting guide"
    echo ""
    exit 0
elif [ "$1" == "--troubleshoot" ]; then
    show_troubleshooting
    exit 0
fi

# Run main setup
main
