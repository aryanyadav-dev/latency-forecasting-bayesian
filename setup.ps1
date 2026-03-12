# Latent Forecasting Network - Environment Setup Script
# For Windows PowerShell

# Requires PowerShell 5.1 or higher

param(
    [switch]$Help,
    [switch]$Troubleshoot
)

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
}

# Check if command exists
function Test-CommandExists {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Show help
function Show-Help {
    Write-Host "Latent Forecasting Network - Setup Script"
    Write-Host ""
    Write-Host "Usage: .\setup.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help              Show this help message"
    Write-Host "  -Troubleshoot      Show troubleshooting guide"
    Write-Host ""
}

# Show troubleshooting
function Show-Troubleshooting {
    Write-Section "Troubleshooting"
    Write-Host ""
    Write-Host "Common issues and solutions:"
    Write-Host ""
    Write-Host "1. CUDA not available:"
    Write-Host "   - Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx"
    Write-Host "   - Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    Write-Host ""
    Write-Host "2. Python not found:"
    Write-Host "   - Install Python 3.9+: https://www.python.org/downloads/"
    Write-Host "   - Add Python to PATH during installation"
    Write-Host ""
    Write-Host "3. Virtual environment activation fails:"
    Write-Host "   - Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Write-Host ""
    Write-Host "4. Out of memory errors:"
    Write-Host "   - Reduce batch_size in config file"
    Write-Host "   - Enable gradient checkpointing"
    Write-Host "   - Use mixed precision training (enabled by default)"
    Write-Host ""
}

# Check Python version
function Test-PythonVersion {
    Write-Section "Step 1: Checking Python Version"
    
    if (-not (Test-CommandExists "python")) {
        Write-Error-Custom "Python is not installed or not in PATH."
        Write-Error-Custom "Please install Python 3.9 or higher from https://www.python.org/downloads/"
        exit 1
    }
    
    $pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    Write-Info "Found Python $pythonVersion"
    
    $versionParts = $pythonVersion.Split('.')
    $major = [int]$versionParts[0]
    $minor = [int]$versionParts[1]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Error-Custom "Python 3.9 or higher is required. Found Python $pythonVersion"
        exit 1
    }
    
    Write-Info "✓ Python version check passed"
}

# Check CUDA availability
function Test-CUDA {
    Write-Section "Step 2: Checking CUDA Availability"
    
    if (Test-CommandExists "nvidia-smi") {
        Write-Info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        
        Write-Info "✓ GPU support available"
    } else {
        Write-Warning-Custom "No NVIDIA GPU detected. Training will use CPU (much slower)."
        Write-Warning-Custom "For GPU support, install NVIDIA drivers and CUDA toolkit."
    }
}

# Create virtual environment
function New-VirtualEnvironment {
    Write-Section "Step 3: Creating Virtual Environment"
    
    $envName = "venv"
    
    if (Test-Path $envName) {
        Write-Warning-Custom "Virtual environment '$envName' already exists."
        $response = Read-Host "Do you want to recreate it? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Write-Info "Removing existing virtual environment..."
            Remove-Item -Recurse -Force $envName
        } else {
            Write-Info "Using existing virtual environment."
            return
        }
    }
    
    Write-Info "Creating virtual environment..."
    python -m venv $envName
    
    Write-Info "✓ Virtual environment created: $envName"
    Write-Info "To activate: .\venv\Scripts\Activate.ps1"
}

# Install dependencies
function Install-Dependencies {
    Write-Section "Step 4: Installing Dependencies"
    
    Write-Info "Activating virtual environment..."
    & .\venv\Scripts\Activate.ps1
    
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    if (-not (Test-Path "requirements.txt")) {
        Write-Error-Custom "requirements.txt not found!"
        exit 1
    }
    
    Write-Info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    Write-Info "✓ Dependencies installed successfully"
}

# Verify installation
function Test-Installation {
    Write-Section "Step 5: Verifying Installation"
    
    Write-Info "Verifying PyTorch installation..."
    & .\venv\Scripts\Activate.ps1
    
    $verifyScript = @"
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
"@
    
    python -c $verifyScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "✓ All packages verified successfully"
    } else {
        Write-Error-Custom "Package verification failed"
        exit 1
    }
}

# Create necessary directories
function New-ProjectDirectories {
    Write-Section "Step 6: Creating Project Directories"
    
    $dirs = @(
        "checkpoints",
        "experiments\logs",
        "experiments\results",
        "experiments\configs",
        "data"
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        } else {
            Write-Info "Directory already exists: $dir"
        }
    }
    
    Write-Info "✓ Project directories ready"
}

# Setup datasets
function Initialize-Datasets {
    Write-Section "Step 7: Dataset Setup"
    
    Write-Info "Dataset setup options:"
    Write-Host "  1. Download datasets now (requires internet)"
    Write-Host "  2. Skip dataset download (download during training)"
    Write-Host ""
    $choice = Read-Host "Choose option (1/2)"
    
    if ($choice -eq "1") {
        Write-Info "Downloading datasets..."
        & .\venv\Scripts\Activate.ps1
        
        $datasetScript = @"
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

print("\nDatasets will be cached in: data\.cache")
"@
        
        python -c $datasetScript
        Write-Info "✓ Dataset download complete"
    } else {
        Write-Info "Skipping dataset download. Datasets will be downloaded during first training run."
    }
}

# Print success message
function Write-SuccessMessage {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    Setup Complete! ✓                       ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host ""
    Write-Host "  1. Activate the virtual environment:" -ForegroundColor Cyan
    Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Green
    Write-Host ""
    Write-Host "  2. Run tests to verify everything works:" -ForegroundColor Cyan
    Write-Host "     pytest tests\ -v" -ForegroundColor Green
    Write-Host ""
    Write-Host "  3. Start training with default configuration:" -ForegroundColor Cyan
    Write-Host "     python main.py train experiments\configs\default.yaml" -ForegroundColor Green
    Write-Host ""
    Write-Host "  4. Monitor training with TensorBoard:" -ForegroundColor Cyan
    Write-Host "     tensorboard --logdir=experiments\logs" -ForegroundColor Green
    Write-Host ""
    Write-Host "For more information, see README.md"
    Write-Host ""
}

# Main setup function
function Start-Setup {
    Write-Section "Latent Forecasting Network - Environment Setup"
    
    # Check execution policy
    $policy = Get-ExecutionPolicy
    if ($policy -eq "Restricted") {
        Write-Warning-Custom "Execution policy is Restricted. You may need to run:"
        Write-Warning-Custom "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    }
    
    # Run setup steps
    Test-PythonVersion
    Test-CUDA
    New-VirtualEnvironment
    Install-Dependencies
    Test-Installation
    New-ProjectDirectories
    Initialize-Datasets
    
    # Final message
    Write-Section "Setup Complete!"
    Write-SuccessMessage
}

# Parse command line arguments
if ($Help) {
    Show-Help
    exit 0
}

if ($Troubleshoot) {
    Show-Troubleshooting
    exit 0
}

# Run main setup
try {
    Start-Setup
} catch {
    Write-Error-Custom "Setup failed with error: $_"
    Write-Host ""
    Write-Host "Run '.\setup.ps1 -Troubleshoot' for help"
    exit 1
}
