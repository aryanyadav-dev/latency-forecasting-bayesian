# Azure Deployment Guide

This guide explains how to deploy and run the Latent Forecasting Network on Microsoft Azure using GPU-enabled Virtual Machines.

## Table of Contents

1. [VM Sizes](#vm-sizes)
2. [Setup Instructions](#setup-instructions)
3. [Low-Priority VMs](#low-priority-vms)
4. [Azure Blob Storage Integration](#azure-blob-storage-integration)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

## VM Sizes

### Recommended GPU VM Series

| VM Size | GPUs | GPU Memory | vCPUs | RAM | Cost (approx/hr) | Best For |
|---------|------|------------|-------|-----|------------------|----------|
| **Standard_NC6s_v3** | 1x V100 | 16 GB | 6 | 112 GB | $3.06 | Small-medium models |
| **Standard_NC12s_v3** | 2x V100 | 32 GB | 12 | 224 GB | $6.12 | Multi-GPU training |
| **Standard_NC24s_v3** | 4x V100 | 64 GB | 24 | 448 GB | $12.24 | Large-scale training |
| **Standard_NC4as_T4_v3** | 1x T4 | 16 GB | 4 | 28 GB | $0.526 | Development/testing |
| **Standard_ND96asr_v4** | 8x A100 | 320 GB | 96 | 900 GB | $27.20 | Cutting-edge research |

### VM Series Overview

- **NC-series**: NVIDIA Tesla K80 (older, budget-friendly)
- **NCv3-series**: NVIDIA Tesla V100 (recommended for training)
- **NCasT4_v3-series**: NVIDIA T4 (cost-effective)
- **ND-series**: NVIDIA Tesla P40 (inference-optimized)
- **NDv2-series**: NVIDIA Tesla V100 (InfiniBand for multi-node)
- **NDv4-series**: NVIDIA A100 (latest, most powerful)

### Regional Availability

Check availability: https://azure.microsoft.com/en-us/global-infrastructure/services/?products=virtual-machines

**Recommended Regions**:
- East US
- West US 2
- West Europe
- Southeast Asia

## Setup Instructions

### Prerequisites

1. **Install Azure CLI**
   ```bash
   # Linux/macOS
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # Windows (PowerShell)
   Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
   Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'
   
   # Login
   az login
   ```

2. **Set Subscription**
   ```bash
   az account list --output table
   az account set --subscription "YOUR_SUBSCRIPTION_ID"
   ```

### Step 1: Create Resource Group

```bash
# Create resource group
az group create \
  --name lfn-resources \
  --location eastus
```

### Step 2: Create GPU VM

```bash
# Create VM with V100 GPU
az vm create \
  --resource-group lfn-resources \
  --name lfn-training \
  --location eastus \
  --size Standard_NC6s_v3 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard \
  --storage-sku Premium_LRS \
  --os-disk-size-gb 128

# Open port for TensorBoard (optional)
az vm open-port \
  --resource-group lfn-resources \
  --name lfn-training \
  --port 6006
```

### Step 3: Connect to VM

```bash
# Get public IP
az vm show \
  --resource-group lfn-resources \
  --name lfn-training \
  --show-details \
  --query publicIps \
  --output tsv

# SSH into VM
ssh azureuser@<public-ip>

# Optional: Port forwarding for TensorBoard
ssh -L 6006:localhost:6006 azureuser@<public-ip>
```

### Step 4: Setup Environment

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Clone repository
git clone https://github.com/your-org/latent-forecasting-network.git
cd latent-forecasting-network

# Run setup script
./setup.sh

# Activate environment
source venv/bin/activate

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi
```

### Step 5: Start Training

```bash
# Train with default config
python main.py train experiments/configs/default.yaml

# Monitor with TensorBoard
tensorboard --logdir=experiments/logs --host=0.0.0.0
```

## Low-Priority VMs

Low-priority VMs can reduce costs by up to 80% but may be evicted when Azure needs capacity.

### Creating Low-Priority VM

```bash
# Add --priority Spot flag
az vm create \
  --resource-group lfn-resources \
  --name lfn-training-spot \
  --location eastus \
  --size Standard_NC6s_v3 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --priority Spot \
  --max-price -1 \
  --eviction-policy Deallocate \
  --storage-sku Premium_LRS \
  --os-disk-size-gb 128
```

### Handling Evictions

1. **Eviction Notice**
   ```bash
   # Check for eviction notice (30-second warning)
   curl -H Metadata:true \
     "http://169.254.169.254/metadata/scheduledevents?api-version=2019-08-01"
   ```

2. **Auto-Resume Training**
   ```bash
   # Training script with auto-resume
   while true; do
     python main.py train config.yaml --resume
     if [ $? -eq 0 ]; then
       break
     fi
     echo "Training interrupted, waiting for VM restart..."
     sleep 300
   done
   ```

3. **Eviction Policy**
   - **Deallocate**: VM is stopped, disk preserved (recommended)
   - **Delete**: VM and disk are deleted (cheaper but risky)

### Low-Priority Best Practices

1. **Frequent Checkpointing**: Save every 500 steps
2. **Managed Disks**: Use separate managed disk for checkpoints
3. **Blob Storage Sync**: Regularly sync to Azure Blob Storage
4. **Max Price**: Set `--max-price` to control costs

## Azure Blob Storage Integration

### Setup Storage Account

```bash
# Create storage account
az storage account create \
  --name lfnstorageaccount \
  --resource-group lfn-resources \
  --location eastus \
  --sku Standard_LRS

# Get connection string
az storage account show-connection-string \
  --name lfnstorageaccount \
  --resource-group lfn-resources \
  --output tsv

# Create container
az storage container create \
  --name checkpoints \
  --account-name lfnstorageaccount
```

### Sync Checkpoints to Blob Storage

```bash
# Install Azure CLI on VM (if not already installed)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login with managed identity (if configured)
az login --identity

# Or set connection string
export AZURE_STORAGE_CONNECTION_STRING="<connection-string>"

# Upload checkpoint
az storage blob upload \
  --container-name checkpoints \
  --file checkpoints/best_model.pt \
  --name best_model.pt \
  --account-name lfnstorageaccount

# Sync directory
az storage blob upload-batch \
  --destination checkpoints \
  --source checkpoints/ \
  --account-name lfnstorageaccount
```

### Download Checkpoints

```bash
# Download specific checkpoint
az storage blob download \
  --container-name checkpoints \
  --name best_model.pt \
  --file checkpoints/best_model.pt \
  --account-name lfnstorageaccount

# Download all checkpoints
az storage blob download-batch \
  --destination checkpoints/ \
  --source checkpoints \
  --account-name lfnstorageaccount
```

### Mount Blob Storage (BlobFuse)

```bash
# Install blobfuse
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse

# Create mount point and temp directory
sudo mkdir -p /mnt/blobstorage
sudo mkdir -p /mnt/blobfusetmp

# Create config file
cat > fuse_connection.cfg << EOF
accountName lfnstorageaccount
accountKey <your-account-key>
containerName checkpoints
EOF

chmod 600 fuse_connection.cfg

# Mount
sudo blobfuse /mnt/blobstorage \
  --tmp-path=/mnt/blobfusetmp \
  --config-file=fuse_connection.cfg \
  -o attr_timeout=240 \
  -o entry_timeout=240 \
  -o negative_timeout=120

# Use mounted directory
python main.py train config.yaml --checkpoint-dir /mnt/blobstorage
```

## Cost Optimization

### 1. Use Low-Priority VMs

```bash
# Savings: Up to 80% compared to regular VMs
# Example: Standard_NC6s_v3
# Regular: $3.06/hr → Spot: $0.61/hr
```

### 2. Reserved Instances

For long-term projects (1-3 years):
- **1-year**: 40% discount
- **3-year**: 60% discount

```bash
# Purchase reservation (via Azure Portal)
# Navigate to: Reservations → Add → Virtual Machines
```

### 3. Azure Hybrid Benefit

If you have Windows Server licenses:
- Save up to 40% on Windows VMs

### 4. Right-Size VMs

```bash
# Get VM recommendations
az advisor recommendation list \
  --category Cost \
  --output table

# Resize VM
az vm deallocate \
  --resource-group lfn-resources \
  --name lfn-training

az vm resize \
  --resource-group lfn-resources \
  --name lfn-training \
  --size Standard_NC4as_T4_v3

az vm start \
  --resource-group lfn-resources \
  --name lfn-training
```

### 5. Stop VMs When Not Training

```bash
# Deallocate VM (stops billing for compute)
az vm deallocate \
  --resource-group lfn-resources \
  --name lfn-training

# Start VM later
az vm start \
  --resource-group lfn-resources \
  --name lfn-training
```

### 6. Use Standard Storage for Checkpoints

```bash
# Standard_LRS is cheaper than Premium_LRS
# Use Premium only for high-IOPS workloads
```

### 7. Auto-Shutdown

```bash
# Enable auto-shutdown at specific time
az vm auto-shutdown \
  --resource-group lfn-resources \
  --name lfn-training \
  --time 2300 \
  --email "your-email@example.com"
```

## Cost Estimation

### Example: Training on Standard_NC6s_v3

| Component | Cost (Regular) | Cost (Spot) |
|-----------|----------------|-------------|
| Standard_NC6s_v3 | $3.06/hr | $0.61/hr |
| 128 GB Premium SSD | $0.20/hr | $0.20/hr |
| Public IP | $0.004/hr | $0.004/hr |
| **Total** | **$3.26/hr** | **$0.81/hr** |

**For 24-hour training run**:
- Regular: ~$78
- Spot: ~$19

### Monitoring Costs

```bash
# View current costs
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --output table

# Set budget
az consumption budget create \
  --budget-name lfn-budget \
  --amount 500 \
  --time-grain Monthly \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --resource-group lfn-resources
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, reinstall driver
sudo apt-get install --reinstall nvidia-driver-525

# Verify CUDA
nvcc --version
```

### Quota Exceeded

```bash
# Check quotas
az vm list-usage \
  --location eastus \
  --output table

# Request quota increase
# Go to: Azure Portal → Subscriptions → Usage + quotas
# Search for "Standard NCv3 Family vCPUs"
# Click "Request increase"
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Solutions:
# 1. Reduce batch size
# 2. Enable gradient checkpointing
# 3. Use larger VM (NC6s_v3 → NC12s_v3)
```

### Slow Data Loading

```bash
# Check disk performance
sudo iostat -x 1

# Upgrade to Premium SSD
az disk update \
  --resource-group lfn-resources \
  --name lfn-training_OsDisk_1 \
  --sku Premium_LRS
```

### SSH Connection Issues

```bash
# Reset SSH
az vm user reset-ssh \
  --resource-group lfn-resources \
  --name lfn-training

# Check NSG rules
az network nsg rule list \
  --resource-group lfn-resources \
  --nsg-name lfn-trainingNSG \
  --output table
```

## Advanced Configuration

### Multi-GPU Training

```bash
# Create VM with multiple GPUs
az vm create \
  --resource-group lfn-resources \
  --name lfn-multi-gpu \
  --location eastus \
  --size Standard_NC24s_v3 \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --storage-sku Premium_LRS

# Run distributed training
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  main.py train experiments/configs/default.yaml
```

### Managed Identity for Storage Access

```bash
# Enable managed identity
az vm identity assign \
  --resource-group lfn-resources \
  --name lfn-training

# Grant storage access
PRINCIPAL_ID=$(az vm show \
  --resource-group lfn-resources \
  --name lfn-training \
  --query identity.principalId \
  --output tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/<subscription-id>/resourceGroups/lfn-resources/providers/Microsoft.Storage/storageAccounts/lfnstorageaccount"

# Access storage without keys
az login --identity
az storage blob list --container-name checkpoints --account-name lfnstorageaccount
```

### Azure Monitor

```bash
# Enable monitoring
az monitor diagnostic-settings create \
  --resource /subscriptions/<subscription-id>/resourceGroups/lfn-resources/providers/Microsoft.Compute/virtualMachines/lfn-training \
  --name lfn-diagnostics \
  --logs '[{"category": "Administrative", "enabled": true}]' \
  --metrics '[{"category": "AllMetrics", "enabled": true}]' \
  --workspace <log-analytics-workspace-id>
```

### VM Scale Sets (Auto-Scaling)

```bash
# Create scale set
az vmss create \
  --resource-group lfn-resources \
  --name lfn-scaleset \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --vm-sku Standard_NC6s_v3 \
  --instance-count 1 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --priority Spot \
  --eviction-policy Deallocate

# Configure autoscale
az monitor autoscale create \
  --resource-group lfn-resources \
  --resource lfn-scaleset \
  --resource-type Microsoft.Compute/virtualMachineScaleSets \
  --name lfn-autoscale \
  --min-count 0 \
  --max-count 10 \
  --count 1
```

## Additional Resources

- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [Data Science Virtual Machines](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/)
- [GPU VM Sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)
- [Low-Priority VMs](https://docs.microsoft.com/en-us/azure/virtual-machines/spot-vms)
- [Azure Blob Storage](https://docs.microsoft.com/en-us/azure/storage/blobs/)
- [Cost Management](https://azure.microsoft.com/en-us/services/cost-management/)
