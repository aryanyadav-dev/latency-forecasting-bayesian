# Google Cloud Platform (GCP) Deployment Guide

This guide explains how to deploy and run the Latent Forecasting Network on Google Cloud Platform using Compute Engine GPU instances.

## Table of Contents

1. [Instance Types](#instance-types)
2. [Setup Instructions](#setup-instructions)
3. [Preemptible Instances](#preemptible-instances)
4. [Cloud Storage Integration](#cloud-storage-integration)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

## Instance Types

### Recommended GPU Configurations

| Machine Type | GPUs | GPU Memory | vCPUs | RAM | Cost (approx/hr) | Best For |
|--------------|------|------------|-------|-----|------------------|----------|
| **n1-highmem-8 + 1x V100** | 1x V100 | 16 GB | 8 | 52 GB | $2.48 | Small-medium models |
| **n1-highmem-16 + 2x V100** | 2x V100 | 32 GB | 16 | 104 GB | $4.96 | Multi-GPU training |
| **a2-highgpu-1g** | 1x A100 | 40 GB | 12 | 85 GB | $3.67 | Latest GPU tech |
| **n1-standard-4 + 1x T4** | 1x T4 | 16 GB | 4 | 15 GB | $0.35 | Development/testing |
| **n1-highmem-8 + 1x P100** | 1x P100 | 16 GB | 8 | 52 GB | $1.46 | Budget training |

### GPU Availability by Region

Check availability: https://cloud.google.com/compute/docs/gpus/gpu-regions-zones

**Recommended Regions** (good availability + pricing):
- `us-central1` (Iowa)
- `us-west1` (Oregon)
- `europe-west4` (Netherlands)
- `asia-east1` (Taiwan)

## Setup Instructions

### Prerequisites

1. **Install Google Cloud SDK**
   ```bash
   # Linux/macOS
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Initialize
   gcloud init
   ```

2. **Set Project and Zone**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud config set compute/zone us-central1-a
   ```

### Step 1: Create GPU Instance

```bash
# Create instance with V100 GPU
gcloud compute instances create lfn-training \
  --zone=us-central1-a \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# Wait for instance to be ready
gcloud compute instances describe lfn-training --zone=us-central1-a
```

### Step 2: Connect to Instance

```bash
# SSH into instance
gcloud compute ssh lfn-training --zone=us-central1-a

# Optional: Port forwarding for TensorBoard
gcloud compute ssh lfn-training --zone=us-central1-a -- -L 6006:localhost:6006
```

### Step 3: Setup Environment

```bash
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

### Step 4: Start Training

```bash
# Train with default config
python main.py train experiments/configs/default.yaml

# Monitor with TensorBoard
tensorboard --logdir=experiments/logs --host=0.0.0.0
```

## Preemptible Instances

Preemptible VMs can reduce costs by up to 80% but may be terminated after 24 hours or when capacity is needed.

### Creating Preemptible Instance

```bash
# Add --preemptible flag
gcloud compute instances create lfn-training-preemptible \
  --zone=us-central1-a \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --metadata="install-nvidia-driver=True"
```

### Handling Preemption

1. **Automatic Checkpointing**
   - Training script saves checkpoints every N steps
   - Resume training with `--resume` flag

2. **Preemption Notice**
   ```bash
   # Check for preemption notice (30-second warning)
   curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted" \
     -H "Metadata-Flavor: Google"
   ```

3. **Auto-Restart Script**
   ```bash
   #!/bin/bash
   # restart-training.sh
   while true; do
     python main.py train config.yaml --resume
     if [ $? -eq 0 ]; then
       break  # Training completed successfully
     fi
     echo "Training interrupted, restarting..."
     sleep 60
   done
   ```

### Preemptible Best Practices

1. **Frequent Checkpointing**: Set `checkpoint_every: 500` in config
2. **Persistent Disks**: Use separate persistent disk for checkpoints
3. **Cloud Storage Sync**: Regularly sync to Google Cloud Storage
4. **Startup Scripts**: Auto-resume training on restart

## Cloud Storage Integration

### Setup Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -l us-central1 gs://your-lfn-bucket

# Set lifecycle policy (auto-delete old checkpoints after 30 days)
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://your-lfn-bucket
```

### Sync Checkpoints to Cloud Storage

```bash
# Manual sync
gsutil -m rsync -r checkpoints/ gs://your-lfn-bucket/checkpoints/

# Automated sync (add to cron)
*/5 * * * * gsutil -m rsync -r /home/user/latent-forecasting-network/checkpoints/ gs://your-lfn-bucket/checkpoints/
```

### Download Checkpoints

```bash
# Download specific checkpoint
gsutil cp gs://your-lfn-bucket/checkpoints/best_model.pt checkpoints/

# Sync entire directory
gsutil -m rsync -r gs://your-lfn-bucket/checkpoints/ checkpoints/
```

### Mount Cloud Storage as Filesystem (FUSE)

```bash
# Install gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

# Mount bucket
mkdir -p ~/gcs-checkpoints
gcsfuse your-lfn-bucket ~/gcs-checkpoints

# Use mounted directory for checkpoints
python main.py train config.yaml --checkpoint-dir ~/gcs-checkpoints
```

## Cost Optimization

### 1. Use Preemptible Instances

```bash
# Savings: Up to 80% compared to regular instances
# Example: n1-highmem-8 + V100
# Regular: $2.48/hr → Preemptible: $0.74/hr
```

### 2. Committed Use Discounts

For long-term projects (1-3 years):
- **1-year commitment**: 25% discount
- **3-year commitment**: 52% discount

```bash
# Purchase commitment
gcloud compute commitments create lfn-commitment \
  --resources=vcpu=8,memory=52GB \
  --plan=12-month \
  --region=us-central1
```

### 3. Sustained Use Discounts

Automatic discounts for running instances >25% of month:
- **25-50% usage**: 20% discount
- **50-75% usage**: 30% discount
- **75-100% usage**: 40% discount

### 4. Right-Size Instances

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Check recommendations
gcloud compute instances get-recommendations lfn-training --zone=us-central1-a
```

### 5. Stop Instances When Not Training

```bash
# Stop instance (preserves disk)
gcloud compute instances stop lfn-training --zone=us-central1-a

# Start instance later
gcloud compute instances start lfn-training --zone=us-central1-a
```

### 6. Use Standard Persistent Disks

```bash
# pd-standard is cheaper than pd-ssd for checkpoints
# Use pd-ssd only for data that needs high IOPS
```

## Cost Estimation

### Example: Training on n1-highmem-8 + V100

| Component | Cost (Regular) | Cost (Preemptible) |
|-----------|----------------|-------------------|
| n1-highmem-8 | $0.47/hr | $0.14/hr |
| 1x V100 GPU | $2.48/hr | $0.74/hr |
| 100 GB pd-ssd | $0.17/hr | $0.17/hr |
| **Total** | **$3.12/hr** | **$1.05/hr** |

**For 24-hour training run**:
- Regular: ~$75
- Preemptible: ~$25

### Monitoring Costs

```bash
# View current costs
gcloud billing accounts list
gcloud billing projects describe YOUR_PROJECT_ID

# Set budget alerts
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="LFN Training Budget" \
  --budget-amount=500USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, reinstall driver
sudo /opt/deeplearning/install-driver.sh

# Verify CUDA
nvcc --version
```

### Quota Exceeded

```bash
# Check quotas
gcloud compute project-info describe --project=YOUR_PROJECT_ID

# Request quota increase
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Filter by "GPUs (all regions)"
# Select and click "EDIT QUOTAS"
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Solutions:
# 1. Reduce batch size
# 2. Enable gradient checkpointing
# 3. Use larger GPU (V100 → A100)
```

### Slow Data Loading

```bash
# Check disk performance
sudo iostat -x 1

# Upgrade to pd-ssd
gcloud compute disks create lfn-data \
  --size=100GB \
  --type=pd-ssd \
  --zone=us-central1-a
```

### Preemption Too Frequent

```bash
# Try different zone
gcloud compute zones list --filter="region:us-central1"

# Or use regular instance for critical runs
```

## Advanced Configuration

### Multi-GPU Training

```bash
# Create instance with multiple GPUs
gcloud compute instances create lfn-multi-gpu \
  --zone=us-central1-a \
  --machine-type=n1-highmem-16 \
  --accelerator=type=nvidia-tesla-v100,count=4 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# Run distributed training
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  main.py train experiments/configs/default.yaml
```

### Startup Script for Auto-Resume

```bash
# Create startup script
cat > startup-script.sh << 'EOF'
#!/bin/bash
cd /home/user/latent-forecasting-network
source venv/bin/activate
python main.py train experiments/configs/default.yaml --resume
EOF

# Add to instance metadata
gcloud compute instances add-metadata lfn-training \
  --zone=us-central1-a \
  --metadata-from-file startup-script=startup-script.sh
```

### Cloud Monitoring

```bash
# Install monitoring agent
curl -sSO https://dl.google.com/cloudagents/add-monitoring-agent-repo.sh
sudo bash add-monitoring-agent-repo.sh
sudo apt-get update
sudo apt-get install stackdriver-agent

# Start agent
sudo service stackdriver-agent start
```

### Managed Instance Groups (Auto-Scaling)

```bash
# Create instance template
gcloud compute instance-templates create lfn-template \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# Create managed instance group
gcloud compute instance-groups managed create lfn-group \
  --base-instance-name=lfn \
  --template=lfn-template \
  --size=1 \
  --zone=us-central1-a
```

## Additional Resources

- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [Deep Learning VM Images](https://cloud.google.com/deep-learning-vm)
- [GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [Preemptible VM Documentation](https://cloud.google.com/compute/docs/instances/preemptible)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Cost Management](https://cloud.google.com/cost-management)
