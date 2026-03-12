# AWS Deployment Guide

This guide explains how to deploy and run the Latent Forecasting Network on Amazon Web Services (AWS) using EC2 GPU instances.

## Table of Contents

1. [Instance Types](#instance-types)
2. [Setup Instructions](#setup-instructions)
3. [Spot Instances](#spot-instances)
4. [S3 Integration](#s3-integration)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

## Instance Types

### Recommended GPU Instances

| Instance Type | GPUs | GPU Memory | vCPUs | RAM | Cost (approx/hr) | Best For |
|--------------|------|------------|-------|-----|------------------|----------|
| **p3.2xlarge** | 1x V100 | 16 GB | 8 | 61 GB | $3.06 | Small-medium models |
| **p3.8xlarge** | 4x V100 | 64 GB | 32 | 244 GB | $12.24 | Multi-GPU training |
| **p4d.24xlarge** | 8x A100 | 320 GB | 96 | 1152 GB | $32.77 | Large-scale experiments |
| **g4dn.xlarge** | 1x T4 | 16 GB | 4 | 16 GB | $0.526 | Development/testing |
| **g5.xlarge** | 1x A10G | 24 GB | 4 | 16 GB | $1.006 | Cost-effective training |

### Choosing an Instance

- **Development/Testing**: g4dn.xlarge or g5.xlarge
- **Production Training**: p3.2xlarge or p3.8xlarge
- **Large Models**: p4d.24xlarge
- **Budget-Conscious**: Use Spot Instances (see below)

## Setup Instructions

### Step 1: Launch EC2 Instance

1. **Go to EC2 Dashboard**
   - Navigate to AWS Console → EC2
   - Click "Launch Instance"

2. **Choose AMI**
   - Select "Deep Learning AMI (Ubuntu 20.04)" from AWS Marketplace
   - This includes CUDA, cuDNN, and PyTorch pre-installed

3. **Choose Instance Type**
   - Select appropriate GPU instance (e.g., p3.2xlarge)

4. **Configure Instance**
   - Storage: At least 100 GB EBS volume (gp3 recommended)
   - Security Group: Allow SSH (port 22) and optionally TensorBoard (port 6006)

5. **Launch**
   - Create or select an existing key pair
   - Download the .pem file if creating new

### Step 2: Connect to Instance

```bash
# Set permissions on key file
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Optional: Enable port forwarding for TensorBoard
ssh -i your-key.pem -L 6006:localhost:6006 ubuntu@<instance-public-ip>
```

### Step 3: Setup Environment

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
```

### Step 4: Start Training

```bash
# Train with default config
python main.py train experiments/configs/default.yaml

# Monitor with TensorBoard (access at http://localhost:6006)
tensorboard --logdir=experiments/logs --host=0.0.0.0
```

## Spot Instances

Spot instances can reduce costs by up to 90% but may be interrupted.

### Launching Spot Instances

1. **Via AWS Console**
   - EC2 Dashboard → Spot Requests → Request Spot Instances
   - Choose same AMI and instance type
   - Set maximum price (optional)

2. **Via AWS CLI**
   ```bash
   aws ec2 request-spot-instances \
     --spot-price "1.50" \
     --instance-count 1 \
     --type "one-time" \
     --launch-specification file://spot-specification.json
   ```

### Handling Interruptions

The training script automatically saves checkpoints, allowing resumption:

```bash
# Training will auto-resume from last checkpoint
python main.py train experiments/configs/default.yaml --resume
```

### Spot Instance Best Practices

1. **Enable Checkpoint Frequency**
   - Set `checkpoint_every: 500` in config (save every 500 steps)

2. **Use Persistent Storage**
   - Mount EBS volume for checkpoints
   - Or sync to S3 regularly (see below)

3. **Monitor Spot Prices**
   ```bash
   aws ec2 describe-spot-price-history \
     --instance-types p3.2xlarge \
     --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
     --product-descriptions "Linux/UNIX" \
     --query 'SpotPriceHistory[*].[Timestamp,SpotPrice]' \
     --output table
   ```

## S3 Integration

### Setup S3 Bucket

```bash
# Create bucket
aws s3 mb s3://your-lfn-bucket

# Set lifecycle policy (optional - auto-delete old checkpoints)
aws s3api put-bucket-lifecycle-configuration \
  --bucket your-lfn-bucket \
  --lifecycle-configuration file://lifecycle.json
```

### Sync Checkpoints to S3

```bash
# Manual sync
aws s3 sync checkpoints/ s3://your-lfn-bucket/checkpoints/

# Automated sync (add to training script or cron)
watch -n 300 'aws s3 sync checkpoints/ s3://your-lfn-bucket/checkpoints/'
```

### Download Checkpoints

```bash
# Download specific checkpoint
aws s3 cp s3://your-lfn-bucket/checkpoints/best_model.pt checkpoints/

# Sync entire directory
aws s3 sync s3://your-lfn-bucket/checkpoints/ checkpoints/
```

### S3 Configuration in Training

Modify `training/trainer.py` to auto-sync:

```python
import boto3

def save_checkpoint_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Checkpoint uploaded to s3://{bucket}/{s3_key}")
```

## Cost Optimization

### 1. Use Spot Instances

- **Savings**: Up to 90% compared to on-demand
- **Risk**: May be interrupted
- **Mitigation**: Frequent checkpointing

### 2. Right-Size Instances

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# If GPU utilization < 80%, consider smaller instance
# If GPU memory < 50% used, consider cheaper GPU type
```

### 3. Stop Instances When Not Training

```bash
# Stop instance (preserves EBS volume)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance later
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

### 4. Use Reserved Instances

For long-term projects (1-3 years):
- **Savings**: Up to 75% compared to on-demand
- **Commitment**: Pay upfront or monthly

### 5. Optimize Storage

```bash
# Use gp3 instead of gp2 (cheaper, better performance)
# Delete old checkpoints
find checkpoints/ -name "checkpoint_step_*.pt" -mtime +7 -delete

# Compress logs
gzip experiments/logs/*.csv
```

### 6. Schedule Training

Use AWS Lambda or cron to start/stop instances:

```bash
# Start training at 2 AM (off-peak hours)
0 2 * * * /home/ubuntu/start-training.sh

# Stop instance after training
python main.py train config.yaml && sudo shutdown -h now
```

## Cost Estimation

### Example: Training on p3.2xlarge

| Component | Cost |
|-----------|------|
| Instance (p3.2xlarge) | $3.06/hr |
| Storage (100 GB gp3) | $0.08/hr |
| Data transfer (out) | $0.09/GB |
| **Total** | ~$3.14/hr |

**For 24-hour training run**: ~$75

**With Spot Instance (70% discount)**: ~$22.50

### Monitoring Costs

```bash
# Install AWS Cost Explorer CLI
pip install awscli

# View current month costs
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
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

### Out of Memory

```bash
# Check GPU memory usage
nvidia-smi

# Solutions:
# 1. Reduce batch size in config
# 2. Enable gradient checkpointing
# 3. Use larger instance type
```

### Slow Data Loading

```bash
# Check EBS volume performance
iostat -x 1

# Upgrade to gp3 with higher IOPS
aws ec2 modify-volume --volume-id vol-xxx --volume-type gp3 --iops 3000
```

### Connection Timeout

```bash
# Check security group allows SSH
aws ec2 describe-security-groups --group-ids sg-xxx

# Add SSH rule if missing
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
```

### Spot Instance Interrupted

```bash
# Check interruption notice (2-minute warning)
curl http://169.254.169.254/latest/meta-data/spot/instance-action

# Auto-save on interruption (add to training script)
import signal
def handle_interruption(signum, frame):
    save_checkpoint("emergency_checkpoint.pt")
    sys.exit(0)
signal.signal(signal.SIGTERM, handle_interruption)
```

## Advanced Configuration

### Multi-GPU Training

```bash
# Use all GPUs on p3.8xlarge (4x V100)
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  main.py train experiments/configs/default.yaml
```

### Auto-Scaling

Use AWS Auto Scaling Groups for multiple training runs:

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name lfn-training \
  --version-description "LFN training template" \
  --launch-template-data file://template.json

# Create auto-scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name lfn-asg \
  --launch-template LaunchTemplateName=lfn-training \
  --min-size 0 \
  --max-size 10 \
  --desired-capacity 1
```

### CloudWatch Monitoring

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure monitoring
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file://cloudwatch-config.json
```

## Additional Resources

- [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/)
- [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)
- [AWS Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)
