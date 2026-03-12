# Deployment Guide

This directory contains comprehensive deployment documentation for the Latent Forecasting Network research project.

## Overview

The Latent Forecasting Network can be deployed in multiple environments:
- **Local Development**: Using setup scripts for Linux/macOS/Windows
- **Docker Containers**: For reproducible environments
- **Cloud Platforms**: AWS, Google Cloud, and Azure

## Quick Start

### Local Setup

**Linux/macOS:**
```bash
./setup.sh
source venv/bin/activate
python main.py train experiments/configs/default.yaml
```

**Windows:**
```powershell
.\setup.ps1
.\venv\Scripts\Activate.ps1
python main.py train experiments\configs\default.yaml
```

### Docker Setup

```bash
# Build and run
docker-compose up -d lfn-training
docker-compose exec lfn-training bash

# Inside container
python main.py train experiments/configs/default.yaml
```

## Documentation Files

### Setup Scripts
- **setup.sh** - Automated environment setup for Linux/macOS
- **setup.ps1** - Automated environment setup for Windows PowerShell

### Docker
- **DOCKER.md** - Complete Docker deployment guide
  - Building images
  - Running containers
  - GPU support
  - Volume mounts
  - TensorBoard integration
  - Troubleshooting

### Cloud Platforms

#### AWS (Amazon Web Services)
- **AWS_SETUP.md** - EC2 GPU instance deployment
  - Instance types (p3, p4, g4dn, g5)
  - Spot instances (up to 90% cost savings)
  - S3 integration for checkpoints
  - Cost optimization strategies
  - Multi-GPU training

#### Google Cloud Platform
- **GCP_SETUP.md** - Compute Engine GPU deployment
  - Instance types (V100, A100, T4, P100)
  - Preemptible instances (up to 80% cost savings)
  - Cloud Storage integration
  - Cost optimization strategies
  - Regional availability

#### Microsoft Azure
- **AZURE_SETUP.md** - Azure VM GPU deployment
  - VM sizes (NC, ND, NV series)
  - Low-priority VMs (up to 80% cost savings)
  - Blob Storage integration
  - Cost optimization strategies
  - Managed identities

## Deployment Comparison

| Platform | Best GPU Option | Cost (Regular) | Cost (Spot/Preemptible) | Best For |
|----------|----------------|----------------|-------------------------|----------|
| **AWS** | p3.2xlarge (V100) | $3.06/hr | $0.92/hr (70% off) | Mature ecosystem, S3 integration |
| **GCP** | n1-highmem-8 + V100 | $2.48/hr | $0.74/hr (70% off) | Cost-effective, good tooling |
| **Azure** | Standard_NC6s_v3 (V100) | $3.06/hr | $0.61/hr (80% off) | Enterprise integration |
| **Local** | Your GPU | $0 | $0 | Development, small experiments |
| **Docker** | Your GPU | $0 | $0 | Reproducibility, portability |

## Cost Optimization Tips

### 1. Use Spot/Preemptible Instances
- **AWS Spot**: Up to 90% savings
- **GCP Preemptible**: Up to 80% savings
- **Azure Low-Priority**: Up to 80% savings
- **Requirement**: Frequent checkpointing (every 500 steps)

### 2. Right-Size Your Instance
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# If GPU utilization < 80%, consider smaller instance
# If GPU memory < 50% used, consider cheaper GPU type
```

### 3. Stop Instances When Not Training
```bash
# AWS
aws ec2 stop-instances --instance-ids i-xxx

# GCP
gcloud compute instances stop lfn-training

# Azure
az vm deallocate --resource-group lfn-resources --name lfn-training
```

### 4. Use Cloud Storage for Checkpoints
- Sync checkpoints to cloud storage regularly
- Allows using spot/preemptible instances safely
- Enables sharing checkpoints across team

### 5. Schedule Training for Off-Peak Hours
- Some cloud providers offer lower prices during off-peak
- Use cron jobs or cloud schedulers to automate

## Typical Training Costs

### Small Model (50M parameters, WikiText-2)
- **Training Time**: ~2 hours on V100
- **AWS p3.2xlarge**: ~$6 (regular) or ~$2 (spot)
- **GCP n1-highmem-8 + V100**: ~$5 (regular) or ~$1.50 (preemptible)
- **Azure NC6s_v3**: ~$6 (regular) or ~$1.20 (spot)

### Medium Model (200M parameters, WikiText-103)
- **Training Time**: ~12 hours on V100
- **AWS p3.2xlarge**: ~$37 (regular) or ~$11 (spot)
- **GCP n1-highmem-8 + V100**: ~$30 (regular) or ~$9 (preemptible)
- **Azure NC6s_v3**: ~$37 (regular) or ~$7 (spot)

### Large Model (500M parameters, OpenWebText)
- **Training Time**: ~48 hours on 4x V100
- **AWS p3.8xlarge**: ~$587 (regular) or ~$176 (spot)
- **GCP n1-highmem-16 + 2x V100**: ~$238 (regular) or ~$71 (preemptible)
- **Azure NC24s_v3**: ~$587 (regular) or ~$117 (spot)

## Troubleshooting

### GPU Not Detected
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory
1. Reduce `batch_size` in config file
2. Enable gradient checkpointing
3. Use mixed precision training (enabled by default)
4. Use larger GPU instance

### Slow Training
1. Check GPU utilization: `nvidia-smi`
2. Increase `num_workers` in data config
3. Use faster storage (SSD instead of HDD)
4. Enable mixed precision training

### Connection Issues
1. Check security group/firewall rules
2. Verify SSH key permissions: `chmod 400 key.pem`
3. Check instance is running
4. Verify public IP address

## Best Practices

### For Research Projects
1. **Use version control** for configs and code
2. **Document experiments** in experiment logs
3. **Save checkpoints frequently** (every 500-1000 steps)
4. **Use cloud storage** for checkpoint backup
5. **Monitor costs** regularly
6. **Use spot/preemptible** for non-critical runs

### For Production
1. **Use reserved instances** for long-term projects
2. **Implement monitoring** (CloudWatch, Stackdriver, Azure Monitor)
3. **Set up alerts** for failures and cost overruns
4. **Use managed services** when possible
5. **Implement auto-scaling** for multiple experiments
6. **Regular backups** to cloud storage

## Security Considerations

### Cloud Deployments
1. **Use IAM roles** instead of access keys
2. **Enable encryption** for storage
3. **Restrict network access** (security groups, firewalls)
4. **Use private subnets** for sensitive data
5. **Enable logging** and monitoring
6. **Regular security updates**

### Docker Deployments
1. **Run as non-root user** when possible
2. **Use official base images**
3. **Scan images** for vulnerabilities
4. **Limit container resources**
5. **Use read-only filesystems** where possible

## Support and Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Docker Documentation](https://docs.docker.com/)

### Cloud Provider Docs
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [GCP Compute Engine](https://cloud.google.com/compute/docs)
- [Azure Virtual Machines](https://docs.microsoft.com/en-us/azure/virtual-machines/)

### Cost Calculators
- [AWS Pricing Calculator](https://calculator.aws/)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)

## Contributing

For deployment-related improvements or issues:
1. Test changes in development environment first
2. Document any new deployment methods
3. Update cost estimates if pricing changes
4. Add troubleshooting tips for common issues

## License

This project is licensed under the terms specified in the LICENSE file in the project root.
