# SageMaker Training Job Integration with Weights & Biases

This repository demonstrates how to integrate Weights & Biases (W&B) with Amazon SageMaker Training Jobs for efficient ML model training, experiment tracking, and checkpoint management.

## Use Case

Deep learning model training often requires tracking experiments, metrics, and managing model checkpoints for resumption in case of failures. This integration addresses:

1. **ML Experimentation & Tracking**: Track model metrics, hyperparameters, and training progress using W&B.
2. **Checkpoint Management**: Store and retrieve model checkpoints through W&B.
3. **Training Resilience**: Resume training from checkpoints after SageMaker cluster repairs due to GPU errors.
4. **Training Job Tagging**: Automatically tag SageMaker Training Jobs with W&B entity, project, and checkpoint information.

## Architecture

This is a reference design for the [second notebook](./notebooks/2.sagemaker-training/2.training-at-SageMaker-Training-Job.ipynb).

![SageMaker Training Job Integration with WANDB](./images/SageMaker%20Training%20Job%20Integration%20with%20WANDB.png)

The architecture workflow:

1. Client initiates SageMaker Training Job
2. Job securely accesses W&B secret for authentication
3. IAM role manages permissions for job execution and resource tagging
4. Training job is configured with appropriate tags
5. [Act after hardware repair] Training job can resume from W&B checkpoints
6. Training output is stored in S3

## Notebooks

### 1. ML Training at Local

The [first notebook](./notebooks/1.ml-training-at-local.ipynb) demonstrates:
- PyTorch Lightning model training with W&B integration
- MNIST image classification implementation
- Checkpoint management with W&B
- How to resume training from checkpoints

### 2. SageMaker Training Job Integration

The [second notebook](./notebooks/2.sagemaker-training/2.training-at-SageMaker-Training-Job.ipynb) demonstrates:
- SageMaker execution role creation with proper permissions
- W&B secret creation in AWS Secrets Manager
- SageMaker PyTorch Training Job configuration
- W&B integration for metrics tracking and checkpoint management
- Training job tagging for checkpoint reference
- Automatic training resumption after failures

## Key Features

1. **SageMaker Training Job Tagging**:
   - The `SageMakerTrainingJobTaggingCallback` in `train.py` automatically tags jobs with:
     - `WANDB_ENTITY`
     - `WANDB_PROJECT`
     - `WANDB_CHECKPOINT_NAME`

2. **Checkpoint Management**:
   - Automatic checkpoint storage in W&B
   - Training resumption from checkpoints
   - Customizable checkpoint tag selection

3. **Environment Configuration**:
   - Secure W&B API key storage using AWS Secrets Manager
   - PyTorch Lightning and W&B runtime installation through requirements.txt

## Getting Started

1. Set up your W&B API key in a `.env` file:
   ```
   WANDB_API_KEY=your-api-key
   ```

2. Run the notebooks in order:
   - Start with local training to understand the basics
   - Move to SageMaker training for production-level deployment

3. For SageMaker training, ensure you have:
   - AWS credentials configured
   - Permissions to create IAM roles, S3 buckets, and Secrets Manager secrets
   - SageMaker execution role with proper permissions

## References

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [SageMaker Training Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
- [Cluster repairs for GPU errors](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints-cluster-repair.html)