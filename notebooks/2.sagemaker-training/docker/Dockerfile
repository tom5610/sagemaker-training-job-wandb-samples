FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker

RUN apt-get update && apt-get install -y build-essential

RUN pip --no-cache-dir install lightning==2.5.3 wandb==0.21.1

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code