import wandb

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer


import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy

import os
import sys
import glob 
import logging
import argparse

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.addHandler(logging.StreamHandler(sys.stdout))

sagemaker_client = boto3.client('sagemaker')

def get_tags_of_training_job(training_job_arn:str):
    response = sagemaker_client.list_tags(ResourceArn=training_job_arn)
    tags = {}
    for tag in response['Tags']:
        tags[tag['Key']] = tag['Value']
    return tags

def put_tags_of_training_job(training_job_arn:str, entity: str, project: str, checkpoint_name: str):
    logger.info("tag_training_job() is invoked")
    response = sagemaker_client.add_tags(
        ResourceArn=training_job_arn,
        Tags=[
            {'Key': "WANDB_ENTITY", "Value": entity},
            {'Key': "WANDB_PROJECT", "Value": project},
            {'Key': "WANDB_CHECKPOINT_NAME", "Value": checkpoint_name}
        ]
    )

class SageMakerTrainingJobTaggingCallback(Callback):

    def __init__(self, training_job_arn:str):
        self.tagging_done = False
        self.training_job_arn = training_job_arn

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logger.info("on_save_checkpoint() is invoked.")
        checkpoint_name = trainer.logger._checkpoint_name
        # for saving a checkpoint for epoch '0', the checkpoint name is not generated yet.
        to_tag_training_job = (checkpoint_name is not None) and (not self.tagging_done)
        logger.info(f"{to_tag_training_job=}")
        if to_tag_training_job: 
            self.tagging_done = True
            entity = trainer.logger._experiment.entity
            project = trainer.logger._project

            # put tags on training job
            put_tags_of_training_job(self.training_job_arn, entity, project, checkpoint_name)
        else:
            # checkpoint name is not generated at wandb server side yet.
            pass

class MNIST_LitModule(pl.LightningModule):

    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        '''
        method used to define our model parameters
        '''
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        # loss
        self.loss = CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        # save Hyperparameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    
    def forward(self, x):
        '''method used for infernce input -> output'''

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # let's do 3 x (linear + relu)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        return preds, loss, acc


    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # log loss and metric
        self.log('train_loss', loss)
        self.log('training_accuracy', acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # log
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNIST(root="./data/MNIST", download=True, transform=transform)
    training_set, validation_set = random_split(dataset, [55_000, 5000])            
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=64)

    return training_loader, validation_loader

def training(args):


    # Verify the checkpoint tag
    # when checkpoint_name tag exists on training, we simplify set tag to the latest when training job is restarted
    # otherwise, we read the data from environment variables
    training_job_arn = os.environ.get('TRAINING_JOB_ARN')
    training_job_tags = get_tags_of_training_job(training_job_arn)

    if "WANDB_CHECKPOINT_NAME" in training_job_tags:
        wandb_project = training_job_tags['WANDB_PROJECT']
        wandb_checkpoint_name = training_job_tags['WANDB_CHECKPOINT_NAME']
        wandb_checkpoint_tag = "latest"
    else:
        wandb_project = os.environ.get("WANDB_PROJECT")
        # if not None, we resume the training from checkpoint.
        wandb_checkpoint_name = os.environ.get("WANDB_CHECKPOINT_NAME", None) 
        # for the training job starts, it may load specified checkpoint version. by default is 'latest'
        wandb_checkpoint_tag = os.environ.get("WANDB_CHECKPOINT_TAG", "latest") 

    wandb_logger = WandbLogger(project=wandb_project, log_model="all")

    if wandb_checkpoint_name is not None:
        logger.info("----Download & Load checkpoint----")
        wandb_entity = wandb_logger.experiment.entity
        checkpoint_reference = f"{wandb_entity}/{wandb_project}/{wandb_checkpoint_name}:{wandb_checkpoint_tag}"
        download_artefact_path = wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
        # load checkpoint
        model_artifacts = glob.glob(f"{download_artefact_path}/*.ckpt")
        model = MNIST_LitModule.load_from_checkpoint(model_artifacts[0]) 
    else:
        model = MNIST_LitModule(n_layer_1=128, n_layer_2=128)

    # callback for checkpoint and tagging
    checkpoint_callback = ModelCheckpoint(
        dirpath="/opt/ml/checkpoint", 
        filename="{epoch:03d}",
        monitor='val_accuracy', 
        mode='max')
    tagging_callback = SageMakerTrainingJobTaggingCallback(training_job_arn)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, tagging_callback],
        accelerator="gpu",
        max_epochs=args.epochs
    )    

    training_loader, validation_loader = load_data()
    trainer.fit(model, training_loader, validation_loader)

    wandb.finish()

def login_wandb(wandb_secret_name: str):
    import boto3
    import json
    from botocore.exceptions import ClientError
    
    # Initialize the Secrets Manager client
    secretsmanager = boto3.client('secretsmanager')
    
    try:
        # Get the secret value
        response = secretsmanager.get_secret_value(SecretId=wandb_secret_name)
        
        # Parse the secret JSON
        secret = json.loads(response['SecretString'])
        
        # Set environment variables from the secret
        for key, value in secret.items():
            os.environ[key] = value
            
        logger.info(f"Successfully loaded secret {wandb_secret_name}")
    except ClientError as e:
        logger.info(f"Error loading secret {wandb_secret_name}: {str(e)}")
    
    wandb.login()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)

    secret_name = os.environ.get("WANDB_SECRET_NAME")
    login_wandb(secret_name)

    args = parser.parse_args()
    training(args)
    