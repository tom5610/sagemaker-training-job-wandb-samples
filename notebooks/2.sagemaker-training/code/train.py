import wandb

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer


import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy

import os
import ast
import argparse

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

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        print("!!!LightningModule-Checkpoint!!!")
        print("checkpoint name", self.logger._checkpoint_name)
        print("project", self.logger._project)
        print("name", self.logger._name)
        print("entity", self.logger._experiment.entity)

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
    checkpoint_callback = ModelCheckpoint(
        dirpath="/opt/ml/checkpoint", 
        filename="{epoch:03d}",
        monitor='val_accuracy', 
        mode='max')

    wandb_project_name = os.environ.get("WANDB_PROJECT_NAME")
    wandb_logger = WandbLogger(project=wandb_project_name, log_model="all")

    wandb_checkpoint_name = os.environ.get("WANDB_CHECKPOINT_NAME", None) # if not None, we resume the training from checkpoint.
    wandb_checkpoint_tag = os.environ.get("WANDB_CHECKPOINT_TAG", "latest")

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        max_epochs=args.epochs
    )    

    model = MNIST_LitModule(n_layer_1=128, n_layer_2=128)

    training_loader, validation_loader = load_data()

    trainer.fit(model, training_loader, validation_loader)

    wandb.finish()

def loging_wandb(wandb_secret_name: str):
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
            
        print(f"Successfully loaded secret {wandb_secret_name}")
    except ClientError as e:
        print(f"Error loading secret {wandb_secret_name}: {str(e)}")
    
    wandb.login()

def get_env_var_value(key:str) -> str:
    if key in os.environ.keys():
        return os.environ[key]
    else:
        return None

def get_wandb_parameters():

    parser.add_argument("--wandb-secret-name", type=str, default=os.environ["WANDB_SECRET_NAME"])
    parser.add_argument("--wandb-project-name", type=str, default=os.environ["WANDB_PROJECT_NAME"])
    parser.add_argument("--wandb-checkpoint-name", type=str, default=os.environ["WANDB_CHECKPOINT_NAME"])
    parser.add_argument("--wandb-checkpoint-tag", type=str, default=os.environ["WANDB_CHECKPOINT_TAG"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)

    secret_name = os.environ.get("WANDB_SECRET_NAME")
    loging_wandb(secret_name)

    args = parser.parse_args()
    training(args)
    