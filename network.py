import torch
import geffnet
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.metrics.functional import to_categorical, confusion_matrix
from pytorch_lightning.metrics.functional.classification import accuracy,  precision, recall
from pytorch_lightning.metrics.functional import f1
from torchvision import models, datasets
from Unet import UNet

class CustomNetwork(pl.LightningModule):

    def __init__(self, batch_size: int = 8, learning_rate: float = 0.01, num_classes: int = 10, training: bool = True, num_workers: int = 8, **kwargs):

        super().__init__()
        self.save_hyperparameters()
        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size


        # Define PyTorch model


        if training:
            # self.wandb = kwargs["wandb"]
            self.train_dataset = kwargs["train_dataset"]
            self.val_dataset = kwargs["val_dataset"]
            self.test_dataset = kwargs["test_dataset"]
            self.hparams = kwargs["hparams"]

        self.model = UNet(n_channels=1, n_classes=self.num_classes)


    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        #cm = confusion_matrix(preds, y)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                                           num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                                           num_workers=self.num_workers, pin_memory=True, drop_last=True)