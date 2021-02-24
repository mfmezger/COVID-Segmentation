
#import geffnet # no use
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import models, datasets

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import to_categorical, confusion_matrix
from pytorch_lightning.metrics.functional.classification import accuracy,  precision, recall
from pytorch_lightning.metrics.functional import f1

#UNet
from Unet import UNet

class CustomNetwork(pl.LightningModule):

    def __init__(self, batch_size: int = 8, learning_rate: float = 0.01, num_classes: int = 1, training: bool = True, num_workers: int = 8, **kwargs):
        
        # 7)
        super().__init__()
        
        # 8)
        self.save_hyperparameters()
        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.num_classes = num_classes 
        self.num_workers = num_workers
        self.batch_size = batch_size


        # Define PyTorch model

        # 9)
        if training:
            # self.wandb = kwargs["wandb"]
            self.train_dataset = kwargs["train_dataset"] # komplettes TorchDataset Objekt wird in self.train_dataset gespeichert
            self.val_dataset = kwargs["val_dataset"]
            self.test_dataset = kwargs["test_dataset"]
            self.hparams = kwargs["hparams"]
        
        # 10)  Model definieren: (n_channels: Anzahl Schichten Eingang; N_classes: Anzahl Schichten Ausgang)
        # from Unet import UNet
        self.model = UNet(n_channels=1, n_classes=self.num_classes)


    def forward(self, x):
        # Modell aufrufen
        x = self.model(x)
        return x
    

# ----- Training ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Feed Forward
        logits = self(x)
        
        # Loss
        loss = F.cross_entropy(logits, y)
        
        # Tensorbard
        tensorboard_logs = {"train_loss": loss}
        
        return {'loss': loss, "log": tensorboard_logs} # als dict zurück
    
    
    
# ----- Validation ------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Feed Forward 
        logits = self(x)
        
        # Loss
        loss = F.cross_entropy(logits, y)
        #preds = torch.argmax(logits, dim=1)
        #cm = confusion_matrix(preds, y)
        
        return {'val_loss': loss} # als dict zurÃ¼ck
    
    def validation_epoch_end(self, outputs): # NEW
    # fÃ¼r Output von validation
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() # mean over losses
        
        # Tensorbard
        tensorboard_logs = {"val_loss": avg_loss}
        
        return {'avg_val_loss': avg_loss, "log": tensorboard_logs}
    
    
# ----- Test ------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch # x: [2, 1, 512, 512] bs,?,bildgröße y: #[2, 512, 512] bs,bildgröße
        
        # Feed Forward 
        logits = self(x) #[2,2,512,512] bs, Anzahl_Bilder, Bild_größe
        #print(logits.shape)
        path = "/home/wolfda/COVID-19-20_v2/Output/" +str(batch_idx)+ ".pt"
        torch.save({"Input": x, "Target": y, "Output": logits}, path)
        
        # Loss
        loss = F.cross_entropy(logits, y)
        
        return {'test_loss': loss} # als dict zurÃ¼ck
    
    def test_epoch_end(self, outputs): # NEW
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean() # mean over losses
        
        # Tensorbard
        tensorboard_logs = {"test_loss": avg_loss}
        
        return {'avg_test_loss': avg_loss, "log": tensorboard_logs}
       

    
# ----- Optimizer ------------------------------------------------------------
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