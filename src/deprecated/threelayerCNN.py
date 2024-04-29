import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import datasets, transforms 
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from PIL import Image


asl_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']



class CNN(LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        # RGB image with 3 channels, output 8 channels with 3x3 kernel 
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3, padding=1)
        
        # set up first fully connected layers, outchanel * image size 
        #self.fc1 = nn.Linear(in_features=16*54*54, out_features=120)
        self.fc1 = nn.Linear(in_features=32*28*28, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=20)
        self.fc4 = nn.Linear(in_features=20, out_features= len(asl_classes))
        
    def forward(self, x):
        # apply first convolutin layer by ReLu
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # = F.dropout(x)
        
        # second convltn. layer
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # = F.dropout(x)
        
        # third layer 
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        
        # flatten
        #x = x.view(-1, 16*54*54)
        x = x.view(-1, 32*28*28)
        #32*28*28
        
        # activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # applies the log softmax function
        output = F.log_softmax(x, dim=1)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)