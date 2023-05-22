import os
import numpy as np
import pandas as pd


import torch
from torch import nn
import torch.optim as optim

from utils import reshape, drawimg, tensor_to_pltimg, multidrawimg
from utils_traintest import train, test


class Trainer:
    def __init__(self,
                 dataloader,
                 model: nn.Module,
                 batchsize: int,
                 optimizer: optim.Optimizer,
                 loss_fn: nn.Module,
                 epochs: int = 10,
                 device: str = 'gpu'):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.batchsize = batchsize
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        
    def train(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(self.dataloader, self.model, self.loss_fn, self.optimizer, device=self.device)
        print("Done!")
        
    def savemodel(self):
        torch.save(self.model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")
        
        
        
        
    