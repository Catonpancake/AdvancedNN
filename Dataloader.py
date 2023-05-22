import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F

class dataloader(Dataset):
    def __init__(self, img_dir, transform=None, test=False):
        self.data = datasets.ImageFolder(img_dir,       
                    transform=transform)
        self.istrain = test is False
        self.img_dir = img_dir
        self.img = self.data
        self.transform = transform
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx: int):
        image = self.img[idx]
        if self.transform:
            image = self.transform(image.astype(float))
        if self.istrain:
            label = self.data[idx][1]
            
            return image, label
        else:
            return image
        
        
        
       
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')