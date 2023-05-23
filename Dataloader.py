import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

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
    

def load_split_train_vaild(datadir, transforms, valid_size = .2, batch_size=4, seed=42):
    setseed(seed)
    dataset = datasets.ImageFolder(datadir,       
                    transform=transforms)
    num_train = int(len(dataset)*(1-valid_size))
    num_valid = len(dataset) - num_train
    print(num_train, num_valid, num_train+num_valid, len(dataset))
    traindata, validdata = train_test_split(dataset, test_size=valid_size, random_state=123, shuffle=True)
    trainloader = DataLoader(traindata, batch_size=batch_size, num_workers=0, shuffle=True)
    validloader = DataLoader(validdata, batch_size=batch_size, num_workers=0, shuffle=False)

    return trainloader, validloader

def setseed(seednum = 42):
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    torch.cuda.manual_seed_all(seednum)