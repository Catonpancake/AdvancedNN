import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import timm
import csv

import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt


def reshape(imgdata: pd.DataFrame):
    img_reshaped = np.reshape(imgdata.values,(28,28))
    return img_reshaped


def drawimg(img, label=None):
    plt.title(f"Label: {label or img[1]}")
    plt.imshow(img[0], cmap='gray')


def tensor_to_pltimg(tensor_image):
    return tensor_image.permute(1,2,0).numpy()
    

def multidrawimg(imgs, labels, batch_size: int=16):
    fig = plt.figure(figsize=[10,10])
    plt.subplots_adjust(top=0.9, wspace=0.2, hspace=0.35)
    for i,data in enumerate(imgs):
        ax = plt.subplot(4,batch_size//4,i+1)
        ax.set_title(f'Label: {labels[i]}')
        ax.imshow(tensor_to_pltimg(data), cmap='gray')
        
        
