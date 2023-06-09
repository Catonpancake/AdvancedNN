import os
import csv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils as ut


def train(dataloader: DataLoader, model, loss_fn, optimizer, device, epoch, seed):
    model.to(device)
    size = len(dataloader.dataset)
    df = pd.DataFrame()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    current = {'loss': loss.cpu().detach().numpy()/len(dataloader), 'accuracy': (correct/size)*100, 'lable': 'train', 'epoch': epoch, 'seed': seed}
    df = df.append(current, ignore_index = True)
    return df
            
def test(dataloader, model, loss_fn, device, epoch, seed, valid=True):
    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    df = pd.DataFrame()
    with torch.no_grad():
        if valid:
            for X, y in dataloader:
                X, y = X.to(device).float(), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            current = {'loss': test_loss, 'accuracy': correct*100, 'lable': 'valid', 'epoch': epoch, 'seed': seed}
            df = df.append(current, ignore_index = True)
        elif valid == False:
            counter = 0
            filename = f"Submissions/Submission_{counter}.csv"
            while os.path.exists(filename):
                counter += 1
                filename = f"Submissions/Submission_{counter}.csv"
            with open(filename, 'a+', newline='') as f:  # 'a+' means append to a file
                thewriter = csv.writer(f)
                for X, y in dataloader:
                    X = X.to(device).float()
                    pred = model(X)
                    result = pred.argmax().item()  # this your label
                    print(result)
                    thewriter.writerow([result])
    
    return df
