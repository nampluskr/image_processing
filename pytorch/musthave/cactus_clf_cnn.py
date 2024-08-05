import os
import random
import numpy as np
import pandas as pd
import skimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from trainer import Trainer, EarlyStopper


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = self.img_dir + img_name
        image = skimage.io.imread(img_path)
        label = self.df.iloc[idx, 1]

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def roc_auc(y_pred, y_true):
    y_pred = torch.softmax(y_pred, dim=1)[:, 1]
    return roc_auc_score(to_numpy(y_true), to_numpy(y_pred))

def accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)   ## int64 (long)
    return torch.eq(y_pred, y_true).float().mean()


if __name__ == "__main__":

    #################################################################
    ## Random Seed
    #################################################################
    seed = 50
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
             
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    #################################################################
    ## Dataset
    #################################################################
    data_dir = '/home/namu/myspace/data/aerial-cactus-identification/'
    labels = pd.read_csv(data_dir + "train.csv")
    
    train_df, valid_df = train_test_split(labels, test_size=0.2, 
                                    stratify=labels["has_cactus"], 
                                    random_state=0)

    transform = transforms.ToTensor()

    img_dir = data_dir + "train/"
    train_dataset = ImageDataset(train_df, img_dir=img_dir, transform=transform)
    valid_dataset = ImageDataset(valid_df, img_dir=img_dir, transform=transform)

    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, **kwargs)

    #################################################################
    ## Modeing
    #################################################################
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics={"acc": accuracy, "roc": roc_auc}

    #################################################################
    ## Training
    #################################################################
    clf = Trainer(model, optimizer, loss_fn, metrics)
    hist = clf.fit(train_loader, n_epochs=50, valid_loader=valid_loader, 
               early_stopper=EarlyStopper(patience=3, min_delta=0.01))

    # >> Epoch[ 1/50] loss=0.243, acc=0.898, roc=0.959 | val_loss=0.198, val_acc=0.917, val_roc=0.975     
    # >> Epoch[ 2/50] loss=0.141, acc=0.947, roc=0.986 | val_loss=0.137, val_acc=0.953, val_roc=0.987     
    # >> Epoch[ 3/50] loss=0.118, acc=0.957, roc=0.991 | val_loss=0.126, val_acc=0.955, val_roc=0.990     
    # >> Epoch[ 4/50] loss=0.105, acc=0.961, roc=0.993 | val_loss=0.135, val_acc=0.952, val_roc=0.993     
    # >> Epoch[ 5/50] loss=0.095, acc=0.964, roc=0.994 | val_loss=0.089, val_acc=0.965, val_roc=0.994     
    # >> Epoch[ 6/50] loss=0.085, acc=0.970, roc=0.995 | val_loss=0.088, val_acc=0.968, val_roc=0.995     
    # >> Epoch[ 7/50] loss=0.086, acc=0.969, roc=0.996 | val_loss=0.085, val_acc=0.969, val_roc=0.995     
    # >> Epoch[ 8/50] loss=0.076, acc=0.973, roc=0.996 | val_loss=0.079, val_acc=0.971, val_roc=0.996     
    # >> Early stopped!
    
