import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import TrainerAE, binary_accuracy, EarlyStopping, set_seed
from pytorch_mssim import ssim

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

set_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ID: 1:37 Class ids            -> 0:36
# SPECIES: 1:Cat 2:Dog          -> 0:1
# BREED ID: 1-25:Cat 1:12:Dog   -> 0:24

data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
df = pd.read_table(data_dir + "annotations/list.txt", sep=' ',
        skiprows=6, header=None, 
        names=["CLASS_ID", "SPECIES", "BREED", "ID"])
df[["SPECIES", "BREED", "ID"]] -= 1
df["path"] = data_dir + "images/" + df["CLASS_ID"] + ".jpg"
# df

# https://github.com/tensorflow/models/issues/3134
images_png = [
    "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
    "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
    "Abyssinian_5", "Abyssinian_34",
]
images_corrupt = ["chihuahua_121", "beagle_116"]

for idx, row in df.iterrows():
    # if row["CLASS_ID"] in images_png:
        # df.loc[idx, "path"] = df.loc[idx, "path"].replace("jpg", "png")
        # df.drop([idx], axis=0, inplace=True)
    if row["CLASS_ID"] in images_corrupt:
        df.drop([idx], axis=0, inplace=True)

# df

train_df, valid_df = train_test_split(df, test_size=0.3, 
                    stratify=df["SPECIES"], random_state=42)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

# len(train_df), len(valid_df)

class ImageDataset(nn.Module):
    def __init__(self, df, transform=None, target_transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        image = cv2.imread(path)
        if len(image.shape) == 3 and image.shape[2] == 4:   # png 파일
            iamge = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.loc[idx, "SPECIES"]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def target_transform(label):
    return torch.tensor(label).long()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
use_cuda = torch.cuda.is_available()
kwargs = {"worker_init_fn": seed_worker, "generator": g,
          "num_workers": 8, "pin_memory": True } if use_cuda else {}

batch_size = 8
train_dataset = ImageDataset(train_df, transform_train, target_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

test_dataset = ImageDataset(valid_df, transform_test, target_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

from autoencoder import AutoEncoder, ConvBlock, UpConvBlock

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

# x = torch.randn(256, 3, 256, 256)
# x = Encoder()(x)
# x.shape

class Decoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.upconv1 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(256, 128)
        self.upconv3 = UpConvBlock(128, 64)
        self.upconv4 = UpConvBlock(64, 3)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x

# x = Decoder()(x)
# x.shape

latent_dim = 128
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.1 * (1 - ssim(pred, target)) + 0.9 * bce(pred, target)

# loss_fn = nn.BCELoss()
# loss_fn = nn.HuberLoss()
loss_fn = recon_loss
metrics = {"acc": binary_accuracy, "ssim": ssim}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=10, valid_loader=test_loader)
            #   early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

# >> Epoch[ 1/10] loss=0.643, acc=0.870, ssim=0.380 | val_loss=0.625, val_acc=0.918, val_ssim=0.447   
# >> Epoch[ 2/10] loss=0.625, acc=0.906, ssim=0.494 | val_loss=0.617, val_acc=0.919, val_ssim=0.520   
# >> Epoch[ 3/10] loss=0.622, acc=0.902, ssim=0.524 | val_loss=0.614, val_acc=0.912, val_ssim=0.538   
# >> Epoch[ 4/10] loss=0.620, acc=0.898, ssim=0.538 | val_loss=0.614, val_acc=0.923, val_ssim=0.538   
# >> Epoch[ 5/10] loss=0.619, acc=0.895, ssim=0.548 | val_loss=0.613, val_acc=0.890, val_ssim=0.560   
# >> Epoch[ 6/10] loss=0.618, acc=0.891, ssim=0.556 | val_loss=0.612, val_acc=0.905, val_ssim=0.561   
# >> Epoch[ 7/10] loss=0.617, acc=0.889, ssim=0.562 | val_loss=0.611, val_acc=0.889, val_ssim=0.572   
# >> Epoch[ 8/10] loss=0.616, acc=0.888, ssim=0.567 | val_loss=0.610, val_acc=0.875, val_ssim=0.582   
# >> Epoch[ 9/10] loss=0.616, acc=0.885, ssim=0.572 | val_loss=0.610, val_acc=0.901, val_ssim=0.576   
# >> Epoch[10/10] loss=0.615, acc=0.883, ssim=0.576 | val_loss=0.609, val_acc=0.886, val_ssim=0.586   
# >> Evaluation: loss=0.609, acc=0.886     

def show_images(images, labels=None, n_images=5, unit=2):
    fig, axes = plt.subplots(ncols=n_images, figsize=(n_images*unit, unit))
    for i in range(n_images):
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img = std * img + mean
        # img = np.clip(img, 0, 1)

        img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_axis_off()
        if labels is not None:
            label = class_names[labels[i].cpu().numpy()]
            axes[i].set_title(label)
    fig.tight_layout()
    plt.show()

x, y = next(iter(test_loader))
images, labels = x.to(device), y.to(device)
model.eval()
pred = model(images)

show_images(images, n_images=8)
show_images(pred.detach(), n_images=8)
