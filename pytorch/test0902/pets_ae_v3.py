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
df

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

df

train_df, valid_df = train_test_split(df, test_size=0.3, 
                    stratify=df["SPECIES"], random_state=42)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

len(train_df), len(valid_df)

class ImageDataset(Dataset):
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True),)

    def forward(self, x):
        return self.block(x)

class ConvBlock2X(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),)
    
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.conv2 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.conv3 = nn.Sequential(
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.conv4 = nn.Sequential(
            ConvBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)
        self.linear = nn.Linear(512 * 16 * 16, latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 512 * 16 * 16)
        x = self.linear(x)
        return x

x = torch.randn(10, 3, 256, 256)
x = Encoder(latent_dim=1024)(x)
x.shape

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True),)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 512 * 16 * 16)
        self.upconv1 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(256, 128)
        self.upconv3 = UpConvBlock(128, 64)
        self.upconv4 = UpConvBlock(64, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 16, 16)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x

x = torch.randn(10, 1024)
x = Decoder(latent_dim=1024)(x)
x.shape

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


latent_dim = 1024
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.5 * (1 - ssim(pred, target)) + 0.5 * bce(pred, target)

# loss_fn = nn.BCELoss()
# loss_fn = nn.HuberLoss()
loss_fn = recon_loss
metrics = {"acc": binary_accuracy, "ssim": ssim}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=100, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

# >> Epoch[  1/100] loss=0.663, acc=0.535, ssim=0.371 | val_loss=0.656, val_acc=0.584, val_ssim=0.382 
# >> Epoch[  2/100] loss=0.656, acc=0.592, ssim=0.381 | val_loss=0.655, val_acc=0.585, val_ssim=0.383 
# >> Epoch[  3/100] loss=0.638, acc=0.682, ssim=0.391 | val_loss=0.615, val_acc=0.770, val_ssim=0.409 
# >> Epoch[  4/100] loss=0.611, acc=0.803, ssim=0.413 | val_loss=0.602, val_acc=0.808, val_ssim=0.423 
# >> Epoch[  5/100] loss=0.600, acc=0.826, ssim=0.428 | val_loss=0.586, val_acc=0.841, val_ssim=0.441 
# >> Epoch[  6/100] loss=0.591, acc=0.838, ssim=0.440 | val_loss=0.579, val_acc=0.859, val_ssim=0.451 
# >> Epoch[  7/100] loss=0.585, acc=0.844, ssim=0.449 | val_loss=0.576, val_acc=0.865, val_ssim=0.456 
# >> Epoch[  8/100] loss=0.580, acc=0.848, ssim=0.456 | val_loss=0.578, val_acc=0.845, val_ssim=0.458 
# >> Epoch[  9/100] loss=0.576, acc=0.851, ssim=0.461 | val_loss=0.568, val_acc=0.858, val_ssim=0.467 
# >> Epoch[ 10/100] loss=0.572, acc=0.855, ssim=0.467 | val_loss=0.568, val_acc=0.872, val_ssim=0.468 
# >> Epoch[ 11/100] loss=0.569, acc=0.856, ssim=0.472 | val_loss=0.563, val_acc=0.870, val_ssim=0.474 
# >> Epoch[ 12/100] loss=0.565, acc=0.858, ssim=0.477 | val_loss=0.563, val_acc=0.866, val_ssim=0.476 
# >> Epoch[ 13/100] loss=0.562, acc=0.859, ssim=0.482 | val_loss=0.560, val_acc=0.863, val_ssim=0.480 
# >> Epoch[ 14/100] loss=0.558, acc=0.862, ssim=0.487 | val_loss=0.558, val_acc=0.875, val_ssim=0.481 
# >> Epoch[ 15/100] loss=0.554, acc=0.864, ssim=0.492 | val_loss=0.558, val_acc=0.867, val_ssim=0.483 
# >> Epoch[ 16/100] loss=0.551, acc=0.867, ssim=0.497 | val_loss=0.555, val_acc=0.877, val_ssim=0.487 
# >> Epoch[ 17/100] loss=0.547, acc=0.870, ssim=0.502 | val_loss=0.549, val_acc=0.870, val_ssim=0.494 
# >> Epoch[ 18/100] loss=0.544, acc=0.873, ssim=0.506 | val_loss=0.548, val_acc=0.874, val_ssim=0.494 
# >> Epoch[ 19/100] loss=0.541, acc=0.876, ssim=0.511 | val_loss=0.541, val_acc=0.891, val_ssim=0.500 
# >> Epoch[ 20/100] loss=0.537, acc=0.879, ssim=0.516 | val_loss=0.543, val_acc=0.876, val_ssim=0.500 
# >> Epoch[ 21/100] loss=0.534, acc=0.882, ssim=0.521 | val_loss=0.538, val_acc=0.893, val_ssim=0.503 
# >> Epoch[ 22/100] loss=0.530, acc=0.884, ssim=0.525 | val_loss=0.541, val_acc=0.890, val_ssim=0.501 
# >> Epoch[ 23/100] loss=0.527, acc=0.885, ssim=0.531 | val_loss=0.536, val_acc=0.897, val_ssim=0.505 
# >> Epoch[ 24/100] loss=0.524, acc=0.887, ssim=0.535 | val_loss=0.530, val_acc=0.898, val_ssim=0.513 
# >> Epoch[ 25/100] loss=0.521, acc=0.888, ssim=0.540 | val_loss=0.534, val_acc=0.884, val_ssim=0.509 
# ...
# >> Epoch[ 35/100] loss=0.490, acc=0.898, ssim=0.588 | val_loss=0.526, val_acc=0.899, val_ssim=0.515 
# >> Epoch[ 36/100] loss=0.487, acc=0.898, ssim=0.594 | val_loss=0.528, val_acc=0.886, val_ssim=0.514 
# >> Early stopped!
# >> Evaluation: loss=0.519, acc=0.900  

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
