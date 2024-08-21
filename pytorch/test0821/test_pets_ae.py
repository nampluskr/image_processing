import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, EarlyStopping, set_seed, accuracy
from trainer import TrainerAE, binary_accuracy

import os
from glob import glob
import numpy as np
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

# train_df

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
          "num_workers": 4, "pin_memory": True } if use_cuda else {}

batch_size = 8
train_dataset = ImageDataset(train_df, 
                             transform_train, target_transform)
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, shuffle=True, **kwargs)

test_dataset = ImageDataset(valid_df,
                            transform_test, target_transform)
test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, shuffle=False, **kwargs)

from autoencoder import AutoEncoder, ConvEncoder, ConvDecoder

latent_dim = 128
encoder = ConvEncoder(latent_dim)
decoder = ConvDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

from autoencoder import AutoEncoder, ResEncoder, ResDecoder

latent_dim = 128
encoder = ResEncoder(latent_dim)
decoder = ResDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
