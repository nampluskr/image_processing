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

def get_label(path):
    class_names = ['dandelion', 'roses', 'daisy', 'sunflowers', 'tulips']
    dirname = os.path.dirname(path)
    classname = dirname.split(os.path.sep)[-1]
    return class_names.index(classname)

data_dir = "/home/namu/myspace/NAMU/datasets/flower_photos/"
df = pd.DataFrame({"path": glob(data_dir + "*/*.jpg")})
df["filename"] = df["path"].apply(os.path.basename)
df["label"] = df["path"].apply(get_label)
# df

train_df, valid_df = train_test_split(df, test_size=0.3, 
                    stratify=df["label"], random_state=42)
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
        iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.loc[idx, "label"]

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

batch_size = 4
train_dataset = ImageDataset(train_df, 
                             transform_train, target_transform)
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, shuffle=True, **kwargs)

test_dataset = ImageDataset(valid_df,
                            transform_test, target_transform)
test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, shuffle=False, **kwargs)

# x, y = next(iter(train_loader))

# print(x.shape, x.dtype, x.min(), x.max())
# print(y.shape, y.dtype, y.min(), y.max())

from torchvision.models import resnet50

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        resnet = resnet50()
        model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
        resnet.load_state_dict(torch.load(model_dir + "resnet50-11ad3fa6.pth"))
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = x.view(-1, 2048)
        x = self.fc(x)
        return x

# encoder = Encoder(latent_dim=64).to(device)
# images = torch.randn(16, 3, 256, 256).to(device)
# print(f">> input: {images.shape}")

# latent = encoder(images)
# print(f">> latent: {latent.shape}")

# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, 
#                         kernel_size=3, padding=1)
#         self.relu = nn.LeakyReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         return x

# class UpConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
#                         kernel_size=4, stride=2, padding=1)  
#         self.relu = nn.LeakyReLU()

#     def forward(self, x):
#         x = self.upconv(x)
#         x = self.relu(x)
#         return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2) 
        self.conv = nn.Conv2d(in_channels, out_channels, 
                    kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.conv1 = Conv(3, 64)
#         self.conv2 = Conv(64, 128)
#         self.conv3 = Conv(128, 256)
#         self.conv4 = Conv(256, 512)
#         self.fc = nn.Linear(512*16*16, latent_dim)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.view(-1, 512*16*16)
#         x = self.fc(x)
#         return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*16*16)
        self.upconv1 = UpConv(512, 512)
        self.upconv2 = UpConv(512, 256)
        self.upconv3 = UpConv(256, 128)
        self.upconv4 = UpConv(128, 64)
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.view(-1, 512, 16, 16)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.output_conv(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

# # Example usage:
# encoder = Encoder(latent_dim=128)
# decoder = Decoder(latent_dim=128)

# images = torch.randn(16, 3, 256, 256)
# print(f">> input: {images.shape}")

# latent = encoder(images)
# print(f">> latent: {latent.shape}")

# output = decoder(latent)
# print(f">> output: {output.shape}")

latent_dim = 128
model = AutoEncoder(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=50, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

