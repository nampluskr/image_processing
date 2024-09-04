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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
os.environ['PYTHONHASHSEED'] = str(seed)

# The below two lines are for deterministic algorithm behavior in CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_df(data_dir):
    def get_product(path):
        return path.split(os.sep)[2]

    def get_filename(path):
        return os.path.basename(path)[:-8]

    def get_pattern(path):
        return os.path.basename(path).split(' ')[0]

    df = pd.DataFrame({"path": glob(data_dir + "*/data_rgb_png/*.png")})
    df["product"] = df["path"].apply(get_product)
    df["pattern"] = df["path"].apply(get_pattern)
    df["filename"] = df["path"].apply(get_filename)
    df["label"] = 1
    return df

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
        iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.loc[idx, "label"]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/"
df = get_df(data_dir)

train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

len(train_df), len(valid_df)

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

# Resize
transform_train_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
])

transform_valid_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset_resize = ImageDataset(train_df, transform_train_resize, target_transform)
train_loader_resize = DataLoader(train_dataset_resize, batch_size=batch_size, shuffle=True, **kwargs)

valid_dataset_resize = ImageDataset(valid_df, transform_valid_resize, target_transform)
valid_loader_resize = DataLoader(valid_dataset_resize, batch_size=batch_size, shuffle=False, **kwargs)

# Random Crop
transform_train_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
])

transform_valid_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((256, 256)),
    transforms.ToTensor(),
])

train_dataset_crop = ImageDataset(train_df, transform_train_crop, target_transform)
train_loader_crop = DataLoader(train_dataset_crop, batch_size=batch_size, shuffle=True, **kwargs)

valid_dataset_crop = ImageDataset(valid_df, transform_valid_crop, target_transform)
valid_loader_crop = DataLoader(valid_dataset_crop, batch_size=batch_size, shuffle=False, **kwargs)

import torch.nn as nn
import torch.optim as optim
from pytorch_mssim import ssim

from trainer import TrainerAE, binary_accuracy, EarlyStopping
from trainer import show_images, save_model, save_history
from autoencoder import Encoder, Decoder, AutoEncoder


def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.5 * (1 - ssim(pred, target)) + 0.5 * bce(pred, target)

def psnr(pred, target):
    mse = nn.MSELoss()(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 ** 2 / mse)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# x, y = next(iter(valid_loader_resize))
# images, labels = x.to(device), y.to(device)
# model.eval()
# pred = model(images)

# show_images(images, n_images=8)
# show_images(pred.detach(), n_images=8)
