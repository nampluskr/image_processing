import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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


class TransformTrain:
    def __init__(self, img_size):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),])

    def __call__(self, x):
        return self.transform(x)


class TransformValid:
    def __init__(self, img_size):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),])

    def __call__(self, x):
        return self.transform(x)


def get_loaders(data_dir, batch_size, img_size):

    df = get_df(data_dir)
    train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    transform_train = TransformTrain(img_size)
    transform_valid = TransformValid(img_size)
    target_transform = lambda x: torch.tensor(x).long()

    train_dataset = ImageDataset(train_df, transform_train, target_transform)
    valid_dataset = ImageDataset(valid_df, transform_valid, target_transform)

    kwargs = {"num_workers": 8, "pin_memory": True }
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader


if __name__ == "__main__":

    data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/"
    batch_size = 8
    img_size = 512
    train_loader, valid_loader = get_loaders(data_dir, batch_size, img_size)

    x, y = next(iter(train_loader))
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
