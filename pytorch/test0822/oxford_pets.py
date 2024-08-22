import random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from sklearn.model_selection import train_test_split


# https://github.com/tensorflow/models/issues/3134
images_png = [
    "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
    "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
    "Abyssinian_5", "Abyssinian_34",
]
images_corrupt = ["chihuahua_121", "beagle_116"]


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
        # image = read_image(path)

        image = cv2.imread(path)
        if len(image.shape) == 3 and image.shape[2] == 4:   # png: images_png
            iamge = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.loc[idx, "SPECIES"]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_dataloaders(data_dir, batch_size, img_shape, use_cuda):

    ## get dataframe for image paths
    data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
    df = pd.read_table(data_dir + "annotations/list.txt",
            sep=' ', skiprows=6, header=None,
            names=["CLASS_ID", "SPECIES", "BREED", "ID"])
    df[["SPECIES", "BREED", "ID"]] -= 1
    df["path"] = data_dir + "images/" + df["CLASS_ID"] + ".jpg"

    ## delete corrupted files
    for idx, row in df.iterrows():
        # if row["CLASS_ID"] in images_png:
        #     df.loc[idx, "path"] = df.loc[idx, "path"].replace("jpg", "png")
        #     df.drop([idx], axis=0, inplace=True)
        if row["CLASS_ID"] in images_corrupt:
            df.drop([idx], axis=0, inplace=True)

    ## split train and valid dataframes
    train_df, valid_df = train_test_split(df, test_size=0.3,
                    stratify=df["SPECIES"], random_state=42)
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    ## transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_shape),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_shape),
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

    train_dataset = ImageDataset(train_df, transform_train, target_transform)
    valid_dataset = ImageDataset(valid_df, transform_valid, target_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader


if __name__ == "__main__":

    data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets/"
    batch_size = 128
    img_shape = (256, 256)
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader = get_dataloaders(data_dir,
                            batch_size, img_shape, use_cuda)

    x, y = next(iter(train_loader))
    print("\n>> Train data:")
    print(x.shape, x.dtype, x.min(), x.max())
    print(y.shape, y.dtype, y.min(), y.max())

    x, y = next(iter(valid_loader))
    print("\n>> Validation data:")
    print(x.shape, x.dtype, x.min(), x.max())
    print(y.shape, y.dtype, y.min(), y.max())
