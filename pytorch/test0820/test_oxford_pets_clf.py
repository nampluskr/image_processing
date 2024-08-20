import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, EarlyStopping, set_seed, accuracy

import os
from glob import glob
import numpy as np
import random
import pandas as pd
import cv2

import torch
import torch.nn as nn
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

batch_size = 16
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

from torchvision.models import efficientnet_b0

model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
model = efficientnet_b0()
model.load_state_dict(torch.load(model_dir + "efficientnet_b0_rwightman-7f5810bc.pth"))

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=37, bias=True),
)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
metrics = {"acc": accuracy}
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
early_stopper = EarlyStopping(patience=3, min_delta=0.001)

clf = Trainer(model, optimizer, loss_fn, metrics)
hist = clf.fit(train_loader, n_epochs=50, valid_loader=test_loader,
               scheduler=scheduler, early_stopper=early_stopper)

res = clf.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

