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

def get_label(path):
    class_names = ['NORMAL', 'PNEUMONIA']
    dirname = os.path.dirname(path)
    classname = dirname.split(os.path.sep)[-1]
    return class_names.index(classname)

data_dir = "/home/namu/myspace/NAMU/datasets/chest-xray-pneumonia/chest_xray/train/"
df = pd.DataFrame({"path": glob(data_dir + "*/*.jpeg")})
df["filename"] = df["path"].apply(os.path.basename)
df["label"] = df["path"].apply(get_label)
# df

test_dir = "/home/namu/myspace/NAMU/datasets/chest-xray-pneumonia/chest_xray/val/"
test_df = pd.DataFrame({"path": glob(test_dir + "*/*.jpeg")})
test_df["filename"] = test_df["path"].apply(os.path.basename)
test_df["label"] = test_df["path"].apply(get_label)
# test_df

train_df, valid_df = train_test_split(df, test_size=0.3, 
                    stratify=df["label"], random_state=42)
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
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

# kwargs = {}
batch_size = 8
train_dataset = ImageDataset(train_df, 
                             transform_train, target_transform)
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, shuffle=True, **kwargs)

valid_dataset = ImageDataset(valid_df,
                            transform_test, target_transform)
valid_loader = DataLoader(valid_dataset, 
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
    nn.Linear(in_features=1280, out_features=2, bias=True),
)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
metrics = {"acc": accuracy}
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
early_stopper = EarlyStopping(patience=3, min_delta=0.001)

clf = Trainer(model, optimizer, loss_fn, metrics)
hist = clf.fit(train_loader, n_epochs=50, valid_loader=valid_loader,
               scheduler=scheduler, early_stopper=early_stopper)

res = clf.evaluate(valid_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

test_dataset = ImageDataset(test_df,
                            transform_test, target_transform)
test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, shuffle=False, **kwargs)

res = clf.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

from torchvision.models import resnet50

model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
model = resnet50()
model.load_state_dict(torch.load(model_dir + "resnet50-11ad3fa6.pth"))

model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
metrics = {"acc": accuracy}
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
early_stopper = EarlyStopping(patience=3, min_delta=0.001)

clf = Trainer(model, optimizer, loss_fn, metrics)
hist = clf.fit(train_loader, n_epochs=50, valid_loader=valid_loader,
               scheduler=scheduler, early_stopper=early_stopper)

res = clf.evaluate(valid_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

res = clf.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
