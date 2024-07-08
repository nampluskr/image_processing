import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import sys
from tqdm import tqdm
import gzip


def load(file_path, image=False):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data

## =====================================================================
## dataset / dataloader

def preprocess(x, y):
    x = x.astype("float32")/255.  # float
    y = y.astype("int64")         # long
    return x.reshape(x.shape[0], -1), y

class FashionMist(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, label = preprocess(self.images[idx], self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            image = self.target_transform(image)
        
        return image, label
    
## =====================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
def accuracy(y_pred, y_true):
    """ Multi-class classification (y_pred: logtis without softmax) """
    y_pred = y_pred.argmax(dim=1)                   ## int64 (long)
    return torch.eq(y_pred, y_true).float().mean()

## =====================================================================
class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = {"loss": loss_fn}
        self.metrics.update(metrics)

        self.hist = {metric: [] for metric in self.metrics}
        self.options = dict(leave=False, file=sys.stdout, ascii=True)   # ncols=100
        self.device = next(model.parameters()).device

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()

        step_res = {"loss": loss.item()}
        for metric, metric_fn in self.metrics.items():
            if metric != "loss":
                step_res[metric] = metric_fn(pred, y).item()
        return step_res

    @torch.no_grad()
    def test_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        step_res = {metric: metric_fn(pred, y).item()
                    for metric, metric_fn in self.metrics.items()}
        return step_res

    def update_history(self, res):
        for metric, value in res.items():
            self.hist[metric].append(value.mean())

    def fit(self, train_loader, n_epochs, valid_loader=None):
        if valid_loader is not None:
            self.hist.update({f"val_{metric}": [] for metric in self.metrics})

        for epoch in range(1, n_epochs + 1):

            ## Training
            self.model.train()
            res = {metric: np.array([]) for metric in self.metrics}
            with tqdm(train_loader, **self.options) as pbar:
                for x, y in pbar:
                    step_res = self.train_step(x, y)
                    for metric in step_res:
                        res[metric] = np.append(res[metric], step_res[metric])

                    train_desc = f"Epoch[{epoch:2d}/{n_epochs:2d}] "
                    train_desc += ', '.join([f"{m}={v.mean():.3f}" for m, v in res.items()])
                    pbar.set_description(train_desc)

            ## Validation
            self.model.eval()
            if valid_loader is None:
                print(train_desc)
                self.update_history(res)
                continue

            val_res = {f"val_{metric}": np.array([]) for metric in self.metrics}
            with tqdm(valid_loader, **self.options) as pbar:
                for x, y in pbar:
                    step_res = self.test_step(x, y)
                    for metric in step_res:
                        val_res[f"val_{metric}"] = np.append(val_res[f"val_{metric}"], step_res[metric])

                    valid_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in val_res.items()])
                    pbar.set_description(">> " + valid_desc)

            print(train_desc, "|", valid_desc)
            self.update_history(res)
            self.update_history(val_res)

        return self.hist

    def evaluate(self, test_loader):
        self.model.eval()
        test_res = {metric: np.array([]) for metric in self.metrics}
        with tqdm(test_loader, **self.options) as pbar:
            for x, y in pbar:
                step_res = self.test_step(x, y)
                for metric in test_res:
                    test_res[metric] = np.append(test_res[metric], step_res[metric])

                test_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in test_res.items()])
                pbar.set_description(">> " + test_desc)

        print(">> " + test_desc)
        return {metric: value.mean() for metric, value in test_res.items()}

if __name__ == "__main__":

  data_dir = ".\\data\\FashionMNIST"
  
  train_images = load(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), image=True)
  train_labels = load(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), image=False)
  test_images = load(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), image=True)
  test_labels = load(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), image=False)
  
  print(train_images.shape, train_labels.shape)
  print(test_images.shape, test_labels.shape)
  
  train_data = FashionMist(train_images, train_labels)
  test_data = FashionMist(test_images, test_labels)
      
  train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN().to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  clf = Trainer(model, optimizer, loss_fn, metrics={"acc": accuracy})
  hist = clf.fit(train_loader, n_epochs=5, valid_loader=test_loader)
