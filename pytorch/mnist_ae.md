```python
import os
import random
import numpy as np
import gzip

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from trainer import Trainer

## dataset / dataloader

def load(file_path, image=False):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()
        image = image.unsqueeze(dim=0) / 255.
        label = torch.tensor(self.labels[idx]).long()
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            image = self.target_transfomr(image)
        
        return image, label
    
def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    dataloader = DataLoader(Dataset(data), 
                            batch_size=batch_size,
                            shuffle=training, **kwargs)
    return dataloader

class AutoencoderMLP(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()   

class AutoencoderCNN(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=1, padding=1),
            nn.Sigmoid(),)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()

manual_seed = 42
random.seed(manual_seed)
np.random.seed(manual_seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# data_dir = "/home/namu/myspace/NAMU_Tutorial/MNIST/Pytorch/MNIST/raw"
data_dir = "/home/namu/myspace/data/fashion_mnist"

train_images = load(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), image=True)
train_labels = load(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), image=False)
test_images = load(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), image=True)
test_labels = load(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), image=False)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

train_loader = get_dataloader((train_images, train_labels), 
                              batch_size=64, training=True, use_cuda=True)
test_loader = get_dataloader((test_images, test_labels),
                             batch_size=32, training=False, use_cuda=True)

x, y = next(iter(train_loader))

print(x.shape, x.dtype, x.min().item(), x.max().item())
print(y.shape, y.dtype, y.min().item(), y.max().item())

class TrainerAE(Trainer):
    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        x_pred = self.model(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()

        res = {"loss": loss.item()}
        for metric, metric_fn in self.metrics.items():
            if metric != "loss":
                res[metric] = metric_fn(x_pred, x).item()
        return res

    @torch.no_grad()
    def test_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        x_pred = self.model(x)
        res = {metric: metric_fn(x_pred, x).item() 
               for metric, metric_fn in self.metrics.items()}
        return res

model = AutoencoderMLP(latent_dim=10).to(device)
loss_fn = nn.BCELoss()
# loss_fn = nn.L1Loss()
# loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
metrics = {"acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics=metrics)
hist = ae.fit(train_loader, n_epochs=10, valid_loader=test_loader)

model = AutoencoderCNN(latent_dim=10).to(device)
loss_fn = nn.BCELoss()
# loss_fn = nn.L1Loss()
# loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
metrics = {"acc": binary_accuracy}

ae = Trainer(model, optimizer, loss_fn, metrics=metrics)
hist = ae.fit(train_loader, n_epochs=10, valid_loader=test_loader)

img = img.squeeze().numpy()
pred = pred.cpu().squeeze().detach().numpy()

print(type(img), img.shape, img.max())
print(type(pred), pred.shape, pred.max())

img, _ = next(iter(test_loader))
pred = model(img.to(device))

import matplotlib.pyplot as plt

idx = 4

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
ax1.imshow(img[idx], cmap="Greys_r")
ax2.imshow(pred[idx], cmap="Greys_r")
plt.show()
```
