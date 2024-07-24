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

class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc(x)
        return logits
    
class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

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

model = MLP(n_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

clf = Trainer(model, optimizer, loss_fn, metrics={"acc": accuracy})
hist = clf.fit(train_loader, n_epochs=5, valid_loader=test_loader)

model = CNN(n_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

clf = Trainer(model, optimizer, loss_fn, metrics={"acc": accuracy})
hist = clf.fit(train_loader, n_epochs=5, valid_loader=test_loader)

res = clf.evaluate(test_loader)
res
```
