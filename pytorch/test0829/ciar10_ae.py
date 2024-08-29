import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import pickle

def unpickle(filename):
    # tar -zxvf cifar-10-python.tar.gz
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    x = np.array(data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(data[b'labels'])
    return x, y


def get_cifar10(data_dir):
    filenames = [os.path.join(data_dir, f"data_batch_{i+1}") for i in range(5)]

    images, labels = [], []
    for filename in filenames:
        x, y = unpickle(filename)
        images.append(x)
        labels.append(y)

    x_train = np.concatenate(images, axis=0)
    y_train = np.concatenate(labels, axis=0)

    filename = os.path.join(data_dir, "test_batch")
    x_test, y_test = unpickle(filename)

    return (x_train, y_train), (x_test, y_test)


def get_classes(data_dir):
    filename = os.path.join(data_dir, "batches.meta")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['label_names']

data_dir = "/home/namu/myspace/NAMU/datasets/cifar-10-batches-py/"
# data_dir = "D:\\Non_Documents\\datasets\\cifar10\\cifar-10-batches-py\\"
    
(X_train, y_train), (X_test, y_test) = get_cifar10(data_dir)
class_names = get_classes(data_dir)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print(X_train.shape, X_train.min(), X_train.max())
print(X_test.shape, X_train.min(), X_train.max())

class ImageDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
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
          "num_workers": 8, "pin_memory": True } if use_cuda else {}

train_data, test_data = get_cifar10(data_dir)
train_dataset = ImageDataset(train_data, transform_train, target_transform)
test_dataset = ImageDataset(test_data, transform_test, target_transform)

batch_size = 256

train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, shuffle=True, **kwargs)

test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, shuffle=False, **kwargs)

def show_images(images, labels=None, n_images=5, unit=2):
    fig, axes = plt.subplots(ncols=n_images, figsize=(n_images*unit, unit))
    for i in range(n_images):
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img = std * img + mean
        # img = np.clip(img, 0, 1)

        img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_axis_off()
        if labels is not None:
            label = class_names[labels[i].cpu().numpy()]
            axes[i].set_title(label)
    fig.tight_layout()
    plt.show()

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.MaxPool2d(2,2))

        self.conv2 = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(2,2))
        self.linear = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 32 * 8 * 8)
        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        )

        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 32, 8, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

from trainer import TrainerAE, binary_accuracy, EarlyStopping
from pytorch_mssim import ssim

device = torch.device("cuda" if use_cuda else "cpu")

latent_dim = 1024
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.1 * (1 - ssim(pred, target)) + 0.9 * bce(pred, target)

loss_fn = recon_loss
metrics = {"acc": binary_accuracy, "ssim": ssim}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=50, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

# >> Epoch[ 1/50] loss=0.575, acc=0.850, ssim=0.546 | val_loss=0.537, val_acc=0.904, val_ssim=0.738   
# >> Epoch[ 2/50] loss=0.529, acc=0.915, ssim=0.769 | val_loss=0.527, val_acc=0.920, val_ssim=0.787   
# >> Epoch[ 3/50] loss=0.524, acc=0.924, ssim=0.800 | val_loss=0.523, val_acc=0.926, val_ssim=0.808   
# >> Epoch[ 4/50] loss=0.521, acc=0.929, ssim=0.816 | val_loss=0.524, val_acc=0.925, val_ssim=0.815   
# >> Epoch[ 5/50] loss=0.519, acc=0.932, ssim=0.827 | val_loss=0.519, val_acc=0.933, val_ssim=0.830   
# >> Epoch[ 6/50] loss=0.518, acc=0.934, ssim=0.836 | val_loss=0.517, val_acc=0.937, val_ssim=0.839   
# >> Epoch[ 7/50] loss=0.516, acc=0.937, ssim=0.843 | val_loss=0.516, val_acc=0.938, val_ssim=0.846   
# >> Epoch[ 8/50] loss=0.515, acc=0.938, ssim=0.850 | val_loss=0.515, val_acc=0.939, val_ssim=0.852   
# >> Epoch[ 9/50] loss=0.514, acc=0.940, ssim=0.855 | val_loss=0.515, val_acc=0.941, val_ssim=0.857   
# >> Epoch[10/50] loss=0.514, acc=0.941, ssim=0.860 | val_loss=0.514, val_acc=0.940, val_ssim=0.860   
# >> Epoch[11/50] loss=0.513, acc=0.942, ssim=0.864 | val_loss=0.513, val_acc=0.942, val_ssim=0.865   
# >> Epoch[12/50] loss=0.512, acc=0.943, ssim=0.868 | val_loss=0.513, val_acc=0.943, val_ssim=0.867   
# >> Epoch[13/50] loss=0.512, acc=0.944, ssim=0.871 | val_loss=0.512, val_acc=0.943, val_ssim=0.872   
# >> Epoch[14/50] loss=0.512, acc=0.945, ssim=0.874 | val_loss=0.512, val_acc=0.945, val_ssim=0.875   
# >> Epoch[15/50] loss=0.511, acc=0.945, ssim=0.876 | val_loss=0.512, val_acc=0.946, val_ssim=0.877   
# >> Epoch[16/50] loss=0.511, acc=0.946, ssim=0.878 | val_loss=0.511, val_acc=0.946, val_ssim=0.879   
# >> Epoch[17/50] loss=0.511, acc=0.946, ssim=0.880 | val_loss=0.511, val_acc=0.948, val_ssim=0.881   
# >> Epoch[18/50] loss=0.510, acc=0.947, ssim=0.882 | val_loss=0.510, val_acc=0.948, val_ssim=0.883   
# >> Epoch[19/50] loss=0.510, acc=0.947, ssim=0.884 | val_loss=0.510, val_acc=0.948, val_ssim=0.884   
# >> Early stopped!

x, y = next(iter(test_loader))
images, labels = x.to(device), y.to(device)
model.eval()
pred = model(images)

show_images(images[:10], n_images=10, unit=1.5)
show_images(pred[:10].detach(), n_images=10, unit=1.5)
