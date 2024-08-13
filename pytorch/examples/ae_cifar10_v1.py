import sys

common_dir = "/home/namu/myspace/pjt_autoencoder/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from trainer import Trainer, EarlyStopping


class ImageDataset(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample1 = nn.Upsample(scale_factor=2) 
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.upsample2 = nn.Upsample(scale_factor=2)  
        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.relu3(self.conv3(x))
        x = self.upsample2(x)
        x = self.conv4(x)
        x = torch.sigmoid(x) # Output pixel values between 0 and 1
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TrainerAE(Trainer):
    def train_step(self, x, y):
        x = x.to(self.device)
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
        x = x.to(self.device)
        x_pred = self.model(x)
        res = {metric: metric_fn(x_pred, x).item() 
               for metric, metric_fn in self.metrics.items()}
        return res
    
def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()  


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
                
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ===============================================================
    # Dataset / Dataloader
    # ===============================================================

    # from mnist import get_mnist
    from cifar10 import get_cifar10

    data_dir = "/home/namu/myspace/NAMU/datasets/cifar-10-batches-py/"
    train_data, test_data = get_cifar10(data_dir)

    print(f">> train data: {train_data[0].shape}, {train_data[1].shape}")
    print(f">> test data:  {test_data[0].shape}, {test_data[1].shape}")

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def target_transform(label):
        return torch.tensor(label).long()

    train_dataset = ImageDataset(train_data, transform_train, target_transform)
    test_dataset = ImageDataset(test_data, transform_test, target_transform)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    batch_size = 64

    kwargs = {"worker_init_fn": seed_worker, "generator": g,
                "num_workers": 4, "pin_memory": True } if use_cuda else {}
    train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset, 
                                batch_size=batch_size, shuffle=False, **kwargs)

    # ===============================================================
    # Modeling / Training
    # ===============================================================

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = Autoencoder(encoder, decoder).to(device)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metrics = {"acc": binary_accuracy}
    early_stopper = EarlyStopping(patience=3, min_delta=0.001)

    ae = TrainerAE(model, optimizer, loss_fn, metrics)
    hist = ae.fit(train_loader, n_epochs=10, valid_loader=test_loader,
                early_stopper=early_stopper)

    res = ae.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
