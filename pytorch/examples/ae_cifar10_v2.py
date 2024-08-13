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
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32 * 8 * 8)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 32, 8, 8)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
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

    train_dataset = ImageDataset(train_data, transform_train, target_transform)
    test_dataset = ImageDataset(test_data, transform_test, target_transform)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    batch_size = 32

    kwargs = {"worker_init_fn": seed_worker, "generator": g,
              "num_workers": 8, "pin_memory": True } if use_cuda else {}
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size, shuffle=False, **kwargs)

    # ===============================================================
    # Modeling / Training
    # ===============================================================

    latent_dim = 64
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    model = Autoencoder(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.BCELoss()
    metrics = {"mse": nn.MSELoss(), 
               "L1": nn.L1Loss(),
               "acc": binary_accuracy}
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    early_stopper = EarlyStopping(patience=3, min_delta=0.0001)

    ae = TrainerAE(model, optimizer, loss_fn, metrics)
    hist = ae.fit(train_loader, n_epochs=50, valid_loader=test_loader,
                  scheduler=scheduler, early_stopper=early_stopper)

    res = ae.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

