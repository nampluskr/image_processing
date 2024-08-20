import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, EarlyStopping, set_seed, accuracy
from trainer import TrainerAE, binary_accuracy
set_seed(42)

from cifar10 import get_dataloaders

data_dir = "/home/namu/myspace/NAMU/datasets/cifar-10-batches-py/"
batch_size = 32
train_loader, test_loader = get_dataloaders(data_dir, batch_size)

x, y = next(iter(train_loader))
print(f">> Batch Size: {batch_size}")
print(f">> Batch Images: {x.shape}, {x.dtype}, min={x.min()}, max={x.max()}")
print(f">> Batch Labels: {y.shape}, {y.dtype}, min={y.min()}, max={y.max()}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                        kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.deconv_block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = DeconvBlock(128, 64)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels,
                            kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 64
model = AutoEncoder(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=50, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
