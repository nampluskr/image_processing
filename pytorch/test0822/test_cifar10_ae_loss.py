import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

datasets_dir = "/home/namu/myspace/NAMU/datasets/"
if datasets_dir not in sys.path:
    sys.path.append(datasets_dir)

from trainer import Trainer, EarlyStopping, set_seed, accuracy
from trainer import TrainerAE, binary_accuracy

import torch
import torch.nn as nn
import torch.optim as optim

set_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from cifar10 import get_dataloaders

data_dir = "/home/namu/myspace/NAMU/datasets/cifar-10-batches-py/"
batch_size = 32
train_loader, test_loader = get_dataloaders(data_dir, batch_size, use_cuda)

x, y = next(iter(train_loader))
print(f">> Batch Size: {batch_size}")
print(f">> Batch Images: {x.shape}, {x.dtype}, min={x.min()}, max={x.max()}")
print(f">> Batch Labels: {y.shape}, {y.dtype}, min={y.min()}, max={y.max()}")

from autoencoder import ConvBlock, UpConvBlock, ResBlock, AutoEncoder

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.linear = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.linear(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 128 * 4 * 4)
        self.upconv_block1 = UpConvBlock(128, 64)
        self.upconv_block2 = UpConvBlock(64, 32)
        self.upconv_block3 = UpConvBlock(32, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        x = self.upconv_block3(x)
        return x

latent_dim = 64
encoder = ConvEncoder(latent_dim)
decoder = ConvDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

# class ResEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.conv_block1 = ConvBlock(3, 32)
#         self.res_block1 = ResBlock(32, 32)
#         self.conv_block2 = ConvBlock(32, 64)
#         self.res_block2 = ResBlock(64, 64)
#         self.conv_block3 = ConvBlock(64, 128)
#         self.res_block3 = ResBlock(128, 128)
#         self.linear = nn.Linear(128 * 4 * 4, latent_dim)
        
#     def forward(self, x):
#         x = self.conv_block1(x)
#         x = self.res_block1(x)
#         x = self.conv_block2(x)
#         x = self.res_block2(x)
#         x = self.conv_block3(x)
#         x = self.res_block3(x)
#         x = x.view(-1, 128 * 4 * 4)
#         x = self.linear(x)
#         return x


# class ResDecoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.linear = nn.Linear(latent_dim, 128 * 4 * 4)
#         self.res_block1 = ResBlock(128, 128)
#         self.upconv_block1 = UpConvBlock(128, 64)
#         self.res_block2 = ResBlock(64, 64)
#         self.upconv_block2 = UpConvBlock(64, 32)
#         self.res_block3 = ResBlock(32, 32)
#         self.upconv_block3 = UpConvBlock(32, 3)

#     def forward(self, x):
#         x = self.linear(x)
#         x = x.view(-1, 128, 4, 4)
#         x = self.res_block1(x)
#         x = self.upconv_block1(x)
#         x = self.res_block2(x)
#         x = self.upconv_block2(x)
#         x = self.res_block3(x)
#         x = self.upconv_block3(x)
#         return x

# latent_dim = 64
# encoder = ResEncoder(latent_dim)
# decoder = ResDecoder(latent_dim)
# model = AutoEncoder(encoder, decoder).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# loss_fn = nn.BCELoss()
# metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

# ae = TrainerAE(model, optimizer, loss_fn, metrics)
# hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
#               early_stopper=EarlyStopping(patience=3, min_delta=0.001))

# res = ae.evaluate(test_loader)
# print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

## Huber loss

latent_dim = 64
encoder = ConvEncoder(latent_dim)
decoder = ConvDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.HuberLoss(delta=0.5)
# loss_fn = nn.SmoothL1Loss(beta=1.0)
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

## SSIM
# https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
from pytorch_mssim import ssim

def ssim_loss(x_pred, x):
    bce = nn.BCELoss()
    return 1 - ssim(x_pred, x, val_range=1) + bce(x_pred, x)

def ssim_fn(x_pred, x):
    return ssim(x_pred, x, val_range=1)

latent_dim = 64
encoder = ConvEncoder(latent_dim)
decoder = ConvDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.BCELoss()
metrics = {"ssim": ssim_fn, "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

## Perceptual Loss
from torchvision.models import vgg16

def get_vgg16(latent_dim, freezed=False):
    model = vgg16()
    model_dir = "/home/namu/myspace/NAMU/pytorch/models/"
    model.load_state_dict(torch.load(model_dir + "vgg16-397923af.pth"))

    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=latent_dim, bias=True),
    )
    if freezed:
        for param in model.parameters():
            param.requires_grad = False

    return model

class PerceptualLoss(nn.Module):
    def __init__(self, latent_dim):
        self.vgg = get_vgg16(latent_dim, freezed=True)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.mse(pred_features, target_features)


def loss_fn(pred, target):
    perc_loss = PerceptualLoss(64)
    recon_loss = nn.MSELoss()
    return recon_loss(pred, target) + perc_loss(pred, target)

latent_dim = 64
encoder = ConvEncoder(latent_dim)
decoder = ConvDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# loss_fn = loss_fn
metrics = {"ssim": ssim_fn, "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
