import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders
from pytorch_mssim import ssim

from unet import UNet
# from autoencoder import ConvBlock, UpConvBlock
# from autoencoder import Encoder, Decoder, AutoEncoder

from trainer import TrainerAE, EarlyStopping, set_seed
from trainer import binary_accuracy, psnr
from trainer import show_images, save_model, save_history


## Set Random seed
set_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#####################################################################
## Hyperparameters
#####################################################################

latent_dim = 1024
img_size = 256
batch_size = 8
split_name = "unet"
patience = 10

#####################################################################
n_epochs = 100
learning_rate = 1e-3
early_stopping = True
lr_scheduling = True

model_dir = f"/home/namu/myspace/NAMU/office/test_0910/{split_name}/"
model_name = f"{split_name}_latent-{latent_dim}_size-{img_size}_batch-{batch_size}"
data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/"


def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.5 * (1 - ssim(pred, target)) + 0.5 * bce(pred, target)

loss_fn = recon_loss
metrics = {"mse": nn.MSELoss(),
           "bce": nn.BCELoss(),
           "acc": binary_accuracy,
           "ssim": ssim,
           "psnr": psnr}


if __name__ == "__main__":

    ## Load data loaders
    train_loader, valid_loader = get_loaders(data_dir, batch_size, img_size)

    ## Train model
    model = UNet(latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=10, min_delta=0.001) \
        if EarlyStopping else None
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) \
        if lr_scheduling else None

    ae = TrainerAE(model, optimizer, loss_fn, metrics)
    hist = ae.fit(train_loader, n_epochs=n_epochs, valid_loader=valid_loader,
                scheduler=scheduler, early_stopper=early_stopper)

    ## Save results
    save_model(model, model_dir, model_name)
    save_history(hist, model_dir, model_name)
