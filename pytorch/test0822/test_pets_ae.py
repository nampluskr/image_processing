import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, EarlyStopping, set_seed, accuracy
from trainer import TrainerAE, binary_accuracy

import os
from glob import glob
import numpy as np
import random
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

set_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



from autoencoder import AutoEncoder, ConvEncoder, ConvDecoder

latent_dim = 128
encoder = ConvEncoder(latent_dim)
decoder = ConvDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

from autoencoder import AutoEncoder, ResEncoder, ResDecoder

latent_dim = 128
encoder = ResEncoder(latent_dim)
decoder = ResDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()
metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss(), "acc": binary_accuracy}

ae = TrainerAE(model, optimizer, loss_fn, metrics)
hist = ae.fit(train_loader, n_epochs=5, valid_loader=test_loader,
              early_stopper=EarlyStopping(patience=3, min_delta=0.001))

res = ae.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
