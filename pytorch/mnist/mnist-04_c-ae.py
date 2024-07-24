import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import Trainer
from mnist import MNIST, get_dataloader


class MLPEncoder(nn.Module):
    def __init__(self, latent_dim=2, n_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28 + n_classes, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),)
        
    def forward(self, x, y):
        x_ = nn.Flatten()(x)
        y_ = nn.functional.one_hot(y, num_classes=10)
        inputs = torch.cat([x_, y_], dim=-1)
        z = self.encoder(inputs)
        return z
        
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=2, n_classes=10):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)
    
    def forward(self, z, y):
        y_ = nn.functional.one_hot(y, num_classes=10)
        inputs = torch.cat([z, y_], dim=-1)
        x_pred = self.decoder(inputs)
        return x_pred


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=2, n_classes=10, embedding_dim=100):
        super().__init__()
        self.labels_reshape = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.Linear(embedding_dim, 1*28*28),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + 1, 32, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),)
        
    def forward(self, x, y):
        y_ = self.labels_reshape(y)
        # print(f"Encoder >> x: {x.shape}, y_: {y_.shape}")
        inputs = torch.cat([x, y_], dim=1)
        return self.encoder(inputs)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=2, n_classes=10, embedding_dim=100):
        super().__init__()
        self.noises_reshape = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),)

        self.labels_reshape = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.Linear(embedding_dim, 1*7*7),
            nn.Unflatten(dim=1, unflattened_size=(1, 7, 7)),)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64 + 1, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=1, padding=1),
            nn.Sigmoid(),)
    
    def forward(self, z, y):
        z_ = self.noises_reshape(z)
        y_ = self.labels_reshape(y)
        # print(f"Dncoder >> z_: {z_.shape}, y_: {y_.shape}")
        inputs = torch.cat([z_, y_], dim=1)
        return self.decoder(inputs)


class ConditionalAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        z = self.encoder(x, y)
        return self.decoder(z, y)

def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()


class TrainerConditionalAE(Trainer):
    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        x_pred = self.model(x, y)
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
        x_pred = self.model(x, y)
        res = {metric: metric_fn(x_pred, x).item() 
               for metric, metric_fn in self.metrics.items()}
        return res


if __name__ == "__main__":

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_type = "cnn"

    data_dir = "/home/namu/myspace/NAMU_Tutorial/MNIST/Pytorch/MNIST/raw"
    # data_dir = "/home/namu/myspace/data/fashion_mnist"
    
    mnist = MNIST(data_dir)
    train_images, train_labels = mnist.get_train_data()
    test_images, test_labels = mnist.get_test_data()
    
    train_loader = get_dataloader((train_images, train_labels), 
                                  batch_size=64, training=True,
                                  use_cuda=use_cuda)
    test_loader = get_dataloader((test_images, test_labels),
                                 batch_size=32, training=False, 
                                 use_cuda=use_cuda)

    print(device)
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)
    
    x, y = next(iter(train_loader))
    print(x.shape, x.dtype, x.min().item(), x.max().item())
    print(y.shape, y.dtype, y.min().item(), y.max().item())


    if model_type == "cnn":
        print(">> Model Type: CNN")
        encoder = CNNEncoder()
        decoder = CNNDecoder()
    else:
        print(">> Model Type: MLP")
        encoder = MLPEncoder()
        decoder = MLPDecoder()
    
    model = ConditionalAE(encoder, decoder).to(device)
    
    loss_fn = nn.BCELoss()
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    metrics = {"acc": binary_accuracy}
    
    cae = TrainerConditionalAE(model, optimizer, loss_fn, metrics=metrics)
    hist = cae.fit(train_loader, n_epochs=10, valid_loader=test_loader)
    
    res = cae.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
