import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import Trainer
from mnist import MNIST, get_dataloader


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
        model = CNN(n_classes=10).to(device)
    else:
        print(">> Model Type: MLP")
        model = MLP(n_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=3, gamma=0.1)
    
    clf = Trainer(model, optimizer, loss_fn, metrics={"acc": accuracy},
                      scheduler=scheduler)
    hist = clf.fit(train_loader, n_epochs=10, valid_loader=test_loader)
    
    res = clf.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
