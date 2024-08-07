import sys

common_dir = "/home/namu/myspace/pytorch_examples/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from trainer import Trainer, EarlyStopping


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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

    # ===============================================================
    # Dataset / Dataloader
    # ===============================================================

    from cifar10 import get_cifar10, get_classes
    
    data_dir = "/home/namu/myspace/data/cifar-10-batches-py/"
    train_data, test_data = get_cifar10(data_dir)
    class_names = get_classes(data_dir)
    
    print(f">> train data: {train_data[0].shape}, {train_data[1].shape}")
    print(f">> test data:  {test_data[0].shape}, {test_data[1].shape}")

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = ImageDataset(train_data, transform=transform_train)
    test_dataset = ImageDataset(test_data, transform=transform_test)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    batch_size = 32
    
    kwargs = {"worker_init_fn": seed_worker, "generator": g,
              "num_workers": 4, "pin_memory": True } if use_cuda else {}
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(test_dataset, 
                              batch_size=batch_size, shuffle=False, **kwargs)

    # ===============================================================
    # Modeling / Training
    # ===============================================================

    model = ConvNet(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = {"acc": accuracy}
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    early_stopper = EarlyStopping(patience=3, min_delta=0.001)
    
    clf = Trainer(model, optimizer, loss_fn, metrics=metrics)
    hist = clf.fit(train_loader, n_epochs=50, valid_loader=test_loader,
                   scheduler=scheduler, early_stopper=early_stopper)
    
    res = clf.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")    
