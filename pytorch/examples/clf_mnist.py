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
    def __init__(self, n_classes, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, n_classes)  

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  
        x = self.fc1(x)
        return x


def accuracy(y_pred, y_true):
    """ Multi-class classification (y_pred: logtis without softmax) """
    y_pred = y_pred.argmax(dim=1)                   ## int64 (long)
    return torch.eq(y_pred, y_true).float().mean()


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

    from mnist import get_mnist
    
    # data_dir = "/home/namu/myspace/NAMU_Tutorial/MNIST/Pytorch/MNIST/raw/"
    data_dir = "/home/namu/myspace/data/fashion_mnist/"
    train_data, test_data = get_mnist(data_dir)
    
    print(f">> train data: {train_data[0].shape}, {train_data[1].shape}")
    print(f">> test data:  {test_data[0].shape}, {test_data[1].shape}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    train_dataset = ImageDataset(train_data, transform=transform)
    test_dataset = ImageDataset(test_data, transform=transform)

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

    model = ConvNet(n_classes=10, n_channels=1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    metrics = {"acc": accuracy}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    early_stopper = EarlyStopping(patience=3, min_delta=0.001)
    
    clf = Trainer(model, optimizer, loss_fn, metrics)
    hist = clf.fit(train_loader, n_epochs=50, valid_loader=test_loader,
                   scheduler=scheduler, early_stopper=early_stopper)
    
    res = clf.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
    
