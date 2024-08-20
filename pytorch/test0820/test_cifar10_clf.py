import sys

common_dir = "/home/namu/myspace/NAMU/pytorch/common/"
if common_dir not in sys.path:
    sys.path.append(common_dir)

from trainer import Trainer, EarlyStopping, set_seed, accuracy
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(n_classes=10).to(device)

loss_fn = nn.CrossEntropyLoss()
metrics = {"acc": accuracy}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
early_stopper = EarlyStopping(patience=5, min_delta=0.001)

clf = Trainer(model, optimizer, loss_fn, metrics)
hist = clf.fit(train_loader, n_epochs=50, valid_loader=test_loader,
               scheduler=scheduler, early_stopper=early_stopper)

res = clf.evaluate(test_loader)
print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")

