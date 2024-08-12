import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet50

import random
import numpy as np

from trainer import EarlyStopping, Trainer


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
        label = torch.tensor(label).long()

        if self.transform:
            image = self.transform(image)

        return image, label


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample1 = nn.Upsample(scale_factor=2) 
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.upsample2 = nn.Upsample(scale_factor=2)  
        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.relu3(self.conv3(x))
        x = self.upsample2(x)
        x = torch.sigmoid(self.conv4(x)) # Output pixel values between 0 and 1
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
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ===============================================================
    # Dataset / Dataloader
    # ===============================================================

    from cifar10 import get_cifar10, get_classes
    
    # data_dir = "/home/namu/myspace/data/cifar-10-batches-py/"
    data_dir = "D:\\Non_Documents\\datasets\\cifar10\\cifar-10-batches-py\\"
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

    batch_size = 64
    kwargs = {}
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(test_dataset, 
                              batch_size=batch_size, shuffle=False, **kwargs)
    
    # ===============================================================
    # Modeling / Training
    # ===============================================================
    
    encoder = Encoder()
    decoder = Decoder()
    model = Autoencoder(encoder, decoder).to(device)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metrics = {"acc": binary_accuracy}
    early_stopper = EarlyStopping(patience=3, min_delta=0.001)

    ae = TrainerAE(model, optimizer, loss_fn, metrics=metrics)
    hist = ae.fit(train_loader, n_epochs=50, valid_loader=test_loader,
                  early_stopper=early_stopper)

    res = ae.evaluate(test_loader)
    print(f">> Evaluation: loss={res['loss']:.3f}, acc={res['acc']:.3f}")
    
    
