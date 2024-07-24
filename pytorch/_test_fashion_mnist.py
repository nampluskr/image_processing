import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import gzip

from pt_trainer import Trainer


def load(file_path, image=False):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
    return data.reshape(-1, 28, 28) if image else data

def preprocess(x, y):
    x = x.astype("float32")/255.  # float
    y = y.astype("int64")         # long
    return x.reshape(x.shape[0], -1), y

class FashionMist(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = preprocess(self.images[idx], self.labels[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            image = self.target_transform(image)

        return image, label


## 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
    """ Multi-class classification (y_pred: logtis without sofrmax) """
    y_pred = y_pred.argmax(dim=1)                   ## int64 (long)
    return torch.eq(y_pred, y_true).float().mean()


if __name__ == "__main__":

    ## 데이터 로딩
    data_dir = "D:\\Non_Documents\\01_work_2024\\ai_prj_2024_v2\\pytorch\\data\\FashionMnist"

    train_images = load(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), image=True)
    train_labels = load(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), image=False)
    test_images = load(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), image=True)
    test_labels = load(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), image=False)

    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    train_data = FashionMist(train_images, train_labels)
    test_data = FashionMist(test_images, test_labels)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    ## Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    clf = Trainer(model, optimizer, loss_fn, metrics={"acc": accuracy})
    
    print("\n[Training]")
    hist = clf.fit(train_loader, n_epochs=100, valid_loader=test_loader)
    # hist = clf.fit(train_loader, n_epochs=10)
    
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
    # ax1.plot(hist["loss"], 'k', label="training")
    # ax1.plot(hist["val_loss"], 'r', label="validation")
    # ax2.plot(hist["acc"], 'k', label="training")
    # ax2.plot(hist["val_acc"], 'r', label="validation")
    # plt.show()

    ## Evaluation
    res = clf.evaluate(test_loader)
    print("\n[Evaluation]")
    print(f">> loss={res['loss']:.3f}, acc={res['acc']:.3f}")
