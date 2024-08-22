import os
import pickle
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def unpickle(filename):
    # tar -zxvf cifar-10-python.tar.gz
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    x = np.array(data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(data[b'labels'])
    return x, y


def get_cifar10(data_dir):
    filenames = [os.path.join(data_dir, f"data_batch_{i+1}") for i in range(5)]

    images, labels = [], []
    for filename in filenames:
        x, y = unpickle(filename)
        images.append(x)
        labels.append(y)

    x_train = np.concatenate(images, axis=0)
    y_train = np.concatenate(labels, axis=0)

    filename = os.path.join(data_dir, "test_batch")
    x_test, y_test = unpickle(filename)

    return (x_train, y_train), (x_test, y_test)


def get_classes(data_dir):
    filename = os.path.join(data_dir, "batches.meta")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['label_names']


class ImageDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_dataloaders(data_dir, batch_size, use_cuda):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def target_transform(label):
        return torch.tensor(label).long()

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    kwargs = {"worker_init_fn": seed_worker, "generator": g,
              "num_workers": 8, "pin_memory": True } if use_cuda else {}

    train_data, test_data = get_cifar10(data_dir)
    train_dataset = ImageDataset(train_data, transform_train, target_transform)
    test_dataset = ImageDataset(test_data, transform_test, target_transform)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


if __name__ == "__main__":

    data_dir = "/home/namu/myspace/NAMU/datasets/cifar-10-batches-py/"
    # data_dir = "D:\\Non_Documents\\datasets\\cifar10\\cifar-10-batches-py\\"
    
    (x_train, y_train), (x_test, y_test) = get_cifar10(data_dir)
    class_names = get_classes(data_dir)

    print(f">> Train Data: {x_train.shape}, {y_train.shape}")
    print(f">> Test Data:  {x_test.shape}, {y_test.shape}")

    batch_size = 32
    train_loader, test_loader = get_dataloaders(data_dir, batch_size)

    x, y = next(iter(train_loader))
    print(f">> Batch Size: {batch_size}")
    print(f">> Batch Images: {x.shape}, {x.dtype}, min={x.min()}, max={x.max()}")
    print(f">> Batch Labels: {y.shape}, {y.dtype}, min={y.min()}, max={y.max()}")
