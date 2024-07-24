import os
import random
import numpy as np
import gzip

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2


class MNIST:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_images = "train-images-idx3-ubyte.gz"
        self.train_labels = "train-labels-idx1-ubyte.gz"
        self.test_images = "t10k-images-idx3-ubyte.gz"
        self.test_labels = "t10k-labels-idx1-ubyte.gz"

    def get_train_data(self):
        images = self.load(os.path.join(self.data_dir, self.train_images), image=True)
        labels = self.load(os.path.join(self.data_dir, self.train_labels), image=False)
        return images, labels

    def get_test_data(self):
        images = self.load(os.path.join(self.data_dir, self.test_images), image=True)
        labels = self.load(os.path.join(self.data_dir, self.test_labels), image=False)
        return images, labels

    def load(self, file_path, image=False):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if image else 8)
        return data.reshape(-1, 28, 28) if image else data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform
        self.target_transform = target_transform

    def preprocess(self, image):
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()
        image = image.unsqueeze(dim=0) / 255.
        label = torch.tensor(self.labels[idx]).long()
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transfomr(label)

        return image, label


def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    dataloader = DataLoader(Dataset(data), 
                            batch_size=batch_size,
                            shuffle=training, 
                            **kwargs)
    return dataloader


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # data_dir = "/home/namu/myspace/NAMU_Tutorial/MNIST/Pytorch/MNIST/raw"
    # data_dir = "/home/namu/myspace/data/fashion_mnist"
    # data_dir = "/mnt/d/datasets/fashion_mnist_29M"
    data_dir = "/mnt/d/datasets/mnist_11M"

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

    
    
