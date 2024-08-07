import os
import pickle
import numpy as np


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


if __name__ == "__main__":

    data_dir = "/home/namu/myspace/data/cifar-10-batches-py/"
    
    (x_train, y_train), (x_test, y_test) = get_cifar10(data_dir)
    class_names = get_classes(data_dir)

    print(f">> train data: {x_train.shape}, {y_train.shape}")
    print(f">> test data:  {x_test.shape}, {y_test.shape}")
