## Set Random seed
import torch
from trainer import set_seed

set_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load test data
import os
import pandas as pd
from glob import glob

dir_list = [
    "/home/namu/myspace/NAMU/datasets/data_2024/dots/",
    "/home/namu/myspace/NAMU/datasets/data_2024/lines/",
]

def get_test_df(dir_list, label):
    def get_filename(path):
        return os.path.basename(path)[:-4]

    data_paths = []
    for data_dir in dir_list:
        data_paths += glob(data_dir + "*.png")

    df = pd.DataFrame({"path": data_paths})
    df["filename"] = df["path"].apply(get_filename)
    df["label"] = label
    return df

test_df = get_test_df(dir_list, label=0)
len(test_df)

# load train/valid data
from dataset import get_df
from sklearn.model_selection import train_test_split

data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/"

df = get_df(data_dir)
train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

len(train_df), len(valid_df)

# data loaders: train / valid / test
import torch
from dataset import TransformTrain, TransformValid, ImageDataset
from torch.utils.data import DataLoader

img_size = 256
batch_size = 1

transform_valid = TransformValid(img_size)
target_transform = lambda x: torch.tensor(x).long()

train_dataset = ImageDataset(train_df, transform_valid, target_transform)
valid_dataset = ImageDataset(valid_df, transform_valid, target_transform)
test_dataset = ImageDataset(test_df, transform_valid, target_transform)

kwargs = {"num_workers": 8, "pin_memory": True }
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

## Load trained model
from autoencoder import Decoder, AutoEncoder
from models import EncoderResnet50

img_size = 256
batch_size = 8

split_name = "resnet50"
latent_dim = 1024

model_dir = f"/home/namu/myspace/NAMU/office/test_0907/{split_name}/"
model_name = f"{split_name}_latent-{latent_dim}_size-{img_size}_batch-{batch_size}"

encoder = EncoderResnet50(latent_dim)
decoder = Decoder(latent_dim, img_size)
model = AutoEncoder(encoder, decoder)

model.load_state_dict(torch.load(model_dir + model_name + ".pth"))

import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from trainer import binary_accuracy, psnr
from pytorch_mssim import ssim

class TransformValid:
    def __init__(self, img_size):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),])

    def __call__(self, x):
        return self.transform(x)

def load_image(path, transform, img_size):
    image = cv2.imread(path)
    iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    return image

def recon_loss(pred, target):
    bce = nn.BCELoss()
    return 0.5 * (1 - ssim(pred, target)) + 0.5 * bce(pred, target)

metrics = {
    "loss": recon_loss,
    "mse": nn.MSELoss(),
    "bce": nn.BCELoss(),
    "acc": binary_accuracy,
    "ssim": ssim,
    "psnr": psnr,
    "l1": nn.L1Loss(),
    "huber": nn.HuberLoss(),
}

def get_result(data_loader, metrics):
    res = {metric: [] for metric in metrics}

    model.to(device)
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, ascii=True) as pbar:
            for x, _ in pbar:
                x = x.to(device)
                pred = model(x)

                for metric, metric_fn in metrics.items():
                    if metric not in ("path", "filename", "label"):
                        res[metric].append(metric_fn(pred, x).item())
    return res

test_result = get_result(test_loader, metrics)
test_result = pd.DataFrame(test_result)
test_result["path"] = test_df["path"]
test_result["filename"] = test_df["filename"]
test_result["label"] = test_df["label"]

test_result.to_csv(f"./results/test_{model_name}.csv", index=False)

valid_result = get_result(valid_loader, metrics)
valid_result = pd.DataFrame(valid_result)
valid_result["path"] = valid_df["path"]
valid_result["filename"] = valid_df["filename"]
valid_result["label"] = valid_df["label"]

valid_result.to_csv(f"./results/valid_{model_name}.csv", index=False)

train_result = get_result(train_loader, metrics)
train_result = pd.DataFrame(train_result)
train_result["path"] = train_df["path"]
train_result["filename"] = train_df["filename"]
train_result["label"] = train_df["label"]

train_result.to_csv(f"./results/train_{model_name}.csv", index=False)
