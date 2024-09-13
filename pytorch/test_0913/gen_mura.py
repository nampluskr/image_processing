import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage

import cv2
import skimage
import torch
from glob import glob

def get_torch_image(path, img_size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = skimage.transform.resize(img, (img_size, img_size), anti_aliasing=True)
    img = img.transpose(2, 0, 1)
    return torch.tensor(img).float().unsqueeze(dim=0)

def get_torch_data(data, img_size=256):
    if data.ndim == 2:
        data = skimage.color.gray2rgb(data)
    img = skimage.transform.resize(data, (img_size, img_size), anti_aliasing=True)
    img = img.transpose(2, 0, 1)
    return torch.tensor(img).float().unsqueeze(dim=0)

def show_images(images, unit=2.5, labels=None, cmap=None, grayscale=False):
    with torch.no_grad():
        if (images.device == torch.device("cuda")):
            images = images.cpu()
        n_images = len(images)
        if len(images) > 1:
            fig, axes = plt.subplots(ncols=n_images, figsize=(n_images*unit, unit))
            for i in range(len(images)):
                img = images[i].numpy().transpose(1, 2, 0)
                if grayscale:
                    img = skimage.color.rgb2gray(img)
                if labels is not None:
                    axes[i].set_title(labels[i])
                axes[i].imshow(img, vmin=0, vmax=1, cmap=cmap)
                axes[i].set_axis_off()
        else:
            fig, ax = plt.subplots(figsize=(unit, unit))
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            if grayscale:
                img = skimage.color.rgb2gray(img)
            if labels is not None:
                axes[i].set_title(lables[i])
            ax.imshow(img, vmin=0, vmax=1, cmap=cmap)
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()

def info(img):
    print(f">> shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min():.2f}, max={img.max():.2f}, "
          f"mean={img.mean():.2f}")

import torch.nn as nn
from pytorch_mssim import ssim
from trainer import binary_accuracy, psnr

metrics = {"mse": nn.MSELoss(),
           "bce": nn.BCELoss(),
           "acc": binary_accuracy,
           "ssim": ssim,
           "psnr": psnr}

def predict(model, img):
    model.eval()
    model.to(device)
    img = img.to(device)
    pred = model(img)
    return torch.concat([img.cpu(), pred.cpu(), (img - pred).cpu()])

def evaluate(model, img, metrics=None):
    model.eval()
    model.to(device)
    img = img.to(device)
    pred = model(img)
    res = {metric: metric_fn(pred, img).item()
               for metric, metric_fn in metrics.items()}
    return res

def make_lines(max, min, n_lines, shift, img_size, angle=np.deg2rad(90)):
    amplitue = (max - min) / 2
    mean = (max + min) / 2
    wavelength = img_size // n_lines
    rng = np.arange(-img_size // 2, img_size // 2, 1)
    X, Y = np.meshgrid(rng, rng)
    rot = X*np.cos(angle) + Y*np.sin(angle)
    res = np.sin(2*np.pi*(rot) / wavelength - 2*np.pi / 10 * shift)
    return amplitue * res + mean

img_size = 256
lines = make_lines(max=0.2, min=0.0, n_lines=5, shift=2, img_size=img_size)
info(lines)
img = get_torch_data(lines)
show_images(img, cmap="jet")

def make_gaussian2d(x0, y0, sig, max, min, img_size):
    x = np.arange(-img_size // 2, img_size // 2, 1)
    X, Y = np.meshgrid(x, x)
    X = np.exp(-(X - x0 * img_size)**2 / sig**2 / 2)
    Y = np.exp(-(Y - y0 * img_size)**2 / sig**2 / 2)
    return X * Y * (max - min) + min

img_size = 256
img1 = make_gaussian2d(-0.1, -0.1, sig=10, max=0.2, min=0.0, img_size=256)
img2 = make_gaussian2d(0.1, 0.1, sig=10, max=0.2, min=0.0, img_size=256)

img = get_torch_data(img1 + img2)
info(img)
show_images(img, cmap="jet")

data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/CT3/data_rgb_png/"
path = glob(data_dir + "*.png")[20]

img = get_torch_image(path)
print(img.shape)
show_images(img)

## Load trained model
import torch
from autoencoder import Encoder, Decoder, AutoEncoder

split_name = "convnet"
latent_dim = 1024
img_size = 256
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = f"/home/namu/myspace/NAMU/office/test_0906/{split_name}/"
model_name = f"{split_name}_latent-{latent_dim}_size-{img_size}_batch-{batch_size}_patience-10"

encoder = Encoder(latent_dim, img_size)
decoder = Decoder(latent_dim, img_size)
model = AutoEncoder(encoder, decoder).to(device)

model.load_state_dict(torch.load(model_dir + model_name + ".pth"))

data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/CT3/data_rgb_png/"
path = glob(data_dir + "*.png")[20]

img = get_torch_image(path)
res = predict(model, img)
print(res.shape)
show_images(res, labels=["imgage", "predition", "difference"])

res = evaluate(model, img, metrics=metrics)
print(res)

img_size = 128
img0 = make_gaussian2d(-0.1, -0.1, sig=10, max=0.11, min=0.09, img_size=256)
img0 = get_torch_data(img0)
res = predict(model, img0)
info(res[0])
info(res[1])
show_images(res, labels=["imgage", "predition", "difference"])

res = evaluate(model, img0, metrics=metrics)
print(res)

img_size = 128
img1 = make_gaussian2d(-0.2, -0.1, sig=10, max=0.4, min=0.0, img_size=256)
img2 = make_gaussian2d(0.1, 0.1, sig=10, max=0.4, min=0.0, img_size=256)
img = get_torch_data(img1 + img2)

res = predict(model, img)
info(res[0])
info(res[1])
show_images(res)
# show_images(res, labels=["imgage", "predition", "difference"])

res = evaluate(model, img, metrics=metrics)
print(res)

img_size = 256
lines = make_lines(max=0.3, min=0.3, n_lines=10, shift=2, img_size=img_size)
img0 = get_torch_data(lines)
res = predict(model, img0)
info(res[0])
info(res[1])
show_images(res)

res = evaluate(model, img0, metrics=metrics)
print(res)

img_size = 256
lines = make_lines(max=0.4, min=0.2, n_lines=10, shift=2, img_size=img_size)
img = get_torch_data(lines)
res = predict(model, img)
info(res[0])
info(res[1])
show_images(res)

res = evaluate(model, img, metrics=metrics)
print(res)

data = skimage.data.astronaut()
img = get_torch_data(data)
res = predict(model, img)
info(res[0])
info(res[1])
show_images(res)

res = evaluate(model, img, metrics=metrics)
print(res)

## Load data loaders
from trainer import set_seed
from dataset import get_loaders

set_seed(42)
data_dir = "/home/namu/myspace/NAMU/datasets/data_2024/"
batch_size = 8
img_size = 256
train_loader, valid_loader = get_loaders(data_dir, batch_size, img_size)

x, y = next(iter(valid_loader))
x.shape, y.shape

images, labels = x.to(device), y.to(device)
model.eval()
pred = model(images).detach()

show_images(images.detach().cpu())
show_images(pred.detach().cpu())

i = 2
res = {metric: metric_fn(pred[i].unsqueeze(0), images[i].unsqueeze(0)).item()
        for metric, metric_fn in metrics.items()}
res
