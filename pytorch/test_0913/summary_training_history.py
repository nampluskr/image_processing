import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_df(path):
    df = pd.read_csv(path, index_col=0)
    basename = os.path.splitext(os.path.basename(path))[0]
    values = basename.split('_')
    df.insert(0, "name", basename)
    df["model"] = values[0]
    df["latent"] = int(values[1].split('-')[-1])
    df["size"] = int(values[2].split('-')[-1])
    # df["batch"] = int(values[3].split('-')[-1])
    return df

def get_ax(ax, hist, name):
    epochs = range(1, len(hist) + 1)
    ax.plot(epochs, hist[name], 'k', label=name)
    ax.plot(epochs, hist["val_" + name], 'r', label="val_" + name)
    ax.legend()
    ax.grid()
    return ax

paths = glob("/home/namu/myspace/NAMU/office/test_0906/*/*.csv")    # convnet
paths += glob("/home/namu/myspace/NAMU/office/test_0907/*/*.csv")   # pretrained
paths += glob("/home/namu/myspace/NAMU/office/test_0910/*/*.csv")   # unet

df = pd.DataFrame()
for path in paths:
    df = pd.concat([df, get_df(path)], ignore_index=True)

def show_hist(df, model, latent):
    hist = df[(df["model"] == model) & (df["latent"] == latent)]

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols=6, figsize=(18, 2.5))
    ax1 = get_ax(ax1, hist, "loss");    ax1.set_title("Loss");      ax1.set_ylim(0.1, 0.5)
    ax2 = get_ax(ax2, hist, "mse");     ax2.set_title("MSE");       ax2.set_ylim(0.0, 0.1)
    ax3 = get_ax(ax3, hist, "bce");     ax3.set_title("BCE");       ax3.set_ylim(0.3, 0.6)

    ax4 = get_ax(ax4, hist, "acc");     ax4.set_title("Accuracy");  ax4.set_ylim(0.5, 1)
    ax5 = get_ax(ax5, hist, "ssim");    ax5.set_title("SSIM");      ax5.set_ylim(0.5, 1)
    ax6 = get_ax(ax6, hist, "psnr");    ax6.set_title("PSNR");      ax6.set_ylim(10, 30)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 100)

    fig.tight_layout()
    plt.show()

show_hist(df, model="convnet", latent=1024)
show_hist(df, model="vgg16", latent=1024)
show_hist(df, model="resnet50", latent=1024)
show_hist(df, model="efficientb7", latent=1024)
show_hist(df, model="unet", latent=1024)

