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

pd.options.display.float_format = "{:.3f}".format

paths = glob("/home/namu/myspace/NAMU/office/test_0906/*/*.csv")

res_convnet = pd.DataFrame()
for path in paths:
    res_convnet = pd.concat([res_convnet, get_df(path).iloc[-1:, :]], ignore_index=True)

res_convnet.sort_values("val_loss")

paths = glob("/home/namu/myspace/NAMU/office/test_0907/*/*.csv")

res_pretrained = pd.DataFrame()
for path in paths:
    res_pretrained = pd.concat([res_pretrained, get_df(path).iloc[-1:, :]], ignore_index=True)

res_pretrained.sort_values("val_loss")

paths = glob("/home/namu/myspace/NAMU/office/test_0910/*/*.csv")

res_unet = pd.DataFrame()
for path in paths:
    res_unet = pd.concat([res_unet, get_df(path).iloc[-1:, :]], ignore_index=True)

res_unet.sort_values("val_loss")

res = pd.concat([res_convnet, res_pretrained, res_unet], ignore_index=True)
res.sort_values("model")

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 3))
sns.barplot(res[res["latent"] > 256 ], x="latent", y="bce", hue="model", ax=ax1)
sns.barplot(res[res["latent"] > 256 ], x="latent", y="ssim", hue="model", ax=ax2)
sns.barplot(res[res["latent"] > 256 ], x="latent", y="psnr", hue="model", ax=ax3)

for ax in (ax1, ax2, ax3):
    ax.set_xlabel("Latent Dimension", fontsize=12)
    # ax.set_ylabel("SSIM", fontsize=12)
    ax.legend(loc="upper left")

fig.tight_layout()
plt.show()

