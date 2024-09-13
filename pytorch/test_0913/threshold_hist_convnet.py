## threshold_hist_convnet.ipynb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

split_name = "convnet"
latent_dim = 1024   # 512, 1024, 2048

result_dir = "/home/namu/myspace/NAMU/office/test_0911_2/results/"
model_name = f"{split_name}_latent-{latent_dim}_size-256_batch-8"

test_result = pd.read_csv(result_dir + f"test_{model_name}.csv")
valid_result = pd.read_csv(result_dir + f"valid_{model_name}.csv")
train_result = pd.read_csv(result_dir + f"train_{model_name}.csv")

len(test_result), len(valid_result), len(train_result)

def plot_hist(col_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    kwargs = {'kde': True, 'bins': 30, 'stat': 'percent'}
    sns.histplot(data=train_result[col_name], color='blue', ax=ax, label="Train", **kwargs)
    sns.histplot(data=valid_result[col_name], color='green', ax=ax, label="Valid", **kwargs)
    sns.histplot(data=test_result[col_name], color='red', ax=ax, label="Test", **kwargs)
    ax.set_xlabel(col_name.upper(), fontsize=12)
    ax.set_ylabel("Percent (%)", fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    plt.show()

plot_hist("loss")
plot_hist("mse")
plot_hist("l1")
plot_hist("huber")
plot_hist("bce")

plot_hist("ssim")
plot_hist("psnr")
plot_hist("acc")
