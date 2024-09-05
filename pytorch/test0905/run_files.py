import subprocess
from itertools import product
from time import time
from datetime import timedelta
import math

def run(files):
    for i, filename in enumerate(files):
        start = time()

        print(f"\n>> File[{i+1}/{len(files)}] {filename}")
        command = ["python", filename]
        subprocess.run(command)

        math.factorial(500000)
        result = str(timedelta(seconds=time() - start)).split(".")[0]
        print(f">> Time: {result}")


if __name__ == "__main__":

    files = [
        "train_convnet_latent-512.py",
        "train_vgg16_latent-512.py",
        "train_resnet50_latent-512.py",

        "train_convnet_latent-1024.py",
        "train_vgg16_latent-1024.py",
        "train_resnet50_latent-1024.py",

        "train_convnet_latent-2048.py",
        "train_vgg16_latent-2048.py",
        "train_resnet50_latent-2048.py",

        "train_vgg19_latent-512.py",
        "train_vgg19_latent-1024.py",
        "train_vgg19_latent-2048.py",

        "train_resnet34_latent-512.py",
        "train_resnet34_latent-1024.py",
        "train_resnet34_latent-2048.py",

        "train_efficientb0_latent-512.py",
        "train_efficientb0_latent-1024.py",
        "train_efficientb0_latent-2048.py",

        "train_efficientb3_latent-512.py",
        "train_efficientb3_latent-1024.py",
        "train_efficientb3_latent-2048.py",

        "train_efficientb7_latent-512.py",
        "train_efficientb7_latent-1024.py",
        "train_efficientb7_latent-2048.py",
    ]
    run(files)
