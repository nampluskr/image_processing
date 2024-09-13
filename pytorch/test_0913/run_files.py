import subprocess
from itertools import product
from time import time
from datetime import timedelta


def run(files):
    for i, filename in enumerate(files):
        start = time()

        print(f"\n>> File[{i+1}/{len(files)}] {filename}")
        command = ["python", filename]
        subprocess.run(command)

        result = str(timedelta(seconds=time() - start)).split(".")[0]
        print(f">> Time: {result}")


if __name__ == "__main__":

    files = [
        "eval_vgg16-1024.py",
    ]
    run(files)
