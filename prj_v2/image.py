from __future__ import annotations
import os

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift, dct, idct


## Data wrapper
class Data:
    def __init__(self, data: ndarray, name: str=""):
        self.data = data
        self.name = name

    def imread(self, path: str):
        self.data = Image(skimage.io.imread(path))
        self.name = os.path.basename(path)
        return self

    def title(self, name: str):
        self.name = name + self.name
        return self

    def info(self, option=0):
        print(f"*** [{self.name}]")
        if option > 0:
            print(f">> shape={self.data.shape}, dtype={self.data.dtype}", end=", ")
            print(f"min={self.data.min():.2f}, max={self.data.max():.2f}")
        return self


## New Data Object Generators
class DataGenerator(Data):
    def copy(self):
        img = Image(self.data, name=self.name)
        img.title("(copy)")
        return img

    def abs(self):
        img = Image(np.abs(self.data), name=self.name)
        img.title("(abs)")
        return img

    def real(self):
        img = Image(np.real(self.data), name=self.name)
        img.title("(real)")
        return img

    def log1p(self):
        img = Image(np.log1p(self.data), name=self.name)
        img.title("(log1p)")
        return img

    def log10(self):
        img = Image(np.log10(self.data), name=self.name)
        img.title("(log10)")
        return img

    def angle(self):
        img = Image(np.angle(self.data), name=self.name)
        img.dtype = "ang"
        img.title("(ang)")
        return img


## Inline Data Modifiers
class DataModifier(Data):
    def _gray(self):
        if self.data.ndim > 2:
            if self.data.shape[-1] > 3:
                self.data = skimage.color.rgba2rgb(self.data)
            self.data = skimage.color.rgb2gray(self.data)
        self.title("(gray)")
        return self

    def _resize(self, height, width):
        self.data = skimage.transform.resize(self.data, (height, width))
        self.title("(resized)")
        return self

    def _minmax(self, min, max):
        data_min_max = self.data.min(), self.data.max()
        self.data = np.interp(self.data, data_min_max, (min, max))
        self.title("(rescaled)")
        return self

    def _gaussian(self, sigma=1) -> Image:
        self.data = gaussian_filter(self.data, sigma=sigma)
        self.title("(gaussian)")
        return self

    def _uniform(self, size=1) -> Image:
        self.data = uniform_filter(self.data, size=size, mode="nearest")
        self.title("(uniform)")
        return self


## Main Objects
class Image(DataGenerator, DataModifier):
    def __init__(self, data: ndarray, name: str=""):
        super().__init__(data, name)
        self.dtype = "img"   # ("img", "amp", "ang")
        self.shifted = False

    def set_dtype(self, dtype: str):
        self.dtype == dtype
        return self



if __name__ == "__main__":

    from viewer import MultiViewer
    img1 = Image(skimage.data.astronaut()).title("astronaut").info(1)
    img2 = img1._gray().copy().info(1)
    img1.info(1)

    viewer = MultiViewer()
    viewer.show(img1)
    viewer.show(img2)

