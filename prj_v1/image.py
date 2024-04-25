from __future__ import annotations
from abc import ABC, abstractmethod
import os

import numpy as np
from numpy import ndarray
import skimage
from scipy.ndimage import uniform_filter, gaussian_filter


## Main Object
class Image:
    def __init__(self, data: ndarray=None):
        self.data = data
        self.path = None
        self.name = None

    def imread(self, path: str) -> Image:
        self.data = Image(skimage.io.imread(path))
        self.path = path
        self.name = os.path.basename(path)
        return self
    
    def info(self) -> Image:
        print(f">> shape={self.data.shape}, dtype={self.data.dtype}", end=", ")
        print(f"min={self.data.min():.2f}, max={self.data.max():.2f}")
        return self


## Image Transformer / Filters
class ImageProcessor(Image):
    def gray(self) -> Image:
        assert self.data is not None
        if self.data.ndim > 2:
            if self.data.shape[-1] > 3:
                self.data = skimage.color.rgba2rgb(self.data)
            self.data = skimage.color.rgb2gray(self.data)
        return self
    
    def resize(self, width, height) -> Image:
        assert self.data is not None
        self.data = skimage.transform.resize(self.data, (width, height))
        return self

    def minmax(self, min, max) -> Image:
        assert self.data is not None
        data_min_max = self.data.min(), self.data.max()
        self.data = np.interp(self.data, data_min_max, (min, max))
        return self
    
    def gaussian(self, sigma=1) -> Image:
        self.data = gaussian_filter(self.data, sigma=sigma)
        return self
    
    def uniform(self, size=1) -> Image:
        self.data = uniform_filter(self.data, size=size, mode="nearest")
        return self


if __name__ == "__main__":
    
    img = ImageProcessor(skimage.data.astronaut()).gray().info()
    
    # img = img.gray().info()
    img = img.resize(200, 300).info()
    img = img.minmax(0, 2).info()
    img = img.gaussian().info()
    img = img.minmax(0, 1).info()
