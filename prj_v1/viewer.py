from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

## Abstract Class
class Viewer(ABC):
    def __init__(self):
        self.cm = "gray"
        self.size = (6, 4)
        self.is_axis_off = False
        self.is_logscale = False

    def cmap(self, cm: str) -> Viewer:
        self.cm = cm
        return self

    def figsize(self, width, height) -> Viewer:
        self.size = width, height
        return self

    def axis_off(self) -> Viewer:
        self.is_axis_off = True
        return self

    def axis_on(self) -> Viewer:
        self.is_axis_off = False
        return self

    def log(self) -> Viewer:
        self.is_logscale = True
        return self

    def linear(self) -> Viewer:
        self.is_logscale = False
        return self
    
    @abstractmethod
    def show() -> None:
        pass


class Viewer2D(Viewer):
    def show(self, img: Image):
        data = np.log1p(img.data) if self.is_logscale else img.data
        fig, ax = plt.subplots(figsize=self.size)
        ax.imshow(data, cmap=self.cm)
        if self.is_axis_off:
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


class Viewer3D(Viewer):
    def show(self, img: Image):
        X = np.linspace(0, img.data.shape[1] - 1, img.data.shape[1])
        Y = np.linspace(0, img.data.shape[0] - 1, img.data.shape[0])
        X, Y = np.meshgrid(X, Y)

        data = np.log1p(img.data) if self.is_logscale else img.data
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(Y, X, img.data, cmap=self.cm)
        ax.set_xlabel("height")
        ax.set_ylabel("width")
        ax.view_init(elev=45, azim=-20)
        if self.is_axis_off:
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    from image import Image, ImageProcessor
    import skimage
    
    img = ImageProcessor(skimage.data.astronaut()).gray().info()
    viewer2d = Viewer2D().cmap("gray").figsize(4, 4)
    viewer3d = Viewer3D().cmap("jet").axis_off()

    viewer2d.show(img)
    viewer3d.show(img)