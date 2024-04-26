from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from image import Image

## Abstract Class
class Viewer(ABC):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = "gray"
        self.size = (6, 4)
        self.is_axis_off = False

    def cmap(self, cm: str) -> Viewer:
        self.cm = cm
        return self

    def figsize(self, width, height) -> Viewer:
        self.size = width, height
        return self

    def axis_off(self) -> Viewer:
        self.is_axis_off = True
        return self

    @abstractmethod
    def show() -> None:
        pass


class Viewer3D(Viewer):
    def show(self, img: Image):
        X = np.linspace(0, img.data.shape[1] - 1, img.data.shape[1])
        Y = np.linspace(0, img.data.shape[0] - 1, img.data.shape[0])
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(Y, X, img.data, cmap=self.cm)
        ax.set_title(img.name)
        ax.set_xlabel("height")
        ax.set_ylabel("width")
        ax.view_init(elev=45, azim=-20)
        if self.is_axis_off:
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


class MultiViewer(Viewer):
    def show(self, *img: tuple):
        ncols = len(img)
        self.figsize(3*ncols, 3)

        if ncols == 1:
            fig, ax = plt.subplots(figsize=self.size)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, len(img), figsize=self.size)

        for i in range(len(img)):
            # axes[i].imshow(img[i].data, cmap=self.cm)
            axes[i].imshow(img[i].data, cmap=self.cm, extent=None)
            axes[i].set_title(img[i].name)
            # if img[i].dtype == "img":
            #     axes[i].set_xticks([])
            #     axes[i].set_yticks([])

            if self.is_axis_off:
                axes[i].set_axis_off()

        fig.tight_layout()
        plt.show()


class PlotOptions:
    def __init__(self, img: Image):
        self.M, self.N = img.shape[:2]
        self.dtype = img.dtype
        self.shifted = img.shifted
        
    def extent(self):
        pass



if __name__ == "__main__":

    import skimage
    from fft import FFT2DShift, InvFFT2DShift

    img = Image(skimage.data.astronaut()).title("RAW").info()
    img = img._gray()._resize(500, 300).info()
    viewer = MultiViewer()

    if 1:
        fft = FFT2DShift(img)
        amp = fft.abs()
        ang = fft.angle()
        ifft = InvFFT2DShift(amp, ang).abs()

        viewer.show(img)
        viewer.show(img, amp.log1p())
        viewer.show(img, amp.log1p(), ang)
        viewer.show(img, amp.log1p(), ang, ifft)


