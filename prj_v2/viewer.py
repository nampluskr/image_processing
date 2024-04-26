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
        self.is_logscale = False

        # self.extent = (0, N, 0, M)
        # self.xticks = (0, N//4, N//2, 3*N//4, N)
        # self.yticks = (0, M//4, M//2, 3*M//4, M)
        # self.tickslabels = (0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$")

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
        fig, (ax, ) = plt.subplots(figsize=self.size)
        ax.imshow(img.data, cmap=self.cm)
        ax.set_title(img.name)
        if self.is_axis_off:
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


class Viewer3D(Viewer):
    def show(self, img: Image):
        X = np.linspace(0, img.data.shape[1] - 1, img.data.shape[1])
        Y = np.linspace(0, img.data.shape[0] - 1, img.data.shape[0])
        X, Y = np.meshgrid(X, Y)

        fig, (ax, ) = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(Y, X, img.data, cmap=self.cm)
        ax.set_title(img.name)
        ax.set_xlabel("height")
        ax.set_ylabel("width")
        ax.view_init(elev=45, azim=-20)
        if self.is_axis_off:
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


class Viewer2Plots(Viewer):
    def show(self, img1: Image, img2: Image):
        self.figsize(6, 3)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.size)
        ax1.imshow(img1.data, cmap=self.cm)
        ax1.set_title(img1.name)

        ax2.imshow(img2.data, cmap=self.cm)
        ax2.set_title(img2.name)

        if self.is_axis_off:
            ax1.set_axis_off()
            ax2.set_axis_off()
        fig.tight_layout()
        plt.show()


class Viewer3Plots(Viewer):
    def show(self, img1: Image, img2: Image, img3: Image):
        self.figsize(9, 3)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.size)
        ax1.imshow(img1.data, cmap=self.cm)
        ax1.set_title(img1.name)

        ax2.imshow(img2.data, cmap=self.cm)
        ax2.set_title(img2.name)

        ax3.imshow(img3.data, cmap=self.cm)
        ax3.set_title(img3.name)

        if self.is_axis_off:
            ax1.set_axis_off()
            ax2.set_axis_off()
            ax3.set_axis_off()
        fig.tight_layout()
        plt.show()


class TestViewer(Viewer):
    def show(self, *img: tuple):
        ncols = len(img)
        self.figsize(3*ncols, 3)

        if ncols == 1:
            fig, ax = plt.subplots(figsize=self.size)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, len(img), figsize=self.size)

        for i in range(len(img)):
            axes[i].imshow(img[i].data, cmap=self.cm)
            axes[i].set_title(img[i].name)

            if self.is_axis_off:
                axes[i].set_axis_off()

        fig.tight_layout()
        plt.show()



if __name__ == "__main__":

    import skimage
    from fft import FFT2DShift

    img = Image(skimage.data.astronaut()).title("RAW")._gray().info()

    if 0:
        viewer2d = Viewer2D().cmap("gray").figsize(4, 4)
        viewer3d = Viewer3D().cmap("jet").axis_off()

        viewer2d.show(img)
        viewer3d.show(img)

        viewer3d.reset()
        viewer3d.show(img)

    if 0:
        fft = FFT2DShift(img)
        amp = fft.amplitude()
        ang = fft.phase()

        viewer2plots = Viewer2Plots()
        viewer2plots.show(img, amp.log1p())

        viewer3plots = Viewer3Plots()
        viewer3plots.show(img, amp.log1p(), ang)

    if 1:
        viewer = TestViewer()
        viewer.show(img)
        viewer.show(img, img)
        viewer.show(img, img, img)

