from __future__ import annotations
import matplotlib.pyplot as plt


## Main Image Viewer Class
class Viewer:
    def __init__(self):
        self.xticks = Xticks()
        self.yticks = Yticks()
        self.xticklabels = XtickLabels()
        self.yticklabels = YticksLabels()
        self.set_default()

    def set_default(self):
        self.cmap = "gray"
        self.unit = 3
        return self

    def set_unit(self, unit):
        self.unit = unit
        return self

    def set_cmap(self, cmap):
        self.cmap = cmap
        return self

    def show(self, *img: tuple):
        ncols = len(img)
        figsize = (self.unit*len(img), self.unit)

        if ncols == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, ncols, figsize=figsize)

        for i in range(ncols):
            axes[i].imshow(img[i].data, cmap=self.cmap)
            axes[i].set_title(img[i].title)
            axes[i].set_xticks(self.xticks(img[i]))
            axes[i].set_yticks(self.yticks(img[i]))
            axes[i].set_xticklabels(self.xticklabels(img[i]))
            axes[i].set_yticklabels(self.yticklabels(img[i]))

            if img[i].axis_off:
                axes[i].set_axis_off()

        fig.tight_layout()
        plt.show()


## Axes Parameters for Viewer
class Xticks:
    def __call__(self, img):
        N = img.data.shape[1]
        return [0, N//4, N//2, 3*N//4, N]


class Yticks:
    def __call__(self, img):
        M = img.data.shape[0]
        return [0, M//4, M//2, 3*M//4, M]


class XtickLabels:
    def __call__(self, img):
        N = img.data.shape[1]
        if img.shifted:
            if img.dtype == "ang":
                return ["-$\pi$", "-$\pi/2$", "0", "$\pi/2$", "$\pi$"]
            else:
                return [-N//2, -N//4, 0, N//4, N//2]
        else:
            if img.dtype == "ang":
                return [0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
            else:
                return [0, N//4, N//2, 3*N//4, N]


class YticksLabels:
    def __call__(self, img):
        M = img.data.shape[0]
        if img.shifted:
            if img.dtype == "ang":
                return ["-$\pi$", "-$\pi/2$", "0", "$\pi/2$", "$\pi$"]
            else:
                return [-M//2, -M//4, 0, M//4, M//2]
        else:
            if img.dtype == "ang":
                return [0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
            else:
                return [0, M//4, M//2, 3*M//4, M]


if __name__ == "__main__":

    import skimage
    from image import Image, Gray, Resize

    viewer = Viewer()
    img1 = Image(skimage.data.astronaut()).set_title("Camera-1")
    img2 = Gray(img1).set_title("Camera-2").add_title("(gray)")
    img3 = Resize(img2, (300, 200)).add_title("(resized)")

    img1.set_dtype("img").set_shifted(False)
    img2.set_dtype("amp").set_shifted(True)
    img3.set_dtype("ang").set_shifted(True)

    viewer.show(img1, img2, img3)
