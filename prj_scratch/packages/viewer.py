from __future__ import annotations
import matplotlib.pyplot as plt
import skimage.color
import skimage.util
from image import Image

## Main Image Viewer Class
class Viewer:
    def __init__(self):
        # self.xticks = Xticks()
        # self.yticks = Yticks()
        # self.xticklabels = XtickLabels()
        # self.yticklabels = YticksLabels()
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
            # axes[i].set_xticks(self.xticks(img[i]))
            # axes[i].set_yticks(self.yticks(img[i]))
            # axes[i].set_xticklabels(self.xticklabels(img[i]))
            # axes[i].set_yticklabels(self.yticklabels(img[i]))

            # if img[i].axis_off:
            #     axes[i].set_axis_off()

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":

    import skimage
    from functions import log1p, ifftshift, fft, fftshift
    
    viewer = Viewer()
    img = skimage.data.astronaut()
    print(img.min(), img.max())

    img = skimage.util.img_as_float(img)
    print(img.min(), img.max())
    
    img = skimage.color.rgb2gray(img)
    img = Image(img).set_title("raw")
    amp = img.fftshift.amplitude
    ang = img.fftshift.phase
    inv = ifftshift(amp, ang).set_title("inv")
    
    viewer.show(img, log1p(amp), ang, abs(inv))
    
