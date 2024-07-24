from __future__ import annotations
import matplotlib.pyplot as plt
import skimage.color
import skimage.util


## Main Image Viewer Class
class Viewer:
    def __init__(self):
        self.xticks = Xticks()
        self.yticks = Yticks()
        self.xticklabels = XtickLabels()
        self.yticklabels = YticksLabels()
        self.default()

    def default(self):
        self._cmap = "gray"
        self._unit = 3
        self._vmin = None
        self._vmax = None
        return self
    
    def vmin(self, _vmin):
        self._vmin = _vmin
        return self
    
    def vmax(self, _vmax):
        self._vmax = _vmax
        return self

    def unit(self, _unit):
        self._unit = _unit
        return self

    def cmap(self, _cmap):
        self._cmap = _cmap
        return self

    def show(self, *img: tuple):
        ncols = len(img)
        figsize = (self._unit*len(img), self._unit)

        if ncols == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, ncols, figsize=figsize)

        for i in range(ncols):
            axes[i].imshow(img[i].data, cmap=self._cmap, vmin=self._vmin, vmax=self._vmax)
            axes[i].set_xticks(self.xticks(img[i]))
            axes[i].set_yticks(self.yticks(img[i]))
            axes[i].set_xticklabels(self.xticklabels(img[i]))
            axes[i].set_yticklabels(self.yticklabels(img[i]))

            if img[i]._axis_off:
                axes[i].set_axis_off()
            else:
                axes[i].set_title(img[i].name)

        fig.tight_layout()
        plt.show()


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
        if "fftshift" in img.name:
            if "phase" in img.name:
                return ["-$\pi$", "-$\pi/2$", "0", "$\pi/2$", "$\pi$"]
            else:
                return [-N//2, -N//4, 0, N//4, N//2]
        else:
            if "phase" in img.name:
                return [0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
            else:
                return [0, N//4, N//2, 3*N//4, N]


class YticksLabels:
    def __call__(self, img):
        M = img.data.shape[0]
        if "fftshift" in img.name:
            if "phase" in img.name:
                return ["-$\pi$", "-$\pi/2$", "0", "$\pi/2$", "$\pi$"]
            else:
                return [-M//2, -M//4, 0, M//4, M//2]
        else:
            if "phase" in img.name:
                return [0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
            else:
                return [0, M//4, M//2, 3*M//4, M]


def show_lines(*pairs, labels=None, h=None, v=None, unit=3):
    ''' pairs = (a1, a2), (b1, b2),          ...: (Image, Image)
        labels = ("a1", "a2"), ("b1", "b2"), ...: (str, str)
    '''
    ncols = len(pairs)
    assert ncols <= 6

    figsize = (1.5*unit*ncols, unit)
    if ncols == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    colors = ('g', 'r', 'b', 'k', 'm', 'c')
    for i in range(ncols):
        if h is None and v is not None:
            for j in range(len(pairs[i])):
                axes[i].plot(pairs[i][j].data[:, v], colors[j], lw=1.5,
                             label=None if labels is None else labels[i][j])
        if h is not None and v is None:
            for j in range(len(pairs[i])):
                axes[i].plot(pairs[i][j].data[h, :], colors[j], lw=1.5,
                             label=None if labels is None else labels[i][j])

        if labels is not None:
            axes[i].legend(loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    import skimage
    from image import Image
    from fft import log1p, ifftshift, fft, fftshift, ifft

    viewer = Viewer()
    img = skimage.data.astronaut()
    print(img.min(), img.max())

    img = skimage.util.img_as_float(img)
    print(img.min(), img.max())

    img = skimage.color.rgb2gray(img)
    img = Image(img).set_name("raw").info()
    amp = img.fftshift.amplitude
    ang = img.fftshift.phase
    inv = ifftshift(amp, ang).set_name("inv")

    viewer.show(img, log1p(amp), ang, abs(inv))
