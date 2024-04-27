import matplotlib.pyplot as plt

class AxesParams:
    def xticks(self, img):
        N = img.data.shape[1]
        return [0, N//4, N//2, 3*N//4, N]

    def yticks(self, img):
        M = img.data.shape[0]
        return [0, M//4, M//2, 3*M//4, M]

    def xticklabels(self, img):
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

    def yticklabels(self, img):
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

class Viewer:
    def __init__(self):
        self.reset()
        self.axes = AxesParams()

    def reset(self):
        self.figsize = 6, 4
        self.cmap = "gray"

    def set_figsize(self, width, height):
        self.figsize = width, height
        return self

    def set_cmap(self, cmap):
        self.cmap = cmap
        return self

    def show(self, *img: tuple):
        ncols = len(img)
        self.figsize = (3*ncols, 3)

        if ncols == 1:
            fig, ax = plt.subplots(figsize=self.figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, len(img), figsize=self.figsize)

        for i in range(len(img)):
            axes[i].imshow(img[i].data, cmap=self.cmap)
            axes[i].set_title(img[i].title)

            axes[i].set_xticks(self.axes.xticks(img[i]))
            axes[i].set_yticks(self.axes.yticks(img[i]))
            axes[i].set_xticklabels(self.axes.xticklabels(img[i]))
            axes[i].set_yticklabels(self.axes.yticklabels(img[i]))

            if img[i].axis_off:
                axes[i].set_axis_off()

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    import skimage
    from image import Image

    viewer = Viewer()
    img1 = Image(skimage.data.astronaut()).set_title("Camera-1")
    img2 = img1.copy().set_title("Camera-2")
    img3 = img1.copy().set_title("Camera-3")

    img1.set_dtype("img").set_shifted(False)
    img2.set_dtype("amp").set_shifted(True)
    img3.set_dtype("ang").set_shifted(True)

    viewer.show(img1, img2, img3)
