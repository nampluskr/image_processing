from abc import ABC, abstractmethod
from image import Image


## Interface for Axes Settings
class BaseAxesParams(ABC):
    def __init__(self, img: Image):
        self.shape = img.data.shape[:2]
        self.dtype = img.dtype
        self.is_shifted = img.shifted

    @abstractmethod
    def params(self):
        pass

## Concrete Axes Settings
class AxesExtent(BaseAxesParams):
    def params(self):
        M, N = self.shape
        if self.is_shifted:
            return [-N//2, N//2, -M//2, M//2]
        else:
            return [0, N, M, 0]


class AxesXticks(BaseAxesParams):
    def params(self):
        M, N = self.shape
        if self.is_shifted:
            return [-N//2, -N//4, 0, N//4, N//2]
        else:
            return [0, N//4, N//2, 3*N//4, N]

class AxesYticks(BaseAxesParams):
    def params(self):
        M, N = self.shape
        if self.is_shifted:
            return [-M//2, -M//4, 0, M//4, M//2]
        else:
            return [M, 3*M//4, M//2, M//4, 0]

class AxesXtickLabels(BaseAxesParams):
    def params(self):
        M, N = self.shape
        if self.is_shifted:
            if self.dtype == "amp":
                return [-N//2, -N//4, 0, N//4, N//2]
            elif self.dtype == "ang":
                return ["-$\pi$", "-$\pi/2$", "0", "$\pi/2$", "$\pi$"]
            else:
                return [0, N//4, N//2, 3*N//4, N]
        else:
            if self.dtype == "ang":
                return [0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
            else:
                return [0, N//4, N//2, 3*N//4, N]

class AxesYtickLabels(BaseAxesParams):
    def params(self):
        M, N = self.shape
        if self.is_shifted:
            if self.dtype == "amp":
                return [-M//2, -M//4, 0, M//4, M//2]
            elif self.dtype == "ang":
                return ["-$\pi$", "-$\pi/2$", "0", "$\pi/2$", "$\pi$"]
            else:
                return [M, 3*M//4, M//2, M//4, 0]
        else:
            if self.dtype == "ang":
                return [0, "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
            else:
                return [M, 3*M//4, M//2, M//4, 0]


## Client Class
class AxesParams:
    def __init__(self, img: Image):
        self.axes = {}
        self.axes["extent"] = AxesExtent(img)
        self.axes["xticks"] = AxesXticks(img)
        self.axes["yticks"] = AxesYticks(img)
        self.axes["xticklabels"] = AxesXtickLabels(img)
        self.axes["yticklabels"] = AxesYtickLabels(img)

    def extent(self):
        return self.axes["extent"].params()

    def xticks(self):
        return self.axes["xticks"].params()

    def yticks(self):
        return self.axes["yticks"].params()

    def xticklabels(self):
        return self.axes["xticklabels"].params()

    def yticklabels(self):
        return self.axes["yticklabels"].params()


if __name__ == "__main__":

    import skimage
    from viewer import MultiViewer

    viewer = MultiViewer()
    img1 = Image(skimage.data.astronaut()).title("RAW")
    img1 = img1._gray()._resize(300, 500)
    img1.name = "img1"

    img2 = img1.copy()
    img2.dtype = "amp"
    img2.is_shifted = True
    img2.name = "img2"

    print(f"img1: {img1.dtype}, shifted={img1.is_shifted}")
    print(f"img2: {img2.dtype}, shifted={img2.is_shifted}")
    
    viewer.show(img1, img2)