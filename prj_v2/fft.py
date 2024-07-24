import numpy as np
import skimage
from scipy.fft import fft2, fftshift, ifft2, ifftshift, dct, idct

from image import Image


class FFT2D(Image):
    def __init__(self, img: Image):
        self.data = fft2(img.data)
        self.name = img.name
        self.title("(fft)")
        self.dtype = "img"
        self.is_shifted = False


class FFT2DShift(Image):
    def __init__(self, img: Image):
        self.data = fftshift(fft2(img.data))
        self.name = img.name
        self.title("(fftshift)")
        self.dtype = "img"
        self.is_shifted = True


class InvFFT2D(Image):
    def __init__(self, amp: Image, ang: Image=None):
        if ang is None:
            self.data = amp.data
            self.name = amp.name
        else:
            self.data = amp.data * np.exp(1j * ang.data)
            self.name = amp.name[5:]
        self.data = ifft2(self.data)
        self.title("(ifft)")
        self.dtype = "img"
        self.is_shifted = False


class InvFFT2DShift(Image):
    def __init__(self, amp: Image, ang: Image=None):
        if ang is None:
            self.data = amp.data
            self.name = amp.name
        else:
            self.data = amp.data * np.exp(1j * ang.data)
            self.name = amp.name[5:]
        self.data = ifft2(ifftshift(self.data))
        self.title("(ifftshift)")
        self.dtype = "img"
        self.is_shifted = False


class DCT2D(Image):
    def __init__(self, img: Image):
        self.data = dct(dct(img.data.T, type=2, norm='ortho').T, type=2, norm='ortho')
        self.name = img.name
        self.title("(dct)")
        self.dtype = "img"
        self.is_shifted = False


class InvDCT2D(Image):
    def __init__(self, img: Image):
        self.data = idct(idct(img.data.T, type=2, norm='ortho').T, type=2, norm='ortho')
        self.name = img.name
        self.title("(idct)")
        self.dtype = "img"
        self.is_shifted = False



if __name__ == "__main__":

    from viewer import MultiViewer
    viewer = MultiViewer()
    img = Image(skimage.data.astronaut()).title("RAW").info()
    img._gray()

    viewer.show(img)

    if 0:
        img_fft = FFT2D(img)
        img_inv = InvFFT2D(img_fft)._abs()
        viewer.show(img_inv)

    if 0:
        img_fft = FFT2DShift(img)
        img_amp = img_fft.abs()
        img_ang = img_fft.angle()
        img_inv = InvFFT2DShift(img_amp, img_ang)._abs()
        viewer.show(img_inv)
        
    if 1:
        img_fft = FFT2DShift(img)
        img_amp = img_fft.abs()
        img_ang = img_fft.angle()
        img_inv = InvFFT2DShift(img_amp, img_ang)._abs()

        viewer.show(img_amp.log1p())
        viewer.show(img_ang)
        viewer.show(img_inv)
