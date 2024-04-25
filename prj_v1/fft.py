import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift, dct, idct

from image import Image, ImageProcessor


class FFT2D(ImageProcessor):
    def __init__(self, img: Image):
        self.data = fft2(img.data)
    
    # ## amp()
    # def amplitude(self) -> ImageProcessor:
    #     return ImageProcessor(self.data).abs()
    
    # ## ang()
    # def phase(self) -> ImageProcessor:
    #     return ImageProcessor(np.angle(self.data))


class FFT2DShift(ImageProcessor):
    def __init__(self, img: Image):
        self.data = fftshift(fft2(img.data))


class InvFFT2D(ImageProcessor):
    def __init__(self, amp: Image, ang: Image=None):
        if ang is None:
            self.data = amp.data
        else:
            self.data = amp.data * np.exp(1j * ang.data)
        self.data = ifft2(self.data)

        
class InvFFT2DShift(ImageProcessor):
    def __init__(self, amp: Image, ang: Image=None):
        if ang is None:
            self.data = amp.data
        else:
            self.data = amp.data * np.exp(1j * ang.data)
        self.data = ifft2(ifftshift(self.data))


class DCT2D(ImageProcessor):
    def __init__(self, img: Image):
        self.data = dct(dct(img.data.T, type=2, norm='ortho').T, type=2, norm='ortho')


class InvDCT2D(ImageProcessor):
    def __init__(self, img: Image):
        self.data = idct(idct(img.data.T, type=2, norm='ortho').T, type=2, norm='ortho')
        

if __name__ == "__main__":
    
    from image import Image, ImageProcessor
    from viewer import Viewer2D

    data = skimage.data.camera()
    img = ImageProcessor(data).info()
    viewer2d = Viewer2D()
    
    if 0:
        viewer2d.show(img)
    
    if 1:
        img_fft = FFT2DShift(img)
        viewer2d.show(img_fft.amp().log1p())
        
        amp_fft = img_fft.amp()
        ang_fft = img_fft.ang()
        
        img_ifft = InvFFT2DShift(img_fft)
        viewer2d.show(img_ifft.abs())
        
        img_ifft = InvFFT2DShift(amp_fft, ang_fft)
        viewer2d.show(img_ifft.abs())
    
    if 0:
        img_dct = DCT2D(img)
        img_idct = InvDCT2D(img_dct)
        viewer2d.show(img_idct)
        
    