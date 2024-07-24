import numpy as np
# from scipy.fft import fft2, fftshift, ifft2, ifftshift, dct, idct
from scipy import fft as fft2d
from image import Image, Uniform, Gaussian, Resize, Rescale


## FFT Functions for Images
class FFT2D(Image):
    def __init__(self, img: Image):
        self.data = fft2d.fft2(img.data)
        self.set_default(img)
        self.shifted = False
        self.axis_off = False


class FFT2DShift(Image):
    def __init__(self, img: Image):
        self.data = fft2d.fftshift(fft2d.fft2(img.data))
        self.set_default(img)
        self.shifted = True
        self.axis_off = False


class InvFFT2D(Image):
    def __init__(self, amp: Image, ang: Image=None):
        data = amp.data if ang is None else amp.data * np.exp(1j * ang.data)
        self.data = fft2d.ifft2(data)
        self.title = "Inverse FFT"
        self.dtype = "img"
        self.shifted = False
        self.axis_off = False


class InvFFT2DShift(Image):
    def __init__(self, amp: Image, ang: Image=None):
        data = amp.data if ang is None else amp.data * np.exp(1j * ang.data)
        self.data = fft2d.ifft2(fft2d.ifftshift(data))
        self.title = "Inverse FFT"
        self.dtype = "img"
        self.shifted = False
        self.axis_off = False


## DCT Functions for Images
class DCT2D(Image):
    def __init__(self, img: Image):
        self.data = fft2d.dct(fft2d.dct(img.data.T, type=2, norm='ortho').T, type=2, norm='ortho')
        self.set_default(img)
        self.axis_off = False


class InvDCT2D(Image):
    def __init__(self, img: Image):
        self.data = fft2d.idct(fft2d.idct(img.data.T, type=2, norm='ortho').T, type=2, norm='ortho')
        self.set_default(img)
        self.title = "Inverse DCT"
        self.axis_off = False


## Math Functions for FFT of Images
class Amplitude(Image):
    def __init__(self, img: Image):
        self.data = np.abs(img.data)
        self.set_default(img)
        self.title = "Amplitude(FFT)"
        self.dtype = "amp"


class Phase(Image):
    def __init__(self, img: Image):
        self.data = np.angle(img.data)
        self.set_default(img)
        self.title = "Phase(FFT)"
        self.dtype = "ang"


## Math Functions for Images
class Abs(Image):
    def __init__(self, img: Image):
        self.data = np.abs(img.data)
        self.set_default(img)


class Log1p(Image):
    def __init__(self, img: Image):
        self.data = np.log1p(img.data)
        self.set_default(img)


class Log10(Image):
    def __init__(self, img: Image):
        self.data = np.log10(img.data)
        self.set_default(img)


class Real(Image):
    def __init__(self, img: Image):
        self.data = np.real(img.data)
        self.set_default(img)

class Conjugate(Image):
    def __init__(self, img: Image):
        self.data = np.conj(img.data)
        self.set_default(img)

class Pow(Image):
    def __init__(self, img: Image, n: int):
        self.data = img.data**n
        self.set_default(img)


## Filters
class Filter(Image):
    def __init__(self, filter: np.ndarray, shape: tuple):
        self.data = Abs(FFT2DShift(Image(filter)))
        self.data = Resize(self.data, shape).data
        self.set_default()
        self.shifted = True

class Mask(Image):
    def __init__(self, shape: tuple, base: float):
        self.data = np.ones(shape, np.uint8) * base
        self.set_default()
        self.title = "mask"
        self.dtype = "mask"
        self.shifted = True
        self.axis_off = False

    def set_center(self, drow, dcol, value):
        rows, cols = self.shape[:2]
        row0, col0 = rows // 2, cols // 2
        self.data[-row0-drow:row0+drow, -col0-dcol:col0+dcol] = value
        return self



## Analysis Tools for 1 image
class Mura(Image):
    def __init__(self, img: Image, threshold: float):
        dct = DCT2D(img)
        dct.data[abs(dct).data < threshold] = 0
        idct = InvDCT2D(dct)
        self.data = Abs(img - idct).data
        self.set_default(img)
        self.title = "Mura"

class MuraRemoved(Image):
    def __init__(self, img: Image, threshold: float):
        dct = DCT2D(img)
        dct.data[abs(dct).data < threshold] = 0
        self.data = InvDCT2D(dct).data
        self.set_default(img)
        self.title = "Mura Removed"

class SaliencyMap(Image):
    def __init__(self, img: Image, size: int=3, sigma: float=2):
        fft = FFT2D(img)
        log_amp = Log1p(Amplitude(fft))
        ang = Phase(fft)

        amp_avg = Uniform(log_amp, size=size)
        amp_res = log_amp - amp_avg
        map = abs(InvFFT2D(amp_res, ang)**2)
        self.data = Gaussian(map, sigma=sigma).data
        self.set_default()
        self.title = "Saliency Map"

class Cepstrum(Image):
    def __init__(self, img: Image):
        fft = FFT2DShift(img)
        amp = Amplitude(fft)
        fft_log_amp = FFT2D(Log1p(amp))
        self.data = Abs(InvFFT2D(fft_log_amp)).data
        self.set_default(img)
        self.title = "Cepstrum"
        self.shifted = True


## Analysis Tools for 2 images
class PhaseCorrelation(Image):
    def __init__(self, img1: Image, img2: Image):
        g1 = FFT2D(img1)
        g2 = FFT2D(img2)
        r = g1 * Conjugate(g2)
        r /= Abs(r)
        self.data = Abs(InvFFT2D(r)).data
        self.set_default()
        self.title = "Phase Only Correlation"


class PhaseDiscrepancy(Image):
    def __init__(self, img1: Image, img2: Image):
        fft1 = FFT2D(img1)
        fft2 = FFT2D(img2)

        amp1, ang1 = Amplitude(fft1), Phase(fft1)
        amp2, ang2 = Amplitude(fft2), Phase(fft2)

        inv1 = abs(InvFFT2D(amp1 - amp2, ang1))
        inv2 = abs(InvFFT2D(amp2 - amp1, ang2))

        self.data = Rescale(inv1 * inv2, (0, 255)).data
        self.set_default()
        self.title = "Phase Discrepancy"


if __name__ == "__main__":

    import skimage
    from image import Image
    from viewer import Viewer

    viewer = Viewer()
    img = Image(skimage.data.camera()).set_title("Original").info()

    if 1:
        fft1 = FFT2D(img)
        amp1 = Amplitude(fft1).info()
        ang1 = Phase(fft1).info()
        inv1 = Abs(InvFFT2D(amp1, ang1)).info()

        viewer.show(img, Log1p(amp1), ang1, inv1)

    if 1:
        fft2 = FFT2DShift(img)
        amp2 = Amplitude(fft2).info()
        ang2 = Phase(fft2).info()
        inv2 = Abs(InvFFT2DShift(amp2, ang2)).info()

        viewer.show(img, Log1p(amp2), ang2, inv2)
