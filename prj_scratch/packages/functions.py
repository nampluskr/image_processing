import numpy as np
from scipy import fft as fft2d
from image import Image, Function


#####################################################################
## Math fucntions

class Log10(Function):
    def forward(self, x):
        return np.log10(x)

class Log1p(Function):
    def forward(self, x):
        return np.log1p(x)

def log10(x):
    return Log10()(x).set_title(x.title)

def log1p(x):
    return Log1p()(x).set_title(x.title)


#####################################################################
## FFT functions

class FFT(Function):
    def forward(self, x):
        return fft2d.fft2(x)

class FFTShift(Function):
    def forward(self, x):
        return fft2d.fftshift(fft2d.fft2(x))

class IFFT(Function):
    def forward(self, amp, ang):
        x = amp if ang is None else amp * np.exp(1j * ang)
        return fft2d.ifft2(x)

class IFFTShift(Function):
    def forward(self, amp, ang):
        x = amp if ang is None else amp * np.exp(1j * ang)
        return fft2d.ifft2(fft2d.ifftshift(x))

class DCT(Function):
    def forward(self, x):
        return fft2d.dct(fft2d.dct(x.T, type=2, norm='ortho').T, type=2, norm='ortho')

class IDCT(Function):
    def forward(self, x):
        return fft2d.idct(fft2d.idct(x.T, type=2, norm='ortho').T, type=2, norm='ortho')


def fft(x):
    return FFT()(x).set_title(x.title).add_title("fft")

def fftshift(x):
    return FFTShift()(x).set_title(x.title).add_title("fftshift")

def ifft(amp, ang=None):
    return IFFT()(amp, ang)

def ifftshift(amp, ang=None):
    return IFFTShift()(amp, ang)

def dct(x):
    return DCT()(x).set_title(x.title).add_title("dct")

def idct(x):
    return IDCT()(x)


if __name__ == "__main__":

    x = Image(np.array([
        [complex(1, 1), complex(2, 2)],
        [complex(2, 2), complex(1, 1)]
    ]))
    x.title = "raw"

    y = x.fft.amplitude

    print(x.data, x.title)
    print(y.data, y.title)
