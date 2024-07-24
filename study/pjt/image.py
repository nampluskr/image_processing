import numpy as np
from numpy import ndarray
import scipy.fft as fft2d
from scipy.ndimage import uniform_filter, gaussian_filter
import skimage


class Image:
    def __init__(self, data: ndarray, name=None):
        if data is not None:
            if not isinstance(data, ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.name = "" if name is None else name
        self._axis_off = False

    def info(self):
        print(f">> {self.shape} dtype={self.dtype}", end=", ")
        if self.dtype == np.float64:
            print(f"min={self.min:.2f}, max={self.max:.2f}", end=" ")
        else:
            print(f"min={self.min}, max={self.max}", end=" ")
        print(f": {[self.name]}")
        return self

    ## ==============================================================
    ## setters
    def set_name(self, name):
        self.name = name
        return self

    def add_name(self, name):
        self.name = name if self.name == "" else self.name + '_' + name
        return self

    def axis_off(self):
        self._axis_off = True
        return self

    def axis_on(self):
        self._axis_off = False
        return self

    ## ==============================================================
    ## properties
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def min(self):
        return self.data.min()

    @property
    def max(self):
        return self.data.max()

    @property
    def mean(self):
        return self.data.mean()

    @property
    def std(self):
        return self.data.std()

    @property
    def var(self):
        return self.data.var()

    @property
    def real(self):
        return Image(self.data.real, self.name)

    @property
    def imag(self):
        return Image(self.data.imag, self.name)

    @property
    def conj(self):
        return Image(np.conj(self.data), self.name)

    @property
    def amplitude(self):
        return Image(np.abs(self.data), self.name).add_name("amplitude")

    @property
    def phase(self):
        return Image(np.angle(self.data), self.name).add_name("phase")

    @property
    def fft(self):
        return Image(fft2d.fft2(self.data), self.name).add_name("fft")

    @property
    def fftshift(self):
        return Image(fft2d.fftshift(fft2d.fft2(self.data)), self.name).add_name("fftshift")

    ## ==============================================================
    ## transformers
    def float(self):
        img = skimage.util.img_as_float(self.data)
        return Image(img, self.name)

    def ubyte(self):
        img = skimage.util.img_as_ubyte(self.data)
        return Image(img, self.name)

    def gray(self):
        if self.ndim > 2:
            if self.shape[-1] > 3:
                img = skimage.color.rgba2rgb(self.data)
            img = skimage.color.rgb2gray(self.data)
            return Image(img, self.name)
        return self

    def rgb(self):
        if self.ndim > 2 and self.shape[-1] > 3:
            img = skimage.color.rgba2rgb(self.data)
            return Image(img, self.name)
        if self.ndim == 2:
            img = skimage.color.gray2rgb(self.data)
            return Image(img, self.name)
        return self

    def resize(self, height, width):
        img = skimage.transform.resize(self.data, (height, width))
        return Image(img, self.name)

    def rotate(self, angle):
        img = skimage.transform.rotate(self.data, angle, resize=True)
        return Image(img, self.name)

    def clip(self, min=1e-9, max=None):
        img = np.clip(self.data, min, self.max if max is None else max)
        return Image(img, self.name)

    def norm(self, min=0., max=1.):
        img = np.interp(self.data, (self.min, self.max), (min, max))
        return Image(img, self.name)

    def scale(self, max):
        return Image(self.data / max, self.name)


#####################################################################
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Image):
        return obj
    return Image(obj)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Image(as_array(y)) for y in ys]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError


#####################################################################
## Operator overloading

class Add(Function):
    def forward(self, x1, x2):
        return x1 + x2

class Mul(Function):
    def forward(self, x1, x2):
        return x1 * x2

class Neg(Function):
    def forward(self, x):
        return -x

class Sub(Function):
    def forward(self, x1, x2):
        return x1 - x2

class Div(Function):
    def forward(self, x1, x2):
        return x1 / x2

class Pow(Function):
    def __init__(self, n):
        self.n = n

    def forward(self, x):
        return x**self.n

class Abs(Function):
    def forward(self, x):
        return np.abs(x)


def add(x1, x2):
    x2 = as_array(x2)
    return Add()(x1, x2)

def mul(x1, x2):
    x2 = as_array(x2)
    return Mul()(x1, x2)

def neg(x):
    return Neg()(x)

def sub(x1, x2):
    x2 = as_array(x2)
    return Sub()(x1, x2)

def rsub(x1, x2):
    x2 = as_array(x2)
    return Sub()(x2, x1)

def div(x1, x2):
    x2 = as_array(x2)
    return Div()(x1, x2)

def rdiv(x1, x2):
    x2 = as_array(x2)
    return Div()(x2, x1)

def pow(x, n):
    return Pow(n)(x).set_name(x.name)

def abs(x):
    return Abs()(x).set_name(x.name)


Image.__add__ = add
Image.__radd__ = add
Image.__mul__ = mul
Image.__rmul__ = mul
Image.__neg__ = neg
Image.__sub__ = sub
Image.__rsub__ = rsub
Image.__truediv__ = div
Image.__rtruediv__ = rdiv
Image.__pow__ = pow
Image.__abs__ = abs


#####################################################################
## Image transformers / filters
class Gaussian(Function):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, x):
        return gaussian_filter(x, sigma=self.sigma)

class Uniform(Function):
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        return uniform_filter(x, size=self.size)


def gaussian(x, sigma=1):
    return Gaussian(sigma)(x).set_name(x.name)

def uniform(x, size=1):
    return Uniform(size)(x).set_name(x.name)


if __name__ == "__main__":

    from viewer import Viewer
    viewer = Viewer()

    data = skimage.data.astronaut()
    raw = Image(data, "raw").resize(300, 200).gray().rotate(90).info()
    img1 = raw.clip(0.2).set_name("clip1").info()
    img2 = raw.clip(0.5).set_name("clip2").info()
    img3 = raw.clip(0.2, 0.6).set_name("clip3").info()
    img4 = img3.norm(0, 0.5).info()

    viewer.show(raw, img1, img2, img3, img4)

