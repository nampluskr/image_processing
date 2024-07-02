import numpy as np
from numpy import ndarray
import scipy.fft as fft2d


class Image:
    def __init__(self, data: ndarray, title: str=None):
        if data is not None:
            if not isinstance(data, ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.title = "" if title is None else title
        
    def set_title(self, title: str):
        self.title = title
        return self
    
    def add_title(self, title: str):
        if self.title == "":
            self.set_title(title)
        else:
            self.title += '_' + title
        return self

    @property
    def shape(self):
        return self.data.shape

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
    def real(self):
        return Image(self.data.real, self.title)

    @property
    def imag(self):
        return Image(self.data.imag, self.title)

    @property
    def conj(self):
        return Image(np.conj(self.data), self.title)

    @property
    def amplitude(self):
        return Image(np.abs(self.data), self.title).add_title("amplitude")

    @property
    def phase(self):
        return Image(np.angle(self.data), self.title).add_title("phase")

    @property
    def fft(self):
        return Image(fft2d.fft2(self.data), self.title).add_title("fft")

    @property
    def fftshift(self):
        return Image(fft2d.fftshift(fft2d.fft2(self.data)), self.title).add_title("fftshift")


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
    return Pow(n)(x).set_title(x.title)

def abs(x):
    return Abs()(x).set_title(x.title)


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


if __name__ == "__main__":

    img = Image(np.array([1, 2, 3])).set_title("data")
    print(img.data)
    print(img.title)

    img = img**2 / 2
    print(img.data)
    print(img.title)
