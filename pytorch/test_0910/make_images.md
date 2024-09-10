```python
import numpy as np
import matplotlib.pyplot as plt
import skimage

def info(img):
    print(f">> shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min():.2f}, max={img.max():.2f}, "
          f"mean={img.mean():.2f}")

# img_name = f"dots_max-{max}_min-{min}_sig-{sig}.png"
def make_gaussian2d(x0, y0, sig, max, min, img_size):
    x = np.arange(-img_size // 2, img_size // 2, 1)
    X, Y = np.meshgrid(x, x)
    X = np.exp(-(X - x0 * img_size)**2 / sig**2 / 2)
    Y = np.exp(-(Y - y0 * img_size)**2 / sig**2 / 2)
    img = X * Y * (max - min) + min
    return skimage.color.gray2rgb(img)

# img_name = f"lines_max-{}_min-{}_nlines-{}_shift-{}_angle-{angle}.png"
def make_lines(max, min, n_lines, shift, img_size, angle=90):
    amplitude = (max - min) / 2
    mean = (max + min) / 2
    wavelength = img_size // n_lines
    angle = np.deg2rad(angle)
    rng = np.arange(-img_size // 2, img_size // 2, 1)
    X, Y = np.meshgrid(rng, rng)
    rot = X*np.cos(angle) + Y*np.sin(angle)
    img = np.sin(2*np.pi*(rot) / wavelength - 2*np.pi / 10 * shift)*amplitude + mean
    return skimage.color.gray2rgb(img)
```
