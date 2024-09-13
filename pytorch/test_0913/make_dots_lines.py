import os
from pathlib import Path
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
def make_lines(max, min, n_lines, shift, angle, img_size):
    amplitude = (max - min) / 2
    mean = (max + min) / 2
    wavelength = img_size // n_lines
    angle = np.deg2rad(angle)
    rng = np.arange(-img_size // 2, img_size // 2, 1)
    X, Y = np.meshgrid(rng, rng)
    rot = X*np.cos(angle) + Y*np.sin(angle)
    img = np.sin(2*np.pi*(rot) / wavelength - 2*np.pi / 10 * shift)*amplitude + mean
    return skimage.color.gray2rgb(img)

def save_to_png(data, img_dir, img_name):
    Path(img_dir).mkdir(exist_ok=True)
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        data *= 255
        data = data.astype("uint8")
        skimage.io.imsave(img_path, data, check_contrast=False)

img_size = 256
img1 = make_gaussian2d(0.1, 0.1, sig=50, max=0.5, min=0.2, img_size=img_size)
img2 = make_lines(max=0.5, min=0.2, n_lines=4, shift=0, angle=90, img_size=img_size)
info(img1)
info(img2)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
ax1.imshow(img1, vmax=1.0, vmin=0.0)
ax2.imshow(img2, vmax=1.0, vmin=0.0)
fig.tight_layout()
plt.show()

img_dir = "/home/namu/myspace/NAMU/datasets/data_2024/lines"
img_size = 256

cnt = 1
max, min, n_lines, shift, angle = 0.5, 0.1, 3, 0, 90
data = make_lines(max, min, n_lines, shift, angle, img_size=img_size)
img_name = f"lines{str(cnt).zfill(4)}_max-{max:.1f}_min-{min:.1f}_nlines-{n_lines}_shift-{shift}_angle-{angle}.png"
# save_to_png(data, img_dir, img_name)

img_dir = "/home/namu/myspace/NAMU/datasets/data_2024/dots"
img_size = 256

sig_list = np.arange(10, 50, 10)
x0_list = [0.3, -0.3, 0.1, -0.1, 0.2, -0.2]
y0_list = [0.3, -0.3, 0.1, -0.1, 0.2, -0.2]
min_list = [0.0]
max_list = [0.2, 0.3, 0.4, 0.5]

cnt = 0
for min in min_list:
    for max in max_list:
        for sig in sig_list:
            for x0 in x0_list:
                for y0 in y0_list:
                    cnt += 1
                    # data = make_gaussian2d(x0, y0, sig, max, min, img_size)
                    img_name = f"dots{str(cnt).zfill(4)}_max-{max:.1f}_min-{min:.1f}_sig-{sig}_({x0:.1f}_{y0:.1f}).png"
                    # save_to_png(data, img_dir, img_name)
print(img_name)

img_dir = "/home/namu/myspace/NAMU/datasets/data_2024/lines"
img_size = 256

min_list = [0.0]
max_list = [0.2, 0.3, 0.4, 0.5]

cnt = 0
for min in min_list:
    for max in max_list:
        for n_lines in [2, 3, 4, 5]:
            for angle in [0, 30, 60, 90]:
                for shift in [0, 2, 4, 6, 8]:
                    cnt += 1
                    # data = make_lines(max, min, n_lines, shift, angle, img_size=img_size)
                    img_name = f"lines{str(cnt).zfill(4)}_max-{max:.1f}_min-{min:.1f}_nlines-{n_lines}_shift-{shift}_angle-{angle}.png"
                    # save_to_png(data, img_dir, img_name)

print(img_name)
