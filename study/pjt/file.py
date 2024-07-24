import os
import json

import numpy as np
import pandas as pd
import skimage

from test.image import Image


#####################################################################
## make file path list
def get_data_paths(data_dir):
    paths = []
    for path, _, files in os.walk(data_dir):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == ".csv":
                if name.endswith("X") or name.endswith("Y") or name.endswith("Z"):
                    paths.append(os.path.join(path, filename))
    return paths


def get_data_df(data_paths):
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in data_paths]
    temp      = [filename.split() for filename in filenames]

    names     = [row[0] for row in temp]
    freqs     = [int(row[1]) for row in temp]
    dimmings  = [int(row[2]) if len(row) > 3 else int(row[2].split('_')[0]) for row in temp]
    levels    = [int(row[-1].split('_')[0]) if len(row) > 3 else 60 for row in temp]
    xyz       = [row[-1].split('_')[1] for row in temp]

    df = pd.DataFrame({
            "filename": filenames,
            "name":     names,
            "freq":     freqs,
            "dimming":  dimmings,
            "level":    levels,
            "type":     xyz,
            "path":     data_paths
            })
    return df


class DataPath:
    def __init__(self, data_dir=None):
        if data_dir is not None:
            data_paths = get_data_paths(data_dir)
            self.df = get_data_df(data_paths)
        else:
            self.df = None
            
    def to_list(self):
        return self.df.path.to_list()

    ## ==============================================================
    ## setters
    def name(self, *name_):
        df = DataPath()
        df.df = self.df[self.df.name.isin(name_)]
        return df

    def freq(self, *freq_):
        df = DataPath()
        df.df = self.df[self.df.freq.isin(freq_)]
        return df

    def dimming(self, *dimming_):
        df = DataPath()
        df.df = self.df[self.df.dimming.isin(dimming_)]
        return df

    def level(self, *sensing_):
        df = DataPath()
        df.df = self.df[self.df.level.isin(sensing_)]
        return df

    def type(self, *type_):
        df = DataPath()
        df.df = self.df[self.df.type.isin(type_)]
        return df
    
    ## ==============================================================
    ## getters
    def get_names(self):
        return sorted(set(self.df.name.to_list()))
    
    def get_freqs(self):
        return sorted(set(self.df.freq.to_list()))
    
    def get_dimmings(self):
        return sorted(set(self.df.dimming.to_list()))
    
    def get_levels(self):
        return sorted(set(self.df.level.to_list()))
    
    def get_types(self):
        return sorted(set(self.df.type.to_list()))

    def get_paths(self):
        return sorted(set(self.df.path.to_list()))

#####################################################################
## read files
def read_img(path):
    img = skimage.io.imread(path)
    if img.ndim > 2 and img.shape[-1] > 3:
        img = skimage.color.rgba2rgb(img)
    img = skimage.util.img_as_float(img)
    return img

def read_csv(path):
    img = np.genfromtxt(path, delimiter=',').clip(1e-9)
    return img

def read_json(path):
    with open(path, "r") as f:
        spec = json.load(f)
    return spec


#####################################################################
## image data conversion
def rgb2xyz(spec):
    Wx, Wy = spec["Wx"], spec["Wy"]
    Rx, Ry = spec["Rx"], spec["Ry"]
    Gx, Gy = spec["Gx"], spec["Gy"]
    Bx, By = spec["Bx"], spec["By"]

    xyz_from_rgb = np.array([
        [Rx / Ry, Gx / Gy, Bx / By],
        [1, 1, 1],
        [(1 - Rx - Ry)/Ry, (1 - Gx - Gy)/Gy, (1 - Bx - By)/By]
    ])
    xyz_w = np.array([ Wx/Wy, 1, (1 - Wx - Wy)/Wy])
    L_ratio = xyz_w @ np.linalg.inv(xyz_from_rgb).T
    return L_ratio * xyz_from_rgb


def rgb_to_xyz(rgb, spec):
    lin_rgb = rgb.data.copy()
    mask = lin_rgb > 0.04045
    lin_rgb[mask] = np.power((lin_rgb[mask] + 0.055) / 1.055, 2.4)
    lin_rgb[~mask] /= 12.92

    XYZ = lin_rgb @ rgb2xyz(spec).T
    X = Image(XYZ[..., 0], "X").clip(1e-9)
    Y = Image(XYZ[..., 1], "Y").clip(1e-9)
    Z = Image(XYZ[..., 2], "Z").clip(1e-9)
    return X, Y, Z


def xyz_to_lxy(X, Y, Z):
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    L = Y / Y.max
    return L.set_name("L"), x.set_name("x"), y.set_name("y")


def xyz_to_rgb(X, Y, Z, spec):
    rgb_from_xyz = np.linalg.inv(rgb2xyz(spec))
    xyz = np.stack([X.data, Y.data, Z.data], axis=-1)
    xyz /= Y.data.max()
    rgb = xyz @ rgb_from_xyz.T

    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
    rgb[~mask] *= 12.92
    return Image(rgb).norm()


if __name__ == "":

    pass
