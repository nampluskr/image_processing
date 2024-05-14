import os
from glob import glob
import pandas as pd


def get_data_path(dirnames):
    filenames, temp, patterns = [], [], []
    freqs, dimmings, sensing, xyz = [], [], [], []
    if not isinstance(dirnames, (list, tuple)):
        dirnames = [dirnames]
    paths = []
    for dirname in dirnames:
        paths += glob(os.path.join(dirname, "*.csv"))

    filenames   = [os.path.basename(path) for path in paths]
    temp        = [name[:-4].split() for name in filenames]
    patterns    = [x[0] for x in temp]
    freqs       = map(int, [x[1] for x in temp])
    dimmings    = map(int, [x[2] for x in temp])
    sensing     = map(int,[x[3].split('_')[0] for x in temp])
    xyz         = [x[3].split('_')[1] for x in temp]

    df = pd.DataFrame({
        "filename": filenames,
        "pattern":  patterns,
        "freq":     freqs,
        "dimming":  dimmings,
        "sensing":  sensing,
        "xyz":      xyz,
        "path":     paths
        })
    return df


def get_pattern_path(dirnames):
    filenames, temp, patterns = [], [], []
    if not isinstance(dirnames, (list, tuple)):
        dirnames = [dirnames]
    paths = []
    for dirname in dirnames:
        paths += glob(os.path.join(dirname, "*.png"))

    filenames   = [os.path.basename(path) for path in paths]
    temp        = [name[:-4].split() for name in filenames]
    patterns    = [x[0] for x in temp]

    df = pd.DataFrame({
        "filename": filenames,
        "pattern":  patterns,
        "path":     paths
        })
    return df


class DataPath:
    def __init__(self, dirnames):
        self.df = get_data_path(dirnames)

    def pattern(self, *pattern_):
        self.df = self.df[self.df.pattern.isin(pattern_)]
        return self

    def freq(self, *freq_):
        self.df = self.df[self.df.freq.isin(freq_)]
        return self

    def dimming(self, *dimming_):
        self.df = self.df[self.df.dimming.isin(dimming_)]
        return self

    def sensing(self, *sensing_):
        self.df = self.df[self.df.sensing.isin(sensing_)]
        return self

    def xyz(self, *xyz_):
        self.df = self.df[self.df.xyz.isin(xyz_)]
        return self

    def to_list(self):
        return self.df.path.to_list()


class PatternPath:
    def __init__(self, dirnames):
        self.df = get_pattern_path(dirnames)

    def pattern(self, *pattern_):
        self.df = self.df[self.df.pattern.isin(pattern_)]
        return self

    def to_list(self):
        return self.df.path.to_list()


if __name__ == "__main__":

    from image import *
    from fft import *
    from viewer import Viewer
    from collections import Counter

    viewer = Viewer()

    dirnames = [
    "E:\\data_2024\\S_L_Optimization_1",
    "E:\\data_2024\\S_L_Optimization_2"
    ]

    paths = DataPath(dirnames).pattern('t2_10_g').xyz('Y').dimming(183, 1600).to_list()
    for path in paths:
        print(path)
        
    def show_fft(path):
        img = Imread(path)
        fft = FFT2DShift(img)
        amp = Amplitude(fft)
        ang = Phase(fft)
        inv = InvFFT2DShift(amp, ang)
        viewer.show(img, Log1p(amp), ang, Abs(inv))

    for path in paths[:3]:   
        show_fft(path)
