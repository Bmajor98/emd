import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.io import arff

def local_extrema(s):
    max = []
    min = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1]:
            max.append(i)
        elif s[i] < s[i - 1] and s[i] < s[i + 1]:
            min.append(i)
    return min, max


def monotonic_decreasing(s):
    return all(a >= b for a, b in zip(s, s[1:]))


def monotonic_increasing(s):
    return all(a <= b for a, b in zip(s, s[1:]))


def monotonic(s):
   return monotonic_increasing(s) or monotonic_decreasing(s)


def sd(h1, h2):
    return np.sum((h2 - h1) ** 2 / h1 ** 2, axis=0)


"""
Empirical Mode Decomposition

This class is an implementation of the algorithm for EMD as demonstrated by Huang et al. (1998).
See https://royalsocietypublishing.org/doi/10.1098/rspa.1998.0193 for further details.
"""


class EMD:
    def __init__(self, criterion="num_sifts", num_sifts=200, sd=0.2):
        self.criterion = criterion
        self.num_sifts = num_sifts
        self.sd = sd

    def __call__(self, s):
        imf = self.imf(s)
        imf_set = imf
        r = s - imf
        while not monotonic(r):
            imf = self.imf(r)
            imf_set = np.vstack((imf_set, imf))
            r = r - imf
        imf_set = np.vstack((imf_set, r))
        return imf_set

    def sift(self, s):
        t = np.arange(len(s))
        s_min, s_max = local_extrema(s)

        if len(s_min) >= 2:
            min_cs = CubicSpline(s_min, s[s_min], bc_type='natural')
            min_envelope = min_cs(t)
        else:
            min_envelope = np.zeros(len(s))

        if len(s_max) >= 2:
            max_cs = CubicSpline(s_max, s[s_max], bc_type='natural')
            max_envelope = max_cs(t)
        else:
            max_envelope = np.zeros(len(s))

        m = max_envelope - min_envelope
        h = s - m
        return h

    def imf(self, s):
        if self.criterion == "num_sifts":
            h = self.sift(s)
            for _ in range(self.num_sifts):
                h = self.sift(s)
            return h
        elif self.criterion == "std_dev":
            h_prev = self.sift(s)
            h = self.sift(h_prev)
            while sd(h_prev, h) < self.sd:
                h_prev = h
                h = self.sift(h)
            return h


if __name__ == "__main__":
    # df = pd.read_csv('data/daily-min-temperatures.csv')
    # s = df.values[:, 1].astype('float')
    t = np.arange(0, 180 * 4)
    s = np.cos(3/30 * np.pi * t) + np.cos(2/34 * np.pi * t)
    emd = EMD("num_sifts")
    imf_set = emd(s)

    fig, axs = plt.subplots(len(imf_set) + 1, 1)
    axs[0].plot(s)
    for i in range(len(imf_set)):
        axs[i + 1].plot(imf_set[i])
    plt.show()

    # plt.plot(np.sum(imf_set, axis=0)[:250])
    # plt.plot(s[:250])
    # plt.show()
