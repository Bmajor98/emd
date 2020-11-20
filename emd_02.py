import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema


def local_extrema(S):
    local_maxima = argrelextrema(S, comparator=np.greater_equal)[0]
    local_minima = argrelextrema(S, comparator=np.less_equal)[0]
    return local_minima, local_maxima


def cubic_spline(S, local_extrema):
    x = local_extrema
    y = S[x]
    return CubicSpline(x, y, bc_type='natural')  # 2nd  order cubic spline


def SD(h0, h1):
    # Standard deviation between h0 and h1
    SD = 0
    for t in range(len(h0)):
        SD += (h0[t] - h1[t]) ** 2 / h0[t] ** 2
    return SD


def is_monotonic(S):
    return (all(S[i] <= S[i + 1] for i in range(len(S) - 1)) or
            all(S[i] >= S[i + 1] for i in range(len(S) - 1)))


class EMD:
    def __init__(self, S, num_sifts=10):
        self.T = np.arange(len(S))
        self.S = S
        self.num_sifts = num_sifts

    def __call__(self, criterion=0.2):
        self.criterion = criterion
        imf_list = []

        c = self.IMF(self.S)
        imf_list.append(c)
        r = self.S - c
        while not is_monotonic(r):
            c = self.IMF(r)
            imf_list.append(c)
            r = r - c
        imf_list.append(r)
        # Note this residue is not an IMF but the baseline trend of original signal
        return np.asarray(imf_list)

    def sift(self, S):
        S_min, S_max = local_extrema(S)
        # checks to see if any extrema have been found - if none then creates a zero array to allow further arithmetic operations
        max_envelope =  np.asarray(cubic_spline(S, S_max)(self.T))
        min_envelope =  np.asarray(cubic_spline(S, S_min)(self.T))
        # np.zeros(len(self.T)) if len(S_max) <= 1 else
        # np.zeros(len(self.T)) if len(S_min) <= 1 else
        m = (max_envelope + min_envelope) / 2.0
        h = S - m
        return h

    def IMF(self, S):
        h = self.sift(S)
        if len(h.shape) > 1:
            h = h[0, :]
        for i in range(self.num_sifts - 1):
            h = self.sift(h)
            if len(h.shape) > 1:
                h = h[0, :]
        return h

def fn(t):
    return np.cos(3/30 * np.pi * t) + np.cos(2/34 * np.pi * t)

if __name__ == '__main__':
    t = np.arange(0, 540) 
    s = fn(t)

    emd = EMD(s)
    imf = emd.sift(s.copy())
    plt.plot(s)
    plt.plot(imf)
    plt.show()
