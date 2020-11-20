import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema


def local_extrema(s):
    local_maxima = argrelextrema(s, comparator=np.greater_equal)[0]
    local_minima = argrelextrema(s, comparator=np.less_equal)[0]
    return local_minima, local_maxima

def envelope(S, local_extrema):
    t = np.arange(len(S))
    return t * 0 if len(local_extrema) <= 1 else np.asarray(CubicSpline(local_extrema, S[local_extrema], bc_type='natural') (t)) 
 
def sift(s):
        s_min, s_max = local_extrema(s)
        max_envelope = envelope(s, s_max)
        min_envelope = envelope(s, s_min)
        mean_envelope = (max_envelope + min_envelope) / 2.0
        return s - mean_envelope

def imf(s, num_sifts):
    for _ in range(num_sifts):
        s = sift(s)
    return s

def monotonic(s):
    return (all(s[i] <= s[i + 1] for i in range(len(s) - 1)) or
            all(s[i] >= s[i + 1] for i in range(len(s) - 1)))

def emd(s, criterion='num_sifts', num_sifts=100, sd=0.2):
    imf_list = []
    if criterion == 'num_sifts':
        while not monotonic(s):
            s_imf = imf(s, num_sifts)
            s = s - s_imf
            imf_list.append(s)
        imf_list.append(s) 

    elif criterion == 'sd':
        pass
    else:
        raise ValueError

    return imf_list
def fn(t):
    return np.cos(3/30 * np.pi * t) + np.cos(2/34 * np.pi * t)

if __name__ == "__main__":
    t = np.arange(0, 540) 
    s = fn(t)
    imf = emd(s.copy())
    plt.plot(s)
    for i in imf:
        plt.plot(i)
    plt.show()
