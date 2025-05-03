#!/usr/bin/env python3

import numpy as np

def randspace(a, b, n, minstep=0.1, maxstep=0.6):
    ''' Generate a sequence from a to b with random steps
        minstep and maxstep define the step magnitude
        '''
    v = minstep + np.random.uniform(size=(n-1))*(maxstep-minstep)
    v = np.hstack([[0.0], v])
    v = v / np.sum(v)
    v = np.cumsum(v)
    return a + v*(b-a)


def rank(A, eps=1e-12):
    u, s, vh = np.linalg.svd(A)
    return len([x for x in s if abs(x) > eps])


def weighted_average(v, w):
    v = np.array(v)
    w = np.array(w)
    return np.sum(w*v) / np.sum(w)


from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def derivative(X, order=1):
    window_size = 15
    poly_order = 3
    return savgol_filter(X, window_size, poly_order, order, axis=0)

def derivative(X, order=1, sigma=20):
    return gaussian_filter1d(X, sigma, order=order, axis=0)
