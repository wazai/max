"""
Utility functions for Covar class

"""


import numpy as np


def ex_post_cor(ret, window, date):
    d = ret[ret.index <= date]
    [nt, n] = d.shape
    if nt < window:
        return np.repeat(np.nan, n * n).reshape(n, n)
    else:
        return d.tail(window).corr().values


def ex_post_cov(ret, window, date):
    d = ret[ret.index <= date]
    [nt, n] = d.shape
    if nt < window:
        return np.repeat(np.nan, n * n).reshape(n, n)
    else:
        return d.tail(window).cov().values


def ex_post_vol(ret, window, date):
    d = ret[ret.index <= date]
    [nt, n] = d.shape
    if nt < window:
        return np.repeat(np.nan, n)
    else:
        return d.tail(window).std().values


def nan_cov(cor, vol):
    voldiag = np.nan_to_num(np.diag(vol))
    cov = voldiag.dot(np.nan_to_num(cor)).dot(voldiag)
    cov[np.isnan(cor)] = np.nan
    return cov


def nan_ema(s0, y1, alpha):
    s1 = alpha * y1 + (1 - alpha) * s0
    s1[np.isnan(s0)] = y1[np.isnan(s0)]
    return s1
