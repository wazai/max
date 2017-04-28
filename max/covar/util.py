"""
Utility functions for Covar class

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_series(series1, series2=None, index=None, y_label=None, legend=None):
    plt.figure()
    plot = pd.Series(series1, index=index).plot(style='b', legend=True)
    if series2 is not None:
        plot = pd.Series(series2, index=index).plot(style='g', legend=True)
    plot.set_ylabel(y_label)
    plt.legend(legend)
    plt.show()


def plot_fit(obj, variable, i, j=None):
    if variable not in ['cov', 'cor', 'vol']:
        raise Exception('variable not valid')
    if variable == 'vol':
        ex_post_series = obj.ex_post[variable][obj.window:, i]
        fit_series = obj.fit[variable][obj.window:, i]
    else:
        if j is None:
            raise Exception('Need to provide both i and j for cor/cov plot')
        else:
            ex_post_series = obj.ex_post[variable][obj.window:, i, j]
            fit_series = obj.fit[variable][obj.window:, i, j]
    plot_series(ex_post_series, fit_series, obj.return_.index[obj.window:],
                y_label=variable, legend=['Ex Post', obj.name])
