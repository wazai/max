"""
Performance class

Contains all performance related analysis: measurement, attribution etc
"""

import logging
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Performance(object):
    """Performance

    @member object_return: pandas series, could be single stock return, portfolio return, alpha factor return etc
    @member benchmark_return: pandas series, benchmark return
    @member metrics, dict, performance measurement including sharpe ratio, IR, alpha, beta etc
    """

    def __init__(self, object_name='target'):
        self.name = object_name
        self.object_return = pd.Series()
        self.benchmark_return = pd.Series()
        self.metrics = dict()

    def set_return_series(self, object_return, benchmark_return):
        if len(object_return) != len(benchmark_return):
            raise Exception('length of object return and benchmark return does not match')
        if not object_return.index.equals(benchmark_return.index):
            logger.warning('object return and benchmark return have different index date')
        self.object_return = object_return
        self.benchmark_return = benchmark_return

    def _check_return_exist(self):
        if not len(self.object_return):
            raise Exception('object return has not been set')
        if not len(self.benchmark_return):
            raise Exception('benchmark return has not been set')

    def get_metrics(self):
        sharpe_ratio = self.object_return.mean() / self.object_return.std()
        active_return = self.object_return.mean() - self.benchmark_return.mean()
        tracking_error = (self.object_return - self.benchmark_return).std()
        information_ratio = active_return / tracking_error
        self.metrics['avg daily return'] = self.object_return.mean()
        self.metrics['daily volatility'] = self.object_return.std()
        self.metrics['sharpe ratio'] = sharpe_ratio
        self.metrics['information ratio'] = information_ratio

        n_dates = len(self.benchmark_return)
        x = self.object_return.values.reshape(n_dates, 1)
        y = self.benchmark_return.values.reshape(n_dates, 1)
        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        [[beta]] = reg.coef_
        [alpha] = reg.intercept_
        self.metrics['alpha'] = alpha
        self.metrics['beta'] = beta

    def show_metrics(self):
        if len(self.metrics) == 0:
            self.get_metrics()
        print('--------- Performance Metrics --------')
        print('Avg Daily Return: ', self.metrics['avg daily return'])
        print('Daily Vol: ', self.metrics['daily volatility'])
        print('Sharpe Ratio: ', self.metrics['sharpe ratio'])
        print('Information Ratio: ', self.metrics['information ratio'])
        print('alpha: ', self.metrics['alpha'])
        print('beta:  ', self.metrics['beta'])

    def plot_return(self):
        self._check_return_exist()
        self.benchmark_return.cumsum().plot(style='b', legend=True)
        plot = self.object_return.cumsum().plot(style='g', legend=True)
        plt.legend(['Benchmark', 'strategy '+self.name])
        plot.set_ylabel('Return')
        plt.show()
