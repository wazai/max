"""
Performance class

Contains all performance related analysis: measurement, attribution etc
"""

import logging
from sklearn import linear_model

logger = logging.getLogger(__name__)

class Performance(object):
    """Performance

    @param object_return: pandas series, could be single stock return, portfolio return, alpha factor return etc
    @param benchmark_return: pandas series, benchmark return
    """

    def __init__(self, object_return, benchmark_return):
        if len(object_return) != len(benchmark_return):
            raise Exception('length of object return and benchmark return does not match')
        if not object_return.index.equals(benchmark_return.index):
            logger.warning('object return and benchmark return have different index date')
        self.object_return = object_return
        self.benchmark_return = benchmark_return
        self.metrics = dict()

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
