"""
Single-Index Shrinkage

Reference: Ledoit, Olivier, and Michael Wolf. "Improved estimation of the covariance matrix of stock
returns with an application to portfolio selection." Journal of empirical finance (2003): 603-621.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import logging

from max.covar.covar import Covar

logger = logging.getLogger(__name__)


class SIS(Covar):

    def __init__(self, df, window, index_name='sh'):
        logger.info('Initializing SIS class')
        if index_name not in df.code.tolist():
            raise Exception('Cannot find index data in df')
        super(SIS, self).__init__(df[df.code != index_name], window)
        self.index_return = df[df.code == index_name].pivot(columns='code', values='return')
        self.name = 'SIS'
        self.parameters = {'weight': np.nan}
        self.parameter_candidates = {'weight': np.array([])}
        self.constants = dict()
        self.estimate = {'cor': np.array([]), 'vol': np.array([]), 'cov': np.array([])}
        n_dates, n_stocks = self.n_dates, self.n_stocks
        self.results = dict()
        self.results['index_variance'] = np.repeat(np.nan, n_dates)
        self.results['beta'] = np.repeat(np.nan, n_dates*n_stocks).reshape(n_dates, n_stocks)
        self.results['mse'] = np.repeat(np.nan, n_dates*n_stocks).reshape(n_dates, n_stocks)
        self.results['F'] = np.repeat(np.nan, n_dates*n_stocks*n_stocks).reshape(n_dates, n_stocks, n_stocks)

    def rolling_regression(self):
        for i in range(self.window, self.n_dates):
            x_i = self.return_[i - self.window:i].values
            y_i = self.index_return.return_[i - self.window:i].values
            self.results['index_variance'][i] = y_i.var()
            lr = LinearRegression(fit_intercept=False)
            for j in range(self.n_stocks):
                x_ij = x_i[:, j].reshape(180, 1)
                fit = lr.fit(x_ij, y_i)
                self.results['beta'][i, j] = fit.coef_.item()
                self.results['mse'][i, j] = np.mean((fit.predict(x_ij) - y_i) ** 2)
                beta_i = self.results['beta'][i, :].reshape(self.n_stocks, 1)

            self.results['F'][i, :, :] = beta_i.dot(beta_i.T) * self.results['index_variance'][i] + \
                np.diag(self.results['mse'][i, :])

    def cal_estimate(self, parameters):
        weight = parameters['weight']
        cov = weight * self.ex_post['cov'] + (1 - weight) * self.results['F']
        return cov
