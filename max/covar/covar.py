import pandas as pd
import numpy as np
import logging
import os
import abc
from max.datacenter import DataCenter
import max.covar.util as util

logger = logging.getLogger(__name__)


class Covar(object):

    def __init__(self, df=pd.DataFrame(), window=90):
        logger.info('Initializing Covariance base class')
        if type(df.index) != pd.tseries.index.DatetimeIndex:
            raise Exception('type of index must be pandas time series')
        if 'code' not in df.columns:
            raise Exception('code column is needed in input data frame')
        if 'return' not in df.columns:
            raise Exception('return column is needed in input data frame')
        self.name = 'Ex Post'
        self.return_ = df.pivot(columns='code', values='return')
        self.n_dates, self.n_stocks = self.return_.shape
        logger.info('Time period from %s to %s', min(df.index).strftime('%Y-%m-%d'), max(df.index).strftime('%Y-%m-%d'))
        logger.info('# of dates: %i, # of stocks: %i', self.return_.shape[0], self.return_.shape[1])
        logger.info('Rolling window %i days', window)
        self.window = window
        self.ex_post = dict()
        self._cal_ex_post()
        self.parameter = dict()
        self.parameter_candidate = dict()
        self.result = dict()
        self.result_candidate = pd.DataFrame()
        self.estimate = self.ex_post

    def _cal_ex_post(self):
        logger.info('Computing rolling %i-days correlation and volatility', self.window)
        self.ex_post['cor'] = np.array([util.ex_post_cor(self.return_, self.window, x) for x in self.return_.index])
        self.ex_post['vol'] = np.array([util.ex_post_vol(self.return_, self.window, x) for x in self.return_.index])
        self.ex_post['cov'] = np.array([util.ex_post_cov(self.return_, self.window, x) for x in self.return_.index])

    def _check_parameter(self, parameter):
        if set(parameter) != set(self.parameter):
            raise Exception('Input parameter dictionary does not match class parameter dictionary')

    def to_csv(self, cov, folder):
        logger.info('Saving files to csv')
        path = os.path.join(DataCenter.get_path('covariance'), folder)
        cols = self.return_.columns
        for i in range(self.window, self.n_dates):
            d = pd.DataFrame(cov[i, :, :], columns=cols)
            d['code'] = cols
            d = d[['code']+cols.tolist()]
            date = self.return_.index[i]
            dir_name = os.path.join(path, str(date.year))
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            filename = os.path.join(dir_name, date.strftime('%Y%m%d')+'.csv')
            logger.info('Saving file to %s', filename)
            d.to_csv(filename, index=False)

    def plot_ex_post(self, variable, i, j=None):
        logger.info('Plotting ex-post %s', variable)
        if variable not in ['cov', 'cor', 'vol']:
            raise Exception('variable not valid')
        if variable == 'vol':
            ex_post_series = self.ex_post[variable][self.window:, i]
        else:
            if j is None:
                raise Exception('Need to provide both i and j for cor/cov plot')
            else:
                ex_post_series = self.ex_post[variable][self.window:, i, j]
        util.plot_series(ex_post_series, index=self.return_.index[self.window:], y_label=variable, legend=['Ex Post'])

    @abc.abstractmethod
    def cal_estimate(self, parameter):
        return

    def calibrate(self, parameter_candidate, metric):
        logger.info('Calibrating parameter using %s', metric.method)
        if type(parameter_candidate) == pd.core.frame.DataFrame:
            self.parameter_candidate = parameter_candidate
            parameter_candidate = parameter_candidate.to_dict('record')
        else:
            self.parameter_candidate = pd.DataFrame(parameter_candidate)
        results = []
        for parameter in parameter_candidate:
            estimate = self.cal_estimate(parameter)
            result = metric.evaluate(estimate['cov'], self.return_, self.window)
            results.append(result)
            if np.isnan(self.result['value']) or metric.is_better(result, self.result):
                self.parameter = parameter
                self.result = result
                self.estimate = estimate
        self.result_candidate = pd.DataFrame(results)
        metric.plot_value_path(self.return_.index, self.result['value_path'])
        metric.plot_candidate_performance(self.parameter_candidate, self.result_candidate)
        logger.info('Optimal parameter: %s, %s value = %.2f', str(self.parameter), metric.method, self.result['value'])
