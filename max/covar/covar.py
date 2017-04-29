import pandas as pd
import numpy as np
import logging
import os
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
        self._get_cov_seq()

    def _get_cov_seq(self):
        logger.info('Computing rolling %i-days correlation and volatility', self.window)
        self.ex_post['cor'] = np.array([util.ex_post_cor(self.return_, self.window, x) for x in self.return_.index])
        self.ex_post['vol'] = np.array([util.ex_post_vol(self.return_, self.window, x) for x in self.return_.index])
        self.ex_post['cov'] = np.array([util.ex_post_cov(self.return_, self.window, x) for x in self.return_.index])

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
