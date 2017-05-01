"""
Exponential Moving Average
"""

import pandas as pd
import numpy as np
import logging

from max.covar.covar import Covar
import max.covar.util as util

logger = logging.getLogger(__name__)


class EMA(Covar):

    def __init__(self, df, window):
        logger.info('Initializing EMA class')
        super(EMA, self).__init__(df, window)
        self.name = 'EMA'
        self.parameter = {'alpha_cor': np.nan, 'alpha_vol': np.nan}
        self.parameter_candidate = {'alpha_cor': np.array([]), 'alpha_vol': np.array([])}
        self.result = {'value': np.nan}
        self.result_candidate = pd.DataFrame()
        self.estimate = {'cor': np.array([]), 'vol': np.array([]), 'cov': np.array([])}

    def cal_estimate(self, parameter):
        self._check_parameter(parameter)
        alpha_cor = parameter['alpha_cor']
        alpha_vol = parameter['alpha_vol']
        logger.debug('Computing EMA covariance, alpha_cor = %.3f, alpha_vol = %.3f', alpha_cor, alpha_vol)
        cor_ema = self.ex_post['cor'].copy()
        vol_ema = self.ex_post['vol'].copy()
        n_dates, n_stocks = self.n_dates, self.n_stocks
        cov_ema = np.repeat(np.nan, n_dates*n_stocks*n_stocks).reshape(n_dates, n_stocks, n_stocks)
        cov_ema[self.window - 1, :, :] = util.nan_cov(cor_ema[self.window - 1, :, :], vol_ema[self.window - 1, :])
        for i in range(self.window, n_dates):
            cor_ema[i, :, :] = util.nan_ema(cor_ema[i-1, :, :], self.ex_post['cor'][i, :, :], alpha_cor)
            vol_ema[i, :] = util.nan_ema(vol_ema[i-1, :], self.ex_post['vol'][i, :], alpha_vol)
            cov_ema[i, :, :] = util.nan_cov(cor_ema[i, :, :], vol_ema[i, :])
        return {'cor': cor_ema, 'vol': vol_ema, 'cov': cov_ema}
