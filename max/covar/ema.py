import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from max.covar.covar import Covar
import max.covar.util as util

logger = logging.getLogger(__name__)


class EMA(Covar):

    def __init__(self, df, window):
        logger.info('Initializing EMA class')
        super(EMA, self).__init__(df, window)
        self.name = 'EMA'
        self.parameters = {'alpha_cor': np.nan, 'alpha_vol': np.nan}
        self.parameter_candidates = {'alpha_cor': np.array([]), 'alpha_vol': np.array([])}
        self.constants = {'eigenvalue_bound': 1e-5}
        self.results = {'log_likelihood': np.array([]), 'log_lik_path': np.array([]), 'cov_det_path': np.array([])}
        self.fit = {'cor': np.array([]), 'vol': np.array([]), 'cov': np.array([])}

    def get_ema_cov(self, alpha_cor, alpha_vol, save=False):
        logger.debug('Computing EMA covariance, alpha_cor = %.3f, alpha_vol = %.3f', alpha_cor, alpha_vol)
        n_dates, n_stocks = self.return_.shape
        cor_ema = self.ex_post['cor'].copy()
        vol_ema = self.ex_post['vol'].copy()
        cov_ema = np.repeat(np.nan, n_dates * n_stocks * n_stocks).reshape(n_dates, n_stocks, n_stocks)
        cov_ema[self.window - 1, :, :] = util.nan_cov(cor_ema[self.window - 1, :, :], vol_ema[self.window - 1, :])
        for i in range(self.window, n_dates):
            cor_ema[i, :, :] = util.nan_ema(cor_ema[i-1, :, :], self.ex_post['cor'][i, :, :], alpha_cor)
            vol_ema[i, :] = util.nan_ema(vol_ema[i-1, :], self.ex_post['vol'][i, :], alpha_vol)
            cov_ema[i, :, :] = util.nan_cov(cor_ema[i, :, :], vol_ema[i, :])
        if save:
            self.fit['cor'] = cor_ema
            self.fit['vol'] = vol_ema
            self.fit['cov'] = cov_ema
        return cov_ema

    def compute_log_likelihood(self, cov, i_pos, j_pos, lag=1):
        logger.debug('Computing log likelihood')
        n_dates = self.return_.shape[0]
        X = np.nan_to_num(self.return_.values.copy())  # fill nan with 0
        log_lik = 0
        cnt = 0
        for i in range(self.window, n_dates - lag):
            cov_i = cov[i, :, :]
            nan_idx = np.isnan(np.diag(cov_i))
            cov_i = cov_i[~nan_idx, :][:, ~nan_idx]
            if not np.any(np.isnan(cov_i)):
                eigval, eigvec = np.linalg.eig(cov_i)
                eigval[eigval < self.constants['eigenvalue_bound']] = 0
                cov_det_i = np.prod(eigval[eigval > 0])
                self.results['cov_det_path'][i, i_pos, j_pos] = cov_det_i
                cnt += 1
                log_lik_i = -(np.sum(eigval > 0) / 2.0) * np.log(2 * np.pi)
                log_lik_i += -0.5 * np.log(cov_det_i)
                xvec = X[i + lag, ~nan_idx]
                zvec = xvec.dot(eigvec)[eigval > 0]
                log_lik_i += -0.5 * np.sum(zvec * zvec * (1 / eigval[eigval > 0]))
                self.results['log_lik_path'][i, i_pos, j_pos] = log_lik_i
                log_lik += log_lik_i
        return log_lik / cnt

    def plot_likelihood_heatmap(self):
        plt.figure()
        plt.imshow(self.results['log_likelihood'], cmap='hot')
        plt.colorbar()
        plt.show()

    def plot_likelihood_path(self, i, j):
        plt.figure()
        plot = pd.Series(self.results['log_lik_path'][:, i, j], index=self.return_.index).plot()
        plot.set_ylabel('Log Likelihood')
        plot.set_title(
            'alpha_cor=' + str(self.parameter_candidates['alpha_cor'][i]) +
            ', alpha_vol=' + str(self.parameter_candidates['alpha_vol'][j]))
        plt.show()

    def calibrate(self, alphas_cor, alphas_vol):
        logger.info('Calibrating EMA parameters')
        n_cor = np.size(alphas_cor)
        n_vol = np.size(alphas_vol)
        log_lik = np.repeat(np.nan, n_cor * n_vol).reshape(n_cor, n_vol)
        n_dates = self.return_.shape[0]
        self.results['log_lik_path'] = np.repeat(np.nan, n_cor*n_vol*n_dates).reshape(n_dates, n_cor, n_vol)
        self.results['cov_det_path'] = np.repeat(np.nan, n_cor*n_vol*n_dates).reshape(n_dates, n_cor, n_vol)
        for i in range(n_cor):
            for j in range(n_vol):
                cov = self.get_ema_cov(alphas_cor[i], alphas_vol[j])
                log_lik[i, j] = self.compute_log_likelihood(cov, i, j)
        max_idx = np.argwhere(log_lik.max() == log_lik)
        self.parameters['alpha_cor'] = alphas_cor[max_idx[0][0]]
        self.parameters['alpha_vol'] = alphas_vol[max_idx[0][1]]
        self.get_ema_cov(self.parameters['alpha_cor'], self.parameters['alpha_vol'], save=True)
        self.results['log_likelihood'] = log_lik
        self.plot_likelihood_heatmap()
        self.parameter_candidates['alpha_cor'] = alphas_cor
        self.parameter_candidates['alpha_vol'] = alphas_vol
        self.plot_likelihood_path(max_idx[0][0], max_idx[0][1])
        logger.info('Optimal solution: alpha_cor = %.2f, alpha_vol = %.2f, avg loglik = %.2f',
                    self.parameters['alpha_cor'], self.parameters['alpha_vol'], log_lik.max())

    def plot_fit(self, variable, i, j=None):
        if variable not in ['cov', 'cor', 'vol']:
            raise Exception('variable not valid')
        if variable == 'vol':
            ex_post_series = self.ex_post[variable][self.window:, i]
            fit_series = self.fit[variable][self.window:, i]
        else:
            if j is None:
                raise Exception('Need to provide both i and j for cor/cov plot')
            else:
                ex_post_series = self.ex_post[variable][self.window:, i, j]
                fit_series = self.fit[variable][self.window:, i, j]
        util.plot_fit(ex_post_series, fit_series, self.return_.index[self.window:],
                      y_label=variable, legend=['Ex Post', self.name])
