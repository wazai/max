"""
Metric class

Contains different metrics to evaluate risk model
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Metric(object):

    def __init__(self, method, options=None):
        logger.info('Initializing Metric class, method = %s', method)
        self.method = method
        if method == 'log_likelihood':
            default_options = {'lag': 1, 'eigenvalue_bound': 1e-5}
        elif method == 'out_of_sample_error':
            default_options = {'forward_window': 21}
        else:
            raise Exception('Cannot recognize the evaluation method')
        self.options = self._fill_options(default_options, options)

    @staticmethod
    def _fill_options(default_options, options=None):
        if options is not None:
            for param in default_options.keys():
                if param in options.keys():
                    default_options[param] = options[param]
        return default_options

    def evaluate(self, cov, return_, window):
        if self.method == 'log_likelihood':
            return self.log_likelihood(cov, return_, window)
        elif self.method == 'out_of_sample_error':
            return self.out_of_sample_error(cov, return_, window)
        else:
            raise Exception('Cannot recognize the evaluation method')

    def is_better(self, result_left, result_right):
        if self.method == 'log_likelihood':
            return result_left['value'] > result_right['value']
        elif self.method == 'out_of_sample_error':
            return result_left['value'] < result_right['value']
        else:
            raise Exception('Cannot recognize the evaluation method')

    def log_likelihood(self, cov, return_, window):
        logger.debug('Computing log likelihood')
        n_dates = return_.shape[0]
        X = np.nan_to_num(return_.values.copy())  # fill nan with 0
        log_lik = 0
        cnt = 0
        log_lik_path = np.repeat(np.nan, n_dates)
        cov_det_path = np.repeat(np.nan, n_dates)
        for i in range(window, n_dates - self.options['lag']):
            cov_i = cov[i, :, :]
            nan_idx = np.isnan(np.diag(cov_i))
            cov_i = cov_i[~nan_idx, :][:, ~nan_idx]
            if not np.any(np.isnan(cov_i)):
                eigval, eigvec = np.linalg.eig(cov_i)
                eigval[eigval < self.options['eigenvalue_bound']] = 0
                cov_det_i = np.prod(eigval[eigval > 0])
                cov_det_path[i] = cov_det_i
                cnt += 1
                log_lik_i = -(np.sum(eigval > 0) / 2.0) * np.log(2 * np.pi)
                log_lik_i += -0.5 * np.log(cov_det_i)
                xvec = X[i + self.options['lag'], ~nan_idx]
                zvec = xvec.dot(eigvec)[eigval > 0]
                log_lik_i += -0.5 * np.sum(zvec * zvec * (1 / eigval[eigval > 0]))
                log_lik_path[i] = log_lik_i
                log_lik += log_lik_i
        return {'value': log_lik/cnt, 'value_path': log_lik_path, 'cov_det_path': cov_det_path}

    def plot_value_path(self, date_index, value_path):
        plt.figure()
        plot = pd.Series(value_path, index=date_index).plot()
        plot.set_ylabel('Value')
        plot.set_title(self.method + ' time series of optimal solution')
        plt.show()

    def plot_candidate_performance(self, parameter_candidate, result_candidate):
        if self.method == 'log_likelihood':
            self.plot_likelihood_heatmap(parameter_candidate, result_candidate['value'])
        elif self.method == 'out_of_sample_error':
            return

    @staticmethod
    def get_likelihood_heatmap(parameter_candidate, log_lik):
        if len(log_lik) != parameter_candidate.shape[0]:
            raise Exception('Likelihood and parameter candidate length not match')
        n_cor = len(set(parameter_candidate['alpha_cor']))
        n_vol = len(set(parameter_candidate['alpha_vol']))
        heatmap = log_lik.values.reshape(n_cor, n_vol).T
        return heatmap

    @staticmethod
    def plot_likelihood_heatmap(parameter_candidate, log_lik):
        plt.figure()
        heatmap = Metric.get_likelihood_heatmap(parameter_candidate, log_lik)
        extent = [parameter_candidate['alpha_cor'].min(), parameter_candidate['alpha_cor'].max(),
                  parameter_candidate['alpha_vol'].max(), parameter_candidate['alpha_vol'].min()]
        plt.imshow(heatmap, cmap='hot', extent=extent)
        plt.xlabel('alpha_cor')
        plt.ylabel('alpha_vol')
        plt.title('Log likelihood heatmap')
        plt.colorbar()
        plt.show()

    def out_of_sample_error(self, cov, return_, window):
        pass
