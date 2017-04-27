import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from max.datacenter import DataCenter
import max.covar.covarutil as CovarUtil

logger = logging.getLogger(__name__)


class Covar:
    
    EIGVAL_THLD = 1e-5

    def __init__(self, df=pd.DataFrame(), window=90):
        logger.info('Initializing Covariance class')
        if 'date' not in df.columns:
            raise Exception('date column is needed in input data frame')
        if 'code' not in df.columns:
            raise Exception('code column is needed in input data frame')
        if 'return' not in df.columns:
            raise Exception('return column is needed in input data frame')
        self.ret = df.pivot(index='date', columns='code', values='return')
        logger.info('Time period from %s to %s', min(df.date), max(df.date))
        logger.info('# of dates: %i, # of stocks: %i', self.ret.shape[0], self.ret.shape[1])
        logger.info('Rolling window %i days', window)
        self.window = window
        self.ex_post_cor, self.ex_post_vol, self.ex_post_cov = self.get_cov_seq()
    
    def get_cov_seq(self):
        logger.info('Computing rolling %i-days correlation and volatility', self.window)
        [T, n] = self.ret.shape
        cor = np.array([CovarUtil.ex_post_cor(self.ret, self.window, x) for x in self.ret.index])
        vol = np.array([CovarUtil.ex_post_vol(self.ret, self.window, x) for x in self.ret.index])
        cov = np.array([CovarUtil.ex_post_cov(self.ret, self.window, x) for x in self.ret.index])
        return cor, vol, cov

    def get_ema_cov(self, alpha_cor, alpha_vol, save=False):
        logger.debug('Computing EMA covariance, alpha_cor = %.3f, alpha_vol = %.3f', alpha_cor, alpha_vol)
        T, n = self.ret.shape
        cor_ema = self.ex_post_cor.copy()
        vol_ema = self.ex_post_vol.copy()
        cov_ema = np.repeat(np.nan, T*n*n).reshape(T,n,n)
        cov_ema[self.window-1,:,:] = CovarUtil.nan_cov(cor_ema[self.window-1,:,:], vol_ema[self.window-1,:])
        for i in range(self.window, T):
            cor_ema[i,:,:] = CovarUtil.nan_ema(cor_ema[i-1,:,:], self.ex_post_cor[i,:,:], alpha_cor)
            vol_ema[i,:]   = CovarUtil.nan_ema(vol_ema[i-1,:], self.ex_post_vol[i,:], alpha_vol)
            cov_ema[i,:,:] = CovarUtil.nan_cov(cor_ema[i,:,:], vol_ema[i,:])
        if save:
            self.cor_ema = cor_ema
            self.vol_ema = vol_ema
            self.cov_ema = cov_ema
        return cov_ema
    
    def compute_log_likelihood(self, cov, ipos, jpos, lag=1):
        logger.debug('Computing log likelihood')
        T = self.ret.shape[0]
        X = np.nan_to_num(self.ret.values.copy())  # fill nan with 0
        loglik = 0
        cnt = 0
        for i in range(self.window, T-lag):
            covi = cov[i,:,:]
            nanidx = np.isnan(np.diag(covi))
            covi_obs = covi[~nanidx,:][:,~nanidx]
            if not np.any(np.isnan(covi_obs)):
                eigval, eigvec = np.linalg.eig(covi_obs)
                eigval[eigval<self.EIGVAL_THLD] = 0
                covi_det = np.prod(eigval[eigval>0])
                self.covdet_path[i,ipos,jpos] = covi_det
                cnt += 1
                logliki = -(np.sum(eigval>0)/2.0) * np.log(2*np.pi)
                logliki += -0.5 * np.log(covi_det)
                xvec = X[i+lag,~nanidx]
                zvec = xvec.dot(eigvec)[eigval > 0]
                logliki += -0.5 * np.sum(zvec * zvec * (1/eigval[eigval>0]))
                self.loglik_path[i,ipos,jpos] = logliki
                loglik += logliki
        return loglik / cnt
    
    def plot_heatmap(self, loglik):
        plt.figure()
        plt.imshow(loglik, cmap='hot')
        plt.colorbar()
        plt.show()
        
    def plot_likelihood_path(self, i, j):
        plt.figure()
        plot = pd.Series(self.loglik_path[:,i,j], index=self.ret.index).plot()
        plot.set_ylabel('Log Likelihood')
        plot.set_title('alpha_cor='+str(self.alpha_cor_candidates[i]))
        plt.show()
    
    def calibrate(self, alphas_cor, alphas_vol):
        logger.info('Calibrating EMA parameters')
        n_cor = np.size(alphas_cor)
        n_vol = np.size(alphas_vol)
        loglik = np.repeat(np.nan, n_cor*n_vol).reshape(n_cor, n_vol)
        T = self.ret.shape[0]
        self.loglik_path = np.repeat(np.nan, n_cor*n_vol*T).reshape(T,n_cor,n_vol)
        self.covdet_path = np.repeat(np.nan, n_cor*n_vol*T).reshape(T,n_cor,n_vol)
        for i in range(n_cor):
            for j in range(n_vol):
                cov = self.get_ema_cov(alphas_cor[i], alphas_vol[j])
                loglik[i, j] = self.compute_log_likelihood(cov, i, j)
        maxidx = np.argwhere(loglik.max() == loglik)
        self.plot_heatmap(loglik)
        self.alpha_cor = alphas_cor[maxidx[0][0]]
        self.alpha_vol = alphas_vol[maxidx[0][1]]
        self.get_ema_cov(self.alpha_cor, self.alpha_vol, save=True)
        self.loglik = loglik
        self.alpha_cor_candidates = alphas_cor
        self.alpha_vol_candidates = alphas_vol
        self.plot_likelihood_path(maxidx[0][0], maxidx[0][1])
        logger.info('Optimal soln: alpha_cor = %.2f, alpha_vol = %.2f, avg loglik = %.2f',
                    self.alpha_cor, self.alpha_vol, loglik.max())

    def plot_ema_vol(self, i):
        plt.figure()
        pd.Series(self.ex_post_vol[self.window:,i], index=self.ret.index[self.window:]).plot(style='b', legend=True)
        plot = pd.Series(self.vol_ema[self.window:,i], index=self.ret.index[self.window:]).plot(style='g', legend=True)
        plt.legend(['Ex post', 'EMA'])
        plot.set_ylabel('Daily Vol')
        plt.show()
    
    def plot_ema_cor(self, i, j):
        plt.figure()
        pd.Series(self.ex_post_cor[self.window:,i,j], index=self.ret.index[self.window:]).plot(style='b', legend=True)
        plot = pd.Series(self.cor_ema[self.window:,i,j], index=self.ret.index[self.window:]).plot(style='g', legend=True)
        plot.set_ylabel('Daily Return Correlation')
        plt.legend(['Ex post', 'EMA'])
        plt.show()
    
    def to_csv(self, cov, folder):
        path = os.path.join(DataCenter.get_path('covariance'), folder)
        cols = self.ret.columns
        T = self.ret.shape[0]
        for i in range(self.window, T):
            d = pd.DataFrame(cov[i,:,:], columns=cols)
            d['code'] = cols
            d = d[['code']+cols.tolist()]
            date = self.ret.index[i]
            dirname = os.path.join(path, str(date.year))
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            filename = os.path.join(dirname, date.strftime('%Y%m%d')+'.csv')
            logger.info('Saving file to %s', filename)
            d.to_csv(filename, index=False)
