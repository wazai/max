import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Covariance:
    
    EIGVAL_THLD = 1e-5

    def ex_post_cor(self, date):
        d = self.ret[self.ret.index <= date]
        [T, n] = d.shape
        if T < self.window:
            return np.repeat(np.nan, n*n).reshape(n, n)
        else:
            return d.tail(self.window).corr().values
    
    def ex_post_cov(self, date):
        d = self.ret[self.ret.index <= date]
        [T, n] = d.shape
        if T < self.window:
            return np.repeat(np.nan, n*n).reshape(n, n)
        else:
            return d.tail(self.window).cov().values
    
    def ex_post_vol(self, date):
        d = self.ret[self.ret.index <= date]
        [T, n] = d.shape
        if T < self.window:
            return np.repeat(np.nan, n)
        else:
            return d.tail(self.window).std().values
    
    @staticmethod
    def nancov(cor, vol):
        voldiag = np.nan_to_num(np.diag(vol))
        cov = voldiag.dot(np.nan_to_num(cor)).dot(voldiag)
        cov[np.isnan(cor)] = np.nan
        return cov
    
    @staticmethod
    def nanema(s0, y1, alpha):
        s1 = alpha*y1 + (1-alpha)*s0
        s1[np.isnan(s0)] = y1[np.isnan(s0)]
        return s1

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
        self.get_cor_seq()
    
    def get_cor_seq(self):
        logger.info('Computing rolling %i-days correlation and volatility', self.window)
        [T, n] = self.ret.shape
        self.cor = np.array([self.ex_post_cor(x) for x in self.ret.index])
        self.vol = np.array([self.ex_post_vol(x) for x in self.ret.index])
        self.cov = np.array([self.ex_post_cov(x) for x in self.ret.index])

    def get_ema_cov(self, alpha_cor, alpha_vol, save=False):
        logger.debug('Computing EMA covariance, alpha_cor = %.3f, alpha_vol = %.3f', alpha_cor, alpha_vol)
        T, n = self.ret.shape
        cor_ema = self.cor.copy()
        vol_ema = self.vol.copy()
        cov_ema = np.repeat(np.nan, T*n*n).reshape(T,n,n)
        cov_ema[self.window-1,:,:] = self.nancov(cor_ema[self.window-1,:,:], vol_ema[self.window-1,:])
        for i in range(self.window, T):
            cor_ema[i,:,:] = self.nanema(cor_ema[i-1,:,:], self.cor[i,:,:], alpha_cor)
            vol_ema[i,:]   = self.nanema(vol_ema[i-1,:], self.vol[i,:], alpha_vol)
            cov_ema[i,:,:] = self.nancov(cor_ema[i,:,:], vol_ema[i,:])
        if save:
            self.cor_ema = cor_ema
            self.vol_ema = vol_ema
            self.cov_ema = cov_ema
        return cov_ema

    def compute_log_likelihood_old(self, cov, ipos, jpos, lag=1):
        logger.debug('Computing log likelihood')
        T = self.ret.shape[0]
        X = np.nan_to_num(self.ret.values.copy()) # fill nan with 0
        loglik = 0
        cnt = 0
        for i in range(self.window, T-lag):
            covi = cov[i,:,:]
            nanidx = np.isnan(np.diag(covi))
            covi_obs = covi[~nanidx,:][:,~nanidx]
            eigval, eigvec = np.linalg.eig(covi_obs)
            covi_det = np.linalg.det(covi_obs)
            self.covdet_path[ipos, jpos, i] = covi_det
            if not np.isnan(covi_det) and abs(covi_det) != 0:
                cnt += 1
                logliki = -(np.sum(~nanidx)/2.0) * np.log(2*np.pi)
                logliki += -0.5 * np.log(np.abs(covi_det))
                xvec = X[i+lag,~nanidx]
                logliki += -0.5 * xvec.dot(np.linalg.inv(covi_obs)).dot(xvec)
                self.loglik_path[ipos,jpos,i] = logliki
                self.covinvdet[ipos,jpos,i] = np.linalg.det(np.linalg.inv(covi_obs))
                self.xTinvSigmax[ipos,jpos,i] = xvec.dot(np.linalg.inv(covi_obs)).dot(xvec)
                loglik += logliki
        return loglik / cnt
    
    def compute_log_likelihood(self, cov, ipos, jpos, lag=1):
        logger.debug('Computing log likelihood')
        T = self.ret.shape[0]
        X = np.nan_to_num(self.ret.values.copy()) # fill nan with 0
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
                self.covdet_path[ipos, jpos, i] = covi_det
                cnt += 1
                logliki = -(np.sum(eigval>0)/2.0) * np.log(2*np.pi)
                logliki += -0.5 * np.log(covi_det)
                xvec = X[i+lag,~nanidx]
                zvec = xvec.dot(eigvec)[eigval > 0]
                logliki += -0.5 * np.sum(zvec * zvec * (1/eigval[eigval>0]))
                self.loglik_path[ipos,jpos,i] = logliki
                loglik += logliki
        return loglik / cnt
    
    def plot_heatmap(self, loglik):
        plt.figure()
        plt.imshow(loglik, cmap='hot')
        plt.colorbar()
        plt.show()
    
    def calibrate(self, alphas_cor, alphas_vol):
        logger.info('Calibrating EMA parameters')
        n_cor = np.size(alphas_cor)
        n_vol = np.size(alphas_vol)
        loglik = np.repeat(np.nan, n_cor*n_vol).reshape(n_cor, n_vol)
        T = self.ret.shape[0]
        self.loglik_path = np.repeat(np.nan, n_cor*n_vol*T).reshape(n_cor, n_vol,T)
        self.covdet_path = np.repeat(np.nan, n_cor*n_vol*T).reshape(n_cor, n_vol,T)
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
        logger.info('Optimal soln: alpha_cor = %.2f, alpha_vol = %.2f, avg loglik = %.2f',
                    self.alpha_cor, self.alpha_vol, loglik.max())

    def plot_ema_vol(self, i):
        plt.figure()
        plot = pd.Series(self.vol[:,i], index=self.ret.index).plot(style='b', legend=True)
        plot = pd.Series(self.vol_ema[:,i], index=self.ret.index).plot(style='g', legend=True)
        plt.legend(['Empirical', 'EMA'])
        plot.set_ylabel('Daily Vol')
        plt.show()
    
    def plot_ema_cor(self, i, j):
        plt.figure()
        plot = pd.Series(self.cor[:,i,j], index=self.ret.index).plot(style='b', legend=True)
        plot = pd.Series(self.cor_ema[:,i,j], index=self.ret.index).plot(style='g', legend=True)
        plt.legend(['Empirical', 'EMA'])
        plot.set_ylabel('Daily Correlation')
        plt.show()