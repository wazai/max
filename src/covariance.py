"""
Covariance class

@author: jingweiwu
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Covariance:

    def ex_post_cor(self, date):
        d = self.ret[self.ret.index <= date]
        [T, n] = d.shape
        if T < self.window:
            return np.repeat(np.nan, n*n).reshape(n, n)
        else:
            return d.tail(self.window).corr().values
    
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
        self.window = window
        self.get_cor_seq()
    
    def get_cor_seq(self):
        logger.info('Computing rolling %i-days correlation and volatility', self.window)
        [T, n] = self.ret.shape
        self.cor = np.array([self.ex_post_cor(x) for x in self.ret.index])
        self.vol = np.array([self.ex_post_vol(x) for x in self.ret.index])

    def get_ema_cov(self, alpha_cor, alpha_vol):
        logger.info('Computing EMA covariance, alpha_cor = %.3f, alpha_vol = %.3f', alpha_cor, alpha_vol)
        T, n = self.ret.shape
        cor_ema = self.cor.copy()
        vol_ema = self.vol.copy()
        cov_ema = np.repeat(np.nan, T*n*n).reshape(T,n,n)
        cov_ema[self.window-1,:,:] = self.nancov(cor_ema[self.window-1,:,:], vol_ema[self.window-1,:])
        for i in range(self.window, T):
            cor_ema[i,:,:] = self.nanema(cor_ema[i-1,:,:], self.cor[i,:,:], alpha_cor)
            vol_ema[i,:]   = self.nanema(vol_ema[i-1,:], self.vol[i,:], alpha_vol)
            cov_ema[i,:,:] = self.nancov(cor_ema[i,:,:], vol_ema[i,:])
        return cov_ema

    def compute_log_likelihood(self, cov, lag=1):
        logger.info('Computing log likelihood')
        T, _ = self.ret.shape
        X = np.nan_to_num(self.ret.values.copy()) # fill nan with 0
        loglik = 0
        cnt = 0
        for i in range(self.window, T-lag):
            covi = cov[i,:,:]
            nanidx = np.isnan(np.diag(covi))
            covi_obs = covi[~nanidx,:][:,~nanidx]
            covi_det = np.linalg.det(covi_obs)
            if not np.isnan(covi_det) and abs(covi_det) != 0:
                cnt += 1
                loglik += -(np.sum(~nanidx)/2.0) * np.log(2*np.pi)
                loglik += -0.5 * np.log(np.abs(covi_det))
                xvec = X[i+lag,~nanidx]
                loglik += -0.5 * xvec.dot(np.linalg.inv(covi_obs)).dot(xvec)
        return loglik / cnt
    
    def calibrate(self, alphas_cor, alphas_vol):
        logger.info('Calibrating EMA parameters')
        n_cor = np.size(alphas_cor)
        n_vol = np.size(alphas_vol)
        pdfs = np.repeat(np.nan, n_cor*n_vol).reshape(n_cor, n_vol)
        for i in range(n_cor):
            for j in range(n_vol):
                cov = self.get_ema_cov(alphas_cor[i], alphas_vol[j])
                pdfs[i, j] = self.compute_log_likelihood(cov)
        maxidx = np.argwhere(pdfs.max() == pdfs)
        return alphas_cor[maxidx[0][0]], alphas_vol[maxidx[0][1]], pdfs
