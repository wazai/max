"""
Portfolio

Assets and quantities in portfolio, get alpha/risk of portfolio, update portfolio
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Portfolio(object):

    def __init__(self, code, share):
        logger.info('Creating portfolio with %i stocks in the universe', len(code))
        if len(code) != len(share):
            logger.error('Number of stocks and shares does not match')
        self.universe = np.array(code)
        self.share = np.array(share)
        self.df = pd.DataFrame({'code': code, 'share': share})
        logger.info('Portfolio created')

    def get_covar(self, date, dc):
        try:
            cov = dc.cov.loc[date, :].copy()
        except KeyError:
            raise Exception('date not found in datacenter cov data')
        missing = self.df[~self.df['code'].isin(cov['code'])].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'covariance'
        cov = cov.set_index('code')
        cov = cov.loc[self.universe, self.universe]
        return cov, missing

    def get_port_risk(self, date, dc):
        cov, missing_cov = self.get_covar(date, dc)
        px, missing_px = self.get_price(date, dc)
        missing = pd.concat([missing_cov, missing_px])
        px['position'] = px['share'] * px['close']
        w = px.loc[self.universe, 'position'].values
        sigma = cov.loc[self.universe, self.universe].values
        w = np.nan_to_num(w)
        sigma = np.nan_to_num(sigma)
        risk = w.dot(sigma).dot(w)
        return risk, missing

    def get_price(self, date, dc):
        try:
            px = dc.price.loc[date, :].copy()
        except KeyError:
            raise Exception('date not found in datacenter price data')
        missing = self.df[~self.df['code'].isin(px['code'])].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'price'
        px = px[['code', 'close']]
        px = pd.merge(self.df, px, on='code', how='left')
        px = px.set_index('code')
        return px, missing

    def summary(self, date, dc):
        stats = dict()

        stats['Universe Size'] = len(self.universe)
        stats['Num of Stocks Hold'] = sum(self.share != 0)
        stats['Num of Stocks Long'] = sum(self.share > 0)
        stats['Num of Stocks Short'] = sum(self.share < 0)

        px, _ = self.get_price(date, dc)
        stats['Notional Gross'] = np.nansum(px['close']*np.abs(px['share']))
        stats['Notional Net'] = np.nansum(px['close']*px['share'])
        stats['Notional Long'] = np.nansum(px[px['share'] > 0]['close'] * px[px['share'] > 0]['share'])
        stats['Notional Short'] = np.nansum(px[px['share'] < 0]['close'] * px[px['share'] < 0]['share'])

        risk, missing = self.get_port_risk(date, dc)
        stats['Risk'] = risk
        stats['Pct Risk'] = risk / stats['Notional Gross']

        return stats, missing
