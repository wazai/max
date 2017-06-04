"""
Portfolio

Assets and quantities in portfolio, get alpha/risk of portfolio, update portfolio
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Portfolio(object):
    """ Portfolio class
    Portfolio does not know date information, it only has the code and quantity.
    To get the position/risk date needs to be given.

    @param code: array-like of shape = [n], stocks in the portfolio
    @param share: array-like of shape = [n], quantities of corresponding stocks
    """

    def __init__(self, code, share):
        logger.info('Creating portfolio with %i stocks in the universe', len(code))
        if len(code) != len(share):
            logger.error('Number of stocks and shares does not match')
        self.universe = np.array(code)
        self.share = np.array(share)
        self.df = pd.DataFrame({'code': code, 'share': share})
        logger.debug('Portfolio created')

    def get_covar(self, date, dc):
        cov = dc.get_covar(date).copy()
        missing = self.df[~self.df['code'].isin(cov.columns)].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'covariance'
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
        px = dc.get_price(date)
        px = px.reset_index()[['code', 'close']]
        px.loc[px.shape[0]] = ['cash', 1.0]
        missing = self.df[~self.df['code'].isin(px['code'])].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'price'
        px = pd.merge(self.df, px, on='code', how='left')
        px = px.set_index('code')
        return px, missing

    def get_position(self, date, dc):
        px, _ = self.get_price(date, dc)
        position = px['close']*px['share']
        return position.values

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

    def get_port_return(self, start_date, end_date):
        # TODO compute portfolio return for a time period
        pass

    def get_alpha(self, date, alpha):
        # TODO get alpha of stocks in the portfolio
        pass
