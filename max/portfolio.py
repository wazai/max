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

    Members
    :param code: array-like of shape = [n], stocks in the portfolio
    :param share: array-like of shape = [n], quantities of corresponding stocks
    :param dc: DataCenter object, used to get price and covariance data
    """

    def __init__(self, code, share, dc):
        logger.info('Creating portfolio with %i stocks in the universe', len(code))
        if len(code) != len(share):
            logger.error('Number of stocks and shares does not match')
        self.universe = np.array(code)
        self.share = np.array(share)
        self.df = pd.DataFrame({'code': code, 'share': share})
        self.price = dc.price
        self.cov = dc.cov
        logger.info('Portfolio created')

    def get_covar(self, date):
        try:
            cov = self.cov.loc[date, :].copy()
        except KeyError:
            raise Exception('date not found in datacenter cov data')
        missing = self.df[~self.df['code'].isin(cov['code'])].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'covariance'
        cov = cov.set_index('code')
        cov = cov.loc[self.universe, self.universe]
        return cov, missing

    def get_port_risk(self, date):
        cov, missing_cov = self.get_covar(date)
        px, missing_px = self.get_price(date)
        missing = pd.concat([missing_cov, missing_px])
        px['position'] = px['share'] * px['close']
        w = px.loc[self.universe, 'position'].values
        sigma = cov.loc[self.universe, self.universe].values
        w = np.nan_to_num(w)
        sigma = np.nan_to_num(sigma)
        risk = w.dot(sigma).dot(w)
        return risk, missing

    def get_price(self, date):
        try:
            px = self.price.loc[date, :].copy()
        except KeyError:
            raise Exception('date not found in datacenter price data')
        missing = self.df[~self.df['code'].isin(px['code'])].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'price'
        px = px[['code', 'close']]
        px = pd.merge(self.df, px, on='code', how='left')
        px = px.set_index('code')
        return px, missing

    def get_position(self, date):
        px, _ = self.get_price(date)
        position = px['close']*np.abs(px['share'])
        return position.values

    def summary(self, date):
        stats = dict()

        stats['Universe Size'] = len(self.universe)
        stats['Num of Stocks Hold'] = sum(self.share != 0)
        stats['Num of Stocks Long'] = sum(self.share > 0)
        stats['Num of Stocks Short'] = sum(self.share < 0)

        px, _ = self.get_price(date)
        stats['Notional Gross'] = np.nansum(px['close']*np.abs(px['share']))
        stats['Notional Net'] = np.nansum(px['close']*px['share'])
        stats['Notional Long'] = np.nansum(px[px['share'] > 0]['close'] * px[px['share'] > 0]['share'])
        stats['Notional Short'] = np.nansum(px[px['share'] < 0]['close'] * px[px['share'] < 0]['share'])

        risk, missing = self.get_port_risk(date)
        stats['Risk'] = risk
        stats['Pct Risk'] = risk / stats['Notional Gross']

        return stats, missing

    def get_port_return(self, start_date, end_date):
        # TODO compute portfolio return for a time period
        pass

    def get_alpha(self, date, alpha):
        # TODO get alpha of stocks in the portfolio
        pass
