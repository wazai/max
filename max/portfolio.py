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

    @member code: array-like of shape = [n], stocks in the portfolio
    @member share: array-like of shape = [n], quantities of corresponding stocks
    @member df: DataFrame of shape [n, 2] with columns code and share
    @member date: current date
    @member price: current price data
    """

    def __init__(self, code, share):
        logger.info('Creating portfolio with %i stocks in the universe', len(code))
        if len(code) != len(share):
            logger.error('Number of stocks and shares does not match')
        if 'cash' not in code:
            code.append('cash')
            share.append(0.0)
        self.universe = np.array(code)
        self.share = np.array(share)
        self.df = pd.DataFrame({'code': code, 'share': share})
        self.date = ''
        self.px = pd.DataFrame()
        logger.debug('Portfolio created')

    # portfolio info
    # --------------

    def get_covar(self, date, dc):
        cov = dc.get_covar(date).copy()
        missing = self.df[~self.df['code'].isin(cov.columns)].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'covariance'
        cov = cov.loc[self.universe, self.universe]
        return cov, missing

    def get_price(self, date, dc=None):
        if self.date == date and not self.px.empty:
            px = self.px
        else:
            px = dc.get_price(date)
            px = px[px['code'].isin(self.universe)]
            px = px.reset_index()[['code', 'close']]
            px.loc[px.shape[0]] = ['cash', 1.0]
            self.date = date
            self.px = px
        missing = self.df[~self.df['code'].isin(px['code'])].copy()
        missing.reset_index(inplace=True)
        missing['type'] = 'price'
        px = pd.merge(self.df, px, on='code', how='left')
        px = px.set_index('code')
        return px, missing

    def get_position(self, date, dc):
        px, _ = self.get_price(date, dc)
        position = px['close'] * px['share']
        return position.values

    def get_share(self, position, date=None, dc=None):
        px, _ = self.get_price(date, dc)
        share = position / px['close'].values
        return share

    def get_alpha(self, date, alpha):
        alpha_values = alpha.get_alpha(date)
        alpha_df = pd.DataFrame({'code': alpha.universe, 'alpha': alpha_values})
        alpha_df.set_index('code', inplace=True)
        if 'cash' in alpha_df.index:
            if alpha_df.loc['cash'].values.item() != 0:
                logger.warning('Alpha of cash is not zero, set it to zero')
                alpha_df.loc['cash'] = 0.0
        else:
            alpha_df.loc['cash'] = 0.0
        return alpha_df.loc[self.universe].values

    def get_port_return(self, start_date, end_date):
        # TODO compute portfolio return for a time period
        pass

    # portfolio analytics
    # -------------------

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

    # portfolio rebalance
    # -------------------

    def trade(self, share_trade):
        share_trade = np.nan_to_num(share_trade)
        self.share = self.share + share_trade
        self.df['share'] = self.share
