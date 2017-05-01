"""
Data Center

Load daily price cache, univ, risk etc 
"""

import pandas as pd
import numpy as np
import os
import datetime
import logging
import sys

logger = logging.getLogger(__name__)


class DataCenter(object):

    @staticmethod
    def get_all_paths():
        paths = dict()
        if sys.platform == 'win32':
            paths['data'] = os.path.join(os.environ['HOMEPATH'], 'Dropbox\\HW\\data')
        else:
            paths['data'] = os.path.join(os.environ['HOME'], 'Dropbox/HW/data')
        paths['marketdata'] = os.path.join(paths['data'], 'marketdata')
        paths['dailycache'] = os.path.join(paths['marketdata'], 'dailycache')
        paths['misc'] = os.path.join(paths['marketdata'], 'misc')
        paths['univ'] = os.path.join(paths['marketdata'], 'univ')
        paths['covariance'] = os.path.join(paths['data'], 'covariance')
        return paths
    
    @staticmethod
    def get_path(name):
        paths = DataCenter.get_all_paths()
        return paths[name]

    @staticmethod
    def get_business_days(bday_t='business'):
        logger.info('Loading business days')
        paths = DataCenter.get_all_paths()
        bdays = pd.read_csv(os.path.join(paths['misc'], bday_t+'_days.csv'),dtype={'date':str})
        return bdays['date'].values

    @staticmethod
    def get_business_days_within(date, n_backward, n_forward):
        dates = DataCenter.get_business_days()
        pos = np.argmax(dates > date)
        if pos == 0:
            logger.warning('Input date outside the range of current business days')
            pos = len(dates)
        n_forward = min(n_forward, len(dates)-pos)
        return dates[pos-n_backward:pos+n_forward]

    def get_business_days_start_end(self, start_date, end_date):
        res = self.price.set_index('date')
        return res[start_date:end_date].index.unique()

    def __init__(self, start_date='2010-01-01', end_date=datetime.date.today().strftime('%Y-%m-%d')):
        logger.info('Initializing data center')
        paths = self.get_all_paths()
        self.paths = paths
        self.business_days = self.get_business_days(bday_t='file')
        self._load_daily_price(start_date, end_date)

        if self.price.empty:
            self.start_date = ''
            self.end_date = ''
        else:
            self.start_date = self.price.index.min()
            self.end_date = self.price.index.max()

        self.univ_dict = dict()
        univ_file_names = os.listdir(paths['univ'])
        for fn in univ_file_names:
            logger.info('Load univ file %s', fn)
            self.univ_dict[fn[:-4]] = pd.read_csv(os.path.join(paths['univ'], fn), dtype={'code': str})
        logger.info('Finish initializing data center')
    
    def _load_daily_price(self, start_date, end_date):
        logger.info('Loading daily cache from %s to %s', start_date, end_date)
        if len(start_date) == 8:
            logger.warning('Use yyyy-mm-dd format for start and end date')
        bdays = self.business_days[(self.business_days >= start_date) & (self.business_days <= end_date)]
        bdays_list = bdays.tolist()
        bdays_list = map(lambda x: x[:4] + x[5:7] + x[8:10], bdays_list)
        file_names = [os.path.join(self.paths['dailycache'], x[:4], x+'.csv') for x in bdays_list]
        px_list = [pd.read_csv(x, index_col='date', dtype={'code': str}, parse_dates=[0]) for x in file_names]
        if not px_list:
            logger.warning('Empty price cache')
            pxcache = pd.DataFrame()
        else:
            pxcache = pd.concat(px_list)
            logger.info('Daily cache loaded')
        self.price = pxcache

    def load_codes_return(self, codes, start_date, end_date):
        if self.price.empty:
            return pd.DataFrame()
        df = self.price[['code', 'return']]
        df = df[df['code'].isin(codes)]
        pivot = df.pivot(values='return', columns='code')
        return pivot[start_date:end_date]

