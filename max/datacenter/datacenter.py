"""
Data Center

Load daily price cache, univ, risk etc 
"""

import pandas as pd
import os
import logging

from max.datacenter.datapath import DataPath
from max.datacenter.businessday import BusinessDay

logger = logging.getLogger(__name__)


class DataCenter(object):

    def __init__(self, start_date, end_date, cov_model='ex_post_180'):
        logger.info('Initializing data center')
        dp = DataPath()
        self.paths = dp.path
        bday = BusinessDay()
        self.business_days = bday.get_business_days(bday_t='file')
        self._load_daily_price(start_date, end_date)
        self._load_daily_covar(start_date, end_date, cov_model)

        if self.price.empty:
            self.start_date = ''
            self.end_date = ''
            self.code_name_dict = dict()
        else:
            self.start_date = self.price.index.min()
            self.end_date = self.price.index.max()
            self.code_name_dict = dict(zip(self.price['code'], self.price['name']))

        self.univ_dict = dict()
        univ_file_names = os.listdir(self.paths['univ'])
        for fn in univ_file_names:
            logger.info('Load univ file %s', fn)
            self.univ_dict[fn[:-4]] = pd.read_csv(os.path.join(self.paths['univ'], fn), dtype={'code': str})
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

    @staticmethod
    def _load_cov_append_date(i, file_names, bdays):
        cov = pd.read_csv(file_names[i], dtype={'code': str})
        cov.index = pd.to_datetime([bdays[i]] * cov.shape[0], format='%Y-%m-%d')
        return cov

    def _load_daily_covar(self, start_date, end_date, model='ex_post_180'):
        logger.info('Loading %s covariance estimate from %s to %s', model, start_date, end_date)
        covar_dir = os.path.join(self.paths['covariance'], model)
        if not os.path.exists(covar_dir):
            logger.error('Covar model %s cannot be found', model)
        bdays = self.business_days[(self.business_days >= start_date) & (self.business_days <= end_date)]
        bdays_list = bdays.tolist()
        bdays_list = map(lambda x: x[:4] + x[5:7] + x[8:10], bdays_list)
        file_names = [os.path.join(covar_dir, x[:4], x + '.csv') for x in bdays_list]
        cov_list = [self._load_cov_append_date(i, file_names, bdays) for i in range(len(file_names))]
        if not cov_list:
            logger.warning('Empty covariance series')
            covcache = pd.DataFrame()
        else:
            covcache = pd.concat(cov_list)
            logger.info('Daily covariance loaded')
        self.cov = covcache

    def load_codes_return(self, codes, start_date, end_date):
        if self.price.empty:
            return pd.DataFrame()
        df = self.price[['code', 'return']]
        df = df[df['code'].isin(codes)]
        pivot = df.pivot(values='return', columns='code')
        return pivot[start_date:end_date]

    def load_codes_column(self, codes, column, start_date, end_date):
        if self.price.empty:
            return pd.DataFrame()
        df = self.price[['code', column]]
        df = df[df['code'].isin(codes)]
        pivot = df.pivot(values=column, columns='code')
        return pivot[start_date:end_date]

    def get_business_days_start_end(self, start_date, end_date):
        return self.price[start_date:end_date].index.unique()
