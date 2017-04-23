import pandas as pd
import numpy as np
import os
import datetime
import logging
import sys
import datetime as dt

logger = logging.getLogger(__name__)

class DataCenter(object):

    @staticmethod
    def get_datapath():
        paths = dict()
        if sys.platform == 'win32':
            paths['data'] = os.path.join(os.environ['HOMEPATH'], 'Dropbox\\HW\\data')
        else:
            paths['data'] = os.path.join(os.environ['HOME'], 'Dropbox/HW/data')
        paths['marketdata'] = os.path.join(paths['data'], 'marketdata')
        paths['dailycache'] = os.path.join(paths['marketdata'], 'dailycache')
        paths['misc'] = os.path.join(paths['marketdata'], 'misc')
        paths['univ'] = os.path.join(paths['marketdata'], 'univ')
        return paths

    @staticmethod
    def get_business_days(bday_t = 'business'):
        logger.info('Loading business days')
        paths = DataCenter.get_datapath()
        bdays = pd.read_csv(os.path.join(paths['misc'], bday_t+'_days.csv'),dtype={'date':str})
        return bdays['date'].values

    @staticmethod
    def get_business_days_within(yyyymmdd, nbackward, nforward):
        dates = DataCenter.get_business_days()
        pos = np.argmax(dates>yyyymmdd)
        if pos == 0:
            logger.warning('Input date outside the range of current business days')
            pos = len(dates)
        nforward = min(nforward, len(dates)-pos)
        return dates[pos-nbackward:pos+nforward]

    def get_business_days_start_end(self, start_date, end_date):
        res = self.price.set_index('date')
        return res[start_date:end_date].index.unique()

    def __init__(self, startdate='2010-01-01', enddate=datetime.date.today().strftime('%Y-%m-%d')):
        logger.info('Initializing data center')
        paths = self.get_datapath()
        self.paths = paths
        self.business_days = self.get_business_days(bday_t='file')
        self._load_daily_price(startdate, enddate)

        # Get start and end dates of DataCenter
        if self.price.empty:
            self.start_date = ''
            self.end_date = ''
        else:
            self.start_date = self.price['date'].min()
            self.end_date = self.price['date'].max()

        self.univ_dict = dict()
        univ_filenames = os.listdir(paths['univ'])
        for fn in univ_filenames:
            logger.info('Load univ file %s', fn)
            self.univ_dict[fn[:-4]] = pd.read_csv(os.path.join(paths['univ'],fn),dtype={'code':str})
        logger.info('Finish initializing data center')
    
    def _load_daily_price(self, startdate, enddate):
        logger.info('Loading daily cache from %s to %s', startdate, enddate)
        bdays = self.business_days[(self.business_days>=startdate) & (self.business_days<=enddate)]
        bdays_list = bdays.tolist()
        bdays_list = map(lambda x: x[:4] + x[5:7] + x[8:10], bdays_list)
        filenames = [os.path.join(self.paths['dailycache'], x[:4], x+'.csv') for x in bdays_list]
        px_list = [pd.read_csv(x, dtype={'date': dt.datetime, 'code': str}, parse_dates=[0]) for x in filenames]
        if not px_list:
            logger.warning('Empty price cache')
            pxcache = pd.DataFrame()
        else:
            pxcache = pd.concat(px_list)
            pxcache = pxcache.reset_index(drop=True)
            logger.info('Daily cache loaded')
        self.price = pxcache

    def load_codes_return(self, codes, start_date, end_date):
        df = self.price[['date', 'code', 'return']]
        df = df[df['code'].isin(codes)]
        pivot = df.pivot_table(values='return', index=['date'], columns=['code'])
        return pivot[start_date:end_date]


