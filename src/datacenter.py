"""
Data Center

@author: jingweiwu
"""

import pandas as pd
import os
import datetime
import logging

logger = logging.getLogger(__name__)

class DataCenter:
    
    def __init__(self, startdate = '20100101', enddate = datetime.date.today().strftime('%Y%m%d')):
        logger.info('Initializing data center')
        self.maxpath = os.path.join(os.environ['HOME'], 'max')
        self.datapath = os.path.join(['HOME'], 'Dropbox/HW/data')
        self.business_days = pd.read_csv(os.path.join(self.datapath,'marketdata/misc/business_days.csv'),dtype={'date':object})
        self.price = self.load_daily_price(startdate, enddate)
        self.univ_dict = dict()
        univ_filenames = os.listdir(os.path.join(self.datapath, 'marketdata/univ'))
        for fn in univ_filenames:
            self.univ_dict[fn[:-4]] = pd.read_csv(os.path.join(self.datapath,'marketdata/univ',fn))
        logger.info('Finish initializing data center')
    
    def load_daily_price(self, startdate, enddate):
        logger.info('Loading daily cache from %s to %s', startdate, enddate)
        bdays = self.business_days[(self.business_days['date']>=startdate) & (self.business_days['date']<=enddate)]
        bdays_list = bdays['date'].tolist()
        filenames = [os.path.join(self.datapath, 'marketdata/dailycache', x[:4], x+'.csv') for x in bdays_list]
        px_list = [pd.read_csv(x) for x in filenames]
        return pd.concat(px_list)
        logger.info('Initializing data center')