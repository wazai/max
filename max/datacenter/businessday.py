import logging
import pandas as pd
import numpy as np
import os

from max.datacenter.datapath import DataPath

logger = logging.getLogger(__name__)


class BusinessDay(object):

    def __init__(self):
        logger.info('Creating business days class')
        paths = DataPath()
        b_days = pd.read_csv(os.path.join(paths.get_path('misc'), 'business_days.csv'), dtype={'date': str})
        f_days = pd.read_csv(os.path.join(paths.get_path('misc'), 'file_days.csv'), dtype={'date': str})
        self.days = {'business': b_days['date'].values, 'file': f_days['date'].values}

    def get_business_days(self, bday_t='business'):
        logger.info('Loading %s days', bday_t)
        return self.days[bday_t]

    def get_business_days_within(self, date, n_backward, n_forward):
        dates = self.get_business_days()
        pos = np.argmax(dates > date)
        if pos == 0:
            logger.warning('Input date outside the range of current business days')
            pos = len(dates)
        n_forward = min(n_forward, len(dates) - pos)
        return dates[pos - n_backward:pos + n_forward]
