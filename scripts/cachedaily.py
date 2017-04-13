"""
Get daily price data from tushare

@author: jingweiwu
"""

import sys
import os
maxpath = os.path.join(os.environ['HOME'], 'max')
sys.path.insert(0, os.path.join(maxpath, 'datacache'))
sys.path.insert(0, os.path.join(maxpath, 'src'))
from cacheutils import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')


codelist = get_code_list(['sz50', 'hs300', 'zz500'])

startdate = '2017-01-01'
enddate = '2017-01-10'

dc = DataCenter('20150101', '20150401')
DataCenter.get_business_days_within('20151231', 3, 5)

px = get_daily_price(codelist[:10], startdate, enddate)

px = enrich(px)

update_business_days()