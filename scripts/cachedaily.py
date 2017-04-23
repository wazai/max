from sys import argv

if len(argv) != 3:
    raise Exception('Incorrect usage. Please run as\n\t python cachedaily.py [start-date] [end-date]')
script, startdate, enddate = argv

import sys
import os
maxpath = os.path.join(os.environ['HOME'], 'max')
sys.path.insert(0, os.path.join(maxpath, 'datacache'))
sys.path.insert(0, os.path.join(maxpath, 'src'))
from cacheutils import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

codelist = get_code_list(['sz50', 'hs300', 'zz500'])

pxorig = get_daily_price(codelist, startdate, enddate)

px = enrich(pxorig)

save_price_to_csv(px)

#update_business_days()