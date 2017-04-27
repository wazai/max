from sys import argv

if len(argv) != 3:
    raise Exception('Incorrect usage. Please run as\n\t python cachedaily.py [start-date] [end-date]')
script, startdate, enddate = argv

if len(startdate)!=10 or len(enddate)!=10:
    raise Exception('Use yyyy-mm-dd format for startdate and enddate')

import sys
import os
maxpath = os.path.join(os.environ['HOME'], 'max')
sys.path.insert(0, os.path.join(maxpath, 'datacache'))
sys.path.insert(0, os.path.join(maxpath, 'max'))
from cacheutils import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

codelist = get_code_list(['sz50', 'hs300', 'zz500'])

pxorig = get_daily_price(codelist, startdate, enddate)

px = enrich(pxorig)

save_price_to_csv(px)

update_file_days()