import logging
import pandas as pd

# from max.alpha import Alpha
from max.datacenter.datacenter import DataCenter
from max.portfolio import Portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)27s  %(levelname)s  %(message)s')

start_date = '2015-01-01'
end_date = '2016-12-31'
dc = DataCenter(start_date, end_date)
date = '2016-12-30'

# portfolio with missing code
# ----------------------------

code = ['601818', '600000', '600028', 'some shit']
share = [20, 50, 30, 100]
p = Portfolio(code, share)

port_summary, missing_info = p.summary(date, dc)
print(pd.DataFrame({'Metric': list(port_summary.keys()), 'Value': list(port_summary.values())}))
print(missing_info)
print(p.get_position(date, dc))

# portfolio with cash
# --------------------

code = ['601818', '600000', '600028', 'some shit', 'cash']
share = [20, 50, 30, 100, 10000]
p = Portfolio(code, share)

port_summary, missing_info = p.summary(date, dc)
print(pd.DataFrame({'Metric': list(port_summary.keys()), 'Value': list(port_summary.values())}))
print(missing_info)
print(p.get_position(date, dc))
