import logging

from max.alpha import Alpha
from max.datacenter.datacenter import DataCenter
from max.portfolio import Portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)27s  %(levelname)s  %(message)s')

start_date = '2015-01-01'
end_date = '2016-12-31'
dc = DataCenter(start_date, end_date)

code = ['601818', '600000', '600028', 'some shit']
share = [20, 50, 30, 100]
p = Portfolio(code, share)

date = '2016-12-30'

port_summary, missing_info = p.summary(date, dc)
print(port_summary)
print(missing_info)

