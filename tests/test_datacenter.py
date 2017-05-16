from max.datacenter.datacenter import DataCenter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)27s  %(levelname)s  %(message)s')

start_date = '2015-01-01'
end_date = '2016-12-31'
d = DataCenter(start_date, end_date)

df = d.load_codes_return(['000001', '000002'], '2015-01-01', '2015-12-31')
df['000001'].cumsum().plot()
print(df.head())

# plot volatility of 浦发银行
d.cov[d.cov.code=='600000']['600000'].plot()
