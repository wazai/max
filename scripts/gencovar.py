import logging
from max.datacenter import DataCenter
from max.covar.covar import Covar

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

# save to csv

dc = DataCenter(startdate='2013-01-01', enddate='2017-03-31')
df = dc.price[['date', 'code', 'return']]
df = df[df.code.isin(dc.univ_dict['sz50']['code'])]

c = Covar(df, 360)
c.to_csv(c.ex_post['cov'], 'ex_post_360')

c = Covar(df, 180)
c.to_csv(c.ex_post['cov'], 'ex_post_180')
