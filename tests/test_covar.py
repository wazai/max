import numpy as np
import logging
from max.datacenter import DataCenter
from max.covar.covar import Covar
from max.covar.ema import EMA


# change the level to logging.DEBUG to see more log
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

# -----------
# test covar

dc = DataCenter(startdate='2015-01-01', enddate='2017-03-31')
df = dc.price[['date', 'code', 'return']]
df = df[df.code.isin(dc.univ_dict['sz50']['code'])]

c = Covar(df, 180)

c.plot_fit('cor', 0, 1)  # empirical correlation of 浦发银行 and 民生银行
c.plot_fit('vol', 0)     # empirical volatility of 浦发银行

# ---------
# test EMA

e = EMA(df, 180)

alphas = np.linspace(0, 1, 11)

e.calibrate(alphas, alphas)
e.plot_fit('cor', 0, 1)  # compare EMA vs ex-post correlation of 浦发银行 and 民生银行
e.plot_fit('vol', 0)     # compare EMA vs ex-post volatility of 浦发银行
