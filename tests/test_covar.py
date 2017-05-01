import logging
import numpy as np
import pandas as pd

from max.datacenter import DataCenter
from max.covar.covar import Covar
from max.covar.ema import EMA
from max.covar.metric import Metric
#from max.covar.sis import SIS
import max.covar.util as util

# change the level to logging.DEBUG to see more log
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')
dc = DataCenter(start_date='2015-01-01', end_date='2017-03-31')

df = dc.price[['code', 'return']]
df = df[df.code.isin(dc.univ_dict['sz50']['code'])]

# -----------
# test covar

c = Covar(df, 180)

c.plot_ex_post('cor', 0, 1)  # empirical correlation of 浦发银行 and 民生银行
c.plot_ex_post('vol', 0)     # empirical volatility of 浦发银行

# --------------------
# test EMA new design

e = EMA(df, 180)
metric = Metric('log_likelihood')
alphas = np.linspace(0, 1, 11)
parameter_candidate = pd.DataFrame(np.array(np.meshgrid(alphas, alphas)).T.reshape(-1, 2),
                                   columns=['alpha_cor', 'alpha_vol'])
e.calibrate(parameter_candidate, metric)
util.plot_estimate(e, 'cor', 0, 1)  # compare EMA vs ex-post correlation of 浦发银行 and 民生银行
util.plot_estimate(e, 'vol', 0)     # compare EMA vs ex-post volatility of 浦发银行


