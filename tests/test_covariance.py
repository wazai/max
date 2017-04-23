import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.stats as stats
import logging
import sys
import os
maxpath = os.path.join(os.environ['HOME'], 'max')
sys.path.insert(0, os.path.join(maxpath, 'src'))
from datacenter import *
from covariance import *

# change the level to logging.DEBUG to see more log
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

dc = DataCenter(startdate='2015-01-01', enddate='2017-03-31')

df = dc.price[['date', 'code', 'return']]
df = df[df.code.isin(dc.univ_dict['sz50']['code'])]

c = Covariance(df, 180)

alphas = np.linspace(0, 1, 11)

c.calibrate(alphas, alphas)

# plot the empirical vol/cor vs the EMA, using the calibrated parameters

c.plot_ema_vol(0) # plot volatility of 浦发银行

c.plot_ema_cor(0, 1) # plot correlation of 浦发银行 and 民生银行
