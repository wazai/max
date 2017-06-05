import numpy as np
import logging

from max.alpha import Alpha
from max.datacenter.datacenter import DataCenter
import max.rule as rule
from max.strategy import Strategy
from max.backtester import Backtester
from max.portfolio import Portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)27s  %(levelname)s  %(message)s')

start_date = '2017-01-01'
end_date = '2017-02-01'
dc = DataCenter(start_date, end_date)
codes = ['601818', '600000', '600028', 'cash', 'some shit']
start_position = [100, 200, 300, 10000, 200]


class RandomAlpha(Alpha):
    def cal(self, date):
        self.alpha = np.random.randn(len(self.universe))

ra = RandomAlpha('RandomAlpha', codes, dc)

# backtest

port = Portfolio(codes, start_position)
rule = rule.EqualWeightRule()
strat = Strategy(rule, ra, port, 'test')
bt = Backtester(strat, dc, start_date, end_date)
bt.backtest()


import pandas as pd
alpha = bt.strategy.alpha
rule = bt.strategy.rule
port = bt.strategy.port

position = pd.DataFrame(columns=alpha.universe)
dates = bt.dc.get_business_days_start_end(bt.start_date, bt.end_date)
date = dates[0]

alpha_value = port.get_alpha(date, alpha)
position_before = port.get_position(date, dc)
cov, cov_missing = port.get_covar(date, dc)
position_trade = rule.generate_trade_list(position_before, alpha_value, cov)
position_after = position_before + position_trade