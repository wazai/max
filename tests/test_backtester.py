import numpy as np
import logging

from max.alpha import Alpha
from max.datacenter.datacenter import DataCenter
from max.rule import SimpleRule
from max.strategy import Strategy
from max.backtester import Backtester
from max.portfolio import Portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)27s  %(levelname)s  %(message)s')

start_date = '2017-01-01'
end_date = '2017-02-01'
dc = DataCenter(start_date, end_date)
codes = ['601818', '600000', '600028', 'some shit']


class RandomAlpha(Alpha):
    def cal(self, date):
        self.alpha = np.random.randn(len(self.universe))

ra = RandomAlpha('RandomAlpha', codes, dc)

# backtest

port = Portfolio(codes, [100, 200, 300, 200], dc)
rule = SimpleRule()
strat = Strategy(rule, ra, port, 'test')
bt = Backtester(strat, dc, start_date, end_date)
bt.backtest()
