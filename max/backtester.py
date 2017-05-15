import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt
class Rule(object):

    def cal_position(self, alpha):
        if not len(alpha):
            return []

        return [ 1 if x > 0 else -1 for x in alpha ]


class Backtester(object):

    def __init__(self, alpha, rule):
        self.alpha = alpha
        self.rule = rule
        self.start_date = alpha.start_date
        self.end_date = alpha.end_date


    def backtest(self):

        print('Running backtester ... ')

        dates = self.alpha.datacenter.get_business_days_start_end(self.start_date, self.end_date)

        res = pd.DataFrame(columns=self.alpha.space)
        for date in dates:
            tmp_alpha = self.alpha.get_alpha(date)
            tmp_position = self.rule.cal_position(tmp_alpha)
            tmp_dict = dict(zip(self.alpha.space, tmp_position))
            tmp_df = pd.DataFrame(data=tmp_dict, index=[date])
            res = res.append(tmp_df)

        self.alpha.historic_position= res
        self.alpha.get_benchmark(self.start_date, self.end_date)
        self.alpha.get_historic_position_return(self.start_date, self.end_date)
        self.alpha.plot_return()
        self.alpha.metrics()

        self.alpha.clean_historic_data()