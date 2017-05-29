"""
Alpha

Object that contains calibration of alpha and connection between alphas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt
from sklearn import linear_model
import logging

logger = logging.getLogger(__name__)


class Alpha(object):

    # Init and utility
    # -----------------

    def __init__(self, name, universe, datacenter):
        logger.info('Initializing Alpha base class')
        # inputs
        self.__name__ = name
        self.universe = universe
        self.datacenter = datacenter
        self.start_date = datacenter.start_date
        self.end_date = datacenter.end_date

        # graph and validation
        self.children = []
        self.valid = 0

        # alpha and position, what we really need to calculate
        self.alpha = np.array([])
        self.position = np.array([])

        # some historic data for backtesting and benchmarking
        self.benchmark = pd.Series()
        self.historic_position_return = pd.Series()
        self.historic_mkt_return = pd.DataFrame(columns=self.universe)
        self.historic_position = pd.DataFrame(columns=self.universe)

        # load historic mkt data from data center
        self.get_historic_mkt_return(self.start_date, self.end_date)

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        self.children.append(child)

    def validate(self):
        # assert(len(self.universe)==len(self.alpha))
        pass

    # Key functions
    # --------------

    def get_alpha(self, date):
        if self.is_leaf():
            self.cal(date)
        else:
            res = []
            for child in self.children:
                res.append(child.get_alpha(date))
            self.alpha = self.blend(res)
        return self.alpha

    @staticmethod
    def blend(alphas):
        # simple blend function, take the average
        return np.average(alphas, axis=0)

    def cal(self, date):
        # Calculate alpha based on mkt data
        # for test only, will be override in derived nodes

        days = 1
        start_date = date - dt.timedelta(days=days)
        end_date = date - dt.timedelta(days=1)
        # dates = self.datacenter.get_business_days_start_end(start_date, date)
        mkt_data = self.historic_mkt_return[start_date:end_date]

        self.alpha = np.average(mkt_data, axis=0)

    def get_historic_position_return(self, start_date, end_date):
        # dot product between historic alpha and historic mkt return. be careful of the date!
        if not len(self.historic_position):
            self.get_historic_position(start_date, end_date)
        if not len(self.historic_mkt_return):
            self.get_historic_mkt_return(start_date, end_date)

        self.historic_position_return = (self.historic_position.shift(1) * self.historic_mkt_return).sum(axis=1)

    def clean_historic_data(self):
        # clean up all historic data. this should be used before backtest
        self.benchmark = pd.Series()
        self.historic_position_return = pd.Series()
        self.historic_mkt_return = pd.DataFrame(columns=self.universe)
        self.historic_position = pd.DataFrame(columns=self.universe)
        self.valid = 0

    # Retrieve data functions
    # ------------------------

    def get_benchmark(self):
        # get benchmark return
        self.benchmark = self.datacenter.load_codes_return(['sh'], self.start_date, self.end_date)['sh']

    def get_historic_mkt_return(self, start_date, end_date):
        # get historic market return
        # dates = pd.date_range(str(start_date), str(end_date) )
        self.historic_mkt_return = self.datacenter.load_codes_return(self.universe, start_date, end_date)

    def get_historic_position(self, start_date, end_date):
        # get historic alpha from storage, this is different from the backtester where we generate
        # historic_position on the fly
        pass

    # Display functions
    # ------------------

    def show(self):
        print('----------- name ---------------')
        print(self.__name__)
        print('valid:', self.valid)
        print('----------- children -----------')
        print(self.children)
        print('----------- alpha --------------')
        print(self.alpha)

    def metrics(self):
        # use benchmark and historic return to calculate metrics
        if not len(self.historic_position_return):
            self.get_historic_position_return(self.start_date, self.end_date)
        if not len(self.benchmark):
            self.get_benchmark()

        # simple sharp
        sharpe_ratio = self.historic_position_return[1:].mean() / self.historic_position_return[1:].std()
        active_return = self.historic_position_return[1:].mean() - self.benchmark.mean()
        tracking_error = (self.historic_position_return[1:]-self.benchmark).std()
        information_ratio = active_return / tracking_error

        print('--------- Alpha Metrics --------')
        print('Avg Daily Return: ', self.historic_position_return[1:].mean())
        print('Daily Vol: ', self.historic_position_return[1:].std())
        print('Sharpe Ratio: ', sharpe_ratio)
        print('Information Ratio: ', information_ratio)

        # simple alpha/beta
        size = len(self.benchmark) - 1
        x = self.historic_position_return[1:].values.reshape(size, 1)
        y = self.benchmark[1:].values.reshape(size, 1)

        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        [[m_beta]] = reg.coef_
        [m_alpha] = reg.intercept_

        print('alpha: ', m_alpha)
        print('beta:  ', m_beta)

    def plot_return(self):

        if not len(self.historic_position_return):
            self.get_historic_position_return(self.start_date, self.end_date)
        if not len(self.benchmark):
            self.get_benchmark()

        self.benchmark.cumsum().plot(style='b', legend=True)
        plot = self.historic_position_return.cumsum().plot(style='g', legend=True)
        plt.legend(['Benchmark', 'Alpha'])
        plot.set_ylabel('Return')

        plt.show()

    def draw_graph(self):
        # draw graph of all alphas, will move this out to a library

        graph = nx.Graph()
        stack = [self]

        while stack:
            node = stack.pop()
            stack += node.children

            graph.add_node(node.__name__)

            for child in node.children:
                graph.add_node(child.__name__)
                graph.add_edge(node.__name__, child.__name__)

        edges = [(u, v) for (u, v, d) in graph.edges(data=True)]
        pos = nx.spring_layout(graph)
        nx.draw_networkx_labels(graph, pos, font_size=15, font_family='sans-serif')
        nx.draw_networkx_nodes(graph, pos, node_size=1000)
        nx.draw_networkx_edges(graph, pos, edgelist=edges, width=3)
        plt.axis('off')
        plt.show()
