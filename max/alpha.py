import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt
from sklearn import linear_model


class Alpha(object):
    """Object that contains calibration of alpha and connection between alphas"""
    '''
    ------------------------------
        Init and utility
    ------------------------------
    '''

    def __init__(self, name, space, datacenter):

        # inputs
        self.__name__ = name
        self.space = space
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
        self.historic_alpha_return = pd.Series()
        self.historic_mkt_return = pd.DataFrame(columns=self.space)
        self.historic_alpha = pd.DataFrame(columns=self.space)

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        self.children.append(child)

    def validate(self):
        # assert(len(self.space)==len(self.alpha))
        pass

    '''
    ------------------------------
        Key functions
    ------------------------------
    '''

    def get_alpha(self, date=dt.date.today()):
        '''get alpha from cal if it is a leaf node,
        otherwise blend children alphas into one'''
        if self.is_leaf():
            if self.valid == 1:
                return self.alpha
            else:
                self.cal(date)
                self.valid = 1
                return self.alpha
        else:
            res = []
            for child in self.children:
                res.append(child.get_alpha(date))
            self.alpha = self.blend(res)
            self.valid = 1

            return self.alpha

    def blend(self, alphas):
        '''simple blend function, take the average'''
        return np.average(alphas, axis=0)

    def cal(self, date=dt.date.today()):
        '''Calculate alpha based on mkt data'''
        # for test only, will be override in derived nodes

        days = 3
        start_date = date - dt.timedelta(days=days)
        dates = pd.date_range(str(start_date), periods=days)
        mkt_data = pd.DataFrame(npr.randn(days,len(self.space)), index=dates, columns=self.space)

        self.alpha = np.average(mkt_data, axis=0)

    def get_historic_alpha_return(self, start_date, end_date):
        '''dot product between historic alpha and historic mkt return. be careful of the date!'''
        if not len(self.historic_alpha):
            self.get_historic_alpha(start_date, end_date)
        if not len(self.historic_mkt_return):
            self.get_historic_mkt_return(start_date, end_date)

        self.historic_alpha_return = (self.historic_alpha.shift(1) * self.historic_mkt_return).sum(axis=1)

    def get_position_from_alpha(self):
        if not len(self.alpha):
            self.get_alpha()

        # simple buy sell position
        self.position = [ 1 if x > 0 else -1 for x in self.alpha ]

    def clean_historic_data(self):
        '''clean up all historic data. this should be used before backtest'''
        self.benchmark = pd.Series()
        self.historic_alpha_return = pd.Series()
        self.historic_mkt_return = pd.DataFrame(columns=self.space)
        self.historic_alpha = pd.DataFrame(columns=self.space)
        self.valid=0

    '''
    ------------------------------
        Retrieve data functions
    ------------------------------
    '''
    def get_benchmark(self, start_date, end_date):
        '''get benchmark return'''
        # for test only, will be override in derived nodes
        dates = self.datacenter.get_business_days_start_end(start_date,end_date)
        self.benchmark = pd.Series(npr.randn(len(dates))/10, index=dates)

    def get_historic_mkt_return(self, start_date, end_date):
        '''get historic mkt return'''
        # dates = pd.date_range(str(start_date), str(end_date) )
        # self.historic_mkt_return = pd.DataFrame(npr.randn(len(dates), len(self.space))+0.5, index=dates)
        self.historic_mkt_return = self.datacenter.load_codes_return(self.space,start_date,end_date)

    def get_historic_alpha(self, start_date, end_date):
        '''
        get historic alpha from storage, this is different from the backtester where we generate
        historic_alpha on the fly
        '''
        dates = self.datacenter.get_business_days_start_end(start_date,end_date)
        self.historic_alpha = pd.DataFrame(npr.randn(len(dates), len(self.space)), columns=self.space, index=dates)

    def get_historic_position_from_alpha(self):
        if not len(self.historic_alpha):
            return


    '''
    ------------------------------
        Display functions
    ------------------------------
    '''
    def show(self):
        print('----------- name ---------------')
        print(self.__name__)
        print('valid:', self.valid)
        print('----------- children -----------')
        print(self.children)
        print('----------- alpha --------------')
        print(self.alpha)

    def metrics(self):
        '''use benchmark and historic return to calculate metrics'''
        if not len(self.historic_alpha_return):
            self.get_historic_alpha_return(self.start_date,self.end_date)
        if not len(self.benchmark):
            self.get_benchmark(self.start_date, self.end_date)

        # simple sharp
        sharp = self.historic_alpha_return[1:].mean() / self.historic_alpha_return[1:].std()

        print('--------- Alpha Metrics --------')
        print('sharp: ', sharp)

        # simple alpha/beta
        size = len(self.benchmark) - 1
        x = self.historic_alpha_return[1:].reshape(size, 1)
        y = self.benchmark[1:].reshape(size, 1)

        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        [[ m_beta ]] = reg.coef_
        [ m_alpha ]  = reg.intercept_

        print('alpha: ', m_alpha)
        print('beta:  ', m_beta)

    def plot_return(self):

        if not len(self.historic_alpha_return):
            self.get_historic_alpha_return(self.start_date,self.end_date)
        if not len(self.benchmark):
            self.get_benchmark(self.start_date,self.end_date)

        plt.figure()
        plot = self.benchmark.cumsum().plot(style='b',legend=True)
        plot = self.historic_alpha_return.cumsum().plot(style='g',legend=True)
        plt.legend(['Benchmark','Alpha'])
        # plot = self.historic_return.plot(secondary_y=True)
        # plot = self.benchmark.plot(secondary_y=True)
        plot.set_ylabel('Return')

        plt.show()

    def draw_graph(self):
        '''draw graph of all alphas, will move this out to a library'''

        G = nx.Graph()
        stack = [ self ]

        while stack:
            node = stack.pop()
            stack += node.children

            G.add_node(node.__name__)

            for child in node.children:
                G.add_node(child.__name__)
                G.add_edge(node.__name__, child.__name__)

        edges = [(u, v) for (u, v, d) in G.edges(data=True)]
        pos = nx.spring_layout(G)
        nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')
        nx.draw_networkx_nodes(G, pos, node_size=1000)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3)
        plt.axis('off')
        plt.show()

    '''
    ------------------------------
        Backtest functions
    ------------------------------
    '''

    def backtest(self, start_date=None, end_date=None):

        print('Running backtester ... ')

        if start_date == None: start_date = self.start_date
        if end_date == None: end_date = self.end_date

        self.clean_historic_data()

        dates = pd.date_range(start_date, end_date)

        res = pd.DataFrame(columns=self.space)
        for date in dates:
            tmp_alpha = self.get_alpha(date)
            tmp_dict = dict(zip(self.space,tmp_alpha))
            tmp_df = pd.DataFrame(data=tmp_dict,index=[date])
            res.append(tmp_df)

        self.historic_alpha = res
        self.get_benchmark(start_date, end_date)
        self.get_historic_alpha_return(start_date, end_date)
        self.plot_return()
        self.metrics()

        self.clean_historic_data()


class Rule(object):

    def cal_position(self,alpha):
        if not len(alpha):
            return []

        return [ 1 if x > 0 else -1 for x in alpha ]


class Backtester(object):

    def __init__(self, alpha, position):
        self.alpha = alpha
        self.position_gen = position
        self.start_date = alpha.start_date
        self.end_date = alpha.end_date

    def backtest(self):
        self.alpha.backtest(self.start_date,self.end_date)