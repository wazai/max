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

    def __init__(self, name='GeneralAlpha', space=list('ABC')):
        self.children = []
        self.valid = 0
        self.__name__ = name
        self.alpha = np.array([])
        self.space = space
        self.mkt_data = pd.DataFrame(columns=self.space)
        self.benchmark = pd.Series()
        self.historic_alpha_return = pd.Series()
        self.historic_mkt_return = pd.DataFrame()
        self.historic_alpha = pd.DataFrame()

    def is_leaf(self):
        return not len(self.children) == 0

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
            # combine children alpha using average
            res = []
            for child in self.children:
                res.append(child.get_alpha(date))
            self.alpha = self.blend(res)
            self.valid = 1

            return self.alpha
        else:
            if self.valid == 1:
                return self.alpha
            else:
                self.cal(date)
                return self.alpha

    def blend(self, alphas):
        '''simple blend function, take the average'''
        return np.average(alphas, axis=0)

    def cal(self, date=dt.date.today()):
        '''Calculate alpha based on mkt data'''
        # for test only, will be override in derived nodes

        self.get_mkt(date)
        self.alpha = np.average(self.mkt_data, axis=0)

    def get_historic_alpha_return(self,start_date=dt.date.today()-dt.timedelta(days=100),
                                  end_date=dt.date.today()):
        '''dot product between historic alpha and historic mkt return. be careful of the date!'''
        if not len(self.historic_alpha):
            self.get_historic_alpha(start_date, end_date)
        if not len(self.historic_mkt_return):
            self.get_historic_mkt_return(start_date, end_date)

        self.historic_alpha_return = (self.historic_alpha.shift(1) * self.historic_mkt_return).sum(axis=1)

    '''
    ------------------------------
        Retrieve data functions
    ------------------------------
    '''
    def get_mkt(self, date):
        '''get all mkt data needed to calculate alpha'''
        # for test only, will be override in derived nodes
        days = 3
        start_date = date - dt.timedelta(days=days)
        dates = pd.date_range(str(start_date), periods=days)
        self.mkt_data =  pd.DataFrame(npr.randn(days,len(self.space)), index=dates, columns=self.space)

    def get_benchmark(self, start_date=dt.date.today()-dt.timedelta(days=100),
                      end_date=dt.date.today()):
        '''get benchmark return'''
        # for test only, will be override in derived nodes
        dates = pd.date_range(str(start_date),str(end_date))
        self.benchmark = pd.Series(npr.randn(len(dates))+0.5, index=dates)

    def get_historic_mkt_return(self, start_date=dt.date.today()-dt.timedelta(days=100),
                                end_date=dt.date.today()):
        '''get historic mkt return'''
        dates = pd.date_range(str(start_date), str(end_date) )
        self.historic_mkt_return = pd.DataFrame(npr.randn(len(dates), len(self.space))+0.5, index=dates)

    def get_historic_alpha(self, start_date=dt.date.today()-dt.timedelta(days=100),
                           end_date=dt.date.today()):
        '''get historic alpha'''
        dates = pd.date_range(str(start_date), str(end_date))
        self.historic_alpha = pd.DataFrame(npr.randn(len(dates), len(self.space)), index=dates)

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
        # print('----------- mkt data------------')
        # print(self.mkt_data)

    def metrics(self):
        '''use benchmark and historic return to calculate metrics'''
        if not len(self.historic_alpha_return):
            self.get_historic_alpha_return()
        if not len(self.benchmark):
            self.get_benchmark()

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
        plt.figure()
        if not len(self.historic_alpha_return):
            self.get_historic_alpha_return()
        if not len(self.benchmark):
            self.get_benchmark()
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

        G.add_node(self.__name__)

        for child in self.children:
            G.add_node(child.__name__)
            G.add_edge(self.__name__, child.__name__)

        nx.draw(G)
        plt.show()

    '''
    ------------------------------
        Backtest functions
    ------------------------------
    '''

    def backtest(self, start_date, end_date, *args):

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
        self.historic_alpha = pd.DataFrame(columns=self.space)
