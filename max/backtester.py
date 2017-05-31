import pandas as pd
import logging

from max.performance import Performance

logger = logging.getLogger(__name__)


class Backtester(object):
    """Backtester class

    @member strategy: Strategy object, to be backtested
    @member dc: DataCenter object
    @member start_date, end_date: string, backtest start and end date
    @member return_: DataFrame of shape = [T, n], daily return of each stock
    @member position: DataFrame of shape = [T, n], daily holding of each stock
    @member benchmark_return: pd series of shape = [n], daily benchmark return
    @member portfolio_return: pd series of shape = [n], daily portfolio return
    @member performance: Performance object, to evaluate the strategy performance
    """

    def __init__(self, strategy, dc, start_date, end_date):
        logger.info('Creating backtester for strategy [%s] from %s to %s', strategy.name, start_date, end_date)
        self.strategy = strategy
        self.universe = strategy.universe
        self.dc = dc
        self.start_date = start_date
        self.end_date = end_date
        self.check_start_end_date()
        self.return_ = dc.load_codes_return(self.universe, self.start_date, self.end_date)
        self.position = pd.DataFrame()
        self.benchmark_return = self.get_benchmark_return()
        self.portfolio_return = pd.Series()
        self.performance = Performance(strategy.name)

    def check_start_end_date(self):
        if self.start_date > self.end_date:
            raise Exception('start date greater than end date')
        dc_start_date = self.dc.start_date.strftime('%Y-%m-%d')
        dc_end_date = self.dc.end_date.strftime('%Y-%m-%d')
        if self.start_date > dc_end_date:
            raise Exception('start date greater than datacenter end date')
        if self.end_date < dc_start_date:
            raise Exception('end date less than datacenter start date')
        if self.start_date < dc_start_date:
            logger.debug('data unavailable, change start date from %s to %s', self.start_date, dc_start_date)
            self.start_date = dc_start_date
        if self.end_date > dc_end_date:
            logger.debug('data unavailable, change end date from %s to %s', self.end_date, dc_end_date)
            self.end_date = dc_end_date

    def reset_start_end_date(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.check_start_end_date()

    def get_benchmark_return(self):
        benchmark = self.strategy.option['benchmark']
        return self.dc.load_codes_return([benchmark], self.start_date, self.end_date)[benchmark]

    def get_portfolio_return(self):
        return (self.position.shift(1) * self.return_).sum(axis=1)

    def clear(self):
        # clean up all historic data. this should be used before backtest
        self.benchmark_return = pd.Series()
        self.portfolio_return = pd.Series()
        self.benchmark_return = pd.DataFrame(columns=self.universe)
        self.position = pd.DataFrame(columns=self.universe)

    def backtest(self):
        logger.info('Running backtester from %s to %s', self.start_date, self.end_date)

        alpha = self.strategy.alpha
        rule = self.strategy.rule
        port = self.strategy.port

        position = pd.DataFrame(columns=alpha.universe)
        dates = alpha.datacenter.get_business_days_start_end(self.start_date, self.end_date)
        for date in dates:
            # TODO need to take into account rebalance frequency here
            alpha_value = alpha.get_alpha(date)
            position_before = port.get_position(date)
            covariance = port.get_covar(date)
            position_trade = rule.generate_trade_list(position_before, alpha_value, covariance)
            position_after = position_before + position_trade
            position_dict = dict(zip(alpha.universe, position_after))
            position_df = pd.DataFrame(data=position_dict, index=[date])
            position = position.append(position_df)

        self.position = position
        self.portfolio_return = self.get_portfolio_return()

        self.performance.set_return_series(self.portfolio_return[1:], self.benchmark_return[1:])
        self.performance.plot_return()
        self.performance.show_metrics()
