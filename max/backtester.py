import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Backtester(object):
    """Backtester class

    :param strategy: Strategy object, to be backtested
    :param dc: DataCenter object
    :param start_date, end_date: string, backtest start and end date
    """

    def __init__(self, strategy, dc, start_date, end_date):
        logger.info('Creating backtester for strategy [%s] from %s to %s', strategy.name, start_date, end_date)
        self.strategy = strategy
        self.dc = dc
        self.start_date = start_date
        self.end_date = end_date
        self.result = pd.DataFrame()

    def reset_start_end_date(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def backtest(self):

        logger.info('Running backtester from %s to %s', self.start_date, self.end_date)

        alpha = self.strategy.alpha
        rule = self.strategy.rule
        port = self.strategy.port

        dates = alpha.datacenter.get_business_days_start_end(self.start_date, self.end_date)

        result = pd.DataFrame(columns=alpha.universe)
        for date in dates:
            alpha_value = alpha.get_alpha(date)
            position_before = port.get_position(date)
            covariance = port.get_covar(date)
            position_trade = rule.generate_trade_list(position_before, alpha_value, covariance)
            position_after = position_before + position_trade
            position_dict = dict(zip(alpha.universe, position_after))
            position_df = pd.DataFrame(data=position_dict, index=[date])
            result = result.append(position_df)

        self.result = result
        alpha.historic_position = result
        alpha.get_benchmark()
        alpha.get_historic_position_return(self.start_date, self.end_date)
        alpha.plot_return()
        alpha.metrics()
