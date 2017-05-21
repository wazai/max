import pandas as pd

class Rule(object):

    def cal_position(self, alpha):
        if not len(alpha):
            return []
        res = [ 1 if x > 0 else 0 for x in alpha ]
        if res == 0:
            return [0 for i in range(len(alpha))]
        else:
            return res / sum(res)

class RuleOpt(object):

    def cal_position(self, alpha, opt, position):
        pass

class Backtester(object):

    def __init__(self, alpha, rule):
        self.alpha = alpha
        self.rule = rule
        self.start_date = alpha.start_date
        self.end_date = alpha.end_date


    def backtest(self):

        print('Running backtester ... ')

        dates = self.alpha.datacenter.get_business_days_start_end(self.start_date, self.end_date)

        res = pd.DataFrame(columns=self.alpha.universe)
        for date in dates:
            tmp_alpha = self.alpha.get_alpha(date)
            tmp_position = self.rule.cal_position(tmp_alpha)
            tmp_dict = dict(zip(self.alpha.universe, tmp_position))
            tmp_df = pd.DataFrame(data=tmp_dict, index=[date])
            res = res.append(tmp_df)

        self.alpha.historic_position= res
        self.alpha.get_benchmark(self.start_date, self.end_date)
        self.alpha.get_historic_position_return(self.start_date, self.end_date)
        self.alpha.plot_return()
        self.alpha.metrics()

 #       self.alpha.clean_historic_data()