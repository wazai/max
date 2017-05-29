"""
Strategy
"""

import logging

logger = logging.getLogger(__name__)


class Strategy(object):
    """Strategy class
    Strategy is a combination of alpha, risk, starting portfolio, rule and some options

    @param rule: Rule object
    @param alpha: Alpha object
    @param port: Portfolio object, contains covariance matrix
    @param name: string
    @param option: dict, including benchmark (default sh) and rebalance frequency (default 1 day)
    """

    def __init__(self, rule, alpha=None, port=None, name='NoOne', option=None):
        logger.info('Creating strategy [%s]', name)
        self.rule = rule
        self.alpha = alpha
        self.port = port
        if set(port.universe) != set(alpha.universe):
            raise Exception('universe of alpha and portfolio does not match')
        self.universe = port.universe
        self.name = name
        if option is None:
            option = {'benchmark': 'sh', 'frequency': 1}
        else:
            if 'benchmark' not in option.keys():
                option['benchmark'] = 'sh'
            if 'frequency' not in option.keys():
                option['frequency'] = 1
        self.option = option
