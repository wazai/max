"""
Strategy
"""

import logging

logger = logging.getLogger(__name__)


class Strategy(object):
    """Strategy class
    Strategy is a combination of alpha, risk, starting portfolio, rule and some options

    :param rule: Rule object
    :param alpha: Alpha object
    :param port: Portfolio object, contains covariance matrix
    :param name: string
    :param option: dict, including rebalance frequency etc
    """

    def __init__(self, rule, alpha=None, port=None, name='NoOne', option=None):
        logger.info('Creating strategy [%s]', name)
        self.rule = rule
        self.alpha = alpha
        self.port = port
        self.name = name
        self.option = dict() if option is None else option
