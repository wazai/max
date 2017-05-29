"""
Rule

alpha, risk, position -> trade list
"""

import numpy as np
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseRule(object):
    """BaseRule class
    BaseRule and its derived class is used to generate trade list based on alpha, covar and current position.
    It's all numerical, i.e., it does not have any stock information.
    """

    def __init__(self, name='Base'):
        logger.info('Creating rule [%s]', name)
        self.name = name

    @abstractmethod
    def generate_trade_list(self, alpha, covar, position):
        """
        Apply rule to (alpha, covar, port) to generate trade list
        :param alpha: array of shape = [n]
        :param covar: array of shape = [n, n]
        :param position: array of shape = [n], current position in yuan
        :return: array of shape = [n], trade list
        """
        pass


class SimpleRule(BaseRule):
    """
    Generate trade list based on a simple rule:
    Assign equal weight to stocks with positive alpha, zero weight to stocks with negative alpha
    """
    def __init__(self):
        super(SimpleRule, self).__init__(name='SimpleRule')

    def generate_trade_list(self, alpha, covar=None, position=None):
        weight = [1 if x > 0 else 0 for x in alpha]
        if len(alpha) == 0 or np.sum(weight) == 0:
            return np.array([0] * len(alpha))
        else:
            return np.array(weight) / np.sum(weight)


class PortOptRule(BaseRule):
    """
    Generate trade list based on classic mean-variance portfolio optimization

    @:param portopt: PortOpt object
    @:param optimize_method: string, simple_optimize or optimize
    """
    def __init__(self, portopt, optimize_method):
        super(PortOptRule, self).__init__(name='PortOptRule')
        self.optimizer = portopt
        self.optimize_method = optimize_method

    def generate_trade_list(self, alpha, covar, position):
        # TODO Apply portfolio optimizer based on alpha, risk and existing portfolio
        pass
