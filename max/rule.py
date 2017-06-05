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

    @staticmethod
    def check_variable_length(position, alpha_value):
        if len(position) == 0:
            raise Exception('Empty position list')
        if len(position) != len(alpha_value):
            raise Exception('Length of position and alpha_value does not match')

    @abstractmethod
    def generate_trade_list(self, position, alpha_value, covar):
        """
        Apply rule to (alpha, covar, port) to generate trade list
        @param alpha_value: array of shape = [n]
        @param covar: array of shape = [n, n]
        @param position: array of shape = [n], current position in yuan
        @return: array of shape = [n], trade list
        """
        pass


class EqualWeightRule(BaseRule):
    """
    Generate trade list based on a simple equal weight rule:
    Assign equal weight to stocks with positive alpha, zero weight to stocks with negative alpha
    """
    def __init__(self):
        super(EqualWeightRule, self).__init__(name='EqualWeight')

    def generate_trade_list(self, position, alpha_value, covar=None):
        self.check_variable_length(position, alpha_value)
        alpha_value[np.isnan(position)] = np.nan
        weight = [1 if x > 0 else 0 for x in alpha_value]
        if np.sum(weight) == 0:
            weight[np.nanargmax(alpha_value)] = 1.0
        position_after = np.array(weight) / np.nansum(weight) * np.nansum(position)
        return position_after - position


class MVORule(BaseRule):
    """
    Generate trade list based on Markowitz mean-variance optimization (MVO)

    @param portopt: PortOpt object
    @param optimize_method: string, simple_optimize or optimize
    """
    def __init__(self, portopt, optimize_method):
        super(MVORule, self).__init__(name='MVO')
        self.optimizer = portopt
        self.optimize_method = optimize_method

    def generate_trade_list(self, position, alpha_value, covar):
        # TODO Apply portfolio optimizer based on alpha_value, risk and existing portfolio
        pass
