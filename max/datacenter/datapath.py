"""
Data Path class
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


class DataPath(object):

    def __init__(self):
        logger.info('Creating DataPath object')
        paths = dict()
        if sys.platform == 'win32':
            paths['data'] = os.path.join(os.environ['HOMEPATH'], 'Dropbox\\HW\\data')
        else:
            paths['data'] = os.path.join(os.environ['HOME'], 'Dropbox/HW/data')
        paths['marketdata'] = os.path.join(paths['data'], 'marketdata')
        paths['dailycache'] = os.path.join(paths['marketdata'], 'dailycache')
        paths['misc'] = os.path.join(paths['marketdata'], 'misc')
        paths['univ'] = os.path.join(paths['marketdata'], 'univ')
        paths['covariance'] = os.path.join(paths['data'], 'covariance')
        self.path = paths

    def get_path(self, name):
        return self.path[name]
