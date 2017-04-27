"""
Test PortOpt

"""

import sys
import os
maxpath = os.path.join(os.environ['HOME'], 'max')
sys.path.insert(0, os.path.join(maxpath, 'max'))
import logging
from portopt import *
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

# case 1 - optimal solution inside constraint

Sigma = np.diag(np.repeat(1.0, 4))
alpha = np.repeat(1.0, 4)
lam = 0.5
w = np.repeat(5.0, 4)

opt = PortOpt(w, alpha, Sigma, lam, 0.0)
opt.simple_optimize() # (4,4,4,4)
opt.optimize() # (4,4,4,4)

# case 2 - optimal solution outside constraint

Sigma = np.diag(np.repeat(1.0, 4))
alpha = np.repeat(1.0, 4)
lam = 0.2
w = np.repeat(2.0, 4)

opt = PortOpt(w, alpha, Sigma, lam, 0.0)
opt.simple_optimize() # (-0.5, -0.5, -0.5, -0.5) optimal solution is to short
opt.optimize() # (0, 0, 0, 0) optimal solution is not to trade

# case 3 - add trading cost

Sigma = np.diag(np.repeat(1.0, 4))
alpha = np.repeat(1.0, 4)
lam = 0.5
w = np.repeat(5.0, 4)

opt = PortOpt(w, alpha, Sigma, lam, 10)
opt.simple_optimize() # (0.1, 0.1, 0.1, 0.1)
opt.optimize() # (1.5, 1.5, 1.5, 1.5)

