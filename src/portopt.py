"""
Portfolio Optimizer

@author jingweiwu
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortOpt:
    
    def __init__(self, position, alpha, Sigma, lam=0.0, mu=0.0):
        logger.info('Start initializing PortOpt object')
        nsecs = len(position)
        if len(alpha) != nsecs:
            logging.error('position and alpha not the same length')
        if Sigma.shape[0] != nsecs or Sigma.shape[1] != nsecs:
            logging.error('Sigma dimension not compatible')
        self.position = position.reshape(nsecs)
        self.alpha = alpha.reshape(nsecs)
        self.Sigma = Sigma
        self.lam = lam
        self.mu = mu
        self.nsecs = nsecs
        logger.info('Finish initializing PortOpt object')
    
    def objective(self, t):
        n = self.nsecs
        tradeout = t[:n] - t[n:]
        endposn = self.position - tradeout
        alpha = np.sum(endposn*self.alpha)
        tc = self.mu * np.sum(np.abs(tradeout))
        risk = self.lam * endposn.dot(self.Sigma).dot(endposn).item()
        return -alpha + tc + risk
    
    def simple_objective(self, t):
        endposn = self.position - t
        alpha = np.sum(endposn*self.alpha)
        tc = self.mu * np.sum(t*t)
        risk = self.lam * endposn.dot(self.Sigma).dot(endposn).item()
        return -alpha + tc + risk
    
    def obj_deriv(self, t):
        n = self.nsecs
        ones = np.repeat(1.0, n)
        dfdt1 = self.alpha + self.mu*ones - 2 * self.lam * \
                (self.Sigma.dot(self.position+t[n:]) - self.Sigma.dot(t[:n]))
        dfdt2 = -self.alpha + self.mu*ones + 2 * self.lam * \
                (self.Sigma.dot(self.position-t[:n]) + self.Sigma.dot(t[n:]))
        return np.append(dfdt1, dfdt2)
    
    def get_constraints(self):
        cons = () 
        def trd(i):
            def take(t):
                return t[i]-t[self.nsecs+i]
            return take
        def endposn(i):
            def take(t):
                return self.position[i]-t[i]+t[self.nsecs+i]
            return take        
        for i in range(self.nsecs):
            cons = cons + \
                ({'type': 'ineq',
                  'fun' : trd(i)},
                 {'type': 'ineq',
                  'fun' : endposn(i)})
        return cons
    
    def optimize(self):
        logger.info('Start optimizing portfolio')
        x0 = np.append(0.5*self.position, 0.2*self.position)
        res = minimize(self.objective, x0, jac=self.obj_deriv,
                       method='SLSQP', constraints=self.get_constraints(),
                       options={'disp':False, 'maxiter':2000, 'ftol':0.01})
        tradelist = res.x[:self.nsecs] - res.x[self.nsecs:]
        self.optres = res
        if res['success']:
            logger.info(res['message'])
        else:
            logger.warning(res['message'])
        logger.info('The minimized objective function: %.2f', res['fun'])
        logger.info('Number of iterations: %i', res['nit'])
        logger.info('Finish optimizing portfolio')
        return tradelist
    
    def simple_optimize(self):
        logger.info('Start simple optimizing')
        mat = self.mu+2*self.lam*self.Sigma
        logger.info('The determinant of dividing matrix: %.2E', np.linalg.det(mat))
        t = np.linalg.inv(mat).dot(2*self.lam*self.Sigma.dot(self.position)-self.alpha)
        objval = self.simple_objective(t)
        logger.info('The minimized objective function: %.2f', objval)
        logger.info('Finish simple optimizing')
        return t