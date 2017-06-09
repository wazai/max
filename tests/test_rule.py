import numpy as np
import max.rule as rule

r = rule.LongTopRule(2)

position = np.array([1, 1, 1, 1, 1])
alpha_value = np.array([np.nan, 3, 2, 4, -2])

r.generate_trade_list(position, alpha_value)

r = rule.LongTopRule(3)

r.generate_trade_list(position, alpha_value)