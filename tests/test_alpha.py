
import max.alpha as ap
import max.datacenter.datacenter as dc
import max.backtester as bt

dcd = dc.DataCenter('2017-01-01','2017-02-01')
codes = ['000001', '000002', '000009']

# create nodes
a = ap.Alpha('a1', codes, dcd)
b = ap.Alpha('a2', codes, dcd)
c = ap.Alpha('a3', codes, dcd)
d = ap.Alpha('a4', codes, dcd)
e = ap.Alpha('a5', codes, dcd)

# add connection
a.add_child(b)
a.add_child(c)
c.add_child(d)
c.add_child(e)

# backtest
#a.backtest('2017-01-01','2017-02-01')
rule = bt.Rule()
backtest = bt.Backtester(a,rule)
backtest.backtest()

