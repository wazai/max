
import max.alpha as ap
import max.datacenter.datacenter as dc

d = dc.DataCenter('2017-01-01','2017-02-01')
codes = ['000001','000002','000009']

# create nodes
a = ap.Alpha('a1', codes, d)
b = ap.Alpha('a2', codes, d)
c = ap.Alpha('a3', codes, d)
d = ap.Alpha('a4', codes, d)
e = ap.Alpha('a5', codes, d)

# add connection
a.add_child(b)
a.add_child(c)
c.add_child(d)
c.add_child(e)

# do some calculation
a.get_alpha()

# show what's in a node
a.show()
#a.metrics()

# plot some graph
a.draw_graph()
a.plot_return()

# backtest
a.backtest('2017-01-01','2017-02-01')
