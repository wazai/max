
import alpha as ap

# create nodes
a = ap.alpha('a1')
b = ap.alpha('a2')
c = ap.alpha('a3')
d = ap.alpha()

# add connection
a.add_child(b)
a.add_child(c)
c.add_child(d)

# do some calculation
a.get_benchmark()
a.get_alpha()

# show what's in a node
a.show()
a.metrics()

# plot some graph
a.draw_graph()
a.plot_return()

# # backtest
a.backtest('2017-01-01','2017-02-01')