
import src.alpha as ap

# create nodes
a = ap.Alpha('a1')
b = ap.Alpha('a2')
c = ap.Alpha('a3')
d = ap.Alpha()

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

# backtest
a.backtest('2017-01-01','2017-02-01')
