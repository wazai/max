"""
Alpha

Object that contains calibration of alpha and connection between alphas
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class Alpha(object):

    # Init and utility
    # -----------------

    def __init__(self, name, universe, datacenter):
        logger.info('Initializing Alpha base class')
        # inputs
        self.__name__ = name
        self.universe = universe
        self.datacenter = datacenter
        self.start_date = datacenter.start_date
        self.end_date = datacenter.end_date

        # graph and validation
        self.children = []
        self.valid = 0

        # alpha and position, what we really need to calculate
        self.alpha = np.array([])

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        self.children.append(child)

    def validate(self):
        # assert(len(self.universe)==len(self.alpha))
        pass

    # Key functions
    # --------------

    def get_alpha(self, date):
        if self.is_leaf():
            self.cal(date)
        else:
            res = []
            for child in self.children:
                res.append(child.get_alpha(date))
            self.alpha = self.blend(res)
        return self.alpha

    @staticmethod
    def blend(alphas):
        # simple blend function, take the average
        return np.average(alphas, axis=0)

    def cal(self, date):
        self.alpha = np.array([0]*len(self.universe))

    # Display functions
    # ------------------

    def show(self):
        print('----------- name ---------------')
        print(self.__name__)
        print('valid:', self.valid)
        print('----------- children -----------')
        print(self.children)
        print('----------- alpha --------------')
        print(self.alpha)

    def draw_graph(self):
        # draw graph of all alphas, will move this out to a library

        graph = nx.Graph()
        stack = [self]

        while stack:
            node = stack.pop()
            stack += node.children

            graph.add_node(node.__name__)

            for child in node.children:
                graph.add_node(child.__name__)
                graph.add_edge(node.__name__, child.__name__)

        edges = [(u, v) for (u, v, d) in graph.edges(data=True)]
        pos = nx.spring_layout(graph)
        nx.draw_networkx_labels(graph, pos, font_size=15, font_family='sans-serif')
        nx.draw_networkx_nodes(graph, pos, node_size=1000)
        nx.draw_networkx_edges(graph, pos, edgelist=edges, width=3)
        plt.axis('off')
        plt.show()
