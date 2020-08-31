import numpy as np
from .data_structures import Node


class Tensor:
    graph = None

    def __init__(self, data):
        self.data = data
        self.grad = None

    def __repr__(self):
        return f"Tensor({self.data})"

    def __mul__(self, rhs):
        self._add_to_graph(Node(self, rhs))
        return Tensor(self.data * rhs)

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __add__(self, rhs):
        self._add_to_graph(Node(self, 1))
        return Tensor(self.data + rhs)

    def __radd__(self, lhs):
        return self.__add__(lhs)

    def log(self):
        self._add_to_graph(Node(self, 1/self.data))
        return Tensor(np.log(self.data))

    def __pow__(self, rhs):
        self._add_to_graph(Node(self, rhs*self.data))
        return Tensor(self.data ** rhs)

    ###

    def _add_to_graph(self, new_node):
        new_node.next = self.__class__.graph
        self.__class__.graph = new_node

    def backward(self, val=1):

        self._add_to_graph(Node(self, val))

        node = self.__class__.graph
        gradient = val

        while node:
            gradient = node.accumulate(gradient)
            node = node.next
