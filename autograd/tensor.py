import numpy as np
from .data_structures import Node, Stack


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

    def _add_to_graph(self, new_node):
        node = self.__class__.graph
        new_node.nexts = [node] if node else None
        self.__class__.graph = new_node

    def backward(self, val=1):

        root = self.__class__.graph
        stack = Stack(root)

        grads = val
        while stack.is_not_empty:

            node = stack.pop()
            if node is None:
                break
            
            print(node.ref)
            grads = node.accumulate_and_store(grads)
            print(grads)

            if node.is_leaf:
                grads = val
                print(" a")
            elif len(node.nexts) == 1:
                left = node.nexts[0]
                stack.push(left)
                print(" b")
            elif len(node.nexts) == 2:
                left, right = node.nexts
                stack.push(right)
                stack.push(left)
                print(" c")
            else:
                raise RuntimeError("uhm")
