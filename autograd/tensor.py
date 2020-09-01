import math
from .data_structures import Node, Stack


class Tensor:

    graph = None

    def __init__(self, data):
        self.data = data
        self.grad = None
        self.grad_fn = None

    def __repr__(self):
        if self.grad_fn:
            return f"Tensor({self.data}, grad_fn=<{self.grad_fn.name}>)"
        else:
            return f"Tensor({self.data})"

    def __mul__(self, rhs):
        # 1. Create any node in grad_fn
        if self.grad_fn:
            self._add_to_graph(self.grad_fn)

        # 2. Create new Tensor
        t = Tensor(self.data * rhs)

        # 3. Set the grad_fn attribute
        t.grad_fn = Node(self, rhs, "MulBackward")

        return t

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __add__(self, rhs):
        if self.grad_fn:
            self._add_to_graph(self.grad_fn)

        t = Tensor(self.data + rhs)

        t.grad_fn = Node(self, 1, "AddBackward")

        return t

    def __radd__(self, lhs):
        return self.__add__(lhs)

    def log(self):
        if self.grad_fn:
            self._add_to_graph(self.grad_fn)

        t = Tensor(math.log(self.data))

        t.grad_fn = Node(self, 1/self.data, "LogBackward")

        return t

    def __pow__(self, rhs):
        if self.grad_fn:
            self._add_to_graph(self.grad_fn)

        t = Tensor(self.data ** rhs)

        t.grad_fn = Node(self, rhs*self.data, "PowerBackward")

        return t

    def _add_to_graph(self, new_node):
        node = self.__class__.graph
        new_node.nexts = [node] if node else None
        self.__class__.graph = new_node

    def backward(self, val=1):

        if self.grad_fn:
            self._add_to_graph(self.grad_fn)

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
