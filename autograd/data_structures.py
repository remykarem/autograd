class Node:
    def __init__(self, ref, g, name):
        self.ref = ref
        self.g = g
        self.nexts = None
        self.name = name

    def __repr__(self):
        return f"Node(ref={self.ref}, g={self.g})"

    def accumulate_and_store(self, grad):
        grad = grad * self.g
        self.ref.grad = grad
        return grad

    @property
    def is_leaf(self):
        return not self.nexts


class Stack:
    def __init__(self, node=None):
        if node:
            self.data = [node]
        else:
            self.data = []

    def push(self, new):
        self.data.append(new)

    def pop(self):
        return self.data.pop()

    @property
    def is_not_empty(self):
        return len(self.data) > 0
