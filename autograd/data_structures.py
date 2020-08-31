class Node:
    def __init__(self, ref, g):
        self.ref = ref
        self.g = g
        self.next = None

    def accumulate(self, grad):
        grad = grad * self.g
        self.ref.grad = grad
        return grad
