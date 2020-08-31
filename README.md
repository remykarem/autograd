# Autograd WIP

Yet another autograd library.

Define a series of operations

```python
>>> a = Tensor(4)
>>> b = a * 3
>>> c = b.log()
>>> d = c * 5
>>> e = d ** 2
```

Then run backpropagation:

```python
>>> e.backward()
```

Get all the gradients

```python
>>> a.grad
31.06133312235
>>> b.grad
10.35377770745
>>> c.grad
124.24533248940001
```
