import numpy as np
from .engine import Array

def relu(x):
    out = Array(np.where(x.data < 0, 0, x.data), (x,), 'ReLU')

    def _backward():
        x.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out


def sin(x):
    out = Array(np.sin(x.data), (x,), 'sin')

    def _backward():
        x.grad += np.cos(x.data) * out.grad
    out._backward = _backward

    return out

def cos(x):
    out = Array(np.cos(x.data), (x,), 'cos')

    def _backward():
        x.grad += -np.sin(x.data) * out.grad
    out._backward = _backward

    return out

def dot(x, y):
    out = Array(np.dot(x.data, y.data), (x, y), 'dot')

    def _backward():
        x.grad += np.dot(out.grad, y.data.T)
        y.grad += np.dot(x.data.T, out.grad)
    out._backward = _backward

    return out