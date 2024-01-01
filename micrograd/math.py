import numpy as np
from .engine import Array

def relu(x):
    out = Array(np.where(x.data < 0, 0, x.data), (x,), 'ReLU')

    def _backward():
        x.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

def tanh(x):
    out = Array(np.tanh(x.data), (x,), 'tanh')

    def _backward():
        x.grad += (1 - out.data**2) * out.grad
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

def tan(x):
    out = Array(np.tan(x.data), (x,), 'tan')

    def _backward():
        x.grad += (1 / np.cos(x.data)**2) * out.grad
    out._backward = _backward

    return out

def exp(x):
    out = Array(np.exp(x.data), (x,), 'exp')

    def _backward():
        x.grad += np.exp(x.data) * out.grad
    out._backward = _backward

    return out

def log(x):
    out = Array(np.log(x.data), (x,), 'log')

    def _backward():
        x.grad += (1 / x.data) * out.grad
    out._backward = _backward

    return out

def dot(x, y):
    out = Array(np.dot(x.data, y.data), (x, y), 'dot')

    def _backward():
        x.grad += np.dot(out.grad, y.data.T)
        y.grad += np.dot(x.data.T, out.grad)
    out._backward = _backward

    return out