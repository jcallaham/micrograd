import numpy as np
from functools import singledispatch
from typing import Union

# Array version of original micrograd.Value
class Array:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other_is_array = isinstance(other, type(self))
        _other = other if other_is_array else Array(other)
        out = Array(self.data + _other.data, (self, _other), '+')

        def _backward():
            self.grad += out.grad
            if other_is_array:
                other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other_is_array = isinstance(other, type(self))
        _other = other if other_is_array else Array(other)
        out = Array(self.data * _other.data, (self, _other), '*')

        def _backward():
            self.grad += _other.data * out.grad
            if other_is_array:
                other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Array(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __getitem__(self, idx):
        out = Array(self.data[idx], (self,), f'getitem[{idx}]')
        def _backward():
            self.grad[idx] += out.grad
        out._backward = _backward

        return out
    
    def __repr__(self):
        return f"Array(data={self.data}, grad={self.grad})"
    
    def backward(self, debug=False):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            v.grad = np.zeros_like(v.data)
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

@singledispatch
def array(x):
    return Array(x)

@array.register
def _(x: Array):
    return x

@array.register
def _(x: Union[list, tuple]):
    # Create new Arrays for elements that aren't already Arrays (existing Arrays won't change)
    _x = [array(xi) for xi in x]  
    out = Array([xi.data for xi in _x], _x, f'array({x})')
    def _backward():
        for i in range(len(x)):
            _x[i].grad += out.grad[i]
    out._backward = _backward
    return out




# TODO:
# - add support for matrix multiplication?
# - add forward mode AD