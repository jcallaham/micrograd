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
            self.grad = self.grad + out.grad
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
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Array(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        out = Array(self.data @ other.data, (self, other), '@')

        def _backward():
            _out_grad = out.grad.reshape(-1, 1) if out.grad.ndim == 1 else out.grad
            _self_data = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
            _other_data = other.data.reshape(-1, 1) if other.data.ndim == 1 else other.data
        
            _self_grad = _out_grad @ _other_data.T  # V @ X.T
            _other_grad = _self_data.T @ _out_grad  # A.T @ V

            self.grad += _self_grad.reshape(self.grad.shape)
            other.grad += _other_grad.reshape(other.grad.shape)

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
    
    def __setitem__(self, idx, value):
        raise NotImplementedError("In-place operations on arrays are not supported")
    
    def __repr__(self):
        return f"Array(data={self.data}, grad={self.grad})"
    
    def zero_grad(self):
        self.grad *= 0.0
        for child in self._prev:
            child.zero_grad()
    
    def backward(self, gradient=None):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            v.zero_grad()
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if gradient is None:
            assert self.data.shape == (), "must pass a gradient for non-scalar arrays"
            gradient = 1.0
        self.grad = np.array(gradient)

        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()

@singledispatch
def array(x):
    return Array(x)

@array.register
def _(x: Array):
    return x

def _array_iter(x: Union[list, tuple]):
    # Create new Arrays for elements that aren't already Arrays (existing Arrays won't change)
    _x = [array(xi) for xi in x]  
    out = Array([xi.data for xi in _x], _x, f'array({x})')

    # The Array creation should be a differentiable operation with respect to `x`
    def _backward():
        for i in range(len(x)):
            _x[i].grad += out.grad[i]
            
    out._backward = _backward
    return out


@array.register
def _(x: list):
    return _array_iter(x)

@array.register
def _(x: tuple):
    return _array_iter(x)




# TODO:
# - add forward mode AD