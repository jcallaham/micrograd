import numpy as np
from functools import singledispatch

class Scalar:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, type(self)) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, type(self)) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Scalar(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            v.grad = 0
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"

def sin(x):
    out = Scalar(np.sin(x.data), (x,), 'sin')

    def _backward():
        x.grad += np.cos(x.data) * out.grad
    out._backward = _backward

    return out

def cos(x):
    out = Scalar(np.cos(x.data), (x,), 'cos')

    def _backward():
        x.grad += -np.sin(x.data) * out.grad
    out._backward = _backward

    return out


# Array version of Scalar
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
    
    def __repr__(self):
        return f"Array(data={self.data}, grad={self.grad})"
    
    def backward(self, debug=False):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if debug: print(f"build_topo({v})")
            v.grad = np.zeros_like(v.data)
            if v not in visited:
                visited.add(v)
                if debug: print(f"visited.add({v}), v._prev={v._prev}")
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

@array.register(Array)
def _(x):
    return x

# @array.register(list)
# def _(x):
#     # Create a new Array, but if the list contains Arrays or Scalars, then track those operations
#     return None
    


# TODO:
# - Add Array creation from list of Scalars
# - add support for matrix multiplication?
# - add forward mode AD