import numpy as np
from .engine import Array

def vjp(f, x, v):
    """Vector-Jacobian product for vector-valued function"""
    y = f(x)
    y.backward(gradient=v)
    return x.grad

def grad(f):
    """Gradient of a scalar-valued function of one variable"""
    def _grad(*args, **kwargs):
        f(*args, **kwargs).backward() # forward and backward passes
        if len(args) == 1:
            return args[0].grad
        return [x.grad for x in args] # extract partial derivatives
    _grad.__name__ = f"grad({f.__name__})"
    return _grad


def jac(f):
    """Jacobian of a vector-valued function"""
    def _jac(*args, **kwargs):
        assert len(args) == 1 and isinstance(args[0], Array)
        x = args[0]

        y = f(*args, **kwargs) # forward pass (return an array)
        assert isinstance(y, Array)

        J = np.zeros((len(y.data), len(x.data)))
        for i in range(len(y.data)):
            # For each output, do a backward pass
            gradient = np.zeros_like(y.data)
            gradient[i] = 1.0

            for x in args:
                x.grad = 0.0

            y.backward(gradient=gradient)
            J[i] = x.grad
        return J
    _jac.__name__ = f"grad({f.__name__})"
    return _jac