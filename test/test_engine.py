import torch
import numpy as np

import micrograd as mg
from micrograd.engine import Scalar

def test_sanity_check():

    x = Scalar(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Scalar(-4.0)
    b = Scalar(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_math():
    for theta in [0.0, np.pi/4, np.pi/3, np.pi/2]:
        # Test sin
        a = Scalar(theta)
        b = mg.sin(a)
        b.backward()
        assert abs(b.data - np.sin(theta)) < 1e-6
        assert abs(a.grad - np.cos(theta)) < 1e-6

        # Test cos
        a = Scalar(theta)
        b = mg.cos(a)
        b.backward()
        assert abs(b.data - np.cos(theta)) < 1e-6
        assert abs(a.grad + np.sin(theta)) < 1e-6

def test_array():
    # Scalar addition
    x = mg.Array([1, 2, 3])
    z = x + 1
    z.backward()
    assert abs(z.data - (x.data + 1)).sum() < 1e-6
    assert abs(x.grad - np.array([1, 1, 1])).sum() < 1e-6

    # Array addition
    x = mg.Array([1, 2, 3])
    y = mg.Array([4, 5, 6])
    z = x + y
    z.backward()
    assert abs(z.data - (x.data + y.data)).sum() < 1e-6
    assert abs(x.grad - np.array([1, 1, 1])).sum() < 1e-6
    assert abs(y.grad - np.array([1, 1, 1])).sum() < 1e-6

    # Scalar multiplication
    x = mg.Array([1, 2, 3])
    z = x * 2
    z.backward()
    assert abs(z.data - x.data * 2).sum() < 1e-6
    assert abs(x.grad - np.array([2, 2, 2])).sum() < 1e-6

    # Array multiplication
    x = mg.Array([1, 2, 3])
    y = mg.Array([4, 5, 6])
    z = x * y
    z.backward()
    assert abs(z.data - (x.data * y.data)).sum() < 1e-6
    assert abs(x.grad - y.data).sum() < 1e-6
    assert abs(y.grad - x.data).sum() < 1e-6

    # Scalar division
    x = mg.Array([1, 2, 3])
    z = x / 2.0
    z.backward()
    assert abs(z.data - x.data / 2.0).sum() < 1e-6
    assert abs(x.grad - np.array([0.5, 0.5, 0.5])).sum() < 1e-6

    # Array division
    x = mg.Array([1, 2, 3])
    y = mg.Array([4, 5, 6])
    z = x / y
    z.backward()
    assert abs(z.data - (x.data / y.data)).sum() < 1e-6
    assert abs(x.grad - 1/y.data).sum() < 1e-6
    assert abs(y.grad - (-x.data/y.data**2)).sum() < 1e-6

    # Power
    x = mg.Array([1, 2, 3])
    z = x**2
    z.backward()
    assert abs(z.data - x.data**2).sum() < 1e-6
    assert abs(x.grad - 2*x.data).sum() < 1e-6

    # Inverse 
    x = mg.Array([1, 2, 3])
    z = x**-1
    z.backward()
    assert abs(z.data - x.data**-1).sum() < 1e-6
    assert abs(x.grad - (-1*x.data**-2)).sum() < 1e-6

    # Sine
    x = mg.Array([0, np.pi/4, np.pi/3, np.pi/2])
    z = mg.sin(x)
    z.backward()
    assert abs(z.data - np.sin(x.data)).sum() < 1e-6
    assert abs(x.grad - np.cos(x.data)).sum() < 1e-6

    # Cosine
    x = mg.Array([0, np.pi/4, np.pi/3, np.pi/2])
    z = mg.cos(x)
    z.backward()
    assert abs(z.data - np.cos(x.data)).sum() < 1e-6
    assert abs(x.grad + np.sin(x.data)).sum() < 1e-6