import torch
import numpy as np

import micrograd as mg

def test_sanity_check():

    x = mg.array(-4.0)
    z = 2 * x + 2 + x
    q = mg.relu(z) + z * x
    h = mg.relu(z * z)
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

    a = mg.array(-4.0)
    b = mg.array(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + mg.relu(b + a)
    d += 3 * d + mg.relu(b - a)
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
        # Sine
        a = mg.array(theta)
        b = mg.sin(a)
        b.backward()
        assert abs(b.data - np.sin(a.data)).sum() < 1e-6
        assert abs(a.grad - np.cos(a.data)).sum() < 1e-6

        # Cosine
        a = mg.array(theta)
        b = mg.cos(a)
        b.backward()
        assert abs(b.data - np.cos(a.data)).sum() < 1e-6
        assert abs(a.grad + np.sin(a.data)).sum() < 1e-6

    # ReLU
    for x in [-1.0, 0.0, 1.0]:
        a = mg.array(x)
        b = mg.relu(a)
        b.backward()
        print(x, b.data)
        true_val = x if x > 0.0 else 0.0
        true_grad = 1.0 if x > 0.0 else 0.0
        assert abs(b.data - true_val).sum() < 1e-6
        assert abs(a.grad - true_grad).sum() < 1e-6

    # Dot product
    a = mg.array([1.0, 2.0, 3.0])
    b = mg.array([4.0, 5.0, 6.0])
    c = mg.dot(a, b)
    c.backward()
    assert abs(c.data - np.dot(a.data, b.data)).sum() < 1e-6
    assert abs(a.grad - b.data).sum() < 1e-6
    assert abs(b.grad - a.data).sum() < 1e-6

class TestArray:
    def test_addition(self):
        # Scalar addition
        x = mg.array([1, 2, 3])
        z = x + 1
        assert abs(z.data - (x.data + 1)).sum() < 1e-6

        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - (x.data + 1)).sum() < 1e-6

        # array addition
        x = mg.array([1, 2, 3])
        y = mg.array([4, 5, 6])
        z = x + y
        assert abs(z.data - (x.data + y.data)).sum() < 1e-6

        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - (x.data + y.data)).sum() < 1e-6
        assert abs(y.grad - (x.data + y.data)).sum() < 1e-6

    def test_multiplication(self):
        # Scalar multiplication
        x = mg.array([1, 2, 3])
        z = x * 2
        assert abs(z.data - x.data * 2).sum() < 1e-6

        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - 2*z.data).sum() < 1e-6

        # array multiplication
        x = mg.array([1, 2, 3])
        y = mg.array([4, 5, 6])
        z = x * y
        assert abs(z.data - (x.data * y.data)).sum() < 1e-6
        
        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - y.data*z.data).sum() < 1e-6
        assert abs(y.grad - x.data*z.data).sum() < 1e-6

    def test_division(self):
        # Scalar division
        x = mg.array([1, 2, 3])
        z = x / 2.0
        assert abs(z.data - x.data / 2.0).sum() < 1e-6
        
        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - 1/2.0*z.data).sum() < 1e-6

        # array division
        x = mg.array([1, 2, 3])
        y = mg.array([4, 5, 6])
        z = x / y
        assert abs(z.data - (x.data / y.data)).sum() < 1e-6
        
        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - 1/y.data*z.data).sum() < 1e-6
        assert abs(y.grad - (-x.data/y.data**2)*z.data).sum() < 1e-6

    def test_power(self):
        # Power
        x = mg.array([1, 2, 3])
        z = x**2
        assert abs(z.data - x.data**2).sum() < 1e-6
        
        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - 2*x.data*z.data).sum() < 1e-6

        # Inverse 
        x = mg.array([1, 2, 3])
        z = x**-1
        assert abs(z.data - x.data**-1).sum() < 1e-6

        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - (-1*x.data**-3)).sum() < 1e-6

    def test_array_ops(self):
        # Indexing (getitem)
        x = mg.array([1, 2, 3])
        c = 3
        z = c*x[0]
        assert abs(z.data - c*x.data[0]).sum() < 1e-6

        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - np.array([c**2, 0, 0])).sum() < 1e-6

        # array creation
        x = mg.array(2)
        c = 2
        z = mg.array([1, c*x, 3])
        assert abs(z.data - np.array([1, 2*c, 3])).sum() < 1e-6

        J = 0.5*mg.dot(z, z)
        J.backward()
        assert abs(x.grad - np.array([2*c**2])).sum() < 1e-6

