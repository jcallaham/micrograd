import micrograd as mg

def test_grad():
    def f(x):
        return mg.dot(x, x)

    grad_f = mg.grad(f)
    x = mg.array([1, 2, 3])

    assert abs(f(x).data - 14) < 1e-6
    assert abs(grad_f(x) - 2*x.data).sum() < 1e-6

def test_vjp():
    def f(x):
        return mg.array([x[0] * x[1], 100 * x[0]**2])
    
     # df/dx = [[2, 1], [200, 0]]
    x = mg.array([1.0, 2.0])

    v = [1.0, 0.0]
    assert abs(mg.vjp(f, x, v) - [2.0, 1.0]).sum() < 1e-6

    v = [0.0, 1.0]
    assert abs(mg.vjp(f, x, v) - [200.0, 0.0]).sum() < 1e-6

def test_jac():
    def f(x):
        return mg.array([x[0] * x[1], 100 * x[0]**2])
    
     # df/dx = [[2, 1], [200, 0]]
    x = mg.array([1.0, 2.0])
    assert abs(mg.jac(f)(x) - [[2.0, 1.0], [200.0, 0.0]]).sum() < 1e-6