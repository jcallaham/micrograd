import micrograd as mg

def test_grad():
    def f(x):
        return mg.dot(x, x)

    grad_f = mg.grad(f)
    x = mg.array([1, 2, 3])

    assert abs(f(x).data - 14) < 1e-6
    assert abs(grad_f(x) - 2*x.data).sum() < 1e-6