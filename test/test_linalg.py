import numpy as np
import numpy.testing as npt
import micrograd as mg
from micrograd.linalg import quadratic_form

def test_quadratic_form():
    x = mg.array([1.0, 2.0, 3.0])
    Q = mg.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    z = quadratic_form(Q, x)

    # Check the forward pass was correct
    npt.assert_allclose(z.data, 0.5 * x.data.T @ Q.data @ x.data)

    # Check the backward pass was correct
    z.backward()
    npt.assert_allclose(x.grad, Q.data @ x.data)


def test_inv():
    X_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    X = mg.array(X_np)
    Y = mg.linalg.inv(X)

    # Check the forward pass was correct
    Y_np = np.linalg.inv(X_np)
    npt.assert_allclose(Y.data, Y_np)

    # Check the backward pass was correct
    # `mg.jac` doesn't support matrices, so we have to do it manually
    for i in range(X.data.shape[0]):
        for j in range(X.data.shape[1]):
            X.zero_grad()
            Y_bar = np.zeros_like(Y.data)
            Y_bar[i, j] = 1.0
            Y.backward(Y_bar)

            # Compare A.grad with the analytic gradient
            X_bar = -Y_np.T @ Y_bar @ Y_np.T

            npt.assert_allclose(X.grad, X_bar)



def test_continuous_lyapunov_equation():
    from scipy.linalg import solve_continuous_lyapunov

    A_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    Q_np = np.array([[1.0, 2.0], [2.0, 1.0]])

    # Note the negative sign in the definition of Q
    # in the scipy implementation compared to micrograd
    P_np = solve_continuous_lyapunov(A_np, -Q_np)

    A = mg.array(A_np)
    P = mg.linalg.solve_continuous_lyapunov(A, Q_np)

    # Check the forward pass was correct
    npt.assert_allclose(P.data, P_np)

    # Check the backward pass was correct
    # `mg.jac` doesn't support matrices, so we have to do it manually
    for i in range(A.data.shape[0]):
        for j in range(A.data.shape[1]):
            A.zero_grad()
            P_bar = np.zeros_like(P.data)
            P_bar[i, j] = 1.0
            P.backward(P_bar)

            # Compare A.grad with the analytic gradient
            W = solve_continuous_lyapunov(A_np.T, -P_bar)
            A_bar = 2 * W @ P_np

            npt.assert_allclose(A.grad, A_bar)


def test_solve():
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_np = np.array([1.0, 2.0])

    x_np = np.linalg.solve(A_np, b_np)

    A = mg.array(A_np)
    b = mg.array(b_np)
    x = mg.linalg.solve(A, b)

    # Check the forward pass was correct
    npt.assert_allclose(x.data, x_np)

    # Check the backward pass was correct
    # `mg.jac` doesn't support matrices, so we have to do it manually
    for i in range(x.data.shape[0]):
        A.zero_grad()
        b.zero_grad()
        x_bar = np.zeros_like(x.data)
        x_bar[i] = 1.0
        x.backward(x_bar)

        # Compare A.grad with the analytic gradient
        b_bar = np.linalg.solve(A_np.T, x_bar)
        A_bar = -np.outer(b_bar, x_np)

        npt.assert_allclose(A.grad, A_bar)
        npt.assert_allclose(b.grad, b_bar)


def _test_qp():
    # Solve the quadratic program
    # min 0.5 x^T Q x + c^T x
    # s.t. x[0] + x[1] = 1.0

    Q = mg.array([[1.0, 2.0], [2.0, 1.0]])
    c = mg.array([[1.0], [2.0]])
    # A = mg.array([[1.0, 1.0]])
    # b = mg.array([1.0])
    A = mg.array([[]])
    b = mg.array([[]])


    x, y = mg.linalg.solve_qp(Q, c, A, b)

    # Check the forward pass was correct
    x_np = np.array([0.25, 0.75])


def test_qp():
    from scipy import sparse
    import osqp

    P_sp = sparse.triu([[4., 1.], [1., 2.]], format='csc')
    q = np.ones(2)
    
    A_sp = sparse.csc_matrix(np.array([[1., 1.]]))
    b = np.array([1.])

    solver = osqp.OSQP()
    solver.setup(P_sp, q, A_sp, b, b)
    osqp_results = solver.solve()

    Q = mg.array(P_sp.todense())
    c = mg.array(np.ones(2))
    A = mg.array(A_sp.todense())
    b = mg.array(np.array([1.]))

    # Check the forward pass was correct
    x, y = mg.linalg.solve_qp(Q, c, A, b)

    npt.assert_allclose(x.data, osqp_results.x)
    npt.assert_allclose(y.data, osqp_results.y)

    # Check the backward pass was correct (this only tests the solution `x`, not the
    # Lagrange multipliers `y`).
    # `mg.jac` doesn't support matrices, so we have to do it manually
    for i in range(x.data.shape[0]):
        x.zero_grad()
        y.zero_grad()
        x_bar = np.zeros_like(x.data)
        y_bar = np.zeros_like(y.data)
        x_bar[i] = 1.0
        x.backward(x_bar)

        # Compare with the analytic gradient
        solver.update(q=-x_bar, l=y_bar, u=y_bar)
        adj_results = solver.solve()
        w_x, w_y = adj_results.x, adj_results.y

        Q_bar = -0.5 * (w_x @ x.data.T + x.data @ w_x.T)
        c_bar = -w_x
        A_bar = -(np.outer(w_y, x.data) + np.outer(y.data, w_x))
        b_bar = w_y

        npt.assert_allclose(Q.grad, Q_bar)
        npt.assert_allclose(c.grad, c_bar)
        npt.assert_allclose(A.grad, A_bar)
        npt.assert_allclose(b.grad, b_bar)