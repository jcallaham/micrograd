import numpy as np
from scipy import linalg as scipy_linalg
from scipy import sparse as scipy_sparse

import osqp

from .engine import Array


def quadratic_form(Q, x: Array):
    # Compute quadratic form (1/2) x^T Q x, where Q is a square matrix and x is a vector
    # Q is considered static data (constant) and x is considered dynamic data (variable)

    # Note that this could just be done directly with
    # `0.5 * x.T @ Q @ x`, since transposition and matrix multiplication are
    # already implemented. However, it's a simple example of how to implement
    # a custom function with a backward pass.

    # Since Q is static, we don't need to worry about its gradient.
    Q = Q.data if isinstance(Q, Array) else Q

    # Forward pass: compute the quadratic form
    out = Array(0.5 * x.data.T @ Q.data @ x.data, (x,), 'quadratic_form')

    # Backward pass: compute the gradient with respect to x
    # using the reverse-mode rule `x_bar = y_bar * (Q @ x)`
    def _backward():
        # The adjoint outputs will be in `out.grad`
        y_bar = out.grad

        # Compute the adjoint inputs
        x_bar = y_bar * (Q @ x.data)

        # Accumulate the adjoint inputs in the gradient
        x.grad += x_bar

    out._backward = _backward

    return out


def inv(X: Array):
    # Matrix inverse of X

    # Forward pass: compute the inverse
    out = Array(scipy_linalg.inv(X.data), (X,), 'inv')

    # Backward pass: compute the gradient with respect to X
    # using the reverse-mode rule `X_bar = -Y^T @ Y_bar @ Y^T`,
    # where Y = X^{-1} and Y_bar = out.grad
    def _backward():
        Y, Y_bar = out.data, out.grad
        X.grad += -Y.T @ Y_bar @ Y.T

    out._backward = _backward

    return out

def solve_continuous_lyapunov(A, Q):
    # Solve the Lyapunov equation A P + P A^T + Q = 0 for X

    # Since Q is static, we don't need to worry about its gradient.
    Q = Q.data if isinstance(Q, Array) else Q

    P = scipy_linalg.solve_continuous_lyapunov(A.data, -Q)  # Primal solution
    out = Array(P, (A,), 'solve_continuous_lyapunov')

    def _backward():
        # Solve the adjoint equation A^T W + W A + P_bar = 0
        P_bar = out.grad
        W = scipy_linalg.solve_continuous_lyapunov(A.data.T, -P_bar)

        # Compute the gradient using the reverse-mode rule
        # and the solution to the adjoint Lyapunov equation
        A_bar = 2 * W @ P

        # Accumulate the gradient to A
        A.grad += A_bar

    out._backward = _backward

    return out

def solve(A, b):
    # Solve the linear system `Ax = b`` for x
    # Both A and b are "dynamic" data, so we need to compute both their gradients.

    # Forward pass: compute the solution
    out = Array(scipy_linalg.solve(A.data, b.data), (A, b), 'solve')

    # Backward pass: compute the gradients with respect to A and b
    def _backward():
        # The adjoint outputs will be in `out.grad`
        x_bar = out.grad.reshape(-1, 1) if out.grad.ndim == 1 else out.grad
        x = out.data.reshape(-1, 1) if out.data.ndim == 1 else out.data

        # Solve the adjoint system A^T w = x_bar
        w = scipy_linalg.solve(A.data.T, x_bar)

        # Compute the adjoint inputs
        A_bar = -w @ x.T
        b_bar = w.reshape(b.grad.shape)

        # Accumulate the adjoint inputs in the gradients
        A.grad += A_bar
        b.grad += b_bar

    out._backward = _backward
    return out


def solve_qp(Q, c, A, b, **settings):
    # Solve the equality-constrained quadratic program
    # min 0.5 x^T Q x + c^T x
    # s.t. A x = b

    # Initialize the OSQP solver
    solver = osqp.OSQP()
    P_sp = scipy_sparse.csc_matrix(Q.data)
    A_sp = scipy_sparse.csc_matrix(A.data)
    solver.setup(P=P_sp, q=c.data, A=A_sp, l=b.data, u=b.data, **settings)
    results = solver.solve()

    x = Array(results.x, (Q, c, A, b), 'solve_qp')  # Solution
    y = Array(results.y, (Q, c, A, b), 'solve_qp')  # Lagrange multipliers

    def _backward():
        x_bar, y_bar = x.grad, y.grad

        # Solve the adjoint system using the same OSQP solver
        solver.update(q=-x_bar, l=y_bar, u=y_bar)
        adj_results = solver.solve()

        w_x, w_y = adj_results.x, adj_results.y

        # Compute the adjoint inputs
        Q_bar = -0.5 * (w_x @ x.data.T + x.data @ w_x.T)
        c_bar = -w_x
        A_bar = -(np.outer(w_y, x.data) + np.outer(y.data, w_x))
        b_bar = w_y

        # Accumulate the adjoint inputs in the gradients
        Q.grad += Q_bar
        c.grad += c_bar
        A.grad += A_bar
        b.grad += b_bar

    x._backward = _backward
    y._backward = _backward
    
    return x, y