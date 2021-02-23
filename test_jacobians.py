#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import unittest

from models import least_squares_jacobian
from models import weighted_least_squares_jacobian


def least_squares_jacobian_single(A, theta, x, y):
    """
    Jacobian of the remove and update function with respect to the update.
    """
    r = (theta.dot(x) - y)
    Ax = A @ x
    return -(r * A + Ax.ger(theta)), Ax


def weighted_least_squares_jacobian_single(A, theta, x, y, w):
    """
    Jacobian of the remove and update function with respect to the update for
    weighted least squares.
    """
    sqw = math.sqrt(w)
    x = sqw * x
    y = sqw * y
    r = (theta.dot(x) - y)
    Ax = A @ x
    return -sqw * (r * A + Ax.ger(theta)), sqw * Ax


def least_squares_update(A, theta, x, y, w=1.0):
    """
    Updates a given solution to the least squares problem to incorporate the
    new data point (x, y).
    A is `(X^T X)^{-1}` and theta are the optimal parameters (`A X^T y`)
    """
    x = math.sqrt(w) * x
    y = math.sqrt(w) * y
    Ax = A @ x
    c = 1 / (1 + x.T @ Ax)
    return theta - c * (x.T @ theta - y) * Ax


def least_squares_remove_and_update(A, theta, X, Y, xr, yr, xn, yn):
    Au = rank_two_update(A, xr, xn, steps=[-1, 1])
    return Au @ (X.T @ Y - xr * yr + xn * yn)


def least_squares_jacobian_update(A, theta, x, y):
    """
    Jacobian of the update function with respect to the update.
    """
    Ax = A @ x
    c = 1 / (1 + x.T @ Ax)
    r = x.dot(theta) - y
    t1 = ((2 * r * c**2) * Ax).ger(Ax)
    t2 = -r * c * A
    t3 = -c *  Ax.ger(theta)
    Jx = t1 + t2 + t3
    Jy = c * Ax
    return Jx, Jy


def rank_one_update(A, x, step=1):
    """
    Compute the updated A when applying the rank-one update of x on A^{-1}.
    E.g. compute `(A^{-1} + step * x x^T)^{-1}` using the Sherman-Morrison formula.
    """
    Ax = A @ x
    c = 1 / (step + x.dot(Ax))
    return A - c * Ax.ger(Ax)


def rank_two_update(A, x1, x2, steps=[1, 1]):
    steps = torch.tensor([1 / s for s in steps])
    U = torch.stack([x1, x2], dim=1)
    AU = A @ U
    D = (U.T @ AU + torch.diag(steps)).inverse()
    return A - AU @ D @ AU.T


class TestJacobian(unittest.TestCase):

    def test_rank_one_update(self):
        d = 10
        A = torch.rand(d, d)
        x = torch.rand(d)
        Aminus = rank_one_update(A, x, step=-1)
        Aplus = rank_one_update(Aminus, x, step=1)
        self.assertTrue(torch.allclose(A, Aplus, rtol=1e-4, atol=1e-4))


    def test_rank_two_update(self):
        d = 10
        A = torch.rand(d, d)
        A = (A + A.T) + 5 * torch.eye(10)
        x1 = torch.rand(d)
        x2 = torch.rand(d)
        Atwo = rank_two_update(A, x1, x2, steps=[1, 1])
        Aone_one = rank_one_update(rank_one_update(A, x1), x2)
        self.assertTrue(torch.allclose(Aone_one, Atwo, rtol=1e-4, atol=1e-4))

        Aminus = rank_two_update(A, x1, x2, steps=[1, 1])
        Aplus = rank_two_update(Aminus, x1, x2, steps=[-1, -1])
        # TODO this seems to be too unstable..
        self.assertTrue(torch.allclose(A, Aplus, rtol=1e-2, atol=1e-2))


    def test_jacobians(self):
        # Make a random sample:
        d = 10
        n = 20
        X = torch.randn(n, d)
        Y = torch.randn(n)

        # Find least squares solution for the full dataset:
        A = torch.inverse(X.T @ X)
        theta = A @ (X.T @ Y)

        xi = X[0, :]
        yi = Y[0]

        # Method 1:, Compute Jacobian w.r.t. x_i, y_i by
        # 1. Removing x_i, y_i from A and theta
        # 2. Expressing theta* as a function of x, y via a rank-one update
        # 3. Computing the Jacobian of 2 at x_i, y_i
        A_minus = rank_one_update(A, xi, step=-1)
        theta_minus = A_minus @ (X.T @ Y - xi * yi)
        def f_x_y(x, y):
            return least_squares_update(A_minus, theta_minus, x, y)

        # Using torch autograd:
        Jx_auto, Jy_auto = torch.autograd.functional.jacobian(f_x_y, (xi, yi))

        # Using closed form:
        Jx, Jy = least_squares_jacobian_update(A_minus, theta_minus, xi, yi)

        self.assertTrue(torch.allclose(Jy.squeeze(), Jy_auto.squeeze(), rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(Jx, Jx_auto, rtol=1e-4, atol=1e-4))

        # Method 2: Compute Jacobian w.r.t. x_i, y_i by
        # 1. Expressing theta* as a function of removing x_i, y_i and adding in
        # x, y via a rank-two update
        # 2. Computing the Jacobian of 2 at x, y
        def f_x_y(x, y):
            return least_squares_remove_and_update(A, theta, X, Y, xi, yi, x, y)
        Jx_auto2, Jy_auto2 = torch.autograd.functional.jacobian(f_x_y, (xi, yi))
        self.assertTrue(torch.allclose(Jy_auto.squeeze(), Jy_auto2.squeeze(), rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(Jx_auto, Jx_auto2, rtol=1e-4, atol=1e-4))

        # Using closed form:
        Jx, Jy = least_squares_jacobian_single(A, theta, xi, yi)
        self.assertTrue(torch.allclose(Jy, Jy_auto, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(Jx, Jx_auto, rtol=1e-4, atol=1e-4))

    def test_weighted_jacobian(self):
        # Make a random sample:
        d = 10
        n = 20
        W = torch.diag(torch.ones(n))
        X = torch.randn(n, d)
        Y = torch.randn(n)

        for w in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:

            W[0, 0] = w

            # Find least squares solution for the full dataset:
            A = torch.inverse(X.T @ W @ X)
            theta = A @ (X.T @ W @ Y)

            xi = X[0, :]
            yi = Y[0]

            A_minus = rank_one_update(A, math.sqrt(w) * xi, step=-1)
            theta_minus = A_minus @ (X.T @ W @ Y - w * xi * yi)
            def f_x_y(x, y):
                return least_squares_update(A_minus, theta_minus, x, y, w)

            # Using torch autograd:
            Jx_auto, Jy_auto = torch.autograd.functional.jacobian(f_x_y, (xi, yi))

            # Using closed form:
            Jx, Jy = weighted_least_squares_jacobian_single(A, theta, xi, yi, w=w)
            self.assertTrue(torch.allclose(Jy, Jy_auto, rtol=1e-4, atol=1e-4))
            self.assertTrue(torch.allclose(Jx, Jx_auto, rtol=1e-4, atol=1e-4))


    def test_batched_jacobian(self):
        d = 10
        n = 20
        X = torch.randn(n, d)
        Y = torch.randn(n)

        # Find least squares solution for the full dataset:
        A = torch.inverse(X.T @ X)
        theta = A @ (X.T @ Y)

        batchJx, batchJy = least_squares_jacobian(A, theta, X, Y)

        singleJs = [least_squares_jacobian_single(A, theta, x, y) for x, y in zip(X, Y)]
        singleJx, singleJy = zip(*singleJs)
        self.assertTrue(torch.allclose(batchJx, torch.stack(singleJx), rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(batchJy, torch.stack(singleJy), rtol=1e-4, atol=1e-4))

        # With weights:
        W = torch.rand(n) * 5
        A = (W.unsqueeze(1) * X).T @ X
        theta = A @ (X.T @ (W * Y))

        batchJx, batchJy = weighted_least_squares_jacobian(A, theta, X, Y, W)

        singleJs = [weighted_least_squares_jacobian_single(A, theta, x, y, w) for x, y, w in zip(X, Y, W)]
        singleJx, singleJy = zip(*singleJs)
        self.assertTrue(torch.allclose(batchJx, torch.stack(singleJx), rtol=1e-3, atol=1e-4))
        self.assertTrue(torch.allclose(batchJy, torch.stack(singleJy), rtol=1e-4, atol=1e-4))


    def test_logistic_jacobian(self):
        def grad(w, x, y, l2):
            s = torch.sigmoid(w @ x)
            return x @ (s - y) + x.shape[1] * l2 * w

        def Hinv(w, x, l2):
            s = torch.sigmoid(w @ x)
            H = s * (1 - s) * x @ x.T + x.shape[1] * l2 * torch.eye(x.shape[0])
            return torch.inverse(H)

        def solve(x, y, l2, its=30):
            w = 1e-1*torch.randn(x.shape[0], dtype=torch.double)
            for it in range(its):
                w = w - Hinv(w, x, l2=l2) @ grad(w, x, y, l2=l2)
            assert grad(w, x, y, l2).norm().item() < 1e-12, "Did not converge."
            return w

        def compute_Jf_exact(w, x, y, l2, i):
            xi = x[:, i]
            yi = y[i]
            si = torch.sigmoid(w.dot(xi))
            nabla_xyw = si * (si - 1) * xi.ger(w) + (yi - si) * torch.eye(xi.shape[0])
            Hi = Hinv(w, x, l2)
            return Hi @ nabla_xyw

        def compute_Jf_fd(x, y, l2, i, epsilon=1e-6):
            Jf_fd = []
            for j in range(x.shape[0]):
                x[j, i] += epsilon
                w_up = solve(x, y, l2)
                x[j, i] -= 2*epsilon
                w_down = solve(x, y, l2)
                Jf_fd.append((w_up - w_down) / (2* epsilon))
                x[j, i] += epsilon
            return torch.stack(Jf_fd, dim=1)

        torch.random.manual_seed(123)
        d = 4
        n = 40
        x = torch.randn((d, n), dtype=torch.double)
        y = torch.randint(high=1, size=(n,))
        l2 = 1e-5

        # Compute Jacobian at x_0 with finite differences
        Jf_fd = compute_Jf_fd(x, y, l2, 0)

        # Compute Jacobian at x_0 analytically
        w_star = solve(x, y, l2)
        Jf = compute_Jf_exact(w_star, x, y, l2, 0)

        self.assertTrue(torch.allclose(Jf, Jf_fd, rtol=1e-7, atol=1e-7))


# run all the tests:
if __name__ == '__main__':
    unittest.main()
