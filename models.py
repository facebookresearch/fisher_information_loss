#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def compute_information_loss(J, target_attribute=None, constrained=False):
    """
    Compute the Fisher information loss, eta, from the largest eigenvalues
    of the unscaled Fisher information matrix for each sample.

    Arguments:
        target_attribute:  Used to specify a range of attributes
                           (default None for all attributes)
        constrained:       If True, constrain the attributes to sum to 1 by
                           multiplying FIM with an orthogonal matrix U
    """
    if target_attribute is not None:
        J = J[:, :, range(*target_attribute)]
    if constrained:
        d = J.size(2)
        assert d > 1, 'Cannot constrain 1-dimensional attribute vector'
        U = torch.vstack([torch.zeros(1, d-1), torch.ones(d-1, d-1).tril()])
        U /= -U.sum(0).unsqueeze(0)
        U += torch.vstack([torch.eye(d-1), torch.zeros(1, d-1)])
        U /= U.norm(2, 0).unsqueeze(0)
        J = J @ U
    return torch.linalg.norm(J, ord=2, dim=(1, 2))


def get_model(model_type):
    if type(model_type) is type:
        return model_type()

    if model_type == "least_squares":
        return LeastSquares()
    elif model_type == "logistic":
        return Logistic()
    raise ValueError(f"Unknown model type {model_type}")


def weighted_least_squares_jacobian(A, theta, X, y, w):
    """
    Jacobian of the remove and update function with respect to the update.
    """
    r = (X @ theta - y)[:, None, None]
    XA = X @ A
    JX = -(r * A.unsqueeze(0) + XA.unsqueeze(2) * theta[None, None, :])
    return w[:, None, None] * JX, w[:, None] * XA


def least_squares_jacobian(A, theta, X, y):
    """
    Jacobian of the remove and update function with respect to the update.
    """
    r = (X @ theta - y)[:, None, None]
    XA = X @ A
    JX = -(r * A.unsqueeze(0) + XA.unsqueeze(2) * theta[None, None, :])
    return JX, XA


class LeastSquares:

    def train(self, data, l2=0, weights=None):
        n = len(data["targets"])
        if weights is None:
            weights = torch.ones(n)
        assert len(weights) == n, "Invalid number of weights"

        # Save the weights for the jacobian:
        self.weights = weights

        X = data["features"]
        y = data["targets"].float()
        # [-1, 1] works much better for regression
        y[y == 0] = -1
        XTX = (weights[:, None] * X).T @ X
        XTXdiag = torch.diagonal(XTX)
        XTXdiag += (n * l2)
        b = X.T @ (weights * y)
        theta = torch.solve(b[:, None], XTX)[0].squeeze(1)
        # Need A to compute the Jacobian.
        A = torch.inverse(XTX)
        self.A = A
        self.theta = theta

    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta

    def predict(self, X, regression=False):
        """
        Given a data matrix X with examples as rows,
        returns a {0, 1} prediction for each x in X.
        """
        if regression:
            return X @ self.theta
        else:
            return (X @ self.theta) > 0

    def loss(self, data):
        """
        Evaluate the loss for each example in a given dataset.
        """
        X = data["features"]
        y = data["targets"].float()
        # [-1, 1] works much better for regression
        y[y == 0] = -1
        return (X @ self.theta - y)**2 / 2

    def influence_jacobian(self, data, weighted=True):
        """
        Compute the Jacobian of the influence of each
        example on the optimal parameters. The resulting
        Jacobian will have shape N x d x (d+1) where N is
        the number of data points.
        """
        X = data["features"]
        y = data["targets"].float()
        y[y == 0] = -1
        if weighted:
            JX, Jy = weighted_least_squares_jacobian(
                self.A, self.theta, X, y, self.weights)
        else:
            JX, Jy = least_squares_jacobian(
                self.A, self.theta, X, y)
        return torch.cat([JX, Jy.unsqueeze(2)], dim=2)


class Logistic:

    def train(self, data, l2=0, init=None, weights=None):
        n = len(data["targets"])
        if weights is None:
            weights = torch.ones(n)
        assert len(weights) == n, "Invalid number of weights"

        # Save for the jacobian:
        self.weights = weights
        self.l2 = n * l2

        X = data["features"]
        y = data["targets"].float()
        theta = torch.randn(X.shape[1], requires_grad=True)
        if init is not None:
            theta.data[:] = init[:]

        crit = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = (crit(X @ theta, y) * weights).sum()
            loss += (self.l2 / 2.0) * (theta**2).sum()
            loss.backward()
            return loss
        for _ in range(100):
            loss = optimizer.step(closure)
        self.theta = theta

    def get_params(self):
        return self.theta

    def set_params(self, theta):
        self.theta = theta

    def predict(self, X):
        """
        Given a data matrix X with examples as rows,
        returns a {0, 1} prediction for each x in X.
        """
        return (X @ self.theta) > 0

    def loss(self, data):
        X = data["features"]
        y = data["targets"].float()
        return torch.nn.BCEWithLogitsLoss(reduction="none")(X @ self.theta, y)

    def influence_jacobian(self, data):
        """
        Compute the Jacobian of the influence of each
        example on the optimal parameters. The resulting
        Jacobian will have shape N x d x (d+1) where N is
        the number of data points.
        """
        X = data["features"]
        y = data["targets"].float()

        # Compute the Hessian at theta for all X:
        s = (X @ self.theta).sigmoid().unsqueeze(1)
        H = (self.weights.unsqueeze(1) * s * (1-s) * X).T @ X
        Hdiag = torch.diagonal(H)
        Hdiag += self.l2
        Hinv = H.inverse()

        # Compute the Jacobian of the gradient w.r.t. theta at each (x, y) pair
        XHinv = X @ Hinv
        JX = -(s * (1-s) * XHinv).unsqueeze(2) * self.theta[None, None, :]
        JX -= (s - y.unsqueeze(1)).unsqueeze(2) * Hinv.unsqueeze(0)
        JX = self.weights[:, None, None] * JX
        JY =  (self.weights[:, None] * XHinv).unsqueeze(2)
        return torch.cat([JX, JY], dim=2)

class MultinomialLogistic:

    def train(self, data):
        X = data["features"]
        y = data["targets"]
        c = torch.max(y) + 1

        theta = torch.randn((X.shape[1], c), requires_grad=True)
        optimizer = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")
        crit = torch.nn.CrossEntropyLoss(reduction="mean")
        def closure():
            optimizer.zero_grad()
            loss = crit(X @ theta, y)
            loss.backward()
            return loss
        for _ in range(100):
            loss = optimizer.step(closure)

        self.theta = theta

    def predict(self, X):
        """
        Given a data matrix X with examples as rows,
        returns a {0, 1} prediction for each x in X.
        """
        return torch.argmax(X @ self.theta, axis=1)
