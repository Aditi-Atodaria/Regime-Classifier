"""Multinomial logistic (softmax) regression in plain NumPy.

Objective (derivation in README):
    J(W, b) = (1/S) * sum_i w_i * CE_i  +  (lambda/2) * ||W||_F^2          S = sum_i w_i
    dJ/dW   = (1/S) * X^T [ w * (P - Y) ] + lambda * W
    dJ/db   = (1/S) * sum_i w_i (p_i - y_i)            (bias is not regularised)
"""
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.3, l2=1e-3, n_iter=4000, class_weight=None, tol=1e-10):
        self.K, self.lr, self.l2, self.n_iter, self.class_weight, self.tol = n_classes, lr, l2, n_iter, class_weight, tol

    def loss_and_grad(self, W, b, X, Y, w):
        S = w.sum()
        P = softmax(X @ W + b)
        ce = -np.log(np.clip((P * Y).sum(1), 1e-15, None))
        loss = (w * ce).sum() / S + 0.5 * self.l2 * np.sum(W * W)
        G = w[:, None] * (P - Y)
        return loss, X.T @ G / S + self.l2 * W, G.sum(0) / S

    def _weights(self, y):
        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=self.K).astype(float)
            return (len(y) / (self.K * np.maximum(counts, 1)))[y]
        return np.ones(len(y))

    def fit(self, X, y):
        """Full-batch gradient descent."""
        Y, w = np.eye(self.K)[y], self._weights(y)
        self.W, self.b = np.zeros((X.shape[1], self.K)), np.zeros(self.K)   # convex: zero init is fine
        self.history, prev = [], np.inf
        for _ in range(self.n_iter):
            loss, gW, gb = self.loss_and_grad(self.W, self.b, X, Y, w)
            self.history.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
            self.W -= self.lr * gW
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
