"""Softmax regression (multi-class logistic regression) in plain NumPy.

The idea, step by step:
  1. Score:       every class gets a score from the features (X @ W + b).
  2. Softmax:     turn the scores into probabilities that add up to 1.
  3. Loss:        measure how much probability the model gave the TRUE class
                  (cross-entropy), plus a small penalty on big weights (L2).
  4. Gradient:    work out how to change W and b to reduce the loss.
                  For softmax this is simply:  (predicted probability - true label).
  5. Descent:     take a small step against the gradient, and repeat.
"""
import numpy as np


def softmax(scores):
    """Turn scores into probabilities. Rows of the result sum to 1."""
    scores = scores - scores.max(axis=1, keepdims=True)   # avoids overflow in exp()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.3, l2=1e-3, n_iter=4000, class_weight=None, tol=1e-10):
        self.K = n_classes               # number of classes (4 regimes)
        self.lr = lr                     # learning rate: size of each step
        self.l2 = l2                     # strength of the penalty on large weights
        self.n_iter = n_iter             # maximum number of steps
        self.class_weight = class_weight # None, or "balanced" to up-weight rare classes
        self.tol = tol                   # stop when the loss barely changes

    def loss_and_grad(self, W, b, X, Y, w):
        """Return the loss and its gradients for W and b.

        X: features, Y: one-hot true labels, w: weight of each sample.
        """
        total_weight = w.sum()
        P = softmax(X @ W + b)                                  # predicted probabilities

        # Loss: -log(probability given to the true class), averaged, plus the L2 penalty
        prob_true_class = np.clip((P * Y).sum(axis=1), 1e-15, None)
        cross_entropy = -np.log(prob_true_class)
        loss = (w * cross_entropy).sum() / total_weight + 0.5 * self.l2 * np.sum(W * W)

        # Gradient: (predicted - true), scaled by sample weights
        error = w[:, None] * (P - Y)
        grad_W = X.T @ error / total_weight + self.l2 * W       # + l2 * W comes from the penalty
        grad_b = error.sum(axis=0) / total_weight               # the bias is not penalised
        return loss, grad_W, grad_b

    def _weights(self, y):
        """Sample weights. "balanced" gives rare classes more weight so they count equally."""
        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=self.K).astype(float)
            per_class = len(y) / (self.K * np.maximum(counts, 1))
            return per_class[y]
        return np.ones(len(y))

    def fit(self, X, y):
        """Train with gradient descent on the whole training set at each step."""
        Y = np.eye(self.K)[y]                       # one-hot labels
        w = self._weights(y)
        self.W = np.zeros((X.shape[1], self.K))     # start from zero (the loss has one minimum)
        self.b = np.zeros(self.K)
        self.history = []                           # loss at each step, used for the loss plot

        previous_loss = np.inf
        for _ in range(self.n_iter):
            loss, grad_W, grad_b = self.loss_and_grad(self.W, self.b, X, Y, w)
            self.history.append(loss)
            if abs(previous_loss - loss) < self.tol:    # loss has stopped changing: done
                break
            previous_loss = loss
            self.W -= self.lr * grad_W                  # step against the gradient
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        """Probability of each class for each row."""
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        """The class with the highest probability."""
        return self.predict_proba(X).argmax(axis=1)
