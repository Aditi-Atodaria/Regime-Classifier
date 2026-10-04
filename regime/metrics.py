"""Classification metrics implemented from scratch."""
import numpy as np


def confusion_matrix(y, p, K):
    M = np.zeros((K, K), dtype=int)
    np.add.at(M, (y, p), 1)          # rows = true, cols = predicted
    return M


def per_class(y, p, K):
    M = confusion_matrix(y, p, K)
    tp = np.diag(M).astype(float)
    prec, rec = tp / np.maximum(M.sum(0), 1), tp / np.maximum(M.sum(1), 1)
    return prec, rec, 2 * prec * rec / np.maximum(prec + rec, 1e-12), M


def summary(y, p, K):
    return {"accuracy": float((y == p).mean()), "macro_f1": float(per_class(y, p, K)[2].mean())}
