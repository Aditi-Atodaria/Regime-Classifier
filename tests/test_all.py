import numpy as np
import pandas as pd
import pytest

from regime.data import load_synthetic
from regime.features import FEATURES, build_dataset, build_features, time_split
from regime.labels import label_regimes
from regime.metrics import confusion_matrix, per_class
from regime.softmax import SoftmaxRegression, softmax


def make_blobs(n=600, d=5, K=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.5, (K, d))
    y = rng.integers(0, K, n)
    return centers[y] + rng.normal(0, 1.2, (n, d)), y


def test_softmax_rows_sum_to_one():
    P = softmax(np.random.default_rng(0).normal(size=(10, 4)) * 50)   # large logits: stability
    assert np.allclose(P.sum(1), 1) and np.isfinite(P).all()


@pytest.mark.parametrize("cw", [None, "balanced"])
def test_gradient_matches_finite_differences(cw):
    X, y = make_blobs(80, 4, 3)
    m = SoftmaxRegression(3, l2=0.05, class_weight=cw)
    Y, w = np.eye(3)[y], m._weights(y)
    rng = np.random.default_rng(1)
    W, b = rng.normal(0, 0.3, (4, 3)), rng.normal(0, 0.3, 3)
    _, gW, gb = m.loss_and_grad(W, b, X, Y, w)
    eps = 1e-6
    for idx in np.ndindex(*W.shape):
        Wp, Wm = W.copy(), W.copy()
        Wp[idx] += eps; Wm[idx] -= eps
        num = (m.loss_and_grad(Wp, b, X, Y, w)[0] - m.loss_and_grad(Wm, b, X, Y, w)[0]) / (2 * eps)
        assert abs(num - gW[idx]) < 1e-6
    for j in range(3):
        bp, bm = b.copy(), b.copy()
        bp[j] += eps; bm[j] -= eps
        num = (m.loss_and_grad(W, bp, X, Y, w)[0] - m.loss_and_grad(W, bm, X, Y, w)[0]) / (2 * eps)
        assert abs(num - gb[j]) < 1e-6


def test_loss_decreases():
    X, y = make_blobs()
    h = SoftmaxRegression(4, lr=0.2, l2=1e-3, n_iter=300).fit(X, y).history
    assert all(b <= a + 1e-12 for a, b in zip(h, h[1:]))


@pytest.mark.parametrize("cw", [None, "balanced"])
def test_matches_scikit_learn(cw):
    sk = pytest.importorskip("sklearn.linear_model")
    X, y = make_blobs(1500, 6, 4)
    X = (X - X.mean(0)) / X.std(0)
    lam = 1e-2
    ours = SoftmaxRegression(4, lr=0.5, l2=lam, n_iter=20000, class_weight=cw).fit(X, y)
    ref = sk.LogisticRegression(C=1 / (lam * len(y)), max_iter=5000, class_weight=cw).fit(X, y)
    assert (ours.predict(X) == ref.predict(X)).mean() > 0.995
    assert np.abs(ours.predict_proba(X) - ref.predict_proba(X)).max() < 1e-2


def test_metrics_match_scikit_learn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert (confusion_matrix(y, p, 4) == skm.confusion_matrix(y, p)).all()
    prec, rec, f1, _ = per_class(y, p, 4)
    assert np.allclose(f1, skm.f1_score(y, p, average=None))
    assert np.allclose(prec, skm.precision_score(y, p, average=None))


def test_labels_and_features_are_causal():
    """Values at date t must not change when future rows are removed."""
    df = next(iter(load_synthetic(1, 900, seed=3).values()))
    cut = 700
    pd.testing.assert_series_equal(label_regimes(df).iloc[:cut], label_regimes(df.iloc[:cut]), check_names=False)
    pd.testing.assert_frame_equal(build_features(df).iloc[:cut], build_features(df.iloc[:cut]))


@pytest.mark.parametrize("h", [0, 5, 20])
def test_time_split_has_no_overlap(h):
    data = build_dataset(load_synthetic(3, 1200, seed=1), h)
    train, test = time_split(data, h)
    dates = np.sort(data.index.unique())
    pos = {d: i for i, d in enumerate(dates)}
    assert train.index.max() < test.index.min()
    assert pos[train.index.max()] + h < pos[test.index.min()]     # target windows never reach the test period
