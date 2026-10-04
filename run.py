"""End-to-end experiment. Usage:
    python run.py                  # real data via yfinance (cached after first run)
    python run.py --synthetic      # offline pipeline check on synthetic prices
"""
import argparse
import time
from pathlib import Path

import numpy as np

from regime.data import load_real, load_synthetic
from regime.features import FEATURES, Standardizer, build_dataset, time_split
from regime.labels import REGIMES
from regime.metrics import per_class, summary
from regime.softmax import SoftmaxRegression

K, OUT = len(REGIMES), Path(__file__).parent / "results"


def run_task(prices, horizon, l2):
    t0 = time.time()
    print(f"\n[horizon={horizon}] building features and labels ...", flush=True)
    data = build_dataset(prices, horizon)
    train, test = time_split(data, horizon)
    sc = Standardizer().fit(train[FEATURES].values)
    Xtr, Xte = sc.transform(train[FEATURES].values), sc.transform(test[FEATURES].values)
    ytr, yte = train["y"].values, test["y"].values
    N, rows, models = len(ytr), [], {}
    print(f"[horizon={horizon}] {len(ytr):,} train rows, {len(yte):,} test rows. Training (a few minutes) ...", flush=True)

    rows.append(("Majority class", summary(yte, np.full_like(yte, np.bincount(ytr).argmax()), K)))
    if horizon > 0:
        rows.append(("Persistence (today's regime)", summary(yte, test["y_now"].values, K)))

    from sklearn.linear_model import LogisticRegression
    for name, cw in [("", None), (" (balanced)", "balanced")]:
        print(f"[horizon={horizon}]   training{name or ' (unweighted)'} ... {time.time() - t0:.0f}s elapsed", flush=True)
        ours = SoftmaxRegression(K, lr=0.3, l2=l2, n_iter=6000, class_weight=cw).fit(Xtr, ytr)
        ref = LogisticRegression(C=1 / (l2 * N), max_iter=5000, class_weight=cw).fit(Xtr, ytr)
        rows.append((f"scikit-learn LogReg{name}", summary(yte, ref.predict(Xte), K)))
        rows.append((f"NumPy softmax (ours){name}", summary(yte, ours.predict(Xte), K)))
        models[cw] = (ours, ref)
    ours, ref = models[None]
    agree = float((ours.predict(Xte) == ref.predict(Xte)).mean())
    pdiff = float(np.abs(ours.predict_proba(Xte) - ref.predict_proba(Xte)).max())
    prec, rec, f1, M = per_class(yte, ours.predict(Xte), K)
    print(f"[horizon={horizon}] done in {time.time() - t0:.0f}s", flush=True)
    return dict(horizon=horizon, train=train, test=test, rows=rows, agree=agree, pdiff=pdiff,
                prec=prec, rec=rec, f1=f1, M=M, ours=ours, ytr=ytr, yte=yte)


def table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    return s + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def plots(res):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for r in res:
        h = r["horizon"]
        plt.figure(figsize=(5, 3.2)); plt.plot(r["ours"].history, color="#202020")
        plt.xlabel("iteration"); plt.ylabel("training loss"); plt.title(f"Gradient descent, horizon={h}")
        plt.tight_layout(); plt.savefig(OUT / f"loss_h{h}.png", dpi=150); plt.close()
        M = r["M"] / np.maximum(r["M"].sum(1, keepdims=True), 1)
        plt.figure(figsize=(5, 4.2)); plt.imshow(M, cmap="Greys", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > .5 else "#202020")
        plt.xticks(range(K), [x.replace("_", "\n") for x in REGIMES], fontsize=7)
        plt.yticks(range(K), [x.replace("_", " ") for x in REGIMES], fontsize=7)
        plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Confusion matrix (row-normalised), horizon={h}")
        plt.tight_layout(); plt.savefig(OUT / f"confusion_h{h}.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--l2", type=float, default=1e-3)
    a = ap.parse_args()
    prices = load_synthetic() if a.synthetic else load_real()
    print(f"{len(prices)} tickers loaded ({'SYNTHETIC' if a.synthetic else 'real'})")
    res = [run_task(prices, h, a.l2) for h in a.horizons]

    md = [f"# Results ({'SYNTHETIC data - pipeline check only, not a market result' if a.synthetic else 'real market data'})\n",
          f"Tickers: {len(prices)} | features: {len(FEATURES)} | L2 lambda: {a.l2} | split: chronological, last 25% of dates = test\n"]
    for r in res:
        h, tr, te = r["horizon"], r["train"], r["test"]
        task = "Nowcast: predict TODAY's rule-based regime from features" if h == 0 else f"Forecast: predict the regime {h} trading days AHEAD"
        dist = np.bincount(te["y"].values, minlength=K) / len(te)
        md += [f"\n## Task: {task}\n",
               f"Train rows: {len(tr):,} ({tr.index.min().date()} to {tr.index.max().date()}) | "
               f"Test rows: {len(te):,} ({te.index.min().date()} to {te.index.max().date()})\n",
               "Test class distribution: " + ", ".join(f"{n} {d:.1%}" for n, d in zip(REGIMES, dist)) + "\n",
               table(["Model", "Accuracy", "Macro-F1"], [(n, f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}") for n, m in r["rows"]]),
               f"\nCorrectness check, ours vs scikit-learn (same objective): predictions agree on {r['agree']:.2%} of test rows, "
               f"max probability difference {r['pdiff']:.1e}.\n",
               "\nPer-class results, NumPy softmax (unweighted):\n",
               table(["Regime", "Precision", "Recall", "F1"], [(REGIMES[i], f"{r['prec'][i]:.3f}", f"{r['rec'][i]:.3f}", f"{r['f1'][i]:.3f}") for i in range(K)]),
               "\nConfusion matrix (rows = true, columns = predicted):\n",
               table([""] + REGIMES, [[REGIMES[i]] + [str(v) for v in r["M"][i]] for i in range(K)]),
               f"\n![loss](loss_h{h}.png) ![confusion](confusion_h{h}.png)\n"]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf8")
    plots(res)
    print("\n" + "=" * 70 + "\n" + "\n".join(md))
    print(f"\nSaved: {OUT / 'results.md'} and plots in {OUT}")


if __name__ == "__main__":
    main()