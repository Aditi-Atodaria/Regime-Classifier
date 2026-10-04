# Regime Classifier: softmax regression from scratch

A small machine-learning project: classify a stock's **market regime** (trending up, trending down, volatile, range-bound) from technical features, using a **softmax regression classifier written in plain NumPy**. I wrote the loss, the gradient, the L2 penalty and gradient descent myself, then compared the model against baselines and scikit-learn.

The focus is on how the machine learning works and on evaluating it honestly, not on building a trading product.

## Summary of results

- **Nowcast works, as expected.** 77.2% accuracy and 0.705 macro-F1 on held-out data (May 2024 to Oct 2026), against 55.9% and 0.179 for the majority class. The labels come from rules on the same indicators the model sees, so this shows the model recovering those rules, not real market insight.
- **Forecast does not beat persistence.** For the regime 5 trading days ahead, the model scores 0.727 accuracy and 0.630 macro-F1. Simply predicting that today's regime continues scores 0.763 and 0.727.
- **The implementation is correct.** Predictions agree with scikit-learn on 99.97% of test rows, and the gradient matches a numerical check.
- **Weak spot:** the volatile regime. Recall is 0.27 (nowcast) and 0.17 (forecast), because a linear model handles threshold-style rules poorly.

## Read this first: what the labels are

The regime labels come from **hand-written rules** (`regime/labels.py`), not from ground truth. The model learns to reproduce or anticipate *those rules*. It does not discover "real" market states.

That is why there are two tasks:

| Task | Question | What it really measures |
|---|---|---|
| **Nowcast** (`horizon=0`) | Given today's features, what is today's rule-based regime? | How well a *linear* model can approximate threshold rules. The features overlap with the rule inputs (ADX, ATR), so a good score here is expected and says little about markets. |
| **Forecast** (`horizon=5`) | Given today's features, what will the regime be 5 trading days from now? | The harder, more honest question. It must beat the **persistence** baseline ("tomorrow = today"). |

## Quick start

```bash
python -m venv venv && venv\Scripts\activate      # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # 11 tests: gradient check, scikit-learn comparison, leakage checks
python run.py                      # real data (30 stocks via yfinance), writes results/results.md + plots
python run.py --synthetic          # offline pipeline check on synthetic prices (NOT a market result)
```

## Pipeline

1. **Data**: 30 liquid stocks (NSE and US), daily OHLCV (`regime/data.py`), cached to CSV. Usable rows run from April 2017 to October 2026.
2. **Indicators** (`regime/indicators.py`): RSI, MACD, ATR, ADX/DI, Bollinger Bands, EMA, written from their definitions.
3. **Labels**: four rule-based regimes. Every label at time *t* uses only data up to *t* (a unit test checks this).
4. **Features** (15): returns (1, 5, 20 days), volatility (10, 20 days), RSI, MACD histogram, Bollinger %B and width, volume ratio, distance to EMA50/EMA200, ADX, DI difference, ATR%.
5. **Time-based split**: first 75% of dates train, last 25% test, with a gap between them (explained below).
6. **Standardise** features using the *training* mean and standard deviation only.
7. **Model**: softmax regression trained with gradient descent (L2 strength 0.001, optional class weights).
8. **Evaluate**: accuracy, macro-F1, per-class precision and recall, confusion matrix, against majority-class, persistence and scikit-learn.

## How the model works (in plain words)

The code is in `regime/softmax.py`. The model has five steps.

### 1. Score each regime
For each day, the model multiplies the 15 features by a set of weights and adds a bias, giving one **score per regime**. A higher score means the model leans towards that regime. The weights start at zero and are learned from the data.

### 2. Turn scores into probabilities (softmax)
Scores can be any number, so softmax converts them into probabilities that are positive and add up to 1. Example with four regimes:

| | Trending up | Trending down | Volatile | Range-bound |
|---|---|---|---|---|
| Score | 2.0 | 1.0 | 0.1 | 0.1 |
| Probability | 0.60 | 0.22 | 0.09 | 0.09 |

The prediction is the regime with the highest probability. In the code, we subtract the largest score before the calculation so that big numbers cannot overflow.

### 3. Measure how wrong the model is (loss)
The loss looks at the probability the model gave to the **true** regime, and is large when that probability is small. It is `-log(probability of the true regime)`:

- True regime got probability 0.90, loss is about 0.11 (good).
- True regime got probability 0.22, loss is about 1.51 (bad).

The average loss over all training rows is what we try to make small.

### 4. Find which way to change the weights (gradient)
The gradient tells us how the loss changes if we nudge each weight. For softmax it has a very simple form: for each regime, take the **predicted probability minus the true label**, where the true label is 1 for the correct regime and 0 for the others.

Using the example above, if the true regime is "trending down":

| | Trending up | Trending down | Volatile | Range-bound |
|---|---|---|---|---|
| Predicted | 0.60 | 0.22 | 0.09 | 0.09 |
| True label | 0 | 1 | 0 | 0 |
| Predicted minus true | +0.60 | -0.78 | +0.09 | +0.09 |

The training step moves against this, so it raises the score for the true regime and lowers the others. The bigger the mistake, the bigger the push. The weights' gradient is these values multiplied by the feature values and averaged over all rows.

### 5. Repeat in small steps (gradient descent)
Imagine walking downhill in fog: you feel which way the ground slopes and take a small step that way, many times. Each step is:

```
weights = weights - learning_rate * gradient
```

The learning rate (0.3 here) sets the step size. We repeat up to several thousand times, and stop early when the loss stops changing. The loss curve plot shows it going down.

### How I checked the gradient
I did not just trust the formula. For each weight, the test nudges it up and down by a tiny amount, sees how the loss changes, and compares that slope with the formula's answer. They agree to better than 0.000001, with and without class weights (`tests/test_all.py`). The tests also check that the loss never goes up during training.

### L2 regularisation
Here the loss gets an extra small penalty for large weights. This keeps the model simple and stops weights growing very large. It helps here because some features are similar to each other (`vol_10`, `vol_20`, `atr_pct`), and without a penalty the model could give such features huge opposite weights that cancel out. The strength of the penalty is set by the number 0.001.

### Class weights
Range-bound days are over half the data, so a model can score well by favouring them. With `class_weight="balanced"`, rare regimes count for more in the loss so every regime matters equally. This improves macro-F1 a little but lowers accuracy, so both are reported.

### Comparing with scikit-learn
scikit-learn has a ready-made logistic regression. I use it only as a check. Its penalty strength is set to match mine (`C = 1 / (lambda * N)`), so both models solve the same problem. If my code is correct, the predictions should almost always agree, and they do (99.97%). The small remaining difference comes from scikit-learn using a different optimiser, so the two stop at very slightly different weights.

## Evaluating honestly

- **No random split.** Daily data is strongly autocorrelated, so tomorrow looks like today. A random split would put near-copies of every test row in the training set and inflate the scores. The split is by date.
- **A gap for the forecast task.** A training row's label looks 5 days ahead, so training rows within 5 days of the test start are dropped. This stops training labels from being built from test-period prices (a test checks it).
- **Scaler fit on training data only**, so test data never influences preprocessing.
- **Causality tests:** removing future rows must not change past labels or features.
- **Baselines:** majority class, persistence (forecast task) and scikit-learn.
- **Macro-F1 and per-class recall**, not just accuracy, because one regime dominates.

## Results

Real data: 30 tickers, 15 features, L2 strength 0.001, chronological split (last 25% of dates are the test set). Full tables and plots are in `results/`.

| Task | Train rows | Test rows | Test period |
|---|---|---|---|
| Nowcast | 52,840 (2017-04-12 to 2024-05-16) | 15,600 | 2024-05-17 to 2026-10-02 |
| Forecast (5d) | 52,600 (2017-04-12 to 2024-05-06) | 15,640 | 2024-05-14 to 2026-09-25 |

Test class mix (nowcast): range-bound 55.9%, volatile 18.8%, trending up 18.2%, trending down 7.0%. The forecast test set is almost identical.

### Headline comparison

| Task | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Nowcast | Majority class | 0.559 | 0.179 |
| Nowcast | scikit-learn LogReg | 0.772 | 0.705 |
| Nowcast | NumPy softmax (ours) | 0.772 | 0.705 |
| Nowcast | NumPy softmax, class-balanced | 0.744 | 0.723 |
| Forecast (5d) | Majority class | 0.564 | 0.180 |
| Forecast (5d) | **Persistence (today's regime)** | **0.763** | **0.727** |
| Forecast (5d) | scikit-learn LogReg | 0.728 | 0.630 |
| Forecast (5d) | NumPy softmax (ours) | 0.727 | 0.630 |
| Forecast (5d) | NumPy softmax, class-balanced | 0.669 | 0.640 |

scikit-learn and our model give identical scores at this precision, in both the unweighted and balanced runs.

### Check against scikit-learn

Predictions agree on **99.97%** of nowcast test rows and **99.96%** of forecast test rows. The largest difference between the two models' probabilities is 0.0085 (nowcast) and 0.0051 (forecast), most likely because gradient descent stops slightly short of the exact optimum that scikit-learn's optimiser finds. The predictions agree almost everywhere, but the two are not numerically identical.

### Per-class results (NumPy softmax, unweighted)

Nowcast:

| Regime | Precision | Recall | F1 |
|---|---|---|---|
| Trending up | 0.850 | 0.855 | 0.853 |
| Trending down | 0.785 | 0.716 | 0.749 |
| Volatile | 0.677 | 0.269 | 0.385 |
| Range-bound | 0.760 | 0.921 | 0.833 |

Forecast (5 days ahead):

| Regime | Precision | Recall | F1 |
|---|---|---|---|
| Trending up | 0.758 | 0.770 | 0.764 |
| Trending down | 0.729 | 0.629 | 0.675 |
| Volatile | 0.649 | 0.173 | 0.273 |
| Range-bound | 0.725 | 0.908 | 0.806 |

### Confusion matrices (rows = true, columns = predicted)

Nowcast:

|  | Trending up | Trending down | Volatile | Range-bound |
|---|---|---|---|---|
| **Trending up** | 2434 | 0 | 29 | 383 |
| **Trending down** | 0 | 781 | 65 | 245 |
| **Volatile** | 107 | 132 | 791 | 1909 |
| **Range-bound** | 322 | 82 | 283 | 8037 |

Forecast (5 days ahead):

|  | Trending up | Trending down | Volatile | Range-bound |
|---|---|---|---|---|
| **Trending up** | 2179 | 3 | 51 | 596 |
| **Trending down** | 6 | 698 | 26 | 379 |
| **Volatile** | 209 | 118 | 500 | 2061 |
| **Range-bound** | 480 | 139 | 194 | 8001 |

![Nowcast loss](results/loss_h0.png) ![Nowcast confusion matrix](results/confusion_h0.png)

![Forecast loss](results/loss_h5.png) ![Forecast confusion matrix](results/confusion_h5.png)

### What the results show

- **Nowcast:** the model beats the majority-class baseline by a wide margin, but this task is close to recovering the labelling rules, so it says little about markets.
- **Forecast:** the model does **not** beat persistence (macro-F1 0.630 vs 0.727). These rule-based regimes tend to continue, and today's regime is a better 5-day-ahead guess than anything this linear model learns from the features.
- **Volatile regime:** most volatile days are predicted as range-bound (1,909 of 2,939 in the nowcast, 2,061 of 2,888 in the forecast). Class weighting raises macro-F1 a little (0.723 and 0.640) but costs accuracy (0.744 and 0.669).
- **Next step:** a non-linear model (a small tree ensemble or neural network) would likely handle the threshold rules better.

## Limitations

- Labels are rule-derived. Nothing here shows the regimes are economically meaningful or tradable.
- A linear model cannot represent the threshold rules exactly; a tree or small neural network would likely do better on the nowcast.
- The forecast model does not beat a trivial persistence baseline, so it has no demonstrated predictive value.
- No transaction costs, no backtest: this is classification, not a trading strategy, and **not financial advice**.
- Survivorship bias: today's liquid large-caps, not a historical universe.
- Pooled across stocks with one shared model; per-stock or per-market models are untested.

## Layout

```
regime/
  data.py         yfinance loader (cached) + synthetic regime-switching generator
  indicators.py   RSI, MACD, ATR, ADX/DI, Bollinger, EMA
  labels.py       rule-based regime labels (causal)
  features.py     features, dataset, time split with gap, standardiser
  softmax.py      the from-scratch model: loss, gradient, gradient descent
  metrics.py      confusion matrix, precision/recall/F1
tests/test_all.py gradient check, scikit-learn comparison, causality, split checks
results/          results.md, loss curves and confusion matrices
run.py            end-to-end experiment -> results/
```
