# Regime Classifier: softmax regression from scratch

A small, fully explained machine-learning project: classify a stock's **market regime** (trending up, trending down, volatile, range-bound) from technical features, using a **multinomial logistic regression written in plain NumPy** (loss, gradient, L2 regularisation and gradient descent all hand-derived), then check it honestly against baselines and scikit-learn.

This is the simpler, from-the-ground-up companion to **StockIQ**, a full-stack regime-detection and strategy platform. Here the focus is the machine learning and the maths, not the product.

## Read this first: what the labels are

The regime labels come from **hand-written rules** (`regime/labels.py`), not from ground truth. The model learns to reproduce or anticipate *those rules*. It does not discover "real" market states.

That is why there are two tasks:

| Task | Question | What it really measures |
|---|---|---|
| **Nowcast** (`horizon=0`) | Given today's features, what is today's rule-based regime? | How well a *linear* model approximates threshold rules. The features overlap with the rule inputs (ADX, ATR), so a good score here is expected and says little about markets. |
| **Forecast** (`horizon=5`) | Given today's features, what will the regime be 5 trading days from now? | The harder, more honest question. It must beat the **persistence** baseline ("tomorrow = today"). |

## Quick start

```bash
python -m venv venv && venv\Scripts\activate      # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # 11 tests: gradients, sklearn parity, leakage checks
python run.py                      # real data (30 stocks via yfinance), writes results/results.md + plots
python run.py --synthetic          # offline pipeline check on synthetic prices (NOT a market result)
```

## Pipeline

1. **Data**: 30 liquid stocks (NSE and US), 10 years of daily OHLCV (`regime/data.py`), cached to CSV.
2. **Indicators** (`regime/indicators.py`): RSI, MACD, ATR, ADX/DI (Wilder), Bollinger, EMA, written from their definitions.
3. **Labels**: four rule-based regimes. Every label at time *t* uses only data up to *t* (a unit test checks this).
4. **Features** (15): returns (1/5/20d), volatility (10/20d), RSI, MACD histogram, Bollinger %B and width, volume ratio, distance to EMA50/200, ADX, DI difference, ATR%.
5. **Time-based split**: first 75% of dates train, last 25% test, with a **purge gap** of `horizon` days (see below).
6. **Standardise** features using *training* mean and std only.
7. **Model**: softmax regression, full-batch gradient descent, optional class weights.
8. **Evaluate**: accuracy, macro-F1, per-class precision/recall, confusion matrix, against majority-class, persistence, and scikit-learn.

## The maths

### Model
For features $x \in \mathbb{R}^d$ and $K$ classes, with weights $W \in \mathbb{R}^{d \times K}$ and bias $b \in \mathbb{R}^K$:

$$z = W^\top x + b, \qquad p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

(implemented with the max-subtraction trick, `z - z.max()`, so large logits cannot overflow).

### Loss
For one sample with one-hot label $y$, the cross-entropy is $\ell = -\sum_k y_k \log p_k$. With sample weights $w_i$ and $S=\sum_i w_i$, plus an L2 penalty on $W$ (not on $b$):

$$J(W,b) = \frac{1}{S}\sum_i w_i\,\ell_i \;+\; \frac{\lambda}{2}\lVert W\rVert_F^2$$

### Gradient, derived by hand
First the softmax Jacobian. Since $p_k = e^{z_k}/\sum_j e^{z_j}$:

$$\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$$

Then, with the chain rule on $\ell = -\sum_k y_k \log p_k$:

$$\frac{\partial \ell}{\partial z_j} = -\sum_k \frac{y_k}{p_k}\,p_k(\delta_{kj} - p_j) = -\sum_k y_k\delta_{kj} + p_j\sum_k y_k = -y_j + p_j$$

(using $\sum_k y_k = 1$). So the gradient with respect to the logits is simply **prediction minus label**, $p - y$. Since $z_j = w_j^\top x + b_j$:

$$\frac{\partial \ell}{\partial w_j} = (p_j - y_j)\,x, \qquad \frac{\partial \ell}{\partial b_j} = p_j - y_j$$

Averaging over the data and adding the penalty's derivative gives the matrix form used in the code:

$$\nabla_W J = \frac{1}{S}X^\top\big[w \odot (P - Y)\big] + \lambda W, \qquad \nabla_b J = \frac{1}{S}\sum_i w_i (p_i - y_i)$$

**Verification:** `tests/test_all.py` compares this analytic gradient with central finite differences, $\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$, for every entry of $W$ and $b$ (agreement better than $10^{-6}$), with and without class weights.

### Gradient descent
$$W \leftarrow W - \eta\,\nabla_W J,\qquad b \leftarrow b - \eta\,\nabla_b J$$

The objective is convex, so zero initialisation is fine and descent reaches the global minimum for a small enough learning rate $\eta$. The tests also check that the loss never increases.

### Why L2 regularisation works
1. **Bayesian view.** Minimising loss plus $\frac{\lambda}{2}\lVert W\rVert^2$ is MAP estimation with a Gaussian prior $W \sim \mathcal{N}(0,\sigma^2 I)$, where $\lambda = 1/(N\sigma^2)$. It encodes the belief that most weights are small.
2. **Weight decay.** The penalty adds $\lambda W$ to the gradient, so each step becomes $W \leftarrow (1-\eta\lambda)W - \eta\,\nabla J_{\text{data}}$: weights shrink a little every step unless the data pushes back.
3. **Well-posedness.** Without the penalty, softmax is not identifiable (adding the same vector to every class's weights changes nothing) and on separable data the weights grow without bound. The penalty adds $\lambda I$ to the Hessian, making the problem **strongly convex** with a unique solution and linear convergence.
4. **Correlated features.** Several features are nearly collinear (`vol_10`, `vol_20`, `atr_pct`). L2 shrinks the weights on such directions instead of letting them blow up, which is exactly the ridge-regression effect.

### Class weights
For imbalanced regimes (range-bound dominates), `class_weight="balanced"` uses $w_i = N/(K\,n_{y_i})$ so each class contributes equally. This trades accuracy for recall on rare regimes, so both versions are reported.

### Matching scikit-learn
scikit-learn minimises $C\sum_i \ell_i + \tfrac12\lVert W\rVert^2$. Dividing by $CN$ gives our objective with $\lambda = 1/(CN)$, so we pass `C = 1/(lambda*N)` and both models solve the *same* problem. Any difference left is optimiser tolerance (gradient descent vs L-BFGS).

## Evaluating honestly

- **No random split.** Daily features and labels are strongly autocorrelated; a random split puts near-duplicates of every test row in the training set and inflates scores. The split is chronological.
- **Purge gap.** For a forecast horizon $h$, a training row's label looks $h$ days ahead. Rows within $h$ days of the test start are dropped, so no training label is built from test-period prices (a test checks the gap).
- **Scaler fit on train only**, so test statistics never leak into preprocessing.
- **Causality tests:** removing future rows must not change past labels or features.
- **Baselines:** majority class, persistence (forecast task), and scikit-learn.
- **Macro-F1 and per-class recall**, not just accuracy, because one regime dominates.

## Results

Running `python run.py` writes `results/results.md` (tables) and plots to `results/`. Paste your numbers here after a real-data run:

| Task | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Nowcast | Majority class | | |
| Nowcast | scikit-learn LogReg | | |
| Nowcast | NumPy softmax (ours) | | |
| Forecast (5d) | Majority class | | |
| Forecast (5d) | Persistence | | |
| Forecast (5d) | NumPy softmax (ours) | | |

Report whatever you get. If persistence wins the forecast task, that is a legitimate finding about how sticky these rule-based regimes are.

## Limitations

- Labels are rule-derived. Nothing here shows the regimes are economically meaningful or tradable.
- A linear model cannot represent the threshold rules exactly; a tree or small neural net would likely do better on the nowcast.
- No transaction costs, no backtest: this is classification, not a trading strategy, and **not financial advice**.
- Survivorship bias: today's liquid large-caps, not a historical universe.
- Pooled across stocks with one shared model; per-stock or per-market models are untested.

## Layout

```
regime/
  data.py         yfinance loader (cached) + synthetic regime-switching generator
  indicators.py   RSI, MACD, ATR, ADX/DI, Bollinger, EMA
  labels.py       rule-based regime labels (causal)
  features.py     features, dataset, purged time split, standardiser
  softmax.py      the from-scratch model: loss, gradient, gradient descent
  metrics.py      confusion matrix, precision/recall/F1
tests/test_all.py gradient check, sklearn parity, causality, split checks
run.py            end-to-end experiment -> results/
```
