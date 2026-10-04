"""Feature matrix + time-aware dataset construction and splitting."""
import numpy as np
import pandas as pd
from .indicators import adx, atr, bollinger, ema, macd_hist, rsi
from .labels import label_regimes

FEATURES = ["ret_1", "ret_5", "ret_20", "vol_10", "vol_20", "rsi", "macd_hist", "pct_b", "bb_width",
            "vol_ratio", "dist_ema50", "dist_ema200", "adx", "di_diff", "atr_pct"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"].replace(0, np.nan)
    lr = np.log(c).diff()
    a, pdi, mdi = adx(df)
    pct_b, width = bollinger(c)
    f = pd.DataFrame(index=df.index)
    f["ret_1"], f["ret_5"], f["ret_20"] = c.pct_change(1), c.pct_change(5), c.pct_change(20)
    f["vol_10"], f["vol_20"] = lr.rolling(10).std(), lr.rolling(20).std()
    f["rsi"] = rsi(c) / 100
    f["macd_hist"] = macd_hist(c) / c
    f["pct_b"], f["bb_width"] = pct_b, width
    f["vol_ratio"] = np.log(v / v.rolling(20).mean())
    f["dist_ema50"], f["dist_ema200"] = c / ema(c, 50) - 1, c / ema(c, 200) - 1
    f["adx"], f["di_diff"] = a / 100, (pdi - mdi) / 100
    f["atr_pct"] = atr(df) / c
    return f.replace([np.inf, -np.inf], np.nan)


def build_dataset(prices: dict, horizon: int) -> pd.DataFrame:
    """Rows = (date, ticker). `y` is the regime `horizon` trading days AHEAD
    (horizon=0 -> today's regime). `y_now` is today's regime."""
    frames = []
    for ticker, df in prices.items():
        out = build_features(df)
        lab = label_regimes(df)
        out["y_now"], out["y"], out["ticker"] = lab, lab.shift(-horizon), ticker
        frames.append(out.dropna())
    data = pd.concat(frames)
    data["y"], data["y_now"] = data["y"].astype(int), data["y_now"].astype(int)
    data.index.name = "date"
    return data


def time_split(data: pd.DataFrame, horizon: int, test_frac: float = 0.25):
    """Chronological split with a purge gap of `horizon` days: training rows
    whose target window (t .. t+horizon) reaches the test period are dropped."""
    dates = np.sort(data.index.unique())
    cut_i = int(len(dates) * (1 - test_frac))
    train = data[data.index < dates[max(cut_i - horizon, 0)]]
    test = data[data.index >= dates[cut_i]]
    return train, test


class Standardizer:
    """z = (x - mean) / std, statistics learned on TRAIN data only."""
    def fit(self, X):
        self.mean, self.std = X.mean(0), X.std(0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
