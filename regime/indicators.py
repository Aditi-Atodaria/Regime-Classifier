"""Technical indicators from their standard definitions (pandas/numpy only)."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    out = 100 - 100 / (1 + ag / al.where(al != 0))
    return out.where(~(al.notna() & (al == 0)), np.where(ag > 0, 100.0, 50.0))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14):
    """Wilder's ADX. Only the larger directional move counts each bar."""
    up, down = df["high"].diff(), -df["low"].diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    a = atr(df, n)
    kw = dict(alpha=1 / n, min_periods=n, adjust=False)
    pdi = 100 * pdm.ewm(**kw).mean() / a
    mdi = 100 * mdm.ewm(**kw).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(**kw).mean(), pdi, mdi


def macd_hist(close: pd.Series) -> pd.Series:
    macd = ema(close, 12) - ema(close, 26)
    return macd - ema(macd, 9)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid, sd = close.rolling(n).mean(), close.rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    return (close - lower) / (upper - lower).replace(0, np.nan), (upper - lower) / mid
