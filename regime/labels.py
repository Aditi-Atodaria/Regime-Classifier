"""Rule-based regime labeller.

IMPORTANT: these labels come from hand-written rules, not ground truth. The
model learns to reproduce / anticipate THESE rules, nothing more.
Every label at time t uses only data up to t (no look-ahead).
"""
import numpy as np
import pandas as pd
from .indicators import adx, atr, ema

REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "RANGE_BOUND"]


def label_regimes(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    a, pdi, mdi = adx(df)
    e50, e200 = ema(c, 50), ema(c, 200)
    atr_pct = atr(df) / c
    thresh = atr_pct.rolling(120).quantile(0.80)          # rolling window = no look-ahead
    up = (a >= 25) & (pdi > mdi) & (e50 > e200)
    down = (a >= 25) & (mdi > pdi) & (e50 < e200)
    code = np.select([up, down, atr_pct > thresh], [0, 1, 2], default=3).astype(float)
    return pd.Series(code, index=df.index).where(a.notna() & thresh.notna())
