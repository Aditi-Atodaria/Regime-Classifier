"""Price data: real (yfinance, cached to CSV) or synthetic regime-switching series."""
from pathlib import Path
import numpy as np
import pandas as pd

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
           "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS",
           "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS",
           "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "WMT", "KO", "PEP"]
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def load_real(tickers=TICKERS, period="10y") -> dict:
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        f = CACHE / f"{t}.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        else:
            try:
                df = yf.Ticker(t).history(period=period, auto_adjust=True)
            except Exception as e:
                print(f"  skip {t}: {e}")
                continue
            if df is None or df.empty:
                print(f"  skip {t}: no data")
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.to_csv(f)
        out[t] = df.dropna()
    return out


def load_synthetic(n_tickers=20, n_days=2500, seed=0) -> dict:
    """Markov regime-switching prices. Pipeline testing only; NOT a market result."""
    rng = np.random.default_rng(seed)
    params = [(0.0012, 0.008), (-0.0012, 0.012), (0.0, 0.025), (0.0, 0.006)]   # (drift, vol) per state
    dates = pd.bdate_range("2012-01-02", periods=n_days)
    out = {}
    for k in range(n_tickers):
        s, ret, volm = 3, np.empty(n_days), np.empty(n_days)
        for i in range(n_days):
            if rng.random() > 0.97:
                s = rng.integers(0, 4)
            mu, sg = params[s]
            ret[i], volm[i] = rng.normal(mu, sg), np.exp(rng.normal(13 + 0.4 * (s == 2), 0.3))
        close = 100 * np.exp(np.cumsum(ret))
        hi = close * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        lo = close * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        out[f"SYN{k:02d}"] = pd.DataFrame({"open": close, "high": hi, "low": lo, "close": close, "volume": volm}, index=dates)
    return out
