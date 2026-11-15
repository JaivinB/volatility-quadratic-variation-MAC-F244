"""
Data layer for the MAC F244 project.

Source: Yahoo Finance via the `yfinance` client (v1.8.0).
  ^NSEI      NIFTY 50 index, daily OHLC, auto-adjusted
  ^INDIAVIX  India VIX, the NSE's 30-day implied volatility index

Downloads once, caches to data/*.csv, and validates before returning. The cache
is what the report's numbers are computed from, so a re-run reproduces them
exactly even if the vendor revises history.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
START, END = "2015-01-01", "2026-11-13"
TRADING_DAYS = 252


def _download(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=START, end=END, progress=False,
                     auto_adjust=True, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"no rows returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):       # yfinance >=0.2 shape
        df.columns = df.columns.get_level_values(0)
    return df


def load(ticker: str, *, refresh: bool = False) -> pd.DataFrame:
    """Cached OHLC frame for one ticker, indexed by date."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{ticker.lstrip('^').lower()}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = _download(ticker)
        df.to_csv(path)
    return df


def nifty_and_vix(*, refresh: bool = False):
    """Closing NIFTY level and India VIX as a decimal, on their common dates."""
    spot = load("^NSEI", refresh=refresh)
    vix = load("^INDIAVIX", refresh=refresh)
    frame = pd.DataFrame({
        "open":  spot["Open"].squeeze(),
        "high":  spot["High"].squeeze(),
        "low":   spot["Low"].squeeze(),
        "spot":  spot["Close"].squeeze(),
        "iv":    vix["Close"].squeeze() / 100.0,     # VIX is quoted in percent
    }).dropna()
    _validate(frame)
    return frame


def _validate(frame: pd.DataFrame) -> None:
    """Fail loudly rather than silently reporting numbers from bad data."""
    assert len(frame) > 2000, f"only {len(frame)} rows; expected a decade"
    assert frame.index.is_monotonic_increasing, "dates out of order"
    assert not frame.index.duplicated().any(), "duplicate dates"
    assert (frame[["open", "high", "low", "spot"]] > 0).all().all(), "non-positive price"
    assert (frame["high"] >= frame["low"]).all(), "high < low"
    assert frame["iv"].between(0.02, 1.50).all(), "implied vol outside a sane band"
    gap = frame.index.to_series().diff().dt.days.dropna()
    assert gap.max() <= 15, f"a {gap.max()}-day hole in the series"


def log_returns(spot: pd.Series) -> pd.Series:
    return np.log(spot).diff().dropna()


def realised_vol(spot: pd.Series, window: int = 21) -> pd.Series:
    """Annualised close-to-close realised volatility over a rolling window."""
    return log_returns(spot).rolling(window).std() * np.sqrt(TRADING_DAYS)


if __name__ == "__main__":
    f = nifty_and_vix()
    r = log_returns(f["spot"])
    print(f"rows            : {len(f)}")
    print(f"span            : {f.index[0].date()} -> {f.index[-1].date()}")
    print(f"NIFTY last      : {f['spot'].iloc[-1]:,.2f}")
    print(f"ann. realised   : {r.std() * np.sqrt(TRADING_DAYS) * 100:.2f}%")
    print(f"mean India VIX  : {f['iv'].mean() * 100:.2f}%")
    print(f"excess kurtosis : {r.kurtosis():.2f}   skew: {r.skew():.2f}")
