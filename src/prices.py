"""
Fetch, cache, and normalise metal spot prices to USD per metric tonne.
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import CACHE_DIR, LOOKBACK_YEARS, METALS

warnings.filterwarnings("ignore")

LT_PARAMS = {
    "copper":    {"mean_usd_tonne": 9_000,  "annual_vol": 0.22},
    "aluminium": {"mean_usd_tonne": 2_500,  "annual_vol": 0.18},
    "steel":     {"mean_usd_tonne":   750,  "annual_vol": 0.20},
    "stainless": {"mean_usd_tonne": 14_000, "annual_vol": 0.35},
}


def _cache_path(metal: str) -> Path:
    return CACHE_DIR / f"{metal}_prices.csv"


def _download(metal: str, years: int) -> pd.Series | None:
    cfg = METALS[metal]
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    try:
        raw = yf.download(cfg["ticker"], start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            return None
        close = raw["Close"].squeeze()
        close = close * cfg["usd_per_tonne_multiplier"]
        close.name = metal
        return close.dropna()
    except Exception:
        return None


def _synthetic(metal: str, years: int) -> pd.Series:
    p = LT_PARAMS[metal]
    n = int(years * 252)
    dt = 1 / 252
    mu = 0.02
    sigma = p["annual_vol"]
    np.random.seed(42)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(n)
    prices = p["mean_usd_tonne"] * np.exp(np.cumsum(log_returns) - np.cumsum(log_returns)[-1] / 2)
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    return pd.Series(prices, index=dates, name=metal)


def fetch_prices(metal: str, years: int = LOOKBACK_YEARS, force_refresh: bool = False) -> pd.Series:
    cfg = METALS[metal]

    if "price_proxy" in cfg:
        proxy_series = fetch_prices(cfg["price_proxy"], years, force_refresh)
        return proxy_series.rename(metal)

    path = _cache_path(metal)
    if not force_refresh and path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        if cached.index[-1].date() >= (datetime.today() - timedelta(days=1)).date():
            cached.name = metal
            return cached

    series = _download(metal, years)
    if series is None or len(series) < 100:
        label = cfg.get("ticker", metal)
        print(f"  [!] Could not download live data for {metal} ({label}) — using synthetic series")
        series = _synthetic(metal, years)
    else:
        print(f"  [+] Downloaded {len(series)} days of {metal} prices ({cfg['ticker']})")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return series


def fetch_all_prices(years: int = LOOKBACK_YEARS, force_refresh: bool = False) -> dict[str, pd.Series]:
    return {metal: fetch_prices(metal, years, force_refresh) for metal in METALS}


def latest_price(metal: str) -> float:
    return float(fetch_prices(metal).iloc[-1])


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> float:
    r = returns.values
    var = float(np.var(r[:20]))
    for ri in r[20:]:
        var = lam * var + (1 - lam) * ri ** 2
    return float(np.sqrt(var))


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def effective_metals(metals: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for m in metals:
        cfg = METALS.get(m, {})
        driver = cfg.get("price_proxy", m)
        seen[driver] = m
    return list(seen.keys())


def merge_exposures_by_driver(
    exposures: dict[str, float],
    returns_map: dict[str, pd.Series],
) -> tuple[dict[str, float], dict[str, pd.Series]]:
    merged_exp: dict[str, float] = {}
    merged_ret: dict[str, pd.Series] = {}

    for metal, exp in exposures.items():
        cfg = METALS.get(metal, {})
        driver = cfg.get("price_proxy", metal)
        merged_exp[driver] = merged_exp.get(driver, 0.0) + exp
        if driver not in merged_ret:
            merged_ret[driver] = returns_map.get(driver, returns_map.get(metal))

    return merged_exp, merged_ret
