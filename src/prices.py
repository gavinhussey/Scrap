"""Fetch, cache, and normalise metal spot prices to USD per metric tonne."""

import warnings
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import CACHE_DIR, LOOKBACK_YEARS, METALS, SCRAP_BASIS_VOL_MULT

warnings.filterwarnings("ignore")

LT_PARAMS = {
    "copper":    {"mean_usd_tonne": 9_000,  "annual_vol": 0.22},
    "aluminium": {"mean_usd_tonne": 2_500,  "annual_vol": 0.18},
    "steel":     {"mean_usd_tonne":   750,  "annual_vol": 0.20},
    "stainless": {"mean_usd_tonne": 14_000, "annual_vol": 0.35},
}

LAST_PRICE_SOURCES: dict[str, str] = {}


def _cache_path(metal: str) -> Path:
    return CACHE_DIR / f"{metal}_prices.csv"


def _synthetic_cache_path(metal: str) -> Path:
    return CACHE_DIR / f"{metal}_synthetic_prices.csv"


def _cache_meta_path(metal: str) -> Path:
    return CACHE_DIR / f"{metal}_prices.meta.json"


def _read_cache(path: Path, metal: str) -> pd.Series:
    cached = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    cached.name = metal
    return cached


def _write_cache(series: pd.Series, metal: str, source: str, ticker: str | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(_cache_path(metal))
    meta = {
        "metal": metal,
        "source": source,
        "ticker": ticker,
        "rows": int(len(series)),
        "start": str(series.index.min().date()) if len(series) else None,
        "end": str(series.index.max().date()) if len(series) else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _cache_meta_path(metal).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


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
        proxy = cfg["price_proxy"]
        proxy_series = fetch_prices(proxy, years, force_refresh)
        LAST_PRICE_SOURCES[metal] = f"proxy:{proxy} ({LAST_PRICE_SOURCES.get(proxy, 'unknown')})"
        return proxy_series.rename(metal)

    path = _cache_path(metal)
    if not force_refresh and path.exists():
        cached = _read_cache(path, metal)
        if cached.index[-1].date() >= (datetime.today() - timedelta(days=1)).date():
            LAST_PRICE_SOURCES.setdefault(metal, "cache")
            return cached

    series = _download(metal, years)
    if series is None or len(series) < 100:
        label = cfg.get("ticker", metal)
        if path.exists():
            cached = _read_cache(path, metal)
            print(f"  [!] Could not download live data for {metal} ({label}) — using stale real cache")
            LAST_PRICE_SOURCES[metal] = "stale-cache"
            return cached
        print(f"  [!] Could not download live data for {metal} ({label}) — using synthetic series (not written to real cache)")
        series = _synthetic(metal, years)
        LAST_PRICE_SOURCES[metal] = "synthetic"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        series.to_csv(_synthetic_cache_path(metal))
        return series
    else:
        print(f"  [+] Downloaded {len(series)} days of {metal} prices ({cfg['ticker']})")
        LAST_PRICE_SOURCES[metal] = "yahoo"

    _write_cache(series, metal, source="yahoo", ticker=cfg.get("ticker"))
    return series


def fetch_all_prices(years: int = LOOKBACK_YEARS, force_refresh: bool = False) -> dict[str, pd.Series]:
    LAST_PRICE_SOURCES.clear()
    return {metal: fetch_prices(metal, years, force_refresh) for metal in METALS}


def price_source_summary() -> dict[str, str]:
    return dict(LAST_PRICE_SOURCES)


def latest_price(metal: str) -> float:
    return float(fetch_prices(metal).iloc[-1])


def vol_multiplier(metal: str) -> float:
    """Total vol scaling for a metal's risk: basis-risk overlay x per-metal factor."""
    return SCRAP_BASIS_VOL_MULT * METALS.get(metal, {}).get("vol_multiplier", 1.0)


def risk_series(metal: str, base: pd.Series) -> pd.Series:
    """Rescale a price series so its returns carry the basis-adjusted vol.

    Levels are not preserved (irrelevant — VaR/MC use returns only); the spot
    price for mark-to-market display still comes from the unscaled series.
    """
    k = vol_multiplier(metal)
    if k == 1.0:
        return base
    log_ret = np.log(base).diff().fillna(0.0) * k
    scaled = float(base.iloc[0]) * np.exp(log_ret.cumsum())
    scaled.name = metal
    return scaled


def fetch_risk_series(years: int = LOOKBACK_YEARS, force_refresh: bool = False) -> dict[str, pd.Series]:
    """Per-metal price series with basis-adjusted volatility, for VaR/Monte Carlo."""
    return {m: risk_series(m, s) for m, s in fetch_all_prices(years, force_refresh).items()}


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> float:
    r = returns.values
    var = float(np.var(r[:20]))
    for ri in r[20:]:
        var = lam * var + (1 - lam) * ri ** 2
    return float(np.sqrt(var))


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
