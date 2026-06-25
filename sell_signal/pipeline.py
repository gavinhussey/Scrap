# sell_signal/pipeline.py
"""
Quantitative sell-signal DATA pipeline for a scrap-metal yard.

This is a *data + feature* foundation for a future sell-signal model. It:

  1. Maps each scrap material to a free futures/ETF/FRED proxy (or weighted basket).
  2. Pulls historical pricing from free sources (yfinance + FRED), with fallbacks.
  3. Cleans / aligns / resamples the series and computes returns.
  4. Builds synthetic proxy baskets (rebased-to-100, weighted).
  5. Estimates correlation relationships (rolling + full sample, multi-horizon, lagged).
  6. Estimates hedge relationships (OLS alpha/beta/R2/p-value/residual vol/tracking error
     + rolling betas).
  7. Engineers model-ready predictive features per material.
  8. Builds placeholder forward-looking target variables for later supervised modeling.

FRED data is pulled from the free, keyless fredgraph CSV endpoint (no API key, no
pandas_datareader — which is broken on Python 3.12+ because it imports the removed
`distutils`). HTTPS uses the `certifi` CA bundle so it works on bare python.org installs
that lack a system cert store. Set FRED_API_KEY to additionally enable the official FRED API.

Install:
    pip install pandas numpy yfinance statsmodels scipy matplotlib certifi requests
"""

from __future__ import annotations

import base64
import html
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

try:
    from fredapi import Fred
except ImportError:  # pragma: no cover
    Fred = None

try:
    import certifi  # ships a CA bundle; needed for HTTPS on bare python.org installs
except ImportError:  # pragma: no cover
    certifi = None

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
@dataclass
class Config:
    """All knobs in one place — edit here, not in the body of the script."""

    # --- time window & sampling ---
    start_date: str = "2015-01-01"
    end_date: str = field(default_factory=lambda: datetime.today().strftime("%Y-%m-%d"))
    # "daily" | "weekly" | "monthly". NOTE: free FRED metal indexes are MONTHLY, so
    # "monthly" gives the most consistent cross-source statistics. Use "daily"/"weekly"
    # only if your proxies are yfinance futures/ETFs (HG=F, ALI=F, HRC=F, SLX, ...).
    frequency: str = "monthly"

    # --- rolling-window lengths (in periods of `frequency`) ---
    rolling_corr_window: int = 24
    rolling_beta_window: int = 24
    z_window: int = 24
    drawdown_window: int = 12
    realized_vol_windows: tuple[int, ...] = (20, 60)
    return_windows: tuple[int, ...] = (1, 5, 10, 20, 60)
    # moving-average / momentum windows (ALSO in periods of `frequency`: at the
    # default monthly frequency, ma_long_window=50 is a 50-MONTH average, not 50 days)
    ma_short_window: int = 20
    ma_long_window: int = 50
    momentum_window: int = 20

    # --- correlation experiment grid ---
    return_horizons: tuple[str, ...] = ("daily", "weekly", "monthly")
    lags: tuple[int, ...] = (0, 1, 2, 4)

    # --- forward target horizons & thresholds ---
    forward_horizons: tuple[int, ...] = (5, 10, 20)
    target_drop_threshold: float = 0.02       # "below -2%" binary target
    target_drawdown_threshold: float = 0.05   # 5% forward drawdown target
    target_drawdown_window: int = 20

    # --- io ---
    output_dir: str = "output/sell_signal"
    material_price_csv: str | None = None  # optional real internal material prices
    make_plots: bool = True               # render PNG charts into output_dir/charts
    make_report: bool = True              # assemble a self-contained HTML report
    report_name: str = "sell_signal_report.html"
    fred_api_key: str | None = field(default_factory=lambda: os.environ.get("FRED_API_KEY"))

    materials: tuple[str, ...] = ("STEEL", "STAINLESS", "ZINC", "COPPER", "ALUMINUM", "BRASS")


CFG = Config()


# --- Proxy candidate sources -------------------------------------------------
# Each proxy key maps to an ORDERED list of candidate sources; the fetcher tries
# them top-to-bottom and records which one actually supplied data. "yf" = yfinance
# ticker, "fred" = FRED series id. Add/remove/reorder freely.
PROXY_CANDIDATES: dict[str, list[dict]] = {
    "copper": [
        {"kind": "yf", "id": "HG=F", "label": "COMEX Copper front future"},
        {"kind": "fred", "id": "PCOPPUSDM", "label": "Global price of Copper (FRED)"},
        {"kind": "yf", "id": "CPER", "label": "US Copper Index Fund ETF"},
    ],
    "aluminum": [
        {"kind": "yf", "id": "ALI=F", "label": "COMEX Aluminum front future"},
        {"kind": "fred", "id": "PALUMUSDM", "label": "Global price of Aluminum (FRED)"},
        {"kind": "yf", "id": "JJU", "label": "Aluminum subindex ETN"},
    ],
    "zinc": [
        {"kind": "fred", "id": "PZINCUSDM", "label": "Global price of Zinc (FRED)"},
        {"kind": "yf", "id": "ZNC=F", "label": "US Zinc front future (illiquid/stale fallback)"},
    ],
    "nickel": [
        {"kind": "fred", "id": "PNICKUSDM", "label": "Global price of Nickel (FRED)"},
    ],
    "steel": [
        {"kind": "yf", "id": "HRC=F", "label": "NYMEX HRC Steel front future"},
        {"kind": "yf", "id": "SLX", "label": "VanEck Steel ETF"},
        {"kind": "fred", "id": "WPU101", "label": "PPI Iron & Steel (FRED)"},
    ],
    # extra proxy, NOT in any basket but available to the "best proxy" search for steel
    "iron_ore": [
        {"kind": "fred", "id": "PIORECRUSDM", "label": "Global price of Iron Ore (FRED)"},
    ],
}

# --- Material -> proxy basket weights ----------------------------------------
# Every material is expressed as a basket (single-proxy materials simply have one
# component with weight 1.0). Weights are renormalized over whatever components
# successfully download, so a missing component degrades rather than breaks.
MATERIAL_BASKETS: dict[str, dict[str, float]] = {
    "COPPER": {"copper": 1.0},
    "ALUMINUM": {"aluminum": 1.0},
    "ZINC": {"zinc": 1.0},
    "STEEL": {"steel": 1.0},
    "STAINLESS": {"nickel": 0.60, "steel": 0.40},   # stainless ~ nickel + steel content
    "BRASS": {"copper": 0.70, "zinc": 0.30},        # brass ~ copper + zinc content
}

# consistent colors for charts/report (material name or proxy key, case-insensitive)
MATERIAL_COLORS: dict[str, str] = {
    "COPPER": "#b87333", "BRASS": "#c9a84c", "ALUMINUM": "#94a3b8", "STEEL": "#607080",
    "STAINLESS": "#6a9ab0", "ZINC": "#7c8f3a", "NICKEL": "#8a7fb0", "IRON_ORE": "#a87c5f",
}


def _color_for(name: str) -> str:
    return MATERIAL_COLORS.get(str(name).replace("proxy_", "").upper(), "#64748b")


# ============================================================================
# LOGGING / IO HELPERS
# ============================================================================
log = logging.getLogger("sell_signal")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_output_dir(cfg: Config) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg.output_dir


def _save(df: pd.DataFrame, cfg: Config, name: str, index: bool = True) -> None:
    path = os.path.join(cfg.output_dir, name)
    df.to_csv(path, index=index)
    log.info("wrote %-32s (%d rows x %d cols)", name, len(df), df.shape[1] if df.ndim == 2 else 1)


# pandas resample rule + periods-per-year for each frequency
_FREQ_RULE = {"daily": "B", "weekly": "W-FRI", "monthly": "ME"}
_FREQ_PPY = {"daily": 252, "weekly": 52, "monthly": 12}


# ============================================================================
# 2. DATA COLLECTION
# ============================================================================
def fetch_yfinance(ticker: str, cfg: Config) -> pd.Series | None:
    """Daily adjusted close from Yahoo Finance, or None on failure."""
    if yf is None:
        log.warning("yfinance not installed; cannot fetch %s", ticker)
        return None
    try:
        raw = yf.download(
            ticker, start=cfg.start_date, end=cfg.end_date,
            progress=False, auto_adjust=True,
        )
        if raw is None or raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):          # MultiIndex single-ticker case
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        return close if not close.empty else None
    except Exception as exc:  # noqa: BLE001 — third-party, want to keep going
        log.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None


def _http_get(url: str, timeout: int = 30) -> str | None:
    """HTTPS GET returning text, using the certifi CA bundle when available.

    Avoids pandas_datareader (broken on Python 3.12+) and works on python.org
    installs that have no system cert store (SSLCertVerificationError otherwise).
    """
    import ssl
    import urllib.request

    ctx = ssl.create_default_context(cafile=certifi.where()) if certifi is not None else None
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sell-signal-pipeline)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        log.warning("HTTP GET failed (%s): %s", url.split("?")[0], exc)
        return None


def _parse_fred_csv(text: str, series_id: str) -> pd.Series | None:
    """Parse a fredgraph CSV (date col + value col; '.' marks missing observations)."""
    import io

    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        return None
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col], errors="coerce"), name=series_id).dropna()
    return s if not s.empty else None


def fetch_fred(series_id: str, cfg: Config) -> pd.Series | None:
    """FRED series via the free keyless fredgraph CSV, with optional API-key fallbacks."""
    # 1. keyless fredgraph CSV endpoint — no API key, no extra dependency.
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd={cfg.start_date}&coed={cfg.end_date}")
    text = _http_get(url)
    if text and not text.lstrip().startswith("<"):  # guard against HTML error pages
        s = _parse_fred_csv(text, series_id)
        if s is not None:
            return s

    # 2. official FRED API (needs a free key) — different host, useful when graph host is blocked.
    if cfg.fred_api_key:
        import json

        api = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
               f"&api_key={cfg.fred_api_key}&file_type=json"
               f"&observation_start={cfg.start_date}&observation_end={cfg.end_date}")
        text = _http_get(api)
        if text:
            try:
                obs = json.loads(text).get("observations", [])
                s = pd.Series(
                    [pd.to_numeric(o["value"], errors="coerce") for o in obs],
                    index=pd.to_datetime([o["date"] for o in obs], errors="coerce"),
                    name=series_id,
                ).dropna()
                if not s.empty:
                    return s
            except (ValueError, KeyError) as exc:
                log.warning("FRED API parse failed for %s: %s", series_id, exc)

    # 3. fredapi library, if installed and keyed.
    if Fred is not None and cfg.fred_api_key:
        try:
            s = pd.Series(Fred(api_key=cfg.fred_api_key).get_series(
                series_id, observation_start=cfg.start_date, observation_end=cfg.end_date))
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not s.empty:
                return s
        except Exception as exc:  # noqa: BLE001
            log.warning("fredapi fetch failed for %s: %s", series_id, exc)

    return None


def fetch_proxy(proxy_key: str, cfg: Config) -> tuple[pd.Series | None, dict]:
    """Try each candidate source for a proxy in order; return (series, provenance)."""
    for cand in PROXY_CANDIDATES[proxy_key]:
        series = fetch_yfinance(cand["id"], cfg) if cand["kind"] == "yf" else fetch_fred(cand["id"], cfg)
        if series is not None and len(series) > 1:
            series.name = proxy_key
            prov = {
                "proxy": proxy_key, "source_kind": cand["kind"], "source_id": cand["id"],
                "label": cand["label"], "status": "ok", "rows": int(len(series)),
                "start": str(series.index.min().date()), "end": str(series.index.max().date()),
            }
            log.info("proxy %-9s <- %-6s %-10s (%d rows)", proxy_key, cand["kind"], cand["id"], len(series))
            return series, prov
        log.info("proxy %-9s candidate %s unavailable, trying next", proxy_key, cand["id"])
    log.warning("proxy %-9s: ALL candidates failed", proxy_key)
    return None, {"proxy": proxy_key, "source_kind": "", "source_id": "", "label": "",
                  "status": "FAILED", "rows": 0, "start": "", "end": ""}


def collect_all_proxies(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download every unique proxy referenced by the baskets (+ extras like iron_ore)."""
    needed = set(PROXY_CANDIDATES)  # fetch all known proxies; harmless extras aid the search
    series_map: dict[str, pd.Series] = {}
    provenance: list[dict] = []
    for key in sorted(needed):
        s, prov = fetch_proxy(key, cfg)
        provenance.append(prov)
        if s is not None:
            series_map[key] = s
    if not series_map:
        raise RuntimeError("No proxy data could be downloaded from any source. Check connectivity.")
    raw = pd.concat(series_map.values(), axis=1, sort=True)
    raw.columns = [f"proxy_{k}" for k in series_map]
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    return raw, pd.DataFrame(provenance)


# ============================================================================
# 3. CLEAN / ALIGN / RESAMPLE / RETURNS
# ============================================================================
def clean_prices(raw: pd.DataFrame, cfg: Config, min_obs: int = 24) -> pd.DataFrame:
    """Numeric-coerce, drop sparse columns, forward-fill, resample to target freq."""
    df = raw.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        log.warning("dropping sparse columns (<%d obs): %s", min_obs, ", ".join(dropped))
    df = df[keep]
    df = resample_frequency(df, cfg)
    df = df.ffill()
    df = df.dropna(how="all")
    return df


def resample_frequency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Resample to the configured frequency taking the last observation in each bucket."""
    rule = _FREQ_RULE.get(cfg.frequency)
    if rule is None:
        raise ValueError(f"frequency must be one of {list(_FREQ_RULE)}")
    return df.resample(rule).last()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple period-over-period percentage returns."""
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)


# ============================================================================
# 4. SYNTHETIC PROXY BASKETS  ->  MATERIAL SERIES
# ============================================================================
def _rebase_100(s: pd.Series) -> pd.Series:
    """Rebase a series so its first valid observation = 100 (makes components comparable)."""
    first = s.dropna().iloc[0]
    return s / first * 100.0 if first else s


def build_material_series(proxy_prices: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct one index series per material from its (renormalized) basket of proxies.

    Returns (material_prices, basket_weights_used).
    """
    if cfg.material_price_csv and os.path.exists(cfg.material_price_csv):
        log.info("loading REAL material prices from %s", cfg.material_price_csv)
        real = pd.read_csv(cfg.material_price_csv, index_col=0, parse_dates=True)
        real = resample_frequency(real.apply(pd.to_numeric, errors="coerce"), cfg).ffill()
        used = pd.DataFrame([{"material": c, "component": "REAL_INTERNAL", "weight": 1.0} for c in real.columns])
        return real, used

    material_cols: dict[str, pd.Series] = {}
    used_rows: list[dict] = []
    for material in cfg.materials:
        basket = MATERIAL_BASKETS[material]
        available = {k: w for k, w in basket.items() if f"proxy_{k}" in proxy_prices.columns}
        if not available:
            log.warning("material %s: no basket components available, skipping", material)
            continue
        total_w = sum(available.values())
        norm = {k: w / total_w for k, w in available.items()}  # renormalize over what we have
        if set(available) != set(basket):
            log.warning("material %s: missing %s; renormalized weights -> %s",
                        material, set(basket) - set(available), norm)
        index = None
        for key, weight in norm.items():
            comp = _rebase_100(proxy_prices[f"proxy_{key}"]) * weight
            index = comp if index is None else index.add(comp, fill_value=np.nan)
            used_rows.append({"material": material, "component": key, "weight": round(weight, 4)})
        material_cols[material] = index
    materials = pd.concat(material_cols.values(), axis=1)
    materials.columns = list(material_cols)
    return materials, pd.DataFrame(used_rows)


def primary_proxy_for(material: str, available_cols: list[str]) -> str | None:
    """The heaviest-weighted basket component that actually downloaded (the hedge anchor)."""
    basket = MATERIAL_BASKETS.get(material, {})
    ordered = sorted(basket.items(), key=lambda kv: kv[1], reverse=True)
    for key, _ in ordered:
        if f"proxy_{key}" in available_cols:
            return f"proxy_{key}"
    return None


# ============================================================================
# 5. CORRELATION RELATIONSHIPS
# ============================================================================
def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Pearson correlation across every material + proxy return series."""
    return returns.corr()


def rolling_correlations(returns: pd.DataFrame, cfg: Config,
                         material_to_proxy: dict[str, str]) -> pd.DataFrame:
    """Rolling correlation of each material vs its primary proxy at the base frequency."""
    out = {}
    for material, proxy in material_to_proxy.items():
        if material in returns and proxy in returns:
            out[f"{material}~{proxy}"] = returns[material].rolling(cfg.rolling_corr_window).corr(returns[proxy])
    return pd.DataFrame(out)


def _is_structural_identity(material: str, proxy: str) -> bool:
    """True when `material` is a single-proxy basket equal to `proxy`.

    In that case the material series IS the proxy (rebased to 100), so any
    correlation / beta / R2 between them is 1.0 *by construction* — a structural
    artifact, not an empirical relationship. Flagged so the report can say so.
    """
    basket = MATERIAL_BASKETS.get(material, {})
    return len(basket) == 1 and f"proxy_{next(iter(basket))}" == proxy


def best_proxy_relationships(prices: pd.DataFrame, cfg: Config,
                             materials: list[str], proxy_cols: list[str],
                             real_prices: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each material, scan {horizon x lag x proxy} and report the strongest |corr|.

    corr( material_return_t , proxy_return_{t-lag} ).

    Returns (best_per_material, full_scan). `best` carries a `structural_identity`
    flag marking materials whose "best proxy" is just themselves (corr==1 trivially).
    """
    rows: list[dict] = []
    for horizon in cfg.return_horizons:
        rule = _FREQ_RULE[horizon]
        px = prices.resample(rule).last().ffill()
        rets = px.pct_change().replace([np.inf, -np.inf], np.nan)
        for material in materials:
            if material not in rets:
                continue
            for proxy in proxy_cols:
                if proxy not in rets:
                    continue
                for lag in cfg.lags:
                    pair = pd.concat([rets[material], rets[proxy].shift(lag)], axis=1).dropna()
                    if len(pair) < max(12, cfg.rolling_corr_window // 2):
                        continue
                    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                    if pd.notna(corr):
                        rows.append({"material": material, "proxy": proxy, "horizon": horizon,
                                     "lag": lag, "n": len(pair), "correlation": round(corr, 4)})
    scan = pd.DataFrame(rows)
    if scan.empty:
        return pd.DataFrame(), scan
    scan["abs_corr"] = scan["correlation"].abs()
    best = (scan.sort_values("abs_corr", ascending=False)
                .groupby("material", as_index=False).first()
                .drop(columns="abs_corr")
                .sort_values("material").reset_index(drop=True))
    # With real internal prices the material is NOT its proxy, so nothing is structural.
    best["structural_identity"] = False if real_prices else best.apply(
        lambda r: _is_structural_identity(r["material"], r["proxy"]), axis=1)
    return best, scan


# ============================================================================
# 6. HEDGE RELATIONSHIPS (regression)
# ============================================================================
def _ols(y: pd.Series, x: pd.Series) -> dict:
    """OLS y = alpha + beta*x; return key stats with or without statsmodels."""
    pair = pd.concat([y, x], axis=1).dropna()
    pair.columns = ["y", "x"]
    n = len(pair)
    if n < 12:
        return {"n": n, "alpha": np.nan, "beta": np.nan, "r_squared": np.nan,
                "p_value": np.nan, "residual_vol": np.nan, "tracking_error": np.nan}
    yv, xv = pair["y"].values, pair["x"].values
    if sm is not None:
        model = sm.OLS(yv, sm.add_constant(xv)).fit()
        alpha, beta = model.params[0], model.params[1]
        r2, pval = model.rsquared, model.pvalues[1]
        resid = model.resid
    elif scipy_stats is not None:
        lr = scipy_stats.linregress(xv, yv)
        alpha, beta, r2, pval = lr.intercept, lr.slope, lr.rvalue ** 2, lr.pvalue
        resid = yv - (alpha + beta * xv)
    else:  # last-resort numpy
        beta, alpha = np.polyfit(xv, yv, 1)
        resid = yv - (alpha + beta * xv)
        r2 = 1 - resid.var() / yv.var() if yv.var() else np.nan
        pval = np.nan
    return {"n": n, "alpha": round(float(alpha), 6), "beta": round(float(beta), 4),
            "r_squared": round(float(r2), 4), "p_value": round(float(pval), 6) if pd.notna(pval) else np.nan,
            "residual_vol": round(float(np.std(resid, ddof=1)), 6),
            "tracking_error": round(float(np.std(yv - xv, ddof=1)), 6)}


def hedge_relationships(returns: pd.DataFrame, cfg: Config,
                        materials: list[str], proxy_cols: list[str],
                        material_to_proxy: dict[str, str]) -> pd.DataFrame:
    """Regress each material return on each proxy return; flag the primary hedge proxy."""
    rows: list[dict] = []
    for material in materials:
        if material not in returns:
            continue
        for proxy in proxy_cols:
            if proxy not in returns:
                continue
            stats = _ols(returns[material], returns[proxy])
            stats.update({"material": material, "proxy": proxy,
                          "is_primary": proxy == material_to_proxy.get(material)})
            rows.append(stats)
    cols = ["material", "proxy", "is_primary", "n", "alpha", "beta", "r_squared",
            "p_value", "residual_vol", "tracking_error"]
    df = pd.DataFrame(rows)
    return df[cols].sort_values(["material", "is_primary", "r_squared"],
                                ascending=[True, False, False]).reset_index(drop=True)


def rolling_betas(returns: pd.DataFrame, cfg: Config,
                  material_to_proxy: dict[str, str]) -> pd.DataFrame:
    """Rolling hedge ratio (beta = cov(m,p)/var(p)) of each material vs its primary proxy."""
    out = {}
    w = cfg.rolling_beta_window
    for material, proxy in material_to_proxy.items():
        if material in returns and proxy in returns:
            cov = returns[material].rolling(w).cov(returns[proxy])
            var = returns[proxy].rolling(w).var()
            out[f"{material}~{proxy}"] = cov / var.replace(0, np.nan)
    return pd.DataFrame(out)


# ============================================================================
# 7. PREDICTIVE FEATURES
# ============================================================================
def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def build_features(material_prices: pd.DataFrame, proxy_prices: pd.DataFrame,
                   cfg: Config, material_to_proxy: dict[str, str]) -> pd.DataFrame:
    """Build a tidy (date, material) feature table for every material.

    EXCLUDED by design: buyer bids, freight, manufacturing/industrial/PMI/ISM/construction.
    """
    ppy = _FREQ_PPY[cfg.frequency]
    frames: list[pd.DataFrame] = []

    for material in material_prices.columns:
        price = material_prices[material].dropna()
        if len(price) < max(cfg.return_windows) + 1:
            log.warning("feature build: %s has too few obs (%d), skipping", material, len(price))
            continue
        ret = price.pct_change()
        f = pd.DataFrame(index=price.index)
        f["material"] = material
        f["price"] = price

        # momentum / multi-window returns
        for w in cfg.return_windows:
            f[f"ret_{w}"] = price.pct_change(w)

        # moving-average ratios (windows in periods of cfg.frequency)
        ma_s, ma_l = cfg.ma_short_window, cfg.ma_long_window
        f[f"price_vs_ma{ma_s}"] = price / price.rolling(ma_s).mean() - 1
        f[f"price_vs_ma{ma_l}"] = price / price.rolling(ma_l).mean() - 1
        f[f"ma{ma_s}_ma{ma_l}_ratio"] = price.rolling(ma_s).mean() / price.rolling(ma_l).mean() - 1

        # realized volatility (annualized)
        for w in cfg.realized_vol_windows:
            f[f"realized_vol_{w}"] = ret.rolling(w).std(ddof=0) * np.sqrt(ppy)

        # drawdown from recent high, z-scores
        f["drawdown_recent_high"] = price / price.rolling(cfg.drawdown_window).max() - 1
        f["price_zscore"] = _zscore(price, cfg.z_window)
        f["ret_zscore"] = _zscore(ret, cfg.z_window)

        # relationship to primary proxy
        proxy = material_to_proxy.get(material)
        if proxy and proxy in proxy_prices:
            pser = proxy_prices[proxy].reindex(price.index).ffill()
            pret = pser.pct_change()
            mw = cfg.momentum_window
            f[f"proxy_momentum_{mw}"] = pser.pct_change(mw)
            f[f"proxy_realized_vol_{mw}"] = pret.rolling(mw).std(ddof=0) * np.sqrt(ppy)
            f["rolling_corr_proxy"] = ret.rolling(cfg.rolling_corr_window).corr(pret)
            cov = ret.rolling(cfg.rolling_beta_window).cov(pret)
            var = pret.rolling(cfg.rolling_beta_window).var()
            f["rolling_beta_proxy"] = cov / var.replace(0, np.nan)
            f["rolling_r2_proxy"] = f["rolling_corr_proxy"] ** 2
            # basis-style spread: rebased material vs rebased proxy
            basis = _rebase_100(price) - _rebase_100(pser)
            f["basis_spread"] = basis
            f["basis_zscore"] = _zscore(basis, cfg.z_window)

        # basket momentum (same as material momentum here; explicit for clarity/extension)
        f[f"basket_momentum_{cfg.momentum_window}"] = price.pct_change(cfg.momentum_window)

        # seasonality
        f["month"] = f.index.month
        f["quarter"] = f.index.quarter

        frames.append(f)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out.index.name = "date"
    return out


# ============================================================================
# 8. PLACEHOLDER TARGET VARIABLES
# ============================================================================
def _forward_min(price: pd.Series, window: int) -> pd.Series:
    """min(price[t+1 .. t+window]) aligned at t (explicit, leakage-aware)."""
    shifted = [price.shift(-k) for k in range(1, window + 1)]
    return pd.concat(shifted, axis=1).min(axis=1)


def build_targets(material_prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Forward-looking, leakage-aware targets for future supervised sell-signal training."""
    drop_pct = int(cfg.target_drop_threshold * 100)

    def _binary(mask: pd.Series, valid: pd.Series) -> pd.Series:
        # keep NA where the forward window ran off the end, so the tail isn't mislabeled 0
        return mask.where(valid.notna()).astype("Int64")

    frames: list[pd.DataFrame] = []
    for material in material_prices.columns:
        price = material_prices[material].dropna()
        t = pd.DataFrame(index=price.index)
        t["material"] = material
        for h in cfg.forward_horizons:
            fwd = price.shift(-h) / price - 1
            t[f"fwd_ret_{h}"] = fwd
            t[f"target_down_{h}"] = _binary(fwd < 0, fwd)
            t[f"target_drop{drop_pct}pct_{h}"] = _binary(fwd < -cfg.target_drop_threshold, fwd)
        fwd_dd = _forward_min(price, cfg.target_drawdown_window) / price - 1
        t[f"fwd_drawdown_{cfg.target_drawdown_window}"] = fwd_dd
        t[f"target_drawdown_{cfg.target_drawdown_window}"] = _binary(
            fwd_dd < -cfg.target_drawdown_threshold, fwd_dd)
        frames.append(t)
    out = pd.concat(frames).sort_index()
    out.index.name = "date"
    return out


# ============================================================================
# 9. CHARTS (PNG)
# ============================================================================
def _rebase_100_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: c / c.dropna().iloc[0] * 100 if c.dropna().size else c)


def _style_ax(ax, title: str = "", ylabel: str = "") -> None:
    """Apply the shared clean look to a matplotlib axis."""
    ax.set_title(title, fontsize=12, fontweight="bold", color="#172033", loc="left", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#667085")
    ax.grid(True, color="#e8edf2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9e0e8")
    ax.tick_params(colors="#667085", labelsize=8)


def _savefig(fig, charts_dir: str, name: str) -> str:
    path = os.path.join(charts_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def render_charts(material_prices: pd.DataFrame, proxy_prices: pd.DataFrame,
                  combined_corr: pd.DataFrame, roll_corr: pd.DataFrame, roll_beta: pd.DataFrame,
                  best_rel: pd.DataFrame, features: pd.DataFrame, targets: pd.DataFrame,
                  cfg: Config) -> dict[str, str]:
    """Render the full chart set to output_dir/charts and return {key: path}."""
    if plt is None:
        log.warning("matplotlib unavailable; skipping charts")
        return {}
    charts_dir = os.path.join(cfg.output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    charts: dict[str, str] = {}

    # A. rebased material indexes
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for col in material_prices.columns:
        ser = _rebase_100_frame(material_prices[[col]])[col]
        ax.plot(ser.index, ser.values, label=col, color=_color_for(col), linewidth=1.6)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(material_prices.columns)))
    _style_ax(ax, "Material proxy indexes (rebased = 100)", "index")
    charts["indexes"] = _savefig(fig, charts_dir, "material_indexes.png")

    # B. correlation heatmap (materials + proxies)
    if not combined_corr.empty:
        cm = combined_corr
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(cm)), max(5, 0.6 * len(cm))))
        im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cm.columns)), labels=cm.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cm.index)), labels=cm.index, fontsize=7)
        for i in range(len(cm.index)):
            for j in range(len(cm.columns)):
                v = cm.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.55 else "#172033")
        ax.set_title("Return correlation matrix", fontsize=12, fontweight="bold", color="#172033", loc="left", pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        charts["heatmap"] = _savefig(fig, charts_dir, "correlation_heatmap.png")

    # C. rolling correlation
    if not roll_corr.empty:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for col in roll_corr.columns:
            ax.plot(roll_corr.index, roll_corr[col].values, label=col.split("~")[0],
                    color=_color_for(col.split("~")[0]), linewidth=1.4)
        ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(frameon=False, fontsize=8, ncol=4)
        _style_ax(ax, f"Rolling correlation to primary proxy ({cfg.rolling_corr_window}-period)", "corr")
        charts["rolling_corr"] = _savefig(fig, charts_dir, "rolling_correlations.png")

    # D. rolling beta
    if not roll_beta.empty:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for col in roll_beta.columns:
            ax.plot(roll_beta.index, roll_beta[col].values, label=col.split("~")[0],
                    color=_color_for(col.split("~")[0]), linewidth=1.4)
        ax.axhline(1, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.legend(frameon=False, fontsize=8, ncol=4)
        _style_ax(ax, f"Rolling hedge ratio (beta) to primary proxy ({cfg.rolling_beta_window}-period)", "beta")
        charts["rolling_beta"] = _savefig(fig, charts_dir, "rolling_betas.png")

    # E. best-proxy correlation bar
    if not best_rel.empty:
        b = best_rel.sort_values("correlation")
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(b) + 1.5))
        ax.barh(b["material"], b["correlation"], color=[_color_for(m) for m in b["material"]])
        for y, (corr, proxy) in enumerate(zip(b["correlation"], b["proxy"])):
            ax.text(corr, y, f"  {corr:.2f} · {proxy.replace('proxy_', '')}", va="center", fontsize=8, color="#172033")
        ax.set_xlim(min(0, b["correlation"].min()) - 0.05, 1.05)
        _style_ax(ax, "Best proxy correlation by material", "correlation")
        charts["best_proxy"] = _savefig(fig, charts_dir, "best_proxy.png")

    # F. forward-down probability by material x horizon
    if not targets.empty:
        down_cols = [c for c in targets.columns if c.startswith("target_down_")]
        if down_cols:
            prob = targets.groupby("material")[down_cols].mean()
            horizons = [c.replace("target_down_", "") for c in down_cols]
            fig, ax = plt.subplots(figsize=(9, 4.4))
            x = np.arange(len(prob.index))
            width = 0.8 / max(1, len(down_cols))
            for i, col in enumerate(down_cols):
                ax.bar(x + i * width, prob[col].values, width, label=f"{horizons[i]}-period",
                       color=plt.cm.Blues(0.4 + 0.5 * i / max(1, len(down_cols))))
            ax.set_xticks(x + width * (len(down_cols) - 1) / 2, labels=prob.index, fontsize=8)
            ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--")
            ax.legend(frameon=False, fontsize=8)
            _style_ax(ax, "Historical P(forward return < 0) by material", "probability")
            charts["targets"] = _savefig(fig, charts_dir, "target_probabilities.png")

    # G. per-material dashboards (price + MAs, realized vol)
    dash: dict[str, str] = {}
    for material in material_prices.columns:
        sub = features[features["material"] == material] if not features.empty else pd.DataFrame()
        price = material_prices[material].dropna()
        if price.empty:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        c = _color_for(material)
        ax1.plot(price.index, price.values, color=c, linewidth=1.6, label=material)
        ax1.plot(price.index, price.rolling(cfg.ma_short_window).mean().values, color="#1d4ed8",
                 linewidth=1.0, label=f"MA{cfg.ma_short_window}")
        ax1.plot(price.index, price.rolling(cfg.ma_long_window).mean().values, color="#b45309",
                 linewidth=1.0, label=f"MA{cfg.ma_long_window}")
        ax1.legend(frameon=False, fontsize=8, ncol=3)
        _style_ax(ax1, f"{material} — price & moving averages ({cfg.frequency} periods)", "index")
        if not sub.empty and "realized_vol_20" in sub:
            ax2.plot(sub.index, sub["realized_vol_20"].values, color="#b91c1c", linewidth=1.2, label="vol 20")
            if "realized_vol_60" in sub:
                ax2.plot(sub.index, sub["realized_vol_60"].values, color="#64748b", linewidth=1.0, label="vol 60")
            ax2.legend(frameon=False, fontsize=8, ncol=2)
        _style_ax(ax2, "", "ann. vol")
        dash[material] = _savefig(fig, charts_dir, f"dashboard_{material}.png")
    charts["dashboards"] = dash  # type: ignore[assignment]

    log.info("rendered %d chart groups to %s", len(charts), charts_dir)
    return charts


# ============================================================================
# 10. SELF-CONTAINED HTML REPORT
# ============================================================================
def _img_tag(path: str) -> str:
    """Embed a PNG as a base64 data URI so the HTML is a single portable file."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" />'


def _esc(v: object) -> str:
    return html.escape(str(v))


def _table(df: pd.DataFrame, floats: int = 3, max_rows: int | None = None) -> str:
    """Render a DataFrame as a styled HTML table."""
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    d = df.head(max_rows) if max_rows else df
    head = "<tr>" + "".join(f"<th>{_esc(c)}</th>" for c in d.columns) + "</tr>"
    body = ""
    for _, row in d.iterrows():
        cells = ""
        for v in row:
            if isinstance(v, float):
                cells += f"<td>{v:,.{floats}f}</td>" if pd.notna(v) else "<td>—</td>"
            elif isinstance(v, bool):
                cells += f'<td>{"✓" if v else ""}</td>'
            else:
                cells += f"<td>{_esc(v)}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


_REPORT_CSS = """
:root{--bg:#f4f6f8;--card:#fff;--text:#172033;--muted:#667085;--border:#d9e0e8;--header:#111827;--blue:#2563eb;}
*{box-sizing:border-box;}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.55;}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px;}
header{background:var(--header);color:#fff;padding:30px 0;}
header h1{margin:0;font-size:22px;font-weight:800;letter-spacing:-.01em;}
.sub{margin-top:6px;color:#9ca3af;font-size:12px;text-transform:uppercase;letter-spacing:.08em;}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:22px 0;}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(16,24,40,.05);}
.kpi .label{color:#9ca3af;font-size:11px;text-transform:uppercase;letter-spacing:.06em;}
.kpi .value{font-size:24px;font-weight:700;margin-top:4px;}
section{margin:30px 0;}
h2{font-size:15px;font-weight:700;margin:0 0 4px;}
.desc{color:var(--muted);font-size:12px;margin-bottom:12px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;box-shadow:0 1px 2px rgba(16,24,40,.05);padding:18px;overflow:auto;margin-bottom:16px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:900px){.grid2{grid-template-columns:1fr;}}
img{width:100%;display:block;border-radius:6px;}
table{border-collapse:collapse;width:100%;font-size:12.5px;}
th{text-align:left;color:var(--muted);border-bottom:1px solid var(--border);padding:8px 10px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;}
td{padding:7px 10px;border-bottom:1px solid #eef2f6;}
tbody tr:hover{background:#f8fafc;}
.muted{color:var(--muted);}
.notice{background:#fffbeb;color:#92400e;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:12.5px;margin:16px 0;}
footer{color:var(--muted);font-size:11px;padding:24px 0 40px;}
"""


def build_html_report(charts: dict, material_prices: pd.DataFrame, proxy_cols: list[str],
                      provenance: pd.DataFrame, best_rel: pd.DataFrame, hedge: pd.DataFrame,
                      cfg: Config, real_prices: bool = False) -> str:
    """Assemble a single self-contained HTML report (base64-embedded charts)."""
    dashboards = charts.get("dashboards", {}) if isinstance(charts.get("dashboards"), dict) else {}
    n_failed = int((provenance["status"] != "ok").sum()) if "status" in provenance else 0
    degraded = ""
    if n_failed:
        failed = ", ".join(provenance.loc[provenance["status"] != "ok", "proxy"])
        degraded = (f'<div class="notice"><b>Degraded data:</b> {n_failed} proxy source(s) unavailable '
                    f'({_esc(failed)}). Affected baskets were renormalized over available components. FRED proxies '
                    f'(nickel, zinc, iron ore) need network access to the keyless fredgraph endpoint; set '
                    f'<code>FRED_API_KEY</code> to enable the official-API fallback.</div>')

    # Methodology banner: real internal prices (genuine relationships, short sample) vs
    # synthetic proxy baskets (single-proxy materials correlate 1.0 by construction).
    n_trivial = int(best_rel["structural_identity"].sum()) if "structural_identity" in best_rel else 0
    methodology_note = ""
    if real_prices:
        methodology_note = (
            f'<div class="notice" style="background:#ecfdf5;color:#065f46;border-color:#a7f3d0;">'
            f'<b>Real internal scrap prices:</b> materials are fixed-weight indexes of your own outbound sale '
            f'tickets, correlated against market proxies — so these are genuine empirical relationships, not '
            f'structural artifacts. Caveat: the sample is short ({len(material_prices)} {cfg.frequency} periods), '
            f'so treat correlations/betas as indicative and re-estimate as history accumulates.</div>')
    elif n_trivial:
        trivial = ", ".join(best_rel.loc[best_rel["structural_identity"], "material"])
        methodology_note = (
            f'<div class="notice"><b>Read these correlations with care:</b> {n_trivial} material(s) '
            f'({_esc(trivial)}) are currently modeled as a single market proxy, so their correlation, beta, and '
            f'R² to that proxy are <b>1.0 by construction</b> — a structural artifact, not an empirical finding. '
            f'Only multi-component baskets (e.g. brass, stainless) reflect a genuine estimated relationship. To make '
            f'every relationship empirical, supply real internal scrap prices via <code>material_price_csv</code>.</div>')

    kpis = [
        ("Materials", len(material_prices.columns)),
        ("Proxies", len(proxy_cols)),
        ("Frequency", cfg.frequency),
        ("Observations", len(material_prices)),
        ("Window", f"{cfg.start_date} → {cfg.end_date}"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="label">{_esc(k)}</div><div class="value">{_esc(v)}</div></div>'
                       for k, v in kpis)

    def chart_card(key: str, title: str, desc: str) -> str:
        tag = _img_tag(charts.get(key, ""))
        if not tag:
            return ""
        return f'<div class="card"><h2>{_esc(title)}</h2><div class="desc">{_esc(desc)}</div>{tag}</div>'

    dash_html = "".join(
        f'<div class="card"><h2>{_esc(m)}</h2>{_img_tag(p)}</div>' for m, p in dashboards.items())

    hedge_primary = hedge[hedge["is_primary"]] if "is_primary" in hedge else hedge

    generated = datetime.today().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scrap Metals Sell-Signal Report</title><style>{_REPORT_CSS}</style></head>
<body>
<header><div class="wrap"><h1>Scrap Metals — Sell-Signal Data Report</h1>
<div class="sub">Proxy mapping · correlation · hedge · features · targets</div></div></header>
<div class="wrap">
{degraded}
{methodology_note}
<div class="kpis">{kpi_html}</div>

<section><div class="card"><h2>Proxy mapping used</h2>
<div class="desc">Which free source actually supplied each proxy, and which materials use it.</div>
{_table(provenance, floats=0)}</div></section>

<section class="grid2">
{chart_card("indexes", "Material indexes", "Each material's proxy/basket, rebased to 100 at the start date.")}
{chart_card("heatmap", "Correlation matrix", "Full-sample return correlation across materials and proxies.")}
</section>

<section><h2>Best proxy & hedge relationships</h2>
<div class="desc">Strongest correlation per material (across horizon × lag) and the primary-proxy OLS hedge fit.</div>
<div class="grid2">
<div class="card"><h2>Best proxy by correlation</h2>{_table(best_rel, floats=3)}</div>
<div class="card"><h2>Primary-proxy hedge (OLS)</h2>{_table(hedge_primary, floats=4)}</div>
</div>
{chart_card("best_proxy", "Best proxy correlation", "Best achievable |correlation| to a single futures/ETF proxy.")}
</section>

<section class="grid2">
{chart_card("rolling_corr", "Rolling correlation", "Stability of the material↔proxy relationship over time.")}
{chart_card("rolling_beta", "Rolling hedge ratio", "Time-varying beta — how many proxy units hedge one material unit.")}
</section>

<section>{chart_card("targets", "Forward-down probability", "Base rate of negative forward returns — the sell-signal's class balance.")}</section>

<section><h2>Per-material dashboards</h2>
<div class="desc">Price with 20/50-period moving averages, and annualized realized volatility.</div>
{dash_html}</section>

<footer class="wrap">Generated {generated} · sell_signal/pipeline.py · charts are illustrative of proxy data, not trading advice.</footer>
</div></body></html>"""


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    setup_logging()
    cfg = CFG
    ensure_output_dir(cfg)
    log.info("=== scrap-metals sell-signal data pipeline ===")
    log.info("window %s..%s  frequency=%s  output=%s",
             cfg.start_date, cfg.end_date, cfg.frequency, cfg.output_dir)

    # 2. collect raw proxy data + provenance
    raw_proxies, provenance = collect_all_proxies(cfg)
    _save(raw_proxies, cfg, "raw_price_data.csv")

    # 3. clean / align / resample proxies
    proxy_prices = clean_prices(raw_proxies, cfg)

    # 4. build material (basket) series, then assemble combined price panel
    material_prices, basket_used = build_material_series(proxy_prices, cfg)
    materials_present = list(material_prices.columns)
    proxy_cols = list(proxy_prices.columns)
    real_mode = bool(cfg.material_price_csv and os.path.exists(cfg.material_price_csv))
    log.info("material series source: %s", "REAL internal prices" if real_mode else "synthetic proxy baskets")

    combined_prices = pd.concat([proxy_prices, material_prices], axis=1).sort_index()
    _save(combined_prices, cfg, "cleaned_price_data.csv")

    returns = compute_returns(combined_prices)
    _save(returns, cfg, "returns_data.csv")

    # provenance + baskets actually used
    prov_out = provenance.merge(
        basket_used.groupby("component")["material"].apply(lambda s: ",".join(sorted(set(s)))).rename("used_by_materials"),
        left_on="proxy", right_index=True, how="left",
    )
    _save(prov_out, cfg, "proxy_mapping_used.csv", index=False)
    _save(basket_used, cfg, "basket_weights_used.csv", index=False)

    # primary-proxy anchor per material (heaviest available component)
    material_to_proxy = {m: primary_proxy_for(m, proxy_cols) for m in materials_present}
    material_to_proxy = {m: p for m, p in material_to_proxy.items() if p}

    # 5. correlations
    corr_mat = correlation_matrix(returns)
    _save(corr_mat, cfg, "correlation_matrix.csv")

    roll_corr = rolling_correlations(returns, cfg, material_to_proxy)
    _save(roll_corr, cfg, "rolling_correlations.csv")

    best_rel, full_scan = best_proxy_relationships(combined_prices, cfg, materials_present, proxy_cols, real_mode)
    if best_rel.empty:
        log.warning("correlation scan produced no rows (insufficient overlapping data)")
    else:
        _save(best_rel, cfg, "best_proxy_relationships.csv", index=False)
        _save(full_scan, cfg, "correlation_scan_full.csv", index=False)

    # 6. hedge relationships
    hedge = hedge_relationships(returns, cfg, materials_present, proxy_cols, material_to_proxy)
    _save(hedge, cfg, "hedge_relationships.csv", index=False)

    roll_beta = rolling_betas(returns, cfg, material_to_proxy)
    _save(roll_beta, cfg, "rolling_betas.csv")

    # 7. predictive features
    features = build_features(material_prices, proxy_prices, cfg, material_to_proxy)
    _save(features, cfg, "predictive_features.csv")

    # 8. targets + final modeling dataset
    targets = build_targets(material_prices, cfg)
    _save(targets, cfg, "target_variables.csv")

    if not features.empty and not targets.empty:
        modeling = features.merge(
            targets.reset_index(), on=["date", "material"], how="left",
        ).set_index("date")
        _save(modeling, cfg, "modeling_dataset.csv")
    else:
        log.warning("features or targets empty; modeling_dataset.csv not written")

    # 9 + 10. charts + self-contained HTML report
    charts: dict = {}
    if cfg.make_plots or cfg.make_report:
        if plt is None:
            log.warning("matplotlib unavailable; skipping charts/report")
        else:
            charts = render_charts(material_prices, proxy_prices, corr_mat, roll_corr, roll_beta,
                                   best_rel, features, targets, cfg)
    if cfg.make_report and charts:
        report_html = build_html_report(charts, material_prices, proxy_cols, prov_out,
                                        best_rel, hedge, cfg, real_mode)
        report_path = os.path.join(cfg.output_dir, cfg.report_name)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report_html)
        log.info("wrote %-32s (self-contained HTML)", cfg.report_name)

    log.info("=== done. %d materials, %d proxies. outputs in %s ===",
             len(materials_present), len(proxy_cols), cfg.output_dir)


if __name__ == "__main__":
    main()
