"""Causal feature engineering.

EVERY feature is computed using only information available at time t (current
and past rows). Rolling windows are trailing (pandas default) — never centered,
never forward-looking. The functions return (feature_frame, feature_names).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import setup_logging

logger = setup_logging()

SKEW_KURT_WINDOWS = [10, 20, 60]


def _consecutive_run(up_mask: pd.Series, want_up: bool) -> pd.Series:
    """Count of consecutive prior days matching direction, as known at day t.

    Uses the sign of the SAME-DAY return (known at the close of day t), so it is
    causal for predicting day t+1.
    """
    target = up_mask if want_up else ~up_mask
    target = target.fillna(False)
    # Reset the counter whenever the streak breaks.
    grp = (target != target.shift()).cumsum()
    run = target.groupby(grp).cumsum()
    return run.where(target, 0).astype(float)


def build_features(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Build the causal feature matrix aligned to ``df`` (same index)."""
    feat_cfg = config["features"]
    windows: List[int] = list(feat_cfg["rolling_windows"])
    has_volume = "volume" in df.columns

    close = df["close"].astype(float)
    open_ = df["open"].astype(float) if "open" in df else pd.Series(np.nan, index=df.index)
    high = df["high"].astype(float) if "high" in df else pd.Series(np.nan, index=df.index)
    low = df["low"].astype(float) if "low" in df else pd.Series(np.nan, index=df.index)

    f = pd.DataFrame(index=df.index)

    # --- core single-bar features (all known at day t) ---
    f["daily_return"] = close.pct_change()
    f["log_return"] = np.log(close / close.shift(1))
    f["open_to_close_return"] = close / open_ - 1.0
    f["high_low_range"] = high / low - 1.0
    rng = (high - low)
    f["close_position_in_range"] = (close - low) / rng.replace(0.0, np.nan)

    daily_ret = f["daily_return"]

    # --- rolling features (trailing windows, min_periods == window) ---
    for w in windows:
        f[f"roll_return_{w}"] = close / close.shift(w) - 1.0          # window-return
        f[f"roll_vol_{w}"] = daily_ret.rolling(w, min_periods=w).std()
        f[f"roll_mean_return_{w}"] = daily_ret.rolling(w, min_periods=w).mean()
        f[f"ma_dist_{w}"] = close / close.rolling(w, min_periods=w).mean() - 1.0
        f[f"momentum_{w}"] = close / close.shift(w) - 1.0

    for w in SKEW_KURT_WINDOWS:
        f[f"roll_skew_{w}"] = daily_ret.rolling(w, min_periods=w).skew()
        f[f"roll_kurt_{w}"] = daily_ret.rolling(w, min_periods=w).kurt()

    # --- direction / streak features ---
    up_mask = daily_ret > 0
    f["prev_day_direction"] = up_mask.shift(0).astype(float)  # sign of today's move
    f["consecutive_up_days"] = _consecutive_run(up_mask, want_up=True)
    f["consecutive_down_days"] = _consecutive_run(up_mask, want_up=False)

    # --- day-of-week one-hot (known in advance) ---
    if feat_cfg.get("include_day_of_week", True):
        dow = df["date"].dt.dayofweek  # Mon=0 .. Sun=6
        for d in range(5):             # trading week Mon..Fri
            f[f"dow_{d}"] = (dow == d).astype(float)

    # --- optional volume features (past-only) ---
    if has_volume and feat_cfg.get("include_volume_features", True):
        vol = df["volume"].astype(float)
        f["volume_pct_change"] = vol.pct_change()
        for w in [10, 20]:
            mean = vol.rolling(w, min_periods=w).mean()
            std = vol.rolling(w, min_periods=w).std()
            f[f"volume_zscore_{w}"] = (vol - mean) / std.replace(0.0, np.nan)

    # Inf -> NaN; the per-window median imputer handles residual NaNs later.
    f = f.replace([np.inf, -np.inf], np.nan)

    feature_names = list(f.columns)
    logger.info("Built %d causal features.", len(feature_names))
    return f, feature_names
