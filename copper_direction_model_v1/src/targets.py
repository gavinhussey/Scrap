"""Direction target construction.

target_up_t = 1 if future_return > threshold else 0
where future_return = close[t + horizon] / close[t] - 1.

``future_return`` is retained ONLY for backtest evaluation; it must never be
used as a model feature (enforced by tests/test_no_leakage.py).
"""
from __future__ import annotations

import pandas as pd

from .utils import setup_logging

logger = setup_logging()

FUTURE_RETURN_COL = "future_return"
TARGET_COL = "target_up"


def create_direction_target(
    df: pd.DataFrame, horizon_days: int = 1, threshold: float = 0.0
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``future_return`` and ``target_up`` columns.

    The final ``horizon_days`` rows have an unknown future and are dropped.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    out = df.copy()
    future_close = out["close"].shift(-horizon_days)
    out[FUTURE_RETURN_COL] = future_close / out["close"] - 1.0
    out[TARGET_COL] = (out[FUTURE_RETURN_COL] > threshold).astype("Int64")

    # Drop rows whose future is unknown (the last `horizon_days` rows).
    before = len(out)
    out = out.dropna(subset=[FUTURE_RETURN_COL]).reset_index(drop=True)
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    logger.info(
        "Target built (horizon=%d, threshold=%.4f). Dropped %d unknown-future rows; "
        "up-share=%.3f.",
        horizon_days, threshold, before - len(out), out[TARGET_COL].mean(),
    )
    return out
