# sell_signal/run_real.py
"""Run the sell-signal pipeline on REAL internal scrap prices.

Pipeline:
    1. Build a fixed-weight weekly scrap-price panel from outbound sale tickets
       (sell_signal/build_scrap_prices.py).
    2. Point the sell-signal pipeline at it via Config.material_price_csv and run.

Because the real ticket history is short (~6 months / ~25 weekly points), this uses
WEEKLY frequency and short lookback/forward windows so features and targets are
actually computable. Re-tune as more history accumulates. Needs network access for
the market proxies (yfinance + FRED); the scrap series itself is fully local.

Run (from the project root):
    python sell_signal/run_real.py
"""

from __future__ import annotations

import dataclasses
import os
import sys

# resolve sibling modules (pipeline, build_scrap_prices) and the project root (`import src`)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import build_scrap_prices as builder
import pipeline as P

SCRAP_CSV = builder.DEFAULT_OUT

# Windows sized for a ~25-week sample (all in WEEKLY periods).
REAL_CONFIG = dataclasses.replace(
    P.Config(),
    frequency="weekly",
    material_price_csv=SCRAP_CSV,
    return_windows=(1, 2, 4, 8, 12),
    realized_vol_windows=(4, 8),
    rolling_corr_window=12,
    rolling_beta_window=12,
    z_window=12,
    drawdown_window=8,
    ma_short_window=4,
    ma_long_window=12,
    momentum_window=4,
    forward_horizons=(1, 2, 4),
    target_drawdown_window=8,
    return_horizons=("weekly",),  # don't upsample weekly scrap into fake daily prints
    lags=(0, 1, 2, 4),
)


def main() -> None:
    builder.build_scrap_prices(out_path=SCRAP_CSV)  # step 1: refresh the scrap panel
    P.CFG = REAL_CONFIG                              # step 2: run the pipeline on it
    P.main()


if __name__ == "__main__":
    main()
