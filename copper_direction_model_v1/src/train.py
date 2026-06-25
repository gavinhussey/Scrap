"""CLI: run the full walk-forward backtest and evaluation.

    python -m src.train --config config.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from .backtest import model_names_from_predictions, run_walk_forward
from .evaluate import run_evaluation
from .utils import load_config, setup_logging

logger = setup_logging()


def parse_args():
    p = argparse.ArgumentParser(description="Copper direction V1 — walk-forward training.")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    return p.parse_args()


def _print_ranking(summary: pd.DataFrame) -> None:
    cols = [
        "model", "balanced_accuracy", "directional_hit_rate",
        "strategy_sharpe", "strategy_max_drawdown", "pct_days_traded",
    ]
    table = summary[cols].copy()
    ranked = table.sort_values(
        ["balanced_accuracy", "directional_hit_rate", "strategy_sharpe"],
        ascending=[False, False, False],
    )
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("\n================ MODEL RANKING (walk-forward, out-of-sample) ================")
    print(ranked.to_string(index=False))
    print("=" * 78)
    print(
        "\nNOTE: V1 research output. Directional accuracy near 0.50 means NO real edge.\n"
        "Do not interpret any single run as evidence of a tradeable signal. Validate on\n"
        "real COMEX copper data and out-of-sample periods before any trading or hedging use."
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    logger.info("Starting walk-forward backtest with config '%s'.", args.config)

    preds = run_walk_forward(config)
    models = model_names_from_predictions(preds)
    if not models:
        raise RuntimeError("No model produced predictions; nothing to evaluate.")

    summary = run_evaluation(preds, models, config)
    _print_ranking(summary)
    logger.info("Done. See reports/backtests/ and reports/figures/.")


if __name__ == "__main__":
    main()
