"""CLI: predict the direction of the NEXT trading day.

    python -m src.predict_next --config config.yaml

Trains each enabled model on the most recent training window, then predicts the
probability that the next session closes higher than the latest close.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import _active_models
from .data_loader import load_and_clean
from .features import SKEW_KURT_WINDOWS, build_features
from .models_baseline import (
    NaiveDirectionModel, make_logistic, make_random_forest, make_svc,
)
from .models_lstm import TF_AVAILABLE, LSTMClassifier, make_sequences
from .preprocessing import Preprocessor
from .targets import create_direction_target
from .utils import ensure_dir, load_config, resolve_path, set_global_seed, setup_logging

logger = setup_logging()

# Weighted ensemble (naive excluded). Renormalized over whatever is available.
ENSEMBLE_WEIGHTS = {"lstm": 0.40, "logistic": 0.20, "random_forest": 0.20, "svc": 0.20}


def parse_args():
    p = argparse.ArgumentParser(description="Predict next-day copper direction.")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def _ensemble(probs: dict) -> float:
    """Weighted average over available weighted models; fall back to plain mean."""
    weighted = {m: probs[m] for m in ENSEMBLE_WEIGHTS if m in probs}
    if weighted:
        total_w = sum(ENSEMBLE_WEIGHTS[m] for m in weighted)
        return sum(probs[m] * ENSEMBLE_WEIGHTS[m] for m in weighted) / total_w
    return float(np.mean(list(probs.values()))) if probs else float("nan")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_global_seed()

    # Build features over the full clean series; keep the latest (target-unknown) row.
    df_clean = load_and_clean(config)
    feats, feature_names = build_features(df_clean, config)
    combined = pd.concat([df_clean, feats], axis=1)
    warmup = max(list(config["features"]["rolling_windows"]) + SKEW_KURT_WINDOWS)
    combined = combined.iloc[warmup:].reset_index(drop=True)

    tcfg = config["target"]
    horizon = int(tcfg["horizon_days"])
    train_frame = create_direction_target(combined, horizon, tcfg["direction_threshold"])

    window = int(config["backtest"]["initial_train_size"])
    train_frame = train_frame.iloc[-window:] if len(train_frame) > window else train_frame

    X_tr_raw = train_frame[feature_names].to_numpy(float)
    y_tr = train_frame["target_up"].to_numpy(int)

    latest = combined.iloc[-1]
    X_pred_raw = combined[feature_names].to_numpy(float)[-1:]

    active = _active_models(config, len(train_frame))
    prep = Preprocessor().fit(X_tr_raw)
    X_tr_s, X_pred_s = prep.transform(X_tr_raw), prep.transform(X_pred_raw)

    probs = {}
    if active["naive"]:
        idx = feature_names.index("daily_return")
        probs["naive"] = float(NaiveDirectionModel(idx).predict_proba_up(X_pred_raw)[0])
    for on, maker in [
        (active["logistic"], make_logistic),
        (active["random_forest"], make_random_forest),
        (active["svc"], make_svc),
    ]:
        if on:
            m = maker().fit(X_tr_s, y_tr)
            probs[m.name] = float(m.predict_proba_up(X_pred_s)[0])

    if active["lstm"] and TF_AVAILABLE:
        lookback = int(config["features"]["lookback_sequence_days"])
        lstm_prep = Preprocessor().fit(X_tr_raw)
        X_seq, y_seq = make_sequences(lstm_prep.transform(X_tr_raw), y_tr, lookback)
        if len(X_seq) >= 10:
            lstm = LSTMClassifier(config["lstm"], lookback).fit(
                X_seq, y_seq, validation_fraction=config["backtest"]["validation_fraction"]
            )
            ctx = combined[feature_names].to_numpy(float)[-lookback:]
            seq = lstm_prep.transform(ctx).reshape(1, lookback, -1)
            probs["lstm"] = float(lstm.predict_proba_up(seq)[0])

    p_ens = _ensemble(probs)

    scfg = config["signals"]
    if p_ens >= scfg["long_threshold"]:
        signal = "Bullish"
    elif p_ens <= scfg["short_threshold"] and scfg.get("allow_short_signal", True):
        signal = "Bearish"
    else:
        signal = "No trade"

    latest_date = pd.to_datetime(latest["date"]).date().isoformat()
    latest_close = float(latest["close"])

    print("\n================ NEXT-DAY COPPER DIRECTION (V1, research) ================")
    print(f"latest date : {latest_date}")
    print(f"latest close: {latest_close:.4f}")
    print("P(up) by model:")
    for m, p in probs.items():
        print(f"  {m:<14s} {p:0.4f}")
    print(f"ensemble P(up): {p_ens:0.4f}")
    print(f"signal        : {signal}")
    print("=" * 73)
    print("V1 research output — NOT financial advice. Validate on real data first.")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_date": latest_date,
        "latest_close": latest_close,
        "horizon_days": horizon,
        "model_probabilities": probs,
        "ensemble_prob_up": p_ens,
        "signal": signal,
        "thresholds": {
            "long": scfg["long_threshold"], "short": scfg["short_threshold"],
            "allow_short": scfg.get("allow_short_signal", True),
        },
        "disclaimer": "V1 research model. Not financial advice. Not production-ready.",
    }
    out_path = resolve_path("reports/latest_prediction.json")
    ensure_dir("reports")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
