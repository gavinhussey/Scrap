"""Walk-forward backtest with strict leakage controls.

For each test day t:
  * Train window = the trailing ``initial_train_size`` rows BEFORE t (rolling).
  * Preprocessing (impute + scale) is fit ONLY on that train window.
  * Non-LSTM models are retrained every day; the LSTM is retrained on the
    configured cadence and reused in between, scoring each day with a causally
    constructed sequence.
Nothing from day t or later ever touches the fit for day t.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_and_clean
from .features import SKEW_KURT_WINDOWS, build_features
from .models_baseline import (
    NaiveDirectionModel,
    make_logistic,
    make_random_forest,
    make_svc,
)
from .models_lstm import TF_AVAILABLE, LSTMClassifier, make_sequences
from .preprocessing import Preprocessor
from .targets import FUTURE_RETURN_COL, TARGET_COL, create_direction_target
from .utils import ensure_dir, set_global_seed, setup_logging

logger = setup_logging()


def prepare_modeling_frame(config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Load -> features -> target -> drop warmup. Returns (frame, feature_names).

    The returned frame has columns: date, close, future_return, target_up, and
    every feature column. Rows are reset to a clean 0..K-1 index.
    """
    df_clean = load_and_clean(config)
    feats, feature_names = build_features(df_clean, config)
    combined = pd.concat([df_clean, feats], axis=1)

    tcfg = config["target"]
    framed = create_direction_target(
        combined, tcfg["horizon_days"], tcfg["direction_threshold"]
    )

    # Drop the rolling warm-up region (longest window of NaNs at the start).
    warmup = max(list(config["features"]["rolling_windows"]) + SKEW_KURT_WINDOWS)
    framed = framed.iloc[warmup:].reset_index(drop=True)

    keep = ["date", "close", FUTURE_RETURN_COL, TARGET_COL] + feature_names
    framed = framed[keep]
    logger.info("Modeling frame ready: %d rows, %d features.", len(framed), len(feature_names))
    return framed, feature_names


def _active_models(config: Dict, n_train_rows: int) -> Dict[str, bool]:
    """Decide which models run, applying the SVC size guard."""
    mcfg = config["models"]
    active = {
        "naive": mcfg.get("run_naive", True),
        "logistic": mcfg.get("run_logistic", True),
        "random_forest": mcfg.get("run_random_forest", True),
        "svc": mcfg.get("run_svc", True),
        "lstm": mcfg.get("run_lstm", True),
    }
    svc_cap = config.get("limits", {}).get("svc_max_train_rows", 4000)
    if active["svc"] and n_train_rows > svc_cap:
        logger.warning(
            "SVC disabled: train window (%d) exceeds svc_max_train_rows (%d).",
            n_train_rows, svc_cap,
        )
        active["svc"] = False
    if active["lstm"] and not TF_AVAILABLE:
        logger.warning("LSTM disabled: TensorFlow not importable.")
        active["lstm"] = False
    return active


def run_walk_forward(config: Dict) -> pd.DataFrame:
    """Execute the rolling backtest and persist the per-day prediction log."""
    set_global_seed()
    M, feature_names = prepare_modeling_frame(config)

    bcfg = config["backtest"]
    window = int(bcfg["initial_train_size"])
    step = int(bcfg["step_size"])
    retrain_every = int(bcfg["retrain_lstm_every"])
    val_frac = float(bcfg["validation_fraction"])
    lookback = int(config["features"]["lookback_sequence_days"])

    K = len(M)
    if K <= window + 1:
        raise ValueError(
            f"Not enough rows ({K}) for initial_train_size={window}. "
            f"Provide more data or lower initial_train_size."
        )

    active = _active_models(config, window)
    logger.info("Active models: %s", [m for m, on in active.items() if on])

    Xall = M[feature_names].to_numpy(dtype=float)
    y_all = M[TARGET_COL].to_numpy(dtype=int)
    close = M["close"].to_numpy(dtype=float)
    fut_ret = M[FUTURE_RETURN_COL].to_numpy(dtype=float)
    dates = M["date"].to_numpy()
    daily_return_idx = feature_names.index("daily_return")

    test_positions = list(range(window, K, step))
    max_days = bcfg.get("max_backtest_days")
    if max_days:
        test_positions = test_positions[: int(max_days)]
    logger.info("Backtesting %d test days (window=%d).", len(test_positions), window)

    # LSTM state, refreshed on the retrain cadence.
    lstm_model = None
    lstm_prep = None

    rows = []
    for n_done, i in enumerate(test_positions):
        tr0 = max(0, i - window)
        X_tr_raw, y_tr = Xall[tr0:i], y_all[tr0:i]
        X_te_raw = Xall[i : i + 1]

        prep = Preprocessor().fit(X_tr_raw)
        X_tr_s = prep.transform(X_tr_raw)
        X_te_s = prep.transform(X_te_raw)

        rec = {
            "date": dates[i],
            "close": close[i],
            "target": int(y_all[i]),
            "future_return": float(fut_ret[i]),
        }

        # --- naive (uses UNSCALED daily_return) ---
        if active["naive"]:
            p = NaiveDirectionModel(daily_return_idx).predict_proba_up(X_te_raw)[0]
            rec["naive_prob"] = float(p)

        # --- sklearn classifiers (scaled features, retrained daily) ---
        sklearn_specs = []
        if active["logistic"]:
            sklearn_specs.append(make_logistic())
        if active["random_forest"]:
            sklearn_specs.append(make_random_forest())
        if active["svc"]:
            sklearn_specs.append(make_svc())
        for mdl in sklearn_specs:
            mdl.fit(X_tr_s, y_tr)
            rec[f"{mdl.name}_prob"] = float(mdl.predict_proba_up(X_te_s)[0])

        # --- LSTM (retrained on cadence, scored every day) ---
        if active["lstm"]:
            need_retrain = (lstm_model is None) or ((i - window) % retrain_every == 0)
            if need_retrain:
                lstm_prep = Preprocessor().fit(X_tr_raw)
                scaled_tr = lstm_prep.transform(X_tr_raw)
                X_seq, y_seq = make_sequences(scaled_tr, y_tr, lookback)
                if len(X_seq) >= 10:
                    lstm_model = LSTMClassifier(config["lstm"], lookback).fit(
                        X_seq, y_seq, validation_fraction=val_frac
                    )
                    logger.info("  [day %d] LSTM retrained on %d sequences.", i, len(X_seq))
                else:
                    lstm_model = None
            if lstm_model is not None and i - lookback + 1 >= 0:
                ctx_raw = Xall[i - lookback + 1 : i + 1]
                ctx_scaled = lstm_prep.transform(ctx_raw)
                seq = ctx_scaled.reshape(1, lookback, -1)
                rec["lstm_prob"] = float(lstm_model.predict_proba_up(seq)[0])

        rows.append(rec)
        if (n_done + 1) % 250 == 0:
            logger.info("  ... %d/%d test days done.", n_done + 1, len(test_positions))

    preds = pd.DataFrame(rows)

    # Derive predicted class (>=0.5) for every model probability column present.
    for col in [c for c in preds.columns if c.endswith("_prob")]:
        model = col[: -len("_prob")]
        preds[f"{model}_pred"] = (preds[col] >= 0.5).astype(int)

    out_dir = ensure_dir("reports/backtests")
    out_path = out_dir / "walk_forward_predictions.csv"
    preds.to_csv(out_path, index=False)
    logger.info("Saved walk-forward predictions -> %s (%d rows).", out_path, len(preds))
    return preds


def model_names_from_predictions(preds: pd.DataFrame) -> List[str]:
    """Model names that have probability columns in the prediction log."""
    return [c[: -len("_prob")] for c in preds.columns if c.endswith("_prob")]
