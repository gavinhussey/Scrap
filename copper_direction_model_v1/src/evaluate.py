"""Evaluation: classification metrics, signal rules, strategy stats, and figures.

Consumes the walk-forward prediction log and the config. Writes metric tables to
reports/backtests/ and charts to reports/figures/.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

from .utils import ensure_dir, setup_logging

logger = setup_logging()


def make_signals(prob: np.ndarray, config: Dict) -> np.ndarray:
    """Map P(up) to a position signal in {-1, 0, +1} using config thresholds."""
    scfg = config["signals"]
    longt, shortt = scfg["long_threshold"], scfg["short_threshold"]
    allow_short = scfg.get("allow_short_signal", True)
    sig = np.zeros(len(prob), dtype=int)
    sig[prob >= longt] = 1
    sig[prob <= shortt] = -1 if allow_short else 0
    return sig


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _sharpe(returns: np.ndarray, ann: int) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns) * np.sqrt(ann))


def _max_drawdown(cum_curve: np.ndarray) -> float:
    if len(cum_curve) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(cum_curve)
    drawdown = (cum_curve - running_max)
    return float(drawdown.min())


def _strategy_returns(signal: np.ndarray, fut_ret: np.ndarray, cost_bps: float) -> np.ndarray:
    """signal * next-day return, minus turnover transaction cost."""
    cost = cost_bps / 1e4
    turnover = np.abs(np.diff(np.concatenate([[0], signal])))
    return signal * fut_ret - turnover * cost


def evaluate_model(preds: pd.DataFrame, model: str, config: Dict) -> Dict:
    """All metrics for one model."""
    y = preds["target"].to_numpy(int)
    fut = preds["future_return"].to_numpy(float)
    prob = preds[f"{model}_prob"].to_numpy(float)
    pred = (prob >= 0.5).astype(int)
    signal = make_signals(prob, config)

    ann = config["evaluation"]["annualization_days"]
    cost_bps = config["evaluation"]["transaction_cost_bps"]
    strat = _strategy_returns(signal, fut, cost_bps)
    cum = np.cumsum(strat)

    traded_mask = signal != 0
    long_mask = signal == 1
    short_mask = signal == -1
    cm = confusion_matrix(y, pred, labels=[0, 1])

    return {
        "model": model,
        "n": len(y),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_up": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "precision_down": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_up": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_down": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": _safe_auc(y, prob),
        "brier": float(brier_score_loss(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "directional_hit_rate": float((pred == y).mean()),
        "pct_days_traded": float(traded_mask.mean()),
        "avg_ret_when_long": float(fut[long_mask].mean()) if long_mask.any() else float("nan"),
        "avg_ret_when_short": float(fut[short_mask].mean()) if short_mask.any() else float("nan"),
        "strategy_total_return": float(strat.sum()),
        "strategy_cumulative_return": float(cum[-1]) if len(cum) else float("nan"),
        "strategy_sharpe": _sharpe(strat[traded_mask] if traded_mask.any() else strat, ann),
        "strategy_max_drawdown": _max_drawdown(cum),
        "confusion_tn": int(cm[0, 0]), "confusion_fp": int(cm[0, 1]),
        "confusion_fn": int(cm[1, 0]), "confusion_tp": int(cm[1, 1]),
    }


def yearly_metrics(preds: pd.DataFrame, models: List[str], config: Dict) -> pd.DataFrame:
    """Directional hit rate + strategy return per calendar year, per model."""
    df = preds.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    rows = []
    for model in models:
        prob = df[f"{model}_prob"].to_numpy(float)
        df["_pred"] = (prob >= 0.5).astype(int)
        df["_strat"] = _strategy_returns(
            make_signals(prob, config), df["future_return"].to_numpy(float),
            config["evaluation"]["transaction_cost_bps"],
        )
        for year, g in df.groupby("year"):
            rows.append({
                "model": model, "year": int(year), "n": len(g),
                "hit_rate": float((g["_pred"] == g["target"]).mean()),
                "strategy_return": float(g["_strat"].sum()),
            })
    return pd.DataFrame(rows)


def regime_metrics(preds: pd.DataFrame, models: List[str], config: Dict) -> pd.DataFrame:
    """Hit rate in high- vs low-volatility regimes (split on median |future_return|).

    Volatility proxied by the realized magnitude of moves; the split is computed
    on the backtest sample only (used for reporting, not for any model input).
    """
    df = preds.copy()
    vol = df["future_return"].abs()
    median = vol.median()
    df["_regime"] = np.where(vol > median, "high_vol", "low_vol")
    rows = []
    for model in models:
        df["_pred"] = (df[f"{model}_prob"].to_numpy(float) >= 0.5).astype(int)
        for regime, g in df.groupby("_regime"):
            rows.append({
                "model": model, "regime": regime, "n": len(g),
                "hit_rate": float((g["_pred"] == g["target"]).mean()),
            })
    return pd.DataFrame(rows)


# ----------------------------- figures --------------------------------------

def _plot_cumulative(preds: pd.DataFrame, models: List[str], config: Dict, fig_dir):
    plt.figure(figsize=(10, 6))
    for model in models:
        prob = preds[f"{model}_prob"].to_numpy(float)
        strat = _strategy_returns(
            make_signals(prob, config), preds["future_return"].to_numpy(float),
            config["evaluation"]["transaction_cost_bps"],
        )
        plt.plot(pd.to_datetime(preds["date"]), np.cumsum(strat), label=model)
    plt.title("Cumulative strategy return (sum of signal * next-day return)")
    plt.xlabel("date"); plt.ylabel("cumulative return"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "cumulative_returns.png", dpi=120); plt.close()


def _plot_lstm_prob(preds: pd.DataFrame, fig_dir):
    if "lstm_prob" not in preds.columns:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(preds["date"]), preds["lstm_prob"], lw=0.8)
    plt.axhline(0.5, color="k", ls="--", lw=0.7)
    plt.title("LSTM P(up) over time"); plt.xlabel("date"); plt.ylabel("P(up)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(fig_dir / "lstm_probability.png", dpi=120); plt.close()


def _plot_confusion(preds: pd.DataFrame, models: List[str], fig_dir):
    for model in models:
        pred = (preds[f"{model}_prob"].to_numpy(float) >= 0.5).astype(int)
        cm = confusion_matrix(preds["target"], pred, labels=[0, 1])
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            plt.text(c, r, str(v), ha="center", va="center")
        plt.xticks([0, 1], ["down", "up"]); plt.yticks([0, 1], ["down", "up"])
        plt.xlabel("predicted"); plt.ylabel("actual"); plt.title(f"Confusion — {model}")
        plt.tight_layout(); plt.savefig(fig_dir / f"confusion_{model}.png", dpi=120); plt.close()


def _plot_rolling_hit(preds: pd.DataFrame, models: List[str], fig_dir, window: int = 60):
    plt.figure(figsize=(10, 5))
    for model in models:
        pred = (preds[f"{model}_prob"].to_numpy(float) >= 0.5).astype(int)
        hit = pd.Series((pred == preds["target"].to_numpy(int)).astype(float))
        plt.plot(pd.to_datetime(preds["date"]), hit.rolling(window, min_periods=window).mean(),
                 label=model, lw=1.0)
    plt.axhline(0.5, color="k", ls="--", lw=0.7)
    plt.title(f"Rolling {window}-day directional hit rate")
    plt.xlabel("date"); plt.ylabel("hit rate"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir / "rolling_hit_rate.png", dpi=120); plt.close()


def run_evaluation(preds: pd.DataFrame, models: List[str], config: Dict) -> pd.DataFrame:
    """Compute every metric table and figure; persist them. Returns the summary."""
    bt_dir = ensure_dir("reports/backtests")
    fig_dir = ensure_dir("reports/figures")

    summary = pd.DataFrame([evaluate_model(preds, m, config) for m in models])
    summary.to_csv(bt_dir / "metrics_summary.csv", index=False)
    yearly_metrics(preds, models, config).to_csv(bt_dir / "yearly_metrics.csv", index=False)
    regime_metrics(preds, models, config).to_csv(bt_dir / "regime_metrics.csv", index=False)

    _plot_cumulative(preds, models, config, fig_dir)
    _plot_lstm_prob(preds, fig_dir)
    _plot_confusion(preds, models, fig_dir)
    _plot_rolling_hit(preds, models, fig_dir)

    logger.info("Saved metrics tables and %d figures.", 3 + len(models))
    return summary
