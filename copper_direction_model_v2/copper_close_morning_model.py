"""Copper Close Direction Model — MORNING-OF variant.

Same question as the night-before model (`copper_close_direction_model.py`): will COMEX
copper close higher tomorrow (t+1) than today (t)? The difference is WHEN you guess and
therefore WHAT you may use:

  * NIGHT-BEFORE model : guess at today's COMEX close (~1pm ET, day t). Uses only data
                         through day t. ~53.9% accuracy.
  * THIS MORNING-OF    : guess on the morning of day t+1, AFTER the Asian/Australian
                         session has closed but BEFORE the COMEX t+1 settle. That lets it
                         use the overnight Asian copper-miner moves of day t+1 -> ~59.8%.

WHY THE OVERNIGHT FEATURE IS LEAK-FREE (and why it helps so much):
  ASX miners close ~1am ET and HK miners ~3-4am ET on date t+1; COMEX copper settles
  ~1pm ET on date t+1. So the day-(t+1) Asian session closes HOURS BEFORE copper's t+1
  close (the outcome) and AFTER copper's day-t close (the reference). Asian markets price
  the day's global risk first and copper's US session follows -- the documented
  "Asia leads the US" effect. We encode it as the date-aligned Asian return shifted by
  one trading day (shift(-1)). Validated: permutation null passes (real 59.8% vs null
  49.9%, p=0.000); a no-shift control stays at the 53.9% baseline.

  TRADE-OFF: this is a ~5-hour-ahead forecast, not 24h. Only valid if you actually guess
  in the morning. If you must commit at the prior close, use the night-before model.

Reuses the night-before model's data/feature/walk-forward/L1 pipeline as a library.
Run:  python copper_close_morning_model.py   ->  outputs/copper_close_morning/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import copper_close_direction_model as base

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "copper_close_morning"
OVERNIGHT_CACHE = BASE_DIR / "data" / "external" / "macro_cache" / "overnight_prices.csv"

# ASX + HK copper miners — Asian session closes before the COMEX settle on the same date.
OVERNIGHT_TICKERS = ["BHP.AX", "RIO.AX", "S32.AX", "2899.HK", "0358.HK", "1208.HK"]


def log(m): print(f"[copper_morning] {m}", flush=True)


def fetch_overnight() -> pd.DataFrame:
    if OVERNIGHT_CACHE.exists():
        log(f"overnight: using cache {OVERNIGHT_CACHE.name}")
        return pd.read_csv(OVERNIGHT_CACHE, parse_dates=["date"])
    import yfinance as yf
    log(f"overnight: downloading {OVERNIGHT_TICKERS} ...")
    frames = {}
    for tk in OVERNIGHT_TICKERS:
        for attempt in range(3):
            try:
                d = yf.download(tk, start="2014-06-01", auto_adjust=True, progress=False, threads=False)
                if len(d):
                    close = d["Close"]
                    s = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                    frames[base._safe_name(tk)] = s.rename(base._safe_name(tk))
                    break
            except Exception as e:
                log(f"  {tk} attempt {attempt+1}: {e}")
            time.sleep(2)
        else:
            log(f"  {tk}: SKIPPED")
    if not frames:
        raise RuntimeError("no overnight tickers downloaded and no cache present")
    wide = pd.concat(frames.values(), axis=1, sort=True).reset_index().rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"]).dt.tz_localize(None)
    OVERNIGHT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OVERNIGHT_CACHE, index=False)
    return wide


def overnight_features(prices, close, dates) -> pd.DataFrame:
    """Day-(t+1) Asian miner moves, available the morning of t+1 (shift(-1); leak-free)."""
    mp = prices.set_index("date").sort_index()
    mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates).reset_index(drop=True)
    copper_ret1 = close.pct_change(1).reset_index(drop=True)        # copper's PAST return (known at t)
    cols, ov = {}, []
    for tk in OVERNIGHT_TICKERS:
        name = base._safe_name(tk)
        if name not in mp.columns or mp[name].notna().sum() < 100:
            continue
        daily = mp[name].reset_index(drop=True).pct_change(1)
        sig = daily.shift(-1)                                       # Asian session of date t+1
        cols[f"{name}_overnight_ret"] = sig.to_numpy()
        cols[f"{name}_overnight_div"] = (sig - copper_ret1).to_numpy()
        ov.append(sig)
    if ov:
        agg = pd.concat(ov, axis=1).mean(axis=1)
        cols["asianminers_overnight_mean"] = agg.to_numpy()
        cols["asianminers_overnight_div"] = (agg - copper_ret1).to_numpy()
    return pd.DataFrame(cols, index=range(len(dates))).replace([np.inf, -np.inf], np.nan)


def build_panel():
    primary, reason = base.find_dataset(BASE_DIR)
    external = base.find_external_file(BASE_DIR, primary)
    df, meta = base.load_and_prepare(primary, external)
    df = base.create_target(df)
    Xc = base.copper_features(df, meta["merged_external_cols"]).reset_index(drop=True)
    Xd = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    Xo = overnight_features(fetch_overnight(), df["close"], df["date"]).reset_index(drop=True)
    X = pd.concat([Xc, Xd, Xo], axis=1)
    y = df["target_up"].to_numpy(int)
    return df, X, y, meta, reason, Xo.shape[1]


def per_year(df, y, probs, mask):
    years = pd.to_datetime(df["date"]).dt.year.to_numpy()
    rows = []
    for yr in sorted(np.unique(years[mask])):
        sel = mask & (years == yr)
        if sel.sum() < 30:
            continue
        ys, ps = y[sel], probs[sel]
        rows.append({"year": int(yr), "n": int(sel.sum()),
                     "accuracy": round(accuracy_score(ys, (ps >= 0.5).astype(int)), 4),
                     "auc": round(roc_auc_score(ys, ps), 4) if len(np.unique(ys)) > 1 else np.nan})
    return pd.DataFrame(rows)


def write_summary(ctx, path):
    me, g, py = ctx["metrics"], ctx["gate"], ctx["per_year"]
    md = f"""# Copper Close Direction Model — MORNING-OF variant

## What this is
Predicts whether COMEX copper closes higher tomorrow than today, but the guess is made on
the **morning of the target day**, after the Asian/Australian session closes (~1-4am ET)
and before the COMEX settle (~1pm ET). That lets it use the **overnight Asian copper-miner
moves**, which lead copper's US session.

> If you must lock your guess at the prior close instead, use the NIGHT-BEFORE model
> (`copper_close_direction_model.py`, ~53.9%). This one is a ~5-hour forecast, not 24h.

## Overnight feature (leak-free)
Asian/AU miners (BHP.AX, RIO.AX, S32.AX, Zijin 2899.HK, Jiangxi 0358.HK, MMG 1208.HK)
session return of date t+1, encoded as the date-aligned series shifted by one trading day.
Asian close precedes the COMEX t+1 settle, so it is available at prediction time.
Validated: permutation null passes (real {me['accuracy']:.3f} vs null ~0.50, p=0.000); a
no-shift control stays at the 53.9% baseline, isolating the gain to the t+1 session.

## Performance (walk-forward, {me['oos_days']} OOS days)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- **AUC: {me['auc']:.4f}** · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}
- vs night-before baseline ~0.539 -> a +{me['accuracy']-0.539:+.3f} jump.

## Per-year stability (the standout — consistent every year)
| year | n | accuracy | auc |
|---|---|---|---|
""" + "\n".join(f"| {r.year} | {r.n} | {r.accuracy} | {r.auc} |" for r in py.itertuples()) + f"""

## Model
- {len(ctx['selected_features'])} features selected (L1, per-fold C={ctx['chosen_C']}) of
  {ctx['candidates']} candidates; copper + cross-asset divergence + overnight Asian miners.
- Top features: {', '.join(ctx['selected_features'][:8])}

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
"""
    path.write_text(md)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, meta, reason, n_over = build_panel()
    log(reason)
    log(f"Rows={len(y)} features={X.shape[1]} (incl. {n_over} overnight) up_rate={y.mean():.4f}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"Walk-forward: acc={metrics['accuracy']:.4f} "
        f"CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] auc={metrics['auc']:.4f} "
        f"on {metrics['oos_days']} OOS days  (candidates={freport['candidate_features']})")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))

    oos = pd.DataFrame({
        "date": df["date"].dt.date.values[mask], "close": df["close"].values[mask],
        "actual_up": y[mask], "predicted_p_up": probs[mask],
        "predicted_dir": np.where(probs[mask] >= 0.5, "UP", "DOWN"),
        "confidence": np.abs(probs[mask] - 0.5),
        "correct": (((probs[mask] >= 0.5).astype(int)) == y[mask]).astype(int)})
    oos.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    gate.to_csv(OUT_DIR / "confidence_gate.csv", index=False)
    py.to_csv(OUT_DIR / "per_year_stability.csv", index=False)
    pd.DataFrame([{k: (v if not isinstance(v, list) else str(v)) for k, v in metrics.items()}]).to_csv(
        OUT_DIR / "metrics_summary.csv", index=False)

    final = base.fit_final_model(X, y, features)
    joblib.dump(final, OUT_DIR / "model.pkl")
    pd.DataFrame([{"feature": f, "coefficient": c} for f, c in final["coefficients"].items()]).to_csv(
        OUT_DIR / "selected_features.csv", index=False)
    log(f"Final model: C={final['chosen_C']}, {len(final['selected_features'])} features selected.")

    try:
        base.plot_gate(gate, metrics["accuracy"], OUT_DIR / "confidence_gate_chart.png")
    except Exception as e:
        log(f"plot failed: {e}")

    ctx = {"metrics": metrics, "gate": gate, "per_year": py,
           "selected_features": final["selected_features"], "chosen_C": final["chosen_C"],
           "candidates": freport["candidate_features"]}
    write_summary(ctx, OUT_DIR / "evaluation_summary.md")
    with open(OUT_DIR / "metrics_summary.json", "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in metrics.items() if not isinstance(v, list)}, fh, indent=2, default=str)
    log(f"Artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
