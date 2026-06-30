"""LME Aluminum Direction Model — NIGHT-BEFORE (honest rebuild).

WHY THIS EXISTS
  The original aluminum model used COMEX ALI=F as the target. ALI=F is a stale,
  barely-traded settlement series (84.6% of days have high==low, 0 median volume), so the
  morning model's headline 92.8% was a LEAKAGE ARTIFACT: next_gap_return = open[t+1]/close[t]
  mechanically equalled the close move because the contract doesn't trade open->close.

  This rebuild swaps the target to the **LME 3-month aluminum settlement**
  (data/external/lme_aluminum.csv), the global benchmark that actually trades — only 1.4% of
  days are flat, up-rate ~0.49. The gap-leak disappears (there is no open column and the
  series moves), and the divergence/walk-forward machinery from aluminum_close_direction_model
  is reused unchanged.

TARGET:  target_up = 1 if lme_al_close[t+1] > lme_al_close[t] else 0.

TIMING / LEAKAGE
  The natural decision point is the EVENING of day t (after the US close, ~4pm ET), forecasting
  the next LME Ring settlement on day t+1 (~7:15am ET / ~12:15 London). Everything dated day t
  -- LME close t, US producer cousins (4pm ET close t), US macro (DXY/VIX/rates close t) -- is
  known by then and precedes the t+1 LME settle, so all day-t features are leak-free.
  (The MORNING model adds the Asian/AU overnight session of day t+1, which also precedes the
  LME settle.)

Run:  python lme_aluminum_direction_model.py  ->  outputs/lme_aluminum_close_direction/
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import aluminum_close_direction_model as base

BASE_DIR = Path(__file__).resolve().parent
LME_CSV = BASE_DIR / "data" / "external" / "lme_aluminum.csv"
EXT_CSV = BASE_DIR / "data" / "external" / "external_drivers.csv"
OUT_DIR = BASE_DIR / "outputs" / "lme_aluminum_close_direction"


def log(m): print(f"[lme_aluminum] {m}", flush=True)


def load_lme() -> tuple[pd.DataFrame, list[str]]:
    """LME aluminum close as the primary series, with US macro drivers merged (past-only).

    The stale COMEX 'aluminum' column in external_drivers is dropped — it is the same ALI=F
    echo we are moving away from and would only add a noisy, mostly-zero cousin.
    """
    raw = pd.read_csv(LME_CSV)
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw = raw.rename(columns={"lme_al_close": "close"})
    df = (raw[["date", "close"]]
          .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
          .dropna(subset=["date"]).sort_values("date")
          .drop_duplicates("date", keep="last").reset_index(drop=True))
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    ext_cols: list[str] = []
    if EXT_CSV.exists():
        ext = pd.read_csv(EXT_CSV)
        ext.columns = [c.lower().strip() for c in ext.columns]
        ext["date"] = pd.to_datetime(ext["date"], errors="coerce")
        ext = ext.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        ext = ext.drop(columns=[c for c in ("aluminum", "aluminium") if c in ext.columns])
        ext_cols = [c for c in ext.columns if c != "date"]
        df = df.merge(ext, on="date", how="left")
        df[ext_cols] = df[ext_cols].ffill()      # past-only fill, no leakage
    return df, ext_cols


def build_panel():
    df, ext_cols = load_lme()
    df = base.create_target(df)
    Xc = base.metal_features(df, ext_cols).reset_index(drop=True)
    Xd = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    X = pd.concat([Xc, Xd], axis=1)
    y = df["target_up"].to_numpy(int)
    return df, X, y, ext_cols


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
    best_gate = g.iloc[g["accuracy"].idxmax()]
    md = f"""# LME Aluminum Close Direction Model — NIGHT-BEFORE

## Question
Will the **LME 3-month aluminum** settlement close higher tomorrow than today?
`target_up = 1 if lme_al_close[t+1] > lme_al_close[t] else 0`. Honest rebuild replacing the
stale COMEX ALI=F target (see ALUMINUM_STEEL_MODEL_REVIEW.md).

## Data
- Target: `lme_aluminum.csv` (LME 3-month, USD/tonne) — actually trades (~1.4% flat days).
- Drivers: `external_drivers.csv` (COMEX echo column dropped) + cross-asset yfinance cousins.
- Rows: {ctx['n_rows']} · Date range: {ctx['date_min']} → {ctx['date_max']}

## Timing (leak-free)
Decision at the evening of day t (after US close); target is the day-(t+1) LME settle
(~7:15am ET). All day-t features precede it.

## Performance (walk-forward, {me['oos_days']} OOS days)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- AUC: {me['auc']:.4f} · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up base rate: {me['up_base_rate']:.4f} (always-guess-majority = {me['majority_baseline_acc']:.4f})
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}
- Confusion (tn,fp,fn,tp): {me['confusion']}

## Deployable confidence gate
| min \\|p-0.5\\| | coverage | accuracy | 95% CI |
|---|---|---|---|
""" + "\n".join(
        f"| {r.min_confidence} | {r.coverage:.1%} | {r.accuracy:.4f} | [{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}] |"
        for r in g.itertuples()) + f"""

Best confident slice: **{best_gate['accuracy']:.4f} on {best_gate['coverage']:.1%} of days**
(min |p-0.5| > {best_gate['min_confidence']}).

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
""" + "\n".join(f"| {r.year} | {r.n} | {r.accuracy} | {r.auc} |" for r in py.itertuples()) + f"""

## Model
- L1 logistic + isotonic calibration, expanding walk-forward (machinery from
  aluminum_close_direction_model). {len(ctx['selected_features'])} features selected (C={ctx['chosen_C']})
  of {ctx['candidates']} candidates.
- Top features: {', '.join(ctx['selected_features'][:8])}

---
*Research model — not financial advice.*
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, ext_cols = build_panel()
    log(f"Rows={len(y)} features={X.shape[1]} up_rate={y.mean():.4f} "
        f"{df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"acc={metrics['accuracy']:.4f} CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f} baseline={metrics['majority_baseline_acc']:.4f} OOS={metrics['oos_days']}")
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
    try:
        base.plot_gate(gate, metrics["accuracy"], OUT_DIR / "confidence_gate_chart.png")
    except Exception as e:
        log(f"plot failed: {e}")

    ctx = {"metrics": metrics, "gate": gate, "per_year": py,
           "selected_features": final["selected_features"], "chosen_C": final["chosen_C"],
           "candidates": freport["candidate_features"], "n_rows": len(df),
           "date_min": df["date"].iloc[0].date(), "date_max": df["date"].iloc[-1].date()}
    write_summary(ctx, OUT_DIR / "evaluation_summary.md")
    with open(OUT_DIR / "metrics_summary.json", "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in metrics.items() if not isinstance(v, list)}, fh, indent=2, default=str)
    log(f"Artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
