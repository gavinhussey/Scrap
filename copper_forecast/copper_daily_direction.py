"""
copper_daily_direction.py — predict whether TOMORROW is an up or down day for
COMEX copper (HG=F), framed as binary classification.

Approach (per the chosen strategy):
  * Features known by the close of day t: copper's own technicals + the SAME-DAY
    moves of cross-asset correlates (all close before tomorrow's copper move, so
    no look-ahead): DXY, WTI, gold, silver, S&P 500, COPX (copper miners),
    aluminum, VIX, 10y yield.
  * Target: sign(close[t+1] - close[t]).  1 = up day, 0 = down/flat.
  * Model: gradient-boosted trees (HistGradientBoostingClassifier) — better
    suited to ~2500 rows of tabular features than an LSTM.
  * Evaluation: expanding-window walk-forward, multi-seed ensemble, honest
    directional accuracy (Dstat) with a pooled binomial 95% CI, against
    always-up (base rate) and momentum baselines.

Daily price is near-memoryless, so the bet is entirely that same-day cross-asset
signals lead next-day copper. Goal: pooled Dstat > 0.55 with a CI clear of the
base rate.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.simplefilter("ignore")

START = "2015-09-14"
END = "2025-08-30"
TICKERS = {
    "cu": "HG=F",      # copper (target + own technicals)
    "dxy": "DX-Y.NYB", # dollar index
    "wti": "CL=F",     # crude oil
    "gold": "GC=F",
    "silver": "SI=F",
    "spx": "^GSPC",     # S&P 500
    "copx": "COPX",     # copper miners ETF
    "alum": "ALI=F",   # aluminum
    "vix": "^VIX",
    "tnx": "^TNX",      # 10y treasury yield
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--init-train-frac", type=float, default=0.5)
    return p.parse_args()


def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def download():
    frames = {}
    for name, tkr in TICKERS.items():
        df = yf.download(tkr, start=START, end=END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[name] = df
    return frames


def build_features(frames):
    cu = frames["cu"]
    o, h, l, c = cu["Open"], cu["High"], cu["Low"], cu["Close"]

    f = pd.DataFrame(index=cu.index)
    # --- copper own technicals (all use data through day t) ---
    f["cu_ret1"] = c.pct_change()
    f["cu_ret3"] = c.pct_change(3)
    f["cu_ret5"] = c.pct_change(5)
    f["cu_ret10"] = c.pct_change(10)
    f["cu_ret1_lag1"] = f["cu_ret1"].shift(1)
    f["cu_ret1_lag2"] = f["cu_ret1"].shift(2)
    f["cu_gap"] = (o - c.shift()) / c.shift()
    f["cu_range"] = (h - l) / c
    f["cu_clpos"] = (c - l) / (h - l).replace(0, np.nan)
    f["cu_vs_ma20"] = c / c.rolling(20).mean() - 1
    f["cu_vol5"] = f["cu_ret1"].rolling(5).std()
    f["cu_vol10"] = f["cu_ret1"].rolling(10).std()
    f["cu_vol20"] = f["cu_ret1"].rolling(20).std()
    f["cu_rsi14"] = rsi(c)

    # --- same-day cross-asset signals (known by close of day t) ---
    for name in ["dxy", "wti", "gold", "silver", "spx", "copx", "alum"]:
        s = frames[name]["Close"].reindex(cu.index).ffill()
        f[f"{name}_ret1"] = s.pct_change()
        f[f"{name}_ret3"] = s.pct_change(3)
    vix = frames["vix"]["Close"].reindex(cu.index).ffill()
    f["vix_level"] = vix
    f["vix_chg1"] = vix.pct_change()
    tnx = frames["tnx"]["Close"].reindex(cu.index).ffill()
    f["tnx_chg1"] = tnx.diff()
    # cross ratios that often lead industrial metals
    f["cu_gold_ratio_chg"] = (c / frames["gold"]["Close"].reindex(cu.index).ffill()).pct_change()

    # --- target: tomorrow up/down ---
    y = (c.shift(-1) > c).astype(int)

    data = f.copy()
    data["__y__"] = y
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    feat_cols = [col for col in data.columns if col != "__y__"]
    return data[feat_cols], data["__y__"], c.reindex(data.index)


# Coverage levels to report: fraction of days we are most confident about.
COVERAGES = [1.00, 0.75, 0.50, 0.30, 0.20, 0.10]
CALIB_FRAC = 0.20   # tail of each training window reserved to set thresholds


def fit_ensemble(Xtr, ytr, seeds):
    """Average predict-proba over a multi-seed GBM ensemble; return a predict fn."""
    models = []
    for s in range(seeds):
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, early_stopping=True, random_state=s)
        clf.fit(Xtr, ytr)
        models.append(clf)
    return lambda Z: np.mean([m.predict_proba(Z)[:, 1] for m in models], axis=0)


def walk_forward(X, y, seeds, folds, init_frac):
    Xv, yv = X.to_numpy(float), y.to_numpy(int)
    T = len(yv)
    first = int(T * init_frac)
    block = (T - first) // folds

    # Per coverage level: accumulate correctness over all SELECTED test days.
    pooled = {q: {"correct": [], "n_total": 0} for q in COVERAGES}

    for k in range(folds):
        lo = first + k * block
        hi = first + (k + 1) * block if k < folds - 1 else T
        if hi <= lo:
            continue

        # Reserve a calibration tail of the TRAIN window to set thresholds OOS.
        cut = int(lo * (1 - CALIB_FRAC))
        predict = fit_ensemble(Xv[:cut], yv[:cut], seeds)
        conv_calib = np.abs(predict(Xv[cut:lo]) - 0.5)     # conviction, OOS in train

        p_te = predict(Xv[lo:hi])
        conv_te = np.abs(p_te - 0.5)
        correct_te = ((p_te >= 0.5).astype(int) == yv[lo:hi])

        for q in COVERAGES:
            thr = np.quantile(conv_calib, 1 - q)           # keep top-q most confident
            mask = conv_te >= thr
            pooled[q]["correct"].append(correct_te[mask])
            pooled[q]["n_total"] += len(correct_te)

        base = max(yv[lo:hi].mean(), 1 - yv[lo:hi].mean())
        print(f"  fold {k+1}: n={len(correct_te):4d}  Dstat(all)={correct_te.mean():.4f}  "
              f"base-rate={base:.4f}")

    return pooled


def main():
    args = parse_args()
    print("Downloading daily data (copper + cross-asset)...")
    frames = download()
    X, y, _ = build_features(frames)
    print(f"Samples: {len(y)} days | features: {X.shape[1]} | up-share: {y.mean():.3f}")

    pooled = walk_forward(X, y, args.seeds, args.folds, args.init_train_frac)

    print("\n========= DSTAT vs COVERAGE (conviction filter) =========")
    print("conviction threshold set on a held-out calibration tail of train, applied OOS\n")
    print(f"{'target cov':>10}  {'acted days':>10}  {'realized cov':>12}  "
          f"{'Dstat':>7}  {'95% CI':>18}")
    for q in COVERAGES:
        c = np.concatenate(pooled[q]["correct"])
        n = len(c); ntot = pooled[q]["n_total"]
        phat = c.mean()
        se = np.sqrt(phat * (1 - phat) / n) if n else 0.0
        print(f"{q:>10.2f}  {n:>10d}  {n/ntot:>12.3f}  {phat:>7.4f}  "
              f"[{phat-1.96*se:.4f}, {phat+1.96*se:.4f}]")

    print("\nGoal: a coverage level where Dstat > 0.55 with CI lower bound above ~0.51 base rate.")


if __name__ == "__main__":
    main()
