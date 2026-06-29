# Aluminum Close-to-Close Direction Model

## Question
Will the aluminum benchmark close **higher tomorrow than it closed today**? Pure up/down,
every day. `target_up = 1 if close[t+1] > close[t] else 0`.

## Data
- Price dataset: `aluminum.csv` (close column `close`)
- Macro drivers merged: `external_drivers.csv` + 20 cross-asset tickers
- Rows: 2999 · Date range: 2014-06-02 → 2026-06-26

## Features (166 candidates -> 77 selected)
- Aluminum price block (returns, momentum, trend ratios, volatility, drawdown, candle, volume)
- **Cross-asset divergence block** (the edge): producer/China/base-metal return minus
  aluminum's own return at 1/5/20d, divergence z-scores and ratio-momentum.
- Non-stationary price LEVELS (ma_*, volume_ma_*) and exact duplicates dropped a priori.
- **L1 selection** keeps ~76 features per fold; the final model uses
  77. Top by |coef|: PICK_div_1d, wti_volatility_20d, brent_volatility_20d, KALU_ratio_mom20, KWEB_div_z20, ASHR_div_z20, DBB_div_5d, KALU_div_z20.

## Model
- **L1-penalized Logistic Regression** + isotonic calibration (logistic beat RF/GB/HistGB —
  the usable signal is linear). C chosen per-fold on validation (C=0.5 final).
- Expanding-window walk-forward, 1999 out-of-sample days.

## Performance (walk-forward, every day)
- **Accuracy: 0.5543** (95% CI [0.5151, 0.5908])
- AUC: 0.5972 · Balanced accuracy: 0.5470
- Up precision/recall: 0.495 / 0.485
- Down precision/recall: 0.600 / 0.609
- Up base rate: 0.4412 (always-guess-majority = 0.5588)
- Confusion (tn,fp,fn,tp): [680, 437, 454, 428]

## Deployable confidence gate (the dead-band that works)
Trust the call only when the model is sure. Accuracy rises with conviction:

| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5543 | [0.515, 0.591] |
| 0.03 | 76.8% | 0.5768 | [0.531, 0.622] |
| 0.05 | 65.1% | 0.5891 | [0.536, 0.639] |
| 0.08 | 51.2% | 0.6074 | [0.544, 0.667] |

Best confident slice: **0.6074 accuracy on 51.2% of days**
(min |p-0.5| > 0.08). Volatility / move-size gates did NOT deploy
(trailing vol predicts move size only weakly), so the gate is on model confidence.

## How to read it
- Beats a coin flip with confidence (AUC CI clears 0.50); ~tied with always-up on raw accuracy.
- Every-day ceiling ≈ 55.4%; the confidence gate buys higher accuracy on a
  selective subset, not a higher every-day number.

## Statistical Notes
- Confidence intervals use **circular block bootstrap** (block_size=20, 3000 iters) to
  account for autocorrelation in rolling-window features. CIs are wider than a naive
  i.i.d. bootstrap would produce — this is the honest estimate.
- Val window is split: first 2/3 for C-selection, last 1/3 for calibration. This prevents
  the isotonic calibration from seeing the same labels used to pick C.
- Feature stability (fraction of folds where each feature had non-zero L1 coefficient) is
  saved in `feature_stability.csv`. Features selected in <20% of folds should be treated
  as unreliable signal.

## Next: more data features (this model is built to extend)
Add genuinely *leading* inputs — handled by the MORNING model: overnight Asian/AU-session
aluminum-producer closes (BHP.AX, RIO.AX, S32.AX, Hongqiao 1378.HK, Chalco 2600.HK, Rusal
0486.HK), LME aluminum overnight settlement, LME–US basis / term structure, and aluminum's
big idiosyncratic driver — energy/power costs (smelting). See DATA_NEEDED.md.

---
*Research model — not financial advice. Validate before any real use.*
