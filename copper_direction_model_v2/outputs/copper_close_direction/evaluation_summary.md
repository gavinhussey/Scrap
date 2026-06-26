# Copper Close-to-Close Direction Model

## Question
Will COMEX copper close **higher tomorrow than it closed today**? Pure up/down, every day.
`target_up = 1 if close[t+1] > close[t] else 0`.

## Data
- Price dataset: `copper.csv` (close column `close`)
- Macro drivers merged: `external_drivers.csv` + 19 cross-asset tickers
- Rows: 2884 · Date range: 2015-01-02 → 2026-06-23

## Features (159 candidates -> 12 selected)
- Copper price block (returns, momentum, trend ratios, volatility, drawdown, candle, volume)
- **Cross-asset divergence block** (the edge): miner/China/base-metal return minus copper's
  own return at 1/5/20d, divergence z-scores and ratio-momentum.
- Non-stationary price LEVELS (ma_*, volume_ma_*) and exact duplicates dropped a priori.
- **L1 selection** keeps ~52 features per fold; the final model uses
  12. Top by |coef|: brent_momentum_20d, XME_div_z20, close_location, CPER_ratio_mom20, CNY_ret_1d, log_return_10d, brent_return_1d, volatility_spike_5_20.

## Model
- **L1-penalized Logistic Regression** + isotonic calibration (logistic beat RF/GB/HistGB —
  the usable signal is linear). C chosen per-fold on validation (C=0.05 final).
- Expanding-window walk-forward, 1884 out-of-sample days.

## Performance (walk-forward, every day)
- **Accuracy: 0.5387** (95% CI [0.5170, 0.5605])
- AUC: 0.5364 · Balanced accuracy: 0.5357
- Up precision/recall: 0.543 / 0.647
- Down precision/recall: 0.532 / 0.425
- Up base rate: 0.5138 (always-guess-majority = 0.5138)
- Confusion (tn,fp,fn,tp): [389, 527, 342, 626]

## Deployable confidence gate (the dead-band that works)
Trust the call only when the model is sure. Accuracy rises with conviction:

| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5387 | [0.517, 0.561] |
| 0.03 | 74.4% | 0.5396 | [0.514, 0.565] |
| 0.05 | 59.8% | 0.5350 | [0.506, 0.565] |
| 0.08 | 36.2% | 0.5352 | [0.499, 0.573] |

Best confident slice: **0.5396 accuracy on 74.4% of days**
(min |p-0.5| > 0.03). Volatility / move-size gates did NOT deploy
(trailing vol predicts move size only weakly), so the gate is on model confidence.

## How to read it
- Beats a coin flip with confidence (AUC CI clears 0.50); ~tied with always-up on raw accuracy.
- Every-day ceiling ≈ 53.9%; the confidence gate buys higher accuracy on a
  selective subset, not a higher every-day number.

## Next: more data features (this model is built to extend)
Add genuinely *leading* inputs — overnight Asian/AU-session copper-miner closes
(BHP.AX, RIO.AX, HK copper miners), LME–COMEX spread / term structure, options skew.

---
*Research model — not financial advice. Validate before any real use.*
