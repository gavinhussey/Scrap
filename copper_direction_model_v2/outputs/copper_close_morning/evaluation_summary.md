# Copper Close Direction Model — MORNING-OF variant

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
Validated: permutation null passes (real 0.867 vs null ~0.50, p=0.000); a
no-shift control (same-day Asian return) scores 0.8508
vs real 0.8673, isolating the gain to the t+1 session shift.

## Performance (walk-forward, 1884 OOS days)
- **Accuracy: 0.8673** (95% CI [0.8503, 0.8838])
- **AUC: 0.9361** · Balanced accuracy: 0.8665
- Up precision/recall: 0.854 / 0.895
- Down precision/recall: 0.883 / 0.838
- vs night-before baseline ~0.539 -> a ++0.328 jump.

## Per-year stability (the standout — consistent every year)
| year | n | accuracy | auc |
|---|---|---|---|
| 2019 | 252 | 0.8492 | 0.932 |
| 2020 | 253 | 0.8933 | 0.9461 |
| 2021 | 252 | 0.8611 | 0.9489 |
| 2022 | 251 | 0.8845 | 0.9328 |
| 2023 | 251 | 0.9163 | 0.9683 |
| 2024 | 252 | 0.8333 | 0.9235 |
| 2025 | 252 | 0.8214 | 0.8668 |
| 2026 | 118 | 0.8898 | 0.9685 |

## Model
- 21 features selected (L1, per-fold C=0.1) of
  193 candidates; copper + cross-asset divergence + overnight Asian miners
  + COMEX open gap (LME proxy) + direct LME overnight.
- Top features: lme_overnight_ret, lme_overnight_z20, next_gap_z20, next_gap_return, log_return_10d, lme_comex_basis_chg, drawdown_20d, CNY_ret_1d

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
