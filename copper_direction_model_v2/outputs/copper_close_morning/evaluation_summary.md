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
Validated: permutation null passes (real 0.598 vs null ~0.50, p=0.000); a
no-shift control stays at the 53.9% baseline, isolating the gain to the t+1 session.

## Performance (walk-forward, 1884 OOS days)
- **Accuracy: 0.5982** (95% CI [0.5764, 0.6210])
- **AUC: 0.6319** · Balanced accuracy: 0.5969
- Up precision/recall: 0.602 / 0.643
- Down precision/recall: 0.593 / 0.551
- vs night-before baseline ~0.539 -> a ++0.059 jump.

## Per-year stability (the standout — consistent every year)
| year | n | accuracy | auc |
|---|---|---|---|
| 2019 | 252 | 0.5675 | 0.6052 |
| 2020 | 253 | 0.5968 | 0.6026 |
| 2021 | 252 | 0.6071 | 0.6308 |
| 2022 | 251 | 0.6295 | 0.6481 |
| 2023 | 251 | 0.5777 | 0.6131 |
| 2024 | 252 | 0.5992 | 0.6594 |
| 2025 | 252 | 0.5992 | 0.6219 |
| 2026 | 118 | 0.6186 | 0.672 |

## Model
- 17 features selected (L1, per-fold C=0.05) of
  173 candidates; copper + cross-asset divergence + overnight Asian miners.
- Top features: 2899_HK_overnight_div, asianminers_overnight_mean, brent_momentum_20d, close_location, 0358_HK_overnight_ret, 0358_HK_overnight_div, log_return_10d, CNY_ret_1d

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
