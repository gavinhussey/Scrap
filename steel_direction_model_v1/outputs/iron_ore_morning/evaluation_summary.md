# Iron Ore (SGX TIO=F) Direction Model — MORNING-OF

## What this is
Predicts whether SGX 62% Fe iron ore closes higher tomorrow, guessed after the China ferrous
complex (SHFE rebar/HRC, DCE iron ore and coking coal) closes at ~3pm Beijing / 2am ET on
day t+1 — before the TIO=F final settlement at ~5-6am ET. See iron_ore_direction_model.py
for the night-before baseline (~51.9%).

## No-shift control (the honesty check)
Real model acc 0.5644 / AUC 0.5980.
No-shift control acc 0.5138 / AUC 0.5353.
**Overnight shift lift: +0.0506 acc / +0.0626 AUC.**
Only a **positive, meaningful lift** makes the morning timing worthwhile.
If lift ≤ 0, the China ferrous "overnight" data is contemporaneous with TIO=F — do not deploy.

## Performance (walk-forward, 2034 OOS days)
- **Accuracy: 0.5644** (95% CI [0.5334, 0.5931])
- AUC: 0.5980 · Balanced accuracy: 0.5628
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.574 / 0.610
- Down precision/recall: 0.553 / 0.516

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5644 | [0.533, 0.593] |
| 0.03 | 85.8% | 0.5771 | [0.544, 0.608] |
| 0.05 | 79.3% | 0.5865 | [0.551, 0.621] |
| 0.08 | 72.6% | 0.5863 | [0.547, 0.624] |

Best confident slice: **0.5865 on 79.3% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.6144 | 0.6186 |
| 2019 | 252 | 0.5159 | 0.4815 |
| 2020 | 253 | 0.6047 | 0.6401 |
| 2021 | 252 | 0.5556 | 0.5831 |
| 2022 | 251 | 0.6056 | 0.6424 |
| 2023 | 250 | 0.588 | 0.629 |
| 2024 | 252 | 0.5317 | 0.5786 |
| 2025 | 251 | 0.5737 | 0.6287 |
| 2026 | 120 | 0.45 | 0.506 |

## Model
- 22 features selected (L1, C=0.05) of 162
  candidates; iron ore price block + cross-asset divergence + China ferrous overnight leads.
- Top features: shfe_hrc_ret, VALE_div_5d, dce_io_ret, CNY_vol_20d, momentum_5d, SID_div_20d, VIX_vol_20d, DBB_div_1d

## China ferrous features (Bloomberg required)
Populated by bloombergSteel.py. If overnight shift lift ≤ 0, the features are not leading
TIO=F and this model should be retracted in favour of iron_ore_direction_model.py.

---
*Research model — not financial advice. Valid only if overnight shift lift is positive.*
