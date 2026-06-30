# Iron Ore (SGX TIO=F) Direction Model — MORNING-OF

## What this is
Predicts whether SGX 62% Fe iron ore closes higher tomorrow, guessed after the China ferrous
complex (SHFE rebar/HRC, DCE iron ore and coking coal) closes at ~3pm Beijing / 2am ET on
day t+1 — before the TIO=F final settlement at ~5-6am ET. See iron_ore_direction_model.py
for the night-before baseline (~51.9%).

## No-shift control (the honesty check)
Real model acc 0.5590 / AUC 0.5748.
No-shift control acc 0.5280 / AUC 0.5463.
**Overnight shift lift: +0.0310 acc / +0.0285 AUC.**
Only a **positive, meaningful lift** makes the morning timing worthwhile.
If lift ≤ 0, the China ferrous "overnight" data is contemporaneous with TIO=F — do not deploy.

## Performance (walk-forward, 2034 OOS days)
- **Accuracy: 0.5590** (95% CI [0.5306, 0.5885])
- AUC: 0.5748 · Balanced accuracy: 0.5573
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.568 / 0.609
- Down precision/recall: 0.547 / 0.506

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5590 | [0.531, 0.588] |
| 0.03 | 84.9% | 0.5597 | [0.527, 0.592] |
| 0.05 | 80.2% | 0.5567 | [0.522, 0.591] |
| 0.08 | 64.7% | 0.5726 | [0.535, 0.610] |

Best confident slice: **0.5726 on 64.7% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.5686 | 0.6038 |
| 2019 | 252 | 0.5079 | 0.4812 |
| 2020 | 253 | 0.5692 | 0.6154 |
| 2021 | 252 | 0.5476 | 0.5827 |
| 2022 | 251 | 0.5777 | 0.6133 |
| 2023 | 250 | 0.54 | 0.5442 |
| 2024 | 252 | 0.5635 | 0.5707 |
| 2025 | 251 | 0.6135 | 0.6255 |
| 2026 | 120 | 0.5333 | 0.4636 |

## Model
- 18 features selected (L1, C=0.05) of 150
  candidates; iron ore price block + cross-asset divergence + China ferrous overnight leads.
- Top features: VALE_div_5d, china_ferrous_mean, CNY_vol_20d, simple_return_5d, BHP_div_1d, SID_div_20d, dce_io_mom5, VIX_vol_20d

## China ferrous features (Bloomberg required)
Populated by bloombergSteel.py. If overnight shift lift ≤ 0, the features are not leading
TIO=F and this model should be retracted in favour of iron_ore_direction_model.py.

---
*Research model — not financial advice. Valid only if overnight shift lift is positive.*
