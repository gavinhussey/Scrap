# Iron Ore (SGX 62% Fe, TIO=F) Direction Model — NIGHT-BEFORE

## Question
Will SGX iron ore close higher tomorrow than today? The liquid ferrous-benchmark rebuild
(steel's LME analog) replacing the thin HRC=F target. See ALUMINUM_STEEL_MODEL_REVIEW.md.

## Data
- Target: TIO=F (SGX 62% Fe iron ore, yfinance) — liquid, ~2.1% flat days.
- Drivers: cross-asset ferrous cousins (steelmakers, iron-ore miners, ETFs) + macro.
- Rows: 3034 · Date range: 2014-06-02 → 2026-06-25

## Timing (leak-free)
Night-before: decision at day-t US close; target is the day-(t+1) SGX settle. All day-t
features precede it.

## Performance (walk-forward, 2034 OOS days)
- **Accuracy: 0.5187** (95% CI [0.4897, 0.5482])
- AUC: 0.5292 · Balanced accuracy: 0.5155
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.530 / 0.610
- Down precision/recall: 0.502 / 0.421

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5187 | [0.490, 0.548] |
| 0.03 | 81.4% | 0.5217 | [0.486, 0.558] |
| 0.05 | 74.1% | 0.5272 | [0.491, 0.565] |
| 0.08 | 63.6% | 0.5290 | [0.486, 0.571] |

Best confident slice: **0.5290 on 63.6% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.5229 | 0.5158 |
| 2019 | 252 | 0.5119 | 0.4727 |
| 2020 | 253 | 0.5455 | 0.5512 |
| 2021 | 252 | 0.5397 | 0.5689 |
| 2022 | 251 | 0.5418 | 0.5722 |
| 2023 | 250 | 0.512 | 0.5105 |
| 2024 | 252 | 0.496 | 0.5219 |
| 2025 | 251 | 0.5299 | 0.5518 |
| 2026 | 120 | 0.4167 | 0.3832 |

## Model
- 14 features selected (L1, C=0.05) of 139
  candidates. Top: VALE_div_5d, CNY_vol_20d, simple_return_5d, BHP_div_1d, VALE_div_1d, SID_div_20d, VIX_vol_20d, momentum_5d

---
*Research model — not financial advice.*
