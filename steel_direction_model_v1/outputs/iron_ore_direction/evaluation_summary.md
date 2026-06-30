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
- **Accuracy: 0.5172** (95% CI [0.4872, 0.5477])
- AUC: 0.5451 · Balanced accuracy: 0.5153
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.530 / 0.574
- Down precision/recall: 0.501 / 0.457

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5172 | [0.487, 0.548] |
| 0.03 | 81.1% | 0.5339 | [0.499, 0.568] |
| 0.05 | 76.5% | 0.5434 | [0.505, 0.579] |
| 0.08 | 61.6% | 0.5483 | [0.504, 0.589] |

Best confident slice: **0.5483 on 61.6% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.549 | 0.6043 |
| 2019 | 252 | 0.5119 | 0.4596 |
| 2020 | 253 | 0.5257 | 0.5833 |
| 2021 | 252 | 0.5238 | 0.5706 |
| 2022 | 251 | 0.5737 | 0.6155 |
| 2023 | 250 | 0.524 | 0.5268 |
| 2024 | 252 | 0.4841 | 0.514 |
| 2025 | 251 | 0.5179 | 0.5601 |
| 2026 | 120 | 0.3917 | 0.3538 |

## Model
- 31 features selected (L1, C=0.1) of 157
  candidates. Top: VALE_div_5d, sgx_io_mom5, CNY_vol_20d, DBB_div_1d, TX_div_20d, VALE_div_1d, dce_io_div, volatility_60d

---
*Research model — not financial advice.*
