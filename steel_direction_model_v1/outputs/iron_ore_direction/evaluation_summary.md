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
- **Accuracy: 0.5280** (95% CI [0.4971, 0.5595])
- AUC: 0.5463 · Balanced accuracy: 0.5256
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.539 / 0.599
- Down precision/recall: 0.513 / 0.452

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5280 | [0.497, 0.559] |
| 0.03 | 87.6% | 0.5342 | [0.501, 0.568] |
| 0.05 | 75.5% | 0.5358 | [0.498, 0.573] |
| 0.08 | 60.7% | 0.5474 | [0.508, 0.587] |

Best confident slice: **0.5474 on 60.7% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.5621 | 0.598 |
| 2019 | 252 | 0.5198 | 0.4701 |
| 2020 | 253 | 0.5257 | 0.5606 |
| 2021 | 252 | 0.5476 | 0.577 |
| 2022 | 251 | 0.5618 | 0.5908 |
| 2023 | 250 | 0.512 | 0.5149 |
| 2024 | 252 | 0.504 | 0.5133 |
| 2025 | 251 | 0.5657 | 0.5714 |
| 2026 | 120 | 0.4 | 0.3913 |

## Model
- 28 features selected (L1, C=0.1) of 155
  candidates. Top: VALE_div_5d, sgx_io_mom5, CNY_vol_20d, DBB_div_1d, TX_div_20d, china_ferrous_div, VALE_div_1d, volatility_60d

---
*Research model — not financial advice.*
