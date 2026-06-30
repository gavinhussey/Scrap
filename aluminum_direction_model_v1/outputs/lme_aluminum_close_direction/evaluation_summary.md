# LME Aluminum Close Direction Model — NIGHT-BEFORE

## Question
Will the **LME 3-month aluminum** settlement close higher tomorrow than today?
`target_up = 1 if lme_al_close[t+1] > lme_al_close[t] else 0`. Honest rebuild replacing the
stale COMEX ALI=F target (see ALUMINUM_STEEL_MODEL_REVIEW.md).

## Data
- Target: `lme_aluminum.csv` (LME 3-month, USD/tonne) — actually trades (~1.4% flat days).
- Drivers: `external_drivers.csv` (COMEX echo column dropped) + cross-asset yfinance cousins.
- Rows: 3051 · Date range: 2014-06-02 → 2026-06-25

## Timing (leak-free)
Decision at the evening of day t (after US close); target is the day-(t+1) LME settle
(~7:15am ET). All day-t features precede it.

## Performance (walk-forward, 2051 OOS days)
- **Accuracy: 0.5110** (95% CI [0.4900, 0.5334])
- AUC: 0.5138 · Balanced accuracy: 0.5114
- Up base rate: 0.4929 (always-guess-majority = 0.5071)
- Up precision/recall: 0.504 / 0.541
- Down precision/recall: 0.519 / 0.482
- Confusion (tn,fp,fn,tp): [501, 539, 464, 547]

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5110 | [0.490, 0.533] |
| 0.03 | 76.9% | 0.5206 | [0.496, 0.547] |
| 0.05 | 63.4% | 0.5077 | [0.481, 0.537] |
| 0.08 | 40.4% | 0.5133 | [0.478, 0.548] |

Best confident slice: **0.5206 on 76.9% of days**
(min |p-0.5| > 0.03).

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 161 | 0.4658 | 0.4437 |
| 2019 | 253 | 0.5257 | 0.5576 |
| 2020 | 254 | 0.5039 | 0.5219 |
| 2021 | 253 | 0.5375 | 0.4941 |
| 2022 | 251 | 0.5259 | 0.5149 |
| 2023 | 251 | 0.498 | 0.4784 |
| 2024 | 254 | 0.5157 | 0.4975 |
| 2025 | 253 | 0.4862 | 0.5142 |
| 2026 | 121 | 0.5372 | 0.531 |

## Model
- L1 logistic + isotonic calibration, expanding walk-forward (machinery from
  aluminum_close_direction_model). 8 features selected (C=0.05)
  of 171 candidates.
- Top features: brent_return_1d, DBB_div_5d, VIX_ret_1d, al_gold_ratio_mom20, al_gold_ratio_z63, VIX_vol_20d, CNY_ret_1d, curve_2s10s_return_5d

---
*Research model — not financial advice.*
