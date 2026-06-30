# LME Aluminum Close Direction Model — MORNING-OF

## What this is
Predicts the next-day **LME 3-month aluminum** settlement, guessed on the morning of day t+1
after the Asian/AU session closes and before the ~7:15am ET LME Ring settle. Honest rebuild of
the corrupted COMEX-ALI morning model: the LME-on-LME self-leak and the open-gap tautology are
both removed (see ALUMINUM_STEEL_MODEL_REVIEW.md).

## Overnight features (leak-free, all close before the LME settle)
ASX/HK aluminum producers (BHP.AX, RIO.AX, S32.AX, 1378.HK, 2600.HK, 0486.HK) + SHFE copper,
day-(t+1) session, shift(-1)-aligned.

## No-shift control (the honesty check)
Real model acc 0.5549 / AUC 0.5701.
No-shift control acc 0.5149 / AUC 0.5129.
**Overnight shift lift: +0.0400 acc / +0.0572 AUC.**
A positive, meaningful lift is the only thing that makes the morning timing worthwhile.

## Performance (walk-forward, 2051 OOS days)
- **Accuracy: 0.5549** (95% CI [0.5316, 0.5802])
- AUC: 0.5701 · Balanced accuracy: 0.5549
- Up base rate: 0.4929 (always-guess-majority = 0.5071)
- Up precision/recall: 0.547 / 0.560
- Down precision/recall: 0.562 / 0.550

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 161 | 0.5714 | 0.5451 |
| 2019 | 253 | 0.5217 | 0.5419 |
| 2020 | 254 | 0.5079 | 0.5202 |
| 2021 | 253 | 0.5968 | 0.5888 |
| 2022 | 251 | 0.5339 | 0.5476 |
| 2023 | 251 | 0.506 | 0.5118 |
| 2024 | 254 | 0.6024 | 0.609 |
| 2025 | 253 | 0.5929 | 0.6228 |
| 2026 | 121 | 0.5785 | 0.5851 |

## Model
- 53 features selected (L1, C=0.25) of 176
  candidates; LME price block + cross-asset divergence + leak-free overnight Asian/SHFE leads.
- Top features: shfe_cu_overnight_ret, asianminers_5d_mom, 2600_HK_overnight_ret, RIO_div_1d, drawdown_60d, DBB_div_5d, S32_AX_overnight_div, XME_div_20d

---
*Research model — not financial advice. The morning-of timing is only valid if the overnight
shift lift above is positive.*
