# Risk Model Overview

## Purpose
The model estimates market-price risk on current physical scrap inventory. It is designed for internal operating decisions: inventory exposure, cash reserves, liquidation risk, and stress losses.

## Current Scope
- Physical inventory source: latest GreenSpark combined inventory snapshot in `daily_inputs/`.
- Valuation: grade-level realized sale price where available; otherwise unpriced rows are held at book for MTM.
- Risk exposure: net realizable value. Unpriced rows are included through a conservative proxy, not treated as zero risk.
- Risk methods: parametric VaR, EWMA VaR, historical VaR, Monte Carlo, named stress scenarios, and basis compression stress.
- Metals: copper, aluminium, steel, stainless, brass, lead, and zinc where mapped in `src/config.py`.

## Main Outputs
- Console report from `python run_risk_model.py`
- HTML report from `python generate_risk_report_html.py`
- Charts in `output/charts/`
- Valuation files in `output/`

## Important Limitations
- Several metals use proxy price series when no usable free ticker exists.
- Synthetic price data may be used if live download and cache are unavailable; this degrades model quality.
- Haircuts, stress shocks, and basis volatility overlays are assumptions, not audited estimates.
- The model measures market and realization risk, not credit, fraud, operational, compliance, environmental, or safety risk.

## Related Notes
- [[Inventory & Grades]]
- [[Price Data]]
- [[VaR and CVaR]]
- [[Monte Carlo]]
- [[Stress Scenarios]]
- [[How to Use This for Business Decisions]]
