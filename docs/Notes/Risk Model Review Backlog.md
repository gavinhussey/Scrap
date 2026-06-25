# Risk Model Review Backlog

This note records the issues, missing pieces, weak assumptions, and recommended fixes from the model review. Use it as a reference checklist when improving the risk model.

## Executive Summary
- The model is a scrap-inventory market-risk model, not a generic risk score.
- It estimates mark-to-market value, net realizable value, VaR/CVaR, Monte Carlo loss distributions, and stress losses for physical inventory.
- Main risks to manage: stale/bad inputs, proxy price data, unpriced inventory, hard-coded assumptions, insufficient tests, and stale documentation.

## Already Improved
- CLI and HTML now use the same shared risk engine.
- HTML chart generation is now refreshed with current model data.
- Price-source metadata now shows whether prices are live, cached, proxied, or synthetic.
- Unpriced inventory is no longer treated as zero-risk.
- Assumptions are now documented in `ASSUMPTIONS` in `src/config.py`.
- Input validation now catches missing required columns and negative GreenSpark weights/costs.
- VaR and Monte Carlo now handle several zero/empty edge cases.
- Vault documentation has been updated to describe the current GreenSpark, multi-metal model.
- Synthetic fallback no longer overwrites real price cache; stale real cache is preferred before synthetic.
- GreenSpark validation now reports stale dates, duplicates, unmapped commodities, zero-cost rows, and outliers.
- HTML now shows data-quality KPIs, input provenance, decision guidance, and run-to-run comparison.
- CLI and HTML support a fail-on-synthetic production control.

## Major Issues To Keep Watching
- Proxy price series can materially distort risk for brass, stainless, lead, and zinc.
- Synthetic price fallback should be treated as degraded model quality and reviewed before business decisions.
- Haircuts, stress shocks, basis-volatility multiplier, and proxy choices still need empirical calibration.
- The model does not cover credit risk, fraud, theft, environmental risk, safety risk, regulatory risk, or operational disruption.
- GreenSpark inventory is treated as physical truth; bad exports can still drive bad model output.
- DTC matching is adjacent to the risk model but is not a market-risk control by itself.

## Missing Risk Categories
- Customer/vendor credit risk.
- Counterparty concentration.
- Yard operational risk.
- Theft/shrink beyond assumed haircuts.
- Environmental and compliance liability.
- Inventory quality/contamination risk.
- Contract pricing or fixed-price exposure.
- Hedge basis and hedge execution risk.
- Liquidity risk by grade and yard.
- Data freshness and data lineage risk.

## Input And Data Quality Backlog
- Validate snapshot date is current and not stale. Implemented for snapshot filename date.
- Detect duplicate material-code/location rows. Implemented for GreenSpark snapshot.
- Flag extreme cost per pound and extreme average cost changes. Implemented for row-level cost/lb; average-change detection still needs historical baseline.
- Flag very large day-over-day inventory changes by metal/grade.
- Validate commodity mappings and report unmapped categories. Implemented for GreenSpark snapshot.
- Validate inbound/outbound date ranges.
- Validate currency/unit consistency.
- Add row-level validation summaries to the HTML report. Implemented through Model Quality warnings.
- Add input file provenance: filename, row count, latest date, hash/checksum. Implemented for GreenSpark snapshot.

## Scoring And Formula Backlog
- There is no single normalized "risk score"; the model reports dollar risk metrics.
- If a score is needed later, define the decision it supports first.
- Prefer separate scores for market risk, data quality, concentration, and liquidity rather than one blended score.
- Avoid arbitrary additive weights unless they are calibrated or governance-approved.
- Consider a tiered status model:
  - `OK`
  - `Review`
  - `Degraded`
  - `Do Not Use`
- Base status on price source, stale data, unpriced inventory share, validation errors, and proxy exposure.

## Risk Categories And Weighting
Included today:
- Flat commodity price risk.
- Scrap basis risk through volatility multiplier and basis stress.
- Realization risk through net-realizable and liquidation haircuts.
- Concentration by metal through exposure breakdown.
- Inventory aging by metal.
- Data quality through valuation confidence and model warnings.

Needs improvement:
- Calibrated liquidity haircuts by grade and sale velocity.
- Explicit concentration limits.
- Separate confidence reserve for unpriced/proxied inventory.
- Stress tests by yard, grade, and customer/vendor concentration.
- Hedge-adjusted risk exposure if hedging is introduced.

## Edge Cases To Test Or Monitor
- Perfectly safe input: zero exposure and zero volatility.
- Extremely risky input: massive exposure or extreme price moves.
- Missing data: absent required CSV columns.
- Conflicting inputs: GreenSpark inventory vs impossible flow netting.
- Very small price sample.
- Very large inventory file.
- Negative weights, costs, prices, or exposures.
- Zero cost basis.
- Outlier costs or market prices.
- Borderline aging thresholds.
- Empty scenario table.
- Missing metal in a stress scenario.
- Synthetic-only price data.

## Output Quality Backlog
- Add clearer report language for what VaR does and does not mean.
- Add plain-English action items by model status.
- Add data-quality KPIs at top of the HTML report. Implemented.
- Add a "Do not use for final decisions if..." box. Implemented as Decision Guidance in Model Quality.
- Add confidence bands or reserves for low-confidence assumptions.
- Add a change log comparing current run to prior run. Implemented as Run Comparison with `output/last_risk_run.json`.

## Code Quality Backlog
- Continue separating model engine, charting, report rendering, and data loading.
- Centralize all constants and assumption metadata.
- Add typed result objects for validation summaries.
- Reduce duplicate valuation logic between scripts.
- Decide whether legacy FIFO netting is supported or deprecated.
- Avoid silent `fillna(0)` where it can hide data problems.
- Replace broad `except Exception` blocks with explicit errors where practical.

## Testing Backlog
- Backtest VaR against realized portfolio changes.
- Test CLI and HTML produce consistent VaR/MC values.
- Test stale snapshot detection.
- Test synthetic-price degraded status.
- Test proxy-price warnings.
- Test unpriced inventory risk reserve.
- Test missing required columns for every loader.
- Test negative and zero values in GreenSpark snapshots.
- Test stress scenarios with missing metals.
- Test all charts referenced by HTML exist after report generation.
- Test report generation does not rely on stale PNG files.

## Security And Reliability Backlog
- Treat CSVs as untrusted input.
- Prevent spreadsheet formula injection in exported CSV/HTML where needed.
- Avoid leaking sensitive customer/vendor data in public artifacts.
- Add deterministic run metadata and price-source provenance. Implemented for report run summary and price sources.
- Make external data failures visible and auditable. Partially implemented through source labels, stale-cache, and synthetic warnings.
- Consider a production mode that refuses synthetic price data unless explicitly allowed. Implemented with `--fail-on-synthetic`.

## Documentation Backlog
- Keep vault docs synchronized with code behavior.
- Document all assumptions with source, confidence, and review date.
- Document how to interpret model-quality warnings.
- Document how to update input files safely.
- Document what risks the model excludes.
- Add a "model governance" note with review cadence and owner.

## Priority Fix List
1. Calibrate realization haircuts using observed sale, freight, shrink, and bid/ask data.
2. Replace weak proxy price series with better market data where possible.
3. Add stale-input and outlier validation.
4. Add run-to-run comparison in the HTML report.
5. Add model governance documentation and assumption review workflow.
6. Add backtesting and exception monitoring.
7. Add production-mode controls for synthetic price data.
8. Add concentration and liquidity limits by metal, grade, yard, and valuation confidence.

## Open Questions
- What is the acceptable maximum share of unpriced inventory?
- Should synthetic price data ever be allowed in a final report?
- What cash reserve policy should VaR feed into?
- Which metals need paid data sources first?
- Who owns approval of haircuts and stress assumptions?
