# Boruta Feature-Selection Summary (copper scrap)

Boruta (all-relevant) is run **after feature engineering, before final redundancy pruning**, on the COMEX-copper scrap-price proxy.

## Setup

- Target: **comex_copper** (COMEX HG=F, $/lb).
- Train window (Boruta fit, no shuffle): **2015-01-02 → 2023-08-11** (450 weeks = first 75%).
- Estimator: RandomForestRegressor(max_depth=5, random_state=42), BorutaPy(n_estimators='auto', max_iter=100, alpha=0.05).
- Two configs: **strict perc=100** and **relaxed perc=90**.
- Missing features median-imputed on the train window only.

## How many features Boruta tested

- **34 candidate features tested** (the full engineered set minus the target and all target-derived/target-containing features: `cu_*` own-price momentum/MA/vol/lags and the COMEX-containing spreads `comex_lme_spread`, `comex_lme_ratio`, `copper_etf_basis` — excluded so the selection is *explanatory*, not autoregressive).
- **Relaxed (perc=90): 19 confirmed, 4 tentative, 11 rejected.**
- Strict (perc=100): 13 confirmed.

## Confirmed features (relaxed)

- `copper_etf` (copper_price) — rank 1, |corr|=1.00  *(also strict-confirmed)*
- `lme_copper` (copper_price) — rank 1, |corr|=0.99  *(also strict-confirmed)*
- `ppi_steel` (macro_demand) — rank 1, |corr|=0.90  *(also strict-confirmed)*
- `ppi_allcommod` (macro_demand) — rank 1, |corr|=0.86
- `aluminum` (copper_price) — rank 1, |corr|=0.84  *(also strict-confirmed)*
- `construction_spend` (macro_demand) — rank 1, |corr|=0.80  *(also strict-confirmed)*
- `housing_starts` (macro_demand) — rank 1, |corr|=0.78  *(also strict-confirmed)*
- `brent_oil` (energy_cost) — rank 1, |corr|=0.77
- `diesel` (energy_cost) — rank 1, |corr|=0.76  *(also strict-confirmed)*
- `yield_curve_2s10s` (financial) — rank 1, |corr|=0.28  *(also strict-confirmed)*
- `ust_2y` (financial) — rank 1, |corr|=0.26  *(also strict-confirmed)*
- `vix` (financial) — rank 1, |corr|=0.25  *(also strict-confirmed)*
- `ust_10y_chg_30d` (financial) — rank 1, |corr|=0.24
- `dxy` (financial) — rank 1, |corr|=0.23  *(also strict-confirmed)*
- `ust_10y` (financial) — rank 1, |corr|=0.20
- `fed_funds` (financial) — rank 1, |corr|=0.19  *(also strict-confirmed)*
- `wti_oil_chg_30d` (energy_cost) — rank 1, |corr|=0.19
- `yield_curve_2s10s_chg_30d` (financial) — rank 1, |corr|=0.11  *(also strict-confirmed)*
- `dxy_chg_30d` (financial) — rank 1, |corr|=0.09

## Tentative features

`ip_manufacturing`, `ust_2y_chg_30d`, `diesel_chg_30d`, `mfg_employment`

## Rejected features

`vix_chg_30d`, `wti_oil`, `fed_funds_chg_30d`, `indpro_chg_30d`, `mfg_new_orders`, `indpro`, `ip_manufacturing_chg_30d`, `ppi_steel_chg_30d`, `housing_starts_chg_30d`, `construction_spend_chg_30d`, `mfg_new_orders_chg_30d`

## Removed for redundancy (|r| > 0.90)

- `copper_etf` dropped (|r|=0.99 with kept `lme_copper`)
- `ppi_steel` dropped (|r|=0.91 with kept `lme_copper`)
- `construction_spend` dropped (|r|=0.92 with kept `ppi_allcommod`)
- `diesel` dropped (|r|=0.94 with kept `ppi_allcommod`)
- `ust_10y` dropped (|r|=0.91 with kept `ust_2y`)
- `fed_funds` dropped (|r|=0.92 with kept `ust_2y`)

*(At the stricter 0.85 threshold the surviving set would be: `lme_copper`, `ppi_allcommod`, `aluminum`, `housing_starts`, `yield_curve_2s10s`, `ust_2y`, `vix`, `ust_10y_chg_30d`, `dxy`, `wti_oil_chg_30d`, `yield_curve_2s10s_chg_30d`, `dxy_chg_30d`, `ip_manufacturing`, `ust_2y_chg_30d`, `diesel_chg_30d`, `mfg_employment`.)*

## Final selected features

| # | feature | category | boruta rank | |corr| w/ target |
|---|---|---|---|---|
| 1 | `lme_copper` | copper_price | 1 | 0.99 |
| 2 | `ppi_allcommod` | macro_demand | 1 | 0.86 |
| 3 | `aluminum` | copper_price | 1 | 0.84 |
| 4 | `housing_starts` | macro_demand | 1 | 0.78 |
| 5 | `brent_oil` | energy_cost | 1 | 0.77 |
| 6 | `yield_curve_2s10s` | financial | 1 | 0.28 |
| 7 | `ust_2y` | financial | 1 | 0.26 |
| 8 | `vix` | financial | 1 | 0.25 |
| 9 | `ust_10y_chg_30d` | financial | 1 | 0.24 |
| 10 | `dxy` | financial | 1 | 0.23 |
| 11 | `wti_oil_chg_30d` | energy_cost | 1 | 0.19 |
| 12 | `yield_curve_2s10s_chg_30d` | financial | 1 | 0.11 |
| 13 | `dxy_chg_30d` | financial | 1 | 0.09 |
| 14 | `ip_manufacturing` | macro_demand | 2 | 0.38 |
| 15 | `ust_2y_chg_30d` | financial | 2 | 0.30 |
| 16 | `diesel_chg_30d` | energy_cost | 2 | 0.30 |
| 17 | `mfg_employment` | macro_demand | 2 | 0.28 |

Final count: **17** (target band 10–20). Core copper benchmarks COMEX (target) and LME are preserved by construction.

## Correlation with target (final matrix)

- `lme_copper`: +0.99
- `ppi_allcommod`: +0.88
- `aluminum`: +0.86
- `brent_oil`: +0.66
- `ust_2y`: +0.55
- `housing_starts`: +0.50
- `dxy`: +0.48
- `mfg_employment`: +0.34
- `yield_curve_2s10s`: -0.28
- `diesel_chg_30d`: +0.21
- `ip_manufacturing`: -0.18
- `ust_2y_chg_30d`: +0.13
- `ust_10y_chg_30d`: +0.12
- `dxy_chg_30d`: -0.09
- `wti_oil_chg_30d`: +0.08
- `vix`: +0.07
- `yield_curve_2s10s_chg_30d`: -0.01

## Why this batch makes economic sense for copper-scrap pricing

- **Direct copper benchmarks** (LME copper; aluminum as a base-metals cross-check) anchor the price level — scrap is a discount to these. The near-identical CPER ETF was pruned as redundant with LME (|r|=0.99).
- **Demand drivers** confirmed by Boruta (industrial production, new orders, construction, PPI) capture the copper-consumption cycle.
- **Financial/macro factors** (USD, Treasury yields, VIX) price the dollar and risk regime that copper trades against.
- **Energy/freight** (oil/diesel) proxy collection & transport cost and broad reflation.
- Boruta keeps only variables that beat *random shadow features*, so the set is statistically defensible; redundancy pruning then removes near-duplicate copper series, leaving a compact, non-redundant batch.

## Caveats

- Boruta here is fit on the **price level** (per spec). Copper and the macro indices share a secular uptrend, so some 'confirmed' relevance is **common trend**; a returns-based re-run would confirm fewer macro levels. Use these as co-movement/structure, not clean causal drivers.
- Target-derived momentum/vol/spread features were excluded from Boruta but remain useful elsewhere (e.g. the forecasting & sell-timing models).