# Copper Direction Model — V1 (research)

A small, leakage-aware framework that predicts whether **tomorrow's COMEX copper
close will be UP or DOWN** versus today's close. It is a **V1 research test
harness**, not a production system and **not financial advice**.

---

## What this is (and isn't)

- **This is direction prediction**, framed as binary classification:
  `target = 1 if close[t+1] > close[t] else 0` (with an optional no-trade
  threshold for tiny moves). The question is *which way*, not *what price*.
- **This is not price prediction.** Predicting the price level is easy to make
  look good (tomorrow ≈ today scores a low error) yet says nothing about
  direction. Directional accuracy ("hit rate") is the honest metric and the one
  we rank on.

It adapts the LSTM/ARIMA/SVR comparison framework from the COMEX-copper paper to
classification, while **fixing that framework's weaknesses**: no full-dataset
preprocessing, no future-aware outlier replacement, no random train/test split,
and a strict walk-forward with per-window scaler/imputer fitting.

---

## Project layout

```
copper_direction_model_v1/
  config.yaml            # all settings
  requirements.txt
  data/raw/              # put your copper.csv here
  data/processed/        # clean_prices.csv is written here
  models/                # reserved for saved models
  reports/backtests/     # prediction log + metric tables (CSV)
  reports/figures/       # PNG charts
  reports/latest_prediction.json
  src/                   # all code (data_loader, features, targets, ... )
  tests/                 # pytest leakage / causality / target tests
```

---

## Setup

From inside `copper_direction_model_v1`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt    # TensorFlow is optional at runtime
```

TensorFlow powers the LSTM arm. If it is not installed, the code logs a warning
and runs every other model; nothing crashes.

---

## Add your data

Drop a CSV at `data/raw/copper.csv` (or point `data.input_csv` at it). Columns
are auto-detected and normalized internally. Accepted names:

| canonical | accepted source columns |
|-----------|-------------------------|
| date      | `Date`, `date`, `timestamp` |
| open      | `Open`, `open` |
| high      | `High`, `high` |
| low       | `Low`, `low` |
| close     | `Close`, `close`, `Settle`, `settle` |
| volume    | `Volume`, `volume` (optional) |

If both `Close` and `Settle` exist, `Settle` is used (toggle with
`data.prefer_settle`). Rows with a missing close are dropped; missing OHL are
forward-filled using **past** values only.

---

## Run

Full walk-forward backtest + evaluation:

```bash
python -m src.train --config config.yaml
```

Outputs: `reports/backtests/walk_forward_predictions.csv`,
`metrics_summary.csv`, `yearly_metrics.csv`, `regime_metrics.csv`, and charts in
`reports/figures/`. The console prints a model ranking by balanced accuracy, hit
rate, strategy Sharpe, and max drawdown.

Predict the next trading day:

```bash
python -m src.predict_next --config config.yaml
```

Prints per-model and ensemble P(up) plus a Bullish / Bearish / No-trade signal,
and writes `reports/latest_prediction.json`.

---

## How the backtest works

Rolling walk-forward, one test day at a time:

1. Train window = the trailing `initial_train_size` rows **before** day *t*.
2. Imputer + scaler are fit on **that window only**.
3. Non-LSTM models retrain every day; the LSTM retrains every
   `retrain_lstm_every` days and scores the days in between with a causally
   built sequence (the last `lookback_sequence_days` rows up to *t*).
4. The model predicts P(up) for day *t*; the realized direction is recorded.

Nothing from day *t* or later influences the fit used to predict day *t*.

---

## Leakage prevention

- **No full-dataset fitting.** Scaler/imputer are fit per train window.
- **Causal features only.** Rolling windows are trailing (never centered); a
  test (`tests/test_features.py`) perturbs future rows and asserts past features
  are unchanged.
- **Outliers are kept, not overwritten** with future-aware medians; extreme
  moves are *described* by features (ranges, vol, skew/kurt) instead.
- **`future_return` and `target_up` are never features** (enforced by
  `tests/test_no_leakage.py`). `future_return` is retained solely to score the
  backtest.
- **No random split.** Strict time order throughout.

Run the guards:

```bash
python -m pytest -q
```

---

## Metrics

Per model: accuracy, **balanced accuracy**, precision/recall for up & down, F1,
ROC AUC, Brier score, confusion matrix, **directional hit rate**, % of days
traded, average next-day return when long/short, a simple strategy return
(`signal * future_return - cost`), cumulative return, annualized Sharpe, max
drawdown, plus per-year and high/low-volatility-regime breakdowns.

**Reading the result honestly:** daily commodity direction is close to a random
walk. A hit rate near **0.50** means **no edge** — treat it as the default
expectation, not a disappointment. Only a hit rate meaningfully above 0.50 with
a confidence interval clear of the base rate is interesting.

---

## ⚠️ Disclaimer

**V1 is research only.** It is **not financial advice**, not validated for
trading or hedging, and not production-ready. Results on synthetic or untested
data mean nothing about real performance. **Validate on real COMEX copper data
and genuine out-of-sample periods before any real-world use.**
