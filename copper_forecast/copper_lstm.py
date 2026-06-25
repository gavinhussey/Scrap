"""
copper_lstm.py — faithful replication of Tang et al. (2026), "Copper Price
Forecasting of COMEX Copper Based on Machine Learning: A Comparative Study of
LSTM, ARIMA, and SVR Models" (IEEE ICPEGE), LSTM arm only.

The paper's protocol (Sec. II–III), reproduced here:
  * Data    : COMEX front-month close (HG=F), 14 Sep 2015 – 29 Aug 2025, 2506 days.
  * Clean   : linear interpolation of gaps; z-score outliers (|z|>3) replaced by
              the median of the adjacent 5 days.
  * Forecast: ROLLING one-step-ahead. Initial estimation window = first 1000
              observations; roll forward one day at a time to produce ~1501
              out-of-sample one-step forecasts. Estimation window is capped at
              1000 days (sliding, not expanding — "the desk's memory limit").
  * Model   : single-layer LSTM, 64 units, dropout 0.15, sigmoid activation,
              time step = 2, batch size = 360, Adam, MSE, early stopping.
  * Metrics : RMSE (cents/lb), MAPE (%), and the paper's tie-inclusive Dstat:
                a_t = 1 if (ŷ_{t+1}-y_t)(y_{t+1}-y_t) >= 0 else 0.

Why this differs from the earlier static-split version: the paper re-anchors to
the true current price every day (genuine one-step-ahead) and re-estimates as the
window rolls. That is what produces its high directional accuracy; a single
static fit extrapolated across a regime shift does not.

Retraining cadence: the paper re-tunes "every 250 days". Retraining all 1501
steps would take ~16h (their 38s x 1501), so we retrain on that 250-day cadence
and run true one-step-ahead inference (on real lagged prices) in between. Use
--retrain-every 1 for full daily-retrain fidelity at much higher runtime.
"""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

# Paper-specified hyperparameters.
WINDOW_LEN = 1000      # estimation window cap (sliding)
TIME_STEP = 2          # LSTM lookback
HIDDEN_UNITS = 64
DROPOUT = 0.15
BATCH_SIZE = 360
SEED = 42              # not in the paper; fixed here for reproducibility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--retrain-every", type=int, default=250,
                   help="retrain cadence in days (paper re-tunes every 250; use 1 for daily)")
    p.add_argument("--epochs", type=int, default=1000,
                   help="epochs per fit (paper parameter table: 1000; with batch 360 on a "
                        "1000-day window the model needs the full run to learn — early "
                        "stopping at patience 20 collapses it to the window mean)")
    return p.parse_args()


# ==========================================
# 1. Data Ingestion from yfinance
# ==========================================
def load_prices():
    print("Downloading COMEX Copper Futures data...")
    raw = yf.download('HG=F', start='2015-09-14', end='2025-08-30', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close'].iloc[:, 0].to_frame()
        df.columns = ['Close']
    else:
        df = raw[['Close']].copy()
    print(f"Data downloaded successfully. Total rows: {len(df)}")
    return df


# ==========================================
# 2. Data Preprocessing (paper Sec. II.C.2)
# ==========================================
def preprocess_copper_data(df, column_name='Close'):
    df[column_name] = df[column_name].interpolate(method='linear')

    col = df[column_name]
    valid = col.dropna()
    z_scores = pd.Series(np.nan, index=col.index)
    z_scores.loc[valid.index] = zscore(valid)
    outlier_mask = z_scores.abs() > 3

    for i in range(len(df)):
        if outlier_mask.iloc[i]:
            start_idx = max(0, i - 2)
            end_idx = min(len(df), i + 3)
            df.iloc[i, df.columns.get_loc(column_name)] = \
                df.iloc[start_idx:end_idx][column_name].median()
    return df


# ==========================================
# 3. Model (paper Sec. II.D.1)
# ==========================================
def build_lstm_model():
    model = Sequential()
    model.add(Input(shape=(TIME_STEP, 1)))
    model.add(LSTM(units=HIDDEN_UNITS, activation='sigmoid'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model


def make_train_sequences(scaled):
    """All (time_step -> next) pairs inside a scaled estimation window."""
    X, y = [], []
    for i in range(len(scaled) - TIME_STEP):
        X.append(scaled[i:i + TIME_STEP, 0])
        y.append(scaled[i + TIME_STEP, 0])
    X = np.asarray(X).reshape(-1, TIME_STEP, 1)
    return X, np.asarray(y)


# ==========================================
# 4. Rolling one-step-ahead walk-forward
# ==========================================
def rolling_forecast(prices, retrain_every, epochs):
    n = len(prices)
    yhat, ytrue, t_index = [], [], []
    model, scaler = None, None
    n_retrains = 0

    # Forecast each day from WINDOW_LEN .. n-1 (one-step-ahead, out-of-sample).
    for t in range(WINDOW_LEN, n):
        train_window = prices[t - WINDOW_LEN:t].reshape(-1, 1)
        # Scaler is refit on the trailing window EVERY day (as the paper rolls
        # daily), so inputs stay anchored to the current price level. Cheap.
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_window)

        # The (more expensive) LSTM is retrained on the stated cadence.
        if (t - WINDOW_LEN) % retrain_every == 0:
            scaled = scaler.transform(train_window)
            X_tr, y_tr = make_train_sequences(scaled)
            tf.keras.backend.clear_session()
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            model = build_lstm_model()
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
            n_retrains += 1

        # One-step-ahead: input is the actual last TIME_STEP prices (known at t-1).
        window_in = prices[t - TIME_STEP:t].reshape(-1, 1)
        scaled_in = scaler.transform(window_in).reshape(1, TIME_STEP, 1)
        pred = scaler.inverse_transform(model.predict(scaled_in, verbose=0))[0, 0]

        yhat.append(pred)
        ytrue.append(prices[t])
        t_index.append(t)

    return np.asarray(yhat), np.asarray(ytrue), np.asarray(t_index), n_retrains


def dstat_paper(yhat, ytrue, prices, t_index):
    """Tie-inclusive Dstat from the paper: sign of forecast move vs actual move,
    both measured from the previous day's ACTUAL price."""
    prev = prices[t_index - 1]
    pred_move = yhat - prev
    true_move = ytrue - prev
    return float(np.mean((pred_move * true_move) >= 0))


def main():
    args = parse_args()
    df = preprocess_copper_data(load_prices())
    prices = df['Close'].to_numpy(dtype=float)

    print(f"Rolling one-step-ahead forecast: window={WINDOW_LEN}, "
          f"retrain every {args.retrain_every} days, max {args.epochs} epochs...")
    yhat, ytrue, t_index, n_retrains = rolling_forecast(
        prices, args.retrain_every, args.epochs)

    # Metrics. RMSE reported in cents/lb to match the paper (HG=F is $/lb).
    rmse_cents = np.sqrt(mean_squared_error(ytrue, yhat)) * 100
    mape = mean_absolute_percentage_error(ytrue, yhat) * 100
    dstat = dstat_paper(yhat, ytrue, prices, t_index)

    print("\n--- LSTM Evaluation (paper protocol) ---")
    print(f"Out-of-sample forecasts: {len(yhat)}  (retrains: {n_retrains})")
    print(f"RMSE:  {rmse_cents:.2f} cents/lb")
    print(f"MAPE:  {mape:.4f}%")
    print(f"Dstat: {dstat:.4f} (tie-inclusive, paper Eq.)")
    print("\nPaper Table II (LSTM): RMSE 6.67 c/lb, MAPE 1.09%, Dstat 0.78")


if __name__ == "__main__":
    main()
