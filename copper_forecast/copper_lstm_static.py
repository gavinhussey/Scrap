"""
copper_lstm_static.py — the paper's OTHER (contradictory) protocol.

Tang et al. Sec II.C.2 says: "The dataset is split into a training set (80%,
2005 samples) and a test set (20%, 501 samples) using a time-based split."
That 501-sample test set — not the abstract's 1501 rolling forecasts — is the
likely source of Table II. This script implements exactly that: ONE static
80/20 split, ONE training run, one-step-ahead predictions over the 501-day test
set, scored with the paper's tie-inclusive Dstat.

Everything else matches the paper's parameter table and our rolling replication:
single-layer LSTM(64, sigmoid), dropout 0.15, time_step 2, batch 360, Adam, MSE,
1000 epochs (no early stopping — patience-20 ES collapses it to the mean).

Scaler is fit on the TRAIN split only (no leakage). Pass --leak to fit on the
full series instead, reproducing the common leaky setup many such papers use.
"""

import argparse

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from copper_lstm import (
    load_prices, preprocess_copper_data, build_lstm_model,
    make_train_sequences, dstat_paper, TIME_STEP, BATCH_SIZE, SEED,
)

TRAIN_FRAC = 0.8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--leak", action="store_true",
                   help="fit scaler on the full series (leaky), not train-only")
    return p.parse_args()


def main():
    args = parse_args()
    prices = preprocess_copper_data(load_prices())['Close'].to_numpy(dtype=float)
    n = len(prices)
    split = int(n * TRAIN_FRAC)            # 2005 train / 501 test for n=2506
    print(f"Static split: {split} train / {n - split} test "
          f"(scaler: {'FULL series (leaky)' if args.leak else 'train-only'})")

    # Scale.
    fit_on = prices.reshape(-1, 1) if args.leak else prices[:split].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(fit_on)
    scaled = scaler.transform(prices.reshape(-1, 1))

    # Train once on the training split.
    X_tr, y_tr = make_train_sequences(scaled[:split])
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    model = build_lstm_model()
    print(f"Training once on {len(X_tr)} sequences for {args.epochs} epochs...")
    model.fit(X_tr, y_tr, epochs=args.epochs, batch_size=BATCH_SIZE, verbose=0)

    # One-step-ahead predictions over the 501-day test set (real prior values).
    yhat, ytrue, t_index = [], [], []
    for t in range(split, n):
        scaled_in = scaled[t - TIME_STEP:t].reshape(1, TIME_STEP, 1)
        yhat.append(scaler.inverse_transform(model.predict(scaled_in, verbose=0))[0, 0])
        ytrue.append(prices[t])
        t_index.append(t)
    yhat = np.asarray(yhat); ytrue = np.asarray(ytrue); t_index = np.asarray(t_index)

    rmse_cents = np.sqrt(mean_squared_error(ytrue, yhat)) * 100
    mape = mean_absolute_percentage_error(ytrue, yhat) * 100
    dstat = dstat_paper(yhat, ytrue, prices, t_index)

    print("\n--- LSTM Evaluation (static 80/20, paper Sec II.C.2) ---")
    print(f"Test forecasts: {len(yhat)}")
    print(f"RMSE:  {rmse_cents:.2f} cents/lb")
    print(f"MAPE:  {mape:.4f}%")
    print(f"Dstat: {dstat:.4f} (tie-inclusive, paper Eq.)")
    print("\nPaper Table II (LSTM): RMSE 6.67 c/lb, MAPE 1.09%, Dstat 0.78")


if __name__ == "__main__":
    main()
