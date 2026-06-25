"""LSTM classifier (optional — degrades gracefully if TensorFlow is absent).

The model outputs P(up tomorrow). Sequences are built causally:
    X_seq[t] = scaled features from rows [t-lookback+1 .. t]
    y[t]     = direction realized from t to t+1
so the sequence ending at row t never contains row t+1's features.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .utils import SEED, setup_logging

logger = setup_logging()

# Best-effort TensorFlow import. Absence is not fatal — the backtest skips LSTM.
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    TF_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    TF_AVAILABLE = False
    logger.warning("TensorFlow unavailable (%s). LSTM arm will be skipped.", exc)


def make_sequences(
    features: np.ndarray, targets: Optional[np.ndarray], lookback: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build rolling sequences.

    Returns X with shape (n_seq, lookback, n_features). If ``targets`` is given,
    also returns the aligned y (the target at the LAST row of each sequence).
    """
    features = np.asarray(features, dtype=float)
    n, n_feat = features.shape
    if n < lookback:
        return np.empty((0, lookback, n_feat)), (None if targets is None else np.empty((0,)))

    X_list, y_list = [], []
    for end in range(lookback - 1, n):
        start = end - lookback + 1
        X_list.append(features[start : end + 1])
        if targets is not None:
            y_list.append(targets[end])
    X = np.stack(X_list)
    y = np.asarray(y_list) if targets is not None else None
    return X, y


class LSTMClassifier:
    """Keras LSTM wrapped to match the baseline model interface."""

    name = "lstm"

    def __init__(self, lstm_cfg: dict, lookback: int):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available; cannot build LSTM.")
        self.cfg = lstm_cfg
        self.lookback = lookback
        self.model = None

    def _build(self, n_features: int):
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        model = Sequential([
            Input(shape=(self.lookback, n_features)),
            LSTM(self.cfg["hidden_units"]),
            Dropout(self.cfg["dropout"]),
            Dense(self.cfg["dense_units"], activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.cfg["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray, validation_fraction: float = 0.15):
        """Train with a TIME-ORDERED validation tail and early stopping."""
        if len(X_seq) == 0:
            raise ValueError("No sequences to train on.")
        n_features = X_seq.shape[2]
        self.model = self._build(n_features)

        n_val = max(1, int(len(X_seq) * validation_fraction))
        if len(X_seq) - n_val < 1:
            n_val = 0  # too small to hold out; train on everything

        callbacks = []
        validation_data = None
        if n_val > 0:
            X_tr, y_tr = X_seq[:-n_val], y_seq[:-n_val]
            validation_data = (X_seq[-n_val:], y_seq[-n_val:])
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg["patience"],
                    restore_best_weights=True,
                )
            )
        else:
            X_tr, y_tr = X_seq, y_seq

        self.model.fit(
            X_tr, y_tr,
            validation_data=validation_data,
            epochs=self.cfg["max_epochs"],
            batch_size=self.cfg["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_proba_up(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTM predict called before fit.")
        if len(X_seq) == 0:
            return np.empty((0,))
        return self.model.predict(X_seq, verbose=0).reshape(-1)
