"""Leakage guards: preprocessing fit only on train; forbidden columns excluded."""
import numpy as np
import pandas as pd

from src.features import build_features
from src.preprocessing import Preprocessor
from src.targets import FUTURE_RETURN_COL, TARGET_COL, create_direction_target

CONFIG = {
    "features": {
        "lookback_sequence_days": 5,
        "rolling_windows": [3, 5],
        "include_day_of_week": True,
        "include_volume_features": False,
    },
    "target": {"horizon_days": 1, "direction_threshold": 0.0},
}


def _toy(n=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 1,
        "low": close - 1, "close": close,
    })


def test_preprocessor_fits_on_train_only():
    """Imputer median must come from train rows, not the whole dataset."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    train, test = X[:50], X[50:]
    prep = Preprocessor().fit(train)
    # The imputer median should be the train median (~24.5), not the full (~49.5).
    assert np.isclose(prep.imputer.statistics_[0], np.median(train))
    assert not np.isclose(prep.imputer.statistics_[0], np.median(X))
    # Transforming test data must not change the fitted statistics.
    before = prep.imputer.statistics_[0]
    prep.transform(test)
    assert prep.imputer.statistics_[0] == before


def test_future_return_not_a_feature():
    df = _toy()
    feats, feature_names = build_features(df, CONFIG)
    assert FUTURE_RETURN_COL not in feature_names
    assert FUTURE_RETURN_COL not in feats.columns


def test_target_not_a_feature():
    df = _toy()
    feats, feature_names = build_features(df, CONFIG)
    combined = pd.concat([df, feats], axis=1)
    framed = create_direction_target(combined, 1, 0.0)
    # target_up and future_return exist on the frame but are NOT feature columns.
    assert TARGET_COL not in feature_names
    assert TARGET_COL in framed.columns
    assert FUTURE_RETURN_COL not in feature_names
