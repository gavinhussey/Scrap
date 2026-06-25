"""Feature causality tests on a tiny, hand-checkable dataset."""
import numpy as np
import pandas as pd

from src.features import build_features

CONFIG = {
    "features": {
        "lookback_sequence_days": 5,
        "rolling_windows": [3, 5],
        "include_day_of_week": True,
        "include_volume_features": False,
    }
}


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 1,
        "low": close - 1, "close": close,
    })


def test_daily_return_known_value():
    df = _toy()
    feats, _ = build_features(df, CONFIG)
    expected = df["close"].iloc[5] / df["close"].iloc[4] - 1
    assert np.isclose(feats["daily_return"].iloc[5], expected)


def test_rolling_uses_only_past():
    """Changing a FUTURE close must not alter features at an earlier row."""
    df = _toy()
    feats_a, names = build_features(df, CONFIG)
    row = 20

    df2 = df.copy()
    df2.loc[30:, "close"] += 50.0          # perturb only the future
    df2.loc[30:, ["open", "high", "low"]] = df2.loc[30:, ["close"]].values + [[0, 1, -1]]
    feats_b, _ = build_features(df2, CONFIG)

    a = feats_a.loc[row, names].to_numpy(float)
    b = feats_b.loc[row, names].to_numpy(float)
    # Equal where both defined (NaNs compare unequal, so mask them).
    mask = ~(np.isnan(a) | np.isnan(b))
    assert np.allclose(a[mask], b[mask]), "future data leaked into past features"


def test_no_centered_window_first_rows_nan():
    df = _toy()
    feats, _ = build_features(df, CONFIG)
    # A 5-day trailing window cannot be defined before row 4.
    assert feats["roll_vol_5"].iloc[:4].isna().all()
    assert not np.isnan(feats["roll_vol_5"].iloc[5])
