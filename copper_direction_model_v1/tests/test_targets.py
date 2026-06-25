"""Target correctness and leakage-at-the-edge tests."""
import numpy as np
import pandas as pd

from src.targets import FUTURE_RETURN_COL, TARGET_COL, create_direction_target


def _toy(closes):
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"date": dates, "close": np.asarray(closes, dtype=float)})


def test_target_uses_tomorrow_not_today():
    # Strictly increasing -> every known day is an UP day.
    df = _toy([10, 11, 12, 13, 14])
    out = create_direction_target(df, horizon_days=1, threshold=0.0)
    # With horizon 1, the final row (unknown future) is dropped -> 4 rows remain.
    assert len(out) == 4
    assert (out[TARGET_COL] == 1).all()

    # future_return on row 0 must equal (close[1]/close[0] - 1), i.e. tomorrow.
    expected = 11 / 10 - 1
    assert np.isclose(out[FUTURE_RETURN_COL].iloc[0], expected)


def test_down_and_threshold():
    df = _toy([10, 10.005, 9, 9])  # tiny up, then down
    out = create_direction_target(df, horizon_days=1, threshold=0.001)
    # row0: +0.05% move <= 0.1% threshold -> 0 ; row1: big drop -> 0
    assert out[TARGET_COL].iloc[0] == 0
    assert out[TARGET_COL].iloc[1] == 0


def test_final_unknown_rows_dropped():
    df = _toy(list(range(1, 11)))  # 10 rows
    out = create_direction_target(df, horizon_days=3, threshold=0.0)
    # last 3 rows have unknown future -> dropped.
    assert len(out) == 7
    assert out[FUTURE_RETURN_COL].notna().all()
