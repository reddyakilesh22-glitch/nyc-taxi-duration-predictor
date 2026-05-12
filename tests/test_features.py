"""
Tests for feature engineering (src/features/engineering.py).

These tests don't need real TLC data — a small synthetic dataframe shaped
like cleaned taxi data is enough to exercise create_features() end-to-end.
"""
import numpy as np
import pandas as pd

from features.engineering import create_features

EXPECTED_NEW_COLUMNS = {
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_am_rush", "is_pm_rush", "is_rush_hour",
    "is_weekend", "is_late_night",
    "is_pu_manhattan", "is_do_manhattan", "is_airport_trip",
    "is_same_zone", "is_both_manhattan",
    "fare_per_mile", "distance_x_rush", "distance_x_night",
    "manhattan_rush",
}


def make_post_cleaning_dataframe(n_rows: int = 100) -> pd.DataFrame:
    """A synthetic dataframe matching what cleaner.py produces."""
    rng = np.random.default_rng(42)
    pickup = pd.date_range("2026-01-01", periods=n_rows, freq="13min")
    return pd.DataFrame({
        "tpep_pickup_datetime":  pickup,
        "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=10),
        "duration_sec":          rng.integers(120, 3600, n_rows),
        "trip_distance":         rng.uniform(0.5, 15.0, n_rows),
        "fare_amount":           rng.uniform(5.0, 60.0, n_rows),
        "PULocationID":          rng.integers(1, 264, n_rows),
        "DOLocationID":          rng.integers(1, 264, n_rows),
        "passenger_count":       rng.integers(1, 6, n_rows),
    })


def test_create_features_adds_expected_columns():
    """Every documented engineered feature should appear in the output."""
    df = make_post_cleaning_dataframe()
    out = create_features(df)
    missing = EXPECTED_NEW_COLUMNS - set(out.columns)
    assert not missing, f"missing engineered features: {missing}"


def test_create_features_produces_no_nans():
    """Engineered columns must be non-null — the model can't handle NaNs."""
    df = make_post_cleaning_dataframe()
    out = create_features(df)
    new_cols = [c for c in out.columns if c not in df.columns]
    null_count = out[new_cols].isnull().sum().sum()
    assert null_count == 0, f"engineered features contain {null_count} NaNs"


def test_engineered_features_have_correct_ranges():
    """Cyclic encodings stay in [-1,1] and binary flags stay in {0,1}."""
    df = make_post_cleaning_dataframe(n_rows=500)
    out = create_features(df)

    # Cyclic encodings: bounded by sin/cos
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert out[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

    # Binary flags: must be 0 or 1
    for col in ["is_am_rush", "is_pm_rush", "is_rush_hour",
                "is_weekend", "is_late_night",
                "is_pu_manhattan", "is_do_manhattan",
                "is_airport_trip", "is_same_zone", "is_both_manhattan"]:
        unique_values = set(out[col].unique())
        assert unique_values.issubset({0, 1}), f"{col} has values outside {{0,1}}: {unique_values}"

    # hour and day_of_week have known fixed ranges
    assert out["hour"].between(0, 23).all()
    assert out["day_of_week"].between(0, 6).all()
