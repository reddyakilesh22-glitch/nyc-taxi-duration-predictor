"""
Tests for the data quality gate (src/data/quality.py).

The gate is the smoke alarm for the pipeline: it should pass when the data
looks plausibly like a taxi dataset and fail loudly when it doesn't.
"""
import numpy as np
import pandas as pd

from data.quality import check_data_quality


def make_clean_dataframe(n_rows: int = 200) -> pd.DataFrame:
    """A small synthetic dataframe shaped like cleaned taxi data."""
    rng     = np.random.default_rng(42)
    pickup  = pd.date_range("2024-01-01", periods=n_rows, freq="5min")
    dropoff = pickup + pd.to_timedelta(rng.integers(60, 1800, n_rows), unit="s")
    return pd.DataFrame({
        "tpep_pickup_datetime":  pickup,
        "tpep_dropoff_datetime": dropoff,
        "duration_sec":          (dropoff - pickup).total_seconds().astype(int),
        "trip_distance":         rng.uniform(0.5, 10.0, n_rows),
        "PULocationID":          rng.integers(1, 264, n_rows).astype("int32"),
        "DOLocationID":          rng.integers(1, 264, n_rows).astype("int32"),
        "fare_amount":           rng.uniform(5.0, 40.0, n_rows),
        "passenger_count":       rng.integers(1, 5, n_rows),
    })


def test_quality_gate_passes_on_clean_data():
    """A 200-row valid dataframe should pass with no critical failures."""
    result = check_data_quality(make_clean_dataframe())
    assert result["success"], f"expected success, got failures: {result['failures']}"
    assert result["statistics"]["total_rows"] == 200


def test_quality_gate_catches_too_few_rows():
    """A 5-row dataframe should fail the row-count check (minimum 100)."""
    result = check_data_quality(make_clean_dataframe(n_rows=5))
    assert not result["success"]
    assert any("Check 2" in f for f in result["failures"])


def test_quality_gate_catches_missing_required_column():
    """Dropping a required column should produce a critical failure."""
    df = make_clean_dataframe()
    df = df.drop(columns=["PULocationID"])
    result = check_data_quality(df)
    assert not result["success"]
    assert any("Missing required columns" in f for f in result["failures"])
