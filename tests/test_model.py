"""
Tests for model training and prediction.

Two layers of testing:
  1. Sanity tests that train a tiny LightGBM in-memory — these run anywhere
     (CI, fresh clones) without needing the production model artifact.
  2. Integration tests that load models/production_model.pkl if it exists.
     These are skipped automatically when the artifact isn't available.
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

PRODUCTION_MODEL = Path(__file__).parents[1] / "models" / "production_model.pkl"


def test_lightgbm_trains_and_predicts():
    """End-to-end sanity: train a tiny LightGBM and verify predictions are finite."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((200, 5)), columns=list("abcde"))
    y = X["a"] * 2 + X["b"] * -1 + rng.normal(0, 0.1, size=200)

    model = lgb.LGBMRegressor(n_estimators=20, num_leaves=15, verbose=-1)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (200,)
    assert np.isfinite(preds).all(), "predictions must be finite"
    # A model that learned anything should have R² > 0 on its training set
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.5, f"trained model R² too low: {r2:.3f}"


@pytest.mark.skipif(
    not PRODUCTION_MODEL.exists(),
    reason="models/production_model.pkl not found — run src/models/run_training.py first",
)
def test_production_model_loads_and_predicts_in_range():
    """Load the real trained model and run a prediction for a typical NYC trip."""
    bundle = joblib.load(PRODUCTION_MODEL)

    assert "model" in bundle
    assert "feature_cols" in bundle
    feature_cols = bundle["feature_cols"]
    model        = bundle["model"]

    # Build a typical trip: 3-mile Midtown → Times Square ride at 8am Monday
    hour, dow, distance = 8, 0, 3.0
    is_rush = 1  # weekday + 7-9am
    row = {
        "VendorID": 2, "passenger_count": 1, "trip_distance": distance,
        "RatecodeID": 1, "PULocationID": 161, "DOLocationID": 230,
        "payment_type": 1,
        "fare_amount": 10.0, "extra": 1.0, "mta_tax": 0.5, "tip_amount": 2.85,
        "tolls_amount": 0.0, "congestion_surcharge": 2.5, "Airport_fee": 0.0,
        "cbd_congestion_fee": 0.75,
        "hour": hour, "day_of_week": dow,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow / 7),
        "dow_cos":  np.cos(2 * np.pi * dow / 7),
        "is_am_rush": 1, "is_pm_rush": 0, "is_rush_hour": is_rush,
        "is_weekend": 0, "is_late_night": 0,
        "is_pu_manhattan": 1, "is_do_manhattan": 1, "is_airport_trip": 0,
        "is_same_zone": 0, "is_both_manhattan": 1,
        "fare_per_mile": 7.5 / distance,
        "distance_x_rush": distance * is_rush,
        "distance_x_night": 0,
    }
    input_df = pd.DataFrame([row])[feature_cols]
    for col in ["PULocationID", "DOLocationID", "RatecodeID",
                "VendorID", "payment_type", "day_of_week"]:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype("category")

    pred_log = model.predict(input_df)[0]
    pred_min = float(np.expm1(pred_log)) / 60

    # A 3-mile Manhattan ride at 8am should be plausibly between 3 and 60 minutes.
    assert 3 <= pred_min <= 60, f"prediction {pred_min:.1f} min outside plausible range"
