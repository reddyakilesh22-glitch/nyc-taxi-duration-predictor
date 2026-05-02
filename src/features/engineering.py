"""
Day 3 — Feature Engineering

Raw data columns are often poor model inputs. This module transforms them
into signals that a model can learn from directly.

Every feature here has a documented reason (the WHY) — that's intentional.
If you can't explain why a feature should predict trip duration, it probably
won't help and might hurt by adding noise.

Usage:
    python src/features/engineering.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "yellow_tripdata_2024-01_cleaned.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2024-01_features.parquet"


# ── NYC Domain Knowledge ─────────────────────────────────────────────────────
# Airport zone IDs in the TLC dataset
AIRPORT_ZONES = {132, 138}   # JFK (132), LaGuardia (138)
EWR_ZONE = {1}               # Newark (technically NJ but in dataset)
ALL_AIRPORT_ZONES = AIRPORT_ZONES | EWR_ZONE

# Manhattan zone IDs run from 4–263 but the core is roughly 4–90 and 125–265
# Simpler: Manhattan borough is encoded in the zone lookup. Here we use a
# known range that covers all Manhattan zones.
MANHATTAN_ZONES = set(range(4, 153)) | {161, 162, 163, 164, 166, 170, 186,
                                         194, 202, 209, 211, 224, 229, 230,
                                         231, 232, 233, 234, 236, 237, 238,
                                         239, 243, 244, 246, 249, 261, 262}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 12+ features across 3 categories:
      1. Temporal — when the trip happened
      2. Domain (geospatial) — where and route type
      3. Interaction — combinations that matter more together than alone

    Args:
        df: Cleaned DataFrame with tpep_pickup_datetime, PULocationID,
            DOLocationID, trip_distance, fare_amount, duration_sec

    Returns:
        DataFrame with all original columns + new engineered features
    """
    df = df.copy()
    pickup = df["tpep_pickup_datetime"]

    # ── Category 1: Temporal Features ────────────────────────────────────────
    # WHY: Traffic volume — and therefore trip duration — follows strong daily
    # and weekly cycles. A model needs these patterns explicitly.

    df["hour"] = pickup.dt.hour
    df["day_of_week"] = pickup.dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = pickup.dt.month

    # WHY: Cyclic encoding — hour 23 and hour 0 are only 1 hour apart, but
    # numerically they're 23 apart. Sin/cos encoding wraps the scale so the
    # model sees them as close. Without this, the model thinks midnight is
    # the most different time from 11pm.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # WHY: Rush hours have the biggest effect on duration — more cars on road.
    # AM rush: 7–9am weekdays. PM rush: 4–7pm weekdays.
    is_weekday = df["day_of_week"] < 5
    df["is_am_rush"] = (is_weekday & df["hour"].between(7, 9)).astype("int8")
    df["is_pm_rush"] = (is_weekday & df["hour"].between(16, 19)).astype("int8")
    df["is_rush_hour"] = ((df["is_am_rush"] == 1) | (df["is_pm_rush"] == 1)).astype("int8")

    # WHY: Weekends have different traffic patterns — less commuter traffic
    # but more leisure trips. Trips on weekends tend to be shorter.
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # WHY: Late night (10pm–5am) streets are nearly empty. These trips move
    # much faster than the same route during the day.
    df["is_late_night"] = (
        (df["hour"] >= 22) | (df["hour"] <= 5)
    ).astype("int8")

    # ── Category 2: Domain (Geospatial) Features ─────────────────────────────
    # WHY: Where a trip starts and ends tells you about expected traffic.
    # Manhattan is the most congested borough — pickups there add time.

    df["is_pu_manhattan"] = df["PULocationID"].isin(MANHATTAN_ZONES).astype("int8")
    df["is_do_manhattan"] = df["DOLocationID"].isin(MANHATTAN_ZONES).astype("int8")

    # WHY: Airport trips have fixed long distances and often use highways.
    # They behave very differently from city trips — often faster per mile.
    df["is_airport_trip"] = (
        df["PULocationID"].isin(ALL_AIRPORT_ZONES) |
        df["DOLocationID"].isin(ALL_AIRPORT_ZONES)
    ).astype("int8")

    # WHY: Trips within the same zone are almost always very short (< 5 min).
    # This is a strong signal that the model would otherwise need many examples
    # to learn from zone ID combinations.
    df["is_same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype("int8")

    # WHY: Both pickup and dropoff in Manhattan = worst congestion scenario.
    df["is_both_manhattan"] = (
        (df["is_pu_manhattan"] == 1) & (df["is_do_manhattan"] == 1)
    ).astype("int8")

    # WHY: Fare per mile is a derived efficiency metric. Slow congested trips
    # have a high fare-per-mile (meter runs on time). Fast trips have low
    # fare-per-mile. This is a proxy for route congestion.
    df["fare_per_mile"] = (df["fare_amount"] / df["trip_distance"].replace(0, np.nan)).fillna(0)

    # ── Category 3: Interaction Features ─────────────────────────────────────
    # WHY: Interaction features capture cases where two things together
    # matter more than either alone. A long trip during rush hour is much
    # worse than a long trip at 3am — the distance × rush hour product
    # captures that compounding effect.

    # WHY: A long trip during rush hour is disproportionately slow.
    # distance=5 miles at 8am could take 40 min; at 3am it takes 10 min.
    df["distance_x_rush"] = df["trip_distance"] * df["is_rush_hour"]

    # WHY: Manhattan during rush hour is the peak congestion scenario.
    # Even a 1-mile trip can take 20+ minutes in midtown at 5pm.
    df["manhattan_rush"] = df["is_pu_manhattan"] * df["is_rush_hour"]

    # WHY: Late night trips cover ground quickly regardless of distance.
    # Captures the interaction between time-of-day and trip length.
    df["distance_x_night"] = df["trip_distance"] * df["is_late_night"]

    return df


def select_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """
    Remove redundant and near-zero-variance features.

    Two removal rules:
      1. Correlation > 0.95 between two features → drop the second one
         (they carry the same information — keeping both is redundant)
      2. Variance < 1% of average feature variance → drop
         (nearly constant columns teach the model nothing)

    Returns:
        (selected_feature_names, reduced_dataframe)
    """
    # Only apply selection to engineered numeric features, not the target
    exclude = {"duration_sec", "tpep_pickup_datetime", "tpep_dropoff_datetime",
               "store_and_fwd_flag"}
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude
    ]

    dropped = {}

    # ── Rule 1: Drop highly correlated features ───────────────────────────
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop_corr = []
    for col in upper.columns:
        if upper[col].max() > 0.95:
            corr_partner = upper[col].idxmax()
            to_drop_corr.append(col)
            dropped[col] = f"correlation {upper[col].max():.3f} with '{corr_partner}'"

    remaining = [c for c in numeric_cols if c not in to_drop_corr]

    # ── Rule 2: Drop near-zero-variance features ──────────────────────────
    variances = df[remaining].var()
    # Fixed absolute threshold: drop only near-constant columns (>99% same value).
    # Using mean-relative threshold is wrong here because high-variance continuous
    # features (trip_distance) inflate the mean, causing binary (0/1) features to
    # be dropped even though they carry real signal.
    threshold = 0.001
    to_drop_var = variances[variances < threshold].index.tolist()

    for col in to_drop_var:
        dropped[col] = f"variance {variances[col]:.6f} below threshold {threshold}"

    selected = [c for c in remaining if c not in to_drop_var]

    # Always keep the target
    if "duration_sec" in df.columns and "duration_sec" not in selected:
        selected.append("duration_sec")

    # Report
    print(f"\nFeature selection:")
    print(f"  Started with : {len(numeric_cols)} features")
    print(f"  Dropped      : {len(dropped)}")
    print(f"  Kept         : {len(selected)}")
    if dropped:
        print(f"\n  Dropped features:")
        for col, reason in dropped.items():
            print(f"    - {col}: {reason}")

    return selected, df[selected + [c for c in ["tpep_pickup_datetime",
                                                  "tpep_dropoff_datetime",
                                                  "store_and_fwd_flag"]
                                     if c in df.columns]]


if __name__ == "__main__":
    print(f"Loading {CLEANED_PATH.name}...")
    df = pd.read_parquet(CLEANED_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    t0 = time.time()
    df_feat = create_features(df)
    elapsed = time.time() - t0

    print(f"\nFeatures created in {elapsed:.1f}s")
    print(f"  Before: {df.shape[1]} columns")
    print(f"  After : {df_feat.shape[1]} columns")
    print(f"\nNew columns added:")
    new_cols = [c for c in df_feat.columns if c not in df.columns]
    for col in new_cols:
        sample_vals = df_feat[col].dropna().head(3).tolist()
        print(f"  {col:<25} sample: {sample_vals}")

    print(f"\nChecking data integrity:")
    inf_count = np.isinf(df_feat.select_dtypes(include="number")).sum().sum()
    null_count = df_feat.isnull().sum().sum()
    print(f"  Infinite values: {inf_count}")
    print(f"  Null values    : {null_count}")

    print("\nRunning feature selection...")
    selected_cols, df_selected = select_features(df_feat)
    print(f"\nFinal shape: {df_selected.shape[0]:,} rows × {df_selected.shape[1]} columns")
