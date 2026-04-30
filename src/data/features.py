"""
Feature engineering for NYC yellow taxi trip duration prediction.

All features are derived from zone IDs, timestamps, and trip_distance.
No raw lat/lon exists in the dataset; borough-level geography comes from
the TLC zone lookup table.
"""

import polars as pl

from src.data.loader import load_zone_lookup, scan_yellow_filtered

# Rush hour windows (weekday only)
AM_RUSH = (7, 9)   # 7:00–9:59
PM_RUSH = (16, 19) # 16:00–19:59

FEATURE_COLS = [
    "trip_distance",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_am_rush",
    "is_pm_rush",
    "pu_borough",
    "do_borough",
    "same_borough",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "congestion_surcharge",
]

TARGET_COL = "duration_sec"


def _attach_borough(lf: pl.LazyFrame, zone_df: pl.DataFrame) -> pl.LazyFrame:
    borough_map = zone_df.select(["location_id", "Borough"]).lazy()

    lf = lf.join(
        borough_map.rename({"location_id": "PULocationID", "Borough": "pu_borough"}),
        on="PULocationID", how="left",
    ).join(
        borough_map.rename({"location_id": "DOLocationID", "Borough": "do_borough"}),
        on="DOLocationID", how="left",
    )
    return lf


def build_features(
    year_start: int = 2023,
    year_end: int = 2025,
) -> pl.LazyFrame:
    """
    Return a lazy frame with all engineered features and the target column.
    Call .collect() to materialise — processes all months end-to-end in one pass.
    """
    zone_df = load_zone_lookup()
    lf = scan_yellow_filtered(year_start, year_end)

    hour_expr = pl.col("tpep_pickup_datetime").dt.hour().alias("hour")
    dow_expr = pl.col("tpep_pickup_datetime").dt.weekday().alias("day_of_week")  # 0=Mon
    month_expr = pl.col("tpep_pickup_datetime").dt.month().alias("month")

    is_weekend = (pl.col("tpep_pickup_datetime").dt.weekday() >= 5).alias("is_weekend")
    is_am_rush = (
        (pl.col("tpep_pickup_datetime").dt.weekday() < 5)
        & pl.col("tpep_pickup_datetime").dt.hour().is_between(AM_RUSH[0], AM_RUSH[1])
    ).alias("is_am_rush")
    is_pm_rush = (
        (pl.col("tpep_pickup_datetime").dt.weekday() < 5)
        & pl.col("tpep_pickup_datetime").dt.hour().is_between(PM_RUSH[0], PM_RUSH[1])
    ).alias("is_pm_rush")
    is_rush_hour = (pl.col("is_am_rush") | pl.col("is_pm_rush")).alias("is_rush_hour")

    lf = (
        lf
        .with_columns([hour_expr, dow_expr, month_expr, is_weekend, is_am_rush, is_pm_rush])
        .with_columns(is_rush_hour)
    )

    lf = _attach_borough(lf, zone_df)
    lf = lf.with_columns(
        (pl.col("pu_borough") == pl.col("do_borough")).alias("same_borough")
    )

    # Cast boroughs to categorical for LightGBM
    lf = lf.with_columns([
        pl.col("pu_borough").cast(pl.Categorical),
        pl.col("do_borough").cast(pl.Categorical),
    ])

    return lf.select(FEATURE_COLS + [TARGET_COL])
