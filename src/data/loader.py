"""
Day 1: Data Loader & Quality Gate

This script answers the most important question before any ML work:
"Is my data actually usable?"

We load one month of yellow taxi data and inspect it thoroughly.
Run it with:  python src/data/loader.py
"""

import sys
from pathlib import Path

import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
# __file__ is this script. We walk up 3 levels to reach the project root,
# then go into data/tlc/yellow/ to find our parquet files.
PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tlc" / "yellow"


def load_sample(filename: str = "yellow_tripdata_2026-01.parquet") -> pd.DataFrame:
    """Load a single month of yellow taxi data."""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: File not found at {path}")
        print(f"Files available in {DATA_DIR}:")
        for f in sorted(DATA_DIR.glob("*.parquet"))[:5]:
            print(f"  {f.name}")
        sys.exit(1)

    print(f"Loading: {path.name}")
    df = pd.read_parquet(path)
    return df


def inspect_shape(df: pd.DataFrame):
    """How big is the dataset?"""
    print("\n" + "=" * 50)
    print("SHAPE")
    print("=" * 50)
    rows, cols = df.shape
    print(f"  Rows    : {rows:,}  (each row = one taxi trip)")
    print(f"  Columns : {cols}")


def inspect_columns(df: pd.DataFrame):
    """What columns exist and what type is each one?"""
    print("\n" + "=" * 50)
    print("COLUMNS & DATA TYPES")
    print("=" * 50)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<35} {str(dtype)}")


def inspect_summary_stats(df: pd.DataFrame):
    """
    For numeric columns: what are the min, max, mean, std?
    Red flags to look for:
      - Negative values where they shouldn't exist (distance, fare)
      - Min/max that are physically impossible
      - Very high std (data spread) suggesting outliers
    """
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS (numeric columns only)")
    print("=" * 50)
    numeric_df = df.select_dtypes(include="number")
    stats = numeric_df.describe().T[["mean", "std", "min", "max"]]
    stats = stats.round(2)
    print(stats.to_string())


def inspect_missing_values(df: pd.DataFrame):
    """
    Which columns have missing values, and how many?
    A column with 30%+ missing values needs special treatment.
    A column with 0% missing is ready to use as-is.
    """
    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    total_rows = len(df)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  No missing values found.")
    else:
        for col, count in missing.items():
            pct = count / total_rows * 100
            print(f"  {col:<35} {count:>8,}  ({pct:.1f}%)")


def inspect_key_columns(df: pd.DataFrame):
    """
    Spot-check the columns we care most about for trip duration prediction.
    These are the columns that will become our features and target.
    """
    print("\n" + "=" * 50)
    print("KEY COLUMNS FOR DURATION PREDICTION")
    print("=" * 50)

    # Compute trip duration if we have pickup + dropoff times
    if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
        df = df.copy()
        df["duration_sec"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds()

        duration_min = df["duration_sec"] / 60
        print("\n  Trip Duration (minutes):")
        print(f"    Min    : {duration_min.min():.1f}")
        print(f"    Median : {duration_min.median():.1f}")
        print(f"    Mean   : {duration_min.mean():.1f}")
        print(f"    Max    : {duration_min.max():.1f}")
        print(f"    Negative durations : {(df['duration_sec'] < 0).sum():,}  ← should be 0")

    if "trip_distance" in df.columns:
        print("\n  Trip Distance (miles):")
        print(f"    Min    : {df['trip_distance'].min():.2f}")
        print(f"    Median : {df['trip_distance'].median():.2f}")
        print(f"    Max    : {df['trip_distance'].max():.2f}")
        print(f"    Zero or negative : {(df['trip_distance'] <= 0).sum():,}  ← should be 0")

    if "PULocationID" in df.columns:
        print("\n  Pickup Zone IDs (valid range: 1–263):")
        print(f"    Unique zones  : {df['PULocationID'].nunique()}")
        print(f"    Out-of-range  : {(~df['PULocationID'].between(1, 263)).sum():,}")


def main():
    print("=" * 50)
    print("NYC TAXI DATA, QUALITY GATE")
    print("=" * 50)

    df = load_sample()

    inspect_shape(df)
    inspect_columns(df)
    inspect_summary_stats(df)
    inspect_missing_values(df)
    inspect_key_columns(df)

    print("\n" + "=" * 50)
    print("QUALITY GATE COMPLETE")
    print("=" * 50)
    print("Review the output above and ask:")
    print("  1. Are there negative durations or distances?")
    print("  2. Any columns with lots of missing values?")
    print("  3. Do the min/max values make physical sense?")


if __name__ == "__main__":
    main()
