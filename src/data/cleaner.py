"""
Day 1: Data Cleaner

Takes raw yellow taxi data and produces a clean version ready for
feature engineering and model training.

Cleaning steps applied (in order):
  1. Compute trip duration (our target variable)
  2. Drop rows where the target is null or invalid
  3. Remove physically impossible values
  4. Drop exact duplicate rows
  5. Handle remaining nulls in feature columns
  6. Enforce correct data types
  7. Save cleaned output as parquet
  8. Re-run quality gate to confirm

Usage:
    python src/data/cleaner.py
"""

from pathlib import Path

import pandas as pd

from quality import check_data_quality, print_result

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR     = PROJECT_ROOT / "data" / "tlc" / "yellow"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean raw taxi data and return (cleaned_df, quality_result).

    Every step prints what it's doing and how many rows it removed.
    That transparency matters: if a step removes 40% of your data,
    you want to know about it.
    """
    original_rows = len(df)
    print(f"\nStarting with {original_rows:,} rows")

    # ── Step 1: Compute the target variable ─────────────────────────────────
    # We're predicting trip duration, but it doesn't exist as a column yet.
    # We calculate it from pickup and dropoff timestamps.
    # Result is in seconds, easier to work with than minutes for the model.
    df = df.copy()
    df["duration_sec"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    print("\nStep 1, Computed duration_sec from timestamps")

    # ── Step 2: Drop rows where target is null or invalid ───────────────────
    # A row with no duration is useless for a duration-prediction model.
    # We also enforce a sensible range: 1 minute to 3 hours.
    # Under 1 min: likely a cancelled/test trip.
    # Over 3 hours: likely a data error or extreme outlier that would skew training.
    before = len(df)
    df = df[df["duration_sec"].notna()]
    df = df[df["duration_sec"].between(60, 10_800)]  # 1 min → 3 hours
    removed = before - len(df)
    print(f"Step 2: Dropped {removed:,} rows with null/invalid duration "
          f"({removed/before*100:.1f}%)")

    # ── Step 3: Remove physically impossible values ──────────────────────────
    # These come from GPS errors, data entry mistakes, or refund records
    # being mixed into the trip table.
    before = len(df)
    df = df[df["trip_distance"] > 0.1]          # must have moved at least 0.1 miles
    df = df[df["trip_distance"] <= 100]          # no trip is longer than 100 miles in NYC
    df = df[df["PULocationID"].between(1, 263)]  # valid NYC taxi zone range
    df = df[df["DOLocationID"].between(1, 263)]
    df = df[df["fare_amount"] >= 0]              # no negative fares
    df = df[df["RatecodeID"].isin([1, 2, 3, 4, 5, 6]) | df["RatecodeID"].isna()]
    removed = before - len(df)
    print(f"Step 3: Dropped {removed:,} rows with impossible values "
          f"({removed/before*100:.1f}%)")

    # ── Step 4: Remove exact duplicates ─────────────────────────────────────
    # Exact row duplicates sometimes appear when data pipelines re-process files.
    # keep='first' means we keep the first occurrence and drop the rest.
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"Step 4: Dropped {removed:,} exact duplicate rows "
          f"({removed/before*100:.1f}%)")

    # ── Step 5: Handle nulls in feature columns ──────────────────────────────
    # passenger_count: fill with 1 (most common value, safe assumption)
    # congestion_surcharge: fill with 0 (if not recorded, assume not charged)
    # store_and_fwd_flag: fill with 'N' (most common)
    df["passenger_count"]      = df["passenger_count"].fillna(1).clip(1, 6)
    df["congestion_surcharge"] = df["congestion_surcharge"].fillna(0)
    df["store_and_fwd_flag"]   = df["store_and_fwd_flag"].fillna("N")
    df["Airport_fee"]          = df["Airport_fee"].fillna(0)

    # RatecodeID: drop rows where it's still null after the range filter above
    before = len(df)
    df = df.dropna(subset=["RatecodeID"])
    removed = before - len(df)
    print(f"Step 5: Filled nulls; dropped {removed:,} rows with null RatecodeID")

    # ── Step 6: Enforce correct data types ───────────────────────────────────
    # Smaller int types save memory on 120M rows.
    df["PULocationID"]   = df["PULocationID"].astype("int16")
    df["DOLocationID"]   = df["DOLocationID"].astype("int16")
    df["RatecodeID"]     = df["RatecodeID"].astype("int8")
    df["payment_type"]   = df["payment_type"].astype("int8")
    df["passenger_count"]= df["passenger_count"].astype("int8")
    print("Step 6: Enforced compact data types")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_rows = len(df)
    total_removed = original_rows - final_rows
    print("\nCleaning complete:")
    print(f"  Before : {original_rows:,} rows")
    print(f"  After  : {final_rows:,} rows")
    print(f"  Removed: {total_removed:,} rows ({total_removed/original_rows*100:.1f}%)")

    # ── Step 7: Run quality gate on cleaned data ──────────────────────────────
    quality_result = check_data_quality(df)

    return df, quality_result


if __name__ == "__main__":
    input_file  = RAW_DIR / "yellow_tripdata_2026-01.parquet"
    output_file = CLEANED_DIR / "yellow_tripdata_2026-01_cleaned.parquet"

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_file.name}...")
    raw_df = pd.read_parquet(input_file)

    cleaned_df, quality_result = clean_data(raw_df)

    # Save as parquet (fast, compressed) and CSV (guide requirement)
    cleaned_df.to_parquet(output_file, index=False)
    csv_file = CLEANED_DIR / "cleaned.csv"
    cleaned_df.to_csv(csv_file, index=False)
    size_mb = output_file.stat().st_size / 1_048_576
    csv_mb  = csv_file.stat().st_size / 1_048_576
    print(f"\nSaved cleaned data → {output_file.name} ({size_mb:.0f} MB)")
    print(f"Saved cleaned data → {csv_file.name} ({csv_mb:.0f} MB)")

    # Show quality gate on cleaned data
    print_result(quality_result)
