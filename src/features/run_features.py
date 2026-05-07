"""
Day 3 — Feature Engineering Pipeline

Ties together create_features() and select_features() into a single
reproducible script. Run this once per dataset to produce the feature
file used by model training.

Usage:
    python src/features/run_features.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from features.engineering import create_features, select_features

PROJECT_ROOT = Path(__file__).parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "yellow_tripdata_2024-01_cleaned.parquet"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "features"
OUTPUT_PATH  = OUTPUT_DIR / "yellow_tripdata_2024-01_features.parquet"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading cleaned data...")
    t0 = time.time()
    df = pd.read_parquet(CLEANED_PATH)
    print(f"  {len(df):,} rows × {df.shape[1]} columns  ({time.time()-t0:.1f}s)")

    # ── Engineer ─────────────────────────────────────────────────────────────
    print(f"\nEngineering features...")
    t1 = time.time()
    df_feat = create_features(df)
    print(f"  {df.shape[1]} → {df_feat.shape[1]} columns  ({time.time()-t1:.1f}s)")

    # ── Select ───────────────────────────────────────────────────────────────
    print(f"\nSelecting features...")
    selected_cols, df_final = select_features(df_feat)

    # ── Save as parquet (fast) and CSV (guide requirement) ───────────────────
    df_final.to_parquet(OUTPUT_PATH, index=False)
    csv_path = OUTPUT_DIR / "features.csv"
    df_final.to_csv(csv_path, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    csv_mb  = csv_path.stat().st_size / 1_048_576

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"  Input  : {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Output : {len(df_final):,} rows × {df_final.shape[1]} columns")
    print(f"  Saved  : {OUTPUT_PATH.name} ({size_mb:.0f} MB)")
    print(f"  Saved  : {csv_path.name} ({csv_mb:.0f} MB)")
    print(f"\n  Kept features ({len(selected_cols)}):")
    for col in sorted(selected_cols):
        print(f"    {col}")


if __name__ == "__main__":
    main()
