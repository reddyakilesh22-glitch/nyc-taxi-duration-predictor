"""
Day 1: Data Quality Gate

Runs 5 automated checks on a DataFrame and returns a structured result.
Think of this as a smoke alarm for your data, it doesn't fix anything,
it just tells you whether something is wrong before you waste time on it.

Usage:
    python src/data/quality.py
"""

from pathlib import Path

import pandas as pd

# ── Required columns for this project ───────────────────────────────────────
# These must exist and be the right type, or we can't build the model at all.
REQUIRED_COLUMNS = {
    "tpep_pickup_datetime":  "datetime64[us]",
    "tpep_dropoff_datetime": "datetime64[us]",
    "PULocationID":          "int32",
    "DOLocationID":          "int32",
    "trip_distance":         "float64",
}

TARGET_COLUMN = "duration_sec"   # computed by cleaner.py, not in raw data yet


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run 5 data quality checks and return a structured result.

    Returns:
        {
            'success':    True if no critical failures,
            'failures':   list of strings describing critical errors,
            'warnings':   list of strings describing non-critical concerns,
            'statistics': dict of useful counts and metrics,
        }
    """
    failures = []
    warnings = []
    statistics = {}

    # ── Check 1: Schema Validation ───────────────────────────────────────────
    # Do the required columns exist? Are they the right type?
    # Wrong types cause silent bugs, e.g., a date stored as a string won't
    # let you subtract timestamps to get duration.
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        failures.append(f"Check 1 FAIL, Missing required columns: {missing_cols}")
    else:
        wrong_types = []
        for col, expected in REQUIRED_COLUMNS.items():
            actual = str(df[col].dtype)
            # Allow close matches: int32/int64, float32/float64
            if not actual.startswith(expected.split("[")[0].rstrip("0123456789")):
                wrong_types.append(f"{col}: expected {expected}, got {actual}")
        if wrong_types:
            warnings.append(f"Check 1 WARN, Unexpected dtypes: {wrong_types}")
        else:
            statistics["schema"] = "OK"

    # ── Check 2: Row Count ───────────────────────────────────────────────────
    # Too few rows = can't train a model / file is probably empty or corrupted.
    total_rows = len(df)
    statistics["total_rows"] = total_rows

    if total_rows < 100:
        failures.append(f"Check 2 FAIL, Only {total_rows} rows. Minimum required: 100.")
    elif total_rows < 1_000:
        warnings.append(f"Check 2 WARN, Only {total_rows:,} rows. Model quality may suffer.")
    else:
        statistics["row_count"] = "OK"

    # ── Check 3: Null Rates ──────────────────────────────────────────────────
    # Columns with > 50% nulls are essentially unusable as features.
    # Columns with > 20% nulls need special handling (imputation or dropping).
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / total_rows * 100).round(1)
    statistics["total_nulls_by_column"] = null_counts[null_counts > 0].to_dict()

    critical_nulls = null_pcts[null_pcts > 50]
    high_nulls = null_pcts[(null_pcts > 20) & (null_pcts <= 50)]

    for col, pct in critical_nulls.items():
        failures.append(f"Check 3 FAIL, '{col}' is {pct}% null (threshold: 50%)")
    for col, pct in high_nulls.items():
        warnings.append(f"Check 3 WARN, '{col}' is {pct}% null (threshold: 20%)")

    # ── Check 4: Value Ranges ────────────────────────────────────────────────
    # Physically impossible values corrupt model training.
    # We know from domain knowledge what "valid" looks like for taxi data.
    range_checks = [
        ("trip_distance",  lambda s: s < 0,           "negative distance"),
        ("trip_distance",  lambda s: s > 500,          "distance > 500 miles"),
        ("PULocationID",   lambda s: ~s.between(1, 265), "pickup zone ID out of range 1–265"),
        ("DOLocationID",   lambda s: ~s.between(1, 265), "dropoff zone ID out of range 1–265"),
        ("fare_amount",    lambda s: s < -10,          "fare below -$10"),
        ("fare_amount",    lambda s: s > 1000,         "fare above $1000"),
    ]

    range_stats = {}
    for col, condition, label in range_checks:
        if col not in df.columns:
            continue
        bad_count = condition(df[col]).sum()
        range_stats[label] = int(bad_count)
        pct = bad_count / total_rows * 100
        if pct > 1.0:
            warnings.append(
                f"Check 4 WARN, {bad_count:,} rows ({pct:.1f}%) have {label}"
            )

    statistics["value_range_issues"] = range_stats

    # ── Check 5: Target Distribution ────────────────────────────────────────
    # For our regression target (duration_sec): we check it exists and isn't
    # degenerate (all zeros, all the same value, or wildly out of range).
    # Unlike classification, we don't check class balance, we check spread.
    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN].dropna()
        target_stats = {
            "min":    round(float(target.min()), 1),
            "median": round(float(target.median()), 1),
            "max":    round(float(target.max()), 1),
            "std":    round(float(target.std()), 1),
        }
        statistics["target_distribution"] = target_stats

        if target.std() == 0:
            failures.append("Check 5 FAIL, Target 'duration_sec' has zero variance (all same value).")
        if (target <= 0).mean() > 0.5:
            failures.append("Check 5 FAIL, More than 50% of durations are zero or negative.")
        if target.max() > 86_400:
            warnings.append(
                f"Check 5 WARN, Max duration is {target.max()/3600:.1f} hours. Outliers present."
            )
    else:
        # Target not present yet, that's fine before cleaning, just note it
        statistics["target_distribution"] = "Not computed yet (run cleaner.py first)"

    # ── Result ───────────────────────────────────────────────────────────────
    return {
        "success":    len(failures) == 0,
        "failures":   failures,
        "warnings":   warnings,
        "statistics": statistics,
    }


def print_result(result: dict):
    """Pretty-print a quality gate result."""
    print("\n" + "=" * 50)
    print("DATA QUALITY GATE RESULT")
    print("=" * 50)

    status = "PASSED" if result["success"] else "FAILED"
    print(f"\n  Status: {status}")
    print(f"  Rows checked: {result['statistics'].get('total_rows', '?'):,}")

    if result["failures"]:
        print(f"\n  FAILURES ({len(result['failures'])}), must fix before proceeding:")
        for f in result["failures"]:
            print(f"    ✗ {f}")

    if result["warnings"]:
        print(f"\n  WARNINGS ({len(result['warnings'])}), should handle before modeling:")
        for w in result["warnings"]:
            print(f"    ⚠ {w}")

    if not result["failures"] and not result["warnings"]:
        print("\n  All checks passed with no warnings.")

    print("\n  Statistics:")
    for k, v in result["statistics"].items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parents[2]
    DATA_FILE = PROJECT_ROOT / "data" / "tlc" / "yellow" / "yellow_tripdata_2026-01.parquet"

    print(f"Loading {DATA_FILE.name}...")
    df = pd.read_parquet(DATA_FILE)

    result = check_data_quality(df)
    print_result(result)
