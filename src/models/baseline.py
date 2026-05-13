"""
Day 4: Baseline Model

The simplest possible model: LinearRegression.
This gives us a "floor", every model we build after this must beat it,
otherwise the added complexity isn't worth it.

Key decision: we train on log1p(duration_sec) because duration is right-skewed
(EDA Day 2). The model predicts log-duration; we convert back to seconds for
human-readable metrics.

Usage:
    python src/models/baseline.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2026-01_features.parquet"
MODEL_DIR     = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Columns to drop, not features
DROP_COLS = ["duration_sec", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "store_and_fwd_flag"]

TARGET_COL = "duration_sec"


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load feature file and return X (features) and y (log-transformed target)."""
    df = pd.read_parquet(FEATURES_PATH)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = np.log1p(df[TARGET_COL])   # log1p(x) = log(1 + x), safe when x=0

    return X, y


def evaluate(model, X_test, y_test_log, label="Model"):
    """
    Print MAE, RMSE, R² on both log scale and real seconds.

    Log-scale metrics: what the model optimises internally.
    Seconds metrics: what actually matters to a human.
    """
    preds_log = model.predict(X_test)
    preds_sec = np.expm1(preds_log)          # reverse the log1p
    actual_sec = np.expm1(y_test_log)

    mae_sec  = mean_absolute_error(actual_sec, preds_sec)
    rmse_log = mean_squared_error(y_test_log, preds_log) ** 0.5
    r2       = r2_score(y_test_log, preds_log)

    print(f"\n{'='*45}")
    print(f"  {label}, Test Set Results")
    print(f"{'='*45}")
    print(f"  MAE   (seconds) : {mae_sec:>8.1f}s  = {mae_sec/60:.1f} min")
    print(f"  RMSE  (log)     : {rmse_log:>8.4f}")
    print(f"  R²    (log)     : {r2:>8.4f}   (1.0 = perfect)")
    print(f"{'='*45}")

    return {"mae_sec": mae_sec, "rmse_log": rmse_log, "r2": r2}


def main():
    print("Loading features...")
    X, y = load_features()
    print(f"  {X.shape[0]:,} rows × {X.shape[1]} features")

    # 80/20 split, random, not temporal, for comparability with guide
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train
    print("\nTraining LinearRegression baseline...")
    t0 = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Evaluate
    metrics = evaluate(model, X_test, y_test, label="LinearRegression (Baseline)")

    # Top 5 most influential features (by absolute coefficient)
    coef_df = pd.Series(model.coef_, index=X.columns)
    print("\nTop 10 most influential features:")
    print(coef_df.abs().sort_values(ascending=False).head(10).round(4).to_string())

    # Save
    model_path = MODEL_DIR / "baseline.pkl"
    joblib.dump({"model": model, "metrics": metrics, "feature_cols": list(X.columns)},
                model_path)
    print(f"\nSaved → {model_path.name}")
    print("\nBaseline scores to beat:")
    print(f"  MAE  : {metrics['mae_sec']:.1f}s ({metrics['mae_sec']/60:.1f} min)")
    print(f"  RMSE : {metrics['rmse_log']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
