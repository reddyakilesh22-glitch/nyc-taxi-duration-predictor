"""
Day 4 — Model Comparison

We compare 3 models with reasoning, not just picking the best number.
Each model is trained with 5-fold cross-validation so we know how
consistent the score is, not just how lucky a single split was.

Models compared:
  1. LinearRegression  — baseline, no assumptions, good interpretability
  2. Ridge             — LinearRegression + L2 regularization, handles correlated features
  3. LightGBM          — gradient boosted trees, captures non-linear patterns,
                         handles categoricals natively, scales to millions of rows

Why these three:
  - Linear models are fast and interpretable but assume relationships are straight lines.
    Rush hour doesn't linearly add time — it compounds. Trees capture that.
  - LightGBM is the industry standard for tabular regression at scale.
    It should win here. If it doesn't, our features might have an issue.

Usage:
    python src/models/compare_models.py
"""

import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

PROJECT_ROOT  = Path(__file__).parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2024-01_features.parquet"
MODEL_DIR     = PROJECT_ROOT / "models"

DROP_COLS = ["duration_sec", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "store_and_fwd_flag"]

# LightGBM handles these better as categorical rather than numeric
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID",
                        "VendorID", "payment_type", "day_of_week"]


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = np.log1p(df["duration_sec"])
    return X, y


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, cv_folds=5):
    """
    Train a model, run 5-fold CV, evaluate on test set, return results row.

    5-fold CV means: split training data into 5 chunks, train on 4, validate
    on 1, repeat 5 times. The average score is more reliable than one split.
    """
    print(f"\nTraining {name}...")
    t0 = time.time()

    # Cross-validation on training set
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=kf, scoring="r2", n_jobs=-1)

    # Final fit on full training set
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    # Test set metrics
    preds_log = model.predict(X_test)
    preds_sec = np.expm1(preds_log)
    actual_sec = np.expm1(y_test)

    test_r2   = r2_score(y_test, preds_log)
    test_mae  = mean_absolute_error(actual_sec, preds_sec)
    test_rmse = mean_squared_error(y_test, preds_log) ** 0.5

    print(f"  CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test  : R²={test_r2:.4f}  MAE={test_mae:.0f}s ({test_mae/60:.1f}min)  "
          f"RMSE={test_rmse:.4f}  Time={elapsed:.1f}s")

    return {
        "Model":         name,
        "CV R² Mean":    round(cv_scores.mean(), 4),
        "CV R² Std":     round(cv_scores.std(), 4),
        "Test R²":       round(test_r2, 4),
        "Test MAE (min)":round(test_mae / 60, 2),
        "Train Time (s)":round(elapsed, 1),
    }, model


def main():
    print("Loading features...")
    X, y = load_features()

    # Cast categoricals for LightGBM
    X_lgb = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_lgb.columns:
            X_lgb[col] = X_lgb[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_lgb = X_lgb.iloc[X_train.index]
    X_test_lgb  = X_lgb.iloc[X_test.index]

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print("\nWhy these 3 models:")
    print("  LinearRegression — baseline, fast, interpretable, assumes linear relationships")
    print("  Ridge            — adds L2 penalty to prevent overfitting on correlated features")
    print("  LightGBM         — gradient boosted trees, captures non-linear patterns,")
    print("                     native categoricals, industry standard for tabular data")

    results = []
    trained_models = {}

    # 1. Linear Regression
    row, m = train_and_evaluate(
        "LinearRegression", LinearRegression(),
        X_train, X_test, y_train, y_test
    )
    results.append(row); trained_models["LinearRegression"] = (m, X_train.columns.tolist())

    # 2. Ridge
    row, m = train_and_evaluate(
        "Ridge (α=1.0)", Ridge(alpha=1.0),
        X_train, X_test, y_train, y_test
    )
    results.append(row); trained_models["Ridge"] = (m, X_train.columns.tolist())

    # 3. LightGBM
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        n_jobs=-1,
    )
    row, m = train_and_evaluate(
        "LightGBM", lgbm,
        X_train_lgb, X_test_lgb, y_train, y_test
    )
    results.append(row); trained_models["LightGBM"] = (m, X_train_lgb.columns.tolist())

    # Comparison table
    print(f"\n{'='*75}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*75}")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Winner
    best = df_results.loc[df_results["Test R²"].idxmax(), "Model"]
    best_r2  = df_results["Test R²"].max()
    best_mae = df_results.loc[df_results["Test R²"].idxmax(), "Test MAE (min)"]
    print(f"\n  Best model: {best}")
    print(f"  Test R²: {best_r2}  |  MAE: {best_mae} min")
    print(f"\n  Why {best} wins:")
    print("  LightGBM captures non-linear patterns (rush hour compounds distance,")
    print("  zone IDs have complex spatial patterns) that linear models cannot.")

    # Save all models
    for name, (model, cols) in trained_models.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") \
                    .replace("α=", "alpha") + ".pkl"
        joblib.dump({"model": model, "feature_cols": cols}, MODEL_DIR / fname)
        print(f"  Saved → {fname}")

    return df_results


if __name__ == "__main__":
    main()
