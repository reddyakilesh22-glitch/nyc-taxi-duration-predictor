"""
Day 4 — Hyperparameter Tuning with Optuna

LightGBM has many knobs. Wrong settings = underfit or overfit model.
Optuna tries 30 combinations automatically using Bayesian search —
smarter than random search because each trial learns from the last.

Key hyperparameters tuned:
  num_leaves     — complexity of each tree (higher = more complex, more overfit risk)
  learning_rate  — step size when learning (lower = more careful, needs more trees)
  min_child_samples — minimum trips per tree leaf (higher = smoother, less overfit)
  feature_fraction  — fraction of features used per tree (prevents feature dominance)

Usage:
    python src/models/tuning.py
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence per-trial logs

PROJECT_ROOT  = Path(__file__).parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2024-01_features.parquet"
MODEL_DIR     = PROJECT_ROOT / "models"

DROP_COLS = ["duration_sec", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "store_and_fwd_flag"]
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID",
                        "VendorID", "payment_type", "day_of_week"]


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = np.log1p(df["duration_sec"])
    return X, y


def objective(trial, X_train, y_train):
    """
    Optuna calls this function 30 times, each time with different hyperparameters.
    It returns the CV score. Optuna uses past results to choose smarter next params.
    """
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 1000),
        "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples":  trial.suggest_int("min_child_samples", 50, 300),
        "feature_fraction":   trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":       trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1":          trial.suggest_float("lambda_l1", 1e-4, 1.0, log=True),
        "lambda_l2":          trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # n_jobs=1 here to avoid nested parallelism crash on macOS (LightGBM already uses all cores internally)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
    return scores.mean()


def main():
    print("Loading features...")
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Sample for tuning — tune on 300k rows, train final model on full data.
    # This is standard practice: hyperparameters that work on a representative
    # sample generalise to the full dataset, and tuning is 7x faster.
    TUNE_SAMPLE = 300_000
    idx = np.random.default_rng(42).choice(len(X_train), TUNE_SAMPLE, replace=False)
    X_tune = X_train.iloc[idx]
    y_tune = y_train.iloc[idx]
    print(f"  Tuning sample: {len(X_tune):,} rows (full train used for final fit)")

    # ── Optuna Search ────────────────────────────────────────────────────────
    print("\nRunning Optuna hyperparameter search (30 trials)...")
    print("  Each trial = one set of hyperparameters evaluated with 5-fold CV")
    t0 = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_tune, y_tune),
        n_trials=30,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\nSearch complete in {elapsed/60:.1f} minutes")
    print(f"  Best CV R²  : {study.best_value:.4f}")
    print("  Best params :")
    for k, v in study.best_params.items():
        print(f"    {k:<25} {v}")

    # Save best params
    params_path = MODEL_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n  Saved best params → {params_path.name}")

    # ── Train Final Model With Best Params ───────────────────────────────────
    print("\nTraining final model with best hyperparameters...")
    best_params = {**study.best_params, "verbose": -1, "n_jobs": -1}
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    preds_log  = final_model.predict(X_test)
    preds_sec  = np.expm1(preds_log)
    actual_sec = np.expm1(y_test)

    test_r2   = r2_score(y_test, preds_log)
    test_mae  = mean_absolute_error(actual_sec, preds_sec)
    test_rmse = mean_squared_error(y_test, preds_log) ** 0.5

    print(f"\n{'='*45}")
    print("  Tuned LightGBM — Final Test Results")
    print(f"{'='*45}")
    print(f"  MAE  (seconds) : {test_mae:.1f}s = {test_mae/60:.1f} min")
    print(f"  RMSE (log)     : {test_rmse:.4f}")
    print(f"  R²   (log)     : {test_r2:.4f}")
    print(f"{'='*45}")
    print("\n  Baseline was: MAE=363.8s (6.1min), R²=0.5890")
    improvement = (test_mae - 363.8) / 363.8 * 100
    print(f"  MAE change   : {improvement:+.1f}%")

    # Save tuned model
    model_path = MODEL_DIR / "tuned_model.pkl"
    joblib.dump({
        "model":        final_model,
        "params":       best_params,
        "metrics":      {"r2": test_r2, "mae_sec": test_mae, "rmse_log": test_rmse},
        "feature_cols": list(X_train.columns),
    }, model_path)
    print(f"\n  Saved → {model_path.name}")


if __name__ == "__main__":
    main()
