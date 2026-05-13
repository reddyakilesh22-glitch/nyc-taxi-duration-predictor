"""
Day 4 — Full Training Pipeline with MLflow Experiment Tracking

MLflow logs every model run — params, metrics, and model artifacts.
This lets you compare runs in a visual UI at http://localhost:5000
without losing track of what you tried.

To start the MLflow UI (in a separate terminal):
    mlflow server --host 127.0.0.1 --port 5000

Then run this script:
    python src/models/run_training.py

Navigate to http://localhost:5000 to see all logged runs.
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT  = Path(__file__).parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2026-01_features.parquet"
MODEL_DIR     = PROJECT_ROOT / "models"

DROP_COLS = ["duration_sec", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "store_and_fwd_flag"]
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID",
                        "VendorID", "payment_type", "day_of_week"]

MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME     = "nyc-taxi-duration"


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = np.log1p(df["duration_sec"])
    return X, y


def compute_metrics(model, X_test, y_test_log):
    preds_log  = model.predict(X_test)
    preds_sec  = np.expm1(preds_log)
    actual_sec = np.expm1(y_test_log)
    return {
        "r2":       round(r2_score(y_test_log, preds_log), 4),
        "mae_sec":  round(mean_absolute_error(actual_sec, preds_sec), 2),
        "rmse_log": round(mean_squared_error(y_test_log, preds_log) ** 0.5, 4),
    }


def run_model(name, model, params, X_train, X_test, y_train, y_test, feature_cols):
    """Train a model and log everything to MLflow."""
    print(f"\n── Running: {name} ──")

    with mlflow.start_run(run_name=name):
        # Log hyperparameters
        mlflow.log_param("model_name", name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        # Metrics on train and test
        train_metrics = compute_metrics(model, X_train, y_train)
        test_metrics  = compute_metrics(model, X_test,  y_test)

        # Log metrics
        mlflow.log_metric("train_r2",       train_metrics["r2"])
        mlflow.log_metric("train_mae_sec",  train_metrics["mae_sec"])
        mlflow.log_metric("test_r2",        test_metrics["r2"])
        mlflow.log_metric("test_mae_sec",   test_metrics["mae_sec"])
        mlflow.log_metric("test_rmse_log",  test_metrics["rmse_log"])
        mlflow.log_metric("train_time_sec", train_time)

        # Save and log model artifact
        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump({"model": model, "params": params,
                     "metrics": test_metrics, "feature_cols": feature_cols},
                    model_path)
        mlflow.log_artifact(str(model_path))

        print(f"  Train R²: {train_metrics['r2']:.4f}  |  "
              f"Test R²: {test_metrics['r2']:.4f}  |  "
              f"MAE: {test_metrics['mae_sec']:.0f}s ({test_metrics['mae_sec']/60:.1f}min)  |  "
              f"Time: {train_time}s")

    return test_metrics


def main():
    # Point MLflow at a local directory (no server needed to log, only to view UI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment          : {EXPERIMENT_NAME}")
    print("\nTo view runs: mlflow server --host 127.0.0.1 --port 5000")
    print("Then open   : http://localhost:5000\n")

    print("Loading features...")
    X, y = load_features()

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

    all_results = []

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    params = {"model_type": "LinearRegression"}
    metrics = run_model("baseline", LinearRegression(), params,
                        X_train, X_test, y_train, y_test, list(X_train.columns))
    all_results.append({"model": "baseline", **metrics})

    # ── 2. Decision Tree ─────────────────────────────────────────────────────
    # A single constrained tree. Shows the contrast between "no non-linearities"
    # (linear regression), "non-linearities via a single tree", and "ensemble of
    # trees" (LightGBM). max_depth and min_samples_leaf prevent the tree from
    # memorising the 1.9M training rows.
    params = {"model_type": "DecisionTree", "max_depth": 12, "min_samples_leaf": 200}
    metrics = run_model(
        "decision_tree",
        DecisionTreeRegressor(max_depth=12, min_samples_leaf=200, random_state=42),
        params,
        X_train, X_test, y_train, y_test, list(X_train.columns),
    )
    all_results.append({"model": "decision_tree", **metrics})

    # ── 3. LightGBM (default params) ─────────────────────────────────────────
    lgbm_params = {
        "model_type": "LightGBM", "n_estimators": 500,
        "learning_rate": 0.05, "num_leaves": 127,
        "min_child_samples": 100, "feature_fraction": 0.8,
    }
    lgbm_model = lgb.LGBMRegressor(
        **{k: v for k, v in lgbm_params.items() if k != "model_type"},
        verbose=-1, n_jobs=-1
    )
    metrics = run_model("lightgbm_default", lgbm_model, lgbm_params,
                        X_train_lgb, X_test_lgb, y_train, y_test,
                        list(X_train_lgb.columns))
    all_results.append({"model": "lightgbm_default", **metrics})

    # ── 4. LightGBM (tuned params, if tuning.py has been run) ────────────────
    best_params_path = MODEL_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            tuned_params = json.load(f)
        tuned_model = lgb.LGBMRegressor(**tuned_params, verbose=-1, n_jobs=-1)
        log_params = {"model_type": "LightGBM_tuned", **tuned_params}
        metrics = run_model("lightgbm_tuned", tuned_model, log_params,
                            X_train_lgb, X_test_lgb, y_train, y_test,
                            list(X_train_lgb.columns))
        all_results.append({"model": "lightgbm_tuned", **metrics})

        # Save as production model
        prod_path = MODEL_DIR / "production_model.pkl"
        joblib.dump({"model": tuned_model, "params": tuned_params,
                     "metrics": metrics, "feature_cols": list(X_train_lgb.columns)},
                    prod_path)
        print(f"\nProduction model saved → {prod_path.name}")
    else:
        print("\nNote: run tuning.py first to include tuned LightGBM")
        # Save default LightGBM as production for now
        prod_path = MODEL_DIR / "production_model.pkl"
        joblib.dump({"model": lgbm_model, "params": lgbm_params,
                     "metrics": all_results[-1],
                     "feature_cols": list(X_train_lgb.columns)},
                    prod_path)
        print(f"Production model saved → {prod_path.name} (default LightGBM)")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    df_res = pd.DataFrame(all_results)
    print(df_res.sort_values("r2", ascending=False).to_string(index=False))
    print("\nAll runs logged to MLflow — run:")
    print("  mlflow server --host 127.0.0.1 --port 5000")
    print("  open http://localhost:5000")


if __name__ == "__main__":
    main()
