# 🚕 NYC Taxi Trip Duration Predictor

[![CI](https://github.com/reddyakilesh22-glitch/nyc-taxi-duration-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/reddyakilesh22-glitch/nyc-taxi-duration-predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-FF4B4B)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6%2B-success)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

> Predicts how long any NYC yellow-taxi trip will take, **before the meter starts** —
> using only pickup zone, dropoff zone, time of day, and distance.

**🔗 Live demo:** _coming soon (Streamlit Community Cloud)_

---

## The Result

| | Baseline (Linear Regression) | **LightGBM (production model)** |
|---|---|---|
| **R²** | 0.589 | **0.978** |
| **Mean Absolute Error** | 6.1 minutes | **~1 minute** |
| **Relative improvement** | — | **83% lower error** |

Trained on **2.69 million cleaned NYC taxi trips** from January 2024 (NYC TLC public data).

---

## What's Inside

A 4-page interactive Streamlit app that walks through the entire ML pipeline:

1. **Project Overview** — KPI cards, tech stack, and the headline result
2. **Explore the Data** — interactive Plotly charts: duration distribution, rush hour patterns, route-type comparisons, correlation matrix
3. **Model Results** — model comparison table, top-15 feature importances, predicted-vs-actual scatter, and a **live prediction form** where you pick a pickup zone, dropoff zone, hour, and day → instant ETA
4. **How I Built This** — architecture diagram, day-by-day timeline, key decisions, and lessons learned

The full project is built around 5 production-grade pieces:

- **Data quality gate** with 5 automated checks (schema, row count, nulls, ranges, target distribution)
- **Feature engineering** producing 32 features across 3 categories — temporal (cyclic sin/cos hour encoding, rush-hour flags), geospatial (Manhattan/airport zone flags), interaction (distance × rush hour, fare-per-mile congestion proxy)
- **Model comparison** with 5-fold cross-validation — Linear Regression, Ridge, LightGBM
- **Hyperparameter tuning** with Optuna Bayesian search (30 trials, MLflow-tracked)
- **Tests + CI** — 8 pytest tests + GitHub Actions running on every push

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data** | Pandas · PyArrow · Parquet |
| **ML** | LightGBM · Scikit-learn · Optuna |
| **Tracking** | MLflow |
| **App** | Streamlit · Plotly |
| **Quality** | Pytest · Ruff · GitHub Actions |
| **Packaging** | Docker · docker-compose |

---

## How to Run

### Option 1 — Docker (recommended)

```bash
docker compose up
```

Open http://localhost:8501. The container mounts `data/` and `models/` from your host, so the live predictor uses the real trained LightGBM model.

### Option 2 — Local Python

```bash
# 1. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2. (Optional) Re-run the full pipeline from raw data
python src/data/cleaner.py            # clean raw TLC parquet → cleaned.parquet
python src/features/run_features.py   # engineer 32 features → features.parquet
python src/models/run_training.py     # train + log all models with MLflow
python src/models/tuning.py           # 30-trial Optuna search (slow: ~3 hours)

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

### Run the test suite

```bash
pytest tests/ -v        # 8 tests
ruff check src/ app/    # lint
```

---

## Project Structure

```
.
├── app/
│   ├── streamlit_app.py         # 4-page interactive dashboard
│   ├── model_results.json       # cached model metrics for the dashboard
│   └── predictions.csv          # test-set predictions for residual plots
├── src/
│   ├── data/
│   │   ├── loader.py            # raw data inspection
│   │   ├── quality.py           # 5-check quality gate
│   │   └── cleaner.py           # 6-step cleaning pipeline
│   ├── features/
│   │   ├── engineering.py       # create_features() + select_features()
│   │   └── run_features.py      # pipeline runner
│   └── models/
│       ├── baseline.py          # LinearRegression baseline
│       ├── compare_models.py    # 5-fold CV comparison
│       ├── run_training.py      # MLflow-tracked training
│       └── tuning.py            # Optuna hyperparameter search
├── tests/                       # 8 pytest tests
├── notebooks/eda.ipynb          # Day 2 EDA notebook
├── notes/learn.md               # plain-English walkthrough of every step
├── Dockerfile · docker-compose.yml
├── .github/workflows/ci.yml     # GitHub Actions: tests + lint
└── requirements.txt
```

---

## The Journey — 6 Days, End to End

| Day | Stage | Key Output |
|---|---|---|
| **1** | Data inspection & quality gate | Removed 9.3% bad rows (negative fares, 300k-mile trips, sub-60s trips) |
| **2** | Exploratory data analysis | Found right-skewed target → log transform; rush hour adds 3–4 min |
| **3** | Feature engineering | 20 raw columns → 32 engineered features (cyclic time, borough flags, interactions) |
| **4** | Model training & tuning | LightGBM hit R²=0.978 / MAE=1min; Optuna ran 30 trials over 3 hours |
| **5** | Interactive dashboard | 4-page Streamlit app with live predictor and 25 popular NYC zones |
| **6** | Production hardening | Dockerized, 8 pytest tests, GitHub Actions CI passing on every push |

A complete plain-English walkthrough — written for someone new to ML — lives in
[`notes/learn.md`](notes/learn.md).

---

## An Honest Finding

Optuna ran 30 hyperparameter trials over **3 hours**. The result:

```
Default LightGBM:  R² = 0.9778,  MAE = 59.16s
Tuned LightGBM:    R² = 0.9774,  MAE = 59.50s
```

Tuning made the model **marginally worse** on the test set. The defaults were already near-optimal for this data. This is a real-world lesson kept in the project rather than hidden: **good defaults + thoughtful features beat blind hyperparameter search almost every time**, and any portfolio that claims otherwise is selectively reporting.

---

## Architecture

```
Raw TLC parquet (2.96M rows)
        │
        ▼
   ┌──────────┐    ┌────────────────┐    ┌─────────────────┐
   │ Cleaning │ → │ Feature        │ → │ Training        │
   │ (9.3%    │   │ engineering    │   │ (Linear, Ridge, │
   │ removed) │   │ (20→32 cols)   │   │ LightGBM)       │
   └──────────┘    └────────────────┘    └─────────────────┘
        │                  │                     │
        │                  │                     ▼
        │                  │           ┌─────────────────┐
        │                  │           │ MLflow tracking │
        │                  │           │ Optuna tuning   │
        │                  │           └─────────────────┘
        │                  │                     │
        │                  ▼                     ▼
        │            features.parquet     production_model.pkl
        │                                        │
        ▼                                        ▼
                       ┌──────────────────────────────┐
                       │ Streamlit app                │
                       │ (4 pages + live predictor)   │
                       └──────────────────────────────┘
                                    │
                                    ▼
                        Docker container · port 8501
```

---

## Data Source

NYC Taxi & Limousine Commission Trip Record Data — January 2024 Yellow Taxi trips
(2,964,624 raw records). Available at
[nyc.gov/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

The TLC data is in the public domain.

---

## License

MIT — feel free to fork, learn from, or build on this project.
