# NYC Taxi Trip Duration Prediction

Predicts yellow taxi trip duration from pickup zone, time of day, and trip distance.
Built as a full-stack ML portfolio project using NYC TLC data (2023–2026).

## Dataset

- **Source:** NYC Taxi & Limousine Commission (TLC) — yellow taxi trip records
- **Size:** ~2.7M clean trips per month, 39 months (2023–2026)
- **Features:** Zone IDs, timestamps, trip distance, fare amounts
- **Target:** Trip duration in seconds (log-transformed for modeling)

## Project Structure

```
src/data/      — data loading, quality gate, cleaning
src/features/  — feature engineering
src/models/    — model training and prediction
app/           — FastAPI API + Streamlit dashboard
notebooks/     — EDA
tests/         — unit tests
```

## Exploratory Data Analysis

Full analysis in [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

**Dataset:** 2,687,584 clean trips (January 2024) across 20 columns — timestamps,
zone IDs (1–263), distances, and fare components.

**Key findings:**

- **Duration is right-skewed** — most trips are 5–20 minutes but the tail extends to 3 hours.
  We use `log(duration)` as the model target to prevent outliers from dominating training.

- **Fare amount correlates strongly with duration** (r ≈ 0.7) — the meter runs on time,
  so fare and duration move together. Both fare and distance carry independent signal.

- **Hour of day is a strong predictor** — rush hour trips (8am, 5–6pm) average 3–4 minutes
  longer than off-peak. Hour of day will be a top feature in the model.

- **Zone IDs have low linear correlation with duration** — but are still valuable.
  LightGBM handles their non-linear patterns (261 categorical values) natively.

- **No nulls after cleaning** — 9.3% of raw rows removed; remaining 2.69M are complete.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quality gate on raw data
python src/data/quality.py

# 3. Clean data
python src/data/cleaner.py
```
