"""
Day 5 — Interactive Streamlit Portfolio App

NYC Taxi Trip Duration Predictor
A 4-page portfolio site showcasing the end-to-end ML pipeline.

Run:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent.parent

TAXI_YELLOW  = "#F7C948"
DARK_NAVY    = "#1a1a2e"
CHARCOAL     = "#2C3E50"
GREEN        = "#27AE60"
RED          = "#E74C3C"
LIGHT_GREY   = "#F8F9FA"
PLOTLY_THEME = "plotly_white"

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a2e;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* KPI cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-left: 5px solid #F7C948;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.kpi-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 32px; font-weight: 800; color: #1a1a2e; line-height: 1.1; }
.kpi-delta { font-size: 13px; color: #27AE60; font-weight: 600; margin-top: 2px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 48px 40px;
    border-radius: 12px;
    margin-bottom: 28px;
}
.hero h1 { font-size: 48px; font-weight: 900; margin: 0; line-height: 1.1; }
.hero .subtitle { font-size: 20px; color: #F7C948; margin-top: 12px; font-weight: 500; }
.hero .desc { font-size: 16px; color: #b0b8d0; margin-top: 16px; max-width: 600px; line-height: 1.6; }

/* Tech badges */
.badge {
    display: inline-block;
    background: #1a1a2e;
    color: #F7C948;
    border: 1px solid #F7C948;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px;
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 800;
    color: #1a1a2e;
    border-bottom: 3px solid #F7C948;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* Callout boxes */
.callout {
    background: #fffbea;
    border-left: 4px solid #F7C948;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.callout-title { font-weight: 700; color: #1a1a2e; font-size: 14px; margin-bottom: 4px; }
.callout-body  { color: #444; font-size: 14px; line-height: 1.5; }

/* Winner badge */
.winner-badge {
    background: #F7C948;
    color: #1a1a2e;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 700;
}

/* Prediction result */
.pred-result {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: white;
    padding: 28px;
    border-radius: 12px;
    text-align: center;
}
.pred-result .time { font-size: 56px; font-weight: 900; color: #F7C948; }
.pred-result .label { font-size: 16px; color: #b0b8d0; margin-top: 4px; }

/* Footer */
.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    padding: 24px 0 8px;
    border-top: 1px solid #eee;
    margin-top: 40px;
}

/* Timeline */
.timeline-item {
    border-left: 3px solid #F7C948;
    padding: 8px 0 8px 20px;
    margin-bottom: 16px;
    position: relative;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -7px;
    top: 12px;
    width: 12px;
    height: 12px;
    background: #F7C948;
    border-radius: 50%;
}
.timeline-day { font-size: 12px; color: #888; font-weight: 600; text-transform: uppercase; }
.timeline-title { font-size: 16px; font-weight: 700; color: #1a1a2e; }
.timeline-desc { font-size: 14px; color: #555; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ── NYC Zone Map ──────────────────────────────────────────────────────────────
AIRPORT_ZONES   = {132, 138, 1}
MANHATTAN_ZONES = set(range(4, 153)) | {161, 162, 163, 164, 166, 170, 186,
                                         194, 202, 209, 211, 224, 229, 230,
                                         231, 232, 233, 234, 236, 237, 238,
                                         239, 243, 244, 246, 249, 261, 262}

POPULAR_ZONES = {
    "Midtown Center":              161,
    "Times Square / Theatre Dist": 230,
    "JFK Airport":                 132,
    "LaGuardia Airport":           138,
    "Penn Station / Madison Sq W": 186,
    "Grand Central":               234,
    "Financial District North":     87,
    "Financial District South":     88,
    "Upper East Side North":       236,
    "Upper East Side South":       237,
    "Upper West Side North":       238,
    "Upper West Side South":       239,
    "Hell's Kitchen North":        113,
    "Midtown East":                162,
    "East Village":                 79,
    "Greenwich Village North":     103,
    "Soho":                        211,
    "Tribeca / Civic Center":      246,
    "Astoria (Queens)":            9,
    "Williamsburg (Brooklyn)":     261,
    "Crown Heights (Brooklyn)":    61,
    "Harlem":                      74,
    "Washington Heights":          152,
    "Lower East Side":             148,
    "Murray Hill":                 170,
}
ZONE_NAMES = list(POPULAR_ZONES.keys())

# ── Data Loading ──────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent

@st.cache_data
def load_model_results():
    path = APP_DIR / "model_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _demo_model_results()

@st.cache_data
def load_predictions():
    path = APP_DIR / "predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return _demo_predictions()

@st.cache_data
def load_feature_sample(n=20_000):
    path = PROJECT_ROOT / "data" / "features" / "yellow_tripdata_2024-01_features.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=[
            "duration_sec", "trip_distance", "hour", "day_of_week",
            "fare_per_mile", "is_rush_hour", "is_weekend",
            "is_pu_manhattan", "is_airport_trip",
        ])
        return df.sample(n=min(n, len(df)), random_state=42)
    return _demo_feature_sample(n)

@st.cache_resource
def load_model():
    path = PROJECT_ROOT / "models" / "production_model.pkl"
    if path.exists():
        return joblib.load(path)
    return None

def _demo_model_results():
    rng = np.random.default_rng(42)
    return {
        "models": [
            {"name": "Linear Regression", "r2": 0.589, "mae_sec": 363.75, "mae_min": 6.1, "rmse_log": 0.446, "description": "Baseline — straight-line relationships only", "winner": False},
            {"name": "Ridge Regression",  "r2": 0.591, "mae_sec": 361.2,  "mae_min": 6.0, "rmse_log": 0.444, "description": "Linear + L2 regularization", "winner": False},
            {"name": "LightGBM (default)","r2": 0.978, "mae_sec": 59.16,  "mae_min": 1.0, "rmse_log": 0.104, "description": "Gradient boosted trees", "winner": False},
            {"name": "LightGBM (tuned)",  "r2": 0.981, "mae_sec": 54.0,   "mae_min": 0.9, "rmse_log": 0.098, "description": "Tuned with 30 Optuna trials", "winner": True},
        ],
        "feature_importances": {"DOLocationID": 13625, "PULocationID": 11013, "fare_per_mile": 9392,
                                  "trip_distance": 7994, "tip_amount": 4543, "hour": 1933},
        "stats": {"total_trips": 2687584, "n_features": 32, "best_r2": 0.981, "best_mae_min": 0.9, "baseline_mae_min": 6.1},
    }

def _demo_predictions():
    rng = np.random.default_rng(42)
    n = 5000
    actual = rng.exponential(scale=700, size=n).clip(60, 10800)
    noise  = rng.normal(0, 50, size=n)
    predicted = (actual + noise).clip(60, 10800)
    return pd.DataFrame({
        "actual_sec": actual, "predicted_sec": predicted,
        "actual_min": actual/60, "predicted_min": predicted/60,
        "error_sec": predicted - actual,
        "trip_distance": rng.exponential(2.5, n).clip(0.1, 30),
        "hour": rng.integers(0, 24, n),
        "is_rush_hour": rng.integers(0, 2, n),
    })

def _demo_feature_sample(n):
    rng = np.random.default_rng(42)
    duration = rng.exponential(700, n).clip(60, 10800)
    hour     = rng.integers(0, 24, n)
    return pd.DataFrame({
        "duration_sec": duration, "trip_distance": rng.exponential(2.5, n).clip(0.1,30),
        "hour": hour, "day_of_week": rng.integers(0, 7, n),
        "fare_per_mile": rng.uniform(2, 12, n),
        "is_rush_hour": ((hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 19)).astype(int),
        "is_weekend": rng.integers(0, 2, n), "is_pu_manhattan": rng.integers(0, 2, n),
        "is_airport_trip": rng.integers(0, 2, n),
    })


# ── Prediction Helper ────────────────────────────────────────────────────────
def build_input_row(pu_id, do_id, distance, hour, dow, passengers, fare_est):
    is_weekday   = int(dow < 5)
    is_am_rush   = int(is_weekday and 7 <= hour <= 9)
    is_pm_rush   = int(is_weekday and 16 <= hour <= 19)
    is_rush      = int(is_am_rush or is_pm_rush)
    is_weekend   = int(dow >= 5)
    is_late_night= int(hour >= 22 or hour <= 5)
    is_pu_man    = int(pu_id in MANHATTAN_ZONES)
    is_do_man    = int(do_id in MANHATTAN_ZONES)
    is_airport   = int(pu_id in AIRPORT_ZONES or do_id in AIRPORT_ZONES)
    is_same      = int(pu_id == do_id)
    is_both_man  = int(is_pu_man and is_do_man)
    fare_per_mile= fare_est / max(distance, 0.1)

    return {
        "VendorID":            2,
        "passenger_count":     passengers,
        "trip_distance":       distance,
        "RatecodeID":          1,
        "PULocationID":        pu_id,
        "DOLocationID":        do_id,
        "payment_type":        1,
        "extra":               1.0,
        "mta_tax":             0.5,
        "tip_amount":          2.85,
        "tolls_amount":        0.0,
        "congestion_surcharge":2.5,
        "Airport_fee":         0.0 if not is_airport else 1.75,
        "hour":                hour,
        "day_of_week":         dow,
        "hour_sin":            np.sin(2 * np.pi * hour / 24),
        "hour_cos":            np.cos(2 * np.pi * hour / 24),
        "dow_sin":             np.sin(2 * np.pi * dow / 7),
        "dow_cos":             np.cos(2 * np.pi * dow / 7),
        "is_am_rush":          is_am_rush,
        "is_pm_rush":          is_pm_rush,
        "is_rush_hour":        is_rush,
        "is_weekend":          is_weekend,
        "is_late_night":       is_late_night,
        "is_pu_manhattan":     is_pu_man,
        "is_do_manhattan":     is_do_man,
        "is_airport_trip":     is_airport,
        "is_same_zone":        is_same,
        "is_both_manhattan":   is_both_man,
        "fare_per_mile":       fare_per_mile,
        "distance_x_rush":     distance * is_rush,
        "distance_x_night":    distance * is_late_night,
    }


# ── Pages ────────────────────────────────────────────────────────────────────
def page_overview():
    results = load_model_results()
    stats   = results["stats"]

    # Hero
    st.markdown("""
    <div class="hero">
        <div style="font-size:14px; color:#F7C948; font-weight:700; letter-spacing:2px; margin-bottom:8px;">
            ML PORTFOLIO PROJECT
        </div>
        <h1>🚕 NYC Taxi Trip<br>Duration Predictor</h1>
        <div class="subtitle">Predicting how long your ride will take — before it starts</div>
        <div class="desc">
            An end-to-end machine learning pipeline trained on <strong>2.69 million NYC yellow taxi trips</strong>.
            The model predicts trip duration to within <strong>~1 minute on average</strong>,
            an 83% improvement over a simple baseline — using only information available at the moment of pickup.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    st.markdown('<div class="section-header">Key Numbers</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Trips Analyzed</div>
            <div class="kpi-value">2.69M</div>
            <div class="kpi-delta">January 2024 NYC TLC data</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Features Engineered</div>
            <div class="kpi-value">{stats['n_features']}</div>
            <div class="kpi-delta">from 15 raw columns</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Accuracy (R²)</div>
            <div class="kpi-value">{stats['best_r2']:.3f}</div>
            <div class="kpi-delta">explains 98.1% of trip variation</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        improvement = (stats['baseline_mae_min'] - stats['best_mae_min']) / stats['baseline_mae_min'] * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Error (MAE)</div>
            <div class="kpi-value">~1 min</div>
            <div class="kpi-delta">↑ {improvement:.0f}% better than baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # What this does + Tech stack
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">What This Project Does</div>', unsafe_allow_html=True)
        st.markdown("""
        This project answers a question every NYC taxi passenger asks: **"How long will this take?"**

        Starting from raw TLC trip records, the pipeline cleans 2.7M trips, engineers 32 domain-aware
        features (rush hour flags, borough signals, cyclic time encoding), and trains a LightGBM model
        that predicts trip duration before the taxi moves.

        The result: an average error of **~1 minute** — good enough to set accurate ETA expectations,
        optimize dispatch routing, and power real-time fare estimates.
        """)

        st.markdown('<br><div class="section-header">Pipeline Overview</div>', unsafe_allow_html=True)
        pipeline_data = pd.DataFrame({
            "Stage": ["Data Cleaning", "Feature Engineering", "Model Training", "Hyperparameter Tuning"],
            "Input": ["2.96M raw rows", "2.69M clean rows", "32 features", "LightGBM model"],
            "Output": ["2.69M clean rows", "32 engineered features", "3 models compared", "Best params (30 trials)"],
            "Key Result": ["Removed 9.3% bad data", "20 → 32 features", "R² 0.589 → 0.978", "R² 0.978 → 0.981"],
        })
        st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
        tech = {
            "Data": ["Pandas", "PyArrow", "Parquet"],
            "ML":   ["LightGBM", "Scikit-learn", "Optuna"],
            "Tracking": ["MLflow"],
            "App": ["Streamlit", "Plotly"],
        }
        for category, tools in tech.items():
            st.markdown(f"**{category}**")
            badges = " ".join(f'<span class="badge">{t}</span>' for t in tools)
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("")

        st.markdown('<br><div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=["Baseline\n(Linear)", "Ridge", "LightGBM\n(default)", "LightGBM\n(tuned)"],
            y=[6.1, 6.0, 1.0, 0.9],
            marker_color=[RED, RED, TAXI_YELLOW, GREEN],
            text=["6.1 min", "6.0 min", "1.0 min", "0.9 min"],
            textposition="outside",
        ))
        fig.update_layout(
            title="MAE by Model (lower = better)",
            yaxis_title="Mean Absolute Error (minutes)",
            template=PLOTLY_THEME,
            height=260,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    _footer()


def page_explore():
    df = load_feature_sample()

    st.markdown('<h2 style="color:#1a1a2e;">📊 Explore the Data</h2>', unsafe_allow_html=True)
    st.caption("Based on 20,000 randomly sampled trips from January 2024")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🕐 Duration Distribution", "⏰ Time Patterns", "🗺️ Location Insights", "🔗 Correlations"
    ])

    with tab1:
        st.markdown('<div class="section-header">Trip Duration Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            log_toggle = st.checkbox("Use log scale (log₁ₓ duration)", value=True,
                                     help="Log scale is what the model actually trains on — compresses the long tail")
            durations = np.log1p(df["duration_sec"]) if log_toggle else df["duration_sec"] / 60

            fig = px.histogram(
                x=durations,
                nbins=80,
                color_discrete_sequence=[TAXI_YELLOW],
                labels={"x": "log(duration)" if log_toggle else "Duration (minutes)"},
                title="Trip Duration Distribution" + (" — Log Scale" if log_toggle else " — Raw (minutes)"),
            )
            fig.update_layout(template=PLOTLY_THEME, height=380, showlegend=False,
                              margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Key Stats</div>', unsafe_allow_html=True)
            dur_min = df["duration_sec"] / 60
            st.metric("Median Trip", f"{dur_min.median():.1f} min")
            st.metric("Mean Trip",   f"{dur_min.mean():.1f} min")
            st.metric("Shortest",    f"{dur_min.min():.1f} min")
            st.metric("Longest",     f"{dur_min.max():.0f} min")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="callout">
                <div class="callout-title">Why Log Scale?</div>
                <div class="callout-body">
                    The raw distribution is right-skewed — a few 3-hour trips
                    would distort training. Log transform compresses the tail
                    so the model focuses on typical trips, not rare outliers.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">How Time Affects Trip Duration</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            avg_by_hour = df.groupby("hour")["duration_sec"].mean().div(60).reset_index()
            avg_by_hour.columns = ["Hour", "Avg Duration (min)"]

            fig = px.line(
                avg_by_hour, x="Hour", y="Avg Duration (min)",
                title="Average Duration by Hour of Day",
                markers=True,
                color_discrete_sequence=[DARK_NAVY],
            )
            fig.add_vrect(x0=7, x1=9,   fillcolor=TAXI_YELLOW, opacity=0.15, annotation_text="AM Rush")
            fig.add_vrect(x0=16, x1=19, fillcolor=TAXI_YELLOW, opacity=0.15, annotation_text="PM Rush")
            fig.update_layout(template=PLOTLY_THEME, height=340, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="callout">
                <div class="callout-title">Rush Hour adds 3–4 minutes</div>
                <div class="callout-body">
                    The same route takes 3–4 minutes longer at 8am vs 3am.
                    This is why hour is the 6th most important feature in the model.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            avg_by_dow = df.groupby("day_of_week")["duration_sec"].mean().div(60).reset_index()
            avg_by_dow["Day"] = avg_by_dow["day_of_week"].map(dict(enumerate(day_labels)))

            fig = px.bar(
                avg_by_dow, x="Day", y="duration_sec",
                title="Average Duration by Day of Week",
                color="duration_sec",
                color_continuous_scale=[[0, TAXI_YELLOW], [1, DARK_NAVY]],
                labels={"duration_sec": "Avg Duration (min)"},
            )
            fig.update_layout(template=PLOTLY_THEME, height=340, margin=dict(t=50, b=20),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="callout">
                <div class="callout-title">Weekdays run longer</div>
                <div class="callout-body">
                    Friday PM has the longest average trips. Weekends
                    are shorter — less commuter traffic, more leisure rides
                    close to home.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">Location and Route Type Effects</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            route_types = []
            for _, row in df.iterrows():
                if row.get("is_airport_trip", 0):
                    route_types.append("Airport Trip")
                elif row.get("is_pu_manhattan", 0):
                    route_types.append("Manhattan Pickup")
                else:
                    route_types.append("Other Boroughs")
            df_plot = df.copy()
            df_plot["Route Type"] = route_types
            df_plot["duration_min"] = df_plot["duration_sec"] / 60

            fig = px.box(
                df_plot.sample(min(5000, len(df_plot)), random_state=42),
                x="Route Type", y="duration_min",
                color="Route Type",
                title="Duration by Route Type",
                color_discrete_map={
                    "Airport Trip": GREEN,
                    "Manhattan Pickup": TAXI_YELLOW,
                    "Other Boroughs": DARK_NAVY,
                },
                labels={"duration_min": "Duration (minutes)"},
            )
            fig.update_layout(template=PLOTLY_THEME, height=360, margin=dict(t=50, b=20),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_plot2 = df.sample(min(3000, len(df)), random_state=42).copy()
            df_plot2["duration_min"] = df_plot2["duration_sec"] / 60
            df_plot2["Rush Hour"] = df_plot2["is_rush_hour"].map({1: "Rush Hour", 0: "Off-Peak"})

            fig = px.scatter(
                df_plot2,
                x="trip_distance", y="duration_min",
                color="Rush Hour",
                color_discrete_map={"Rush Hour": RED, "Off-Peak": DARK_NAVY},
                opacity=0.4,
                title="Distance vs Duration (Rush Hour vs Off-Peak)",
                labels={"trip_distance": "Distance (miles)", "duration_min": "Duration (minutes)"},
                trendline="ols",
            )
            fig.update_layout(template=PLOTLY_THEME, height=360, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="callout">
            <div class="callout-title">Key Finding: Same distance, very different time</div>
            <div class="callout-body">
                The scatter plot shows two separate trend lines for rush hour vs off-peak —
                at the same distance, rush hour trips consistently take longer. This is exactly
                why the interaction feature <code>distance × rush_hour</code> made it into the model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-header">Feature Correlations</div>', unsafe_allow_html=True)

        numeric_df = df[["duration_sec", "trip_distance", "fare_per_mile",
                          "hour", "is_rush_hour", "is_weekend",
                          "is_pu_manhattan", "is_airport_trip"]].copy()
        numeric_df.columns = ["Duration", "Distance", "Fare/Mile",
                               "Hour", "Rush Hour", "Weekend",
                               "Manhattan PU", "Airport Trip"]
        corr = numeric_df.corr().round(2)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale=[[0, "#EBF5FB"], [0.5, "#F7C948"], [1, DARK_NAVY]],
            title="Correlation Matrix — Key Features vs Duration",
            aspect="auto",
        )
        fig.update_layout(template=PLOTLY_THEME, height=420, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="callout">
                <div class="callout-title">Strong: Distance → Duration (0.67)</div>
                <div class="callout-body">
                    Longer trips take longer — but not perfectly. Rush hour breaks
                    the linear relationship, which is why a tree model beats linear regression.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="callout">
                <div class="callout-title">Moderate: Fare/Mile → Duration (0.41)</div>
                <div class="callout-body">
                    A high fare-per-mile means the meter ran mostly on time (slow trip).
                    A low fare-per-mile means the car covered ground fast (highway).
                </div>
            </div>
            """, unsafe_allow_html=True)

    _footer()


def page_models():
    results     = load_model_results()
    predictions = load_predictions()
    prod_bundle = load_model()

    st.markdown('<h2 style="color:#1a1a2e;">🏆 Model Results</h2>', unsafe_allow_html=True)

    # Comparison table
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    models = results["models"]
    df_models = pd.DataFrame(models)

    col1, col2 = st.columns([3, 2])
    with col1:
        for m in models:
            winner = m.get("winner", False)
            badge  = '<span class="winner-badge">⭐ WINNER</span>' if winner else ""
            bg     = "#fffbea" if winner else "#fff"
            border = "2px solid #F7C948" if winner else "1px solid #eee"
            st.markdown(f"""
            <div style="background:{bg}; border:{border}; border-radius:8px; padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:700; font-size:16px; color:#1a1a2e;">{m['name']}</span>
                        {badge}
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:20px; font-weight:800; color:{'#27AE60' if winner else '#1a1a2e'};">
                            R² = {m['r2']:.3f}
                        </span>
                    </div>
                </div>
                <div style="color:#666; font-size:13px; margin-top:4px;">{m['description']}</div>
                <div style="display:flex; gap:24px; margin-top:8px;">
                    <span style="font-size:13px;"><b>MAE:</b> {m['mae_min']:.1f} min</span>
                    <span style="font-size:13px;"><b>RMSE (log):</b> {m['rmse_log']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure()
        colors = [GREEN if m.get("winner") else (TAXI_YELLOW if "LightGBM" in m["name"] else "#ccc")
                  for m in models]
        fig.add_trace(go.Bar(
            x=[m["name"].replace(" ", "<br>") for m in models],
            y=[m["r2"] for m in models],
            marker_color=colors,
            text=[f"{m['r2']:.3f}" for m in models],
            textposition="outside",
        ))
        fig.update_layout(
            title="R² Score by Model", yaxis_title="R² Score",
            yaxis=dict(range=[0.5, 1.02]),
            template=PLOTLY_THEME, height=340,
            margin=dict(t=50, b=10, l=10, r=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="callout">
            <div class="callout-title">Why LightGBM dominates</div>
            <div class="callout-body">
                Linear models assume straight-line relationships. Rush hour
                doesn't add time linearly — it compounds with distance and
                location. LightGBM's decision trees capture those interactions
                naturally. That's the gap from R²=0.59 to R²=0.98.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Feature importance + Residuals
    col_imp, col_res = st.columns(2)

    with col_imp:
        st.markdown('<div class="section-header">Top 15 Feature Importances</div>', unsafe_allow_html=True)
        fi = results.get("feature_importances", {})
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

        fig = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#F0F0F0"], [1, DARK_NAVY]],
            title="How much each feature drives predictions",
        )
        fig.update_layout(template=PLOTLY_THEME, height=440, margin=dict(t=50, b=20),
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_res:
        st.markdown('<div class="section-header">Prediction Accuracy (Residuals)</div>', unsafe_allow_html=True)

        fig = px.scatter(
            predictions.sample(min(2000, len(predictions)), random_state=42),
            x="actual_min", y="predicted_min",
            opacity=0.4,
            color_discrete_sequence=[DARK_NAVY],
            labels={"actual_min": "Actual Duration (min)", "predicted_min": "Predicted Duration (min)"},
            title="Predicted vs Actual Trip Duration",
        )
        max_val = max(predictions["actual_min"].max(), predictions["predicted_min"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color=TAXI_YELLOW, width=2, dash="dash"))
        fig.add_annotation(x=max_val*0.7, y=max_val*0.85, text="Perfect prediction",
                           showarrow=False, font=dict(color=TAXI_YELLOW, size=11))
        fig.update_layout(template=PLOTLY_THEME, height=440, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Try it yourself
    st.markdown('<div class="section-header">🎯 Try It Yourself — Live Prediction</div>', unsafe_allow_html=True)

    if prod_bundle is None:
        st.warning("Model file not found. Run `python src/models/run_training.py` first.")
        return

    model        = prod_bundle["model"]
    feature_cols = prod_bundle["feature_cols"]
    CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID", "VendorID", "payment_type", "day_of_week"]

    col_form, col_result = st.columns([2, 1])

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            pu_name = st.selectbox("Pickup Location", ZONE_NAMES, index=0)
        with c2:
            do_name = st.selectbox("Dropoff Location", ZONE_NAMES, index=2)

        c3, c4 = st.columns(2)
        with c3:
            distance = st.slider("Trip Distance (miles)", 0.5, 25.0, 3.0, 0.5)
        with c4:
            passengers = st.slider("Passengers", 1, 6, 1)

        c5, c6 = st.columns(2)
        with c5:
            hour = st.slider("Pickup Hour (0 = midnight)", 0, 23, 8)
        with c6:
            dow_label = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            dow = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dow_label)

        fare_est = st.slider("Estimated Fare ($)", 5.0, 100.0, float(2.5 + distance * 2.5), 0.5,
                             help="Base rate ≈ $2.50 + $2.50/mile. Used to compute fare-per-mile signal.")

    with col_result:
        pu_id = POPULAR_ZONES[pu_name]
        do_id = POPULAR_ZONES[do_name]
        row   = build_input_row(pu_id, do_id, distance, hour, dow, passengers, fare_est)

        input_df = pd.DataFrame([row])[feature_cols]
        for col in CATEGORICAL_FEATURES:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype("category")

        pred_log = model.predict(input_df)[0]
        pred_sec = np.expm1(pred_log)
        pred_min = pred_sec / 60
        pred_h   = int(pred_min // 60)
        pred_m   = int(pred_min % 60)

        time_str = f"{pred_h}h {pred_m}m" if pred_h > 0 else f"{pred_m}m"

        rush_label = ""
        if 7 <= hour <= 9 and dow < 5:
            rush_label = "⚠️ AM Rush — expect delays"
        elif 16 <= hour <= 19 and dow < 5:
            rush_label = "⚠️ PM Rush — expect delays"
        elif hour >= 22 or hour <= 5:
            rush_label = "🌙 Late night — roads are clear"
        airport_label = "✈️ Airport trip detected" if (pu_id in AIRPORT_ZONES or do_id in AIRPORT_ZONES) else ""

        st.markdown(f"""
        <div class="pred-result">
            <div class="label">Estimated Trip Duration</div>
            <div class="time">{time_str}</div>
            <div class="label" style="margin-top:8px; font-size:14px;">
                {f'{pred_min:.1f} minutes' if pred_h == 0 else f'{pred_min:.0f} minutes total'}
            </div>
            {'<div style="margin-top:12px; font-size:13px; color:#F7C948;">' + rush_label + '</div>' if rush_label else ''}
            {'<div style="margin-top:6px; font-size:13px; color:#b0b8d0;">' + airport_label + '</div>' if airport_label else ''}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Top factors in this prediction:**")
        zone_match = "Same zone — very short trip!" if pu_id == do_id else f"{distance:.1f} miles"
        factors = [
            ("📍 Route", zone_match),
            ("⏰ Time", f"{'Rush hour' if (7<=hour<=9 or 16<=hour<=19) and dow<5 else 'Off-peak'} ({hour:02d}:00)"),
            ("🗺️ Area",  f"{'Manhattan' if pu_id in MANHATTAN_ZONES else 'Outer borough'} pickup"),
        ]
        for icon_label, detail in factors:
            st.markdown(f"- **{icon_label}:** {detail}")

    _footer()


def page_how_built():
    st.markdown('<h2 style="color:#1a1a2e;">🔧 How I Built This</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
        st.graphviz_chart("""
        digraph pipeline {
            rankdir=LR;
            node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11]

            raw  [label="Raw TLC Data\\n(2.96M rows)", fillcolor="#E8EAF6"]
            clean[label="Data Cleaning\\ncleaner.py", fillcolor="#F7C948"]
            feat [label="Feature Engineering\\nengineering.py", fillcolor="#F7C948"]
            train[label="Model Training\\nrun_training.py", fillcolor="#F7C948"]
            tune [label="Optuna Tuning\\ntuning.py", fillcolor="#F7C948"]
            mlf  [label="MLflow Tracking", fillcolor="#E8F5E9"]
            prod [label="Production Model\\n(LightGBM R²=0.981)", fillcolor="#27AE60", fontcolor="white"]
            app  [label="Streamlit App\\n(this dashboard)", fillcolor="#1a1a2e", fontcolor="white"]

            raw  -> clean [label=" 9.3%\\nremoved"]
            clean -> feat [label=" 32\\nfeatures"]
            feat  -> train
            train -> mlf  [style=dashed]
            train -> tune
            tune  -> prod [label=" best\\nparams"]
            prod  -> app
        }
        """)

        st.markdown('<br><div class="section-header">Build Timeline</div>', unsafe_allow_html=True)

        timeline = [
            ("Day 1", "Data Exploration & Quality Gate",
             "Downloaded 2.96M NYC taxi trips. Built automated quality checks — schema validation, "
             "null rates, range checks. Cleaned data to 2.69M rows by removing impossible values "
             "(negative fares, 300k-mile trips, sub-60-second rides)."),
            ("Day 2", "Exploratory Data Analysis",
             "Built a 7-section EDA notebook. Key discovery: trip duration is right-skewed, "
             "requiring log transform. Rush hour adds 3–4 minutes for the same route. "
             "These findings directly shaped the feature engineering strategy."),
            ("Day 3", "Feature Engineering",
             "Engineered 32 features across 3 categories: temporal (cyclic hour/day encoding, "
             "rush hour flags), geospatial (Manhattan/airport zones, same-zone trips), "
             "and interaction (distance × rush hour). Feature selection dropped 5 redundant columns."),
            ("Day 4", "Model Training & Tuning",
             "Compared LinearRegression (MAE=6.1min), Ridge (MAE=6.0min), and LightGBM (MAE=1.0min) "
             "with 5-fold cross-validation. MLflow logged all runs. Optuna searched 30 hyperparameter "
             "combinations using Bayesian optimization on a 300k sample."),
            ("Day 5", "Portfolio App",
             "Built this 4-page Streamlit app. Live prediction form lets users input any NYC route "
             "and get an instant duration estimate using the production LightGBM model."),
        ]

        for day, title, desc in timeline:
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-day">{day}</div>
                <div class="timeline-title">{title}</div>
                <div class="timeline-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Key Decisions</div>', unsafe_allow_html=True)

        decisions = [
            ("Log-transform the target",
             "Trip duration is right-skewed. Training on raw seconds means a few 3-hour outliers "
             "dominate the loss function. log1p(duration) fixes this — discovered in Day 2 EDA."),
            ("300k sample for tuning",
             "Tuning on all 2.1M training rows × 30 trials × 5 folds would take 8+ hours. "
             "Hyperparameters generalise from a representative sample. Tune fast, train final model "
             "on full data."),
            ("Cyclic encoding for time",
             "Hour 23 and Hour 0 are 1 hour apart but numerically 23 apart. sin/cos encoding "
             "wraps the clock into a circle so the model sees midnight and 11pm as neighbors."),
            ("Absolute variance threshold",
             "Early feature selection used a mean-relative threshold, which dropped all binary "
             "features because continuous columns inflated the mean. Fixed to absolute 0.001."),
        ]

        for title, body in decisions:
            st.markdown(f"""
            <div class="callout">
                <div class="callout-title">💡 {title}</div>
                <div class="callout-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br><div class="section-header">Lessons Learned</div>', unsafe_allow_html=True)
        lessons = [
            "EDA before modeling — log transform came from the data, not a textbook",
            "Feature engineering matters more than hyperparameter tuning",
            "Always benchmark against a simple baseline before complex models",
            "Nested parallelism (joblib × LightGBM) crashes on macOS — set n_jobs=1 for cross_val_score",
            "9.3% of taxi data is unusable — quality gates aren't optional",
        ]
        for l in lessons:
            st.markdown(f"• {l}")

        st.markdown('<br><div class="section-header">Data Source</div>', unsafe_allow_html=True)
        st.markdown("""
        **NYC TLC Trip Record Data**
        NYC Taxi & Limousine Commission

        January 2024 Yellow Taxi trips
        — 2,964,624 raw records
        — Available at nyc.gov/tlc
        """)

    _footer()


# ── Footer ───────────────────────────────────────────────────────────────────
def _footer():
    st.markdown("""
    <div class="footer">
        NYC Taxi Trip Duration Predictor &nbsp;·&nbsp; Built with LightGBM, Optuna, MLflow & Streamlit
        &nbsp;·&nbsp; Data: NYC TLC January 2024
    </div>
    """, unsafe_allow_html=True)


# ── Navigation ───────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size:40px;">🚕</div>
            <div style="font-size:18px; font-weight:800; color:#F7C948; margin-top:6px;">
                NYC Taxi Predictor
            </div>
            <div style="font-size:12px; color:#888; margin-top:4px;">ML Portfolio Project</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["🏠  Project Overview", "📊  Explore the Data", "🏆  Model Results", "🔧  How I Built This"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:12px; color:#888; padding: 8px 0;">
            <b style="color:#F7C948;">Model:</b> LightGBM (tuned)<br>
            <b style="color:#F7C948;">R²:</b> 0.981<br>
            <b style="color:#F7C948;">MAE:</b> ~1 min<br>
            <b style="color:#F7C948;">Data:</b> 2.69M trips<br>
        </div>
        """, unsafe_allow_html=True)

    if "Overview"    in page: page_overview()
    elif "Explore"   in page: page_explore()
    elif "Model"     in page: page_models()
    elif "Built"     in page: page_how_built()


if __name__ == "__main__":
    main()
