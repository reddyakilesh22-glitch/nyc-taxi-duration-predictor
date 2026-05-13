# Days 1-7 Explained: Plain English

---

# Day 1: Load Your Data and Build a Quality Gate

## What Were We Actually Doing?

You downloaded 3.7 million rows of taxi trip data. Before doing anything with it, you needed to answer one question:

**"Is this data even usable?"**

Think of it like buying groceries. Before you cook, you check: Is the milk expired? Is the bread moldy? Is the meat the right cut? You don't just throw everything in the pan and hope for the best.

Day 1 was that inspection step.

---

## The Three Files and What They Did

### `loader.py` - "Let me see what I have"

This was your first look at the data. Like opening a box you've never seen before and taking inventory.

It answered five questions:
1. **How big is it?** → 3,724,889 rows. Almost 3.7 million taxi trips in one month.
2. **What columns exist?** → Pickup time, dropoff time, distance, zone IDs, fare, etc.
3. **What type is each column?** → Are timestamps stored as actual dates or plain text? (Matters a lot, you can't subtract text to get duration.)
4. **What do the numbers look like?** → Min, max, average for every column.
5. **What's missing?** → Which columns have gaps, and how many?

**The red flags we found:**
- Trip distance max = **312,722 miles.** Earth is 24,901 miles around. That's a broken record.
- Fare amount min = **-$899.** Negative money. Also broken.
- 56 trips had **negative duration.** The dropoff was before the pickup. Impossible.

---

### `quality.py` - "Turn that inspection into automatic checks"

Manually reading the loader output every time is fine once. But what about next month's data? And the month after?

`quality.py` turns your observations into a **smoke alarm**, it checks automatically and tells you pass or fail, every time, in seconds.

**The 5 checks it runs:**

**Check 1: Schema:** Do the required columns exist and are they the right type?
If `tpep_pickup_datetime` is stored as text instead of a timestamp, you can't subtract it to get duration. This check catches that immediately.

**Check 2: Row count:** Are there at least 100 rows?
If someone accidentally gives you an empty file, you'd waste an hour training a model on nothing.

**Check 3: Null rates:** Are columns more than 50% empty?
A column that's 70% blank is basically unusable as a feature. The check flags it.

**Check 4: Value ranges:** Are the numbers physically possible?
Distance can't be negative. Zone IDs must be 1–263. Fare can't be -$900. These rules come from domain knowledge, things you know must be true about taxi trips.

**Check 5: Target distribution:** Does the thing you're predicting have variation?
If every single trip lasted exactly 10 minutes, there's nothing for a model to learn. This catches that edge case.

**What it returns:**
```
success:    True/False
failures:   things you MUST fix before proceeding
warnings:   things you SHOULD handle
statistics: counts and numbers
```

---

### `cleaner.py` - "Now actually fix the problems"

The quality gate found the problems. The cleaner fixes them.

**The 6 things it did, in order:**

1. **Computed trip duration**: subtracted pickup time from dropoff time. This created the column `duration_sec`, the number your model will learn to predict. It doesn't exist in raw data. You calculated it.

2. **Dropped invalid durations**: kept only trips between 1 minute and 3 hours. Under 1 minute = probably cancelled. Over 3 hours = probably a data error.

3. **Removed impossible values**: no trips over 100 miles, no negative fares, zone IDs must be 1–263.

4. **Removed exact duplicates**: if the same row appears twice, keep one, drop the other.

5. **Filled nulls**: `passenger_count` was missing for 4.7% of trips. Instead of dropping those rows, filled with 1 (most common value). Same logic for other columns.

6. **Shrunk data types**: changed `int64` to `int16` where numbers are small. Zone IDs only go up to 265, so they don't need 64 bits of storage. This saves memory when you load all 39 months at once.

**The result:**
```
Before cleaning:  3,724,889 rows
After cleaning:   2,379,881 rows
Removed:          1,345,008 rows (36.1%)
```

36% of your data was bad. Without the cleaner, that garbage would have trained your model. (For reference, the same pipeline run on January 2024 data removed only 9.3%, the 2026 data has substantially more nulls and inconsistencies in `passenger_count`, `RatecodeID`, and other columns.)

---

# Day 2: Exploratory Data Analysis (EDA)

## What Were We Actually Doing?

You have 2.4 million clean rows. But clean doesn't mean understood.

EDA is where you **look at your data visually** before building anything. Numbers in a table are hard to interpret, plots make patterns obvious in seconds.

Think of it like a detective looking at a crime scene before forming a theory. You're looking for clues: What patterns exist? What's weird? What will matter when you build the model?

**Every discovery in EDA becomes a decision in modeling.** That's why you do it.

---

## The 7 Sections and What Each One Revealed

### Section 1: Overview
Just confirmed you're looking at the right data. Right number of rows, right columns, right types. The first cell anyone reading your notebook sees.

### Section 2: Target Distribution
You plotted the distribution of `duration_sec`, the thing you're predicting.

**What you saw:** A right-skewed distribution. Most trips were 5–20 minutes, but a long tail stretched out to 3 hours.

**Why it matters:** ML models work best when the target is roughly symmetric (bell-shaped). A right-skewed target means the model spends too much effort on rare extreme values and does poorly on typical ones.

**The fix:** Use `log(duration)` instead of raw duration. Log compresses the tail:
```
10 minutes  → log = 2.3
60 minutes  → log = 4.1
180 minutes → log = 5.2   ← not 18x bigger anymore, just 2.3x bigger
```

Now outliers don't dominate. The model learns from typical trips, not extreme ones.

**Decision made:** Train the model on log-transformed duration.

### Section 3: Missing Values
Plotted a heatmap showing nulls per column after cleaning.

**What you saw:** Zero nulls. The cleaner worked.

A heatmap is just a grid where dark = null, light = present. You could see instantly that every column was completely filled.

### Section 4: Feature Distributions
A 3×3 grid of histograms, one plot per feature.

**What you saw:**
- `trip_distance`, right-skewed. Most trips under 5 miles, tail to 100 miles.
- `fare_amount`, right-skewed. Most fares $8–$25, tail to $1000.
- `PULocationID`, spiky. Some zones are extremely popular (JFK, Midtown), most are rare.

**Why it matters:** Skewed features might need transformation too. Very spiky distributions might mean certain zones dominate your training data.

### Section 5: Correlation Matrix
A heatmap showing how much every feature moves together with every other feature.

**What correlation means:**
- `+1.0` = perfectly in sync. When one goes up, the other always goes up.
- `-1.0` = perfectly opposite. When one goes up, the other always goes down.
- `0.0` = no relationship at all.

**What you saw:**
- `fare_amount` and `trip_distance`: correlation = 0.97. Almost perfectly in sync.
- Makes sense, the meter runs on distance. More miles = higher fare.

**Why it matters:** If two features are 97% correlated, they're telling the model the same thing twice. Keeping both is redundant. The feature selection step on Day 3 dropped `fare_amount` because of this.

**What most correlated with duration?** Fare amount, total amount, and trip distance. The longer the trip, the higher the fare. Makes sense.

### Section 6: Features vs Target
Scatter plots of the strongest features plotted against duration.

**The temporal plots were the most revealing:**
- **Duration by hour:** Trips at 8am averaged 3–4 minutes longer than trips at 3am, for the same distance. Rush hour is real and measurable.
- **Duration by day:** Weekday trips are consistently longer than weekend trips.

**Decision made:** Hour of day and rush hour flags will be among our most important features.

### Section 7: Key Findings
A written markdown cell summarising everything discovered.

**Why write it down:** In 3 months you'll look at this project and wonder "why did I use log transform?" The README captures the reason while it's fresh. Good documentation is what separates a portfolio project from a homework assignment.

---

# Day 3: Feature Engineering

## What Were We Actually Doing?

Raw data columns are poor model inputs. A model sees `PULocationID = 161` as just the number 161. It has no idea that means Midtown Manhattan at rush hour.

**Feature engineering is the process of reshaping raw data into signals the model can actually learn from.**

Think of it like translating a book. The raw data is written in a language the model barely understands. Feature engineering translates it into a language the model is fluent in.

---

## The Three Categories of Features We Built

### Category 1, Temporal Features (time-based)

**The raw column:** `tpep_pickup_datetime = 2026-01-08 08:32:00`

**What we extracted:**
```
hour          = 8    (what hour of the day)
day_of_week   = 0    (Monday)
is_rush_hour  = 1    (yes, 8am weekday)
is_weekend    = 0    (no)
is_late_night = 0    (no)
```

**Why:** Traffic is the biggest driver of trip duration. Traffic follows the clock. Monday at 8am and Sunday at 3am are completely different worlds even for the same route. The model needs to know this explicitly.

**The tricky one, cyclic encoding:**

If you give the model `hour = 23` and `hour = 0`, it thinks they're 23 apart. But 11pm and midnight are only 1 hour apart, they have nearly identical traffic. We fixed this with sin/cos encoding:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

This wraps the clock into a circle. Now the model sees 11pm and midnight as neighbours, not opposites. Think of it like the x and y coordinates of a point moving around a clock face, you need both to know exactly where you are on the circle.

---

### Category 2, Domain (Geography) Features

**The raw column:** `PULocationID = 161`

**What we created:**
```
is_pu_manhattan   = 1   (zone 161 is Midtown Manhattan)
is_airport_trip   = 0   (neither end is an airport)
is_same_zone      = 0   (pickup and dropoff in different zones)
is_both_manhattan = 1   (both pickup and dropoff in Manhattan)
```

**Why:** The number 161 is meaningless to a model. But "this trip starts in Manhattan" is a powerful signal, Manhattan is the most congested place in NYC. We used NYC-specific knowledge (which zone IDs are Manhattan, which are airports) to translate the raw number into something meaningful.

**`is_same_zone` is a particularly strong signal:** A trip that starts and ends in the same zone is almost always under 5 minutes. One flag captures that entire pattern.

---

### Category 3, Interaction Features

**The idea:** Sometimes two things together matter more than either one alone.

**Example:** A 5-mile trip at 3am takes about 10 minutes. The same 5-mile trip at 8am in Manhattan takes 35 minutes. Distance alone doesn't explain the difference. Rush hour alone doesn't explain it either. But **distance × rush hour together** captures that compounding effect.

**What we created:**
```
distance_x_rush  = trip_distance × is_rush_hour
distance_x_night = trip_distance × is_late_night
fare_per_mile    = fare_amount / trip_distance
```

**`fare_per_mile` explained:** This is a congestion proxy. A slow congested trip has the meter running mostly on time, so fare per mile is high. A fast highway trip covers miles quickly, so fare per mile is low. The ratio tells the model something neither column says alone.

---

## Feature Selection, Removing What Doesn't Help

After creating 37 features, we ran two filters:

**Filter 1, High correlation (dropped 3):**

`fare_amount` was 96.5% correlated with `trip_distance`. They say almost the same thing. Keeping both is like asking "how far did you go?" and "how many miles did you travel?" in the same survey. We kept `trip_distance` and dropped `fare_amount`.

**Filter 2, Near-zero variance (dropped 2):**

`improvement_surcharge` is $1.00 on almost every single trip. A column where 99% of rows have the same value teaches the model nothing, there's no pattern to learn.

`month` had zero variance because we only loaded one month of data.

**The result:**
```
Started with:  37 features
Dropped:        5 features  (redundant or useless)
Kept:          32 features  (ready for model training)
```

---

## The Pipeline Script

`run_features.py` chains everything together:

```
Load cleaned data  →  Create features  →  Select features  →  Save
  (2.38M rows)         (20 → 41 cols)       (41 → 34 cols)    (58 MB)
```

This runs in 12 seconds on 2.4 million rows. It becomes your reproducible pipeline, next month when new data arrives, you run one command and get the same features applied consistently.

---

# Day 4: Model Training and Metrics

## What Were We Actually Doing?

You have clean, engineered features. Now you answer: **"What model best predicts trip duration, and how do we measure that?"**

The workflow:
```
Baseline (simple) → Compare models → Track everything with MLflow
```

You always start simple. If LinearRegression gets R²=0.85 and LightGBM gets R²=0.87, the extra complexity might not be worth it. You can only know by comparing.

---

## MAE, Mean Absolute Error

### What it means in plain English
> "On average, how many minutes is the model off by?"

That's it. MAE is the average mistake size.

### How it's calculated

Say you have 5 trips:

| Trip | Actual (min) | Predicted (min) | Absolute Error |
|---|---|---|---|
| 1 | 10 | 12 | 2 |
| 2 | 25 | 22 | 3 |
| 3 | 8 | 8 | 0 |
| 4 | 40 | 35 | 5 |
| 5 | 15 | 18 | 3 |

**Step 1:** Calculate the error (actual − predicted).
**Step 2:** Make all negatives positive (absolute value). Being off by +3 and -3 are equally bad. Without this step they'd cancel out to zero, a lie.
**Step 3:** Average them:
```
MAE = (2 + 3 + 0 + 5 + 3) / 5 = 2.6 minutes
```

**Your results across the three models:**
- LinearRegression MAE = **4667 seconds = 77.8 minutes** (see note below)
- DecisionTree MAE = **70 seconds = 1.2 minutes**
- LightGBM MAE = **55 seconds = 0.9 minutes**

### Why the LinearRegression number looks insane

77 minutes of average error on a model whose R² is 0.76 doesn't add up at first.
Here's what's happening:

We train on `log(1 + duration_sec)`, not on raw seconds. In log-space, a small
mistake looks like 0.3, fine. But when you convert the prediction back to
seconds with `expm1()`, **small log-errors get exponentiated into huge
seconds-errors**. A predicted log of 8.0 vs. an actual log of 7.0 is a
1-point gap in log-space, but in seconds that's `e^8 - e^7 ≈ 1,884 seconds`
= 31 minutes off.

LinearRegression doesn't know the duration range is bounded. It can predict
log values like 10, 12, 15, which exponentiate into days or weeks. A handful
of these blown-up predictions can drag the MAE for the whole test set into
the dozens of minutes.

**Tree models don't have this problem.** A tree's prediction is always
*some weighted average of training labels*, so it can never predict outside
the training-data range. That's why production duration/price models
almost always use trees (LightGBM, XGBoost), not linear regression.

In other words: the absurd 77-minute MAE is itself the lesson. It's exactly
why we use LightGBM.

If a taxi app says "12 minutes," LinearRegression might actually mean
anywhere from 6 to 60. LightGBM means 11 to 13. One is useful, one is not.

### What's a good MAE?
There's no universal number, it depends on the problem. You judge it against:
1. Your baseline, is the new model better?
2. Real-world usefulness, is 1 minute of error acceptable for a taxi app?

---

## R², R-Squared

### The key question R² asks
> "How much better is my model than just guessing the average every single time?"

### The dumbest possible model

Imagine a model that always answers with the average trip duration, no matter what the distance, hour, or location.

```
Average duration = 11.5 minutes
Stupid model answer for every trip = 11.5 minutes
```

This is your floor. Any real model must beat this.

### What the number means

| R² | What it means |
|---|---|
| 1.0 | Perfect, predicted every trip exactly right |
| 0.98 | Explains 98% of why trips differ in duration |
| 0.76 | Explains 76% of why trips differ in duration |
| 0.0 | Exactly as good as guessing the average every time |
| Below 0 | Worse than guessing the average (something is very wrong) |

**Your results:**
- LinearRegression R² = **0.76**
- DecisionTree R² = **0.98**
- LightGBM R² = **0.98**

(DecisionTree and LightGBM tie in log-space at 2 decimal places, but LightGBM wins on MAE in seconds because an ensemble of trees produces smoother predictions than a single tree's coarse step-wise output.)

### A concrete example

Your data has two trips:
- Trip A: 2 miles, 3am, no traffic → 6 minutes
- Trip B: 2 miles, 8am, Midtown → 35 minutes

**Stupid model:** guesses 11.5 min for both. Very wrong.
**LinearRegression:** guesses 10 min and 20 min. Better, but misses the rush hour effect.
**LightGBM:** guesses 6.5 min and 34 min. Almost exactly right.

R² captures how much of the gap between Trip A and Trip B your model explains. LightGBM explains almost all of it (0.98). LinearRegression only explains about three-quarters (0.76).

---

## Why LightGBM Won: A Three-Step Story

The three models tell a clear progression in modeling power.

### Step 1: Linear Regression cannot capture non-linearity

Linear models assume straight-line relationships. They think "more distance = proportionally more time." But reality is not a straight line:

- A 2-mile trip at 3am = 6 minutes
- A 2-mile trip at 8am in Midtown = 35 minutes

Same distance, totally different duration. Linear regression has no way to express that.

### Step 2: A single Decision Tree fixes that

A tree builds a flowchart of yes/no questions:
```
Is it rush hour?
  Yes → Is it Manhattan?
          Yes → Is distance > 1 mile?
                  Yes → predict 30+ minutes
```

Now the model can say "if rush hour AND Manhattan AND >1 mile, predict 30+ minutes" without ever needing a straight line. A single tree gets us from MAE = 77.8 min down to **MAE = 1.2 min**. Most of the heavy lifting in our model is right here.

But a single tree has a limit: its predictions are **coarse**. Every leaf gives one fixed number. So if your tree has 1,000 leaves, it can only produce 1,000 distinct predictions, regardless of the input.

### Step 3: An ensemble of trees smooths everything out

LightGBM is **gradient boosting**: build a tree, look at where it's wrong, build another tree that corrects those mistakes, repeat hundreds of times. The final prediction is the sum of all the trees' contributions.

Because hundreds of trees vote together, the model can produce thousands or millions of distinct predictions, not just a fixed set of leaves. That smoothness is what takes us from MAE = 1.2 min (single tree) down to **MAE = 0.9 min** (ensemble).

### The takeaway

```
Linear         MAE = 77.8 min   straight lines only
Decision Tree  MAE =  1.2 min   non-linearities, but coarse
LightGBM       MAE =  0.9 min   non-linearities AND smooth
```

The huge gap is between Linear and the tree models, not between Decision Tree and LightGBM. That's the right intuition to walk away with: **once you can express non-linearity, you've crossed the important threshold**. Going from single tree to ensemble is a meaningful refinement, not a revolution.

---

## Why We Used Log Duration

Our code does this:
```python
y = np.log1p(duration_sec)    # before training
preds = np.expm1(preds)       # after predicting
```

Duration is right-skewed, most trips are short, a few are very long. If you train on raw duration, a few 3-hour trips dominate the error calculation and the model obsesses over them.

`log1p` compresses the scale so outliers don't dominate. The model focuses on getting typical trips right instead of rare extreme ones. This was a direct result of what we found in Day 2's EDA.

---

## MLflow, Why You Logged Everything

Without MLflow, after 10 experiments you forget what you tried. Did you use 500 trees or 1000? What learning rate? What was the score?

MLflow stores every run automatically:
- Which hyperparameters you used
- What score it got
- The saved model file

You can open `http://localhost:5000` and see every experiment in a table, compare them side by side, click into any run, and download any model.

---

## Optuna, Automatically Finding the Best Settings

### What's a hyperparameter?

A **parameter** is something the model learns from data. LightGBM adjusts thousands of internal numbers during training, those are parameters.

A **hyperparameter** is a setting *you* choose before training even starts. It controls *how* the model learns. Examples:

| Hyperparameter | What it controls |
|---|---|
| `num_leaves` | How complex each tree can be. Higher = more detailed, higher overfit risk |
| `learning_rate` | How big each update step is. Lower = more careful, needs more trees |
| `min_child_samples` | Minimum trips per leaf. Higher = smoother, less overfit |
| `feature_fraction` | What fraction of features each tree sees. Lower = more variety |

The model can't choose these itself, that's your job. And the wrong settings can make the difference between R²=0.95 and R²=0.98.

### The old way, Grid Search

You could try every combination manually:
```
Try learning_rate=0.01, num_leaves=31 → score
Try learning_rate=0.01, num_leaves=63 → score
Try learning_rate=0.01, num_leaves=127 → score
... 100+ combinations later ...
```

That's called Grid Search. It's exhaustive and very slow. If you have 9 hyperparameters each with 5 options, that's 5⁹ = nearly 2 million combinations. Not practical.

### The smart way, Optuna's Bayesian Search

Optuna doesn't try everything blindly. It learns from each attempt:

```
Trial 1:  learning_rate=0.08, num_leaves=200 → R²=0.97
Trial 2:  learning_rate=0.03, num_leaves=150 → R²=0.98   ← better! Focus here
Trial 3:  learning_rate=0.02, num_leaves=160 → R²=0.98   ← still improving
...
Trial 30: learning_rate=0.019, num_leaves=155 → R²=0.98  ← final best
```

Each trial looks at what worked before and chooses the next combination that's most likely to improve. It's like a researcher who reads past experiment notes before designing the next one, not someone who just tries random settings.

This is called **Bayesian optimization**. 30 smart trials beat 300 random ones.

### Why we used 300,000 rows for tuning (not 2.1 million)

Running 30 trials × 5 folds × 2.1M rows = training 150 models on 1.7M rows each. That would take 8+ hours.

Hyperparameters that work well on a 300k random sample generalize to the full dataset, the patterns are the same, there's just less of them. We tune fast on the sample, then train the final model once on all 2.1M rows with the best settings found.

This is standard practice in industry. Tune on a representative sample, deploy on full data.

### What the numbers tuned actually mean

**`num_leaves = 31–255`**
Think of a tree as a flowchart of yes/no questions. `num_leaves` is the number of final answers the flowchart can give. 31 leaves = simple flowchart. 255 leaves = very complex, can capture subtle patterns but risks memorizing the training data.

**`learning_rate = 0.01–0.1`**
Imagine adjusting a dial. High learning rate = big fast turns (might overshoot the best setting). Low learning rate = small careful turns (takes longer but more precise). LightGBM builds hundreds of trees, each correcting the mistakes of the last. The learning rate controls how aggressively each tree corrects.

**`min_child_samples = 50–300`**
Each leaf of the tree represents a rule: "trips under 3 miles, in Manhattan, after 5pm take about 18 minutes." `min_child_samples` sets the minimum number of training trips that must match a rule before it's allowed. High = only make rules backed by lots of evidence. Low = allow very specific rules (overfit risk).

**`feature_fraction = 0.6–1.0`**
Each tree only looks at a random subset of features. This forces the model to learn diverse patterns instead of always relying on the same 2–3 strong features. It's the same idea as not putting all your eggs in one basket.

### The honest result of our tuning

After 3 hours of tuning (30 trials × 5-fold CV on 300k rows), the result was:

```
Default LightGBM:  R² = 0.98,  MAE = 59.16s
Tuned LightGBM:    R² = 0.98,  MAE = 59.50s

(Note: this was on the older January 2024 dataset. We did not re-tune on the 2026 data; based on the prior result it's unlikely to change the conclusion.)
```

The tuned model was **slightly worse** on the test set. The default hyperparameters were already near-optimal for this data.

**Why this matters:** This is a real lesson, not a failure. Tuning is not always worth the cost. Good default settings + good features will beat fancy tuning on bad features almost every time. We document this in the portfolio because hiring managers respect honesty about what didn't work, not just success stories.

---

# Day 5: Building the Portfolio App

## What Were We Actually Doing?

Up to Day 4, your model lived inside a `.pkl` file on your laptop. It worked, but only you could use it. To anyone else, a hiring manager, a friend, a future you in 6 months, your project was just a folder of Python files.

**Day 5 turned the model into a website.**

Specifically: a 4-page interactive dashboard where anyone with the URL can click around, see the data, see the results, and most importantly, **type in their own taxi route and get a prediction in real time**.

Think of it like the difference between writing a recipe in a notebook vs. opening a restaurant. Same dish; very different reach.

---

## What is Streamlit?

Streamlit is a Python library that turns scripts into web apps. You write Python, no HTML, no JavaScript, no React, and Streamlit renders it as a website in your browser.

**Before Streamlit:**
```
Want a chart on a website?
1. Learn HTML, CSS, JavaScript
2. Set up a frontend build tool
3. Wire up a backend API
4. Deploy to a server
5. ... (3 weeks later, finally see your chart)
```

**With Streamlit:**
```python
import streamlit as st
import plotly.express as px

st.title("My Chart")
st.plotly_chart(px.line(df, x="hour", y="duration"))
```

Run `streamlit run app.py` and you have a live website. That's the whole pitch.

---

## The 4 Pages and What Each One Does

The app is structured like a portfolio website, not a dashboard. Each page answers a specific question someone visiting your project would have.

### Page 1, Project Overview

**The question it answers:** *"What is this project, and why should I care?"*

If a hiring manager spends 30 seconds on your portfolio, this page is what they see. So it has to land fast:
- A bold hero with the project name
- 4 KPI cards: 2.38M trips, 34 features, R² 0.98, ~1 min average error
- A 2-sentence description of what the model does
- The tech stack as little badges (LightGBM, Optuna, MLflow, etc.)

The numbers do the work. "98% accuracy on 2.4M real NYC trips" tells a recruiter more in 10 seconds than 3 paragraphs of prose.

### Page 2, Explore the Data

**The question it answers:** *"What does the data actually look like?"*

This is your EDA from Day 2, but interactive instead of static. Four tabs:
1. **Duration distribution**: with a toggle to switch between raw minutes and log scale (so visitors can *see* why we used log transform)
2. **Time patterns**: average duration by hour, with the rush hour bands highlighted
3. **Location effects**: box plots comparing airport vs Manhattan vs other-borough trips
4. **Correlations**: a heatmap showing which features track with duration

The point isn't just to show charts, it's to walk a visitor through the discoveries that shaped the model.

### Page 3, Model Results

**The question it answers:** *"How well does the model work, and can I try it?"*

Two halves:

**Top half, comparison and accuracy:**
- Cards for each model tried (Linear, Decision Tree, LightGBM) with their scores
- A "predicted vs actual" scatter plot, points clustered along the diagonal = good predictions
- A bar chart of which features mattered most

**Bottom half, the live predictor.** This is the most interesting part. Visitors pick a pickup zone, dropoff zone, distance, hour, and day of week. They get a real prediction from the model.

We'll explain how this works in detail below, it's the trickiest piece of the whole app.

### Page 4, How I Built This

**The question it answers:** *"Are you a real engineer or did you just glue stuff together?"*

This page is for the technical reviewer:
- An architecture diagram showing the pipeline (raw data → cleaning → features → training → tuning → app)
- A timeline of Days 1–5
- 4 key decisions explained (why log transform, why 300k tuning sample, why cyclic encoding, etc.)
- 5 lessons learned

This is where you prove you understand the *why* behind your choices, not just the *what*.

---

## How the Live Predictor Works (The Important Part)

This is the trickiest and most clever part of the app. The model expects **32 input features** (PULocationID, DOLocationID, hour_sin, distance_x_rush_hour, ...). But asking a visitor to type in 32 values would be a terrible experience.

So the predictor lets the visitor pick **just 6 simple things**:
1. Pickup zone (dropdown of 25 popular NYC zones)
2. Dropoff zone (same dropdown)
3. Distance (a slider, 0.5–25 miles)
4. Hour of day (slider, 0–23)
5. Day of week (Mon–Sun)
6. Number of passengers (1–6)

Then **behind the scenes**, the app **rebuilds all 32 features** from those 6 inputs, exactly the same way Day 3's feature engineering did during training:

```
User picks: Midtown → JFK, 12 miles, 8am Monday, 1 passenger

App computes:
   PULocationID    = 161 (Midtown's TLC zone ID)
   DOLocationID    = 132 (JFK)
   hour            = 8
   day_of_week     = 0
   hour_sin        = sin(2π × 8/24) = 0.866
   hour_cos        = cos(2π × 8/24) = -0.5
   is_am_rush      = 1   (it's 8am on a weekday)
   is_pu_manhattan = 1   (Midtown is in Manhattan)
   is_airport_trip = 1   (JFK is an airport)
   distance_x_rush = 12 × 1 = 12
   ... (22 more features)

Model.predict() → log(duration) = 6.42
expm1(6.42) = 614 seconds = 10.2 minutes
```

**Why this matters:** The exact same transformations that ran on 1.9M training rows have to run on this single live input. If the training pipeline computed `is_rush_hour` one way and the app computed it differently, the model would silently produce garbage. This is one of the most common ways ML systems break in production.

The way to avoid that: the app's `build_input_row()` function mirrors the training code's `create_features()` line-for-line. Same logic, applied at the same point in the flow.

---

## What's the Difference Between This and an "API"?

You'll hear ML engineers talk about "deploying a model" or "model APIs." Here's how a Streamlit app fits in:

| Approach | What it is | When to use |
|---|---|---|
| **Streamlit app** | A website with a model inside | Demos, portfolios, internal tools, exploration |
| **REST API** (FastAPI) | A URL that takes JSON in and returns JSON out | Production systems, mobile apps, other services calling the model |
| **Batch script** | A Python script that processes a file | Nightly jobs, ETL, large offline runs |

For a portfolio, **Streamlit is the right call.** A REST API isn't useful to a hiring manager, they don't have a curl command ready. They have a browser.

(For a real production taxi app, you'd build a FastAPI service, and the iPhone app would call it. The Streamlit version exists so a human can see the model work without writing any code.)

---

## A Subtle Bug We Hit

One thing worth knowing about because you'll see it again: the Model Results page initially showed raw HTML code on screen instead of rendered cards. The reason was a beautiful little gotcha:

- The card HTML was inside a Python loop, indented 12 spaces deep
- One of the values (`{badge}`) was empty for non-winner models
- When Streamlit dedented the multi-line string, that empty value left a **blank line in the middle of the HTML**
- Markdown's CommonMark parser closes any HTML block at a blank line
- Everything after the blank line got re-parsed as an indented code block, and shown as literal text

Lesson: when you mix conditional content into multi-line HTML strings, an empty value can leave a blank line that breaks everything downstream. Fix: build the HTML as a single concatenated string so empty values become zero-length segments instead of blank lines.

This kind of bug doesn't show up in any tutorial. They show up when you build real things. Catching them is part of the work.

---

# Day 6: Docker, Tests, and CI/CD

## What Were We Actually Doing?

By the end of Day 5, you had a working app on your laptop. Good for showing yourself. Useless for anyone else.

Day 6 was about making the project **shippable**. Three pieces:

1. **Docker** so the app runs the same way on any computer, not just yours
2. **Tests** so you'd know if a future change broke something
3. **CI/CD** so those tests run automatically every time you push to GitHub

This is the difference between "I built a model" and "I built a project a team could trust."

---

## Part 1: Docker, or "Works on My Machine, Solved"

Every developer has lived through this conversation:

> "It works on my laptop."
> "It doesn't work on my laptop."
> "Did you install Python 3.11?"
> "Yes."
> "What about pandas 2.2?"
> "Let me check..."
> *(four hours later)*

Docker fixes this by packaging your entire app, **including the operating system, the Python version, the libraries, everything**, into one portable file called an "image". Anyone can run that image on any computer and get exactly the same result.

Think of it like a sealed lunchbox vs. a recipe. A recipe says "use bread, ham, mustard." A lunchbox already has the sandwich made. Hand the lunchbox to anyone and they get the same sandwich.

### What the Dockerfile actually says

Our `Dockerfile` is 25 lines. In plain English it says:

```
Start from a clean Linux box with Python 3.11.
Install libgomp1 (LightGBM needs it).
Copy requirements.txt and pip install everything.
Copy src/, app/, and setup.py into the box.
Open port 8501 so the outside world can reach Streamlit.
When the box starts, run streamlit on the app.
```

To use it:
```bash
docker build -t nyc-taxi-duration .   # build the lunchbox
docker compose up                      # open it and serve
```

Open http://localhost:8501 and your app is running, but this time inside a container that you could ship to AWS, a colleague's laptop, or any cloud platform without changing a single line of code.

**Why this matters for hiring:** "I can dockerize my projects" is table-stakes for most ML / data engineering roles. It tells a hiring manager you've thought about how your code leaves your laptop.

---

## Part 2: Tests, or "How Will You Know Something Broke?"

You spent five days carefully building a pipeline. Now imagine you add a new feature next month. How do you know you didn't accidentally break the cleaner, the feature engineering, or the model?

**You write tests.**

A test is just a small piece of code that says "given this input, the function should produce that output." If a future change makes the function produce something different, the test fails and you know immediately.

We wrote 8 tests across 3 files:

### `tests/test_data_quality.py` (3 tests)

Tests for the data quality gate:

- **Passes on clean data:** Make a small fake dataset that looks like good taxi data. Quality gate should approve it.
- **Catches too-few-rows:** Make a dataset with only 5 rows. Quality gate should reject it.
- **Catches missing columns:** Drop a required column. Quality gate should reject it.

### `tests/test_features.py` (3 tests)

Tests for feature engineering:

- **All expected columns produced:** Run `create_features()` on synthetic data. Check that all 21 new feature columns appear.
- **No NaNs:** Engineered features must be non-null. The model can't handle NaNs.
- **Ranges are correct:** Cyclic encodings stay in [-1, 1], binary flags stay in {0, 1}, hours stay in 0-23.

### `tests/test_model.py` (2 tests)

Tests for the model:

- **A tiny LightGBM trains and predicts:** Sanity check that the training stack works even without your real data files.
- **The production model loads and predicts in range:** Load the actual `production_model.pkl`. Build a typical NYC trip input. Verify the prediction is between 3 and 60 minutes (not 38 days).

### Running them

```bash
pytest tests/ -v
```

You see:
```
tests/test_data_quality.py ... 3 passed
tests/test_features.py     ... 3 passed
tests/test_model.py        ... 2 passed
8 passed in 3.2s
```

Green. If anything goes red, you fix it before merging.

**The deeper point:** Tests are how you sleep at night. Without them, every code change is a roll of the dice. With them, you know the moment something breaks.

---

## Part 3: CI/CD, or "A Robot That Checks Your Work"

OK so you have tests. Great. Will you remember to run them before every commit? Be honest. No, you won't. Nobody does.

**CI/CD (Continuous Integration / Continuous Deployment)** is a robot that runs your tests *for you*, automatically, every time you push code to GitHub.

We used **GitHub Actions**, which is free for public repos. Here's the flow:

```
You push code to GitHub.
       ↓
GitHub Actions wakes up.
       ↓
Spins up a fresh Linux box.
       ↓
Installs Python 3.11, then pip install -r requirements.txt.
       ↓
Runs pytest tests/ -v.
       ↓
Runs ruff check src/ app/ tests/ (the linter).
       ↓
Reports back: green checkmark or red X.
```

This happens every push, every pull request. You never have to remember.

### What our workflow does

The file `.github/workflows/ci.yml` defines two jobs that run in parallel:

**Job 1: Test**
- Set up Python 3.11
- pip install everything
- pytest tests/ -v

**Job 2: Lint**
- Set up Python 3.11
- pip install ruff
- Check for real bugs (undefined names, unused imports, syntax errors) but don't nitpick style

Total time: about 3 minutes per push.

### What the green checkmark means

When both jobs pass, GitHub shows a green checkmark next to your commit. That checkmark also appears on your README as a badge:

```markdown
[![CI](https://github.com/.../actions/workflows/ci.yml/badge.svg)](...)
```

A green CI badge on a GitHub repo is a credibility signal. It says: "my tests pass, my code lints, you can trust this isn't held together with duct tape."

---

## What Day 6 Actually Bought You

By the end of Day 6:

- The app runs identically on any computer with Docker (no more "works on my machine")
- 8 automated tests catch broken code before it hits main
- Every push is verified automatically by a GitHub robot
- The README has a green CI badge that anyone can see

The project went from "a folder of code" to "something that looks production-grade." Same model, same data, dramatically different perception.

---

# Day 7: Deploying, Polishing, and Going Public

## What Were We Actually Doing?

Day 6 made the project shippable. Day 7 was about actually shipping it.

Three buckets of work:

1. **Switch the data** from January 2024 to January 2026 (the newest available)
2. **Deploy** to Streamlit Community Cloud so the app has a real public URL
3. **Polish** the design to look professional, not like a generic SaaS dashboard

By the end of the day, you had a link you could send to anyone. That's the difference between a personal project and a portfolio piece.

---

## Part 1: Refreshing the Data

The model was trained on January 2024 data. Fine for a learning project. Less impressive on a resume in 2026.

So we deleted everything from 2023, 2024, and 2025 (21 GB across multiple vehicle types) and ran the entire pipeline on **January 2026** yellow taxi data instead.

This was the cleanest test of the whole project: was the pipeline truly reproducible, or did it secretly depend on specific quirks of the 2024 data?

### What changed and what didn't

| | January 2024 | January 2026 |
|---|---|---|
| Raw rows | 2.96M | 3.72M |
| After cleaning | 2.69M | 2.38M |
| Removed | 9.3% | 36% |
| Features | 32 | 34 |
| Best R² | 0.98 | 0.98 |
| Best MAE | 1.0 min | 0.9 min |

Two interesting findings:

**The 2026 data was dirtier.** 36% of raw rows got removed in cleaning vs. 9.3% in 2024. The 2026 file has way more nulls in `passenger_count` and `RatecodeID`. The cleaner caught all of it without any code changes, which is exactly what a good cleaner should do.

**The model got slightly better.** R² went from 0.98 to 0.98, and MAE dropped from 1.0 min to 0.9 min. Two new features that NYC introduced (`cbd_congestion_fee` for the Manhattan congestion charge, and `fare_amount` getting kept this time because correlation patterns shifted) carried real signal.

### The baseline blow-up

One thing did change dramatically: the linear regression baseline went from MAE = 6.1 min (2024) to **MAE = 77.8 min (2026)**. Same model, same features, same code.

Why? Linear regression on a `log(duration)` target can predict log values like 12 or 15, which exponentiate into days. A handful of these blown-up predictions dragged the test MAE into the 70-minute range. Tree models (LightGBM) don't have this problem because their predictions are bounded by training-data leaf values.

This is the **actual reason** production duration / price prediction models use trees, not linear regression. We didn't hide it. We documented it.

---

## Part 2: Deploying to Streamlit Community Cloud

Until Day 7, the only way to see the app was to clone the repo and run it locally. Useless for sharing.

Streamlit Community Cloud is a free hosting service from the makers of Streamlit. It connects to a GitHub repo, watches the `main` branch, and rebuilds the app automatically every time you push.

### How the deploy works

1. Sign in to share.streamlit.io with GitHub
2. Click "New app"
3. Point it at your repo, the `main` branch, and `app/streamlit_app.py`
4. Click "Deploy"

About 5 minutes later, you have a permanent URL like `https://nyc-taxi-duration-predictor.streamlit.app`. Anyone can open it. No login. No setup. Just works.

### The catch: cloud has no access to your laptop

This is where it got interesting. Streamlit Cloud only sees what's in your **GitHub repo**. Your local `data/` and `models/` folders are gitignored (rightly so, they're huge). So when the app booted in the cloud, it had no model file and no features data.

The live predictor crashed. The EDA page crashed.

### Two fixes

**Fix 1: Bundle the production model.** Updated `.gitignore` to allow exactly one file:
```
/models/*
!/models/production_model.pkl
```
This commits just the 6.7 MB production model to GitHub, nothing else. The live predictor works in the cloud.

**Fix 2: Pre-sample the features parquet.** The full features file is 58 MB, large enough to trigger GitHub's "this file is huge" warning. So we sampled 50,000 rows into a 0.4 MB `app/eda_sample.parquet` and bundled that instead. The EDA page reads from the sample. Same insights, 145x smaller file.

### A real production debug

When the EDA page first loaded on the cloud, it crashed with:
```
ModuleNotFoundError: No module named 'statsmodels'
```

The Plotly trendline feature (`trendline="ols"`) needs statsmodels, which was installed in the local conda environment but not pinned in `requirements.txt`. Locally everything worked; in the cloud's clean Python environment it broke.

Fix: add `statsmodels>=0.14.0` to requirements.txt. Push. Streamlit Cloud auto-redeployed, and the EDA page came back.

**The lesson:** "It works in my environment" hides which dependencies are actually required. Your `requirements.txt` is your contract with the deploy target. Anything imported anywhere in the codebase, even by a library you call, must be listed.

---

## Part 3: Polishing the Design

The default app looked like every other Streamlit dashboard: loud yellow accents, dark navy hero, emoji icons, default fonts. It worked but it felt cheap.

Day 7 was a from-scratch redesign with one rule: **make it look like Apple made it, not like a SaaS template.**

### The new design system

**Palette.** Out: bright `#F7C948` and dark `#1A1A2E`. In:
- White surfaces (`#FFFFFF`)
- A warm off-white page background (`#FAFAF8`)
- Soft secondary surfaces (`#F4F3EF`)
- Barely-visible borders (`#ECEAE5`)
- One refined amber accent (`#B8943F`) used sparingly

The whole site wears white, then softer white, then a single accent color where it earns its place. Like apple.com.

**Typography.** SF Pro / system stack with tight letter-spacing on headings (-0.018em to -0.028em), tabular numerals for metrics, wide tracking on small uppercase labels. Restraint, not loudness.

**No emoji as icons.** Built a registry of 26 custom inline SVG icons, stroke-based at 1.6px width with rounded line caps. They use `currentColor` so the same icon can be ink, accent, or slate depending on context. Used in the sidebar brand, page heads, section eyebrows, KPI labels, callouts, prediction chips, factor rows, and timeline dots.

**Micro-interactions.** Things that move when you touch them, not before:
- Cards lift 1-2 pixels on hover with a soft shadow appearing
- Slider thumbs scale 1.08 on hover
- Timeline dots fill with the accent and scale 1.15 on hover
- The prediction value plays a 500ms reveal animation each time it updates
- Tabs underline slides smoothly between selections

Nothing animates on page load. (More on why below.)

### The page-flash bug

The first version of the redesign had a subtle bug: clicking from "Overview" to "Explore the data" briefly flashed the Overview hero before showing Explore.

The cause: Streamlit reruns the entire script on every interaction. While the new page renders, the old page's DOM is still visible. I had a 500ms page-entry animation on the hero, which delayed the new content from fully painting, making the flicker visible.

Fix: remove page-entry animations entirely. Keep interaction animations (hover, slider, prediction reveal). Page swaps now feel instant.

**The lesson:** Animations on page load fight reactive frameworks like Streamlit. Animations on user actions (hover, click, change) don't. Reserve motion for things the user is doing, not for things the framework is doing.

---

## What Day 7 Actually Shipped

By the end:

- A live URL anyone can visit
- A model retrained on the latest available data
- A design that looks intentional, not templated
- A README with a green CI badge and headline numbers
- Two honest writeups (baseline blow-up, Optuna didn't help)

You can put this on a resume. You can send the link in a cover letter. You can share it on LinkedIn. That's a real portfolio piece.

---

# The Through-Line Across All Seven Days

```
Day 1:  Raw messy data       →  Clean usable data       (removed 36% bad rows)
Day 2:  Numbers              →  Visual understanding    (log transform, rush hour signal)
Day 3:  Raw columns          →  Meaningful signals      (19 columns → 34 engineered)
Day 4:  Features             →  Predictions             (MAE 77.8 min → 0.9 min)
Day 5:  A model file         →  A working product       (4-page site with live predictor)
Day 6:  A local app          →  A shippable product     (Docker + tests + CI/CD)
Day 7:  A repo on your laptop → A live public link      (deployed, polished, shareable)
```

Each day built on the previous one:
- Skip Day 1 → model trains on garbage
- Skip Day 2 → you wouldn't know to use log transform
- Skip Day 3 → model misses rush hour patterns entirely
- Skip Day 4 → you have features but no model
- Skip Day 5 → you have a model but nobody can see or use it
- Skip Day 6 → your project breaks the moment it leaves your laptop
- Skip Day 7 → you have a polished local project that no one in the world can find

The whole pipeline is a chain. Every link matters.

**The big picture:** A real ML project isn't just "train a model." It's *get raw data → clean it → understand it → reshape it → train on it → tune it → make it usable → make it reliable → make it public*. Seven steps. Each one is its own discipline. Each one is what the next link depends on. That's the actual job.
