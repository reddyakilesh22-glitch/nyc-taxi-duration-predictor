# Days 1–4 Explained — Plain English

---

# Day 1 — Load Your Data and Build a Quality Gate

## What Were We Actually Doing?

You downloaded 3 million rows of taxi trip data. Before doing anything with it, you needed to answer one question:

**"Is this data even usable?"**

Think of it like buying groceries. Before you cook, you check: Is the milk expired? Is the bread moldy? Is the meat the right cut? You don't just throw everything in the pan and hope for the best.

Day 1 was that inspection step.

---

## The Three Files and What They Did

### `loader.py` — "Let me see what I have"

This was your first look at the data. Like opening a box you've never seen before and taking inventory.

It answered five questions:
1. **How big is it?** → 2,964,624 rows. Almost 3 million taxi trips in one month.
2. **What columns exist?** → Pickup time, dropoff time, distance, zone IDs, fare, etc.
3. **What type is each column?** → Are timestamps stored as actual dates or plain text? (Matters a lot — you can't subtract text to get duration.)
4. **What do the numbers look like?** → Min, max, average for every column.
5. **What's missing?** → Which columns have gaps, and how many?

**The red flags we found:**
- Trip distance max = **312,722 miles.** Earth is 24,901 miles around. That's a broken record.
- Fare amount min = **-$899.** Negative money. Also broken.
- 56 trips had **negative duration.** The dropoff was before the pickup. Impossible.

---

### `quality.py` — "Turn that inspection into automatic checks"

Manually reading the loader output every time is fine once. But what about next month's data? And the month after?

`quality.py` turns your observations into a **smoke alarm** — it checks automatically and tells you pass or fail, every time, in seconds.

**The 5 checks it runs:**

**Check 1 — Schema:** Do the required columns exist and are they the right type?
If `tpep_pickup_datetime` is stored as text instead of a timestamp, you can't subtract it to get duration. This check catches that immediately.

**Check 2 — Row count:** Are there at least 100 rows?
If someone accidentally gives you an empty file, you'd waste an hour training a model on nothing.

**Check 3 — Null rates:** Are columns more than 50% empty?
A column that's 70% blank is basically unusable as a feature. The check flags it.

**Check 4 — Value ranges:** Are the numbers physically possible?
Distance can't be negative. Zone IDs must be 1–263. Fare can't be -$900. These rules come from domain knowledge — things you know must be true about taxi trips.

**Check 5 — Target distribution:** Does the thing you're predicting have variation?
If every single trip lasted exactly 10 minutes, there's nothing for a model to learn. This catches that edge case.

**What it returns:**
```
success:    True/False
failures:   things you MUST fix before proceeding
warnings:   things you SHOULD handle
statistics: counts and numbers
```

---

### `cleaner.py` — "Now actually fix the problems"

The quality gate found the problems. The cleaner fixes them.

**The 6 things it did, in order:**

1. **Computed trip duration** — subtracted pickup time from dropoff time. This created the column `duration_sec` — the number your model will learn to predict. It doesn't exist in raw data. You calculated it.

2. **Dropped invalid durations** — kept only trips between 1 minute and 3 hours. Under 1 minute = probably cancelled. Over 3 hours = probably a data error.

3. **Removed impossible values** — no trips over 100 miles, no negative fares, zone IDs must be 1–263.

4. **Removed exact duplicates** — if the same row appears twice, keep one, drop the other.

5. **Filled nulls** — `passenger_count` was missing for 4.7% of trips. Instead of dropping those rows, filled with 1 (most common value). Same logic for other columns.

6. **Shrunk data types** — changed `int64` to `int16` where numbers are small. Zone IDs only go up to 265, so they don't need 64 bits of storage. This saves memory when you load all 39 months at once.

**The result:**
```
Before cleaning:  2,964,624 rows
After cleaning:   2,687,584 rows
Removed:            277,040 rows (9.3%)
```

9.3% of your data was bad. Without the cleaner, that garbage would have trained your model.

---

# Day 2 — Exploratory Data Analysis (EDA)

## What Were We Actually Doing?

You have 2.7 million clean rows. But clean doesn't mean understood.

EDA is where you **look at your data visually** before building anything. Numbers in a table are hard to interpret — plots make patterns obvious in seconds.

Think of it like a detective looking at a crime scene before forming a theory. You're looking for clues: What patterns exist? What's weird? What will matter when you build the model?

**Every discovery in EDA becomes a decision in modeling.** That's why you do it.

---

## The 7 Sections and What Each One Revealed

### Section 1 — Overview
Just confirmed you're looking at the right data. Right number of rows, right columns, right types. The first cell anyone reading your notebook sees.

### Section 2 — Target Distribution
You plotted the distribution of `duration_sec` — the thing you're predicting.

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

### Section 3 — Missing Values
Plotted a heatmap showing nulls per column after cleaning.

**What you saw:** Zero nulls. The cleaner worked.

A heatmap is just a grid where dark = null, light = present. You could see instantly that every column was completely filled.

### Section 4 — Feature Distributions
A 3×3 grid of histograms — one plot per feature.

**What you saw:**
- `trip_distance` — right-skewed. Most trips under 5 miles, tail to 100 miles.
- `fare_amount` — right-skewed. Most fares $8–$25, tail to $1000.
- `PULocationID` — spiky. Some zones are extremely popular (JFK, Midtown), most are rare.

**Why it matters:** Skewed features might need transformation too. Very spiky distributions might mean certain zones dominate your training data.

### Section 5 — Correlation Matrix
A heatmap showing how much every feature moves together with every other feature.

**What correlation means:**
- `+1.0` = perfectly in sync. When one goes up, the other always goes up.
- `-1.0` = perfectly opposite. When one goes up, the other always goes down.
- `0.0` = no relationship at all.

**What you saw:**
- `fare_amount` and `trip_distance`: correlation = 0.97. Almost perfectly in sync.
- Makes sense — the meter runs on distance. More miles = higher fare.

**Why it matters:** If two features are 97% correlated, they're telling the model the same thing twice. Keeping both is redundant. The feature selection step on Day 3 dropped `fare_amount` because of this.

**What most correlated with duration?** Fare amount, total amount, and trip distance. The longer the trip, the higher the fare. Makes sense.

### Section 6 — Features vs Target
Scatter plots of the strongest features plotted against duration.

**The temporal plots were the most revealing:**
- **Duration by hour:** Trips at 8am averaged 3–4 minutes longer than trips at 3am — for the same distance. Rush hour is real and measurable.
- **Duration by day:** Weekday trips are consistently longer than weekend trips.

**Decision made:** Hour of day and rush hour flags will be among our most important features.

### Section 7 — Key Findings
A written markdown cell summarising everything discovered.

**Why write it down:** In 3 months you'll look at this project and wonder "why did I use log transform?" The README captures the reason while it's fresh. Good documentation is what separates a portfolio project from a homework assignment.

---

# Day 3 — Feature Engineering

## What Were We Actually Doing?

Raw data columns are poor model inputs. A model sees `PULocationID = 161` as just the number 161. It has no idea that means Midtown Manhattan at rush hour.

**Feature engineering is the process of reshaping raw data into signals the model can actually learn from.**

Think of it like translating a book. The raw data is written in a language the model barely understands. Feature engineering translates it into a language the model is fluent in.

---

## The Three Categories of Features We Built

### Category 1 — Temporal Features (time-based)

**The raw column:** `tpep_pickup_datetime = 2024-01-08 08:32:00`

**What we extracted:**
```
hour          = 8    (what hour of the day)
day_of_week   = 0    (Monday)
is_rush_hour  = 1    (yes, 8am weekday)
is_weekend    = 0    (no)
is_late_night = 0    (no)
```

**Why:** Traffic is the biggest driver of trip duration. Traffic follows the clock. Monday at 8am and Sunday at 3am are completely different worlds even for the same route. The model needs to know this explicitly.

**The tricky one — cyclic encoding:**

If you give the model `hour = 23` and `hour = 0`, it thinks they're 23 apart. But 11pm and midnight are only 1 hour apart — they have nearly identical traffic. We fixed this with sin/cos encoding:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

This wraps the clock into a circle. Now the model sees 11pm and midnight as neighbours, not opposites. Think of it like the x and y coordinates of a point moving around a clock face — you need both to know exactly where you are on the circle.

---

### Category 2 — Domain (Geography) Features

**The raw column:** `PULocationID = 161`

**What we created:**
```
is_pu_manhattan   = 1   (zone 161 is Midtown Manhattan)
is_airport_trip   = 0   (neither end is an airport)
is_same_zone      = 0   (pickup and dropoff in different zones)
is_both_manhattan = 1   (both pickup and dropoff in Manhattan)
```

**Why:** The number 161 is meaningless to a model. But "this trip starts in Manhattan" is a powerful signal — Manhattan is the most congested place in NYC. We used NYC-specific knowledge (which zone IDs are Manhattan, which are airports) to translate the raw number into something meaningful.

**`is_same_zone` is a particularly strong signal:** A trip that starts and ends in the same zone is almost always under 5 minutes. One flag captures that entire pattern.

---

### Category 3 — Interaction Features

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

## Feature Selection — Removing What Doesn't Help

After creating 37 features, we ran two filters:

**Filter 1 — High correlation (dropped 3):**

`fare_amount` was 96.5% correlated with `trip_distance`. They say almost the same thing. Keeping both is like asking "how far did you go?" and "how many miles did you travel?" in the same survey. We kept `trip_distance` and dropped `fare_amount`.

**Filter 2 — Near-zero variance (dropped 2):**

`improvement_surcharge` is $1.00 on almost every single trip. A column where 99% of rows have the same value teaches the model nothing — there's no pattern to learn.

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
  (2.69M rows)         (20 → 41 cols)       (41 → 36 cols)    (58 MB)
```

This runs in 12 seconds on 2.7 million rows. It becomes your reproducible pipeline — next month when new data arrives, you run one command and get the same features applied consistently.

---

# Day 4 — Model Training and Metrics

## What Were We Actually Doing?

You have clean, engineered features. Now you answer: **"What model best predicts trip duration, and how do we measure that?"**

The workflow:
```
Baseline (simple) → Compare models → Track everything with MLflow
```

You always start simple. If LinearRegression gets R²=0.85 and LightGBM gets R²=0.87, the extra complexity might not be worth it. You can only know by comparing.

---

## MAE — Mean Absolute Error

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
**Step 2:** Make all negatives positive (absolute value). Being off by +3 and -3 are equally bad. Without this step they'd cancel out to zero — a lie.
**Step 3:** Average them:
```
MAE = (2 + 3 + 0 + 5 + 3) / 5 = 2.6 minutes
```

**Your results:**
- LinearRegression MAE = **364 seconds = 6.1 minutes**
- LightGBM MAE = **59 seconds = 1.0 minute**

If a taxi app says "12 minutes," LinearRegression might actually mean anywhere from 6 to 18. LightGBM means 11 to 13. One is useful, one is not.

### What's a good MAE?
There's no universal number — it depends on the problem. You judge it against:
1. Your baseline — is the new model better?
2. Real-world usefulness — is 1 minute of error acceptable for a taxi app?

---

## R² — R-Squared

### The key question R² asks
> "How much better is my model than just guessing the average every single time?"

### The dumbest possible model

Imagine a model that always answers with the average trip duration — no matter what the distance, hour, or location.

```
Average duration = 11.5 minutes
Stupid model answer for every trip = 11.5 minutes
```

This is your floor. Any real model must beat this.

### What the number means

| R² | What it means |
|---|---|
| 1.0 | Perfect — predicted every trip exactly right |
| 0.978 | Explains 97.8% of why trips differ in duration |
| 0.589 | Explains 58.9% of why trips differ in duration |
| 0.0 | Exactly as good as guessing the average every time |
| Below 0 | Worse than guessing the average (something is very wrong) |

**Your results:**
- LinearRegression R² = **0.589**
- LightGBM R² = **0.978**

### A concrete example

Your data has two trips:
- Trip A: 2 miles, 3am, no traffic → 6 minutes
- Trip B: 2 miles, 8am, Midtown → 35 minutes

**Stupid model:** guesses 11.5 min for both. Very wrong.
**LinearRegression:** guesses 10 min and 20 min. Better, but misses the rush hour effect.
**LightGBM:** guesses 6.5 min and 34 min. Almost exactly right.

R² captures how much of the gap between Trip A and Trip B your model explains. LightGBM explains almost all of it (0.978). LinearRegression only explains about half (0.589).

---

## Why LightGBM Won by Such a Large Margin

Linear models assume straight-line relationships. They think "more distance = proportionally more time." But reality is not a straight line:

- A 2-mile trip at 3am = 6 minutes
- A 2-mile trip at 8am in Midtown = 35 minutes

LightGBM builds a tree of decisions:
```
Is it rush hour?
  Yes → Is it Manhattan?
          Yes → Is distance > 1 mile?
                  Yes → predict 30+ minutes
```

Those branching decisions capture what linear models can't. That's why the gap is so large.

---

## Why We Used Log Duration

Our code does this:
```python
y = np.log1p(duration_sec)    # before training
preds = np.expm1(preds)       # after predicting
```

Duration is right-skewed — most trips are short, a few are very long. If you train on raw duration, a few 3-hour trips dominate the error calculation and the model obsesses over them.

`log1p` compresses the scale so outliers don't dominate. The model focuses on getting typical trips right instead of rare extreme ones. This was a direct result of what we found in Day 2's EDA.

---

## MLflow — Why You Logged Everything

Without MLflow, after 10 experiments you forget what you tried. Did you use 500 trees or 1000? What learning rate? What was the score?

MLflow stores every run automatically:
- Which hyperparameters you used
- What score it got
- The saved model file

You can open `http://localhost:5000` and see every experiment in a table — compare them side by side, click into any run, and download any model.

---

## Optuna — Automatically Finding the Best Settings

### What's a hyperparameter?

A **parameter** is something the model learns from data. LightGBM adjusts thousands of internal numbers during training — those are parameters.

A **hyperparameter** is a setting *you* choose before training even starts. It controls *how* the model learns. Examples:

| Hyperparameter | What it controls |
|---|---|
| `num_leaves` | How complex each tree can be. Higher = more detailed, higher overfit risk |
| `learning_rate` | How big each update step is. Lower = more careful, needs more trees |
| `min_child_samples` | Minimum trips per leaf. Higher = smoother, less overfit |
| `feature_fraction` | What fraction of features each tree sees. Lower = more variety |

The model can't choose these itself — that's your job. And the wrong settings can make the difference between R²=0.95 and R²=0.98.

### The old way — Grid Search

You could try every combination manually:
```
Try learning_rate=0.01, num_leaves=31 → score
Try learning_rate=0.01, num_leaves=63 → score
Try learning_rate=0.01, num_leaves=127 → score
... 100+ combinations later ...
```

That's called Grid Search. It's exhaustive and very slow. If you have 9 hyperparameters each with 5 options, that's 5⁹ = nearly 2 million combinations. Not practical.

### The smart way — Optuna's Bayesian Search

Optuna doesn't try everything blindly. It learns from each attempt:

```
Trial 1:  learning_rate=0.08, num_leaves=200 → R²=0.971
Trial 2:  learning_rate=0.03, num_leaves=150 → R²=0.975   ← better! Focus here
Trial 3:  learning_rate=0.02, num_leaves=160 → R²=0.976   ← still improving
...
Trial 30: learning_rate=0.019, num_leaves=155 → R²=0.978  ← final best
```

Each trial looks at what worked before and chooses the next combination that's most likely to improve. It's like a researcher who reads past experiment notes before designing the next one — not someone who just tries random settings.

This is called **Bayesian optimization**. 30 smart trials beat 300 random ones.

### Why we used 300,000 rows for tuning (not 2.1 million)

Running 30 trials × 5 folds × 2.1M rows = training 150 models on 1.7M rows each. That would take 8+ hours.

Hyperparameters that work well on a 300k random sample generalize to the full dataset — the patterns are the same, there's just less of them. We tune fast on the sample, then train the final model once on all 2.1M rows with the best settings found.

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

---

# The Through-Line Across All Four Days

```
Day 1:  Raw messy data   →  Clean usable data       (removed 9.3% bad rows)
Day 2:  Numbers          →  Visual understanding    (found log transform + rush hour signal)
Day 3:  Raw columns      →  Meaningful signals      (20 columns → 36 engineered features)
Day 4:  Features         →  Predictions             (MAE 6.1 min → 1.0 min)
```

Each day built on the previous one:
- Skip Day 1 → model trains on garbage
- Skip Day 2 → you wouldn't know to use log transform
- Skip Day 3 → model misses rush hour patterns entirely
- Skip Day 4 → you have features but no model

The whole pipeline is a chain. Every link matters.
