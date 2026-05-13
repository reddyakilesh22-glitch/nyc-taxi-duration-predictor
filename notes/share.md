# Sharing Copy

Drop-in copy for LinkedIn, dev.to, your portfolio site, and recruiters.
Swap `LIVE_URL` and `REPO_URL` with the real ones before posting.

Replacements:
- `LIVE_URL` = the Streamlit Community Cloud URL
- `REPO_URL` = `https://github.com/reddyakilesh22-glitch/nyc-taxi-duration-predictor`

---

## 1. LinkedIn post (long-form, ~280 words)

Use this as the primary share. Personal voice, headline result, one
unexpected finding, link to the demo.

```
I spent the last week building an end-to-end ML project I'm actually
proud of: an NYC taxi trip duration predictor trained on 2.38 million
real January 2026 yellow taxi trips.

You pick a pickup zone, a dropoff zone, an hour, and a day. The model
returns:

  - Estimated duration (from LightGBM, ~1 minute average error on
    unseen test data)
  - Typical fare (median of actual trips paid on that route)

Try it: LIVE_URL

The honest version of the story:

1) LightGBM hit R² = 0.98 and MAE = 0.9 min on test. Solid.

2) Linear regression hit R² = 0.76 in log-space but MAE = 77 minutes
   in seconds-space, because a small log-space error explodes after
   exp() back-transform. This is the actual reason most production
   duration / ETA models use trees, not linear regression. Documented
   it as a feature, not a bug.

3) I burned 3 hours running 30 Optuna trials. The tuned model was
   marginally worse than the default. Kept this in the writeup
   because pretending tuning always helps is selling a fairy tale.

The repo: REPO_URL

Stack: Python, Pandas, LightGBM, Scikit-learn, Optuna, MLflow,
Streamlit, Plotly, Docker, GitHub Actions, Streamlit Community Cloud.

What I'm taking away: a deployable model is 30% of the work. The
other 70% is data quality, feature engineering, deciding what to
show users, and being honest about what didn't work.

Open to data science / ML engineering roles. DMs open.
```

---

## 2. Short LinkedIn post (~120 words, for daily-feed posts)

Use this as a quicker share or as a follow-up in the comments of the
long post.

```
Shipped an end-to-end ML project this week.

Predicts how long any NYC taxi trip will take, plus the typical fare,
before the meter starts. Trained on 2.38 million real January 2026
trips, ~1 minute average error.

Pick two zones, pick a time, get an answer. Live demo:

LIVE_URL

Built with LightGBM, Streamlit, Docker, and GitHub Actions CI.

The most interesting finding: 30 Optuna tuning trials didn't beat the
default hyperparameters. Wrote that into the project instead of
hiding it.

Repo: REPO_URL
```

---

## 3. Twitter / X thread (5 tweets)

Use this if you're sharing on X.

```
1/  I just shipped an end-to-end ML project: NYC taxi trip duration
    predictor. Real Jan 2026 data, 2.38M trips, ~1 minute average
    error.

    Pick two zones + a time. Get the duration AND the typical fare.

    LIVE_URL

2/  The pipeline:

    Raw TLC parquet (3.7M rows)
       -> cleaning (removed 36% bad rows)
       -> 34 engineered features
       -> LightGBM model
       -> Streamlit app
       -> Docker + tests + CI
       -> public live URL

    7 days, end to end.

3/  Most interesting finding: linear regression hit R² = 0.76 in
    log-space, but MAE was 77 MINUTES in seconds.

    Reason: small log-space errors get exponentiated into huge
    seconds-space errors after expm1.

    This is the actual reason production duration models use trees.

4/  Other honest finding: I ran 30 Optuna hyperparameter trials.
    They didn't beat the defaults.

    Kept that in the project writeup because pretending tuning
    always wins is a lie. Good features + good defaults beat blind
    search almost every time.

5/  Stack: Python, Pandas, LightGBM, Optuna, MLflow, Streamlit,
    Docker, GitHub Actions.

    Code: REPO_URL
    Live: LIVE_URL

    Open to ML / data engineering roles. DMs open.
```

---

## 4. Resume / portfolio bullet (~50 words)

For the project section of your resume or a portfolio site card.

```
NYC Taxi Trip Duration Predictor (Python, LightGBM, Streamlit, Docker)

End-to-end ML pipeline: cleaned 3.7M raw TLC trips down to 2.38M,
engineered 34 features, trained and compared three models, deployed
a 4-page interactive web app with live duration + fare estimation.
Test MAE under 1 minute (R² 0.98). Live at LIVE_URL.
```

---

## 5. Cold email to a hiring manager (~100 words)

Use this when you're emailing someone specific. Personalize the
opening line.

```
Hi <name>,

I came across your team's work on <something specific> and wanted to
share an ML project I just shipped that might be a useful starting
point for a conversation.

It's a duration + fare predictor for NYC taxi trips, trained on
2.38 million real January 2026 records. End-to-end: data cleaning,
feature engineering, model comparison, deployment, CI.

Live demo (try it): LIVE_URL
Code: REPO_URL

I document the parts that didn't work (Optuna tuning didn't beat
defaults; linear regression has a log-target failure mode) alongside
what did. Would love your read on it.

Best,
Akilesh
```

---

## 6. README CTA (one line)

For the top of the README, replacing the current "_coming soon_"
placeholder.

```markdown
**Live demo:** [LIVE_URL](LIVE_URL)
```

---

## Posting tips

- LinkedIn algorithm rewards comments. Reply to your own post in the
  first hour with the short version (#2) as a follow-up. People who
  see the comment but missed the post will read it there.

- Always paste a screenshot in the LinkedIn post. The new Overview
  with the dual duration + fare card is the right one. The platform
  surfaces image posts to more people than link-only posts.

- Tag 1 or 2 specific people who'd find it relevant. Don't spray-tag.

- Best posting times: Tuesday-Thursday, 8am to 10am your local time.
- Avoid Mondays (everyone is catching up) and Fridays (everyone is
  checked out).

- Keep the first two lines hook-worthy because LinkedIn truncates
  after that on mobile. The opening of post #1 above ("I spent the
  last week building...") is designed to survive the truncation.
