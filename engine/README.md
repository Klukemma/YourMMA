# MMA Prediction Engine v4

Ensemble fight predictor over 8,229 UFC fights (2000-11-17 → 2025-12-06),
converted from `MMA_predict_v4_full.ipynb` into a runnable script.

## Running it

```bash
pip install -r engine/requirements.txt
python engine/predict_card.py
```

Edit the **USER SETTINGS** block at the top of `engine/predict_card.py` to set
`EVENT_NAME`, `EVENT_DATE`, `FIGHT_CARD` and `FIGHT_CONTEXTS`, then re-run.
Each run appends its picks to `engine/data/prediction_history.json`.

A full run takes a few minutes. Set `RUN_OPTUNA = True` for hyperparameter
tuning (much slower); it defaults to pre-tuned parameters.

## Configuration

Secrets and paths come from the environment — copy `.env.example` to `.env`:

| Variable | Purpose |
| --- | --- |
| `ODDS_API_KEY` | [The Odds API](https://the-odds-api.com/) key. Without it the engine still predicts, but the odds / edge / value columns stay empty. |
| `UFC_CSV` | Override the fight-history CSV path. |
| `PREDICTIONS_LOG` | Override the prediction log path — useful for test runs you don't want recorded. |

**Never hardcode the API key.** This repository is public.

## Model

Win probability is an ensemble of LogisticRegression + RandomForest + XGBoost
with Platt calibration, on a 70/15/15 temporal train/cal/test split. Separate
heads predict method (KO/TKO, Submission, Decision) and round.

Measured on the held-out test set:

| Target | Accuracy | Brier | Log loss |
| --- | --- | --- | --- |
| Win | 73.3% | 0.186 | 0.553 |
| Method | 48.5% | | |
| Round | 59.6% | | |

Walk-forward evaluation (expanding window, 2015–2025) averages **69.7%**, which
is the more honest number — the single test-set figure benefits from a
favourable slice. Per-year accuracy ranges 65.2%–72.9% with no downward drift.

Seven automated leakage audits run on every execution (rolling-feature shifts,
pre-fight streaks, layoff dates, target columns, per-fight outcome stats,
time-travel correlation, Bayesian skill priors). All currently pass.

### Features

TrueSkill mu/sigma skill ratings with uncertainty, MMR, L3/L5 rolling averages
and momentum, age curves weighted by weight class, layoff and ring-rust
transforms, win/loss streaks, finish rates, durability, cardio, cage control,
submission defense, opponent quality (strength of schedule), damage
accumulation, weight-class movement, stance matchups, and fighter archetype
classification.

Optional per-fight context: home/away, cage size, altitude, short notice.

## Files

| Path | What it is |
| --- | --- |
| `predict_card.py` | The engine. Settings block up top, everything else below. |
| `compare_fighters.py` | Debug helper — side-by-side fighter stats. Needs an interactive session that already ran `predict_card.py`. |
| `data/UFC_with_mmr_rebuilt_dedup.csv` | Fight history, 130 columns per bout. |
| `data/prediction_history.json` | Logged predictions and running accuracy. |

## Known issues

- **Unknown names are silently fuzzy-matched to the wrong fighter.** A name
  absent from the CSV resolves to the closest match instead of erroring, and
  the engine then predicts that fight with full confidence. In the committed
  example card, `Leon Shahbazyan vs Levan Chokheli` resolved to
  `Cameron Saaiman vs Levan Makashvili`, and `Shane Collins` to `Jared Rollins`.
  Always check the names echoed in the output against the card you entered.
- **All 270 logged predictions are still `pending`** — `update_prediction_result()`
  exists but has never been called, so the tracked accuracy is unmeasured.
- The dataset ends 2025-12-06, so any 2026 fight is predicted from stats that
  omit the fighter's most recent bouts.
