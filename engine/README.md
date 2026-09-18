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

## Name resolution

The engine refuses to predict a fighter it cannot identify. It will **not**
substitute a similar name or fall back to invented "average fighter" stats.

Resolution order: curated alias → exact match → all-tokens match → unique
surname → high-confidence typo. A typo is only auto-accepted at
`FUZZY_AUTO_ACCEPT_RATIO` (0.90) **and** when the surname matches exactly, so a
wrong first name can never pull in a different fighter. Anything else returns
`NO DATA` with ranked suggestions.

Settings live in the USER SETTINGS block:

| Setting | Default | Effect |
| --- | --- | --- |
| `STRICT_NAMES` | `True` | Refuse unidentifiable fighters. Off = best guess with a loud warning. |
| `FUZZY_AUTO_ACCEPT_RATIO` | `0.90` | Typo acceptance threshold (surname must also match). |
| `FUZZY_SUGGEST_RATIO` | `0.60` | Threshold for "did you mean...". |
| `MIN_FIGHTS_FOR_PREDICTION` | `1` | Refuse fighters with fewer recorded bouts. |
| `LOW_DATA_FIGHT_COUNT` | `3` | Below this, predict but flag as thin data. |

If a fighter is simply spelled differently in the dataset, add a mapping to
`data/fighter_aliases.json` rather than loosening the thresholds:

```json
{ "Bones Jones": "Jon Jones" }
```

## Tests

```bash
pip install pytest
python -m pytest engine/tests/ -q
```

23 tests covering resolution, refusals, ranking, aliases and search. They run
against the real fighter list from the CSV, not a fixture.

## Data pipeline

`sync_kaggle.py` pulls [the upstream dataset](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
and merges fights newer than the local file.

```bash
python engine/sync_kaggle.py inspect        # report upstream schema, change nothing
python engine/sync_kaggle.py sync --dry-run # report what would be added
python engine/sync_kaggle.py sync           # merge, rate, back up, write
```

Run `inspect` first. If upstream column names differ, fill in `COLUMN_MAP` in
that file rather than letting it guess.

It also runs as a GitHub Action (`.github/workflows/update-dataset.yml`),
weekly and on demand — Actions → Update UFC dataset → Run workflow, which works
from the GitHub mobile app. Needs a `KAGGLE_API_TOKEN` repository secret.

### Ratings

Upstream has no rating columns, so new fights get them from `ratings.py`.
Two things were measured from the shipped data rather than assumed:

- **`mmr_pre == mu_pre - 3 * sigma_pre`, exactly** (max error 0.0 over all
  8,229 rows). MMR is the TrueSkill conservative rating, not a separate Elo.
- **The ratings were built with β ≈ 5.0**, not the `TRUESKILL_BETA = 4.17` that
  `predict_card.py` uses. Fitted against 13,725 consecutive-bout pairs, β=5.0
  and τ=0.2 reproduce 97% of updates within 0.05 of mu; β=4.17 has a median
  error 17× larger.

New fights continue each fighter's existing rating series instead of recomputing
history, so rows the model was trained on never move. Exact historical
reproduction is not possible from this file: same-day bouts have no recorded
order, and the dedup step appears to postdate the rating build.

## Measured performance

Backtest, walk-forward (retrains per year on all prior fights):

| year | 2022 | 2023 | 2024 | 2025 | **2026** |
| --- | --- | --- | --- | --- | --- |
| accuracy | 70.6% | 73.0% | 73.5% | 72.9% | **60.4%** |
| Brier | 0.186 | 0.182 | 0.178 | 0.187 | **0.241** |

Live, graded against real results (145 unique bouts, 2025-12 to 2026-06):
**59.3%**, Brier 0.264. See `grade.py`.

**The live number matches the 2026 backtest.** The model is not degrading in
production - 2026 is simply harder for it than any year since 2017. Comparing
live 59.3% against the 69.7% eleven-year walk-forward mean is the wrong
comparison; against 2026's own 60.4% it is in line.

### What the drop is not

`experiments/adjustment_ablation.py` scores each layer of the live stack on
the same bouts, with the model trained only on pre-cutoff data:

| variant | accuracy | Brier |
| --- | --- | --- |
| raw ensemble (LR+RF+XGB) | 58.5% | 0.2583 |
| + Platt calibration | 59.2% | 0.2688 |
| + context adjustment | 59.9% | 0.2597 |

All within 1.4 points. The adjustment layers are not the cause, and the raw
ensemble is just as miscalibrated (its 80-90% band wins 61.5%), so the
overconfidence comes from the base model rather than from anything layered on
top.

### What it looks like instead

Accuracy on the graded bouts, split by the less experienced fighter's prior
UFC bouts:

| prior bouts | n | accuracy |
| --- | --- | --- |
| 0 (debut) | 2 | 50.0% |
| 1-2 | 45 | 57.8% |
| 3-5 | 41 | 56.1% |
| **6+** | **53** | **66.0%** |

And 2026 has far more thin-history bouts than recent years: **27.7%** of 2026
fights involve a UFC debutant, against 16.5% across 2024-2025. So the harder
cases are both more common and much less predictable. On experienced matchups
the model still performs near its historical level.

### Recalibration was tried and did not help

`experiments/calibration_selection.py` picks a calibration recipe on 2025 and
confirms it once on 2026, so the recipe is not fitted to the period used to
judge it. Earlier runs that scored every variant directly on 2026 looked far
more promising; that gap is what selecting on the test set buys you.

| recipe | accuracy | Brier | ECE |
| --- | --- | --- | --- |
| no calibrator | 58.0% | 0.2343 | 0.112 |
| production (all history, C=1e10) | 59.4% | 0.2407 | 0.123 |
| best chosen on 2025 (all, C=1) | 59.1% | 0.2364 | **0.109** |

ECE 0.123 to 0.109 is not worth a production change, and on the 2025 selection
period the **uncalibrated** ensemble was the best calibrated of all (ECE 0.034),
so calibration was adding error there rather than removing it.

The reading: 2026 miscalibration cannot be fixed by refitting on pre-2026 data,
because the problem is that 2026 differs from everything before it. No
calibrator fitted on the past anticipates that. What would work is refitting as
results arrive - now possible, since `grade.py` closes that loop.

### Selective prediction works; the meta-model adds nothing

`experiments/meta_model.py` trains a second model on 2024 to predict whether
the base model's pick is right, chooses a variant on 2025, and looks at the
held-out period once.

On the 352 held-out bouts, accuracy over the top slice by each ranking:

| ranking | top 50% | top 30% | top 20% |
| --- | --- | --- | --- |
| no selection (58.0% base) | 56.8% | 57.5% | 60.0% |
| **the model's own confidence** | **69.3%** | **78.3%** | 81.4% |
| meta-model (GBM) | 68.2% | 75.5% | 82.9% |

**Skipping fights is worth far more than any modelling.** The base model is 58%
on everything and 78% on the third of fights it is most sure about.

**The meta-model does not beat sorting by confidence.** Its single largest
feature is `confidence` at 0.466 importance - it mostly rediscovers the sort.
Not worth the machinery.

Simple thresholds on the held-out period:

| rule | coverage | accuracy |
| --- | --- | --- |
| confidence >= 0.2 | 68.5% | 64.7% |
| confidence >= 0.4 | 43.8% | 72.7% |
| confidence >= 0.6 | 23.0% | 79.0% |

**Accuracy is not edge.** High-confidence picks are heavy favourites, and 79%
on fighters the market prices at -400 still loses money. Nothing here has been
measured against odds; these figures say which picks are *reliable*, not which
are *profitable*. That needs the odds data joined to the graded results.

### The experience gate does not replicate

An earlier look at the 145 graded live predictions showed 66.0% where both
fighters had 6+ prior UFC bouts. On the full held-out set with a fixed base
model that does not hold: the gate covers 27.6% of bouts at **59.8%**, against a
58.0% base. The original figure came from a smaller sample and a differently
trained model. Experience is a weak selector on its own; confidence is the
strong one.

### ROI: the picks lose money

`roi.py` settles one unit per pick against historical odds
(`data/odds.csv`, fetched by `sync_kaggle.py fetch-odds`).

From 2026-01-01, of 130 graded picks the 65 that could be priced:

| strategy | bets | won | ROI |
| --- | ---: | ---: | ---: |
| every pick | 65 | 56.9% | **−8.9%** |
| confidence >= 0.2 | 52 | 59.6% | −7.0% |
| confidence >= 0.4 | 38 | 60.5% | −9.2% |
| confidence >= 0.6 | 19 | 63.2% | **−13.3%** |

**Every strategy loses, and tightening confidence makes it worse.** The hit
rate rises with confidence (56.9% to 63.2%) while the return falls, because
the confident picks are shorter favourites: you win more often and are paid
less each time, and the second effect is larger. This is what "accuracy is not
edge" means in practice.

Caveats: only half the picks could be priced - the odds source covers 275 of
the ~358 bouts in the window and spells some names differently - and 19 bets at
the tightest filter is a small sample. The direction is consistent across every
tier, which is the part worth trusting.

### Pruning the weak features makes it worse

`experiments/feature_pruning.py`, choosing on 2025 and confirming once on 2026:

| feature set | n | 2025 accuracy | Brier |
| --- | ---: | ---: | ---: |
| **all 57** | 57 | **73.8%** | 0.1863 |
| drop negative-importance | 31 | 70.1% | 0.1869 |
| drop negative and near-zero | 23 | 69.7% | 0.1878 |
| top 20 | 20 | 68.3% | 0.1905 |
| top 10 | 10 | 68.3% | 0.1922 |
| top 5 | 5 | 71.6% | 0.1913 |
| winrate_diff alone | 1 | 69.1% | 0.1984 |

Every reduced set is worse. Dropping the 26 features with negative or
near-zero importance costs **3.7 points**.

This is not a contradiction of the audit. Permutation importance measures what
one feature adds *given all the others*, so a feature can score zero or
negative and still carry information the ensemble uses in combination with
something else. Scoring near zero individually is not the same as being
useless, and the 56 features around `winrate_diff` are collectively worth about
4.7 points (69.1% alone against 73.8% together).

The practical conclusion: **do not prune, and do not rewrite the weak features
one at a time.** Their individual contributions are too small for that work to
pay, and the measurement says removing them costs accuracy.

## Known issues

- **`TRUESKILL_BETA = 4.17` contradicts the data** (β ≈ 5.0, above). It feeds
  `bayesian_win_prob`, so that feature is computed on the wrong scale.
- **`opp_quality` mixes two scales.** Real values run −2.6 to 31.4, but a
  debutant gets the constant `1500` (`calc_opponent_quality`). That fires on
  **18.8% of fights**, making `opp_quality_diff` effectively a debut flag
  multiplied by ~1486: its overall std is 628.8, versus 6.30 among fights where
  both fighters have history.
- **`base_prob` is nearly constant.** `MMR_SCALE = 120.0` was tuned for an
  Elo-like scale. With actual `mmr_diff` spanning −25…+31, `base_prob` only
  spans 0.448–0.565 (std 0.015), so a documented headline feature carries
  almost no signal.
- **All 270 logged predictions are still `pending`.** Their results are not yet
  obtainable: predictions cover 2025-12-14 → 2026-06-20, the dataset ends
  2025-12-06. Fixed by running the sync.
- `predict_fight_prod()` is called three times per fight per run (card table,
  compact summary, bettability analysis). Worth caching before this sits behind
  an API.
