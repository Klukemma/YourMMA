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
| `ODDS_API_KEY` | [The Odds API](https://the-odds-api.com/) key. Without it the engine still predicts, but the odds / edge / value columns stay empty. Free tier is 500 credits a month; a credit is `markets x regions` and this engine asks for one of each, so that is 500 real calls. Verify a key with `python3 engine/check_odds.py`, or the `check-odds` workflow mode. |
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

### The model does not beat the market

`edge.py` compares the model's probability against the bookmaker's own, on the
same 65 fights, with the market de-vigged first.

| | Brier | log loss |
| --- | ---: | ---: |
| model | 0.2661 | 0.7417 |
| **market (de-vigged)** | **0.2057** | **0.5993** |

The market is substantially better calibrated. And the decisive number:

**On the 24 fights where the model and the market picked different fighters,
the model's pick won 9 — 37.5%.**

If the model held information the market lacked, its disagreements should win
*more* often than the market's. They win considerably less. Those
disagreements are not edge; they are the model being wrong in a way the market
already knows about.

| segment | n | won | model said | market said | ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| backing a favourite | 44 | 70.5% | 73.6% | 68.8% | −0.4% |
| backing an underdog | 21 | 28.6% | 68.3% | 36.4% | **−26.7%** |
| heavy favourites (≤ −300) | 22 | 81.8% | 73.9% | 77.9% | +0.7% |
| long underdogs (≥ +200) | 7 | 14.3% | 61.3% | 27.6% | **−41.4%** |

The underdog rows are where it breaks. The model thinks a +200 underdog wins
61% of the time; they won 14%. Backing favourites is roughly break-even, but
that is just following the market and paying the vig.

Caveats: 65 fights, half the picks unpriced, and the source does not say
whether these are opening or closing lines. The direction is consistent across
every segment and the disagreement result points the wrong way for an edge
thesis, so the burden of proof sits with anyone claiming otherwise.

### What the 2026 drop actually was, in part

`auc_by_year` walks a model forward year by year, training on everything before
each year and scoring it:

    2015-2025   AUC 0.749-0.826, mean 0.776, std 0.023
    2026        AUC 0.646  ->  5.6 standard deviations below its own mean

Eleven years of real ranking ability then a cliff, which is a break rather than
a model that never worked. Three candidate explanations were tested and two
were ruled out.

**Not the blank window.** The dataset has a four-month hole where every
striking, takedown and control statistic is missing (2025-09-13 to 2025-12-06,
145 fights, October and November 100% blank). 56.6% of 2026 fights involve a
fighter who fought inside it, which looked decisive. It is not:

    2026, split by exposure       n   accuracy    AUC
      neither fighter           148      60.1%  0.637
      one fighter               142      63.4%  0.653
      both fighters              51      62.7%  0.660

Unexposed fights rank slightly *worse*. A placebo running the same split on
2025 against the equivalent 2024 window separates by +0.027, confirming the
method rather than the hypothesis.

**Not missing fight history.** Bucketing by the less-experienced fighter's
record, against 2024 as a reference:

    bucket                2026            2024
    debut            54.3%  0.650     82.1%  0.892
    1-2 prior        61.9%  0.664     69.6%  0.795
    3-5 prior        63.8%  0.630     72.6%  0.823
    6+ prior         67.7%  0.694     69.4%  0.817

Debutants were the model's *best* bucket in 2024 at AUC 0.892, so non-UFC
history cannot improve them. And 2026 is worse in every bucket including 6+,
where extra history adds nothing. This is what closed Path B in its original
form.

**Partly a unit mismatch, which was ours.** `schema_map` mapped
`r_reach_inches` onto `r_reach` and `r_weight_lbs` onto `r_weight` as straight
renames into columns holding metric. Height arrived as `6' 3"` into a
centimetres column and dob with a different separator:

    r_reach    before 181.88     after  71.54     (inches -> cm)
    r_weight   before  72.20     after 160.60     (lbs -> kg)

358 rows, every one this project synced. Fixed in `transform`, with `fix-units`
converting the rows already written (2,864 cells) and `sync` now refusing a
column that sits 2+ SD from existing rows. Worth **+0.019 AUC**:

    2026 before   61.9%   AUC 0.646   Brier 0.2433
    2026 after    63.0%   AUC 0.665   Brier 0.2366

That is about 15% of the gap. The rest is still open.

### The professional record is missing from every synced row

`adversarial_rows` trains a classifier to tell our rows from the original
builder's, with a same-builder pair as control:

    CONTROL  2024 vs 2025 (same builder)   AUC 0.477
    TEST     2025 vs 2026 (transform)      AUC 0.670   excess +0.193

    top distinguishing columns:
      b_losses  b_wins  r_wins  r_losses  b_draws  r_draws

Same-builder rows are interchangeable. Every column still giving 2026 away is
the professional win/loss/draw record: 69.9% of 2026 debutants have none, where
the same bucket in 2024 has none missing.

`carry_forward_records` carries each fighter's record forward from local history
and leaves a fighter it has never seen blank rather than assuming 0-0. That is
correct given its inputs - `inspect-fighter` confirms upstream's `fighter.csv`
carries only `height, dob, stance, str_acc, sapm, str_def, td_avg, td_acc,
td_def, sub_avg` and **no record column at all**.

So the record has to come from somewhere else, and this is the one place a
non-UFC source is genuinely needed - not to model debutants better, which 2024
shows was never the weakness, but because the record counts bouts outside the
UFC and 358 rows are structurally unlike the other 8,229 without it.

Candidates found by `search-mma`, not yet inspected for a record column or for
name overlap with the 2,584-fighter roster:

    binduvr/pro-mma-fighters                    273KB  2021-08-12
    cullenwatson/every-ufc-bellator-one-fc-pfl  151KB  2023-10-29
    leandroiber/mmastats                        9.2MB  2026-06-02

### The MMR formula costs more than it adds

`mmr = mu - 3*sigma`. Sweeping the constant, as AUC of the rating difference
alone:

    k       2021    2022    2023    2024    2025    2026
    0      0.558   0.593   0.574   0.621   0.587   0.603
    1.0    0.538   0.578   0.548   0.593   0.549   0.543
    2.0    0.520   0.559   0.519   0.564   0.523   0.497
    3.0    0.504   0.542   0.498   0.541   0.506   0.466

**k=0 wins in every year and k=3 loses in every year.** Sigma spans 2.54-8.49,
so the penalty swings 7.6-25.5 points against a mu spread of 14-48 and
dominates the rating. The rating the model actually consumes is below chance in
2026 (0.466) while the skill estimate underneath it is at 0.603.

This is an in-sample sweep and needs a walk-forward confirmation before the
constant changes, but monotonicity across eight independent years is not a
selection artifact.


### The features are not the constraint

Three rounds of correctness work, measured the same way each time:

    unit mismatch (reach, weight, height)   2026 AUC  0.646 -> 0.665   +0.019
    Elo-era rating constants                          0.665 -> 0.666   +0.001
    full feature rebuild                              0.666 -> 0.669   +0.003

The rebuild was not cosmetic. It fixed the fill rule on fifteen features,
where a missing percentage became 0 and told the model a debutant lands
nothing; it added levels, known flags and eight matchup interactions the
old difference-only design could not express at all; it computed the
trajectory features that had been a constant; and it corrected an ape index
that read 0.39 or 2.60 instead of 1.02 on 9.4% of fights. 63 features became
111. Nine of eleven historical years improved and 2024 accuracy rose from
72.1% to 75.8%.

The headline is still +0.002 on the historical mean.

    2015-2025 mean AUC   0.777 -> 0.779
    2026 AUC             0.666 -> 0.669

2026 remains about 0.11 below its own history, and that gap has now survived
adjustment layers, recalibration, the blank data window, missing fight
history, unit errors, rating constants, and the whole feature layer. Every
one of those was a real defect and fixing it was right. None of them was the
reason.

What that leaves is information rather than encoding. On the same 2026
fights the market scores AUC 0.741 against the model's 0.669, and it is
better in 98% of bootstrap resamples. The market prices things this dataset
does not contain: camp changes, injuries, weight-cut trouble, late
replacements. No amount of rearranging the columns we have will recover
what was never recorded.


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
- **No professional record on synced rows.** 69.9% of 2026 debutants have no
  win/loss/draw record, and `adversarial_rows` makes it the dominant structural
  difference between our rows and the original builder's. Upstream carries no
  record column, so it needs another source (above).
- **`mmr = mu - 3*sigma` is worse than `mu` alone** in every year measured
  (above). The constant has not been changed yet, pending a walk-forward test.
- **One bout is still missing its statistics**, `2025-10-04 Ateba Gautier vs
  Tre'ston Vines`, which `repair` could not match upstream. It refuses rather
  than guessing.
- `predict_fight_prod()` is called three times per fight per run (card table,
  compact summary, bettability analysis). Worth caching before this sits behind
  an API.
