# YourSoccer

Predictions for Europe's top five leagues, built the way the MMA and NFL apps are: every
number point-in-time, measured walk-forward against the betting market before it ships,
picks logged before kickoff and graded afterwards, and a phone app that shows the numbers
with their measured record beside them.

The phone app is at **https://klukemma.github.io/YourMMA/soccer/app/**. Open it on a phone and
use Add to Home Screen. `soccer/app/yoursoccer.html` is a single offline copy of the same app.

## Data

football-data.co.uk, snapshotted into `soccer/data/` by `soccer_data.py` (run on GitHub's
machines; the development sandbox cannot reach the site):

- the Premier League, La Liga, Serie A, the Bundesliga and Ligue 1, and their second tiers
  (rated, never predicted - that is where promoted clubs come from), every season since 2005-06;
- each match's result, shots, shots on target, corners and cards;
- two prices: the **pre-match** average, collected Friday afternoon for weekend games and
  Tuesday afternoon for midweek ones - the same moment the fixtures file the app predicts from
  is published - and the **closing** price;
- kickoff times, on the UK clock, converted to UTC.

## What the app shows, and why

Three models were built and measured (`experiments/`), choosing on 2006-2015 and confirming
once on 2016-2025, against the pre-match market with the bookmakers' cut removed:

| | 2006-2015 log loss | 2016-2025 log loss | 2016-2025 results picked |
|---|---|---|---|
| Dixon-Coles (goals, time-decayed) | 0.9920 | 0.9799 | 53.0% |
| Goal-based Elo | 0.9917 | 0.9803 | 53.1% |
| Shots-based ratings (goals, shots on target, shots) | **0.9878** | **0.9754** | 53.3% |
| The market | 0.9816 | 0.9652 | 54.1% |
| **The pick: the market, recalibrated on earlier seasons** | **0.9813** | **0.9645** | 54.0% |

None of the models beat the market, and none added to it: pooled with the market, each lost to
the market pooled with itself on 2006-2015. (Dixon-Coles and Elo did help on 2016-2025, but only
with a negative weight - betting against themselves - which was not there on 2006-2015, so it
was not shipped.) So:

- **The pick** is the market's home/draw/away chances, recalibrated: p is proportional to
  q^beta * exp(c), each season fitted on every earlier season (`soccer_model.pick`). The market's
  favourites win slightly more often than its prices say and the draw is priced slightly short;
  for 2026-27, beta = 1.11 and the draw gets +0.04.
- **The model's own view** - the shots-based ratings, the best of the three - is shown beside
  every pick, labelled "Model alone", and drives the club ratings on the Tables tab. It is never
  blended into the pick. Where no price was posted it is all there is, and the app says so.
- **Goals:** the over/under 2.5 market with the cut removed. The model's own chance is shown
  beside it (log loss 0.6752 against the market's 0.6707 on 2016-2025).
- **No match is labelled a bet.** Betting the model wherever it disagreed with the price, at the
  best price any book posted, made money on 2006-2015 and lost 5-7% on 2016-2025.

The 2016-2025 pick log loss by league: Premier League 0.9520 (market 0.9520), La Liga 0.9660
(0.9673), Serie A 0.9471 (0.9491), Bundesliga 0.9794 (0.9788), Ligue 1 0.9823 (0.9828).

## Point in time

Every model predicts a match only from matches on earlier days, checked by
`soccer_backtest.check_point_in_time` (which scrambles every result from a date on and requires
that date's predictions not to move) on dates in both periods, and re-checked by an independent
judge with a stricter version (prices and match statistics scrambled too). The recalibration
for a season is fitted on earlier seasons only. A pick is logged before kickoff and frozen when
the match starts (`soccer_log.py`); the live record grades only those.

## The runner

`.github/workflows/soccer.yml` runs `soccer_predict.py` on Friday and Tuesday evenings, when the
fixtures and prices are posted, on Saturday and Sunday mornings before kickoff, and on Monday
and Thursday to grade the round just played (August to May). One run: sync this season from
football-data.co.uk, run the model over every match since 2005-06 (about 15 seconds), recalibrate
the market, log the week's picks, grade the finished ones, and write `soccer/app/data/` and the
standalone page.

    python soccer/engine/soccer_predict.py --no-sync   # locally, on the snapshot as it is
    python -m pytest soccer/engine/tests -q
