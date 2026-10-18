# YourNFL

NFL game predictions, built the way the MMA engine next door is built: point-in-time
features, a model measured walk-forward against the closing market, picks logged before
kickoff and graded afterwards, and a phone app that shows the numbers with their measured
record beside them.

The phone app is at **https://klukemma.github.io/YourMMA/nfl/app/**. Open it on a phone and
use Add to Home Screen. `nfl/app/yournfl.html` is a single offline copy of the same app.

## How a pick is made

| Layer | What it is | Measured, 2016-2025 (walk-forward) |
|---|---|---|
| Elo | 538-style team ratings with a margin-of-victory multiplier and offseason regression | 64.0% of winners |
| Model | logistic regression on Elo, offensive/defensive EPA per play, the starting QB's EPA per dropback, rest, byes, short weeks, time zones, division and neutral-site flags | 64.3% |
| Pick | the model blended with the no-vig market price; the blend is fitted on earlier seasons' out-of-sample predictions | 66.7% (closing favourite: 66.6%) |

The confidence bands (STRONG 75%+, SOLID 65%+, LEAN 55%+, TOSS-UP) are probability bands,
and each one's hit rate is measured on 2016-2025 and printed in the app. No game is ever
labelled a bet. Backing the model wherever it disagreed with the closing moneyline made a
little in 2006-2015 and lost 7-10% in 2016-2025, so that label is switched off by
`nfl_backtest.value_gate`. It turns itself on only if a re-measured edge is profitable in
both periods.

## The rules it keeps

- **Point in time.** Games are replayed in kickoff order. A game reads every team's and
  quarterback's state before kickoff, and shared state (the league scoring average) waits
  for a strictly later kickoff. `tests/test_nfl_features.py` hides every result from a
  kickoff onwards, rebuilds, and checks the features are identical.
- **Walk-forward.** Season S is predicted by models fitted only on seasons before S. The
  live week is predicted by the very object the backtest used for its season.
- **Logged before kickoff.** A pick is written to `nfl/data/prediction_history.json` while
  its game has not started and is frozen once it does. The live record grades only those
  picks.
- **Shown, not applied.** Injury reports appear under each game. The one structural input
  is the starting quarterback: if the schedule's projected starter is listed Out, the model
  uses the team's last starter instead, and the app says so.
- **Weather is left out.** The schedule records temperature and wind after the game, and no
  forecast was ever stored.

## Files

    nfl/engine/nfl_data.py        nflverse snapshot: sync + loaders (franchise codes, UTC kickoffs)
    nfl/engine/nfl_elo.py         Elo
    nfl/engine/nfl_features.py    one kickoff-ordered pass -> pre-game features
    nfl/engine/nfl_model.py       winner / margin / total models, market blend, walk-forward
    nfl/engine/nfl_backtest.py    protocol, metrics, confidence bands, value gate
    nfl/engine/nfl_odds.py        the-odds-api (ODDS_API_KEY) with cache; schedule lines as fallback
    nfl/engine/nfl_form.py        record and last five beside each team
    nfl/engine/nfl_predict.py     the weekly run: picks, log, grading, app data
    nfl/engine/nfl_export.py      app payloads and caveats
    nfl/engine/nfl_standalone.py  the offline single-file app
    nfl/data/                     snapshots (games, team games, QB games, injuries), odds cache, pick log
    nfl/app/                      the phone app (PWA)
    .github/workflows/nfl.yml     runner: predict / sync / backtest, on a schedule in season

## Running it

    pip install -r engine/requirements.txt
    python nfl/engine/nfl_data.py --full      # rebuild every season's snapshot
    python nfl/engine/nfl_predict.py          # sync this season, predict the week, write the app
    python nfl/engine/nfl_backtest.py         # the walk-forward report
    python -m pytest nfl/engine/tests

On GitHub, the **NFL predictions** workflow runs `predict` on Tuesday mornings, Thursday
afternoons and Sunday mornings during the season, and commits the app data. The Thursday and
Sunday runs buy fresh prices (3 credits each). Run it by hand from the Actions tab with
`refresh_odds` to price the slate now.

Data: [nflverse](https://github.com/nflverse) schedules with closing lines, team and player
statistics, and injury reports. For entertainment and analysis, not betting advice.
