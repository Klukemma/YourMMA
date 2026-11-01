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
| Model | L1-regularised logistic regression on Elo, offensive/defensive EPA per play, the starting QB's EPA per dropback, rest, byes, short weeks, time zones, division and neutral-site flags | 64.4%, log loss 0.6245 |
| Pick | the model blended with the market's no-vig moneyline and its spread; the blend is fitted on earlier seasons' (from 2003) out-of-sample predictions | 66.4%, log loss 0.6080 (closing favourite: 66.6%, 0.6085) |

The confidence bands (STRONG 75%+, SOLID 65%+, LEAN 55%+, TOSS-UP) are probability bands,
and each one's hit rate is measured on 2016-2025 and printed in the app. No game is ever
labelled a bet. Backing the model wherever it disagreed with the closing moneyline made a
little in 2006-2015 and lost 7-10% in 2016-2025, so that label is switched off by
`nfl_backtest.value_gate`. It turns itself on only if a re-measured edge is profitable in
both periods.

## Players

Every game card also carries:

- **Most valuable players**, for each team. Offence is ranked by expected points added (EPA)
  this season, the points a player's plays added above an average play in the same situation.
  Before week 5, last season's per-game rate fills in. Defenders are ranked by the plays they
  make themselves: sacks, takeaways, passes defended and tackles for loss.
- **Absence alerts.** A listed player who is out or doubtful, or a most valuable player who is
  questionable, leads the card. This is read from the official injury report, the weekly roster
  (injured reserve and game-day inactives) and, once a game is over, its snap counts.
- **Most likely to score a touchdown.** For each team's top three, this is the chance of a
  rushing or receiving TD if the player plays (`nfl_td.py`). It is built from his recent carries,
  targets, snaps and scoring, and his team's expected points from the betting line. Measured
  walk-forward on 2013+ snap-count games against a position base rate (log loss 0.444) and
  usage alone (0.403), it scores 0.395 in 2020-2025. Each team's most likely scorer scored
  43.5% of the time against the 41.1% the model gave, and at least one of the top three scored
  74% of the time. These picks are logged before kickoff and graded from the box score.

  Below the top three, folded, is everyone else a bettor can back to score: every QB, RB, FB, WR
  and TE on the team's latest roster who is active on it - or was a game-day inactive in an
  earlier week, which says nothing about this one (a team on a bye, or a run made before the
  new week's roster is out, would otherwise drop a starter coming back) - less players ruled
  out, off the active roster, or on another team's roster since (`nfl_predict._roster_now`,
  `nfl_td.predict_roster`; the top three draw from the same roster). A chance is printed only for a group the walk-forward measured as
  well calibrated - mean chance within 20% of the scoring rate and log loss under the position
  base rate, in both 2014-2019 and 2020-2025 (`report()["by_usage"]`, re-measured every run).
  The top three's candidates (weighted carries + targets of 1 or more) pass: 1-3 said 9.5% and
  9.7% where 9.0% and 8.4% scored, 3+ said 24.6% and 25.0% where 25.2% and 26.0% did. The
  others do not, so they are listed with a dash: under 0.5 said 4.2% and 3.7% where 3.1% and
  2.7% scored, 0.5-1 said 6.2% where 4.9% scored in 2014-2019, and players with no NFL game said
  3% where 8% scored. A team's last starting quarterback is a candidate at any usage, as
  before; under 1 his chance follows his group like everyone else's (no player under 1 ranked
  in a team's top three in the walk-forward, 6,806 team-games of 2014-2026, so the top three
  are unchanged). The top three are players with a chance only. The whole list is logged
  before kickoff (`home_td_all`, `away_td_all`) and graded as the top three are
  (`grade_touchdown_lists`).
- **Touchdown meters.** Beside every listed player, on each team (Teams, and Record's table) and
  overall (Record), two meters from the live record, as "X of Y" and a percentage:
  - **Top 3 → scored**: of the times he was in his team's logged top three and played, how often
    he scored a rushing or receiving touchdown (a pick who did not play is left out);
  - **Not top 3 → no TD**: of the times he played outside it, how often he did not score.

  For games logged with the whole list, "not top 3" is the rest of that list who played. Games
  whose pick was logged before the list was (the top three only: games of 2026 weeks 6 to 8;
  a week-8 game still to kick off when the list arrived was re-logged with it) rebuild it from
  the snap counts: every other player of the team who took an offensive snap at QB, RB, FB, WR
  or TE, his touchdowns from the box score (none without a box-score line); the app marks each
  week that has such games, and says "in some games" when a week mixes the two. A team's sheet
  counts each player's games for that team only, so its players add up to the team's meters
  (a player who moved teams carries `by_team`).
  They are computed at every run from the log and the box scores (`nfl_export.td_meters`), so
  a game whose snap counts land late counts as soon as they do. A meter with no games is a
  dash, never 0%.
- **Projected yards.** Passing, rushing and receiving yards for each key player, plus each
  team's projected leaders (`nfl_props.py`). Each projection is the middle of what his recent
  usage and his team's expected points point to. Measured 2020-2025:
  - passing: 56 yards off on average, against 67 for the QB's own recent average;
  - rushing: 18 off, against 19;
  - receiving: 18 off, against 20.

  Players went over about half the time, as a middle number should. Each team's leaders are
  logged before kickoff and graded.
- **Player news.** Headlines from public NFL news feeds (`nfl_news.py`, fetched by the runner)
  that name either team or one of their rostered players, published before kickoff.

All of these are shown, not applied. The pick moves only for the starting quarterback.

## What was tried and not shipped

Each family was explored in parallel against the same walk-forward harness. The rule:
choose on 2006-2015, and ship only what does not make 2016-2025 worse for the pick.

- **Shipped, from the learner family:** an L1 winner model, the spread as a third blend
  input, and blend history from 2003. The pick's 2016-2025 log loss went from 0.6087 to
  0.6080, under the market's 0.6085.
- **Model alone better, pick not, so not shipped:**
  - a starting-QB adjustment to Elo with a moving home-field edge;
  - a "QB form gap" feature (the starter against the quarterbacks behind the team's recent
    EPA).

  Both improve the model in both periods. They do not improve the pick in 2016-2025,
  because the closing line already prices quarterback changes.
- **No gain:**
  - EPA variants: decay, pass/rush splits, opponent adjustment, turnover luck;
  - Elo tuning;
  - recency weighting, rolling windows, gradient boosting, calibration;
  - roughly 200 spread and total configurations. None beats -110 in both periods.

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
    nfl/engine/nfl_players.py     most valuable players and whether they play
    nfl/engine/nfl_td.py          anytime-touchdown model, walk-forward and live
    nfl/engine/nfl_props.py       passing/rushing/receiving yard projections
    nfl/engine/nfl_news.py        player news from public RSS feeds
    nfl/engine/nfl_predict.py     the weekly run: picks, log, grading, app data
    nfl/engine/nfl_export.py      app payloads and caveats
    nfl/engine/nfl_standalone.py  the offline single-file app
    nfl/engine/ncaa_data.py       college football (FBS), data only so far: sportsdataverse schedules
                                  and lines, ESPN's live board and teams (sync + loaders)
    nfl/data/                     snapshots (games, team games, QB games, injuries), odds cache, pick log
    nfl/data/ncaa/                college snapshot: schedule and lines per season, teams, the live week
    nfl/app/                      the phone app (PWA)
    .github/workflows/nfl.yml     runner: predict / sync / backtest / ncaa-sync, on a schedule in season

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
