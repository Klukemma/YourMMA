# YourSoccer

Predictions for Europe's top five leagues, built the way the MMA and NFL apps are: every
number point-in-time, measured walk-forward against the betting market before it ships,
picks logged before kickoff and graded afterwards, and a phone app that shows the numbers
with their measured record beside them.

The phone app is at **https://klukemma.github.io/YourMMA/soccer/app/**. Open it on a phone and
use Add to Home Screen. `soccer/app/yoursoccer.html` is a single offline copy of the same app.

## Data

Two kinds of source, with different jobs. Both are snapshotted into `soccer/data/` on GitHub's
machines (the development sandbox cannot reach either).

**football-data.co.uk** (`soccer_data.py`) is the source of record and the only source of prices:

- the Premier League, La Liga, Serie A, the Bundesliga and Ligue 1, and their second tiers
  (rated, never predicted - that is where promoted clubs come from), every season since 2005-06;
- each match's result, shots, shots on target, corners and cards - the model reads only these;
- two prices: the **pre-match** average, collected Friday afternoon for weekend games and
  Tuesday afternoon for midweek ones - the same moment the fixtures file the app predicts from
  is published - and the **closing** price;
- kickoff times, on the UK clock, converted to UTC.

It posts the weekend's fixtures only on Friday afternoon, the midweek's on Tuesday afternoon,
and results days after a round. So, for **schedule and scores only**, two free, unofficial feeds
(`soccer_feed.py`, snapshot `soccer/data/feed.json`):

- **ESPN's public scoreboard** (one request per league per day, four days back to ten ahead):
  kickoff, whether ESPN calls the time valid, status (scheduled, live, final, postponed,
  cancelled, abandoned), score. Its betting lines are not read.
- **fixturedownload.com** (one request per league for the season): kickoff and score. A round
  with no real time yet carries a placeholder clock, read as "no time".

Clubs are mapped by explicit tables (ESPN by team id), never fuzzy-matched. Neither feed has an
agreement with this project or any promise to stay as it is: either can change or vanish, and
the app would fall back to football-data.co.uk alone. What the probe of 2026-10-26 measured
against football-data.co.uk (every id, kickoff and score of the matches both had) is in
`soccer_feed.py`'s docstring.

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

## Goals, scores and how sure the pick is

Opened, every match also shows both teams to score, over 1.5 and over 3.5 goals, the three likeliest
exact scores, and the pick's **band** - all measured walk-forward the same way
(`experiments/score_markets.py`, choosing on 2006-2015 and confirming once on 2016-2025; the
production code is `soccer_scores.py`, which the experiment imports, so the app runs the code that was
measured).

**One score matrix per match**, everything read off it. Three sources were compared against each league's
own rates from earlier seasons (B): the model's own Dixon-Coles matrix (M), the matrix that best
reproduces the pick's home/draw/away chances and the over/under 2.5 price (K, market-implied), and the
market's total split the model's way (KM). K had the lowest 2006-2015 exact-score log loss, so it was
chosen, and a number ships only if it beats B in both periods. All four did. Log loss, the same
18,260 / 17,937 matches for every source:

| | M 2006-15 | **K** 2006-15 | B 2006-15 | M 2016-25 | **K** 2016-25 | B 2016-25 |
|---|---|---|---|---|---|---|
| Both teams to score | 0.68979 | **0.68972** | 0.69312 | 0.68456 | **0.68319** | 0.69047 |
| Over 1.5 | 0.56468 | **0.56332** | 0.57117 | 0.52792 | **0.52554** | 0.53889 |
| Over 3.5 | 0.57977 | **0.57823** | 0.59079 | 0.59869 | **0.59508** | 0.61558 |
| Exact score (37 ways) | 2.83389 | **2.82672** | 2.94606 | 2.87252 | **2.85793** | 3.00865 |

K's likeliest score happened in 12.9% / 13.1% of matches (B's: 11.5% / 11.9%), one of its three
likeliest in 34.9% / 34.2% (B: 31.1% / 29.6%). Over 2.5 stays the over/under market's: K reproduces it
(2016-2025 log loss 0.67067 against 0.67066) and adds nothing.

**The rule as shipped** (`soccer_scores.shown_matrices`):

- **K** where the home/draw/away **and** the over/under 2.5 are priced, and the fit misses none of its four
  targets (home, draw, away, over 2.5) by more than **0.04**;
- **M**, the model's own matrix, where only the home/draw/away is priced (K's path from the 1X2 alone was
  measured on 2 matches a period - effectively unmeasured), where nothing is priced (the model-alone
  picks between football-data.co.uk's postings), or where K's fit misses a target by more than 0.04.
  The card says which, and why;
- nothing (shown as missing) where the model has no rates for the match.

The 0.04 was set on 2006-2015 from the fit alone, no result read: the largest per-target miss of K has a
body (median 0.006, 99th percentile 0.0387) and a long tail (0.056 at the 99.5th, 0.17 at the 99.9th) of
matches whose prices contradict any Poisson matrix - mostly Serie A run-ins of 2006-2010, the worst a 71%
draw beside a 43% over 2.5, which K "reproduces" with over 2.5 at 0.9% and 0-0 at about 70%. The rule is
that 99th percentile rounded up to the next 0.01. It catches 168 of 18,258 fits on 2006-2015, 8 of 17,935
on 2016-2025 and none of 2026-27's so far. **It is a display rule, and it costs a little:** on those 168
DEV matches the model's matrix scored the exact score worse than K's (log loss 2.797 against 2.753). The
rule as shipped, measured by every run on the same matches (the Method tab prints this table):

| | 2006-2015 | 2016-2025 |
|---|---|---|
| Both teams to score | 0.68964 (K 0.68972) | 0.68318 (K 0.68319) |
| Over 1.5 | 0.56341 (K 0.56332) | 0.52553 (K 0.52554) |
| Over 3.5 | 0.57830 (K 0.57823) | 0.59509 (K 0.59508) |
| Exact score | 2.82712 (K 2.82672) | 2.85794 (K 2.85793) |
| Likeliest score happened | 12.9% | 13.1% |
| From the market / the model | 18,090 / 170 (1 no price, 1 no over/under, 168 fit) | 17,927 / 10 (1, 1, 8) |

Every one still beats the league rates in both periods. Run by the experiment's rule (K from the 1X2 alone,
no fit limit), the in-run table reproduces the experiment's published numbers exactly (checked on the
whole history; a test holds the two to the same numbers on a made-up league).

**The card** shows the pick's own home/draw/away as before - the matrix's are beside them ("Scores'
1X2"), never in their place (K's draw is on average 0.005 below the pick's). For a pick by the model
alone, the matrix is the model's and its 1X2 is the pick itself.

**The band** is how sure the pick is, by its top chance, at the 2006-2015 quartiles (0.423, 0.497,
0.616) rounded: **Strong** 60% or more, **Solid** 50-60%, **Lean** 40-50%, **Toss-up** under 40%. Each is
shown with its measured hit rate - the pick's (the market's where priced) or, for a pick by the model
alone, the model's own picks at the same edges (measured on both periods, no new choice; they are close,
and the pick's rate would not be the one measured for that pick):

| Band | Picks won 2006-15 | Picks won 2016-25 | Model's own 2006-15 | Model's own 2016-25 |
|---|---|---|---|---|
| Strong | 69.8% of 5,087 | 70.9% of 5,659 | 69.9% of 4,211 | 71.4% of 4,710 |
| Solid | 53.0% of 3,880 | 55.0% of 3,957 | 55.2% of 4,279 | 55.3% of 4,379 |
| Lean | 44.8% of 6,259 | 44.9% of 5,445 | 44.8% of 6,535 | 45.6% of 5,958 |
| Toss-up | 39.2% of 3,034 | 36.8% of 2,876 | 39.3% of 3,235 | 36.8% of 2,890 |

A band is a measured hit rate, never "bet this".

**The live record** of these numbers starts with this change: every pick is logged with `p_btts`,
`p_o15`, `p_o35`, the three likeliest scores (with the matrix's 1X2 and why it is the model's where it
is), `score_source`, `band` and `band_basis`; both teams to score is graded beside over/under 2.5 (the
side its chance favoured) and counted on the Record tab. Picks logged before it carry none of these and
say so.

## Point in time

Every model predicts a match only from matches on earlier days, checked by
`soccer_backtest.check_point_in_time` (which scrambles every result from a date on and requires
that date's predictions not to move) on dates in both periods, and re-checked by an independent
judge with a stricter version (prices and match statistics scrambled too). The recalibration
for a season is fitted on earlier seasons only. A pick is logged before kickoff and frozen when
the match starts - or at 00:00 UK time on match day where the kickoff time is not confirmed (see
the rules below) - and the live record grades only those (`soccer_log.py`); a pick logged at or
after a match's real kickoff is void, never graded.

## Between football-data.co.uk's postings: the rules

`soccer_schedule.py` joins the feeds with football-data.co.uk; `soccer_log.py` keeps the picks.

- **One match, one row, one pick.** A pairing (league, season, home, away) is played once a
  season. Where football-data.co.uk has posted the fixture, its row is the match - its id and its
  prices; the feeds add only their kickoff. A match only the feeds list is shown and picked with
  **no price**: the model's own chances, labelled "No price yet". When football-data.co.uk posts
  it with prices before kickoff, the market's pick replaces the model's under the same id (or the
  pick moves to football-data.co.uk's id if the feeds dated it differently; nothing is voided).
- **Kickoff.** The earlier of football-data.co.uk's and the feeds' times, so a pick is frozen no
  later than any source's kickoff.
- **Freeze.** At kickoff where the time is confirmed (football-data.co.uk's own, or ESPN and
  fixturedownload agreeing). A time the feeds do not confirm freezes the pick at **00:00 UK time
  on match day**. A pick whose freeze time has passed never changes, even if its time is confirmed
  later; a match first listed after it has none.
- **Never graded if logged after kickoff.** A result that carries its kickoff is checked against
  the pick's last log; a pick written at or after the real kickoff (a match brought forward that
  no source had moved yet) is void.
- **Postponed, cancelled, abandoned** (as ESPN says, on football-data.co.uk's date for it): not
  picked, shown as such. A pick logged before is **held** on the card - kept in the log, not
  graded - until a pick on the new date replaces it, or it is void three weeks after kickoff with
  no result. A football-data.co.uk fixture whose time has passed with no result while the feeds
  list the match later was moved: the feeds' date stands until football-data.co.uk catches up.
- **No date.** A match the feeds list with no time from either source (a round not scheduled
  yet): "date and time to be confirmed", never picked.
- **Provisional results.** ESPN's final score, with fixturedownload.com's agreeing or not posted
  yet (a contradiction is no result), for a match football-data.co.uk has not posted. It grades a
  pick **provisionally**, and is listed in the results, the form lines and the tables, flagged.
  When football-data.co.uk posts the result, that is the result: the same score confirms the
  grade, a different one corrects it, and the correction is kept and shown. A provisional grade
  the feeds later contradict each other on is withdrawn.
- **Never in the model.** Provisional results never reach the model's input: its ratings use
  football-data.co.uk's results only, which carry the shots it reads.
- **A feed outage never fails a run or wipes data.** Each request's failure is caught and keeps
  what is on file for it (a league's requests stop after three failures in a row, and none starts
  after ten minutes); a snapshot that cannot be read or joined means football-data.co.uk alone,
  said so in the log. A final score on file is never dropped by a re-fetch.

## Caveats

- **Feed results are provisional** until football-data.co.uk posts its own, usually days later.
  The app marks them (a dashed edge in the last five, an asterisk in the table, "Provisional" on
  the card and in the live record) and a provisional grade can still change.
- **The model's ratings include results through the date shown** ("ratings include results
  through ...") - football-data.co.uk's latest - not the provisional ones after it.
- **Prices come only from football-data.co.uk**, so between rounds the next matches are picked by
  the model alone until Friday (weekend) or Tuesday (midweek) afternoon. Those picks count in the
  live record like any other, and the model alone is measured to pick worse than the market
  (2016-2025: 53.3% against 54.0%, above).
- **The feeds are unofficial and free.** Their times and scores were checked against
  football-data.co.uk on the probe, not guaranteed; fixturedownload.com's 2026-27 Ligue 1 lists one
  pairing twice with home and away swapped (warned, the played one used).
- **Both teams to score runs low.** Across 2016-2025 its chance averaged 51.1% where both teams scored in
  53.7% of matches, low in every decile (2006-2015: 48.8% against 51.0%). It still beat the league rates
  in both periods. A recalibration was not shipped: it would have to be fitted on earlier seasons and
  measured on a window not yet read.
- **Exact scores are long shots.** The likeliest score happens about one match in eight (13.1% across
  2016-2025); one of the three likeliest about one in three.
- **With no price, the goals and scores numbers are the model's own** (and the band's rate the model's own
  picks'): measured too, a little less accurate than the market-implied ones (2016-2025 both teams to
  score log loss 0.6846 against 0.6832; exact score 2.8725 against 2.8579). The card says so.
- **The score matrix cannot always reproduce the prices.** Its largest per-target miss is usually under
  0.01 (median 0.006 on 2006-2015, smaller since); the card prints it. Beyond 0.04 the model's matrix is
  shown instead (above), which scored worse on those matches - the rule keeps a card from printing numbers
  that contradict the prices beside them.

## The runner

`.github/workflows/soccer.yml` runs `soccer_predict.py` through the season (August to May):

| When (UTC) | Why |
|---|---|
| Friday 13:47 and 16:47 | the weekend's fixtures and prices from football-data.co.uk |
| Saturday and Sunday 07:17 | refresh before the day's games; grade the day before |
| Saturday and Sunday 21:47 | the day's results from the feeds, the evening they happen (provisional); the next round |
| Monday 09:17 | grade the weekend |
| Tuesday 13:47 and 16:47 | the midweek fixtures and prices |
| Wednesday 07:17 | Tuesday's results; the weekend's round early (the model alone until Friday) |
| Thursday 09:17 | grade the midweek round |
| every morning 08:17, Dec 20 - Jan 3 | the Christmas and New Year fixtures |

GitHub starts scheduled runs late under load; a match that kicks off before any run has logged
it gets no pick. One run: sync this season from football-data.co.uk and the feeds, run the model
over every match since 2005-06 (about 15 seconds), recalibrate the market, fit every match's score
matrix and measure the goals and scores numbers on 2006-2025 (about 6 seconds), log the picks that
have not frozen, grade the finished ones, and write `soccer/app/data/`, the standalone page and
`soccer/data/` (football-data.co.uk's snapshot, the feeds' snapshot and the prediction log),
then commit.

    python soccer/engine/soccer_predict.py --no-sync   # locally, on the snapshot as it is
    python -m pytest soccer/engine/tests -q
