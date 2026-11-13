# YourSoccer

Predictions for every league football-data.co.uk carries that has a current season - 37 of its
38 today, from the Premier League to MLS and the J1 League (Russia's file stopped in August) -
built the way the MMA and NFL apps are: every number point-in-time, measured walk-forward against
the betting market before it ships, league by league, picks logged before kickoff and graded
afterwards, and a phone app that shows the numbers with their measured record beside them. A
number a league did not earn on its own seasons is withheld, and shown as missing.

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

## Every league: what the run does with each

`soccer_leagues.py` holds what `experiments/leagues.py` measured and decided for each of
football-data.co.uk's 38 leagues (its docstring has the rules, decided before any number was
seen); the run reads those decisions and never re-decides them:

- **Which leagues.** The model is run over all 38 (`soccer_data.load_matches(leagues="all")`):
  a country's rating pool and a group's regression learn from every league of them, and the
  pick's weights are pooled over every league priced the same way, so a frame missing one - even
  Russia's, which the app does not show - would give numbers nobody measured (`soccer_model`
  refuses it). The app carries the leagues with a current season in the data now
  (`soccer_leagues.in_app(matches, fixtures, now)`: a result in the last 45 days or a fixture
  posted) - 37 today. The league picker lists them in `league_order`, every one, even with no
  match on screen; Russia is under `not_in_app`, with why.
- **The pick** of each league is its own market's price (the pre-match average; an extra league's
  history has only the close, which it was measured on), recalibrated on earlier matches - or the
  raw price with the bookmakers' cut removed in the nine leagues where recalibrating scored worse
  on their development period (D1 E1 E3 SC0 SC1 SC3 AUT FIN POL). With no price posted yet, the
  model's own pick only where the model beat the league's base rates in both periods (every league
  in the app, today); elsewhere "No pick until the price is posted". Every match entry and every
  logged pick carries its `basis` (`recalibrated` / `raw` / `model` / `unpriced`) and the price it
  was read from (`price_kind`: `pre` / `close`).
- **The numbers beside it** (both teams to score, over 1.5 / 2.5 / 3.5, the likeliest scores) are
  shown for a league only from a source - the market-implied matrix or the model's own - that beat
  the league's own rates in both of its periods (`soccer_leagues.withheld`). A withheld number is
  null and listed in `scores.withheld`; over 2.5 beside the card follows the same rule. The extra
  leagues have no over/under price, so their numbers are the model's own, where it earned them.
- **Bands** are read against the league's group's measured hit rates (`matches.json` `bands`, by
  `leagues[code].group` and the card's `band_basis`), never the top five's for another league.
- **Each league on its own dates**: its season and label on its own calendar ("2026 season" in
  ARG BRA CHN FIN IRL NOR SWE USA, "2026-27 season" elsewhere), its own latest football-data.co.uk
  result (`model_through` - the last its ratings include), its own latest round in the results, its
  own table (`tables.json`, with what the table is not where its format splits or has play-offs:
  `table_note`), its own periods (2013-2018 / 2019-2025 for the extra leagues) and its own record
  (`performance.json` `backtest.by_league`, which reproduces `soccer_leagues.SHIPPED`'s measured
  numbers; the live `track_record.by_league`, `by_group` and `by_basis`). The pooled backtest
  blocks (`result`, `over_2_5`, `value`, `calibration`) are the top five's, and say so (`scope`).
- **Caveats are never deleted.** Each league carries every caveat `soccer_leagues` holds for it,
  and beside them - never instead - what the live feeds cover for it (`coverage`,
  `coverage_caveat`): seven leagues (SC2 SC3 FIN IRL POL ROU SWZ) have no feed, so their matches
  appear only when football-data.co.uk posts them, with no provisional results.
- **Where a pairing repeats in a season** (a split, a third round robin, play-offs: the 15 leagues
  in `soccer_feed.REPEATS`), a pairing is not a match: the schedule, the log and the cards look a
  match up by its pairing and its day (`soccer_schedule.key`), and a pick is never graded against,
  moved onto or hidden behind another meeting of the same clubs.

## What the app shows, and why

What follows is how the model and the pick were chosen, on the top five (the numbers in this
section are theirs); every other league was then measured on its own seasons by the same
procedure, and ships only what it earned there (above).

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
  q^beta * exp(c), each league-season's weights fitted on the matches dated before it of every
  league priced the same way - the European divisions' pre-match prices together, the extra
  leagues' closing prices together (`soccer_model.shipped_pick`, `soccer_leagues.PICK_WEIGHTS`) -
  or the raw price where recalibrating lost (the Bundesliga among them). The market's favourites
  win slightly more often than its prices say and the draw is priced slightly short; for the
  Premier League's 2026-27, beta = 1.131 and the draw gets +0.043.
- **The model's own view** - the shots-based ratings, the best of the three - is shown beside
  every pick, labelled "Model alone", and drives the club ratings on the Tables tab. It is never
  blended into the pick. Where no price was posted it is all there is, and the app says so.
- **Goals:** the over/under 2.5 market with the cut removed. The model's own chance is shown
  beside it (log loss 0.6752 against the market's 0.6707 on 2016-2025).
- **No match is labelled a bet.** Betting the model wherever it disagreed with the price, at the
  best price any book posted, made money on 2006-2015 and lost 5-7% on 2016-2025.

The 2016-2025 pick log loss by league, as shipped (`soccer_leagues.SHIPPED`): Premier League
0.9522 (market 0.9520), La Liga 0.9659 (0.9673), Serie A 0.9466 (0.9491), Bundesliga 0.9788 (the
raw market), Ligue 1 0.9822 (0.9828). Every league's are on the Record tab.

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

- **K** where the home/draw/away **and** the over/under 2.5 are priced. The card prints how far K misses the
  prices it was fitted to (its largest miss over home, draw, away and over 2.5, and on which), and **flags**
  it where that is more than **0.04**: "no score chances fit these prices closely - read them with care";
- **M**, the model's own matrix, where only the home/draw/away is priced (K's path from the 1X2 alone was
  measured on 1 match a period - effectively unmeasured) or where nothing is priced (the model-alone picks
  between football-data.co.uk's postings). The card says which, and why;
- nothing (shown as missing) where the model has no rates for the match.

The 0.04 was set on 2006-2015 from the distribution of K's largest miss: a body (median 0.006, 99th
percentile 0.0387) and a long tail (0.056 at the 99.5th, 0.17 at the 99.9th) of matches whose prices
contradict any Poisson matrix - mostly Serie A run-ins of 2006-2010, the worst a 71% draw beside a 43% over
2.5, which K "reproduces" with over 2.5 at 0.9% and 0-0 at about 70%. The flag is that 99th percentile
rounded up to the next 0.01; it marks 168 of 18,258 fits on 2006-2015, 8 of 17,935 on 2016-2025 and none of
2026-27's so far. **It is a flag, not a swap.** An earlier version showed M on those matches "so a card
never prints numbers that contradict the prices beside it"; measured, that was backwards. On those matches
M misses the card's prices by *more* than K (median largest miss 0.10 against 0.06; in 163 of the 176
over both periods) and scores worse (2006-2015 exact-score log loss 2.797 against 2.753; 2016-2025 2.858
against 2.857). DEV outcomes for each candidate threshold, 0.03 to 0.10, were looked at when it was set,
and showed the swap costing at every one. So the card shows K and says how far it misses; in 4 of those
168 DEV matches K is near-degenerate (its likeliest score above 30%).

The rule as shipped, measured by every run on the same matches (the Method tab prints this table):

| | 2006-2015 | 2016-2025 |
|---|---|---|
| Both teams to score | 0.68973 (K 0.68972) | 0.68320 (K 0.68319) |
| Over 1.5 | 0.56332 (K 0.56332) | 0.52555 (K 0.52554) |
| Over 3.5 | 0.57823 (K 0.57823) | 0.59508 (K 0.59508) |
| Exact score | 2.82672 (K 2.82672) | 2.85794 (K 2.85793) |
| Likeliest score happened | 12.9% | 13.1% |
| From the market / the model | 18,258 / 2 (1 no price, 1 no over/under); 168 flagged | 17,935 / 2 (1, 1); 8 flagged |

Every one still beats the league rates in both periods. It differs from K only on the 2 matches a period
with no over/under price. Run by the experiment's rule (K from the 1X2 alone), the in-run table
reproduces the experiment's published numbers exactly (checked on the whole history; a test holds the two
to the same numbers on a made-up league).

**The card** shows the pick's own home/draw/away as before - the matrix's are beside them (the "Results"
row), never in their place (K's draw is on average 0.005 below the pick's). That row is a by-product,
measured no better than the pick (2016-2025 log loss 0.96488 against the pick's 0.96447; the Method tab
prints it). For a pick by the model alone, the matrix is the model's and its 1X2 is the pick itself.

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
`p_o15`, `p_o35`, the three likeliest scores (with the matrix's 1X2, its fit miss and flag, and why it is
the model's where it is), `score_source`, `band` and `band_basis`; both teams to score is graded beside
over/under 2.5 (the side its chance favoured) and counted on the Record tab. That call, measured: right in
55.6% of 2016-2025 matches against 53.7% for always "yes" (2006-2015: 53.7% against 51.0%). Picks logged
before it carry none of these and say so.

## Point in time

Every model predicts a match only from matches on earlier days, checked by
`soccer_backtest.check_point_in_time` (which scrambles every result from a date on and requires
that date's predictions not to move) on dates in both periods, and re-checked by an independent
judge with a stricter version (prices and match statistics scrambled too). The recalibration
for a season is fitted on earlier seasons only. A pick is logged before kickoff and frozen when
the match starts - or at 00:00 UK time on match day where the kickoff time is not confirmed, or
earlier still where any source lists an earlier kickoff (see the rules below) - and the live record
grades only those (`soccer_log.py`); a pick logged at or after a match's real kickoff is void, never
graded. A real run reads the clock again after its sync, so a pick is never logged at a moment
before the prices and kickoffs it was built from; each feed row carries the moment its own request
was answered.

## Between football-data.co.uk's postings: the rules

`soccer_schedule.py` joins the feeds with football-data.co.uk; `soccer_log.py` keeps the picks.

- **One match, one row, one pick.** A pairing (league, season, home, away) is played once a
  season. Where football-data.co.uk has posted the fixture, its row is the match - its id and its
  prices; the feeds add only their kickoffs. A match only the feeds list is shown and picked with
  **no price**: the model's own chances, labelled "No price yet". When football-data.co.uk posts
  it with prices before kickoff, the market's pick replaces the model's under the same id (or the
  pick moves to football-data.co.uk's id if the feeds dated it differently; nothing is voided).
  **A priced pick is never replaced by an unpriced one**: if football-data.co.uk's fixtures file
  (replaced whole on every sync) stops listing a match it had priced, the pick stays the market's
  and only its time follows the feeds.
- **Kickoff, as shown** - the match's day, its id, when the card says "Kicked off":
  football-data.co.uk's own time where it has the fixture (or the feeds' confirmed time where that is
  earlier); for a match only the feeds list, **ESPN's** valid time, else fixturedownload.com's. Not
  simply the earliest: on 2026-10-27 ESPN had moved three Ligue 1 matches by days that
  fixturedownload.com still listed on the Saturday (below), and the earliest would have dated them
  wrong and shown them "Kicked off" for up to two days. Where the sources differ, the card lists each
  one's time ("Date to be confirmed" where they give different days).
- **Close (freeze).** At the shown kickoff where that time is confirmed (football-data.co.uk's own,
  or ESPN and fixturedownload agreeing); at **00:00 UK time on match day** where it is not - and
  **earlier**, at any source's time before that, so no pick stays open past any source's kickoff.
  (A source's time that has passed while ESPN still lists the match as scheduled later is not the
  kickoff, and is left out.) After it closes and before the shown kickoff, the card is "Locked" with
  the pick as logged. A pick whose freeze time has passed keeps its pick, its chances and its freeze
  time for good, even if its time is confirmed later; a match first listed after it has none.
- **A match whose id changes.** A pick logged under one id (a feed's date) and listed later under
  another (football-data.co.uk's, or ESPN's new day) is the same match: it moves to the new id. One
  that had not frozen is logged afresh; one that had moves as it is - nothing new is logged after a
  pick froze - with its time following the match. A pick is voided as "rescheduled" only when its
  match is known to be off - every time on file for it (its kickoff and each source's) passed with
  no result, or the feeds list it postponed, cancelled or abandoned - and a pick on the new date
  replaces it.
- **Never graded if logged after kickoff.** A result that carries its kickoff is checked against
  the pick's last log; a pick written at or after the real kickoff (a match brought forward that
  no source had moved yet) is void. A provisional result is checked against ESPN's kickoff - the
  final's own time - never fixturedownload.com's alone, and a void on the feeds' time is checked
  again when football-data.co.uk's result lands: logged before its kickoff, the pick is restored and
  graded.
- **Postponed, cancelled, abandoned** (as ESPN says, on football-data.co.uk's date for it): not
  picked, shown as such. A pick logged before is **held** on the card - kept in the log, not
  graded - until a pick on the new date replaces it, or it is void three weeks after kickoff with
  no result. A football-data.co.uk fixture whose time has passed with no result while **ESPN** lists
  the match as scheduled more than three hours later was moved: ESPN's date stands, with no price,
  until football-data.co.uk catches up. fixturedownload.com alone never moves a match: its
  "scheduled" only means no score posted yet.
- **No date.** A match the feeds list with no time from either source (a round not scheduled
  yet): "date and time to be confirmed", never picked.
- **Provisional results.** ESPN's final score, with fixturedownload.com's agreeing or not posted
  yet, for a match football-data.co.uk has not posted - dated by ESPN's kickoff. It grades a pick
  **provisionally**, and is listed in the results, the form lines and the tables, flagged. When
  football-data.co.uk posts the result, that is the result: the same score confirms the grade, a
  different one corrects it, and the correction is kept and shown. A provisional grade is
  withdrawn - the pick pending again, written down - when the feeds later contradict each other on
  the score, or ESPN no longer calls the match final (abandoned, postponed). Where the feeds'
  scores contradict, the card says the result is withheld and gives both.
- **Never in the model.** Provisional results never reach the model's input: its ratings use
  football-data.co.uk's results only, which carry the shots it reads.
- **A feed outage never fails a run or wipes data.** Each request's failure is caught and keeps
  what is on file for it (a league's requests stop after three failures in a row, and none starts
  after ten minutes). An ESPN day that answers with no events while matches still to play are on
  file for it is taken for a glitch and kept as on file, until the next run gets the same empty
  answer. A snapshot that cannot be read is rebuilt from the next fetch that answers (every
  committed version is in git), and until then the run is football-data.co.uk alone, said so in the
  log. A final score on file is never dropped by a re-fetch.

## Caveats

- **Feed results are provisional** until football-data.co.uk posts its own, usually days later.
  The app marks them (a dashed edge in the last five, an asterisk in the table, "Provisional" on
  the card and in the live record) and a provisional grade can still change.
- **The model's ratings include results through the date shown** ("ratings include results
  through ...") - football-data.co.uk's latest in that league (each league's own: its file can lag
  another's by weeks) - not the provisional ones after it.
- **An extra league's live pick uses the pre-match price; its record was measured on the close**,
  the only price its history has. The close scored 0.0035 better in log loss on the European
  divisions' 2019-2025, so its live picks should do about that much worse than measured; every
  such league says so in its own caveats.
- **A league's table is the regular season's.** Play-off rounds, MLS Cup, the liguilla and the rounds
  after a split are left out (`soccer_form`), so where a league splits (and in Belgium, Austria and
  Romania halves the points) or has play-offs, the table is not the official standings - its
  `table_note` says how.
- **Prices come only from football-data.co.uk**, so between rounds the next matches are picked by
  the model alone until Friday (weekend) or Tuesday (midweek) afternoon. Those picks count in the
  live record like any other, and the model alone is measured to pick worse than the market
  (2016-2025: 53.3% against 54.0%, above).
- **The feeds are unofficial and free.** Their times and scores were checked against
  football-data.co.uk on the probe, not guaranteed; fixturedownload.com's 2026-27 Ligue 1 lists one
  pairing twice with home and away swapped (warned, the played one used).
- **Two feeds agreeing on a time is not a guarantee a week out.** On the probe of 2026-10-26 ESPN
  (its time marked valid) and fixturedownload.com agreed on all 130 matches ESPN listed. A day later
  ESPN had moved three of the 58 still to play - all Ligue 1, round 10: Nice v Rennes 22 hours
  *earlier*, Lyon v Angers a day later, Lorient v Brest two days later - and fixturedownload.com none
  (its later Ligue 1 rounds read like a template). What protects the record is that every run reads
  the feeds again before it logs, ESPN's "under way" stops a pick, and a pick logged at or after the
  real kickoff is void. A "confirmed" feed time is not frozen at midnight on purpose: football-data.co.uk
  prices Tuesday's and Friday's matches that same afternoon (3 of the 12 priced picks so far were
  priced on match day), and a midnight close would keep those the model's alone.
- **The club tables need the promoted clubs every summer.** Each feed's names map to
  football-data.co.uk's by explicit tables (`soccer_feed.ESPN_CLUBS`, `FD_CLUBS`) that hold 2026-27's
  clubs. A club missing from them is skipped, its matches with it, and listed in the feeds'
  snapshot (`last_run.unmapped`); a snapshot check (reported, never a gate) fails when
  football-data.co.uk has a club this season that either table does not map. Add the promoted clubs
  each August.
- **Both teams to score runs low.** Across 2016-2025 its chance averaged 51.1% where both teams scored in
  53.7% of matches, low in every decile (2006-2015: 48.8% against 51.0%). It still beat the league rates
  in both periods. A recalibration was not shipped: it would have to be fitted on earlier seasons and
  measured on a window not yet read.
- **Exact scores are long shots.** The likeliest score happens about one match in eight (13.1% across
  2016-2025); one of the three likeliest about one in three.
- **With no price, the goals and scores numbers are the model's own** (and the band's rate the model's own
  picks'): measured too, a little less accurate than the market-implied ones (2016-2025 both teams to
  score log loss 0.6846 against 0.6832; exact score 2.8725 against 2.8579). The card says so, and gives the
  model's own caveats: its both teams to score ran 1.7 points low across 2016-2025 (52.0% against 53.7%;
  the card says "about 2"), its likeliest score happened 13.0% of the time.
- **The score matrix cannot always reproduce the prices.** Its largest per-target miss is usually under
  0.01 (median 0.006 on 2006-2015, smaller since); the card prints it, and flags it beyond 0.04 - prices no
  Poisson matrix can match (above), where the market's matrix is still the better of the two measured.

## The runner

`.github/workflows/soccer.yml` runs `soccer_predict.py` every month - the calendar-year leagues
(ARG BRA CHN FIN IRL NOR SWE USA) play through June and July:

| When (UTC) | Why |
|---|---|
| Friday 13:47 and 16:47 | the weekend's fixtures and prices from football-data.co.uk |
| Saturday and Sunday 07:17 | refresh before the day's games; grade the day before |
| Saturday and Sunday 22:17 | the day's results from the feeds, the evening they happen (provisional) - after a 21:00 CET kickoff's full time, about 21:55 UTC in winter; the next round |
| Monday 09:17 | grade the weekend |
| Tuesday 13:47 and 16:47 | the midweek fixtures and prices |
| Wednesday 07:17 | Tuesday's results; the weekend's round early (the model alone until Friday) |
| Thursday 09:17 | grade the midweek round |
| every day 11:37 | the Americas' evenings: MLS, Liga MX, Brazil and Argentina kick off about 22:00-04:00 UTC |
| every day 17:37 | Asia's mornings: the J1 League and the Chinese Super League kick off about 04:00-12:00 UTC the next day |
| every morning 08:17, Dec 20 - Jan 3 | the Christmas and New Year fixtures |

GitHub starts scheduled runs late under load - 3 to 10 hours late on every run from Oct 26 to
Nov 5 - so the daily slots are set that far ahead of the kickoffs they are for; a match that kicks
off before any run has logged it gets no pick. One run: sync this season of all 38 leagues from
football-data.co.uk and the feeds for the 30 app leagues they cover (462 requests), run the model
over every match of every league since 2005-06, pick each league as decided, fit every match's
score matrix and measure the goals and scores numbers per group, log the picks that have not
frozen, grade the finished ones, and write `soccer/app/data/`, the standalone page and
`soccer/data/` (football-data.co.uk's snapshot, the feeds' snapshot and the prediction log),
then commit. Measured offline on the snapshot of 2026-11-10 (228,665 matches of 38 leagues, a replay of
Friday 2026-11-06 19:30 with 250 matches to pick): about 2.5 minutes - the model 78 s, the pick 24 s, the score
matrices 16 s, the Record tab 11 s, the scores measure 7 s, everything else under 20 s - on a quiet
4-core machine (up to 4 minutes under load), before the sync.

The prediction log stays small enough to commit on every run: one pick per line, compact, and a
pick that can never change again (settled on football-data.co.uk's result, or void for good) moves
four weeks after the newest pick's kickoff to `soccer/data/prediction_history/YYYY-MM.json`.
`soccer_log.load` reads them all back as one log.

    python soccer/engine/soccer_predict.py --no-sync   # locally, on the snapshot as it is
    python -m pytest soccer/engine/tests -q
