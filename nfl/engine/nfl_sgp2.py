"""Same-game tickets, v2: only confident games, the pick's side, and the
players the touchdown model rates most likely to score.

PRE-REGISTERED 2026-11-24, before any v2 ticket was built or any v2 result
computed. v1 (nfl_sgp.py, sgp-v1) stays as it is and is measured beside v2.

WHY A V2. The user builds his own two-leg tickets - a confident game's
winner plus a player the system rates very likely to score - and they land
far more often than v1's. v1 built tickets in every game, toss-ups
included, took result legs on either team, and filled slots with whatever
TD leg ranked highest even at a 22% chance, or a coin-flip total. v2 keeps
v1's ticket shapes, legs, grading, meters and the user's cap (a leg in at
most USE_CAP of a game's tickets), and narrows what may go in them:

  GAMES   only a confident game gets tickets: the pick's blend chance is
          CONFIDENT or more. Every other game shows no tickets, and why.
  RESULT  result legs only on the pick's side: the pick's WIN and the
          pick's SPREAD. Never the other team's.
  TD      only players the model rates very likely: a TD chance of
          TD_FLOOR or more (the rookie leg keeps v1's ROOKIE_FLOOR - a
          promising rookie is rarely rated that high - and must be on the
          pick's team, since its ticket pairs him with "team winning").
  SMALL   the pick's best TD scorer: the best TD leg on the pick's team if
          one clears TD_FLOOR, else the game's best that does. A team that
          wins scores more: P(TD | his team won) was 0.527 against 0.360 when
          it lost (2016-2025, the research before v1), so the pick's own
          scorer rides with the pick's WIN.
  TOTAL   unchanged from v1 (the market's no-vig side), and only where v1
          puts one: MEDIUM's third slot competes with the TD legs, LARGE's
          fourth slot is a total.

  Everything else is v1's: the build order SMALL, ROOKIE, MEDIUM, LARGE;
  each slot takes the highest-chance leg still available; one result leg,
  one total and distinct TD players per ticket; the identical-to-SMALL
  rule; a slot with nothing available leaves its ticket unbuilt, with the
  reason; voids (tie, push, did not play) and payout on the remaining legs;
  the product chance beside the measured hit rate; est. payouts from the
  consensus single prices; Stake's quoted price typed for the exact payout.

THE TWO NUMBERS, CHOSEN ON DEV ONLY. CONFIDENT and TD_FLOOR are picked from
the grids below on 2014-2019 alone, by one objective fixed here: the
highest SMALL hit rate (won / (won + lost)) among the pairs that build a
SMALL ticket in at least COVERAGE_MIN of DEV's games - so most weeks still
have a handful of tickets - ties to the larger coverage, then the lower
CONFIDENT, then the lower TD_FLOOR. Hit rate always rises as the bars rise,
so the coverage floor is what stops the choice at a usable pair. The chosen
pair is then measured, untouched, on 2020-2025 and on the current season,
and every other pair of the grid is reported beside it (report-only: it
cannot move the choice). Nothing is chosen on 2020 or later.

  CONFIDENT_GRID  0.60, 0.65, 0.70, 0.75 - the app's SOLID band starts at
                  0.65 and STRONG at 0.75.
  TD_FLOOR_GRID   0.35, 0.40, 0.45, 0.50 - 0.50 is "more likely than not".
  COVERAGE_MIN    0.25 - a SMALL ticket in a quarter of the games, about four
                  a week of a sixteen-game slate.

REPORTED BESIDE V2 (report-only): v1 on the same games; SMALL with the
game's best TD instead of the pick's team's first; and the whole grid.

Nothing here claims an edge. Narrowing to likely legs raises the hit rate
and lowers the payout; the market prices both. These are the legs the
engine rates most likely to land, not bets it rates profitable.
"""

VERSION = "sgp-v2"
CONFIDENT_GRID = (0.60, 0.65, 0.70, 0.75)
TD_FLOOR_GRID = (0.35, 0.40, 0.45, 0.50)
COVERAGE_MIN = 0.25
DEV = (2014, 2019)             # the only seasons the two numbers are chosen on
CONFIRM = (2020, 2025)
# CONFIDENT and TD_FLOOR are set from the DEV choice, once, by the backtest
# (experiments/nfl_sgp2_backtest.py --choose) and written here with the
# date they were chosen; until then v2 builds no live tickets.
CONFIDENT = None
TD_FLOOR = None
