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

import nfl_sgp as sgp          # v1's legs, slots, grading, meters and constants

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
CHOSEN_ON = None

KINDS = sgp.KINDS              # v1's build order and ticket shapes, unchanged
LEGS = sgp.LEGS

# The objective in the words the payload shows beside the chosen pair.
OBJECTIVE = (f"the highest SMALL hit rate (won / (won + lost)) among the pairs that build a SMALL "
             f"ticket in at least {COVERAGE_MIN:.0%} of {DEV[0]}-{DEV[1]}'s games; ties to the larger "
             "coverage, then the lower CONFIDENT, then the lower TD_FLOOR")
NO_EDGE = ("Nothing here claims an edge. Narrowing to likely legs raises the hit rate and lowers the "
           "payout; the market prices both. These are the legs the engine rates most likely to land, "
           "not bets it rates profitable.")

# ---------------------------------------------------------------- why no ticket
#
# v1's reasons where v2 keeps v1's words (a TOTAL, the rookie's WIN); v2's
# own where the docstring narrows a slot, each with the bar it used - the
# grid's pairs are replayed side by side, so a reason names its own floor.

NOT_CHOSEN = ("v2's two numbers (CONFIDENT, TD_FLOOR) are not chosen yet "
              "(experiments/nfl_sgp2_backtest.py --choose)")
NO_PICK = "not a confident game: no chance on file for the pick"
NO_RESULT = "no result leg on the pick's side (its WIN or SPREAD) available"
NO_TOTAL = sgp.NO_TOTAL
NO_ROOKIE = (f"no rookie TD leg on the pick's team with a chance of {sgp.ROOKIE_FLOOR:.0%} or more "
             "available")
ROOKIE_IS_SMALL = (f"the only rookie TD leg on the pick's team at {sgp.ROOKIE_FLOOR:.0%} or more, with "
                   "the pick's WIN, is SMALL's ticket")
NO_ROOKIE_WIN = sgp.NO_ROOKIE_WIN


def not_confident(confident):
    """The one reason a game below the bar shows for all four tickets."""
    return f"not a confident game: the pick's chance is below {confident:.0%}"


def no_td(td_floor):
    return f"no TD leg with a chance of {td_floor:.0%} or more available"


def no_second_td(td_floor):
    return f"no TD leg with a chance of {td_floor:.0%} or more on a second player available"


def no_third(td_floor):
    return f"no TOTAL leg or TD leg with a chance of {td_floor:.0%} or more on another player available"


# ---------------------------------------------------------------- the game

def pick_of(home, away, p_home):
    """(the pick's team, the pick's blend chance) - nfl_predict's pick: the
    home side when P(home) is 0.5 or more, else the away side at 1 - it.
    (None, None) without a chance."""
    p = sgp._chance(p_home)
    if p is None:
        return None, None
    return (home, p) if p >= 0.5 else (away, 1.0 - p)


def eligible(p_pick, confident=CONFIDENT):
    """GAMES: a confident game - the pick's blend chance is `confident` or
    more. A missing chance, or a bar not chosen yet, is not confident."""
    p, bar = sgp._chance(p_pick), sgp._number(confident)
    return p is not None and bar is not None and p >= bar


def _floor_td(td_floor):
    """TD: a TD leg the model rates very likely - TD_FLOOR or more."""
    return lambda leg: sgp._is_td(leg) and leg["chance"] >= td_floor


def _rookie_td(pick_team):
    """The rookie leg: v1's ROOKIE_FLOOR, on the pick's team only."""
    return lambda leg: (sgp._is_td(leg) and bool(leg.get("rookie")) and leg.get("team") == pick_team
                        and leg["chance"] >= sgp.ROOKIE_FLOOR)


def pool(legs, pick_team, td_floor=TD_FLOOR):
    """The legs v2 may use, of a game's legs (nfl_sgp.game_legs), highest
    chance first: the pick's WIN and SPREAD (RESULT - never the other
    team's), every TOTAL (v1's), TD legs at `td_floor` or more, and rookie
    TD legs at nfl_sgp.ROOKIE_FLOOR or more on the pick's team - which only
    the ROOKIE ticket may take when they are under `td_floor`."""
    floor, rookie = _floor_td(td_floor), _rookie_td(pick_team)
    keep = []
    for leg in legs:
        if sgp._is_result(leg):
            ok = leg["team"] == pick_team
        elif sgp._is_total(leg):
            ok = True
        else:
            ok = floor(leg) or rookie(leg)
        if ok:
            keep.append(leg)
    return sgp._ranked(keep)


def _gate(pick_team, p_pick, confident, td_floor):
    """Why the game gets no tickets at all, or None when it gets them."""
    if confident is None or td_floor is None:
        return NOT_CHOSEN
    if pick_team is None or sgp._chance(p_pick) is None:
        return NO_PICK
    return None if eligible(p_pick, confident) else not_confident(confident)


# ---------------------------------------------------------------- the tickets

def _small_td(pool_, floor, pick_team):
    """SMALL's TD: the best on the pick's team clearing the floor, else the
    game's best that does."""
    return pool_.best(lambda leg: floor(leg) and leg["team"] == pick_team) or pool_.best(floor)


def _small(pool_, small, floor, pick_team, td_floor):
    return sgp._slots((_small_td(pool_, floor, pick_team), no_td(td_floor)),
                      (pool_.best(sgp._is_result), NO_RESULT))


def _rookie(pool_, small, floor, pick_team, td_floor):
    """v1's ROOKIE on the pick's team: the best rookie there at ROOKIE_FLOOR
    or more with the pick's WIN; the next one when that pair is SMALL's."""
    small_keys = None if small is None else {leg["key"] for leg in small}
    held = False
    for td in pool_.all(_rookie_td(pick_team)):
        win = pool_.find(f"WIN:{td['team']}")
        if win is not None and {td["key"], win["key"]} == small_keys:
            held = True
            continue
        if win is None or not pool_.available(win):
            return None, NO_ROOKIE_WIN
        return [td, win], None
    return None, ROOKIE_IS_SMALL if held else NO_ROOKIE


def _medium(pool_, small, floor, pick_team, td_floor):
    td = pool_.best(floor)
    return sgp._slots((td, no_td(td_floor)), (pool_.best(sgp._is_result), NO_RESULT),
                      (lambda: pool_.best(lambda leg: sgp._is_total(leg) or (
                          floor(leg) and leg["gsis_id"] != td["gsis_id"])), no_third(td_floor)))


def _large(pool_, small, floor, pick_team, td_floor):
    td = pool_.best(floor)
    return sgp._slots((td, no_td(td_floor)),
                      (lambda: pool_.best(lambda leg: floor(leg) and leg["gsis_id"] != td["gsis_id"]),
                       no_second_td(td_floor)),
                      (pool_.best(sgp._is_result), NO_RESULT), (pool_.best(sgp._is_total), NO_TOTAL))


_FILL = {"small": _small, "rookie": _rookie, "medium": _medium, "large": _large}


def build_tickets(legs, pick_team, p_pick, confident=CONFIDENT, td_floor=TD_FLOOR):
    """The game's four v2 tickets, in v1's KINDS order: a game whose pick's
    chance is under `confident` (or has none) gets four unbuilt tickets with
    one reason; a confident one builds from pool() as v1 builds - each slot
    the highest-chance leg still available (in fewer than nfl_sgp.USE_CAP of
    the tickets built so far; ties by key), SMALL's TD from the pick's team
    first, ROOKIE moved off a ticket identical to SMALL's, and a ticket with
    an empty slot unbuilt with why. Tickets are nfl_sgp's (_ticket: legs,
    product chance, est_payout), so nfl_sgp grades and meters them."""
    why = _gate(pick_team, p_pick, confident, td_floor)
    if why is not None:
        return [sgp._unbuilt(kind, why) for kind in KINDS]
    pool_, floor, small, out = sgp._Pool(pool(legs, pick_team, td_floor)), _floor_td(td_floor), None, []
    for kind in KINDS:
        chosen, reason = _FILL[kind](pool_, small, floor, pick_team, td_floor)
        if chosen is None:
            out.append(sgp._unbuilt(kind, reason))
            continue
        assert len(chosen) == LEGS[kind], (kind, chosen)
        pool_.take(chosen)
        if kind == "small":
            small = chosen
        out.append(sgp._ticket(kind, chosen))
    return out


def game_best_small(legs, pick_team, p_pick, confident=CONFIDENT, td_floor=TD_FLOOR):
    """REPORTED BESIDE V2 (report-only): SMALL with the game's best TD at
    `td_floor` or more, whichever team, instead of the pick's team's first -
    the same game gate, the same result leg. It never builds a live ticket."""
    why = _gate(pick_team, p_pick, confident, td_floor)
    if why is not None:
        return sgp._unbuilt("small", why)
    pool_ = sgp._Pool(pool(legs, pick_team, td_floor))
    chosen, reason = sgp._slots((pool_.best(_floor_td(td_floor)), no_td(td_floor)),
                                (pool_.best(sgp._is_result), NO_RESULT))
    return sgp._unbuilt("small", reason) if chosen is None else sgp._ticket("small", chosen)


# ---------------------------------------------------------------- the rule's words

def rules(confident=CONFIDENT, td_floor=TD_FLOOR):
    """The docstring's narrowing in the words the app can show beside v2's
    tickets, each number from the constants (None until they are chosen)."""
    if confident is None or td_floor is None:
        return None
    return {
        "games": (f"only a confident game gets tickets: the pick's chance is {confident:.0%} or more; "
                  "every other game shows no tickets, and why"),
        "result": "result legs only on the pick's side: the pick's WIN and the pick's SPREAD",
        "td": (f"only players the model rates very likely: a TD chance of {td_floor:.0%} or more (the "
               f"rookie leg keeps v1's {sgp.ROOKIE_FLOOR:.0%} and must be on the pick's team)"),
        "small": (f"the pick's best TD scorer: the best TD leg on the pick's team at {td_floor:.0%} or "
                  "more, else the game's best that clears it, + the best result leg on the pick's side"),
        "total": ("unchanged from v1 (the market's no-vig side): MEDIUM's third slot competes with the "
                  "TD legs, LARGE's fourth slot is a total"),
        "chosen": (f"{confident:.0%} and {td_floor:.0%} were chosen on {DEV[0]}-{DEV[1]} only, by "
                   f"{OBJECTIVE}; measured untouched on {CONFIRM[0]}-{CONFIRM[1]} and the current season"),
    }
