"""Same-game tickets: four parlays per NFL game, built before kickoff from the
engine's own chances, with every leg from that one game.

PRE-REGISTERED 2026-11-23, before any ticket was built or any result
computed. The ticket shapes below are the user's (small, medium, large,
rookie); the rule fixes how each slot is filled, so that the backtest
replays exactly what the app shows. Changing a number here after the
backtest has run is a new version (V2...), with this one kept and measured
beside it.

THE LEGS (v1). Four kinds, each with a chance the engine has measured and a
price the market posts:

  WIN     a team to win (moneyline, overtime included). Chance: the pick's
          blend - P(home) = the walk-forward blend_prob, away = 1 - it
          (nfl_model; 66.4% of winners and log loss 0.6080 on 2016-2025,
          the closing favourite 66.6% and 0.6085). A tie voids the leg.
  SPREAD  a team at the posted spread (the handicap). Chance: the market's
          no-vig at the consensus spread prices - the model does not beat
          the line (its own cover chance scored 0.707 log loss on
          2016-2025, worse than a coin's 0.693). A push voids.
  TOTAL   over or under the posted game total. Chance: the market's no-vig
          at the consensus total prices. A push voids.
  TD      a player scores 1+ rushing or receiving touchdown, overtime
          included - passing, return and fumble-recovery touchdowns do not
          count, which is also Stake's "anytime touchdown" rule. Chance:
          nfl_td's walk-forward probability, which is conditional on the
          player playing; only players in a usage group the walk-forward
          measured as calibrated (nfl_td.shipped_groups: a live chance that
          is not None). A player who does not play voids the leg, as Stake
          voids a player listed inactive.

  A ROOKIE leg is a TD leg on a player in his first NFL season: no game in
  player_games in any earlier season. Live and in the backtest alike.

  Not in v1, because nothing on file measures them yet: QB pass attempts,
  yards over a line, first/second-half points, team or game touchdown
  totals, alternate ("Asian") lines. Each needs a probability measured
  against outcomes and a posted price; they arrive as V2 legs with their
  own measurement, never as an unmeasured guess.

THE TICKETS. Built in this order, each slot taking the highest-chance leg
still AVAILABLE (ties: the leg key, alphabetically). A leg is available to
a ticket while it is in fewer than USE_CAP of the game's tickets, so one
bad leg sinks at most two of the four. Within a ticket: one result leg
(WIN or SPREAD, either team), one TOTAL leg, and TD legs on different
players.

  SMALL   (2 legs) the best TD + the best result leg.
  ROOKIE  (2 legs) the best rookie TD with a chance of ROOKIE_FLOOR or more
          + WIN for the rookie's own team. A ticket identical to SMALL's
          moves to the next qualifying rookie. No qualifying rookie, or
          that WIN no longer available: no ticket, and the reason.
  MEDIUM  (3 legs) the best TD + the best result leg + the best of the
          TOTAL legs and the TD legs on other players.
  LARGE   (4 legs) the best TD + the next-best TD (another player) + the
          best result leg + the best TOTAL leg.

  A slot with nothing available leaves its ticket unbuilt, with the reason
  shown; a ticket is never filled with a leg outside its rule.

THE CHANCE AND THE PRICE.
  chance        the product of the legs' chances - what the ticket would
                hit if its legs were independent. Legs in one game are not
                (a team winning and its running back scoring go together),
                so the backtest measures how often each kind actually hit
                beside this product, and the app shows both.
  est. payout   the product of the legs' consensus single prices, marked
                "est.". Stake's bet builder prices correlated legs BELOW
                the product of the singles, so this is an upper bound on
                what Stake pays. The exact payout is the price Stake quotes
                for the slip: the app takes it as typed and computes the
                return on the stake from it.

THE ROI METERS. For each kind, independently, as if $10 had been bet on
every ticket of that kind:
  history   the backtest (experiments/nfl_sgp_backtest.py) over every game
            from the first season the TD walk-forward covers (2014) to the
            last one played, with the periods 2014-2019, 2020-2025 and the
            current season reported apart. Its inputs are point-in-time:
            the walk-forward blend and TD chances, the closing lines and
            prices (games.csv.gz), and the rookie flag. Tickets hit and
            tickets built are measured. Return needs every leg's price and
            no past anytime-TD price is on file, so return is shown as
            missing with that reason until a price history is added.
  live      every ticket logged before kickoff (prediction_history.json,
            the game's "tickets") and graded from the box score, its return
            at the logged est. payout (so an upper bound, and marked so).

  KNOWN BIAS, said where the meter is: in the backtest a TD candidate is a
  player who took a snap, so the replay knows who was active. Live, a pick
  who does not play is voided and the ticket is graded on its other legs.

Nothing here claims an edge. The market is hard to beat, and parlays
multiply the margin of every leg; these tickets are the legs the engine
rates most likely to land, not bets it rates profitable.
"""

import math
from collections import Counter

VERSION = "sgp-v1"
KINDS = ("small", "rookie", "medium", "large")     # build order
LEGS = {"small": 2, "rookie": 2, "medium": 3, "large": 4}
USE_CAP = 2                    # a leg is in at most this many of a game's tickets
ROOKIE_FLOOR = 0.25            # a rookie TD chance this high or more qualifies
LEG_KINDS = ("WIN", "SPREAD", "TOTAL", "TD")
RESULT_KINDS = ("WIN", "SPREAD")
HISTORY_FROM = 2014            # the TD walk-forward's first scored season
PERIODS = {"2014-2019": (2014, 2019), "2020-2025": (2020, 2025)}
STAKE_UNIT = 10.0              # the meters' bet per ticket, in dollars

# The words the meters and the backtest show beside a number they cannot
# give, or beside a number that leans one way: one wording, live and history.
KNOWN_BIAS = ("In the backtest a TD candidate is a player who took a snap, so the replay knows "
              "who was active. Live, a pick who does not play is voided and the ticket is graded "
              "on its other legs.")
NO_TD_PRICE = "no anytime-TD price history on file"
NO_EDGE = ("Nothing here claims an edge. The market is hard to beat, and parlays multiply the "
           "margin of every leg; these tickets are the legs the engine rates most likely to land, "
           "not bets it rates profitable.")
EST_PAYOUT = ("est. - the product of the legs' consensus single prices; Stake's bet builder prices "
              "correlated legs below it, so it is an upper bound on what Stake pays")

WON, LOST, VOID = "won", "lost", "void"
TOTAL_SIDES = ("OVER", "UNDER")

# Why a ticket was not built: one sentence per empty slot, so the history
# can count them and the app can show the same words.
NO_TD = "no TD leg available"
NO_SECOND_TD = "no TD leg on a second player available"
NO_RESULT = "no result leg (WIN or SPREAD) available"
NO_TOTAL = "no TOTAL leg available"
NO_THIRD = "no TOTAL leg or TD leg on another player available"
NO_ROOKIE = f"no rookie TD leg with a chance of {ROOKIE_FLOOR:.0%} or more available"
ROOKIE_IS_SMALL = (f"the only rookie TD leg at {ROOKIE_FLOOR:.0%} or more, with his team's WIN, "
                   "is SMALL's ticket")
NO_ROOKIE_WIN = "no WIN leg available for the rookie's own team"

# The rule in the words the app shows beside the tickets and their meters
# (nfl_export.sgp_block), each from the docstring above, its numbers from
# the constants: one wording, live and history.
LEG_RULES = {
    "WIN": "a team to win, overtime included; chance: the pick's blend; a tie voids the leg",
    "SPREAD": ("a team at the posted spread; chance: the market's no-vig at the consensus spread "
               "prices, since the model does not beat the line; a push voids the leg"),
    "TOTAL": ("over or under the posted game total; chance: the market's no-vig at the consensus "
              "total prices; a push voids the leg"),
    "TD": ("a player scores 1+ rushing or receiving touchdown, overtime included - passing, return "
           "and fumble-recovery touchdowns do not count; chance: the touchdown model's, if he plays, "
           "only where his usage group was measured as calibrated; a player who does not play voids "
           "the leg"),
}
ROOKIE_RULE = ("A ROOKIE leg is a TD leg on a player in his first NFL season: no game in player_games "
               "in any earlier season.")
TICKET_RULES = {
    "small": "the best TD + the best result leg (WIN or SPREAD)",
    "rookie": (f"the best rookie TD with a chance of {ROOKIE_FLOOR:.0%} or more + WIN for the rookie's "
               "own team"),
    "medium": ("the best TD + the best result leg + the best of the TOTAL legs and the TD legs on "
               "other players"),
    "large": ("the best TD + the next-best TD (another player) + the best result leg + the best "
              "TOTAL leg"),
}
SLOT_RULE = (f"Built in the order {', '.join(k.upper() for k in KINDS)}, each slot taking the "
             f"highest-chance leg still available: a leg is in at most {USE_CAP} of a game's "
             f"{len(KINDS)} tickets. A slot with nothing available leaves its ticket unbuilt, with "
             "the reason.")
CHANCE_NOTE = ("The chance is the product of the legs' chances - what the ticket would hit if its "
               "legs were independent. Legs in one game are not (a team winning and its running back "
               "scoring go together), so the backtest measures how often each kind actually hit "
               "beside this product, and both are shown.")
PRICE_NOTE = ("Every price is the books' consensus single price, not Stake's: an estimate, marked "
              "est.")
EXACT_PAYOUT = ("The exact payout is the price Stake quotes for the slip: enter it as typed and the "
                "return on the stake is worked out from it.")
METER_NOTE = (f"Each kind is metered on its own, as if ${STAKE_UNIT:.0f} had been bet on every ticket "
              "of that kind: hit rate is won / (won + lost), beside the mean product chance of the "
              "same tickets.")
LIVE_ROI = ("Live: every ticket logged before kickoff, graded from the box score, its return at the "
            "logged est. payout - so an upper bound, and marked so.")
NOT_IN_V1 = ("Not in v1, because nothing on file measures them yet: QB pass attempts, yards over a "
             "line, first/second-half points, team or game touchdown totals, alternate (\"Asian\") "
             "lines.")


# ---------------------------------------------------------------- numbers

def _number(x):
    """x as a finite float, or None (None, NaN, text, inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _chance(x):
    """A probability strictly between 0 and 1, or None: a leg that is
    certain or impossible is not a measured chance."""
    v = _number(x)
    return v if v is not None and 0.0 < v < 1.0 else None


def _decimal(x):
    """A decimal price (more than 1), or None."""
    v = _number(x)
    return v if v is not None and v > 1.0 else None


def decimal_from_american(odds):
    """The decimal price of American odds (+150 -> 2.5, -200 -> 1.5), or
    None where there is no valid price (missing, or between -100 and +100)."""
    a = _number(odds)
    if a is None or -100.0 < a < 100.0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def no_vig(first, second):
    """The market's chance of the FIRST side of a two-way market, its margin
    removed in proportion to each side's implied chance:
    (1/a) / (1/a + 1/b) for decimal prices a and b. None without both."""
    a, b = _decimal(first), _decimal(second)
    if a is None or b is None:
        return None
    return (1.0 / a) / (1.0 / a + 1.0 / b)


# ---------------------------------------------------------------- the legs

def _leg(key, kind, chance, decimal, label, **fields):
    p = _chance(chance)
    if p is None:
        raise ValueError(f"{key}: a leg needs a chance strictly between 0 and 1, not {chance!r}")
    return dict(key=key, kind=kind, **fields, chance=p, decimal=_decimal(decimal), label=label)


def win_leg(team, chance, decimal=None):
    """`team` to win, overtime included; a tie voids it. `chance` is the
    blend's for that team."""
    return _leg(f"WIN:{team}", "WIN", chance, decimal, f"{team} to win", team=team)


def spread_leg(team, line, chance, decimal=None):
    """`team` at the handicap `line` (its own: -3.5 gives 3.5 points away):
    it covers when its margin plus `line` is above 0; exactly 0 is a push
    and voids the leg."""
    line = float(line)
    return _leg(f"SPREAD:{team}", "SPREAD", chance, decimal, f"{team} {line:+g}", team=team, line=line)


def total_leg(side, line, chance, decimal=None):
    """OVER or UNDER the game total `line`; landing on it is a push and
    voids the leg."""
    side = str(side).upper()
    if side not in TOTAL_SIDES:
        raise ValueError(f"a total leg's side is OVER or UNDER, not {side!r}")
    line = float(line)
    return _leg(f"TOTAL:{side}", "TOTAL", chance, decimal, f"{side.title()} {line:g}",
                side=side, line=line)


def td_leg(gsis_id, name, position, team, chance, rookie=False, decimal=None):
    """The player scores 1+ rushing or receiving touchdown. `chance` is
    nfl_td's, IF HE PLAYS - so a player who does not play voids the leg.
    `rookie`: no game in player_games in any earlier season (is_rookie)."""
    return _leg(f"TD:{gsis_id}", "TD", chance, decimal, f"{name} anytime TD",
                gsis_id=gsis_id, name=name, position=position, team=team, rookie=bool(rookie))


def game_legs(home, away, p_home=None, spread_line=None, home_spread_price=None,
              away_spread_price=None, total_line=None, over_price=None, under_price=None,
              home_ml=None, away_ml=None, td=()):
    """Every candidate leg of one game, highest chance first (ties: key).

    Prices are DECIMAL (decimal_from_american converts the schedule's and
    the odds feed's American prices); a missing one leaves the leg's
    `decimal` None and only the WIN and TD legs can do without one, since a
    SPREAD's and a TOTAL's chance IS the no-vig of their two prices.

      p_home       P(home wins), the walk-forward blend_prob; the away team
                   gets 1 - it. None: no WIN legs.
      spread_line  the HOME side's expected margin (nflverse's spread_line;
                   positive = home favoured), so home plays at -spread_line
                   and away at +spread_line.
      td           [{gsis_id, name, position, team, prob, rookie[, decimal]}]
                   - only candidates whose prob is not None become legs
                   (nfl_td gives None to a usage group not measured as
                   calibrated).
    """
    legs = []
    p = _chance(p_home)
    if p is not None:
        legs += [win_leg(home, p, home_ml), win_leg(away, 1.0 - p, away_ml)]
    line, q = _number(spread_line), no_vig(home_spread_price, away_spread_price)
    if line is not None and q is not None:
        legs += [spread_leg(home, -line, q, home_spread_price), spread_leg(away, line, 1.0 - q, away_spread_price)]
    line, q = _number(total_line), no_vig(over_price, under_price)
    if line is not None and q is not None:
        legs += [total_leg("OVER", line, q, over_price), total_leg("UNDER", line, 1.0 - q, under_price)]
    seen = set()
    for c in td or ():
        if _chance(c.get("prob")) is None or c["gsis_id"] in seen:
            continue
        seen.add(c["gsis_id"])
        legs.append(td_leg(c["gsis_id"], c.get("name"), c.get("position"), c.get("team"), c["prob"],
                           c.get("rookie", False), c.get("decimal")))
    return _ranked(legs)


def _ranked(legs):
    return sorted(legs, key=lambda leg: (-leg["chance"], leg["key"]))


def first_seasons(player_games):
    """{gsis_id: the first season he has a game in player_games} - what
    is_rookie() reads. Season S's rookie flags need only the rows of
    seasons before S, so passing the whole file is point in time."""
    if player_games is None or not len(player_games):
        return {}
    return player_games.groupby("player_id")["season"].min().astype(int).to_dict()


def is_rookie(first, gsis_id, season):
    """A player in his first NFL season: no game in player_games in any
    season before `season` (`first` is first_seasons())."""
    seen = first.get(gsis_id)
    return seen is None or int(seen) >= int(season)


# ---------------------------------------------------------------- the tickets

def _is_td(leg):
    return leg["kind"] == "TD"


def _is_result(leg):
    return leg["kind"] in RESULT_KINDS


def _is_total(leg):
    return leg["kind"] == "TOTAL"


class _Pool:
    """A game's legs in rank order, and how many built tickets hold each."""

    def __init__(self, legs):
        self.legs = _ranked(legs)
        self.used = Counter()

    def available(self, leg):
        return self.used[leg["key"]] < USE_CAP

    def all(self, ok):
        return [leg for leg in self.legs if self.available(leg) and ok(leg)]

    def best(self, ok):
        for leg in self.legs:
            if self.available(leg) and ok(leg):
                return leg
        return None

    def find(self, key):
        return next((leg for leg in self.legs if leg["key"] == key), None)

    def take(self, legs):
        for leg in legs:
            self.used[leg["key"]] += 1


def _slots(*picks):
    """The filled slots, or the reason of the first empty one."""
    legs = []
    for leg, reason in picks:
        leg = leg() if callable(leg) else leg
        if leg is None:
            return None, reason
        legs.append(leg)
    return legs, None


def _small(pool, small):
    return _slots((pool.best(_is_td), NO_TD), (pool.best(_is_result), NO_RESULT))


def _rookie(pool, small):
    """The best rookie TD at ROOKIE_FLOOR or more and his own team's WIN;
    the next rookie when that pair is SMALL's ticket."""
    small_keys = None if small is None else {leg["key"] for leg in small}
    held = False
    for td in pool.all(lambda leg: _is_td(leg) and leg.get("rookie") and leg["chance"] >= ROOKIE_FLOOR):
        win = pool.find(f"WIN:{td['team']}")
        if win is not None and {td["key"], win["key"]} == small_keys:
            held = True
            continue
        if win is None or not pool.available(win):
            return None, NO_ROOKIE_WIN
        return [td, win], None
    return None, ROOKIE_IS_SMALL if held else NO_ROOKIE


def _medium(pool, small):
    td = pool.best(_is_td)
    return _slots((td, NO_TD), (pool.best(_is_result), NO_RESULT),
                  (lambda: pool.best(lambda leg: _is_total(leg) or (
                      _is_td(leg) and leg["gsis_id"] != td["gsis_id"])), NO_THIRD))


def _large(pool, small):
    td = pool.best(_is_td)
    return _slots((td, NO_TD),
                  (lambda: pool.best(lambda leg: _is_td(leg) and leg["gsis_id"] != td["gsis_id"]),
                   NO_SECOND_TD),
                  (pool.best(_is_result), NO_RESULT), (pool.best(_is_total), NO_TOTAL))


_FILL = {"small": _small, "rookie": _rookie, "medium": _medium, "large": _large}


def _ticket(kind, legs):
    legs = [dict(leg) for leg in legs]
    prices = [leg["decimal"] for leg in legs]
    unpriced = [leg["label"] for leg in legs if leg["decimal"] is None]
    return {"kind": kind, "built": True, "reason": None, "legs": legs,
            "chance": math.prod(leg["chance"] for leg in legs),
            "est_payout": None if unpriced else math.prod(prices),
            "est_payout_reason": ("no price on file for " + ", ".join(unpriced)) if unpriced else None}


def _unbuilt(kind, reason):
    return {"kind": kind, "built": False, "reason": reason, "legs": [], "chance": None,
            "est_payout": None, "est_payout_reason": "no ticket"}


def build_tickets(legs):
    """The game's four tickets, in KINDS order, as the docstring builds them:
    each slot takes the highest-chance leg still available (in fewer than
    USE_CAP of the tickets built so far; ties by key), and a ticket with an
    empty slot is not built and says why. Each built ticket has `legs`, its
    product `chance` and its `est_payout` (None, with the reason, when a leg
    has no price)."""
    pool, small, out = _Pool(legs), None, []
    for kind in KINDS:
        chosen, reason = _FILL[kind](pool, small)
        if chosen is None:
            out.append(_unbuilt(kind, reason))
            continue
        assert len(chosen) == LEGS[kind], (kind, chosen)
        pool.take(chosen)
        if kind == "small":
            small = chosen
        out.append(_ticket(kind, chosen))
    return out


# ---------------------------------------------------------------- grading

def outcome(home, away, home_score=None, away_score=None, players=None):
    """What grading reads: the final score (overtime included; None until
    final) and `players`, {gsis_id: rushing + receiving TDs} of every player
    who played - one absent did not play - or None until the box score is in."""
    return {"home": home, "away": away, "home_score": _number(home_score),
            "away_score": _number(away_score), "players": players}


def grade_leg(leg, result):
    """WON, LOST, VOID, or None while the result is not in. A tie voids a
    WIN, a push a SPREAD or TOTAL, and a player who did not play a TD."""
    if leg["kind"] == "TD":
        players = result.get("players")
        if players is None:
            return None
        if leg["gsis_id"] not in players:
            return VOID
        tds = _number(players[leg["gsis_id"]])
        return None if tds is None else (WON if tds >= 1 else LOST)
    hs, as_ = _number(result.get("home_score")), _number(result.get("away_score"))
    if hs is None or as_ is None:
        return None
    if leg["kind"] == "TOTAL":
        diff = (hs + as_ - leg["line"]) * (1 if leg["side"] == "OVER" else -1)
    else:
        if leg["team"] not in (result["home"], result["away"]):
            raise ValueError(f"{leg['key']} is not a leg of {result['away']} @ {result['home']}")
        diff = (hs - as_) * (1 if leg["team"] == result["home"] else -1)
        if leg["kind"] == "SPREAD":
            diff += leg["line"]
    return VOID if diff == 0 else (WON if diff > 0 else LOST)


def grade_ticket(ticket, result):
    """The ticket with its grade: `result` LOST when any leg lost, else None
    while a leg waits, VOID when every leg voided, else WON; `leg_results`
    by key; and the ticket re-priced on the legs that stand - `payout`, the
    product of their prices (None when one has none; 1 when all voided, the
    stake back) and `chance_graded`, the product of their chances, which is
    what a graded ticket needed to land."""
    out = dict(ticket, result=None, leg_results={}, payout=None, chance_graded=None)
    if not ticket["built"]:
        return out
    graded = {leg["key"]: grade_leg(leg, result) for leg in ticket["legs"]}
    out["leg_results"] = graded
    standing = [leg for leg in ticket["legs"] if graded[leg["key"]] != VOID]
    prices = [leg["decimal"] for leg in standing]
    out["payout"] = math.prod(prices) if all(p is not None for p in prices) else None
    out["chance_graded"] = math.prod(leg["chance"] for leg in standing)
    states = set(graded.values())
    if LOST in states:
        out["result"] = LOST
    elif None in states:
        out["result"] = None
    elif states == {VOID}:
        out["result"] = VOID
    else:
        out["result"] = WON
    return out


# ---------------------------------------------------------------- meters

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _profit(ticket):
    if ticket["result"] == WON:
        return STAKE_UNIT * (ticket["payout"] - 1.0)
    return -STAKE_UNIT if ticket["result"] == LOST else 0.0


def meter(tickets, no_price_reason=None):
    """One kind's record (see summarise)."""
    built = [t for t in tickets if t["built"]]
    graded = [t for t in built if t.get("result") in (WON, LOST, VOID)]
    decided = [t for t in graded if t["result"] != VOID]
    won = sum(t["result"] == WON for t in decided)
    out = {
        "tickets": len(built), "graded": len(graded), "won": won, "lost": len(decided) - won,
        "void": len(graded) - len(decided), "pending": len(built) - len(graded),
        "not_built": len(tickets) - len(built),
        "not_built_why": dict(Counter(t["reason"] for t in tickets if not t["built"]).most_common()),
        "hit_rate": won / len(decided) if decided else None,
        "mean_chance": _mean(t["chance"] for t in decided),
        "mean_chance_graded": _mean(t["chance_graded"] for t in decided),
        "hit_rate_reason": None if decided else "no ticket decided yet",
        "stake": STAKE_UNIT, "profit": None, "roi": None, "roi_reason": None,
    }
    unpriced = [t for t in graded if t.get("est_payout") is None]
    if not graded:
        out["roi_reason"] = "no ticket graded yet"
    elif unpriced:
        out["roi_reason"] = no_price_reason or (
            f"{len(unpriced)} of {len(graded)} graded tickets have a leg with no price on file")
    else:
        profit = sum(_profit(t) for t in graded)
        out["profit"], out["roi"] = profit, profit / (STAKE_UNIT * len(graded))
    return out


def summarise(tickets, no_price_reason=None):
    """{kind: meter} over graded tickets (grade_ticket), each kind on its own:
    tickets built, graded, won, lost, void (every leg voided), pending, the
    tickets not built and why; hit rate = won / (won + lost); mean_chance =
    the product chance shown, over the same decided tickets, and
    mean_chance_graded = the product over the legs that stood (a voided leg
    drops out of a ticket, so its hit rate is measured against this one);
    and, as if STAKE_UNIT had been bet on every graded ticket (a void
    returns the stake), profit and roi = profit / staked - only when every
    graded ticket has every leg's price, else None with `roi_reason`
    (`no_price_reason` when the caller knows it, e.g. NO_TD_PRICE). A
    ticket's payout is its est. one, so the return is an upper bound."""
    return {kind: meter([t for t in tickets if t["kind"] == kind], no_price_reason) for kind in KINDS}
