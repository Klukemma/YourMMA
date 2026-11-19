"""Same-day parlays: the likeliest 3-5x ticket a day, from any of the three sports.

THE RULE - fixed here before the backtest (experiments/sameday_backtest.py)
was run, and not changed after any of its numbers was seen:

  DAY      the America/New_York calendar date of each event's scheduled
           start. A UFC card takes the ET date of its first bell (live: the
           card's first_start, else the logged first_start, else the event
           date; historically the event date). An NFL game takes its
           schedule day (ET). A top-five soccer match takes its kickoff's ET
           date - the UK date, which equals it for every timed top-five
           match 2019-2026. Every leg of a ticket is on one such day, so the
           whole ticket settles that day (a UFC main event can end after
           midnight ET; a non-US card's local date can differ from ET).
  LEGS     one per event, and only that sport's SHIPPED pick at its price:
           UFC the blended pick side (model + market at the shipped weight),
           NFL the blend's pick (home when it gives the home side 0.5 or
           more), soccer the shipped pick H/D/A (the market recalibrated).
           A leg needs a chance of at least MIN_P (0.50) and a price.
  TICKET   among the day's POOL (12) likeliest legs, the combination of 1 to
           MAX_LEGS (6) legs whose payout lies in BAND (3.0-5.0x) with the
           highest combined chance - the product of the legs' chances; on a
           tie, fewer legs. One ticket a day; none when no combination fits.
  CONTROL  the same search ranked by the market's no-vig chance alone
           (market_only): what the ticket would be with no model at all.

"Most realistic" means exactly that: the ticket most likely to land among
those paying 3-5x. It is not a bet expected to profit - the market is hard
to beat, and a parlay pays the bookmaker's margin once for every leg.

THE UNDERDOG TICKET (v1) - a second ticket a day, fixed here on 2026-11-12
before any of its results was computed (only outcome-free numbers - counts,
prices, price-implied returns - were computed to write it), and not to be
changed after any of them is seen:

  WHY      the likeliest ticket is favourites by construction (every leg at
           least a 50% chance), and the user asked for tickets that at times
           take the underdog, by a randomizer that makes some sense. A draw
           cannot create an edge - every 3-5x ticket pays the bookmaker's
           margin - so this ticket is for variety and a bigger payout, never
           sold as the better bet.
  DOG      per event, the side the no-vig market rates less likely to win,
           at the same single price snapshot that pays (UFC the card's line,
           NFL the moneyline, soccer the pre-match average): its chance q is
           under 0.5. In soccer the less likely of the two TEAMS - never the
           draw. No candidate when the sides are level (equal no-vig
           chances). A candidate is priced within DOG_PRICE (2.00-4.00
           decimal, +100 to +300) and its shipped chance - that side's chance
           as its sport ships it: UFC and NFL the blend's, soccer the shipped
           pick's three-way chance (recalibrated, or the raw no-vig price
           where the league ships it raw) - is at least DOG_FLOOR (0.20; the
           4.00 cap already means about 1 in 4 by the price, so the floor
           only guards against a model far below the market). The dog may be
           the sport's own pick (a UFC blend that picks the market's
           underdog).
  LOTTERY  each candidate gets u = (the first 8 bytes of SHA-256 of
           "salt|day|lot key", read big-endian, + 0.5) / 2**64 - never 0 or
           1 - and the score -ln(u) / p, p its shipped chance; the candidates
           are tried in ascending score. That is weighted sampling without
           replacement (Efraimidis-Spirakis): the first is drawn with a
           chance proportional to p, and the order of any two candidates
           never depends on which other events are on the day, so a price
           moving elsewhere does not reshuffle the draw. Candidates' chances
           run about 0.26-0.46, so a 45% underdog comes up about 1.5 times as
           often as a 30% one: close to an even draw among them. The salt is
           DOG_SALT live; the lot key is dog_lot_key(), one string live and
           in the backtest (NFL the game id, soccer the match id, UFC the
           two corners' normalised names, sorted).
  FILL     with the dog fixed: from the POOL (12) likeliest legs of the day
           on OTHER events (the likeliest ticket's legs - shipped picks, at
           least a 50% chance, priced - with the dog's event removed first),
           the combination of 1 to MAX_LEGS - 1 legs that brings the ticket
           into BAND (3.0-5.0x) with the highest product of chances; ties to
           fewer legs. So a ticket has 2 to 6 legs. No fill: the next dog in
           score order. None fits: no underdog ticket that day.
  SHIP     this ticket ships beside the likeliest one, with its measured
           record, whatever the record shows. Nothing measured beside it can
           replace it in v1. Any change is a v2 (a new salt), judged only on
           tickets logged live after v1, with v1's record kept.
  MEASURED (experiments/sameday_backtest.py) every day 2016-08-01 to
           2026-08-08, as the likeliest ticket, in its periods FIRST
           (2016-08-01..2023-08-08) and LAST3 (2023-08-09..2026-08-08): the
           live salt's record - the headline, with a 95% interval from a
           bootstrap over days (4,000 resamples, seed 7); the mean over the
           1,000 salts DOG_SIM_SALTS, with a day-bootstrap interval of each
           day's mean profit over the salts; the salts' 5th, 50th and 95th
           percentiles, labelled "how much the draw alone moves it, same
           games" - that spread is not a confidence interval - and the live
           salt's rank among them. Beside it, report-only: the largest model
           gap first (shipped minus market chance; in soccer that gap is the
           recalibration curve, not a model opinion - never called "value"),
           the likeliest dog first (the largest shipped chance), and the
           market-only lottery (weights, floor and fill by the no-vig
           chance, same salt); the likeliest ticket paired on the days both
           exist, and both tickets held (2 units a day). Every row beside its
           price-implied return (the product of its legs' no-vig chances
           times its multiplier, less 1). Splits by sport, dog price and year
           are descriptive: none can move the band, the floor, the weight or
           the sports in v1.
  EXPECTED (written before the results) by the prices the underdog ticket
           lands about 1 time in 4 (a market chance of about 0.25 at about
           3.8x), the likeliest about 2 in 7 at about 3.1x; price-implied
           about -8% a ticket. Bookmakers load more margin on underdogs
           (about 5% on a dog leg against 3% on a favourite) and the
           proportional no-vig spreads it evenly, so a dog leg's market
           chance reads a little generous: it is expected to do somewhat
           worse than its prices imply.
  PRIOR EXPOSURE the rule's own results were not seen before it was fixed,
           but underdog results over the same years had been: the UFC
           profiles' risker and risk-lover tickets (engine/experiments/
           parlay_profiles.json) and the NFL and soccer edge-betting checks.
           Its record is not an out-of-sample test of underdogs.

TICKETS v2 - fixed here on 2026-11-14, before any of their results was
computed. Asked for by the user ("the builder barely builds; no KO/sub/
decision picks; pure moneyline - make it more active and more diverse").
Only outcome-free numbers (coverage, counts, prices, overrounds, price-
implied returns, an estimator's error against real PRICES) were computed to
write it: three scouts' reports, scratchpad scout_*/. v1's rules and records
stay as they are (kept on the page as v1); v2 is measured on its own.

  LEGS v2  - the pool every v2 ticket draws from, one leg per event (key =
           the event, so a match, game or fight gives at most one leg to a
           ticket), each priced at the snapshot named here and live at the
           price the app files carry:
    1X2    soccer: the shipped pick of EVERY league in the app
           (soccer_leagues.in_app at the time), at the pre-match average
           (history avg_*; live prices.avg). The 15 extra leagues' history
           has closing prices only, so their legs are measured at the close
           while live pays the pre-match price; every record also reports
           itself without them.
    OU25   soccer over/under 2.5 goals, in each league that ships over 2.5
           from the market (soccer_leagues.withheld): the side the
           pre-match no-vig over/under price favours (level: no leg), its
           chance that no-vig chance (the market's - what the app ships; the
           model was measured worse), its price the pre-match average
           (history avg_o25/avg_u25; live prices.avg.o25/u25). Graded on the
           total goals.
    SPREAD / TOTAL  NFL: the side whose no-vig chance at the spread (total)
           prices is at least 0.5 - level prices: the shipped model's side,
           chance 0.5; a zero spread gives no leg - its chance that no-vig
           chance (the model was measured not to beat the line), its price
           the closing price in history (nflverse home/away_spread_odds,
           over/under_odds) and live the consensus price at the consensus
           line. A push voids the leg.
    METHOD UFC: for each corner, "wins by KO/TKO", "wins by submission" and
           "wins by decision" - the props history prices (engine/data/
           method_odds.csv; no real price exists for "goes the distance" or
           "inside the distance", so neither is offered). Chance: the
           corner's shipped win chance (the blend, as the UFC moneyline leg)
           times the share of that corner's wins going that way - by the
           six-way no-vig market in history, by METHOD ESTIMATOR live. Price:
           in history the real prop; LIVE THERE IS NO PROP PRICE on our feed
           (the odds api prices no method market), so the leg carries the
           estimator's price, shown as an estimate with its measured error,
           and a ticket holding one shows its multiplier as an estimate.
           METHOD ESTIMATOR: the scouts' market-only ridge (the moneyline's
           no-vig chance with square and cube terms, men's/women's division,
           five rounds, each corner's UFC record before the fight, shrunk to
           a prior) with the price 1 / (chance x (1 + loading(chance))), the
           loading fitted from prices; frozen for live use from 2016-2024
           with a hash of its coefficients; in the backtest refitted each
           year on earlier years only.
    plus the v1 legs: each sport's shipped moneyline/1X2 pick (UFC, NFL).

  KINDS v2 - four tickets a day, each built, locked at its own first start,
           logged and graded on its own; each 3-5x (BAND), 2-6 legs unless
           said, ties to fewer legs:
    LIKELIEST v2  the v1 rule over the v2 moneyline/1X2 legs (soccer from
           every league; UFC and NFL as v1).
    UNDERDOG v2   the v1 underdog rule, salt "sameday-dog-v2", over the v2
           moneyline/1X2 events (dogs from every soccer league).
    MIXED  the combination with the highest product of chances among the
           POOL (12) likeliest moneyline/1X2 legs and the 6 likeliest
           non-moneyline legs (OU25, SPREAD, TOTAL, METHOD), each at least a
           0.50 chance except METHOD (no floor), that holds at least one
           non-moneyline leg; 1-6 legs.
    PROPS  on days with a UFC card only: exactly one METHOD leg, drawn by
           the v1 lottery (weights its chance, salt "sameday-props-v1", lot
           key dog_lot_key's corners plus the leg type), among METHOD legs
           priced 1.50-4.00 (live: the estimate), plus the likeliest fill
           from the LIKELIEST v2 pool on other events (other fights and
           other sports) to the band. Live, the search reads the estimated
           price and uses the band 3.6-4.2x, the scouts' measured choice
           that lands the real price in 3-5x most often (82% for any method
           leg, 69% for a single prop on 2021-24 cards); the backtest
           reports both "as priced" (chosen and paid at the real props, band
           3-5x) and "as live" (chosen by the yearly-refitted estimator in
           3.6-4.2x, paid at the real props) - the headline is "as live".
           Measured 2016-08-01 to 2024-12-07 only: history has no prop
           after that.

  MEASURED as v1's MEASURED (periods all/FIRST/LAST3, a day bootstrap of
           4,000 resamples with seed 7, price-implied return beside every
           record, the 1,000 simulation salts for each lottery, the live
           salt's rank), plus per kind: tickets per day, the share of days
           with a ticket, and splits by leg type and sport (descriptive:
           none can move a rule in v2).
  SHIP     all four kinds ship with their records, whatever the records
           show; none is presented as an edge.
  EXPECTED (written before the results, from prices only) tickets on about
           70-80% of days instead of about a third; price-implied returns
           about -15% (LIKELIEST v2 and MIXED) and -16% to -20% (PROPS) a
           ticket - a prop carries about 22% margin across its six outcomes
           against about 4% for a moneyline pair.
  PRIOR EXPOSURE v1's likeliest and underdog records had been seen when v2
           was asked for; the soccer over/under market-against-model
           measure, the NFL spread/total measure (no configuration beat
           -110 in both periods) and the UFC prop flat-bet results (underdog
           KO props -24%, submission props -52%: parlay_profiles' docstring)
           had been seen. The extra leagues and the new markets were chosen
           on the scouts' outcome-free counts, not on any result.

Legs are dicts parlay_profiles' search and settlement read (engine/
parlay_profiles.py): key (sport-prefixed, so one leg per event holds across
sports), decimal, p, p_market, side "pick", market "moneyline" and result
{"winner": "pick" | "other" | None (void)} or None (not known yet).
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))

import parlay_profiles as pp  # noqa: E402

DAY_TZ = "America/New_York"
ET = ZoneInfo(DAY_TZ)
BAND = (3.0, 5.0)
MAX_LEGS = 6
POOL = 12
MIN_P = 0.50
SPORTS = ("UFC", "NFL", "Soccer")

# The underdog ticket (v1): see the module docstring.
DOG_VERSION = 1
DOG_SALT = "sameday-dog-v1"
DOG_SIM_SALTS = tuple(f"sameday-dog-sim{i}" for i in range(1000))
DOG_PRICE = (2.00, 4.00)
DOG_FLOOR = 0.20
DOG_ORDERS = ("lottery", "gap", "likeliest")

# Tickets v2: see the module docstring.
V2_VERSION = 2
DOG_SALT_V2 = "sameday-dog-v2"
PROPS_SALT = "sameday-props-v1"
PROPS_SIM_SALTS = tuple(f"sameday-props-sim{i}" for i in range(1000))
DOG_SIM_SALTS_V2 = tuple(f"sameday-dog-v2-sim{i}" for i in range(1000))
PROPS_PRICE = (1.50, 4.00)
PROPS_LIVE_BAND = (3.6, 4.2)
PROPS_HISTORY_END = "2024-12-07"
MIXED_POOL_OTHER = 6
LEG_MARKETS = ("1X2", "OU25", "SPREAD", "TOTAL", "METHOD")
METHOD_TYPES = ("KO/TKO", "Submission", "Decision")


def et_day(when):
    """The America/New_York date of an instant (aware, or naive UTC, or an
    ISO string ending in Z), as YYYY-MM-DD. None for nothing."""
    if when is None or (isinstance(when, float) and math.isnan(when)):
        return None
    if isinstance(when, str):
        if not when.strip():
            return None
        when = datetime.fromisoformat(when.replace("Z", "+00:00"))
    if hasattr(when, "to_pydatetime"):
        when = when.to_pydatetime()
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(ET).strftime("%Y-%m-%d")


def leg(sport, key, day, event, pick, p, decimal, p_market=None, p_model=None,
        opponent=None, kickoff=None, result=None, price_source=None, number=None):
    """One leg, or None when it is not one: no price, or a chance under MIN_P."""
    dec = pp._price(decimal) if hasattr(pp, "_price") else decimal
    if dec is None or not (isinstance(dec, (int, float)) and math.isfinite(dec)) or dec <= 1.0:
        return None
    if p is None or not math.isfinite(float(p)):
        return None
    return {
        "key": f"{sport}:{key}", "sport": sport, "day": day, "event": event,
        "pick": pick, "opponent": opponent, "kickoff": kickoff, "number": number,
        "side": "pick", "market": "moneyline", "decimal": float(dec),
        "p": float(p), "p_model": None if p_model is None else float(p_model),
        "p_market": None if p_market is None else float(p_market),
        "edge": None if (p_model is None or p_market is None) else float(p_model) - float(p_market),
        "favourite": p_market is not None and float(p_market) >= 0.5,
        "price_source": price_source, "result": result,
    }


def build(legs, objective="p"):
    """The day's ticket: the likeliest 1-6 leg combination in BAND among the
    POOL likeliest legs, by the blended chance ("p") or, for the control, the
    market's ("p_market"). None when nothing fits."""
    usable = [l for l in legs if l and l.get(objective) is not None and l[objective] >= MIN_P]
    pool = sorted(usable, key=lambda l: -l[objective])[:POOL]
    picked = pp._search([(pool, 1, MAX_LEGS)], BAND, objective=objective, max_legs=MAX_LEGS)
    if not picked:
        return None
    t = pp.ticket(picked, "same_day" if objective == "p" else "same_day_market_only", BAND)
    t["legs"] = sorted(t["legs"], key=lambda l: (l.get("kickoff") or "", l["sport"], l["key"]))
    t["sports"] = sorted({l["sport"] for l in t["legs"]})
    t["day"] = t["legs"][0].get("day")
    return t


def settle(ticket):
    return pp.settle(ticket)


def summarise(tickets):
    return pp.summarise(tickets)


def american(decimal):
    return pp.american_from_decimal(decimal)


def reason(l):
    """One line from the leg's own numbers."""
    bits = [f"{l['sport']}: chance {l['p']:.0%}"]
    if l.get("p_model") is not None and l["sport"] != "Soccer":
        bits.append(f"model {l['p_model']:.0%}")
    if l.get("p_market") is not None:
        bits.append(f"market {l['p_market']:.0%}")
    return ", ".join(bits)


# ---------------------------------------------------------------- the underdog ticket (v1)

def _norm_name(name):
    """A corner's name for the lot key: accents dropped, lower case, runs
    of anything but letters and digits made one space."""
    t = unicodedata.normalize("NFKD", str(name or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def dog_lot_key(sport, game_id=None, match_id=None, corners=None):
    """The string the lottery hashes for an event, the same live and in the
    backtest: NFL the game id (nflverse's, e.g. 2026_10_WAS_NYG), soccer the
    match id (e.g. D1-20261106-hamburg-mgladbach), UFC the two corners'
    normalised names sorted and joined by "|" - corner order and accents do
    not move the draw. The day is hashed beside it (lottery_u)."""
    if sport == "NFL":
        return str(game_id)
    if sport == "Soccer":
        return str(match_id)
    if sport == "UFC":
        return "|".join(sorted(_norm_name(c) for c in corners))
    raise ValueError(sport)


def lottery_u(salt, day, lot_key):
    """The event's lottery number on the day, in (0, 1): the first 8 bytes
    of SHA-256("salt|day|lot key") as a big-endian integer, + 0.5, / 2**64."""
    h = hashlib.sha256(f"{salt}|{day}|{lot_key}".encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2.0 ** 64


def dog(sport, key, day, event, pick, p, decimal, p_market, lot_key, p_model=None,
        opponent=None, kickoff=None, result=None, price_source=None, number=None):
    """An underdog candidate, or None when it is not one: the side's no-vig
    chance p_market must be under 0.5 (the market's underdog; level sides
    give none), its price within DOG_PRICE. The shipped chance p is kept for
    the floor and the weight (build_dog applies the floor, so the market-only
    control can apply its own to p_market). key is the event's, sport-
    prefixed as leg()'s, so the fill never takes the dog's event."""
    dec = pp._price(decimal) if hasattr(pp, "_price") else decimal
    if dec is None or not (isinstance(dec, (int, float)) and math.isfinite(dec)):
        return None
    if not (DOG_PRICE[0] <= float(dec) <= DOG_PRICE[1]):
        return None
    if p is None or p_market is None or not math.isfinite(float(p)) or not math.isfinite(float(p_market)):
        return None
    if not float(p_market) < 0.5:
        return None
    return {
        "key": f"{sport}:{key}", "sport": sport, "day": day, "event": event,
        "pick": pick, "opponent": opponent, "kickoff": kickoff, "number": number,
        "side": "pick", "market": "moneyline", "decimal": float(dec),
        "p": float(p), "p_model": None if p_model is None else float(p_model),
        "p_market": float(p_market),
        "edge": None if p_model is None else float(p_model) - float(p_market),
        "favourite": False, "dog": True, "lot_key": str(lot_key),
        "price_source": price_source, "result": result,
    }


def dog_order(dogs, salt=DOG_SALT, order="lottery", objective="p"):
    """The candidates in the order they are tried. "lottery": ascending
    -ln(u) / chance (the rule); "gap": the largest shipped-minus-market
    chance first; "likeliest": the largest chance first (both report-only).
    Ties by key, so the order is total."""
    if order == "lottery":
        def score(d):
            return (-math.log(lottery_u(salt, d["day"], d["lot_key"])) / d[objective], d["key"])
    elif order == "gap":
        def score(d):
            return (-(d["p"] - d["p_market"]), d["key"])
    elif order == "likeliest":
        def score(d):
            return (-d[objective], d["key"])
    else:
        raise ValueError(order)
    return sorted(dogs, key=score)


def build_dog(legs, dogs, salt=DOG_SALT, order="lottery", objective="p"):
    """The day's underdog ticket, or None. legs: the day's legs as build()
    takes them (shipped picks); dogs: the day's candidates (dog()).
    objective "p" is the rule; "p_market" the market-only control, whose
    floor, weights and fill read only the no-vig chance."""
    cands = [d for d in dogs if d and d.get(objective) is not None and d[objective] >= DOG_FLOOR]
    usable = [l for l in legs if l and l.get(objective) is not None and l[objective] >= MIN_P]
    for rank, d in enumerate(dog_order(cands, salt, order, objective)):
        others = [l for l in usable if l["key"] != d["key"]]
        pool = sorted(others, key=lambda l: -l[objective])[:POOL]
        picked = pp._search([([d], 1, 1), (pool, 1, MAX_LEGS - 1)], BAND, objective=objective,
                            max_legs=MAX_LEGS)
        if not picked:
            continue
        t = pp.ticket(picked, "same_day_dog" if objective == "p" else "same_day_dog_market_only", BAND)
        t["legs"] = sorted(t["legs"], key=lambda l: (not l.get("dog"), l.get("kickoff") or "", l["sport"], l["key"]))
        t["sports"] = sorted({l["sport"] for l in t["legs"]})
        t["day"] = d["day"]
        t["dog_key"] = d["key"]
        t["dog_rank"] = rank
        t["candidates"] = len(cands)
        t["salt"] = salt if order == "lottery" else None
        t["order"] = order
        return t
    return None


def opposes(dog_ticket, ticket):
    """Whether the underdog ticket's dog is the other side of one of the
    likeliest ticket's legs - then the two tickets cannot both win."""
    if not dog_ticket or not ticket:
        return False
    d = next((l for l in dog_ticket["legs"] if l.get("dog")), None)
    return d is not None and any(l["key"] == d["key"] and l["pick"] != d["pick"] for l in ticket["legs"])


# ---------------------------------------------------------------- tickets v2

WINNER_BETS = (None, "1X2")


def is_winner(l):
    """A leg on who wins (a v1 leg carries no "bet"; a v2 one carries "1X2")."""
    return l.get("bet") in WINNER_BETS


def market_leg(sport, key, day, event, pick, p, decimal, bet, bet_detail=None, p_market=None,
               p_model=None, opponent=None, kickoff=None, result=None, price_source=None,
               number=None, lot_key=None, estimated=False, est_decimal=None):
    """A v2 leg of any market (LEG_MARKETS), or None when it has no price or
    no chance. key is the EVENT's (sport-prefixed as leg()'s), so a match,
    game or fight gives at most one leg to any ticket - its winner, its
    over/under, its spread or a prop, never two that move together.
    "market" stays "moneyline" because parlay_profiles' settlement reads a
    resolved yes/no bet that way; "bet" names the market. Floors are the
    builders' (the likeliest legs need MIN_P; a METHOD leg has none).
    estimated: the price is the METHOD ESTIMATOR's, not a book's (live props);
    est_decimal: the estimator's price beside a real one (the backtest's
    "as live")."""
    if bet not in LEG_MARKETS:
        raise ValueError(bet)
    dec = pp._price(decimal) if hasattr(pp, "_price") else decimal
    if dec is None or not (isinstance(dec, (int, float)) and math.isfinite(dec)) or dec <= 1.0:
        return None
    if p is None or not math.isfinite(float(p)) or not 0.0 < float(p) < 1.0:
        return None
    return {
        "key": f"{sport}:{key}", "sport": sport, "day": day, "event": event,
        "pick": pick, "opponent": opponent, "kickoff": kickoff, "number": number,
        "side": "pick", "market": "moneyline", "bet": bet, "bet_detail": bet_detail,
        "decimal": float(dec), "p": float(p),
        "p_model": None if p_model is None else float(p_model),
        "p_market": None if p_market is None else float(p_market),
        "edge": None if (p_model is None or p_market is None) else float(p_model) - float(p_market),
        "favourite": p_market is not None and float(p_market) >= 0.5,
        "lot_key": None if lot_key is None else str(lot_key),
        "estimated": bool(estimated),
        "est_decimal": None if est_decimal is None else float(est_decimal),
        "price_source": price_source, "result": result,
    }


def method_lot_key(corners, corner_name, method_type):
    """The props lottery's key for one prop: the fight's (dog_lot_key's
    corners), the prop's corner (normalised as dog_lot_key's names) and the
    type - one string live and in the backtest.

    CLARIFICATION (2026-11-19, not a rule change - the draw measured and the
    draw live are this function's): PROPS in the module docstring names the
    key "dog_lot_key's corners plus the leg type"; the corner's own name is
    in it as well, because without it both corners' "wins by KO/TKO" props
    of one fight would hash to one string, share one u and so never be
    drawn apart by the lottery's order - only by their chances."""
    return "|".join([dog_lot_key("UFC", corners=corners), _norm_name(corner_name), str(method_type)])


def _finish(t, legs_order=None):
    t["legs"] = sorted(t["legs"], key=legs_order or (lambda l: (l.get("kickoff") or "", l["sport"], l["key"])))
    t["sports"] = sorted({l["sport"] for l in t["legs"]})
    t["bets"] = sorted({l.get("bet") or "1X2" for l in t["legs"]})
    t["day"] = t["legs"][0].get("day")
    return t


def build_likeliest_v2(legs, objective="p"):
    """LIKELIEST v2: the v1 rule over the v2 winner legs (soccer from every
    league in the app)."""
    t = build([l for l in legs if l and is_winner(l)], objective=objective)
    if t is not None:
        t["profile"] = "same_day_v2" if objective == "p" else "same_day_v2_market_only"
        t["bets"] = ["1X2"]
    return t


def build_underdog_v2(legs, dogs, salt=DOG_SALT_V2, order="lottery", objective="p"):
    """UNDERDOG v2: the v1 underdog rule with its own salt, filled from the
    v2 winner legs."""
    t = build_dog([l for l in legs if l and is_winner(l)], dogs, salt=salt, order=order, objective=objective)
    if t is not None:
        t["profile"] = "same_day_dog_v2" if objective == "p" else "same_day_dog_v2_market_only"
    return t


def build_mixed(legs, objective="p"):
    """MIXED: the highest product of chances, in BAND, 1-6 legs one per
    event, among the POOL likeliest winner legs (chance >= MIN_P) and the
    MIXED_POOL_OTHER likeliest other-market legs (each >= MIN_P, but a METHOD
    leg has no floor), holding at least one other-market leg. None when
    nothing fits."""
    legs = [l for l in legs if l and l.get(objective) is not None]
    winners = sorted([l for l in legs if is_winner(l) and l[objective] >= MIN_P], key=lambda l: -l[objective])[:POOL]
    others = sorted([l for l in legs if not is_winner(l) and (l.get("bet") == "METHOD" or l[objective] >= MIN_P)],
                    key=lambda l: -l[objective])[:MIXED_POOL_OTHER]
    picked = pp._search([(winners + others, 1, MAX_LEGS)], BAND, objective=objective, max_legs=MAX_LEGS,
                        require=lambda ls: any(not is_winner(l) for l in ls))
    if not picked:
        return None
    t = pp.ticket(picked, "same_day_mixed" if objective == "p" else "same_day_mixed_market_only", BAND)
    return _finish(t)


def _priced_as(l, price_key, chance_key=None, objective="p"):
    """The leg as the search sees it: priced at price_key when it has one
    (the estimator's price for a prop "as live"), else at its own price; and
    with chance_key, its chance (the `objective` field) read from chance_key
    when it has one (the estimator's chance for a prop "as live"), else its
    own. WHY THE CHANCE TOO: live, a prop's chance is the shipped win chance
    times the ESTIMATOR's share of that corner's wins, and that chance
    weights the lottery; a backtest that swapped only the price would draw by
    the real six-way book's split, which live never has."""
    v = l.get(price_key) if price_key != "decimal" else None
    c = l.get(chance_key) if chance_key else None
    if v is None and c is None:
        return l
    out = dict(l)
    if v is not None:
        out["decimal"] = float(v)
    if c is not None:
        out[objective] = float(c)
    return out


def build_props(legs, method_legs, salt=PROPS_SALT, band=BAND, price_key="decimal", objective="p",
                chance_key=None):
    """PROPS: exactly one METHOD leg priced within PROPS_PRICE, drawn by the
    lottery (weights its chance), plus the likeliest fill from the
    likeliest-v2 pool (winner legs, chance >= MIN_P, the POOL likeliest) on
    other events to `band`; 2-6 legs; the next draw when a prop has no fill,
    None when none fits.

    price_key "decimal": choose and pay at the legs' own prices (live, where a
    prop's own price IS the estimate and its chance the estimator's; the
    backtest's "as priced"). Another key ("est_decimal"): choose by that
    price - the band, the PROPS_PRICE window and the search read it - and pay
    at the real one (the backtest's "as live"); the ticket's multiplier is
    then the real product and may fall outside the band, which is the point
    of measuring it. chance_key ("est_p"): the lottery's weight and the
    search read that chance for a prop in place of its own (the backtest's
    "as live": the estimator's chance, as live has it); the paid prop keeps
    its real price, result and no-vig chance (p_market, the price-implied
    return's) and carries the chance it was drawn by, its own kept as
    p_book. None: the prop's own chance (live; "as priced"; and the as-live
    reading weighted by the real book's split, reported beside it)."""
    seen = {}
    for m in method_legs or []:
        if not m or m.get("bet") != "METHOD" or m.get(objective) is None:
            continue
        s = _priced_as(m, price_key, chance_key, objective)
        if chance_key and m.get(chance_key) is None:
            continue                            # no estimated chance: as live it could not be drawn
        if PROPS_PRICE[0] <= s["decimal"] <= PROPS_PRICE[1]:
            seen[id(s)] = (s, dict(m, **{objective: s[objective], "p_book": m[objective]}) if chance_key else m)
    cands = [s for s, _ in seen.values()]
    real = {id(s): m for s, m in seen.values()}
    pool_all = sorted([l for l in legs if l and is_winner(l) and l.get(objective) is not None
                       and l[objective] >= MIN_P], key=lambda l: -l[objective])
    for rank, s in enumerate(dog_order(cands, salt, "lottery", objective)):
        pool = [_priced_as(l, price_key) for l in pool_all if l["key"] != s["key"]][:POOL]
        picked = pp._search([([s], 1, 1), (pool, 1, MAX_LEGS - 1)], band, objective=objective, max_legs=MAX_LEGS)
        if not picked:
            continue
        searched = float(math.prod(l["decimal"] for l in picked))
        back = {l["key"]: l for l in pool_all}
        paid_legs = [real[id(s)] if l is s else back[l["key"]] for l in picked]
        t = pp.ticket(paid_legs, "same_day_props" if objective == "p" else "same_day_props_market_only", band)
        t = _finish(t, lambda l: (l.get("bet") != "METHOD", l.get("kickoff") or "", l["sport"], l["key"]))
        t.update({"method_key": real[id(s)]["key"], "method_lot_key": real[id(s)]["lot_key"],
                  "method_rank": rank, "candidates": len(cands), "salt": salt,
                  "price_key": price_key, "searched_multiplier": searched,
                  "estimated": any(l.get("estimated") for l in paid_legs)})
        return t
    return None
