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

Legs are dicts parlay_profiles' search and settlement read (engine/
parlay_profiles.py): key (sport-prefixed, so one leg per event holds across
sports), decimal, p, p_market, side "pick", market "moneyline" and result
{"winner": "pick" | "other" | None (void)} or None (not known yet).
"""

from __future__ import annotations

import math
import sys
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
