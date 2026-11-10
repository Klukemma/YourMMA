"""The parlay profiles over ten years - the pre-registered protocol.

THE QUESTION. The app offers three tickets a card, aimed at the payout
bands a bettor asked for, built by engine/parlay_profiles.py. Before the app
shows a record beside them, what would those tickets have done, card by
card, using only what was known before each card?

THE PROFILES - copied from engine/parlay_profiles.py, where they were FIXED
BEFORE ANY NUMBER IN THIS FILE WAS SEEN. Nothing below tunes them; a
profile that loses is reported losing. The bands changed once, as PROTOCOL
v2 (below).

  steady      ~5x     (band 4.5-6.5)  favourites only: every leg the market
                                      favourite, the model agreeing, blended
                                      probability >= 0.60
  risker      ~10x    (band 9.5-14)   one or two RISKY legs - a price of 2.0
                                      or longer that the model rates at least
                                      3 points above the market (blended >=
                                      0.35) - filled with steady legs
  risk_lover  50-100x (band 50-100)   one or two LONG legs - an underdog at 2.5
                                      or longer the model rates at least as
                                      high as the market - on top of two or
                                      more confident favourites (blended >=
                                      0.65). A long leg's reason names its
                                      likeliest finishing route (KO or
                                      submission) where the fighter's own UFC
                                      record has a finish.
  mixed       every band              the 12 likeliest legs on the card with a
                                      market price and a blended probability
                                      >= 0.30 (favourite or underdog, any
                                      profile's pool or none), at least one
                                      leg the model rates above the market
  market_only every band              the 12 legs the de-vigged market rates
                                      likeliest (market chance >= 0.30),
                                      searched by the market's chance alone -
                                      the control that says whether the model
                                      adds anything

plus risk_lover_method, the risk lover with its long legs priced as the
KO/submission prop of their likeliest route where the props were priced.
Within its pool and band every profile takes the combination with the
highest combined (blended; market_only: the market's) probability, so a
ticket usually pays near its band's floor; one leg per fight; a card with
no combination inside the band gives no ticket.

PROTOCOL v2 - THE BANDS MOVED AFTER v1 WAS SEEN. v1 searched risker in 8-14
and the risk lover in 40-120, and its tickets sat at those floors: no
"~10x" ticket reached 10x (median about 8.3x) and the "50-100x" tickets had
a median about 41x. v2 sets the bands to the payouts the bettor asked for -
risker 9.5-14, the risk lover 50-100; steady unchanged at 4.5-6.5 - from
the request, not from any result: the labels, not the outcomes, drove it.
v2 also prices each card from ONE snapshot (below), where v1 let a ticket
mix closing and opening prices that could never have been bought together.
Every profile rule, the periods and the ship rule are unchanged. The JSON
keeps v1's results, ship decision and caveats under "v1", and records the
multipliers v2's tickets actually carry ("multipliers": median, min, max per
rule and period).

THE PROTOCOL.

  rows      experiments/parlay_legs.py: every walk-forward fight from
            2016-01-01 to the end of the cache, the shipped recipe's
            model probability (each year fitted on earlier years only), the
            closing and the opening moneyline, method props, each corner's
            record strictly before the fight. Each CARD is priced from one
            snapshot - the one that prices more of its fights, ties to the
            close - and every leg on it takes that snapshot's price, so no
            ticket mixes two snapshots; a fight priced only in the other
            snapshot gives no leg and is counted. Rebuilt on every run, so
            the record and the rows it was read from cannot drift.
  cards     rows grouped by (event, date). Each card's legs come from
            parlay_profiles.fight_legs, which rebuilds the blended probability
            from the de-vigged market at the shipped weight
            (market_blend.MARKET_WEIGHT) - the number the live card shows.
  tickets   one per card per profile: steady, risker, risk_lover,
            risk_lover_method, and mixed and market_only at each of the three
            bands. Flat 1 unit, settled by the archive's result (a void leg
            pays 1.0; every leg void returns the stake).
  periods   DEV    2016-11-01 .. 2023-10-31
            WINDOW 2023-11-01 .. the last walk-forward date - the bettor's
                   "last three years". The cache ends 2026-08-08; the cards
                   after it have no walk-forward probability until the
                   runner refits the cache, and are counted, not guessed.
  record    parlay_profiles.summarise per period and profile (hit rate, the
            model's expected hit rate, the market's, mean multiplier, ROI with
            a bootstrap 95% interval over tickets = cards); the window again
            split by price snapshot (close / open), which covers every
            ticket because a card has one snapshot; and the multipliers the
            tickets carry (median, min, max), so the record says what the
            tickets paid, not only what their band promised.
  ship      PRE-REGISTERED: at each band, mixed ships in place of the named
            profile only if its ROI beats the named profile's in BOTH DEV and
            WINDOW. Otherwise the named profile ships. A profile with no
            settled tickets has no ROI to beat or to be beaten by, so the
            named profile ships. Shipping is what the app OFFERS - it is not
            a claim of profit: the market is hard to beat, the control
            (market_only) sits beside every band, and the app says so.

Outputs, beside this file: parlay_profiles.json (the contract the app
reads), parlay_tickets.csv.gz (one row per ticket, every leg and price, so
any number in the JSON can be re-derived by hand) and parlay_legs.csv.gz.

    python engine/experiments/parlay_profiles.py
"""

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import market_blend  # noqa: E402
import parlay_profiles as pp  # noqa: E402
from experiments import parlay_legs  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "parlay_profiles.json"
TICKETS = HERE / "parlay_tickets.csv.gz"
ODDS_CACHE = ENGINE / "data" / "odds_cache.json"

PROTOCOL_VERSION = pp.PROTOCOL_VERSION
PROTOCOL = ("parlay-profiles-v2: profiles fixed in engine/parlay_profiles.py before "
            "any backtest number; bands steady 4.5-6.5, risker 9.5-14, risk lover 50-100 "
            "(v1's 8-14 and 40-120 moved to the asked targets after v1 was seen - the "
            "labels, not the outcomes, drove it); each card priced from one snapshot; "
            "one ticket per card per profile, flat 1 unit; "
            "DEV 2016-11-01..2023-10-31, WINDOW 2023-11-01..end of walk-forward cache; "
            "mixed ships at a band only if its ROI beats the named profile's in both")
PERIODS = {"dev": ("2016-11-01", "2023-10-31"), "window": ("2023-11-01", None)}
BASE = ("steady", "risker", "risk_lover", "risk_lover_method")
NAMES = BASE + tuple(f"{p}@{b}" for p in ("mixed", "market_only") for b in pp.ORDER)
# ADDED AFTER THE FIRST RUN (post hoc, so not pre-registered): mixed ticket,
# else the named profile's. The first run shipped mixed at 50-100x, and mixed
# builds no ticket on a card whose top-12 legs are mostly favourites (the
# Nov 7 card): the live card falls back to the named profile there, and this
# measures that combined procedure - on the same cards, so its record is not
# independent of the decision that created it.
FALLBACK = tuple(f"mixed_else@{b}" for b in pp.ORDER)
NAMES = NAMES + FALLBACK
SNAPSHOTS = parlay_legs.SNAPSHOTS
SETTLED = ("won", "lost", "void")
# What a v1 file is lifted into the "v1" block with, for the record.
V1_FIELDS = ("generated", "protocol", "periods", "results", "by_snapshot", "cards",
             "ship", "caveats")


def spec(name):
    """NAME -> (profile, band) for parlay_profiles.build."""
    if "@" in name:
        profile, band = name.split("@")
        return profile, pp.BANDS[band]
    return name, None


# ---------------------------------------------------------------- the cards

def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def fight_from_row(r):
    """A parlay_legs row as the fight dict fight_legs reads. p_blend is left
    out on purpose: fight_legs rebuilds it from the market at the shipped
    weight, the same way the live card does."""
    winner = r["winner"]
    if winner == "unresolved":
        result = None
    else:
        result = {"winner": None if pd.isna(winner) else winner,
                  "method": None if pd.isna(r["method"]) else r["method"]}
    fight = {"key": r["fight_id"], "event": r["event"], "date": r["date"],
             "number": int(r["number"]), "red": r["red"], "blue": r["blue"],
             "p_model": float(r["p_model"]),
             "facts": {"red": json.loads(r["facts_red"]), "blue": json.loads(r["facts_blue"])},
             "result": result}
    for col in ("dec_red", "dec_blue", "ko_red", "ko_blue", "sub_red", "sub_blue",
                "decision_red", "decision_blue"):
        fight[col] = _num(r[col])
    return fight


def cards(frame):
    """[(event, date, [fight, ...], {fight_id: price_source})] in date order."""
    out = []
    for (event, date), g in frame.groupby(["event", "date"], sort=False):
        rows = g.to_dict("records")
        out.append((event, date, [fight_from_row(r) for r in rows],
                    {r["fight_id"]: (None if pd.isna(r["price_source"]) else r["price_source"])
                     for r in rows}))
    return sorted(out, key=lambda c: (c[1], c[0]))


def period_of(date, last):
    """"dev", "window" or None (before DEV, or after the cache's last date,
    where no point-in-time probability exists)."""
    for name, (lo, hi) in PERIODS.items():
        if lo <= date <= (hi or last):
            return name
    return None


def snapshot(t, sources):
    """"close" / "open" when every leg was priced from that snapshot, else
    "mixed" - such a ticket belongs to neither split."""
    s = {sources.get(l["key"]) for l in t["legs"]}
    return s.pop() if len(s) == 1 else "mixed"


def card_tickets(fights):
    """({NAME: ticket or None}, the card's legs). Every profile reads the
    same legs, so a difference between two profiles on one card is the
    rule, never the data."""
    legs = [l for f in fights for l in pp.fight_legs(f)]
    out = {}
    for name in NAMES:
        if name in FALLBACK:
            continue
        profile, band = spec(name)
        out[name] = pp.build(legs, profile, band)
    for name in FALLBACK:
        band = name.split("@")[1]
        out[name] = out[f"mixed@{band}"] or out[band]
    return out, legs


# ---------------------------------------------------------------- the record

def ship(results):
    """The pre-registered rule, band by band."""
    out = {}
    for band in pp.ORDER:
        mixed = [results[p].get(f"mixed@{band}", {}) for p in PERIODS]
        named = [results[p].get(band, {}) for p in PERIODS]
        beats = all(m.get("tickets") and n.get("tickets") and m["roi"] > n["roi"]
                    for m, n in zip(mixed, named))
        out[band] = "mixed" if beats else band
    return out


def _spread(values):
    m = sorted(float(x) for x in values)
    if not m:
        return {"tickets": 0, "median": None, "min": None, "max": None}
    return {"tickets": len(m), "median": round(float(np.median(m)), 2),
            "min": round(m[0], 2), "max": round(m[-1], 2)}


def multipliers(rows, names=NAMES):
    """{period: {NAME: {"tickets", "median", "min", "max"}}}: the multiplier
    each settled ticket carried (what it paid had every leg won), over the
    same tickets parlay_profiles.summarise counts. rows: ticket rows
    (ticket_row's columns), a list of dicts or a frame. A rule with no
    settled ticket has None, not a number."""
    frame = pd.DataFrame(rows, columns=["period", "profile", "multiplier", "status"]) \
        if not isinstance(rows, pd.DataFrame) else rows
    frame = frame[frame["status"].isin(SETTLED)]
    return {p: {name: _spread(frame.loc[(frame["period"] == p) & (frame["profile"] == name),
                                        "multiplier"])
                for name in names}
            for p in PERIODS}


def v1_block(previous, v1_tickets=None):
    """v1's published record, kept beside v2's: carried forward unchanged
    from a v2 file, or lifted from a v1 file - its protocol, results, ship
    decision and caveats as published, the bands it searched, and (when the
    v1 ticket file is handed over) the multipliers its tickets carried.
    None when `previous` is neither."""
    if not isinstance(previous, dict):
        return None
    if (previous.get("protocol_version") or 1) >= 2:
        return previous.get("v1")
    if not str(previous.get("protocol") or "").startswith("parlay-profiles-v1"):
        return None
    block = {k: previous.get(k) for k in V1_FIELDS}
    block["protocol_version"] = 1
    block["bands"] = {b: list(pp.V1_BANDS[b]) for b in pp.ORDER}
    block["window_tickets_mixed_snapshot"] = (previous.get("coverage") or {}).get(
        "window_tickets_mixed_snapshot")
    if v1_tickets is not None and len(v1_tickets):
        present = set(v1_tickets["profile"])
        block["multipliers"] = multipliers(v1_tickets, [n for n in NAMES if n in present])
    return block


def _read_v1_tickets(path=TICKETS):
    """The ticket file beside a v1 JSON, when it is still v1's (its risker
    rows searched v1's band); None otherwise."""
    try:
        t = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    lo, hi = pp.V1_BANDS["risker"]
    risker = t.loc[t["profile"] == "risker", "band"]
    return t if len(risker) and (risker == f"{lo:g}-{hi:g}").all() else None


def leg_text(l):
    pick = l["pick"] if l["market"] == "moneyline" else f"{l['pick']} ({l['market']})"
    return f"{pick}@{l['decimal']:.3f}"


def ticket_row(period, name, t, sources):
    s = pp.settle(t)
    lo, hi = t["band"]
    return {"period": period, "profile": name, "band": f"{lo:g}-{hi:g}",
            "event": t["event"], "date": t["date"], "n_legs": t["n_legs"],
            "multiplier": round(t["multiplier"], 4), "p": round(t["p"], 6),
            "p_market": None if t["p_market"] is None else round(t["p_market"], 6),
            "status": s["status"], "profit": s["profit"],
            "legs": "|".join(leg_text(l) for l in t["legs"]),
            "price_source": snapshot(t, sources)}


def run(frame):
    """Every card of DEV and WINDOW through every profile: the summaries, the
    window split by snapshot, and one row per ticket for the ticket file."""
    last = frame["date"].max()
    tickets = {p: defaultdict(list) for p in PERIODS}
    by_snap = {s: defaultdict(list) for s in SNAPSHOTS}
    rows, n_cards, priced_cards = [], Counter(), Counter()
    snaps, method_years = Counter(), Counter()
    for event, date, fights, sources in cards(frame):
        period = period_of(date, last)
        if period is None:
            continue
        n_cards[period] += 1
        built, legs = card_tickets(fights)
        if legs:
            priced_cards[period] += 1
        for name in NAMES:
            t = built[name]
            if t is None:
                continue
            tickets[period][name].append(t)
            snap = snapshot(t, sources)
            if period == "window":
                snaps[(name, snap)] += 1
                if snap in by_snap:
                    by_snap[snap][name].append(t)
            if name == "risk_lover_method":
                method_years[(period, date[:4])] += 1
            rows.append(ticket_row(period, name, t, sources))

    results = {p: {name: pp.summarise(tickets[p][name]) for name in NAMES} for p in PERIODS}
    by_snapshot = {"window": {s: {name: pp.summarise(by_snap[s][name]) for name in NAMES}
                              for s in SNAPSHOTS}}
    return {
        "results": results, "by_snapshot": by_snapshot, "rows": rows,
        "multipliers": multipliers(rows),
        "cards": dict(n_cards), "priced_cards": dict(priced_cards),
        "snapshots": snaps, "method_years": method_years, "window_end": last,
    }


# ---------------------------------------------------------------- caveats

def _cards_after(last):
    """Cards in the archive after the walk-forward cache ends."""
    a = pd.read_csv(parlay_legs.ARCHIVE, low_memory=False, usecols=["event_name", "date"])
    a = a[a["date"].astype(str).str[:10] > last]
    return int(a.groupby(["event_name", "date"]).ngroups), (str(a["date"].max())[:10] if len(a) else None)


def _cache_rows_in(frame, lo, hi, path=None):
    """Close-priced fights in [lo, hi] whose price is an odds_cache.json entry
    (best price per side across books). A cache key carries the bout's UTC
    commence date, and a US evening card starts on the next UTC day, so the
    fight's local date or one day either side counts - as parlay_legs
    matches the price files."""
    from odds_cache import fight_key  # the cache's own key, so no second spelling of it

    try:
        cached = json.loads(Path(path or ODDS_CACHE).read_text())["fights"]
    except (OSError, ValueError, KeyError):
        return None, None
    keys = set(cached)
    tol = parlay_legs.DATE_TOLERANCE_DAYS
    span = frame[(frame["date"] >= lo) & (frame["date"] <= hi) & (frame["price_source"] == "close")]

    def in_cache(r, b, d):
        day = pd.Timestamp(d).date()
        return any(fight_key(r, b, (day + timedelta(days=k)).isoformat()) in keys
                   for k in range(-tol, tol + 1))

    hits = sum(in_cache(r, b, d) for r, b, d in zip(span["red"], span["blue"], span["date"]))
    dates = sorted(v.get("date") for v in cached.values() if v.get("date"))
    return hits, (dates[0] if dates else None)


def _x(v):
    return "n/a" if v is None else f"{v:g}x"


def _spread_text(m):
    if not m or not m.get("tickets"):
        return "no ticket"
    return f"median {_x(m['median'])} ({_x(m['min'])} to {_x(m['max'])}, {m['tickets']} tickets)"


def caveats(frame, out, coverage, shipped=None, v1=None):
    lo_w = PERIODS["window"][0]
    last = out["window_end"]
    win = frame[frame["date"] >= lo_w]
    by_year = win.groupby(win["date"].str[:4])["price_source"].value_counts().unstack(fill_value=0)
    src = {y: {s: int(by_year.loc[y].get(s, 0)) for s in SNAPSHOTS} for y in by_year.index}
    no_price = win[win["price_source"].isna()].to_dict("records")
    off_w = sum(any(parlay_legs.snapshot_pair(r, s) for s in SNAPSHOTS) for r in no_price)
    unpriced_w = len(no_price) - off_w
    method_last = str(pd.read_csv(parlay_legs.METHOD_ODDS, usecols=["date"])["date"].max())
    rl = out["results"]["window"]["risk_lover"]
    rlm_years = sorted({y for (p, y), n in out["method_years"].items() if p == "window" and n})
    mixed_snap = sum(n for (name, s), n in out["snapshots"].items() if s == "mixed")
    n_after, after_last = _cards_after(last)
    cache_hits, cache_first = _cache_rows_in(frame, lo_w, last)
    tot = coverage["total"]
    c = []
    v1 = v1 or {}
    v1m = (v1.get("multipliers") or {}).get("window") or {}
    v1_floor = "; ".join(
        f"{b} {_spread_text(v1m.get(b))}" for b in ("risker", "risk_lover") if v1m.get(b))
    c.append("PROTOCOL v2 - the bands moved after v1 was seen. v1 searched risker in 8-14 and "
             "the risk lover in 40-120; the search takes the likeliest combination in a band, "
             "so v1's tickets sat at the floor"
             + (f" (window multipliers: {v1_floor})" if v1_floor else "")
             + " - below the '~10x' and '50-100x' they were labelled. v2 moved the bands to "
             "the payouts the bettor asked for (risker 9.5-14, the risk lover 50-100; steady "
             "unchanged at 4.5-6.5) AFTER v1 was seen: the labels, not the outcomes, drove "
             "it - the bands were set from the request, not from any result - and every "
             "profile rule, the periods and the ship rule are unchanged. v1's results, ship "
             "decision"
             + (f" ({', '.join(f'{b} -> {v}' for b, v in (v1.get('ship') or {}).items())})"
                if v1.get("ship") else "")
             + " and caveats are kept under \"v1\" for the record. v1 also let a ticket mix "
             "closing and opening prices"
             + (f" ({v1['window_tickets_mixed_snapshot']} window tickets)"
                if v1.get("window_tickets_mixed_snapshot") is not None else "")
             + "; v2 prices each card from one snapshot. v2 is a second look at the same "
             "cards, so its record is not independent of v1's.")
    mult = out["multipliers"]["window"]
    rules = []
    for b in pp.ORDER:
        rules.append(b)
        if shipped and shipped.get(b) == "mixed":
            rules.append(f"mixed_else@{b}")
    c.append("A band is a range, not a payout: a ticket pays its multiplier only if every "
             "leg wins, and the likeliest combination in a band sits near its floor. What "
             "v2's window tickets carried: "
             + "; ".join(f"{r} {_spread_text(mult.get(r))}" for r in rules)
             + ". Every rule and period is under \"multipliers\" (median, min, max).")
    c.append("The price snapshots differ. Window fights priced from the CLOSING line by year: "
             + ", ".join(f"{y} {v['close']}" for y, v in src.items())
             + "; from the OPENING line: " + ", ".join(f"{y} {v['open']}" for y, v in src.items())
             + f"; unpriced {unpriced_w}; priced only in the snapshot their card did not use "
             f"{off_w} (no leg). odds.csv has no 2025 prices, so 2025 is opening-only: "
             "an opening line is quoted days before the close, without fight-week news. "
             "Each card is priced from one snapshot - the one that prices more of its "
             "fights, ties to the close - because a ticket mixing the two could never have "
             f"been bought; {mixed_snap} window tickets mix them, so by_snapshot splits "
             "every window ticket.")
    c.append("Best-of-books prices overstate a multiplier. odds_cache.py keeps the best price "
             "per side across books, which no single book offered and which can even sum "
             "below 100% (devig rejects those). The live card is priced that way; of the "
             f"window's close-priced fights, {cache_hits} came from odds_cache.json (its "
             f"entries start {cache_first}), so the record is on the price files' closing "
             "lines, while the live card's multipliers can run higher than any one book "
             "would pay for the same legs.")
    c.append(f"Method props stop on {method_last}, so risk_lover_method has no 2025 or 2026 "
             "tickets (window years with a method ticket: "
             + (", ".join(rlm_years) if rlm_years else "none")
             + "). Its window record is 2023-11 to 2024 only; a prop leg's chance is the "
             "win chance times the six-way method market's share for that route, because "
             "the model has no route view here.")
    rare = []
    for name in ("risk_lover", "mixed@risk_lover", "mixed_else@risk_lover", "risk_lover_method"):
        for p in PERIODS:
            s = out["results"][p][name]
            if s.get("tickets"):
                rare.append(f"{name} {p} {s['wins']} of {s['tickets']} "
                            f"(ROI {s['roi']:+.2f}, interval {s['roi_low']:+.2f} to "
                            f"{s['roi_high']:+.2f}; one hit more or fewer moves it "
                            f"{s['mean_multiplier'] / s['tickets']:.2f})")
    if rare:
        c.append("The 50-100x band lands rarely - wins of tickets: " + "; ".join(rare)
                 + ". The ship decision at this band rests on a handful of hits. Read "
                 "these as 'no evidence either way', not as a measured edge or loss.")
    c.append("Closing prices include fight-week news (weigh-ins, late replacements, "
             "injuries); a bettor who bets earlier gets a different price, and the "
             "blend weight (market_blend.MARKET_WEIGHT = "
             f"{market_blend.MARKET_WEIGHT}) was fitted on closing lines and is applied "
             "unchanged to opening lines.")
    c.append(f"The walk-forward cache ends {last}; {n_after} cards after it (to "
             f"{after_last}) have no point-in-time model probability and are not scored. "
             "They need the runner to refit model_compare_baseline_symmetric.csv - a "
             "probability from a model fitted after the fight would be a leak.")
    c.append(f"Draws and no contests ({tot.get('excluded_fights', 0)} fights 2016-"
             f"{last[:4]}) are not in the walk-forward set, so they never become legs: a "
             "live ticket would sometimes carry a void leg (paid at 1.0) that this record "
             "never meets. The void rule is tested, not exercised.")
    c.append(f"{tot.get('unpriced', 0)} of {tot.get('fights', 0)} fights since 2016 have no "
             f"usable price and {tot.get('off_snapshot', 0)} more a price only in the snapshot "
             f"their card did not use; neither gives a leg. {tot.get('close_rows_alias_gap', 0)} closing-line "
             "rows are one fighter under two names (a married name, a family-name-first "
             "order) that only an entry in data/fighter_aliases.json can join.")
    c.append("The walk-forward models are up to a year staler than production (a December "
             "card is predicted by models fitted through the previous December), which "
             "understates a fresh model - the same for every profile and for the control.")
    c.append("Every ticket shares its card with the other profiles' tickets, so the profiles' "
             "records are not independent and a difference between two ROIs is not a "
             "measured difference unless it survives both periods. The bootstrap resamples "
             "tickets (one a card), not fighters.")
    c.append("The market is hard to beat in MMA. A combined probability is the model's "
             "blended belief; market_only sits beside every band as the control, and a "
             "profile that does not beat it has shown nothing the market did not already "
             "price. No profile here is presented as an edge.")
    return c


# ---------------------------------------------------------------- output

def _fmt(name, period, s):
    if not s.get("tickets"):
        return f"  {name:<24}{period:<8}{0:>8}   (no ticket)"
    mkt = "  n/a" if s["market_hit_rate"] is None else f"{s['market_hit_rate']:>5.1%}"
    return (f"  {name:<24}{period:<8}{s['tickets']:>8}{s['wins']:>6}{s['hit_rate']:>7.1%}"
            f"{s['expected_hit_rate']:>7.1%}{mkt:>7}{s['mean_multiplier']:>7.1f}x"
            f"{s['mean_legs']:>6.1f}{s['roi']:>+8.3f}  [{s['roi_low']:+.3f}, {s['roi_high']:+.3f}]"
            f"{s['profit']:>+9.1f}{s['biggest_win']:>+8.1f}")


def table(payload):
    head = (f"  {'profile':<24}{'period':<8}{'tickets':>8}{'wins':>6}{'hit':>7}{'model':>7}"
            f"{'market':>7}{'mult':>8}{'legs':>6}{'ROI':>8}  {'[95% interval]':<18}"
            f"{'profit':>9}{'best':>8}")
    lines = [head]
    for name in NAMES:
        for period in PERIODS:
            lines.append(_fmt(name, period, payload["results"][period][name]))
        for s in SNAPSHOTS:
            lines.append(_fmt(name, f"w:{s}", payload["by_snapshot"]["window"][s][name]))
    lines.append("  hit = tickets that won; model / market = the mean combined probability "
                 "each gave; ROI per 1-unit ticket")
    return "\n".join(lines)


def write_tickets(rows, path=TICKETS):
    cols = ["period", "profile", "band", "event", "date", "n_legs", "multiplier", "p",
            "p_market", "status", "profit", "legs", "price_source"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, compression="gzip")


def _previous(path=OUT):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def multiplier_table(mult):
    lines = [f"  {'multiplier':<24}{'period':<8}{'tickets':>8}{'median':>9}{'min':>9}{'max':>9}"]
    for name in NAMES:
        for period in PERIODS:
            m = mult[period][name]
            if not m["tickets"]:
                lines.append(f"  {name:<24}{period:<8}{0:>8}   (no ticket)")
                continue
            lines.append(f"  {name:<24}{period:<8}{m['tickets']:>8}{m['median']:>8.2f}x"
                         f"{m['min']:>8.2f}x{m['max']:>8.2f}x")
    return "\n".join(lines)


def main():
    t0 = time.time()
    # v1's record, read before this run overwrites the files it lives in
    previous = _previous()
    v1 = v1_block(previous, _read_v1_tickets() if previous and not previous.get("protocol_version")
                  else None)
    frame, coverage = parlay_legs.build()
    parlay_legs.write(frame)
    # score the rows as written, so the record is the file's
    frame = pd.read_csv(parlay_legs.OUT)
    print(parlay_legs.coverage_table(coverage))
    out = run(frame)
    last = out["window_end"]
    shipped = ship(out["results"])
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL,
        "bands": {b: list(pp.BANDS[b]) for b in pp.ORDER},
        "periods": {"dev": list(PERIODS["dev"]), "window": [PERIODS["window"][0], last]},
        "results": out["results"],
        "by_snapshot": out["by_snapshot"],
        "multipliers": out["multipliers"],
        "cards": {p: int(out["cards"].get(p, 0)) for p in PERIODS},
        "coverage": {
            "fights": coverage["total"].get("fights", 0),
            "priced_close": coverage["total"].get("priced_close", 0),
            "priced_open": coverage["total"].get("priced_open", 0),
            "unpriced": coverage["total"].get("unpriced", 0),
            "off_snapshot": coverage["total"].get("off_snapshot", 0),
            "method_priced": coverage["total"].get("method_priced", 0),
            "cards_with_a_priced_fight": {p: int(out["priced_cards"].get(p, 0)) for p in PERIODS},
            "window_tickets_mixed_snapshot": sum(n for (_, s), n in out["snapshots"].items()
                                                 if s == "mixed"),
            "legs": coverage,
        },
        "ship": shipped,
        "caveats": caveats(frame, out, coverage, shipped, v1),
        "v1": v1,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    write_tickets(out["rows"])
    print()
    print(table(payload))
    print()
    print(f"  cards: dev {payload['cards']['dev']}, window {payload['cards']['window']} "
          f"({payload['periods']['window'][0]} .. {last})")
    print("  ship (pre-registered: mixed only if its ROI beats the named profile in BOTH periods):")
    for band, choice in payload["ship"].items():
        m = [payload["results"][p][f"mixed@{band}"] for p in PERIODS]
        n = [payload["results"][p][band] for p in PERIODS]
        roi = lambda s: "n/a" if not s.get("tickets") else f"{s['roi']:+.3f}"  # noqa: E731
        print(f"    {band:<11} -> {choice:<11} (mixed {roi(m[0])} dev / {roi(m[1])} window; "
              f"{band} {roi(n[0])} dev / {roi(n[1])} window)")
    print()
    print(multiplier_table(payload["multipliers"]))
    print(f"  wrote {OUT.name}, {TICKETS.name} ({len(out['rows']):,} tickets), "
          f"{parlay_legs.OUT.name} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
