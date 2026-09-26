"""Remember every price we are ever given, and stop asking for it again.

THE UNIT OF SPENDING IS A CALL, NOT A FIGHT. The Odds API returns every
upcoming MMA event in one response, so there is no way to ask for three fights
and be charged for three - and no saving to be had from asking for fewer. The
only lever is whether to call at all.

So the rule is: if every fight on the card already has a price on file, make no
request. One call covers a whole card, a card is roughly weekly, and the free
tier is 500 credits a month at one credit a call. Used this way the quota is
not a constraint; used carelessly - a call per run, several runs a day - it is
gone in a fortnight.

ODDS MOVE, AND NOT RE-ASKING MEANS ACCEPTING A STALE PRICE. That is the trade
being made deliberately rather than by accident: a line taken ten days out is
not the line at the bell, and an edge computed against it is an edge against a
price nobody can still get. Every entry therefore carries when it was first
seen and when it was last confirmed, callers can read the age, and `refresh`
overrides the rule outright for the day of the card.

A SETTLED FIGHT'S PRICE IS FINAL AND BECOMES HISTORY. Once a bout has happened
its closing line can never change, so it is flushed to data/odds.csv, where the
backtest reads it. That is the part worth having beyond saving credits: the
odds file has no prices at all for 2025, and every card watched from here fills
that gap permanently instead of being thrown away with the runner.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from roi import norm_name

ENGINE = Path(__file__).resolve().parent
CACHE = ENGINE / "data" / "odds_cache.json"
ODDS_CSV = ENGINE / "data" / "odds.csv"

SCHEMA = 1

# A price this old is reported as stale. Not enforced - the caller decides -
# but a line a fortnight out has usually moved by fight night.
STALE_AFTER_DAYS = 3


def fight_key(fighter_a, fighter_b, date):
    """Stable across corner order, so one bout is never stored twice.

    The API calls them home and away, which for MMA is arbitrary and does not
    match the dataset's red and blue. Sorting the normalised names removes the
    question entirely.
    """
    a, b = sorted((norm_name(fighter_a), norm_name(fighter_b)))
    return f"{str(date)[:10]}|{a}|{b}"


def empty():
    return {"schema": SCHEMA, "fights": {}, "fetches": []}


def load(path=CACHE):
    if not Path(path).exists():
        return empty()
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        # An unreadable cache must not silently behave like an empty one; that
        # would spend a credit and look like normal operation.
        raise ValueError(
            f"{path} has schema {data.get('schema')}, expected {SCHEMA}")
    data.setdefault("fights", {})
    data.setdefault("fetches", [])
    return data


def save(cache, path=CACHE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True,
                               allow_nan=False) + "\n")
    return path


def _best(outcomes):
    """The friendliest price on offer, and who is offering it."""
    best_price, best_book = None, None
    for book, price in outcomes:
        if price is None:
            continue
        if best_price is None or price > best_price:
            best_price, best_book = price, book
    return best_price, best_book


def parse_events(events):
    """Turn an API response into {key: entry}, keeping the best price a side.

    An event with no usable h2h market is skipped rather than stored empty: a
    fight recorded with no price would satisfy the "already have it" test and
    stop us ever asking again.
    """
    out = {}
    for event in events or []:
        # A null or non-dict entry must skip its own fight, not abort the whole
        # response: one malformed event would otherwise discard a card's worth
        # of prices that had already been paid for.
        if not isinstance(event, dict):
            continue
        home = event.get("home_team")
        away = event.get("away_team")
        when = (event.get("commence_time") or "")[:10]
        if not (home and away and when):
            continue

        offers = {home: [], away: []}
        for book in event.get("bookmakers", []):
            title = book.get("title") or book.get("key")
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    if name in offers:
                        offers[name].append((title, outcome.get("price")))

        home_price, home_book = _best(offers[home])
        away_price, away_book = _best(offers[away])
        if home_price is None or away_price is None:
            continue

        # Stored in the same sorted order fight_key uses, so a and b always
        # mean the same fighter however the API happened to label the corners.
        pairs = sorted(((norm_name(home), home, home_price, home_book),
                        (norm_name(away), away, away_price, away_book)))
        (_, name_a, odds_a, book_a), (_, name_b, odds_b, book_b) = pairs
        out[fight_key(home, away, when)] = {
            "date": when,
            "fighter_a": name_a,
            "fighter_b": name_b,
            "odds_a": odds_a,
            "odds_b": odds_b,
            "book_a": book_a,
            "book_b": book_b,
        }
    return out


def merge(cache, parsed, now=None):
    """Fold a parsed response into the cache. Returns (added, refreshed).

    A fight already on file keeps its ORIGINAL price and gains a last_seen.
    Overwriting would quietly replace the line a prediction was judged against
    with a later one, which is how a backtest ends up scoring bets at prices
    that were never available when the pick was made.
    """
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added, refreshed = 0, 0
    for key, entry in parsed.items():
        existing = cache["fights"].get(key)
        if existing is None:
            cache["fights"][key] = {**entry, "first_seen": now,
                                    "last_seen": now}
            added += 1
        else:
            existing["last_seen"] = now
            # Recorded, never applied. What the line moved to is worth knowing
            # and is not what the pick was priced at.
            if (existing.get("odds_a") != entry["odds_a"]
                    or existing.get("odds_b") != entry["odds_b"]):
                existing["latest_a"] = entry["odds_a"]
                existing["latest_b"] = entry["odds_b"]
            refreshed += 1
    return added, refreshed


def missing(fights, cache):
    """The card's fights that have no price on file.

    `fights` is [(fighter_a, fighter_b, date)].
    """
    return [f for f in fights if fight_key(*f) not in cache["fights"]]


def age_days(entry, now=None):
    """How long ago this price was first seen, in days."""
    now = now or datetime.now(timezone.utc)
    if isinstance(now, str):
        now = datetime.strptime(now, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    seen = entry.get("first_seen")
    if not seen:
        return None
    first = datetime.strptime(seen, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    return (now - first).total_seconds() / 86400.0


def decide(fights, cache, refresh=False, now=None):
    """Whether to spend a credit, and why. The whole point of the module."""
    absent = missing(fights, cache)
    if refresh:
        return True, f"refresh requested; {len(fights)} fights on the card"
    if not fights:
        return False, "no fights to price"
    if absent:
        return True, (f"{len(absent)} of {len(fights)} fights have no price "
                      f"on file")
    stale = [f for f in fights
             if (age_days(cache["fights"][fight_key(*f)], now) or 0)
             > STALE_AFTER_DAYS]
    if stale:
        return False, (f"every fight is priced, {len(stale)} from more than "
                       f"{STALE_AFTER_DAYS} days ago - use refresh on the day "
                       f"of the card if a current line matters")
    return False, f"every fight on the card is already priced ({len(fights)})"


def settled_rows(cache, before=None):
    """Cached fights that have already happened, as odds.csv rows.

    Final by definition, so they belong in the historical file where the
    backtest can reach them rather than only in a cache the backtest ignores.
    """
    before = before or datetime.now(timezone.utc)
    if isinstance(before, str):
        before = datetime.strptime(before[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc)
    rows = []
    for entry in cache["fights"].values():
        try:
            when = datetime.strptime(entry["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        # A day's grace: a card commencing today has not finished today.
        if when + timedelta(days=1) > before:
            continue
        rows.append({"date": entry["date"],
                     "fighter_a": entry["fighter_a"],
                     "fighter_b": entry["fighter_b"],
                     "odds_a": entry["odds_a"],
                     "odds_b": entry["odds_b"]})
    return sorted(rows, key=lambda r: (r["date"], r["fighter_a"]))


def record_fetch(cache, status, events, remaining, now=None):
    """Keep a log of every call, so the spend is visible rather than inferred."""
    now = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache["fetches"].append({"at": now, "status": status, "events": events,
                             "credits_remaining": remaining})
    cache["fetches"] = cache["fetches"][-200:]
    return cache


def as_current_odds(cache, existing=None):
    """The cache in the shape predict_card's matcher expects.

    Keyed by lowercased fighter name, both sides, so a fight priced on an
    earlier run is still priced on this one without another call. A live
    response already merged into the cache is served from here too, so there
    is one path rather than two that can disagree.
    """
    out = dict(existing or {})
    for entry in cache["fights"].values():
        for name, price, book, opponent in (
                (entry["fighter_a"], entry["odds_a"], entry.get("book_a"),
                 entry["fighter_b"]),
                (entry["fighter_b"], entry["odds_b"], entry.get("book_b"),
                 entry["fighter_a"])):
            if price is None:
                continue
            out[name.lower()] = {
                "name": name,
                "opponent": opponent,
                "odds": [{"book": book or "cached", "odds": price}],
                "best_odds": price,
                "event": "",
                "time": entry["date"],
                "cached_at": entry.get("first_seen"),
            }
    return out


def flush_settled(cache, csv_path=ODDS_CSV, before=None):
    """Append newly settled fights to odds.csv. Returns how many were added.

    Appending rather than rewriting, and skipping anything already present, so
    running twice cannot duplicate a row and inflate the backtest's sample.
    """
    import pandas as pd

    rows = settled_rows(cache, before)
    if not rows:
        return 0

    csv_path = Path(csv_path)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        seen = {fight_key(r.fighter_a, r.fighter_b, r.date)
                for r in existing.itertuples(index=False)}
    else:
        existing = pd.DataFrame(
            columns=["date", "fighter_a", "fighter_b", "odds_a", "odds_b"])
        seen = set()

    fresh = [r for r in rows
             if fight_key(r["fighter_a"], r["fighter_b"], r["date"]) not in seen]
    if not fresh:
        return 0

    combined = pd.concat([existing, pd.DataFrame(fresh)], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    return len(fresh)
