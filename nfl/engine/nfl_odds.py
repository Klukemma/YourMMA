"""NFL prices: moneyline, spread and total for the games still to come.

From the-odds-api (the provider the MMA engine already uses, key in the
ODDS_API_KEY environment variable - never in the repo, never printed), with
the nflverse schedule's own posted lines as the fallback when there is no key
or no reply. Every price says where it came from.

ONE CALL PRICES A WEEK. A call costs markets x regions credits: three
markets in one region is 3 credits for every NFL game on the board, so the
cache's job is to make the call only when it is needed - a game this week
has no price on file, the newest price is days old, or ODDS_REFRESH=1 asks
for today's lines on purpose (game day). A price a game has already started
on is an in-play line and is never stored.

THE MARKET PROBABILITY is the median, across books, of each book's
no-vig moneyline probability (its two prices scaled to sum to one). The
median, not the best price: the best price on each side comes from
different books and together they under-count the margin, which tilts the
market toward whichever side one book happens to be generous on.
"""

import json
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import nfl_data as nd

CACHE = nd.DATA / "odds_cache.json"
SCHEMA = 1
ENDPOINT = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
PARAMS = {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}
STALE_AFTER_DAYS = 3
WINDOW_DAYS = 8          # only this week's games need a price
MATCH_HOURS = 36         # a listed kickoff this close to the schedule's is the same game

NAME_TO_CODE = {name: code for code, (name, *_rest) in nd.TEAMS.items()}
NAME_TO_CODE.update({
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
    "Oakland Raiders": "LV", "San Diego Chargers": "LAC", "St. Louis Rams": "LA",
    "St Louis Rams": "LA",
})


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(text):
    return datetime.strptime(str(text)[:19].replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def redact(text, key):
    """The key out of any string that might be printed (requests puts the
    full URL, query string and all, into its exception text)."""
    return str(text).replace(key, "<redacted>") if key else str(text)


def empty():
    return {"schema": SCHEMA, "games": {}, "fetches": []}


def load(path=CACHE):
    path = Path(path)
    if not path.exists():
        return empty()
    cache = json.loads(path.read_text())
    if cache.get("schema") != SCHEMA:
        # An unreadable cache must not behave like an empty one: that would
        # spend credits and look like normal operation.
        raise ValueError(f"odds cache is schema {cache.get('schema')}, expected {SCHEMA}")
    return cache


def save(cache, path=CACHE):
    Path(path).write_text(json.dumps(cache, indent=1, sort_keys=True, allow_nan=False) + "\n")


def game_key(away, home, kickoff):
    return f"{pd.Timestamp(kickoff).strftime('%Y-%m-%d')}|{away}|{home}"


def _implied(o):
    o = float(o)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def _american(p):
    """Implied probability -> American odds (the inverse of _implied)."""
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def consensus_price(prices):
    """The middle book's price, taken in probability space. American odds
    jump from -100 to +100 at even money, so the median of the raw numbers
    across books that straddle it (-110 and +105) is a price nobody quoted
    (-2.5)."""
    return round(_american(statistics.median(_implied(o) for o in prices)), 1)


def offered_line(points):
    """A line some book actually offered: the lower-middle of the sorted
    lines. The plain median of 3 and 3.5 is 3.25, which no book hangs and
    which would turn the commonest NFL result, a push on 3, into a loss."""
    points = sorted(points)
    return points[(len(points) - 1) // 2]


def _no_vig(home_price, away_price):
    def implied(o):
        o = float(o)
        return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)
    h, a = implied(home_price), implied(away_price)
    return h / (h + a)


def parse_events(events):
    """{key: entry} from one API reply. Teams are mapped to franchise codes;
    an event naming a team not in the table is skipped and reported."""
    out, unknown = {}, set()
    for ev in events or []:
        home, away = NAME_TO_CODE.get(ev.get("home_team")), NAME_TO_CODE.get(ev.get("away_team"))
        if not home or not away:
            unknown.update(n for n in (ev.get("home_team"), ev.get("away_team")) if n not in NAME_TO_CODE)
            continue
        probs, home_prices, away_prices, spreads, totals, books = [], [], [], [], [], []
        for book in ev.get("bookmakers") or []:
            books.append(book.get("title") or book.get("key"))
            for market in book.get("markets") or []:
                outcomes = {o.get("name"): o for o in market.get("outcomes") or []}
                if market.get("key") == "h2h":
                    h, a = outcomes.get(ev["home_team"]), outcomes.get(ev["away_team"])
                    if h and a and h.get("price") and a.get("price"):
                        home_prices.append((float(h["price"]), books[-1]))
                        away_prices.append((float(a["price"]), books[-1]))
                        probs.append(_no_vig(h["price"], a["price"]))
                elif market.get("key") == "spreads":
                    h = outcomes.get(ev["home_team"])
                    if h and h.get("point") is not None:
                        # The API quotes the home side's handicap (-3.5 =
                        # home gives 3.5); the schedule's convention is the
                        # home side's expected margin, so the sign flips.
                        spreads.append(-float(h["point"]))
                elif market.get("key") == "totals":
                    over = outcomes.get("Over")
                    if over and over.get("point") is not None:
                        totals.append(float(over["point"]))
        if not probs and not spreads:
            continue
        best_home = max(home_prices, key=lambda x: _payout(x[0])) if home_prices else (None, None)
        best_away = max(away_prices, key=lambda x: _payout(x[0])) if away_prices else (None, None)
        out[game_key(away, home, ev.get("commence_time"))] = {
            "home": home, "away": away,
            "commence_time": ev.get("commence_time"),
            "home_prob": statistics.median(probs) if probs else None,
            "home_ml": consensus_price([p for p, _ in home_prices]) if home_prices else None,
            "away_ml": consensus_price([p for p, _ in away_prices]) if away_prices else None,
            "best_home_ml": best_home[0], "best_home_book": best_home[1],
            "best_away_ml": best_away[0], "best_away_book": best_away[1],
            "spread": offered_line(spreads) if spreads else None,
            "total": offered_line(totals) if totals else None,
            "books": len(books),
        }
    return out, sorted(unknown)


def _payout(odds):
    return 100.0 / -odds if odds < 0 else odds / 100.0


def merge(cache, parsed, now=None):
    """Fold a reply into the cache. Returns (added, refreshed, skipped_live).

    The first price ever seen for a game is kept as its opening line; every
    later one replaces `latest`. A game that has already kicked off is
    skipped - its line is in-play and says nothing about the pre-game market.
    """
    now = now or _now()
    added = refreshed = live = 0
    for key, entry in parsed.items():
        start = _parse_time(entry["commence_time"]) if entry.get("commence_time") else None
        if start is not None and start <= now:
            live += 1
            continue
        stamped = dict(entry, seen=_stamp(now))
        if key in cache["games"]:
            cache["games"][key]["latest"] = stamped
            refreshed += 1
        else:
            cache["games"][key] = {"opening": stamped, "latest": stamped}
            added += 1
    return added, refreshed, live


def record_fetch(cache, status, events, remaining, now=None):
    cache["fetches"].append({"at": _stamp(now or _now()), "status": status,
                             "events": events, "credits_remaining": remaining})
    cache["fetches"] = cache["fetches"][-200:]


def for_game(cache, away, home, kickoff):
    """The cached entry for one game ({opening, latest}) or None. Matched by
    the two teams and a kickoff within MATCH_HOURS - a flexed game keeps its
    price."""
    kickoff = pd.Timestamp(kickoff)
    best = None
    for entry in cache["games"].values():
        latest = entry["latest"]
        if latest["home"] != home or latest["away"] != away:
            continue
        start = pd.Timestamp(_parse_time(latest["commence_time"]).replace(tzinfo=None))
        gap = abs((start - kickoff).total_seconds()) / 3600
        if gap <= MATCH_HOURS and (best is None or gap < best[0]):
            best = (gap, entry)
    return best[1] if best else None


def needs_fetch(cache, upcoming, refresh=False, now=None):
    """Why a call is worth its credits, or None. `upcoming` is a frame of
    games not yet started (away, home, kickoff)."""
    now = now or _now()
    if refresh:
        return "refresh asked for"
    horizon = pd.Timestamp((now + timedelta(days=WINDOW_DAYS)).replace(tzinfo=None))
    soon = upcoming[upcoming["kickoff"] <= horizon]
    for g in soon.itertuples(index=False):
        entry = for_game(cache, g.away, g.home, g.kickoff)
        if entry is None:
            return f"{g.away} @ {g.home} has no price"
        seen = _parse_time(entry["latest"]["seen"])
        if now - seen > timedelta(days=STALE_AFTER_DAYS):
            return f"{g.away} @ {g.home} price is {(now - seen).days} days old"
    return None


def fetch(api_key, log=print):
    """One call. Returns (status, events or [], credits remaining)."""
    import requests
    try:
        response = requests.get(ENDPOINT, params=dict(PARAMS, apiKey=api_key), timeout=20)
    except requests.RequestException as err:
        log(f"   [odds] request failed: {redact(err, api_key)[:200]}")
        return None, [], None
    remaining = response.headers.get("x-requests-remaining")
    if response.status_code != 200:
        log(f"   [odds] HTTP {response.status_code}: {redact(response.text, api_key)[:200]}")
        return response.status_code, [], remaining
    return 200, response.json(), remaining


def refresh(cache, upcoming, api_key=None, refresh_asked=None, log=print, now=None):
    """Buy prices if they are needed and a key exists. Returns the reason a
    call was made, or why not."""
    api_key = (api_key if api_key is not None else os.environ.get("ODDS_API_KEY", "")).strip()
    refresh_asked = os.environ.get("ODDS_REFRESH") == "1" if refresh_asked is None else refresh_asked
    reason = needs_fetch(cache, upcoming, refresh_asked, now)
    if reason is None:
        log("   [odds] every game this week has a fresh price on file; no call made")
        return "cached"
    if not api_key:
        log(f"   [odds] {reason}, but ODDS_API_KEY is not set: using the schedule's posted lines")
        return "no key"
    status, events, remaining = fetch(api_key, log)
    parsed, unknown = parse_events(events)
    if unknown:
        log(f"   [odds] team names not recognised: {unknown}")
    added, refreshed, live = merge(cache, parsed, now)
    record_fetch(cache, status, len(parsed), remaining, now)
    log(f"   [odds] {reason}: {len(parsed)} games priced ({added} new, {refreshed} refreshed, "
        f"{live} in play skipped). Credits left: {remaining}")
    return "fetched" if status == 200 else f"failed ({status})"


def market_for(cache, game, now=None):
    """The market for one game as the app shows it: the API's latest price if
    it is fresh, else the schedule's posted line. Probabilities are the
    home side's. None where nothing is known.

    A cached price older than STALE_AFTER_DAYS loses to the schedule's line,
    which is re-synced every run: when a refresh fails (a spent quota, an
    outage) a days-old price must not feed the pick, or the ATS grade, while
    a fresher line sits on disk."""
    now = now or _now()
    entry = for_game(cache, game["away"], game["home"], game["kickoff"])
    if entry and now - _parse_time(entry["latest"]["seen"]) > timedelta(days=STALE_AFTER_DAYS):
        entry = None
    if entry:
        latest, opening = entry["latest"], entry["opening"]
        return {
            "source": "the-odds-api", "as_of": latest["seen"], "books": latest.get("books"),
            "home_prob": latest.get("home_prob"),
            "home_ml": latest.get("home_ml"), "away_ml": latest.get("away_ml"),
            "best_home_ml": latest.get("best_home_ml"), "best_home_book": latest.get("best_home_book"),
            "best_away_ml": latest.get("best_away_ml"), "best_away_book": latest.get("best_away_book"),
            "spread": latest.get("spread"), "total": latest.get("total"),
            "opening_spread": opening.get("spread"), "opening_total": opening.get("total"),
        }
    home_ml, away_ml = game.get("home_ml"), game.get("away_ml")
    has_ml = home_ml == home_ml and away_ml == away_ml and home_ml is not None and away_ml is not None
    spread, total = game.get("spread_line"), game.get("total_line")
    if not has_ml and not (spread == spread and spread is not None):
        return None
    return {
        "source": "nflverse", "as_of": None, "books": None,
        "home_prob": _no_vig(home_ml, away_ml) if has_ml else None,
        "home_ml": home_ml if has_ml else None, "away_ml": away_ml if has_ml else None,
        "spread": spread if spread == spread else None,
        "total": total if total == total else None,
    }
