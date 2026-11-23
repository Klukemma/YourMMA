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

THE SPREAD AND TOTAL PRICES. The same-day parlays (parlays/engine/sameday.py,
TICKETS v2 SPREAD / TOTAL) price a spread or total leg live, so the price is
kept beside the line: each side's consensus_price - the moneyline's own
consensus - AT THE CONSENSUS LINE (offered_line, unchanged), from the books
that priced BOTH sides at that line. A book's pair counts only then: a price
at another line (a book on 3 while the consensus is 3.5) is a price for a
different bet, and one side alone cannot be de-vigged. A price always comes
from the same source as its line (the-odds-api's line with its price, the
schedule's with the schedule's), and a missing price says why
(PRICE_MISSING). The prices are read by no model and no pick.

ANYTIME-TD PRICES. The same-game tickets (nfl_sgp) price a TD leg at the
books' consensus "Yes" price for player_anytime_td. The odds endpoint above
does not carry player markets: each game is its own call to the per-event
endpoint, 1 credit a game (one market, one region), so the budget is per
game. A game is bought only once it kicks off within TD_WINDOW_HOURS (the
names a book lists settle late in the week), at most once in TD_STALE_HOURS
unless ODDS_REFRESH=1 asks for game-day prices, never once it has started,
and never when the last credits counted are below TD_RESERVE_CREDITS - the
lines above are worth more than a TD price. The event ids come from the
odds reply (kept as event_id) or, for a game cached before it was kept,
from the events list, which costs no credit.

Kept per game in the cache's "td" block: {feed name: consensus Yes price
(the median across books in probability space, as consensus_price), the
best price and its book, how many books}, and when it was seen. The feed
names a player but not his team, so a price reaches a gsis id only through
the names the game's own two rosters give him (td_prices_for): the exact
name first, then a strict normalised one (case, accents, punctuation and a
generational suffix - nothing looser, no nickname, no near spelling), and
only where that name belongs to one player of the game and one feed entry.
Anything else leaves him unpriced with the reason (TD_PRICE_MISSING): the
MMA cache's lesson (engine/odds_cache.py, for_card) - a wrong price is
worse than none, and a price is only ever the game's own.
"""

import json
import os
import re
import statistics
import unicodedata
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

EVENTS_ENDPOINT = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
EVENT_ODDS_ENDPOINT = EVENTS_ENDPOINT + "/{event_id}/odds"
TD_MARKET = "player_anytime_td"
TD_PARAMS = {"regions": "us", "markets": TD_MARKET, "oddsFormat": "american"}
TD_WINDOW_HOURS = 36     # a game's anytime-TD prices are bought only this close to kickoff
TD_STALE_HOURS = 20      # and at most once in this long, unless ODDS_REFRESH=1
TD_RESERVE_CREDITS = 30  # credits left for the lines above: no TD call below this

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
    return {"schema": SCHEMA, "games": {}, "fetches": [], "td": {}}


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


# Why a spread or total price is None in market_for's output.
PRICE_MISSING = {
    "no_line": "no line",
    "before": "fetched before prices were kept",
    "unpaired": "no book priced both sides at the line",
    "schedule_none": "the schedule posts none",
    "schedule_one": "the schedule posts one side only",
}
LINE_TOL = 1e-9


def _pair(a, b, sign):
    """(a's price, b's price, the line) when both outcomes are priced at the
    same line, else None. The line is sign x a's point - the schedule's
    convention: a spread's home handicap flipped (sign -1), a total as it is
    (sign +1) - and in both markets it is the point b quotes when b is on
    the same line (the away side's handicap; the under's total)."""
    if not a or not b or not a.get("price") or not b.get("price"):
        return None
    if a.get("point") is None or b.get("point") is None:
        return None
    line = sign * float(a["point"])
    if abs(float(b["point"]) - line) > LINE_TOL:
        return None
    return float(a["price"]), float(b["price"]), line


def _at_line(pairs, line):
    """Each side's consensus price at `line` from the books whose pair is
    at it, and how many such books: (first, second, books). (None, None, 0)
    where none is; (None, None, None) where there is no line."""
    if line is None:
        return None, None, None
    at = [(x, y) for x, y, point in pairs if abs(point - line) <= LINE_TOL]
    if not at:
        return None, None, 0
    return consensus_price([x for x, _ in at]), consensus_price([y for _, y in at]), len(at)


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
        spread_pairs, total_pairs = [], []      # (first price, second price, line) per book
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
                    pair = _pair(h, outcomes.get(ev["away_team"]), -1)
                    if pair:
                        spread_pairs.append(pair)
                elif market.get("key") == "totals":
                    over = outcomes.get("Over")
                    if over and over.get("point") is not None:
                        totals.append(float(over["point"]))
                    pair = _pair(over, outcomes.get("Under"), 1)
                    if pair:
                        total_pairs.append(pair)
        if not probs and not spreads:
            continue
        best_home = max(home_prices, key=lambda x: _payout(x[0])) if home_prices else (None, None)
        best_away = max(away_prices, key=lambda x: _payout(x[0])) if away_prices else (None, None)
        spread = offered_line(spreads) if spreads else None
        total = offered_line(totals) if totals else None
        sp_home, sp_away, sp_books = _at_line(spread_pairs, spread)
        to_over, to_under, to_books = _at_line(total_pairs, total)
        out[game_key(away, home, ev.get("commence_time"))] = {
            "home": home, "away": away,
            "commence_time": ev.get("commence_time"),
            # The per-event endpoint's id: what an anytime-TD call is made with.
            "event_id": ev.get("id"),
            "home_prob": statistics.median(probs) if probs else None,
            "home_ml": consensus_price([p for p, _ in home_prices]) if home_prices else None,
            "away_ml": consensus_price([p for p, _ in away_prices]) if away_prices else None,
            "best_home_ml": best_home[0], "best_home_book": best_home[1],
            "best_away_ml": best_away[0], "best_away_book": best_away[1],
            "spread": spread,
            "total": total,
            # American odds at the line above; None with a count of 0 where
            # no book priced both sides at it, None with None where there is
            # no line.
            "spread_price_home": sp_home, "spread_price_away": sp_away, "spread_price_books": sp_books,
            "total_price_over": to_over, "total_price_under": to_under, "total_price_books": to_books,
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


def record_fetch(cache, status, events, remaining, now=None, **extra):
    """One line of the fetch log. `extra` says what else the call was (an
    anytime-TD round: its market and how many calls it made)."""
    cache["fetches"].append(dict({"at": _stamp(now or _now()), "status": status,
                                  "events": events, "credits_remaining": remaining}, **extra))
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


def _get(url, params, api_key, log, failed):
    """One GET to the-odds-api: (status, the reply's JSON or `failed`,
    credits remaining). The key goes in the query string, so every text
    that could carry the URL is redacted before it is logged."""
    import requests
    try:
        response = requests.get(url, params=dict(params, apiKey=api_key), timeout=20)
    except requests.RequestException as err:
        log(f"   [odds] request failed: {redact(err, api_key)[:200]}")
        return None, failed, None
    remaining = response.headers.get("x-requests-remaining")
    if response.status_code != 200:
        log(f"   [odds] HTTP {response.status_code}: {redact(response.text, api_key)[:200]}")
        return response.status_code, failed, remaining
    return 200, response.json(), remaining


def fetch(api_key, log=print):
    """One call. Returns (status, events or [], credits remaining)."""
    return _get(ENDPOINT, PARAMS, api_key, log, [])


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


# ---------------------------------------------------------------- anytime TD

def _credits(remaining):
    """The credits-remaining header as a number, or None."""
    try:
        return float(remaining)
    except (TypeError, ValueError):
        return None


def _last_credits(cache):
    """The credits left as the newest fetch on file counted them, or None."""
    for f in reversed(cache.get("fetches") or []):
        left = _credits(f.get("credits_remaining"))
        if left is not None:
            return left
    return None


def parse_td_event(event):
    """(key, entry) from one per-event reply for player_anytime_td, or
    (None, None) for a team not in the table.

    A book's price counts for a player only on its "Yes" outcome with the
    player named (`description`) and a real American price; a book that
    quotes one player at two different prices has neither counted. Per
    player: `price`, the consensus across books (consensus_price - the
    median in probability space), `best_price` and `best_book`, and `books`."""
    home, away = NAME_TO_CODE.get(event.get("home_team")), NAME_TO_CODE.get(event.get("away_team"))
    if not home or not away:
        return None, None
    quotes, books = {}, 0
    for book in event.get("bookmakers") or []:
        title = book.get("title") or book.get("key")
        mine, torn = {}, set()
        for market in book.get("markets") or []:
            if market.get("key") != TD_MARKET:
                continue
            for o in market.get("outcomes") or []:
                who, price = str(o.get("description") or "").strip(), o.get("price")
                if str(o.get("name") or "").strip().lower() != "yes" or not who:
                    continue
                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue
                if abs(price) < 100:
                    continue
                if who in mine and mine[who] != price:
                    torn.add(who)
                mine[who] = price
        for who in torn:
            mine.pop(who)
        books += bool(mine)
        for who, price in mine.items():
            quotes.setdefault(who, []).append((price, title))
    players = {}
    for who, q in quotes.items():
        best = max(q, key=lambda x: _payout(x[0]))
        players[who] = {"price": consensus_price([p for p, _ in q]), "best_price": best[0],
                        "best_book": best[1], "books": len(q)}
    return game_key(away, home, event.get("commence_time")), {
        "event_id": event.get("id"), "home": home, "away": away,
        "commence_time": event.get("commence_time"), "books": books, "players": players}


def merge_td(cache, key, entry, now=None):
    """Keep one game's anytime-TD prices, replacing any older ones. A game
    that has started is skipped - its prices are in-play. True if kept."""
    now = now or _now()
    start = _parse_time(entry["commence_time"]) if entry.get("commence_time") else None
    if start is None or start <= now:
        return False
    cache.setdefault("td", {})[key] = dict(entry, seen=_stamp(now))
    return True


def td_for_game(cache, away, home, kickoff):
    """One game's anytime-TD entry, or None: the two teams and a kickoff
    within MATCH_HOURS, as for_game() - never another game's."""
    kickoff = pd.Timestamp(kickoff)
    best = None
    for entry in (cache.get("td") or {}).values():
        if entry.get("home") != home or entry.get("away") != away or not entry.get("commence_time"):
            continue
        start = pd.Timestamp(_parse_time(entry["commence_time"]).replace(tzinfo=None))
        gap = abs((start - kickoff).total_seconds()) / 3600
        if gap <= MATCH_HOURS and (best is None or gap < best[0]):
            best = (gap, entry)
    return best[1] if best else None


def td_due(cache, upcoming, refresh=False, now=None):
    """The games worth a credit for anytime-TD prices now: (away, home,
    kickoff) of each that kicks off within TD_WINDOW_HOURS and has not
    started, whose prices on file are TD_STALE_HOURS old or more (or that
    has none) - every game in the window when `refresh`."""
    now = now or _now()
    clock = pd.Timestamp(now.replace(tzinfo=None))
    horizon = clock + pd.Timedelta(hours=TD_WINDOW_HOURS)
    due = []
    for g in upcoming.itertuples(index=False):
        kickoff = pd.Timestamp(g.kickoff)
        if not clock < kickoff <= horizon:
            continue
        entry = td_for_game(cache, g.away, g.home, kickoff)
        if (refresh or entry is None
                or now - _parse_time(entry["seen"]) >= timedelta(hours=TD_STALE_HOURS)):
            due.append((g.away, g.home, kickoff))
    return due


def fetch_events(api_key, log=print):
    """The events list - ids, teams, kickoffs. Costs no credit."""
    return _get(EVENTS_ENDPOINT, {}, api_key, log, [])


def fetch_td(api_key, event_id, log=print):
    """One game's anytime-TD prices: 1 credit. (status, event or None,
    credits remaining)."""
    return _get(EVENT_ODDS_ENDPOINT.format(event_id=event_id), TD_PARAMS, api_key, log, None)


def _event_index(events):
    """{(away, home): [(kickoff, id)]} of an events list (or an odds reply)."""
    out = {}
    for ev in events or []:
        home, away = NAME_TO_CODE.get(ev.get("home_team")), NAME_TO_CODE.get(ev.get("away_team"))
        if home and away and ev.get("id") and ev.get("commence_time"):
            start = pd.Timestamp(_parse_time(ev["commence_time"]).replace(tzinfo=None))
            out.setdefault((away, home), []).append((start, ev["id"]))
    return out


def _event_id(index, away, home, kickoff):
    best = None
    for start, eid in index.get((away, home), []):
        gap = abs((start - kickoff).total_seconds()) / 3600
        if gap <= MATCH_HOURS and (best is None or gap < best[0]):
            best = (gap, eid)
    return best[1] if best else None


def event_ids(cache, due, api_key, log=print):
    """{(away, home, kickoff): event id} for the due games: from the odds
    reply on file where it kept the id, else from the free events list -
    called once, and only when a game needs it."""
    out, missing = {}, []
    for g in due:
        entry = for_game(cache, *g)
        eid = entry["latest"].get("event_id") if entry else None
        if eid:
            out[g] = eid
        else:
            missing.append(g)
    if missing:
        status, events, _ = fetch_events(api_key, log)
        index = _event_index(events if status == 200 else [])
        for g in missing:
            eid = _event_id(index, *g)
            if eid:
                out[g] = eid
    return out


def refresh_td(cache, upcoming, api_key=None, refresh_asked=None, log=print, now=None):
    """Buy anytime-TD prices for the games td_due() names, one call each,
    if a key exists. Returns what happened."""
    api_key = (api_key if api_key is not None else os.environ.get("ODDS_API_KEY", "")).strip()
    refresh_asked = os.environ.get("ODDS_REFRESH") == "1" if refresh_asked is None else refresh_asked
    now = now or _now()
    due = td_due(cache, upcoming, refresh_asked, now)
    if not due:
        log(f"   [odds] anytime TD: no game within {TD_WINDOW_HOURS} hours of kickoff needs a price; "
            "no call made")
        return "cached"
    if not api_key:
        log(f"   [odds] anytime TD: {len(due)} games due, but ODDS_API_KEY is not set: no TD prices")
        return "no key"
    ids = event_ids(cache, due, api_key, log)
    remaining, status = _last_credits(cache), None
    calls = priced = 0
    for g in due:
        if g not in ids:
            continue
        if remaining is not None and _credits(remaining) < TD_RESERVE_CREDITS:
            log(f"   [odds] anytime TD: {remaining} credits left, under the {TD_RESERVE_CREDITS} kept "
                "for the lines; no more TD calls")
            break
        status, event, left = fetch_td(api_key, ids[g], log)
        calls += 1
        remaining = left if left is not None else remaining
        if status != 200 or not event:
            continue
        key, entry = parse_td_event(event)
        if key is not None and merge_td(cache, key, entry, now):
            priced += 1
    if calls:
        record_fetch(cache, status, priced, remaining, now, market=TD_MARKET, calls=calls)
    log(f"   [odds] anytime TD: {len(due)} games due, {calls} calls ({len(due) - len(ids)} without an "
        f"event id), {priced} games priced. Credits left: {remaining}")
    return "fetched" if calls and status == 200 else ("no event id" if not calls else f"failed ({status})")


# Why a TD leg has no price (td_prices_for).
TD_PRICE_MISSING = {
    "window": f"anytime-TD prices are bought only within {TD_WINDOW_HOURS} hours of kickoff",
    "none": "no anytime-TD prices on file for this game",
    "unposted": "no book had posted anytime-TD prices when they were fetched",
    "unmatched": "no anytime-TD price under his name",
    "ambiguous": "his name matches more than one player or price, so none rather than a wrong one",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def td_name_key(name):
    """A name as td_prices_for compares it when the exact name fails:
    'D.J. Moore' and 'DJ Moore', 'Kenneth Walker III' and 'Kenneth Walker',
    'Ja'Marr Chase' and 'JaMarr Chase' alike - accents, case, punctuation
    and a generational suffix gone. Nothing looser: 'Gabe' is not 'Gabriel'."""
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    parts = re.sub(r"[^a-z ]", "", text.lower().replace("-", " ")).split()
    while len(parts) > 2 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def td_prices_for(cache, away, home, kickoff, players, now=None):
    """{gsis_id: price} for one game's listed players, from that game's own
    anytime-TD prices only (td_for_game) and only those seen before kickoff.

    `players` is [(gsis_id, [the names he goes by])] - every player listed
    for either team (the TD list's name, the roster's), so a feed name that
    could be someone else of this game is seen to be ambiguous. Each price:
    `price` (consensus American), `best_price`, `best_book`, `books`,
    `as_of`, the `feed_name` it came under and `match` (exact or
    normalised) - or `price` None and `missing`, why (TD_PRICE_MISSING).

    A player is priced from one feed entry: the one carrying exactly a name
    of his, else the one whose td_name_key() is his - and only when no other
    player of the game shares that key, and (for the normalised match) no
    other feed entry carries it. Two exact hits (his TD-list name and his
    roster name both quoted) are ambiguous too.
    """
    now = now or _now()
    kickoff = pd.Timestamp(kickoff)
    players = [(g, sorted({n for n in names if n})) for g, names in players]

    def missing(why):
        return {g: {"price": None, "missing": TD_PRICE_MISSING[why]} for g, _ in players}

    entry = td_for_game(cache, away, home, kickoff)
    if entry is not None and pd.Timestamp(_parse_time(entry["seen"]).replace(tzinfo=None)) >= kickoff:
        entry = None                                    # never an in-play price
    if entry is None:
        far = kickoff - pd.Timestamp(now.replace(tzinfo=None)) > pd.Timedelta(hours=TD_WINDOW_HOURS)
        return missing("window" if far else "none")
    feed = entry.get("players") or {}
    if not feed:
        return missing("unposted")
    owners, feed_keys = {}, {}
    for g, names in players:
        for n in names:
            owners.setdefault(td_name_key(n), set()).add(g)
    for name in feed:
        feed_keys.setdefault(td_name_key(name), set()).add(name)
    out = {}
    for g, names in players:
        exact = {n for n in names if n in feed}
        if exact:
            hits, match = exact, "exact"
        else:
            hits, match = set().union(*(feed_keys.get(td_name_key(n), set()) for n in names)), "normalised"
        if not hits:
            out[g] = {"price": None, "missing": TD_PRICE_MISSING["unmatched"]}
            continue
        name = next(iter(hits))
        key = td_name_key(name)
        if len(hits) > 1 or owners.get(key) != {g} or (match == "normalised" and len(feed_keys[key]) > 1):
            out[g] = {"price": None, "missing": TD_PRICE_MISSING["ambiguous"]}
            continue
        out[g] = dict(feed[name], as_of=entry["seen"], feed_name=name, match=match, missing=None)
    return out


# The two sides of each line's price, as market_for names them.
PRICE_SIDES = {"spread": ("home", "away"), "total": ("over", "under")}
# The schedule's columns for those prices (nflverse; nfl_data.load_games).
SCHEDULE_PRICES = {"spread": ("home_spread_odds", "away_spread_odds"), "total": ("over_odds", "under_odds")}


def _price_fields(market, first, second, books, source, missing):
    """One line's price fields: <market>_price_<side> (American odds) for
    both sides, _books (how many books priced both sides at the line; None
    where the source does not say), _source (whose price - always the
    line's own source) and _missing (why the prices are None; None when
    they are not)."""
    a, b = PRICE_SIDES[market]
    if missing is not None:
        first = second = source = None
    return {f"{market}_price_{a}": first, f"{market}_price_{b}": second, f"{market}_price_books": books,
            f"{market}_price_source": source, f"{market}_price_missing": missing}


def _api_prices(latest):
    """The spread and total prices of a cached the-odds-api entry, at its
    own lines. An entry cached before parse_events kept prices has none of
    the fields: said so, never read as "no book priced it"."""
    out = {}
    for market, (a, b) in PRICE_SIDES.items():
        first, second = latest.get(f"{market}_price_{a}"), latest.get(f"{market}_price_{b}")
        books = latest.get(f"{market}_price_books")
        if latest.get(market) is None:
            missing, books = PRICE_MISSING["no_line"], None
        elif f"{market}_price_books" not in latest:
            missing = PRICE_MISSING["before"]
        elif first is None or second is None:
            missing = PRICE_MISSING["unpaired"]
        else:
            missing = None
        out.update(_price_fields(market, first, second, books, "the-odds-api", missing))
    return out


def _schedule_prices(game, lines):
    """The spread and total prices the schedule posts beside its own lines
    (`lines`: {"spread": ..., "total": ...}, None where none). A pair counts
    only when both sides are posted."""
    out = {}
    for market, cols in SCHEDULE_PRICES.items():
        posted = [game.get(c) for c in cols]
        posted = [float(x) if x is not None and x == x else None for x in posted]
        n = sum(x is not None for x in posted)
        missing = (PRICE_MISSING["no_line"] if lines[market] is None else
                   PRICE_MISSING["schedule_none"] if n == 0 else
                   PRICE_MISSING["schedule_one"] if n == 1 else None)
        out.update(_price_fields(market, posted[0], posted[1], None, "nflverse", missing))
    return out


def with_price_fields(market):
    """A market logged before the prices were kept (a game under way or
    over shows the market its pick was made against), with the price
    fields present as missing and why - never absent, never invented. A
    market that has them, or none at all, is returned as it is. A new dict:
    the log is never changed."""
    if not market or "spread_price_missing" in market:
        return market
    out = dict(market)
    for m in PRICE_SIDES:
        out.update(_price_fields(m, None, None, None, None,
                                 PRICE_MISSING["no_line"] if market.get(m) is None else PRICE_MISSING["before"]))
    return out


def market_for(cache, game, now=None):
    """The market for one game as the app shows it: the API's latest price if
    it is fresh, else the schedule's posted line. Probabilities are the
    home side's. None where nothing is known.

    A cached price older than STALE_AFTER_DAYS loses to the schedule's line,
    which is re-synced every run: when a refresh fails (a spent quota, an
    outage) a days-old price must not feed the pick, or the ATS grade, while
    a fresher line sits on disk.

    Beside each line, its price (_price_fields): from the same source as
    the line, at that line - the API's consensus from the books that priced
    both sides at it, or the schedule's own posted pair - and, where None,
    why (PRICE_MISSING). The pick reads none of them."""
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
            **_api_prices(latest),
        }
    home_ml, away_ml = game.get("home_ml"), game.get("away_ml")
    has_ml = home_ml == home_ml and away_ml == away_ml and home_ml is not None and away_ml is not None
    spread, total = game.get("spread_line"), game.get("total_line")
    if not has_ml and not (spread == spread and spread is not None):
        return None
    lines = {"spread": spread if spread == spread else None, "total": total if total == total else None}
    return {
        "source": "nflverse", "as_of": None, "books": None,
        "home_prob": _no_vig(home_ml, away_ml) if has_ml else None,
        "home_ml": home_ml if has_ml else None, "away_ml": away_ml if has_ml else None,
        "spread": lines["spread"],
        "total": lines["total"],
        **_schedule_prices(game, lines),
    }
