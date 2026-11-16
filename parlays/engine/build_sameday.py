"""The live same-day tickets: built from the three apps' data, logged, graded.

Reads what each sport's runner last committed - app/data/card.json (UFC),
nfl/app/data/week.json, soccer/app/data/matches.json - and, by the rules in
sameday.py, builds TWO tickets a day: the likeliest 3-5x ticket and the
underdog ticket (v1). Writes parlays/app/data/sameday.json for the page and
appends to parlays/data/history.json.

THE CLOCK DECIDES, NOT A STATUS. An app file is rewritten only when its
sport's runner runs, so a match it lists as "upcoming" may have started; a
leg - or an underdog candidate - is used only while its scheduled start is
after `now`.

DAYS SHOWN: today, plus every day of the next WINDOW_DAYS (7) that has at
least one leg or underdog candidate (or a ticket logged and not yet
started, so a day whose games left the files is rebuilt - to a "none" row -
rather than left standing). WHY: the sports play on different days - a
Thursday may have one NFL game and no ticket while Saturday has a UFC card -
and the page should always show the next tickets, not an empty today.
Today's "none" names the next day that has a ticket of its kind.

IN PLAY: before today, the last IN_PLAY_DAYS (2) days whose newest row of
either kind is a ticket still pending and locked are shown too, in date
order, flagged past and in_play. WHY: the day rolls at midnight ET, and the
05:23 UTC run comes while a Saturday main event or a late Monday game can
still be on - and UFC results arrive later, with the dataset sync. Shown
from today only, the ticket a reader saw that evening would vanish until it
settled, reduced to a "pending" count. Those days are shown, never rebuilt:
every event on a past day has started.

TWO KINDS, EACH ON ITS OWN. history.json rows carry "kind": "likeliest" or
"underdog"; a row with no kind was written before the underdog ticket
existed and is read as "likeliest" (old rows are never rewritten). Each kind
is built, logged, locked, superseded and graded by itself (latest, locked,
update, grade_all, record all take the kind), so one kind's start never
locks the other and a "none" for one never supersedes the other.

LOCKED AT ITS FIRST START. Before a ticket's first leg starts, every run
rebuilds it at the latest prices and logs the new one (or a "none" row, with
its reason, when nothing fits any more), and the newest logged row of that
day and kind is the ticket. From its first start on it is locked: no run
replaces it with one built from the games still to come - the ticket a
reader saw in the morning is the one that is graded. Its first start is the
earlier of the one logged and, by the files as they are now
(starts_on_file), any of its legs' events: a kickoff moved earlier locks the
ticket at the new time even if no run logged it. A row is logged again when
any leg's event, side, price or start changes (or, for the underdog, the
dog's event, side or the candidate count): the side is compared because
both sides of an event share its key.

THE APP FILES ARE READ STRICTLY (_app_file): a missing file is a sport with
no events, but a file that is there and does not parse, or has lost its
list, stops the run before history.json is written - read as "no events" it
would supersede every unstarted ticket with a leg in it. history.json itself
likewise (load_history).

THE UNDERDOG CANDIDATES (sameday.py's DOG, built by sameday.dog() - the
rule is never re-implemented here) come from the same three files as the
legs, one per event, the side the no-vig market rates under 0.5 at the price
pair the leg is priced at:
  UFC     the side other than the pick, or the pick itself when it is the
          market's underdog. Priced at the card's MEDIAN pair across books
          (odds_median / odds_other_median, written by engine/predict_card)
          when it has one from at least 2 books; otherwise at the card's best
          pair, with price_source "best of books" on the leg and a note on the
          page. WHY: the best price of each side, often from two books,
          flatters an underdog's multiplier (dog prices spread wider across
          books than favourites'), and a parlay is bought at one book. A
          "median" of 1 book is the best price again (odds_cache keeps only
          the best of each side), so it does not count as typical.
          Chance: the blend's (win_prob for the pick, 1 - it for the other).
  NFL     market.home_ml / away_ml (the consensus moneyline); chance the
          blend's (pick_prob for the pick's side, 1 - it for the other).
  Soccer  only SOCCER_MEASURED leagues, as the legs; the team (never the draw)
          with the lower no-vig market chance (market.p_h / p_a), priced at
          prices.avg, its chance the shipped p_h / p_a. Level sides: none.
Each candidate's `names` lets leg_result grade it exactly as a leg: the
dog's side (UFC the corners and the dog's name, NFL whether it is home,
soccer "H" / "A").

THE UNDERDOG ROW also logs its draw - "dog": {key, pick, lot_key, rank,
candidates, salt, version} - so any later run (and a reader) can see which
underdog was drawn, from how many, and recompute the draw: lottery_u(salt,
day, lot_key) depends on nothing else. A "none" row says why: "no_dog" (no
underdog priced 2.00-4.00 with at least a 20% chance) or "no_fill" (no
favourites on other events bring any of them to 3-5x).

THE PAYLOAD (parlays/app/data/sameday.json):
  days     [{day, today, past, in_play, likeliest: T, underdog: T,
           cannot_both_win}], in date order: the days in play (past, before
           today), today, the days ahead; T is one ticket (ticket_payload):
           status none / open / locked / won / lost / void, legs (each with
           its outcome and, when void, void_reason), multiplier, p, p_market,
           missing_reason (None unless status is none) and none_reason; the
           underdog's T adds dog {pick, rank, candidates, salt, version},
           lands_1_in (1 / its market chance, rounded), only_one (a single
           candidate fit), best_of_books (its dog leg priced at the best
           price) and price_note. cannot_both_win: the dog is the other
           side of a likeliest leg (sameday.opposes).
  tickets  the likeliest T of each day from today on, today first, in the
           shape the page read before the underdog ticket (kept so an older
           cached page, which takes the first as today, still works).
  records  {likeliest, underdog}: the live record per kind; `record` is the
           likeliest's, as before.
  measured None until the backtest has run; else the likeliest's block as
           before - at the top level for an older page AND as
           measured.likeliest - plus measured.underdog, read from
           sameday_backtest.json's "dog" block (None, with
           measured.underdog_missing_reason, until that block exists and is
           this rule's: version and salt must match sameday.py's). Every
           record carries implied_roi, the price-implied return of the same
           tickets, beside the realised one (the likeliest's from the
           backtest's ticket list, implied_from_tickets).
  caveats  every caveat the backtest carries, the dog block's, and this
           builder's (prices, what the underdog ticket is not).

GRADING reads each sport's own results: UFC through engine/grade.py's
name-tolerant match on the archive (a draw or no contest is void; a bout
with no result stays open, unless engine/grade.py's own rule for parlays
says it never happened - the card has results in, it is a day past its
first bell, and neither fighter has a result near it: void as "cancelled"),
NFL the schedule's played games (a tie is void; a game with no score stays
open), soccer the prediction log's
finals - the WHOLE log, archived months included (soccer_picks), because
every run settles every day's ticket again - and its voids for good (a
match that never produced a result, or was rescheduled: soccer_results).

TICKETS v2 (sameday.py's TICKETS v2 - the rule lives there and is never
re-implemented here: every v2 leg is made by sameday.market_leg, every v2
ticket by build_likeliest_v2 / build_underdog_v2 / build_mixed /
build_props). WHY: the user asked for a builder that builds more often and
more kinds of bet; v1 left two days in three without a ticket and took
winners only. FOUR MORE KINDS a day, each built, locked at its own first
start, superseded, logged and graded on its own exactly as v1's two are:
history rows of kind "likeliest_v2", "underdog_v2", "mixed" and "props"
(the props ticket on a day with a UFC card only). v1's two kinds keep being
built beside them, unchanged (top-five soccer, winners only), so their live
records keep growing; the page shows them labelled v1.
  THE v2 POOL (live_v2) from the same three files:
  1X2     each sport's shipped winner pick as v1's leg, soccer from EVERY
          league the soccer app shows (matches.json holds only those).
  OU25    soccer over/under 2.5 at prices.avg.o25 / u25 (soccer_predict.
          prices_of), only where the match ships market_o25 - a withheld
          over 2.5 gives no leg - the side the no-vig pair favours, its
          chance that no-vig chance. A missing key reads as not posted.
  SPREAD / TOTAL  NFL at week.json's market.spread_price_home/away and
          total_price_over/under (American odds at the consensus line);
          a missing price is no leg, with nfl_odds' reason counted.
  METHOD  UFC card.json's method_legs (engine/app_export.method_legs_
          payload): each corner by KO/TKO, submission or decision, priced
          at the ESTIMATOR's price (estimated=True: no feed prices a prop),
          its measured error_pct carried to the page.
  Every leg that could not be offered is counted by its reason, per day
  and market, and the page shows the counts: a missing number is shown as
  missing, with why.
  THE PROPS TICKET is searched in PROPS_LIVE_BAND at the estimated price
  (sameday.py: the band that lands the real price in 3-5x most often), so
  its multiplier, and its record, are estimates, and the page says so.
  GRADED per market: OU25 on the soccer log final's goals, SPREAD / TOTAL
  on the schedule's margin and total points (a push is void), METHOD on
  the UFC archive's method class (engine/grade.categorise_method): won
  only when that corner wins by that type; void with no winner, or as
  "cancelled" exactly as a moneyline leg on the bout.
  MEASURED: read from sameday_backtest.json's "v2" block (v2_measured),
  defensively - until it exists every kind says it is not measured yet,
  and why, and shows no number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "parlays" / "engine"))
import sameday as sdy  # noqa: E402

OUT = ROOT / "parlays" / "app" / "data" / "sameday.json"
HISTORY = ROOT / "parlays" / "data" / "history.json"
BACKTEST = ROOT / "parlays" / "data" / "sameday_backtest.json"
BACKTEST_TICKETS = ROOT / "parlays" / "data" / "sameday_tickets.csv.gz"
UFC_CARD = ROOT / "app" / "data" / "card.json"
UFC_LOG = ROOT / "engine" / "data" / "prediction_history.json"
UFC_ARCHIVE = ROOT / "engine" / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
NFL_WEEK = ROOT / "nfl" / "app" / "data" / "week.json"
NFL_GAMES = ROOT / "nfl" / "data" / "games.csv.gz"
SOCCER_MATCHES = ROOT / "soccer" / "app" / "data" / "matches.json"
SOCCER_LOG = ROOT / "soccer" / "data" / "prediction_history.json"
SOCCER_LOG_SCHEMA = 1          # soccer_log.SCHEMA
# soccer_log's void reasons that football-data.co.uk's result may still take
# back (soccer_log.LATE on the feeds' kickoff, "void_by" not it): open, not void.
SOCCER_LATE = "logged after kickoff"
SOCCER_RECORD = "football-data.co.uk"

KINDS_V1 = ("likeliest", "underdog")
KINDS_V2 = ("likeliest_v2", "underdog_v2", "mixed", "props")
KINDS = KINDS_V1 + KINDS_V2
# The kinds whose row logs a lottery draw ("dog"), and its version.
DOG_KINDS = {"underdog": sdy.DOG_VERSION, "underdog_v2": sdy.V2_VERSION}
WINDOW_DAYS = 7
IN_PLAY_DAYS = 2
# The fewest books a UFC median pair must come from to count as the typical
# book's price (one book's "median" is the best price odds_cache kept).
MEDIAN_MIN_BOOKS = 2
BEST_OF_BOOKS = "best of books"
NFL_PRICE = "consensus moneyline"
SOCCER_PRICE = "average price"


def _iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s):
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _dec(american):
    a = _num(american)
    if a is None or a == 0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


# ---------------------------------------------------------------- live legs

def ufc_first_start(card):
    """The card's first bell: card.json's event.first_start, else the
    first_start the UFC log carries for that event date, else None."""
    ev = (card or {}).get("event") or {}
    if ev.get("first_start"):
        return _parse(ev["first_start"])
    log = _load(UFC_LOG, {}) or {}
    starts = sorted(p.get("first_start") for p in log.get("predictions", [])
                    if str(p.get("event_date", ""))[:10] == str(ev.get("date"))[:10] and p.get("first_start"))
    return _parse(starts[0]) if starts else None


def ufc_legs(card, now):
    if not card:
        return []
    ev = card.get("event") or {}
    start = ufc_first_start(card)
    if start is not None and start <= now:
        return []
    day = sdy.et_day(start) if start else str(ev.get("date"))[:10]
    if start is None and day < sdy.et_day(now):
        return []
    out = []
    for f in card.get("fights", []):
        pick = f.get("pick")
        red, blue = f.get("red"), f.get("blue")
        opp = blue if pick == red else red
        dec = _num(f.get("decimal")) or _dec(f.get("odds"))
        dec_o = _num(f.get("decimal_other")) or _dec(f.get("odds_other"))
        p_mk = None
        if dec and dec_o:
            ia, ib = 1 / dec, 1 / dec_o
            p_mk = ia / (ia + ib)
        elif _num(f.get("market_prob")) is not None:
            p_mk = _num(f.get("market_prob"))
        l = sdy.leg("UFC", f"{ev.get('date')}:{red}:{blue}", day, f"{red} vs {blue}", pick,
                    _num(f.get("win_prob")), dec, p_market=p_mk, p_model=_num(f.get("win_prob_model")),
                    opponent=opp, kickoff=_iso(start) if start else None, number=f.get("number"),
                    price_source=BEST_OF_BOOKS)
        if l:
            l["names"] = {"red": red, "blue": blue, "event_date": str(ev.get("date"))[:10]}
            out.append(l)
    return out


def nfl_legs(week, now):
    out = []
    for g in (week or {}).get("games", []):
        k = _parse(g.get("kickoff"))
        if k is None or k <= now or not g.get("pick"):
            continue
        mk = g.get("market") or {}
        home = g["pick"] == g.get("home")
        dec = _dec(mk.get("home_ml") if home else mk.get("away_ml"))
        hp = _num(g.get("market_prob"))
        mp = _num(g.get("model_prob"))
        l = sdy.leg("NFL", g["game_id"], sdy.et_day(k), f"{g.get('away')} @ {g.get('home')}", g["pick"],
                    _num(g.get("pick_prob")), dec,
                    p_market=None if hp is None else (hp if home else 1 - hp),
                    p_model=None if mp is None else (mp if home else 1 - mp),
                    opponent=g.get("away") if home else g.get("home"), kickoff=_iso(k),
                    price_source=NFL_PRICE)
        if l:
            l["names"] = {"game_id": g["game_id"], "home": home}
            out.append(l)
    return out


# The soccer legs the rule was measured on: the top five leagues
# (legs_history.py builds them from main's top-five engine). The soccer app
# now shows 37 leagues; their picks stay out of the same-day ticket until a
# backtest of the rule includes them.
SOCCER_MEASURED = ("E0", "SP1", "I1", "D1", "F1")


def soccer_legs(data, now, leagues=SOCCER_MEASURED):
    """The soccer winner legs: the shipped pick H/D/A at the average price,
    from `leagues` (v1: SOCCER_MEASURED; None: every league in the file -
    tickets v2's 1X2, which takes every league the soccer app shows)."""
    out = []
    for m in (data or {}).get("matches", []):
        k = _parse(m.get("kickoff"))
        pick = m.get("pick")
        if leagues is not None and m.get("league") not in leagues:
            continue
        if k is None or k <= now or pick not in ("H", "D", "A") or m.get("price_status") != "posted":
            continue
        avg = ((m.get("prices") or {}).get("avg") or {})
        dec = _num(avg.get(pick.lower()))
        home, away = (m.get("home") or {}).get("name"), (m.get("away") or {}).get("name")
        name = {"H": home, "D": "Draw", "A": away}[pick]
        p_mk = _num((m.get("market") or {}).get(f"p_{pick.lower()}"))
        l = sdy.leg("Soccer", m["match_id"], sdy.et_day(k), f"{home} v {away}", name,
                    _num(m.get("pick_prob")), dec, p_market=p_mk, opponent=f"{m.get('league')}",
                    kickoff=_iso(k), price_source=SOCCER_PRICE)
        if l:
            l["names"] = {"match_id": m["match_id"], "pick": pick}
            out.append(l)
    return out


def _app_file(path, list_key):
    """One sport's app file, read strictly: None when it is not there (that
    sport has nothing on file: no events), else the parsed file - and a file
    that is there but does not parse, or has lost its list (`list_key`),
    stops the run with an error.

    WHY STRICT: read as "no events", a truncated or renamed file would turn
    every unstarted ticket with a leg from that sport into another ticket or
    a "none" row with the wrong reason ("no combination fits"), and once the
    first start passes, the ticket a reader saw is gone for good. Stopping
    before history.json is written leaves the log as it was until the file
    is fixed - as _soccer_file does for the soccer log."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} does not read ({err}); nothing written") from err
    if not isinstance(data, dict) or not isinstance(data.get(list_key), list):
        raise ValueError(f"{path} has no {list_key!r} list; nothing written")
    return data


def load_app_files():
    """The three app files the tickets are built from, each read once and
    strictly (_app_file): {"card": UFC, "week": NFL, "matches": soccer}."""
    return {"card": _app_file(UFC_CARD, "fights"), "week": _app_file(NFL_WEEK, "games"),
            "matches": _app_file(SOCCER_MATCHES, "matches")}


def starts_on_file(files):
    """{event key (as leg() keys it): its start as the files have it NOW},
    started events included - the legs leave them out. WHY: a ticket locks
    at its first start, and a start can move earlier after the ticket was
    logged (a corrected soccer kickoff, an earlier UFC bout added to the
    log). locked() reads this beside the logged first_start, so a ticket
    already in play is never rebuilt because the run came after the moved
    start."""
    out = {}
    card = files.get("card")
    if card:
        ev = card.get("event") or {}
        start = ufc_first_start(card)
        if start is not None:
            for f in card.get("fights", []):
                out[f"UFC:{ev.get('date')}:{f.get('red')}:{f.get('blue')}"] = start
    for g in (files.get("week") or {}).get("games", []):
        k = _parse(g.get("kickoff"))
        if k is not None and g.get("game_id"):
            out[f"NFL:{g['game_id']}"] = k
    for m in (files.get("matches") or {}).get("matches", []):
        k = _parse(m.get("kickoff"))
        if k is not None and m.get("match_id"):
            out[f"Soccer:{m['match_id']}"] = k
    return out


def live_legs(now, files=None):
    files = load_app_files() if files is None else files
    return (ufc_legs(files.get("card"), now) + nfl_legs(files.get("week"), now)
            + soccer_legs(files.get("matches"), now))


# ---------------------------------------------------------------- live underdog candidates

def _no_vig(dec, dec_other):
    """The first side's chance with the margin taken out in proportion - the
    project's no-vig everywhere (it flatters an underdog a little: the
    caveats say so)."""
    ia, ib = 1.0 / dec, 1.0 / dec_other
    return ia / (ia + ib)


def ufc_pair(f):
    """(pick decimal, other decimal, price_source) a UFC dog is priced and
    sided at: the card's median pair when it comes from MEDIAN_MIN_BOOKS
    books or more, else its best pair ("best of books"), else None."""
    dm = _num(f.get("decimal_median")) or _dec(f.get("odds_median"))
    dom = _num(f.get("decimal_other_median")) or _dec(f.get("odds_other_median"))
    n = _num(f.get("books_median"))
    if dm and dom and dm > 1 and dom > 1 and n is not None and n >= MEDIAN_MIN_BOOKS:
        return dm, dom, f"median of {int(n)} books"
    dec = _num(f.get("decimal")) or _dec(f.get("odds"))
    dec_o = _num(f.get("decimal_other")) or _dec(f.get("odds_other"))
    if dec and dec_o and dec > 1 and dec_o > 1:
        return dec, dec_o, BEST_OF_BOOKS
    return None


def ufc_dogs(card, now):
    """The card's underdog candidates: per fight, the corner the no-vig
    market rates under 0.5 at the pair ufc_pair() gives - the other corner,
    or the pick when the blend picks the market's underdog. Same day and
    started-card rule as ufc_legs."""
    if not card:
        return []
    ev = card.get("event") or {}
    start = ufc_first_start(card)
    if start is not None and start <= now:
        return []
    day = sdy.et_day(start) if start else str(ev.get("date"))[:10]
    if start is None and day < sdy.et_day(now):
        return []
    out = []
    for f in card.get("fights", []):
        red, blue, pick = f.get("red"), f.get("blue"), f.get("pick")
        p = _num(f.get("win_prob"))
        pair = ufc_pair(f)
        if not red or not blue or pick not in (red, blue) or p is None or pair is None:
            continue
        dec, dec_o, source = pair
        other = blue if pick == red else red
        q_pick = _no_vig(dec, dec_o)
        pm = _num(f.get("win_prob_model"))
        if q_pick < 0.5:
            name, price, q, chance, model = pick, dec, q_pick, p, pm
        else:                                   # level (q = 0.5) gives no candidate in dog()
            name, price, q, chance, model = other, dec_o, 1 - q_pick, 1 - p, None if pm is None else 1 - pm
        c = sdy.dog("UFC", f"{ev.get('date')}:{red}:{blue}", day, f"{red} vs {blue}", name, chance, price, q,
                    lot_key=sdy.dog_lot_key("UFC", corners=[red, blue]), p_model=model,
                    opponent=other if name == pick else pick, kickoff=_iso(start) if start else None,
                    price_source=source, number=f.get("number"))
        if c:
            c["names"] = {"red": red, "blue": blue, "event_date": str(ev.get("date"))[:10]}
            out.append(c)
    return out


def nfl_dogs(week, now):
    """Each game's underdog at the consensus moneyline: the side whose no-vig
    chance from home_ml / away_ml is under 0.5, its chance the blend's."""
    out = []
    for g in (week or {}).get("games", []):
        k = _parse(g.get("kickoff"))
        if k is None or k <= now or not g.get("pick") or not g.get("game_id"):
            continue
        mk = g.get("market") or {}
        dh, da = _dec(mk.get("home_ml")), _dec(mk.get("away_ml"))
        pick_p = _num(g.get("pick_prob"))
        if not dh or not da or dh <= 1 or da <= 1 or pick_p is None:
            continue
        q_home = _no_vig(dh, da)
        home = q_home < 0.5                     # level: the away side at 0.5, which dog() refuses
        pick_home = g["pick"] == g.get("home")
        mp = _num(g.get("model_prob"))          # the model's HOME chance
        c = sdy.dog("NFL", g["game_id"], sdy.et_day(k), f"{g.get('away')} @ {g.get('home')}",
                    g.get("home") if home else g.get("away"),
                    pick_p if home == pick_home else 1 - pick_p, dh if home else da,
                    q_home if home else 1 - q_home, lot_key=sdy.dog_lot_key("NFL", game_id=g["game_id"]),
                    p_model=None if mp is None else (mp if home else 1 - mp),
                    opponent=g.get("away") if home else g.get("home"), kickoff=_iso(k), price_source=NFL_PRICE)
        if c:
            c["names"] = {"game_id": g["game_id"], "home": home}
            out.append(c)
    return out


def soccer_dogs(data, now, leagues=SOCCER_MEASURED):
    """Each measured-league match's underdog TEAM (never the draw): the lower
    of the no-vig market's home and away chances, at the average price, its
    chance the shipped p_h / p_a. Level home and away: none. `leagues` None:
    every league in the file (UNDERDOG v2's dogs)."""
    out = []
    for m in (data or {}).get("matches", []):
        if (leagues is not None and m.get("league") not in leagues) or not m.get("match_id"):
            continue
        k = _parse(m.get("kickoff"))
        if k is None or k <= now or m.get("price_status") != "posted":
            continue
        mk = m.get("market") or {}
        qh, qa = _num(mk.get("p_h")), _num(mk.get("p_a"))
        if qh is None or qa is None or qh == qa:
            continue
        side = "H" if qh < qa else "A"
        avg = ((m.get("prices") or {}).get("avg") or {})
        home, away = (m.get("home") or {}).get("name"), (m.get("away") or {}).get("name")
        c = sdy.dog("Soccer", m["match_id"], sdy.et_day(k), f"{home} v {away}", home if side == "H" else away,
                    _num(m.get(f"p_{side.lower()}")), _num(avg.get(side.lower())), qh if side == "H" else qa,
                    lot_key=sdy.dog_lot_key("Soccer", match_id=m["match_id"]), opponent=f"{m.get('league')}",
                    kickoff=_iso(k), price_source=SOCCER_PRICE)
        if c:
            c["names"] = {"match_id": m["match_id"], "pick": side}
            out.append(c)
    return out


def live_dogs(now, files=None):
    files = load_app_files() if files is None else files
    return (ufc_dogs(files.get("card"), now) + nfl_dogs(files.get("week"), now)
            + soccer_dogs(files.get("matches"), now))


# ---------------------------------------------------------------- tickets v2: the pool

# Why a leg of a v2 market is not offered, as the page says it (counted per
# day and market; no file path, no word the page does not define).
OU_NOT_POSTED = "no over/under 2.5 price posted"
OU_WITHHELD = "the app does not ship this league's over 2.5 from the market"
LEVEL = "level prices"
NFL_NOT_IN_FILE = "not in this week's file (written before the prices were kept)"
NFL_NO_PRICE = "no price at the line"
NFL_ZERO_SPREAD = "a zero spread (that is the moneyline's bet)"
NFL_LEVEL_NO_SIDE = "level prices and no model side"
METHOD_NOT_IN_FILE = "not estimated in this card's file (written before the method legs)"
METHOD_NOT_ESTIMATED = "not estimated"
NFL_LINE_PRICE = "consensus price at the consensus line"
METHOD_PRICE = "estimated"


def _miss(missing, day, bet, reason, n=1):
    """Count one leg of `bet` on `day` not offered, and why."""
    if missing is None or day is None:
        return
    per = missing.setdefault((day, bet), {})
    per[reason] = per.get(reason, 0) + n


def _v2_winner(l):
    """A v1 winner leg as a v2 1X2 leg (sameday.market_leg, bet "1X2"): the
    same event key, pick, chance and price, and the names it is graded by."""
    if not l:
        return None
    v = sdy.market_leg(l["sport"], l["key"].split(":", 1)[1], l["day"], l["event"], l["pick"], l["p"],
                       l["decimal"], "1X2", bet_detail="winner", p_market=l.get("p_market"),
                       p_model=l.get("p_model"), opponent=l.get("opponent"), kickoff=l.get("kickoff"),
                       price_source=l.get("price_source"), number=l.get("number"))
    if v:
        v["names"] = l.get("names")
    return v


def soccer_ou_legs(data, now, missing=None):
    """OU25: per match not yet started, the side of over/under 2.5 goals the
    no-vig pair at prices.avg.o25 / u25 favours, at that average price, its
    chance that no-vig chance - the number the app ships as market_o25
    (soccer_predict: bt.no_vig_total of the same two averages). Only where
    market_o25 is not null: a league whose over 2.5 the app withholds
    gives no leg. A price key missing from the file (written before the
    prices were kept) reads as not posted. Level prices: no leg."""
    out = []
    for m in (data or {}).get("matches", []):
        k = _parse(m.get("kickoff"))
        if k is None or k <= now or not m.get("match_id"):
            continue
        day = sdy.et_day(k)
        avg = ((m.get("prices") or {}).get("avg") or {})
        o, u = _num(avg.get("o25")), _num(avg.get("u25"))
        if o is None or u is None or o <= 1 or u <= 1:
            _miss(missing, day, "OU25", OU_NOT_POSTED)
            continue
        if _num(m.get("market_o25")) is None:
            _miss(missing, day, "OU25", OU_WITHHELD)
            continue
        q_over = _no_vig(o, u)
        if q_over == 0.5:
            _miss(missing, day, "OU25", LEVEL)
            continue
        over = q_over > 0.5
        label = f"{'Over' if over else 'Under'} 2.5 goals"
        home, away = (m.get("home") or {}).get("name"), (m.get("away") or {}).get("name")
        q = q_over if over else 1 - q_over
        l = sdy.market_leg("Soccer", m["match_id"], day, f"{home} v {away}", label, q, o if over else u,
                           "OU25", bet_detail=label, p_market=q, opponent=f"{m.get('league')}",
                           kickoff=_iso(k), price_source=SOCCER_PRICE)
        if l:
            l["names"] = {"match_id": m["match_id"], "over": over, "line": 2.5}
            out.append(l)
    return out


def _line_text(x):
    """A handicap as a bet slip prints it: -3.5, +2."""
    return f"{x:+g}"


def nfl_line_legs(week, now, missing=None):
    """SPREAD and TOTAL: per game not yet started, the side whose no-vig
    chance at the consensus spread (total) prices - market.spread_price_home
    / _away and total_price_over / _under, American odds at the consensus
    line - is above 0.5, its chance that no-vig chance (the model was
    measured not to beat the line). Level prices: the shipped model's side
    (spread_pick / total_pick), chance 0.5. A zero spread gives no leg (the
    moneyline is that bet). A price nfl_odds left missing gives no leg, its
    reason counted; a market from a file written before the prices were
    kept, likewise. market.spread is the home side's expected margin (the
    schedule's convention), so home lays -spread and away takes +spread."""
    out = []
    for g in (week or {}).get("games", []):
        k = _parse(g.get("kickoff"))
        if k is None or k <= now or not g.get("game_id"):
            continue
        day = sdy.et_day(k)
        mk = g.get("market") or {}
        home, away = g.get("home"), g.get("away")
        for bet, market, sides in (("SPREAD", "spread", ("home", "away")), ("TOTAL", "total", ("over", "under"))):
            a, b = (mk.get(f"{market}_price_{s}") for s in sides)
            line = _num(mk.get(market))
            if f"{market}_price_missing" not in mk and f"{market}_price_{sides[0]}" not in mk:
                _miss(missing, day, bet, NFL_NOT_IN_FILE)
                continue
            if mk.get(f"{market}_price_missing"):
                _miss(missing, day, bet, str(mk[f"{market}_price_missing"]))
                continue
            da, db = _dec(a), _dec(b)
            if line is None or da is None or db is None or da <= 1 or db <= 1:
                _miss(missing, day, bet, NFL_NO_PRICE)
                continue
            if bet == "SPREAD" and line == 0:
                _miss(missing, day, bet, NFL_ZERO_SPREAD)
                continue
            q_first = _no_vig(da, db)
            if q_first != 0.5:
                first, q = q_first > 0.5, max(q_first, 1 - q_first)
            else:
                shipped = g.get("spread_pick" if bet == "SPREAD" else "total_pick") or {}
                side = shipped.get("team") if bet == "SPREAD" else shipped.get("side")
                if side not in ((home, away) if bet == "SPREAD" else ("OVER", "UNDER")):
                    _miss(missing, day, bet, NFL_LEVEL_NO_SIDE)
                    continue
                first, q = side in (home, "OVER"), 0.5
            if bet == "SPREAD":
                label = f"{home} {_line_text(-line)}" if first else f"{away} {_line_text(line)}"
                names = {"game_id": g["game_id"], "home": first, "line": line}
            else:
                label = f"{'Over' if first else 'Under'} {line:g}"
                names = {"game_id": g["game_id"], "over": first, "line": line}
            l = sdy.market_leg("NFL", g["game_id"], day, f"{away} @ {home}", label, q, da if first else db, bet,
                               bet_detail=label, p_market=q, opponent=away if first else home, kickoff=_iso(k),
                               price_source=NFL_LINE_PRICE)
            if l:
                l["names"] = names
                out.append(l)
    return out


def _ufc_card_day(card, now):
    """(the card's ET day, its first bell) while it has not started, as
    ufc_legs reads it; None once it has, or with no card."""
    if not card:
        return None
    ev = card.get("event") or {}
    start = ufc_first_start(card)
    if start is not None and start <= now:
        return None
    day = sdy.et_day(start) if start else str(ev.get("date"))[:10]
    if start is None and day < sdy.et_day(now):
        return None
    return day, start


def ufc_method_legs(card, now, missing=None):
    """METHOD: every estimated method leg on the card (card.json's
    method_legs, engine/app_export.method_legs_payload) as a v2 leg on the
    fight's event key: "<fighter> by <type>", its chance the leg's chance
    (the corner's shipped win chance x the estimator's share of its wins
    that way), its price the ESTIMATOR's (estimated=True: no feed prices a
    prop live), its market chance the estimator's own, its lot key
    sameday.method_lot_key's. A leg the card did not estimate is counted
    with the card's reason; a fight with no method_legs (a card written
    before them) counts its six as not in the file."""
    cd = _ufc_card_day(card, now)
    if cd is None:
        return []
    day, start = cd
    ev = card.get("event") or {}
    out = []
    for f in card.get("fights", []):
        red, blue = f.get("red"), f.get("blue")
        if not red or not blue:
            continue
        legs = f.get("method_legs")
        if not isinstance(legs, list):
            _miss(missing, day, "METHOD", METHOD_NOT_IN_FILE, 2 * len(sdy.METHOD_TYPES))
            continue
        for m in legs:
            fighter, mtype, corner = m.get("fighter"), m.get("type"), m.get("corner")
            chance, est = _num(m.get("chance")), _num(m.get("est_decimal"))
            if (fighter not in (red, blue) or mtype not in sdy.METHOD_TYPES or corner not in ("red", "blue")
                    or chance is None or est is None):
                _miss(missing, day, "METHOD", str(m.get("missing_reason") or METHOD_NOT_ESTIMATED))
                continue
            label = f"{fighter} by {mtype}"
            l = sdy.market_leg("UFC", f"{ev.get('date')}:{red}:{blue}", day, f"{red} vs {blue}", label, chance,
                               est, "METHOD", bet_detail=mtype, p_market=_num(m.get("market_chance")),
                               opponent=blue if fighter == red else red, kickoff=_iso(start) if start else None,
                               price_source=METHOD_PRICE, number=f.get("number"),
                               lot_key=sdy.method_lot_key([red, blue], fighter, mtype), estimated=True,
                               est_decimal=est)
            if l:
                l["names"] = {"red": red, "blue": blue, "event_date": str(ev.get("date"))[:10],
                              "corner": corner, "type": mtype}
                l["error_pct"] = _num(m.get("error_pct"))
                l["moneyline_source"] = m.get("moneyline_source")
                out.append(l)
            else:
                _miss(missing, day, "METHOD", METHOD_NOT_ESTIMATED)
    return out


def live_v2(now, files=None):
    """The v2 pool from the three app files: {"legs": the 1X2 legs (soccer
    from every league) and every other market's (OU25, SPREAD, TOTAL,
    METHOD), "dogs": the underdog candidates (soccer from every league),
    "ufc_days": the card's day while it has not started, "missing":
    {(day, bet): {reason: count}} of the legs not offered}."""
    files = load_app_files() if files is None else files
    card, week, matches = files.get("card"), files.get("week"), files.get("matches")
    missing = {}
    winners = [_v2_winner(l) for l in ufc_legs(card, now) + nfl_legs(week, now)
               + soccer_legs(matches, now, leagues=None)]
    legs = ([l for l in winners if l] + soccer_ou_legs(matches, now, missing) + nfl_line_legs(week, now, missing)
            + ufc_method_legs(card, now, missing))
    dogs = ufc_dogs(card, now) + nfl_dogs(week, now) + soccer_dogs(matches, now, leagues=None)
    cd = _ufc_card_day(card, now)
    ufc = {cd[0]} if cd and (card or {}).get("fights") else set()
    return {"legs": legs, "dogs": dogs, "ufc_days": ufc, "missing": missing}


# ---------------------------------------------------------------- logging

LEG_FIELDS = ("sport", "key", "event", "pick", "opponent", "kickoff", "decimal", "p", "p_model", "p_market",
              "names", "price_source", "dog")
# A v2 leg also logs its market, whether its price is an estimate (and the
# estimate's measured error), and a prop's lot key. v1 rows keep their shape.
LEG_FIELDS_V2 = LEG_FIELDS + ("bet", "bet_detail", "estimated", "est_decimal", "error_pct", "lot_key",
                              "moneyline_source")


def kind_of(row):
    """A row's ticket kind; a row written before the underdog ticket existed
    has none and was the likeliest ticket."""
    return (row or {}).get("kind") or "likeliest"


def build_day(day, legs, dogs):
    """This run's two tickets for `day`, each (ticket or None, none_reason or
    None): the likeliest (sameday.build) and the underdog (sameday.build_dog,
    from the day's legs and candidates - the rule exactly as sameday.py has
    it). The underdog's none_reason is "no_dog" when no candidate clears the
    floor, else "no_fill"; the likeliest's is "no_fit"."""
    dl = [l for l in legs if l and l.get("day") == day]
    dd = [d for d in dogs if d and d.get("day") == day]
    t = sdy.build(dl)
    td = sdy.build_dog(dl, dd)
    fit = [d for d in dd if d.get("p") is not None and d["p"] >= sdy.DOG_FLOOR]
    return {"likeliest": (t, None if t else "no_fit"),
            "underdog": (td, None if td else ("no_fill" if fit else "no_dog")),
            "candidates": len(fit)}


def build_day_v2(day, v2):
    """This run's four v2 tickets for `day`, each (ticket or None,
    none_reason or None), by sameday's v2 builders from the day's v2 pool
    (live_v2): LIKELIEST v2 ("no_fit"), UNDERDOG v2 ("no_dog" / "no_fill"),
    MIXED ("no_other": no other-market leg clears its floor; else
    "no_fit") and PROPS - on a day with a UFC card only ("no_card"
    otherwise), searched in PROPS_LIVE_BAND at the estimated prices
    ("no_estimate": no method leg estimated at all - the reason is counted
    in `missing`; "no_prop": none priced within PROPS_PRICE; else
    "no_fill")."""
    legs = [l for l in (v2 or {}).get("legs") or [] if l and l.get("day") == day]
    dogs = [d for d in (v2 or {}).get("dogs") or [] if d and d.get("day") == day]
    props = [l for l in legs if l.get("bet") == "METHOD"]
    t1 = sdy.build_likeliest_v2(legs)
    t2 = sdy.build_underdog_v2(legs, dogs)
    fit = [d for d in dogs if d.get("p") is not None and d["p"] >= sdy.DOG_FLOOR]
    t3 = sdy.build_mixed(legs)
    others = [l for l in legs if not sdy.is_winner(l) and (l.get("bet") == "METHOD" or l["p"] >= sdy.MIN_P)]
    priced = [m for m in props if sdy.PROPS_PRICE[0] <= m["decimal"] <= sdy.PROPS_PRICE[1]]
    ufc_day = day in ((v2 or {}).get("ufc_days") or ())
    t4 = sdy.build_props(legs, props, band=sdy.PROPS_LIVE_BAND) if ufc_day else None
    missing = {bet: dict(per) for (d, bet), per in ((v2 or {}).get("missing") or {}).items() if d == day}
    return {"likeliest_v2": (t1, None if t1 else "no_fit"),
            "underdog_v2": (t2, None if t2 else ("no_fill" if fit else "no_dog")),
            "mixed": (t3, None if t3 else ("no_fit" if others else "no_other")),
            "props": (t4, None if t4 else ("no_card" if not ufc_day else "no_fill" if priced
                                           else "no_prop" if props else "no_estimate")),
            "candidates_v2": len(fit), "props_candidates": len(priced), "ufc_day": ufc_day,
            "missing": missing}


def ticket_row(day, t, now, kind="likeliest", none_reason=None):
    v2 = kind in KINDS_V2
    if t is None:
        row = {"day": day, "kind": kind, "built_at": _iso(now), "status": "none", "multiplier": None,
               "legs": [], "none_reason": none_reason}
        if v2:
            row["version"] = sdy.V2_VERSION
        return row
    starts = [l["kickoff"] for l in t["legs"] if l.get("kickoff")]
    row = {"day": day, "kind": kind, "built_at": _iso(now), "status": "pending",
           "multiplier": round(t["multiplier"], 4), "p": round(t["p"], 6),
           "p_market": None if t["p_market"] is None else round(t["p_market"], 6),
           "first_start": min(starts) if starts else None, "last_start": max(starts) if starts else None,
           "sports": t["sports"],
           "legs": [{k: l.get(k) for k in (LEG_FIELDS_V2 if v2 else LEG_FIELDS)} for l in t["legs"]]}
    if kind in DOG_KINDS:
        d = next(l for l in t["legs"] if l.get("dog"))
        row["dog"] = {"key": d["key"], "pick": d["pick"], "lot_key": d.get("lot_key"),
                      "rank": t.get("dog_rank"), "candidates": t.get("candidates"), "salt": t.get("salt"),
                      "version": DOG_KINDS[kind]}
    if v2:
        for l in row["legs"]:
            l["bet"] = l.get("bet") or "1X2"           # an underdog candidate is a winner leg
            l["estimated"] = bool(l.get("estimated"))
        row["version"] = sdy.V2_VERSION
        row["bets"] = sorted({l["bet"] for l in row["legs"]})
        row["estimated"] = any(l["estimated"] for l in row["legs"])
    if kind == "props":
        m = next(l for l in t["legs"] if l["key"] == t.get("method_key") and l.get("bet") == "METHOD")
        row["props"] = {"key": m["key"], "pick": m["pick"], "lot_key": t.get("method_lot_key"),
                        "rank": t.get("method_rank"), "candidates": t.get("candidates"), "salt": t.get("salt"),
                        "version": sdy.V2_VERSION, "band": list(sdy.PROPS_LIVE_BAND),
                        "price_key": t.get("price_key")}
    return row


def latest(history, day, kind="likeliest"):
    rows = [r for r in history.get("tickets", []) if r.get("day") == day and kind_of(r) == kind]
    return rows[-1] if rows else None


def locked(row, now, starts=None):
    """Whether a logged ticket is in play: its logged first_start has
    passed, or - by the files as they are now (starts_on_file) - any of its
    legs' events has started. The second is for a start moved earlier after
    the row was logged: the row's first_start is then later than the real
    one, and reading it alone would rebuild a ticket already under way. A
    "none" row is never locked (there is nothing to keep)."""
    if row is None or row.get("status") == "none":
        return False
    start = _parse(row.get("first_start"))
    if start is not None and start <= now:
        return True
    starts = starts or {}
    return any(starts.get(l.get("key")) is not None and starts[l.get("key")] <= now
               for l in row.get("legs") or [])


def _leg_sig(l):
    """What a logged leg commits to: its event, its SIDE, its price and its
    start. WHY THE SIDE: the event key is the same for both sides, so a pick
    (or the market's underdog) that flips to the other side at an equal
    price would read as unchanged and the old side would be shown and
    graded. WHY THE START: the lock reads the logged first_start, so a
    kickoff moved earlier must be logged, or the ticket would stay open
    past its real first start."""
    return (l.get("key"), l.get("pick"), l.get("decimal"), l.get("kickoff"), l.get("bet") or "1X2")


def _same(prev, row):
    """Whether a rebuilt row says what the newest logged one does: the same
    status, legs (event, side, price, start) - and for the underdog the same
    draw (the dog's event and side, and the candidate count) - so an
    unchanged ticket is not logged again."""
    if prev is None or prev.get("status") != row["status"]:
        return False
    if [_leg_sig(l) for l in prev.get("legs", [])] != [_leg_sig(l) for l in row["legs"]]:
        return False
    if row["status"] == "none":
        return prev.get("none_reason") == row.get("none_reason")
    if row.get("kind") in DOG_KINDS or row.get("kind") == "props":
        draw = "props" if row.get("kind") == "props" else "dog"
        a, b = prev.get(draw) or {}, row.get(draw) or {}
        return ((a.get("key"), a.get("pick"), a.get("candidates"))
                == (b.get("key"), b.get("pick"), b.get("candidates")))
    return True


def in_play_days(history, now, starts=None):
    """The last IN_PLAY_DAYS days before today whose newest row of either
    kind is a ticket still pending (not settled by the last grading) and
    locked - a ticket in play, or played and not graded yet - in order."""
    t0 = datetime.strptime(sdy.et_day(now), "%Y-%m-%d")
    out = []
    for back in range(IN_PLAY_DAYS, 0, -1):
        d = (t0 - timedelta(days=back)).strftime("%Y-%m-%d")
        rows = [latest(history, d, k) for k in KINDS]
        if any(r is not None and r.get("status") == "pending" and locked(r, now, starts) for r in rows):
            out.append(d)
    return out


def days_shown(history, now, legs, dogs=(), starts=None):
    """The days in play before today (in_play_days), today, then every day
    of the next WINDOW_DAYS with a leg or a candidate - or with a ticket
    logged and not started, so it is rebuilt (to "none" when its games have
    left the files) - in order."""
    today = sdy.et_day(now)
    end = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    ahead = {x["day"] for x in list(legs) + list(dogs) if x and x.get("day") and today < x["day"] <= end}
    ahead |= {r["day"] for r in history.get("tickets", [])
              if r.get("day") and today < r["day"] <= end and r.get("status") != "none" and not locked(r, now, starts)}
    return in_play_days(history, now, starts) + [today] + sorted(ahead)


def update(history, now, legs, dogs=(), why=None, starts=None, v2=None):
    """Build and log every kind of ticket for every day shown; returns the
    days. A kind whose newest row is locked is left alone; a kind with no row
    and nothing built logs nothing (there was never a ticket to supersede).
    `why`, when given a dict, gets {(day, kind): none_reason} and
    {(day, "candidates"): n} for this run, so the page can say why a day
    without a logged row has no ticket. A day before today (in play) is
    shown, never built: its events have all started, so a rebuild could only
    log a wrong "none" over what was there.

    v2: the v2 pool (live_v2). Without it only v1's two kinds are built -
    and no v2 row is touched, so a run that could not read the pool never
    supersedes a v2 ticket. With it, the four v2 kinds too (build_day_v2),
    and `why` also gets the day's "candidates_v2", "props_candidates",
    "ufc_day" and "missing" (the legs not offered, by market and reason)."""
    v2legs, v2dogs = list((v2 or {}).get("legs") or []), list((v2 or {}).get("dogs") or [])
    show = days_shown(history, now, list(legs) + v2legs, list(dogs) + v2dogs, starts)
    today = sdy.et_day(now)
    for day in show:
        if day < today:
            continue
        built = build_day(day, legs, dogs)
        if v2 is not None:
            built.update(build_day_v2(day, v2))
        if why is not None:
            why[(day, "candidates")] = built["candidates"]
            for k in ("candidates_v2", "props_candidates", "ufc_day", "missing"):
                if k in built:
                    why[(day, k)] = built[k]
        for kind in KINDS:
            if kind not in built:
                continue
            t, reason_ = built[kind]
            if why is not None:
                why[(day, kind)] = reason_
            prev = latest(history, day, kind)
            if locked(prev, now, starts):
                continue
            if prev is None and t is None:
                continue
            row = ticket_row(day, t, now, kind, reason_)
            if not _same(prev, row):
                history.setdefault("tickets", []).append(row)
    return show


# ---------------------------------------------------------------- grading

def _results():
    """The three sports' results, loaded once: "ufc" (grade.py and its
    archive index), "nfl" {game_id: home margin}, "nfl_total" {game_id:
    total points} (SPREAD / TOTAL legs), "soccer" {match_id: H / D / A /
    void} and "soccer_goals" {match_id: (home, away) / "void" / None} (OU25
    legs: None when the final carries no goals)."""
    sys.path.insert(0, str(ROOT / "engine"))
    import grade
    import pandas as pd
    out = {"ufc": None, "nfl": {}, "nfl_total": {}, "soccer": {}, "soccer_goals": {}}
    try:
        out["ufc"] = (grade, grade.load_results(UFC_ARCHIVE))
    except Exception:
        out["ufc"] = None
    try:
        g = pd.read_csv(NFL_GAMES)
        for r in g.itertuples():
            hs, as_ = getattr(r, "home_score", None), getattr(r, "away_score", None)
            if hs is not None and as_ is not None and not (pd.isna(hs) or pd.isna(as_)):
                out["nfl"][r.game_id] = float(hs) - float(as_)
                out["nfl_total"][r.game_id] = float(hs) + float(as_)
    except Exception:
        pass
    full = soccer_results(with_goals=True)
    out["soccer"] = {k: v if v == "void" else v["result"] for k, v in full.items()}
    out["soccer_goals"] = {k: v if v == "void" else v["goals"] for k, v in full.items()}
    return out


def _soccer_file(path):
    """One soccer log file's picks. A file that is there but does not read
    as the log stops the run: read as empty, it would turn every ticket
    with a leg in it back to pending and write that to history.json."""
    try:
        log = json.loads(Path(path).read_text())
    except (OSError, ValueError) as err:
        raise ValueError(f"soccer log {path} does not read ({err})") from err
    if log.get("schema") != SOCCER_LOG_SCHEMA:
        raise ValueError(f"soccer log {path} is schema {log.get('schema')}, expected {SOCCER_LOG_SCHEMA}")
    return log.get("picks") or {}


def soccer_picks(path=None):
    """Every pick of the soccer log, as soccer_log.load reads it: the main
    file and every archived month (prediction_history/YYYY-MM.json beside
    it), a pick in both being the main file's.

    WHY THE ARCHIVE: soccer_log.save moves a pick settled for good to its
    kickoff's month once that kickoff is ARCHIVE_DAYS (28) before the newest
    pick's - about three to four weeks after the match - and grade_all
    settles every day's ticket again on every run. Read from the main file
    alone, a won or lost ticket went back to pending, paid and profit None,
    once its soccer leg aged out, and left the record.

    WHY NOT IMPORT soccer_log: its modules and the UFC engine's (put on the
    path by _results for grade.py) both have a package named experiments,
    and the parlays runner installs only the UFC engine's requirements. The
    test writes a split log with soccer_log.save itself and reads it here,
    so a change to the layout fails it. No main file (no soccer yet): no
    picks."""
    path = Path(SOCCER_LOG if path is None else path)
    picks = _soccer_file(path) if path.exists() else {}
    folder = path.with_suffix("")
    old = {}
    if folder.is_dir():
        for f in sorted(folder.glob("*.json")):
            old.update(_soccer_file(f))
    return dict(old, **picks)


def _void_for_good(p):
    """Whether a soccer pick is void for a reason no later run takes back
    (soccer_log.settled_for_good): every void but "logged after kickoff" on
    the feeds' time, which football-data.co.uk's kickoff may restore - that
    leg stays open until it does or confirms it."""
    v = p.get("void")
    return bool(v) and not (v == SOCCER_LATE and p.get("void_by") not in (None, SOCCER_RECORD))


def _goals(final):
    """(home goals, away goals) from a soccer log final, or None."""
    h, a = _num((final or {}).get("home")), _num((final or {}).get("away"))
    return None if h is None or a is None else (h, a)


def soccer_results(path=None, with_goals=False):
    """{match_id: "H" / "D" / "A" / "void"} for every soccer pick with a
    result or void for good; a match with neither is left out (its leg
    stays open). with_goals: each result as {"result": H / D / A, "goals":
    (home, away) or None} instead (a void stays "void") - the over/under
    2.5 leg (tickets v2's OU25) is graded on the final's goals, which the
    same final carries beside its result.

    VOID is read off soccer_log's own key, "void" (it never writes a
    "status"): "no result" (no result STALE_DAYS after kickoff: postponed
    and never played), "rescheduled" (off, back under a new id),
    "duplicate", and "logged after kickoff" on football-data.co.uk's time -
    a leg on a match that started before it was listed to is void too, as a
    bet taken after kickoff is, never a win or a loss. Before this a
    postponed match's leg kept its ticket pending for good.

    A MOVED MATCH (soccer_log: the same match under an id that changed,
    "id_was" the first one) is graded under the id it moved to, so a leg
    keyed by the old id reads its result there; a pick still under the old
    id wins."""
    picks = soccer_picks(path)
    out = {}
    for mid, p in picks.items():
        fin = p.get("final") or {}
        if fin.get("result") in ("H", "D", "A") and not p.get("void"):
            r = {"result": fin["result"], "goals": _goals(fin)} if with_goals else fin["result"]
        elif _void_for_good(p):
            r = "void"
        else:
            continue
        out[mid] = r
        was = p.get("id_was")
        if was and was != mid and was not in picks:
            out.setdefault(was, r)
    return out


def ufc_cancel_index(history, ufc, log=None):
    """What engine/grade.py's rule for a bout that never happened needs:
    {"ufc_cards": {card date: result dates its bouts matched}, "ufc_fighters":
    {name: dates}} - grade._logged_cards over every bout the UFC log holds
    and every UFC leg logged here, and grade._fighter_dates. The rule is
    grade_parlays', reused, never re-written here."""
    grade, index = ufc
    if log is None:
        log = _load(UFC_LOG, {}) or {}
    legs = [{"event_date": (l.get("names") or {}).get("event_date"),
             "legs": [{"red": (l.get("names") or {}).get("red"), "blue": (l.get("names") or {}).get("blue")}]}
            for r in history.get("tickets", []) for l in r.get("legs") or [] if l.get("sport") == "UFC"]
    return {"ufc_cards": grade._logged_cards({"predictions": log.get("predictions") or [], "parlays": legs}, index),
            "ufc_fighters": grade._fighter_dates(index)}


def _line_result(diff, pick_first):
    """A spread or total leg's result from the margin over the line (home
    margin less the line; total points less the line): a push (0) is void,
    as books return it."""
    if diff == 0:
        return {"winner": None, "void_reason": "push"}
    return {"winner": "pick" if (diff > 0) == bool(pick_first) else "other"}


def _ufc_bout(l, res, now):
    """(the archive's (date, winner, method) for the leg's bout, None), or
    (None, the leg's result when the bout is not there): void as
    "cancelled" by engine/grade.py's rule (leg_result), else None - not
    known yet. Shared by the moneyline and the method legs, so a cancelled
    bout voids both alike."""
    if res.get("ufc") is None:
        return None, None
    grade, index = res["ufc"]
    n = l.get("names") or {}
    found = grade.find_result({"red_corner": n.get("red"), "blue_corner": n.get("blue"),
                               "event_date": n.get("event_date")}, index)
    if found is not None:
        return found, None
    cards, fighters, day = res.get("ufc_cards"), res.get("ufc_fighters"), n.get("event_date")
    if now is None or cards is None or fighters is None or not day:
        return None, None
    card = {"first_start": l.get("kickoff"), "event_date": day}
    if (day in cards and grade.card_is_complete(card, now)
            and grade._cancelled({"red": n.get("red"), "blue": n.get("blue")}, day, fighters)):
        return None, {"winner": None, "void_reason": "cancelled"}
    return None, None


def market_result(l, res, now=None):
    """A v2 other-market leg's result (OU25, SPREAD, TOTAL, METHOD), in
    leg_result's contract.
      OU25    the soccer log final's goals: over wins above 2.5 (no push at
              2.5); a void match is void; a final without goals stays open.
      SPREAD  the schedule's home margin against the line (market.spread,
              the home side's expected margin): home covers above it, away
              below, a push is void.
      TOTAL   the schedule's total points against the line; a push is void.
      METHOD  the UFC archive's row for the bout: void with no winner (draw,
              no contest), "cancelled" as a moneyline leg on a bout that
              never happened; lost when the other corner won; else won only
              when engine/grade.categorise_method reads the archive's method
              as this leg's type (a DQ or an overturned result reads as none
              of the three: lost). A winner whose method is not recorded yet
              stays open."""
    n = l.get("names") or {}
    bet = l.get("bet")
    if bet == "OU25":
        g = (res.get("soccer_goals") or {}).get(n.get("match_id"))
        if g is None:
            return None
        if g == "void":
            return {"winner": None}
        return {"winner": "pick" if (sum(g) > float(n.get("line", 2.5))) == bool(n.get("over")) else "other"}
    if bet in ("SPREAD", "TOTAL"):
        src = res.get("nfl") if bet == "SPREAD" else res.get("nfl_total")
        x, line = (src or {}).get(n.get("game_id")), _num(n.get("line"))
        if x is None or line is None:
            return None
        return _line_result(x - line, n.get("home") if bet == "SPREAD" else n.get("over"))
    if bet == "METHOD":
        found, r = _ufc_bout(l, res, now)
        if found is None:
            return r
        grade = res["ufc"][0]
        _, winner, method = found
        if not isinstance(winner, str) or not winner.strip():
            return {"winner": None, "void_reason": "draw or no contest"}
        corner = grade._corner(winner, n.get("red"), n.get("blue"))
        if corner is None:
            return None
        if corner != n.get("corner"):
            return {"winner": "other"}
        if not isinstance(method, str) or not method.strip():
            return None
        return {"winner": "pick" if grade.categorise_method(method) == n.get("type") else "other"}
    raise ValueError(bet)


def leg_result(l, res, now=None):
    """A leg's result as parlay_profiles settles it: {"winner": "pick" /
    "other"}, {"winner": None, "void_reason": ...} (void: books return the
    leg at 1.0), or None (not known yet - the leg stays open). A v2 leg of
    another market (bet OU25 / SPREAD / TOTAL / METHOD) is graded by
    market_result; a winner leg (no bet, or "1X2") as below.

    A UFC BOUT THAT NEVER HAPPENED (called off on fight night, after the
    ticket locked) has no archive row, so "not found" alone would keep its
    ticket open for good - and a ticket with an open leg still loses the
    moment another leg loses, so the record would keep the losses and never
    the wins. It is void as "cancelled" exactly when engine/grade.py's
    grade_parlays voids it: the card has results in (res["ufc_cards"]), the
    card is old enough to be complete at `now` (grade.card_is_complete on
    the leg's first bell) and neither fighter has a result within the date
    tolerance (grade._cancelled). Otherwise it stays open."""
    if l.get("bet") not in sdy.WINNER_BETS:
        return market_result(l, res, now)
    n = l.get("names") or {}
    if l["sport"] == "NFL":
        m = res["nfl"].get(n.get("game_id"))
        if m is None:
            return None
        if m == 0:
            return {"winner": None, "void_reason": "tie"}
        return {"winner": "pick" if (m > 0) == bool(n.get("home")) else "other"}
    if l["sport"] == "Soccer":
        r = res["soccer"].get(n.get("match_id"))
        if r is None:
            return None
        return {"winner": None} if r == "void" else {"winner": "pick" if r == n.get("pick") else "other"}
    found, r = _ufc_bout(l, res, now)
    if found is None:
        return r
    grade = res["ufc"][0]
    _, winner, _ = found
    if not isinstance(winner, str) or not winner.strip():
        return {"winner": None, "void_reason": "draw or no contest"}
    corner = grade._corner(winner, n.get("red"), n.get("blue"))
    if corner is None:
        return None
    picked = "red" if grade.norm_name(l["pick"]) == grade.norm_name(n.get("red")) else "blue"
    return {"winner": "pick" if corner == picked else "other"}


def grade_all(history, now=None):
    """Settle each day's ticket of each kind (the newest row of that day and
    kind) in place; every older row of the same day and kind that was a
    ticket is "superseded". Returns the settled-or-pending rows, both kinds
    (kind_of tells them apart). `now` (default: the clock) decides whether a
    UFC card is complete, so a bout missing from it can be void as
    cancelled (leg_result); a void leg carries its void_reason."""
    now = now or datetime.now(timezone.utc)
    res = _results()
    if res.get("ufc") is not None and "ufc_cards" not in res:
        res = dict(res, **ufc_cancel_index(history, res["ufc"]))
    days = {}
    for r in history.get("tickets", []):
        days[(r["day"], kind_of(r))] = r
    graded = []
    for _, r in sorted(days.items()):
        if r.get("status") == "none":
            continue
        legs = []
        for l in r["legs"]:
            result = leg_result(l, res, now)
            l["outcome"] = sdy.pp.leg_outcome({"side": "pick", "market": "moneyline", "result": result})
            why = (result or {}).get("void_reason")
            if why and l["outcome"] == "void":
                l["void_reason"] = why
            else:
                l.pop("void_reason", None)
            legs.append({"side": "pick", "market": "moneyline", "decimal": float(l["decimal"]), "result": result})
        s = sdy.settle({"legs": legs})
        r["status"] = "pending" if s["status"] == "open" else s["status"]
        r["paid"], r["profit"] = s["paid"], s["profit"]
        graded.append(r)
    for r in history.get("tickets", []):
        if days.get((r["day"], kind_of(r))) is not r and r.get("status") not in ("none",):
            r["status"] = "superseded"
    return graded


def record(graded, kind=None):
    """The live record of the graded rows - of one kind when `kind` is given.
    "implied": the mean price-implied return of the settled tickets (their
    market chance times their multiplier, less 1), beside the realised one,
    so a reader sees what the prices expected of the same tickets."""
    rows = [r for r in graded if kind is None or kind_of(r) == kind]
    done = [r for r in rows if r.get("status") in ("won", "lost", "void")]
    won = sum(r["status"] == "won" for r in done)
    units = round(sum(r.get("profit") or 0.0 for r in done), 2)
    exp = [r["p_market"] * r["multiplier"] - 1 for r in done
           if r.get("p_market") is not None and r.get("multiplier") is not None]
    return {"settled": len(done), "won": won, "units": units,
            "roi": round(units / len(done), 4) if done else None,
            "implied_roi": round(sum(exp) / len(exp), 4) if exp and len(exp) == len(done) else None,
            # settled tickets paid at an ESTIMATED price (a UFC prop): their
            # units and implied return are estimates too, and the page says so
            "estimated": sum(bool(r.get("estimated")) for r in done),
            "pending": sum(r.get("status") == "pending" for r in rows),
            "last": [{k: r.get(k) for k in ("day", "status", "multiplier", "paid", "profit", "sports")}
                     for r in sorted(done, key=lambda r: r["day"])[-10:]]}


# ---------------------------------------------------------------- the page

def _band_text():
    lo, hi = sdy.DOG_PRICE
    return f"{lo:.2f}-{hi:.2f} (+{sdy.american(lo)} to +{sdy.american(hi)})"


def _x(band):
    return f"{band[0]:g}-{band[1]:g}x"


OTHER_MARKETS = "over/under, spread, total or method"


def none_reason_text(kind, code, candidates=None):
    """Why a day has no ticket of this kind, in a reader's words - every
    number read from sameday.py's constants."""
    if kind == "likeliest_v2":
        return (f"No combination of this day's priced winner picks (each at least a {sdy.MIN_P:.0%} chance) "
                f"pays {_x(sdy.BAND)} with {sdy.MAX_LEGS} legs or fewer.")
    if kind == "mixed":
        if code == "no_other":
            return (f"No mixed ticket: no {OTHER_MARKETS} leg is on offer this day (each but a method leg "
                    f"needs at least a {sdy.MIN_P:.0%} chance).")
        return (f"No mixed ticket: no combination holding an {OTHER_MARKETS} leg pays {_x(sdy.BAND)} "
                f"with {sdy.MAX_LEGS} legs or fewer.")
    if kind == "props":
        lo, hi = sdy.PROPS_PRICE
        if code == "no_card":
            return "No props ticket: there is no UFC card on this day."
        if code == "no_estimate":
            return ("No props ticket: no method prop on this card could be estimated (the reason is beside "
                    "the day's markets).")
        if code == "no_prop":
            return f"No props ticket: no method prop on this card is priced {lo:.2f}-{hi:.2f} (estimated)."
        n = candidates or 0
        return (f"No props ticket: {n} method prop{'' if n == 1 else 's'} priced {lo:.2f}-{hi:.2f}, but no "
                f"favourites on other events bring {'it' if n == 1 else 'any of them'} to "
                f"{_x(sdy.PROPS_LIVE_BAND)} at the estimated price.")
    if kind == "likeliest":
        return ("No combination of this day's priced picks (each at least a 50% chance) "
                "pays 3-5x with six legs or fewer.")
    if code == "no_fill":
        n = candidates or 0
        some = "an underdog" if n == 1 else (f"{n} underdogs" if n else "underdogs")
        return (f"No underdog ticket: there is {some} priced {_band_text()}, but no favourites on "
                f"other events bring {'it' if n == 1 else 'any of them'} to 3-5x.")
    if code == "no_dog":
        return (f"No underdog ticket: no underdog on this day is priced {_band_text()} with at least a "
                f"{sdy.DOG_FLOOR:.0%} chance.")
    return "No underdog ticket fits this day."


def _day_name(day):
    d = datetime.strptime(day, "%Y-%m-%d")
    return f"{d:%A}, {d:%b} {d.day}"


# The market each v2 leg is on, as the page tags it.
MARKET_NAME = {"OU25": "Over/under", "SPREAD": "Spread", "TOTAL": "Total", "METHOD": "Method"}


def leg_label(l):
    """A leg as a bet slip names it: "Arsenal to win", "Draw", "Over 2.5
    goals", "BUF -3.5", "Over 44.5", "Ciryl Gane by KO/TKO" - a v2 other-
    market leg's pick already is its label."""
    if l.get("bet") not in sdy.WINNER_BETS:
        return l.get("pick")
    if l.get("sport") == "Soccer" and (l.get("names") or {}).get("pick") == "D":
        return "Draw"
    return f"{l.get('pick')} to win"


def market_name(l):
    if l.get("bet") in MARKET_NAME:
        return MARKET_NAME[l["bet"]]
    return "1X2" if l.get("sport") == "Soccer" else "Moneyline"


def leg_reason(l):
    """One line from a v2 leg's own numbers. A method leg's chance is the
    corner's shipped win chance times the estimator's share of its wins that
    way, and its market chance the estimator's - never called the market's."""
    if l.get("bet") == "METHOD":
        bits = [f"UFC: chance {l['p']:.0%}"]
        if l.get("p_market") is not None:
            bits.append(f"the estimator's market chance {l['p_market']:.0%}")
        return ", ".join(bits)
    if l.get("bet") in MARKET_NAME:
        return f"{l['sport']}: chance {l['p']:.0%}, the market's (no-vig)"
    return sdy.reason(l)


def ticket_payload(day, row, now, kind="likeliest", none_reason=None, candidates=None, starts=None):
    """One ticket for the page. A day with no ticket says why: the reason
    logged on its "none" row, else this run's (update's `why`). A v2 ticket's
    legs carry their market (bet, market, label), and an estimated price its
    measured error; the ticket says whether any price on it is estimated."""
    v2 = kind in KINDS_V2
    if row is None or row.get("status") == "none":
        code = ((row or {}).get("none_reason") or none_reason
                or ("no_fit" if kind in ("likeliest", "likeliest_v2", "mixed") else None))
        out = {"day": day, "kind": kind, "status": "none", "legs": [], "multiplier": None, "p": None,
               "p_market": None, "none_reason": code,
               "missing_reason": none_reason_text(kind, code, candidates)}
        if v2:
            out.update(version=sdy.V2_VERSION, estimated=False)
        return out
    legs = []
    for l in row["legs"]:
        x = dict({k: l.get(k) for k in ("sport", "event", "pick", "opponent", "kickoff", "decimal",
                                        "p", "p_model", "p_market", "price_source")},
                 american=sdy.american(l["decimal"]), reason=leg_reason(l) if v2 else sdy.reason(l),
                 outcome=l.get("outcome"), void_reason=l.get("void_reason"), dog=bool(l.get("dog")))
        if v2:
            x.update(bet=l.get("bet") or "1X2", bet_detail=l.get("bet_detail"), label=leg_label(l),
                     market=market_name(l), estimated=bool(l.get("estimated")), error_pct=l.get("error_pct"),
                     moneyline_source=l.get("moneyline_source"))
        legs.append(x)
    status = row.get("status", "pending")
    if status == "pending":
        status = "locked" if locked(row, now, starts) else "open"
    out = {"day": day, "kind": kind, "status": status, "legs": legs, "multiplier": row.get("multiplier"),
           "p": row.get("p"), "p_market": row.get("p_market"), "sports": row.get("sports"),
           "first_start": row.get("first_start"), "last_start": row.get("last_start"),
           "built_at": row.get("built_at"), "paid": row.get("paid"), "profit": row.get("profit"),
           "missing_reason": None, "none_reason": None}
    if v2:
        out.update(version=row.get("version"), bets=row.get("bets"),
                   estimated=any(l.get("estimated") for l in legs))
    if kind == "props":
        pr = row.get("props") or {}
        ml = next((l for l in row["legs"] if l.get("key") == pr.get("key") and l.get("bet") == "METHOD"), {})
        out.update({"props": {"pick": ml.get("pick"), "event": ml.get("event"), "rank": pr.get("rank"),
                              "candidates": pr.get("candidates"), "version": pr.get("version"),
                              "band": pr.get("band")},
                    "only_one": pr.get("candidates") == 1})
    if kind in DOG_KINDS:
        dg = row.get("dog") or {}
        dl = next((l for l in row["legs"] if l.get("dog")), {})
        pm = row.get("p_market")
        best = dl.get("price_source") == BEST_OF_BOOKS
        out.update({
            "dog": {"pick": dl.get("pick"), "event": dl.get("event"), "sport": dl.get("sport"),
                    "rank": dg.get("rank"), "candidates": dg.get("candidates"), "salt": dg.get("salt"),
                    "version": dg.get("version")},
            "lands_1_in": round(1 / pm) if pm else None,
            "only_one": dg.get("candidates") == 1,
            "best_of_books": best,
            "price_note": ("This underdog is priced at the best price seen across books, which may not be "
                           "the book its other legs are best at: one book may pay less than this multiplier."
                           if best else None)})
    return out


def _json_safe(x):
    """The backtest's numbers as JSON can carry them: a NaN or infinity read
    from the file is missing (None), never a number."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def implied_from_tickets(bt, path=None):
    """{rule: {period: price-implied return}} for the likeliest ticket and its
    market-only control, from the backtest's own ticket list: the mean of
    each ticket's market chance times its multiplier, less 1 - outcome-free,
    what the prices expected of the same tickets. WHY HERE: the backtest's
    json carries it for the underdog ticket but not for these two, and the
    page shows it beside every realised return. A period whose ticket count
    does not match the json's (the two files out of step) is left missing."""
    import csv
    import gzip
    path = Path(BACKTEST_TICKETS if path is None else path)
    try:
        with gzip.open(path, "rt", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, EOFError, csv.Error):
        return {}
    out = {}
    for rule, res in (bt.get("results") or {}).items():
        for period, span in (bt.get("periods") or {}).items():
            s = (res or {}).get(period) or {}
            vals = []
            for r in rows:
                if r.get("rule") != rule or not (span[0] <= str(r.get("day")) <= span[1]):
                    continue
                q, m = _num(r.get("p_market")), _num(r.get("multiplier"))
                vals.append(None if q is None or m is None else q * m - 1)
            if vals and None not in vals and len(vals) == s.get("tickets"):
                out.setdefault(rule, {})[period] = round(sum(vals) / len(vals), 4)
    return out


def dog_measured(bt):
    """(measured underdog block, None) from sameday_backtest.json's "dog"
    block, or (None, the reason it is not shown). Read defensively: the
    block is written by the backtest, separately, and a block for another
    version or salt is another rule's record - never shown as this one's.

    The reasons are reader-facing text, shown on the page: no file path
    (an unbreakable token that runs past a phone's card) and no "salt" (the
    page calls it the draw). The block is written by parlays/engine/
    experiments/sameday_backtest.py."""
    blk = (bt or {}).get("dog")
    if not isinstance(blk, dict):
        return None, ("Not measured yet: the underdog ticket's backtest has not been run, so no record "
                      "is shown for it.")
    if blk.get("version") != sdy.DOG_VERSION or blk.get("salt") != sdy.DOG_SALT:
        return None, (f"Not shown: the underdog record on file was measured for another version of the "
                      f"rule (v{blk.get('version')}), not the live v{sdy.DOG_VERSION}, so it is not this "
                      f"ticket's record.")
    res = blk.get("results") if isinstance(blk.get("results"), dict) else {}
    live = res.get("live") if isinstance(res.get("live"), dict) else {}
    if not any(isinstance(v, dict) and v.get("tickets") for v in live.values()):
        return None, "Not measured yet: the underdog record on file has no tickets at the live draw."
    return _json_safe({
        "version": blk.get("version"), "salt": blk.get("salt"), "rule": blk.get("rule"),
        "periods": blk.get("periods"), "headline": blk.get("headline"), "missing": blk.get("missing"),
        "live": live, "salts": res.get("salts"),
        "report_only": {k: res.get(k) for k in ("gap", "likeliest_dog", "market_only", "paired", "both")},
        "splits": blk.get("splits")}), None


# ---------------------------------------------------------------- tickets v2: measured

# Where a v2 kind's record may sit in the "v2" block, by the names the
# measurement could give it (the history kinds first), and which of its
# readings is the headline: a lottery's live draw, the props ticket "as
# live" (chosen at the estimate, paid at the real props - sameday.py).
V2_BLOCK_NAMES = {"likeliest_v2": ("likeliest_v2", "likeliest", "same_day_v2"),
                  "underdog_v2": ("underdog_v2", "underdog", "dog", "same_day_dog_v2"),
                  "mixed": ("mixed", "same_day_mixed"),
                  "props": ("props", "same_day_props")}
V2_HEADLINE = {"likeliest_v2": ("rule", "results", "live"), "underdog_v2": ("live", "results"),
               "mixed": ("rule", "results", "live"), "props": ("as_live", "live", "results")}
V2_SALTS = {"underdog_v2": sdy.DOG_SALT_V2, "props": sdy.PROPS_SALT}


def _periods(x):
    """x when it is {period: record} with a record carrying tickets."""
    if not isinstance(x, dict):
        return None
    recs = {k: v for k, v in x.items() if isinstance(v, dict) and "tickets" in v}
    return recs if any(v.get("tickets") for v in recs.values()) else None


def _reading(kb, keys):
    """(the first of kb's readings named `keys` that is {period: record},
    its name), looked for at the top of the block and under "results"."""
    res = kb.get("results") if isinstance(kb.get("results"), dict) else {}
    for k in keys:
        for src in (kb, res):
            p = _periods(src.get(k))
            if p:
                return p, k
    p = _periods(res) or _periods(kb)
    return (p, "results") if p else (None, None)


def v2_measured(bt):
    """({kind: measured block or None}, {kind: why it is None, or None}) for
    the four v2 kinds, from sameday_backtest.json's "v2" block - read
    defensively: the measurement writes it separately, in its own shape, and
    may not have yet. A block for another version, or a lottery measured at
    another draw, is another rule's record and is never shown as this one's.
    Each block: headline {period: record} (the reading named in
    headline_from), periods, and what the block carries beside it - the
    lottery's mean over its simulation draws (salts), the props ticket "as
    priced", the market-only control, splits, activity (tickets per day,
    days with a ticket) and caveats - each passed through as written, NaN
    as missing. The reasons are reader-facing (no path, no "salt")."""
    blk = (bt or {}).get("v2")
    if not isinstance(blk, dict):
        why = ("Not measured yet: the v2 tickets' backtest has not been run, so no record is shown for them."
               if bt else "Not measured yet: the backtest of these tickets has not been run, so no record "
                          "is shown.")
        return {k: None for k in KINDS_V2}, {k: why for k in KINDS_V2}
    if blk.get("version") is not None and blk.get("version") != sdy.V2_VERSION:
        why = (f"Not shown: the v2 record on file was measured for another version of the rule "
               f"(v{blk.get('version')}), not the live v{sdy.V2_VERSION}.")
        return {k: None for k in KINDS_V2}, {k: why for k in KINDS_V2}
    kinds = blk.get("kinds") if isinstance(blk.get("kinds"), dict) else blk
    out, why = {}, {}
    for kind in KINDS_V2:
        kb = next((kinds[n] for n in V2_BLOCK_NAMES[kind] if isinstance(kinds.get(n), dict)), None)
        if kb is None:
            out[kind], why[kind] = None, "Not measured yet: the v2 backtest on file has no record of this ticket."
            continue
        if kind in V2_SALTS and kb.get("salt") is not None and kb.get("salt") != V2_SALTS[kind]:
            out[kind], why[kind] = None, ("Not shown: the record on file was measured at another draw, not "
                                          "this ticket's live one.")
            continue
        head, name = _reading(kb, V2_HEADLINE[kind])
        if head is None:
            out[kind], why[kind] = None, "Not measured yet: the v2 record on file for this ticket has no tickets."
            continue
        res = kb.get("results") if isinstance(kb.get("results"), dict) else {}
        extra = {k: kb.get(k, res.get(k)) for k in ("salts", "as_priced", "market_only", "splits", "missing",
                                                    "rule", "activity", "per_day")}
        out[kind] = _json_safe(dict({k: v for k, v in extra.items() if v is not None},
                                    headline=head, headline_from=name,
                                    periods=kb.get("periods") or blk.get("periods") or (bt or {}).get("periods"),
                                    activity_rows={p: {k: r.get(k) for k in ("days", "days_with_ticket",
                                                                             "share_days", "tickets_per_day")
                                                       if k in r} for p, r in head.items()},
                                    caveats=[c for c in kb.get("caveats") or [] if isinstance(c, str)]))
        why[kind] = None
    return out, why


# The builder's own v2 caveats: what the new markets are priced at, what an
# estimated price is, and what v2 is not. No number typed: each sits beside
# the measured figure it qualifies on the page, or is read from sameday.py.
V2_CAVEATS = [
    "Tickets v2 add markets: soccer over or under 2.5 goals at the average posted price, in the leagues "
    "whose over 2.5 the app ships from the market; NFL spreads and totals at the consensus price at the "
    "consensus line (a push voids the leg); and UFC method props, each corner to win by knockout, by "
    "submission or by decision. Their winner legs take soccer from every league in the app.",
    "No book on the feeds prices a UFC method prop, so each prop's price is an ESTIMATE (marked est.), from "
    "a market-only estimator, shown with the median error it was measured to miss real prop prices by. A "
    "ticket holding one shows its multiplier as an estimate, and its live record is at the estimated "
    "prices: a real book may pay less, or more.",
    "A prop's book carries far more margin than a moneyline pair's, spread over its six outcomes, so the "
    "props and mixed tickets pay that margin on their props; by their prices they are expected to return "
    "less than the likeliest ticket.",
    "The props estimate reads the fight's moneyline at the card's median pair across books where the card "
    "carries it - the closest live match to the closing line the estimator was fitted on - else at the best "
    "pair; each prop on file says which.",
    "Tickets v2 were fixed before any of their results was computed, and ship whatever their records show; "
    "none is presented as an edge. The market is hard to beat.",
    "The v1 tickets (the likeliest and the underdog, top-five soccer, winners only) keep being built, logged "
    "and graded beside v2 on their own rules, each with its own record, labelled v1.",
]


# Shown until the backtest's dog block (with its own, measured caveats) exists.
DOG_CAVEATS = [
    "The underdog ticket is for variety and a bigger payout, not a better bet: drawing an underdog "
    "cannot create an edge, and every 3-5x ticket pays the bookmaker's margin on each leg. Written "
    "before it was measured: by the prices it lands about 1 time in 4 at about 3.8x (the likeliest "
    "about 2 in 7 at about 3.1x), about -8% a ticket.",
    "A dog leg's market chance reads a little generous: bookmakers load more margin on underdogs (about "
    "5% on a dog leg against 3% on a favourite), and taking the margin out in proportion spreads it "
    "evenly. It is expected to do somewhat worse than its prices imply.",
    "Its record is not an out-of-sample test of underdogs: the rule's own results were not seen before "
    "it was fixed, but underdog results over the same years had been (the UFC risker tickets, the NFL "
    "and soccer edge checks).",
    "The underdog is priced at the typical book where the files allow: UFC at the median across the books "
    "on file when the card carries one from two books or more (else the best price, flagged on the leg: "
    "a multiplier no single book may pay), NFL the consensus moneyline, soccer the average price. Its "
    "favourites are priced as the likeliest ticket's legs.",
    "The two tickets often share a leg, and on some days the underdog is the other side of a likeliest "
    "leg: then the two cannot both win, and the page says so.",
]


def payload(history, show, now, graded, why=None, starts=None):
    why = why or {}
    bt = _load(BACKTEST)
    measured = None
    caveats = []
    if bt:
        sp = bt["splits"]["same_day"]
        for rule, per in implied_from_tickets(bt).items():
            for period, x in per.items():
                bt["results"][rule][period].setdefault("implied_roi", x)
        likeliest = _json_safe({"rule": bt["results"]["same_day"], "market_only": bt["results"]["market_only"],
                                "periods": bt["periods"], "multipliers": sp["multipliers"],
                                "splits": {k: sp[k] for k in ("by_sports", "by_mix", "by_weekday", "by_year")},
                                "legs_used": sp.get("legs_used")})
        underdog, missing = dog_measured(bt)
        measured = dict(likeliest, likeliest=likeliest, underdog=underdog, underdog_missing_reason=missing)
        caveats = list(bt.get("caveats") or [])
        if underdog is None:
            caveats.append(missing)
        else:
            caveats += [c for c in (bt["dog"].get("caveats") or []) if isinstance(c, str) and c not in caveats]
    else:
        # (the backtest is parlays/engine/experiments/sameday_backtest.py; no
        # path in the reader's text - see dog_measured)
        caveats = ["Not measured yet: the backtest of these tickets has not been run, so no record is "
                   "shown. The market is hard to beat."]
    caveats.append("Prices are the ones on file when the sports' runners last ran: UFC the best price seen "
                   "across books, NFL the consensus moneyline, soccer the average posted price. One book "
                   "may pay less. Check every price before betting; a leg whose game has started is never used.")
    # This builder's underdog caveats stand in for the backtest's own until its
    # dog block exists; once it does, its caveats say the same with numbers.
    if not (measured or {}).get("underdog"):
        caveats += [c for c in DOG_CAVEATS if c not in caveats]
    v2m, v2why = v2_measured(bt)
    caveats += [c for c in V2_CAVEATS if c not in caveats]
    caveats += [w for w in dict.fromkeys(v2why.values()) if w and w not in caveats]
    v2blk = (bt or {}).get("v2") if isinstance((bt or {}).get("v2"), dict) else {}
    caveats += [c for c in v2blk.get("caveats") or [] if isinstance(c, str) and c not in caveats]
    for blk in v2m.values():
        caveats += [c for c in (blk or {}).get("caveats") or [] if c not in caveats]

    today = sdy.et_day(now)
    days = []
    for d in show:
        rl, ru = latest(history, d, "likeliest"), latest(history, d, "underdog")
        lk = ticket_payload(d, rl, now, "likeliest", why.get((d, "likeliest")), starts=starts)
        dg = ticket_payload(d, ru, now, "underdog", why.get((d, "underdog")), why.get((d, "candidates")), starts)
        past = d < today
        x = {"day": d, "today": d == today, "past": past, "likeliest": lk, "underdog": dg,
             "cannot_both_win": lk["status"] != "none" and dg["status"] != "none" and sdy.opposes(ru, rl)}
        # v2: the four kinds - the props ticket on a day with a UFC card (or
        # with a props row logged, so a card moved off the day says so).
        r2 = {k: latest(history, d, k) for k in KINDS_V2}
        ufc_day = bool(why.get((d, "ufc_day"))) or r2["props"] is not None
        x.update({"ufc_day": ufc_day, "missing": why.get((d, "missing")) or {},
                  "likeliest_v2": ticket_payload(d, r2["likeliest_v2"], now, "likeliest_v2",
                                                 why.get((d, "likeliest_v2")), starts=starts),
                  "underdog_v2": ticket_payload(d, r2["underdog_v2"], now, "underdog_v2",
                                                why.get((d, "underdog_v2")), why.get((d, "candidates_v2")), starts),
                  "mixed": ticket_payload(d, r2["mixed"], now, "mixed", why.get((d, "mixed")), starts=starts),
                  "props": (ticket_payload(d, r2["props"], now, "props", why.get((d, "props")),
                                           why.get((d, "props_candidates")), starts) if ufc_day else None)})
        x["v2_cannot_both_win"] = (x["likeliest_v2"]["status"] != "none" and x["underdog_v2"]["status"] != "none"
                                   and sdy.opposes(r2["underdog_v2"], r2["likeliest_v2"]))
        x["in_play"] = past and any(t and t["status"] == "locked" for t in (x[k] for k in KINDS))
        days.append(x)
    # Today's "none" names the next day with a ticket of its kind (days in
    # play come before today, so today is found, not assumed first).
    at = next((i for i, x in enumerate(days) if x["today"]), None)
    labels = {"likeliest": "likeliest", "underdog": "underdog", "likeliest_v2": "likeliest", "underdog_v2":
              "underdog", "mixed": "mixed", "props": "props"}
    for kind in KINDS:
        first = days[at][kind] if at is not None else None
        if first is None or first["status"] != "none":
            continue
        label = labels[kind]
        nxt = next((x["day"] for x in days[at + 1:] if x[kind] and x[kind]["status"] != "none"), None)
        first["next_day"] = nxt
        first["missing_reason"] += (f" The next {label} ticket is on {_day_name(nxt)}." if nxt else
                                    f" No {label} ticket in the next {WINDOW_DAYS} days yet.")
    records = {k: record(graded, k) for k in KINDS}
    return {"generated": _iso(now), "day_tz": sdy.DAY_TZ,
            "rule": {"band": list(sdy.BAND), "max_legs": sdy.MAX_LEGS, "pool": sdy.POOL, "min_p": sdy.MIN_P,
                     "window_days": WINDOW_DAYS,
                     "dog": {"version": sdy.DOG_VERSION, "salt": sdy.DOG_SALT, "price": list(sdy.DOG_PRICE),
                             "floor": sdy.DOG_FLOOR, "sim_salts": len(sdy.DOG_SIM_SALTS)},
                     "v2": {"version": sdy.V2_VERSION, "kinds": list(KINDS_V2), "markets": list(sdy.LEG_MARKETS),
                            "method_types": list(sdy.METHOD_TYPES), "dog_salt": sdy.DOG_SALT_V2,
                            "props_salt": sdy.PROPS_SALT, "props_price": list(sdy.PROPS_PRICE),
                            "props_live_band": list(sdy.PROPS_LIVE_BAND),
                            "mixed_pool_other": sdy.MIXED_POOL_OTHER,
                            "sim_salts": len(sdy.DOG_SIM_SALTS_V2), "props_sim_salts": len(sdy.PROPS_SIM_SALTS)}},
            "kinds": {"v1": list(KINDS_V1), "v2": list(KINDS_V2)},
            "days": days,
            "tickets": [x["likeliest"] for x in days if not x["past"]],
            "record": records["likeliest"], "records": records,
            "measured": measured,
            "measured_v2": {"kinds": v2m, "missing_reason": v2why},
            "caveats": caveats}


def load_history(path=None):
    """history.json, or a fresh log when there is none yet. A file that is
    there but does not read, or has no "tickets" list, stops the run: read
    as empty, the run would write an empty log over every ticket logged so
    far - the live record, gone."""
    path = Path(HISTORY if path is None else path)
    if not path.exists():
        return {"schema": 1, "tickets": []}
    try:
        history = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} does not read ({err}); nothing written") from err
    if not isinstance(history, dict) or not isinstance(history.get("tickets"), list):
        raise ValueError(f"{path} has no 'tickets' list; nothing written")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", help="ISO time to build as of (testing)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    now = _parse(args.now) if args.now else datetime.now(timezone.utc)
    history = load_history()
    # The app files are read (strictly) before anything is written: a file
    # that is there but unreadable stops the run here, history.json untouched.
    files = load_app_files()
    starts = starts_on_file(files)
    legs, dogs, why = live_legs(now, files), live_dogs(now, files), {}
    show = update(history, now, legs, dogs, why, starts, v2=live_v2(now, files))
    graded = grade_all(history, now)
    out = payload(history, show, now, graded, why, starts)
    text = json.dumps(out, indent=1, allow_nan=False)
    if args.no_write:
        print(text)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    HISTORY.write_text(json.dumps(history, indent=1, allow_nan=False))
    for x in out["days"]:
        for kind in KINDS:
            t = x[kind]
            if t is None:
                continue
            legs = ", ".join(f"{l['sport']} {l.get('label') or l['pick']}{' (dog)' if l.get('dog') else ''} "
                             f"{l['decimal']:.2f}{' est.' if l.get('estimated') else ''}" for l in t["legs"])
            print(f"{t['day']} {kind}: {t['status']} {t['multiplier'] or ''} {legs or t['missing_reason']}")


if __name__ == "__main__":
    main()
