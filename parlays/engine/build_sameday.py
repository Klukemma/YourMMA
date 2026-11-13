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

KINDS = ("likeliest", "underdog")
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


def soccer_legs(data, now):
    out = []
    for m in (data or {}).get("matches", []):
        k = _parse(m.get("kickoff"))
        pick = m.get("pick")
        if m.get("league") not in SOCCER_MEASURED:
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


def soccer_dogs(data, now):
    """Each measured-league match's underdog TEAM (never the draw): the lower
    of the no-vig market's home and away chances, at the average price, its
    chance the shipped p_h / p_a. Level home and away: none."""
    out = []
    for m in (data or {}).get("matches", []):
        if m.get("league") not in SOCCER_MEASURED or not m.get("match_id"):
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


# ---------------------------------------------------------------- logging

LEG_FIELDS = ("sport", "key", "event", "pick", "opponent", "kickoff", "decimal", "p", "p_model", "p_market",
              "names", "price_source", "dog")


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


def ticket_row(day, t, now, kind="likeliest", none_reason=None):
    if t is None:
        return {"day": day, "kind": kind, "built_at": _iso(now), "status": "none", "multiplier": None,
                "legs": [], "none_reason": none_reason}
    starts = [l["kickoff"] for l in t["legs"] if l.get("kickoff")]
    row = {"day": day, "kind": kind, "built_at": _iso(now), "status": "pending",
           "multiplier": round(t["multiplier"], 4), "p": round(t["p"], 6),
           "p_market": None if t["p_market"] is None else round(t["p_market"], 6),
           "first_start": min(starts) if starts else None, "last_start": max(starts) if starts else None,
           "sports": t["sports"],
           "legs": [{k: l.get(k) for k in LEG_FIELDS} for l in t["legs"]]}
    if kind == "underdog":
        d = next(l for l in t["legs"] if l.get("dog"))
        row["dog"] = {"key": d["key"], "pick": d["pick"], "lot_key": d.get("lot_key"),
                      "rank": t.get("dog_rank"), "candidates": t.get("candidates"), "salt": t.get("salt"),
                      "version": sdy.DOG_VERSION}
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
    return (l.get("key"), l.get("pick"), l.get("decimal"), l.get("kickoff"))


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
    if row.get("kind") == "underdog":
        a, b = prev.get("dog") or {}, row.get("dog") or {}
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


def update(history, now, legs, dogs=(), why=None, starts=None):
    """Build and log both kinds of ticket for every day shown; returns the
    days. A kind whose newest row is locked is left alone; a kind with no row
    and nothing built logs nothing (there was never a ticket to supersede).
    `why`, when given a dict, gets {(day, kind): none_reason} and
    {(day, "candidates"): n} for this run, so the page can say why a day
    without a logged row has no ticket. A day before today (in play) is
    shown, never built: its events have all started, so a rebuild could only
    log a wrong "none" over what was there."""
    show = days_shown(history, now, legs, dogs, starts)
    today = sdy.et_day(now)
    for day in show:
        if day < today:
            continue
        built = build_day(day, legs, dogs)
        if why is not None:
            why[(day, "candidates")] = built["candidates"]
        for kind in KINDS:
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
    """The three sports' results, loaded once."""
    sys.path.insert(0, str(ROOT / "engine"))
    import grade
    import pandas as pd
    out = {"ufc": None, "nfl": {}, "soccer": {}}
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
    except Exception:
        pass
    out["soccer"] = soccer_results()
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


def soccer_results(path=None):
    """{match_id: "H" / "D" / "A" / "void"} for every soccer pick with a
    result or void for good; a match with neither is left out (its leg
    stays open).

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
            r = fin["result"]
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


def leg_result(l, res, now=None):
    """A leg's result as parlay_profiles settles it: {"winner": "pick" /
    "other"}, {"winner": None, "void_reason": ...} (void: books return the
    leg at 1.0), or None (not known yet - the leg stays open).

    A UFC BOUT THAT NEVER HAPPENED (called off on fight night, after the
    ticket locked) has no archive row, so "not found" alone would keep its
    ticket open for good - and a ticket with an open leg still loses the
    moment another leg loses, so the record would keep the losses and never
    the wins. It is void as "cancelled" exactly when engine/grade.py's
    grade_parlays voids it: the card has results in (res["ufc_cards"]), the
    card is old enough to be complete at `now` (grade.card_is_complete on
    the leg's first bell) and neither fighter has a result within the date
    tolerance (grade._cancelled). Otherwise it stays open."""
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
    if res["ufc"] is None:
        return None
    grade, index = res["ufc"]
    found = grade.find_result({"red_corner": n.get("red"), "blue_corner": n.get("blue"),
                               "event_date": n.get("event_date")}, index)
    if found is None:
        cards, fighters, day = res.get("ufc_cards"), res.get("ufc_fighters"), n.get("event_date")
        if now is None or cards is None or fighters is None or not day:
            return None
        card = {"first_start": l.get("kickoff"), "event_date": day}
        if (day in cards and grade.card_is_complete(card, now)
                and grade._cancelled({"red": n.get("red"), "blue": n.get("blue")}, day, fighters)):
            return {"winner": None, "void_reason": "cancelled"}
        return None
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
            "pending": sum(r.get("status") == "pending" for r in rows),
            "last": [{k: r.get(k) for k in ("day", "status", "multiplier", "paid", "profit", "sports")}
                     for r in sorted(done, key=lambda r: r["day"])[-10:]]}


# ---------------------------------------------------------------- the page

def _band_text():
    lo, hi = sdy.DOG_PRICE
    return f"{lo:.2f}-{hi:.2f} (+{sdy.american(lo)} to +{sdy.american(hi)})"


def none_reason_text(kind, code, candidates=None):
    """Why a day has no ticket of this kind, in a reader's words."""
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


def ticket_payload(day, row, now, kind="likeliest", none_reason=None, candidates=None, starts=None):
    """One ticket for the page. A day with no ticket says why: the reason
    logged on its "none" row, else this run's (update's `why`)."""
    if row is None or row.get("status") == "none":
        code = (row or {}).get("none_reason") or none_reason or ("no_fit" if kind == "likeliest" else None)
        return {"day": day, "kind": kind, "status": "none", "legs": [], "multiplier": None, "p": None,
                "p_market": None, "none_reason": code,
                "missing_reason": none_reason_text(kind, code, candidates)}
    legs = []
    for l in row["legs"]:
        legs.append(dict({k: l.get(k) for k in ("sport", "event", "pick", "opponent", "kickoff", "decimal",
                                                 "p", "p_model", "p_market", "price_source")},
                         american=sdy.american(l["decimal"]), reason=sdy.reason(l), outcome=l.get("outcome"),
                         void_reason=l.get("void_reason"), dog=bool(l.get("dog"))))
    status = row.get("status", "pending")
    if status == "pending":
        status = "locked" if locked(row, now, starts) else "open"
    out = {"day": day, "kind": kind, "status": status, "legs": legs, "multiplier": row.get("multiplier"),
           "p": row.get("p"), "p_market": row.get("p_market"), "sports": row.get("sports"),
           "first_start": row.get("first_start"), "last_start": row.get("last_start"),
           "built_at": row.get("built_at"), "paid": row.get("paid"), "profit": row.get("profit"),
           "missing_reason": None, "none_reason": None}
    if kind == "underdog":
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

    today = sdy.et_day(now)
    days = []
    for d in show:
        rl, ru = latest(history, d, "likeliest"), latest(history, d, "underdog")
        lk = ticket_payload(d, rl, now, "likeliest", why.get((d, "likeliest")), starts=starts)
        dg = ticket_payload(d, ru, now, "underdog", why.get((d, "underdog")), why.get((d, "candidates")), starts)
        past = d < today
        days.append({"day": d, "today": d == today, "past": past,
                     "in_play": past and any(t["status"] == "locked" for t in (lk, dg)),
                     "likeliest": lk, "underdog": dg,
                     "cannot_both_win": lk["status"] != "none" and dg["status"] != "none" and sdy.opposes(ru, rl)})
    # Today's "none" names the next day with a ticket of its kind (days in
    # play come before today, so today is found, not assumed first).
    at = next((i for i, x in enumerate(days) if x["today"]), None)
    for kind, label in (("likeliest", "likeliest"), ("underdog", "underdog")):
        first = days[at][kind] if at is not None else None
        if first is None or first["status"] != "none":
            continue
        nxt = next((x["day"] for x in days[at + 1:] if x[kind]["status"] != "none"), None)
        first["next_day"] = nxt
        first["missing_reason"] += (f" The next {label} ticket is on {_day_name(nxt)}." if nxt else
                                    f" No {label} ticket in the next {WINDOW_DAYS} days yet.")
    records = {k: record(graded, k) for k in KINDS}
    return {"generated": _iso(now), "day_tz": sdy.DAY_TZ,
            "rule": {"band": list(sdy.BAND), "max_legs": sdy.MAX_LEGS, "pool": sdy.POOL, "min_p": sdy.MIN_P,
                     "window_days": WINDOW_DAYS,
                     "dog": {"version": sdy.DOG_VERSION, "salt": sdy.DOG_SALT, "price": list(sdy.DOG_PRICE),
                             "floor": sdy.DOG_FLOOR, "sim_salts": len(sdy.DOG_SIM_SALTS)}},
            "days": days,
            "tickets": [x["likeliest"] for x in days if not x["past"]],
            "record": records["likeliest"], "records": records,
            "measured": measured, "caveats": caveats}


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
    show = update(history, now, legs, dogs, why, starts)
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
            legs = ", ".join(f"{l['sport']} {l['pick']}{' (dog)' if l.get('dog') else ''} {l['decimal']:.2f}"
                             for l in t["legs"])
            print(f"{t['day']} {kind}: {t['status']} {t['multiplier'] or ''} {legs or t['missing_reason']}")


if __name__ == "__main__":
    main()
