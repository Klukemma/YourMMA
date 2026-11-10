"""The live same-day ticket: built from the three apps' data, logged, graded.

Reads what each sport's runner last committed - app/data/card.json (UFC),
nfl/app/data/week.json, soccer/app/data/matches.json - and, by the rule in
sameday.py, builds the ticket for TODAY (America/New_York) and for the next
day that has legs. Writes parlays/app/data/sameday.json for the page and
appends to parlays/data/history.json.

THE CLOCK DECIDES, NOT A STATUS. An app file is rewritten only when its
sport's runner runs, so a match it lists as "upcoming" may have started; a
leg is used only while its scheduled start is after `now`.

ONE TICKET A DAY, LOCKED AT ITS FIRST START. Before a day's ticket's first
leg starts, every run rebuilds it at the latest prices and logs the new one
(or a "none" row when nothing fits any more), and the newest logged row is
the day's ticket. From its first start on it is locked: no run replaces it
with one built from the games still to come - the ticket a reader saw in
the morning is the one that is graded.

GRADING reads each sport's own results: UFC through engine/grade.py's
name-tolerant match on the archive (a bout with no result stays open), NFL
the schedule's played games (a tie is void), soccer the prediction log's
finals (postponed, cancelled or abandoned is void).
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
UFC_CARD = ROOT / "app" / "data" / "card.json"
UFC_LOG = ROOT / "engine" / "data" / "prediction_history.json"
UFC_ARCHIVE = ROOT / "engine" / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
NFL_WEEK = ROOT / "nfl" / "app" / "data" / "week.json"
NFL_GAMES = ROOT / "nfl" / "data" / "games.csv.gz"
SOCCER_MATCHES = ROOT / "soccer" / "app" / "data" / "matches.json"
SOCCER_LOG = ROOT / "soccer" / "data" / "prediction_history.json"
VOID_STATUSES = {"postponed", "cancelled", "canceled", "abandoned", "void", "suspended"}


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
                    opponent=opp, kickoff=_iso(start) if start else None, number=f.get("number"))
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
                    opponent=g.get("away") if home else g.get("home"), kickoff=_iso(k))
        if l:
            l["names"] = {"game_id": g["game_id"], "home": home}
            out.append(l)
    return out


def soccer_legs(data, now):
    out = []
    for m in (data or {}).get("matches", []):
        k = _parse(m.get("kickoff"))
        pick = m.get("pick")
        if k is None or k <= now or pick not in ("H", "D", "A") or m.get("price_status") != "posted":
            continue
        avg = ((m.get("prices") or {}).get("avg") or {})
        dec = _num(avg.get(pick.lower()))
        home, away = (m.get("home") or {}).get("name"), (m.get("away") or {}).get("name")
        name = {"H": home, "D": "Draw", "A": away}[pick]
        p_mk = _num((m.get("market") or {}).get(f"p_{pick.lower()}"))
        l = sdy.leg("Soccer", m["match_id"], sdy.et_day(k), f"{home} v {away}", name,
                    _num(m.get("pick_prob")), dec, p_market=p_mk, opponent=f"{m.get('league')}",
                    kickoff=_iso(k))
        if l:
            l["names"] = {"match_id": m["match_id"], "pick": pick}
            out.append(l)
    return out


def live_legs(now):
    return (ufc_legs(_load(UFC_CARD), now) + nfl_legs(_load(NFL_WEEK), now)
            + soccer_legs(_load(SOCCER_MATCHES), now))


# ---------------------------------------------------------------- logging

def ticket_row(day, t, now):
    if t is None:
        return {"day": day, "built_at": _iso(now), "status": "none", "multiplier": None, "legs": []}
    starts = [l["kickoff"] for l in t["legs"] if l.get("kickoff")]
    return {"day": day, "built_at": _iso(now), "status": "pending",
            "multiplier": round(t["multiplier"], 4), "p": round(t["p"], 6),
            "p_market": None if t["p_market"] is None else round(t["p_market"], 6),
            "first_start": min(starts) if starts else None, "last_start": max(starts) if starts else None,
            "sports": t["sports"],
            "legs": [{k: l.get(k) for k in ("sport", "key", "event", "pick", "opponent", "kickoff",
                                            "decimal", "p", "p_model", "p_market", "names")}
                     for l in t["legs"]]}


def latest(history, day):
    rows = [r for r in history.get("tickets", []) if r.get("day") == day]
    return rows[-1] if rows else None


def locked(row, now):
    start = _parse((row or {}).get("first_start"))
    return row is not None and row.get("status") != "none" and start is not None and start <= now


def update(history, now, legs):
    """Build and log today's and the next day's tickets. Returns the days."""
    today = sdy.et_day(now)
    days = sorted({l["day"] for l in legs if l["day"] and l["day"] >= today})
    show = [today] + [d for d in days if d > today][:1]
    for day in show:
        prev = latest(history, day)
        if locked(prev, now):
            continue
        t = sdy.build([l for l in legs if l["day"] == day])
        row = ticket_row(day, t, now)
        same = prev is not None and prev.get("status") == row["status"] and \
            [(l["key"], l["decimal"]) for l in prev.get("legs", [])] == [(l["key"], l["decimal"]) for l in row["legs"]]
        if prev is None and t is None:
            continue
        if not same:
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
    for mid, p in ((_load(SOCCER_LOG, {}) or {}).get("picks") or {}).items():
        fin = p.get("final") or {}
        if fin.get("result") in ("H", "D", "A"):
            out["soccer"][mid] = fin["result"]
        elif str(p.get("status", "")).lower() in VOID_STATUSES:
            out["soccer"][mid] = "void"
    return out


def leg_result(l, res):
    n = l.get("names") or {}
    if l["sport"] == "NFL":
        m = res["nfl"].get(n.get("game_id"))
        if m is None:
            return None
        return {"winner": None} if m == 0 else {"winner": "pick" if (m > 0) == bool(n.get("home")) else "other"}
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
        return None
    _, winner, _ = found
    if not isinstance(winner, str) or not winner.strip():
        return {"winner": None}
    corner = grade._corner(winner, n.get("red"), n.get("blue"))
    if corner is None:
        return None
    picked = "red" if grade.norm_name(l["pick"]) == grade.norm_name(n.get("red")) else "blue"
    return {"winner": "pick" if corner == picked else "other"}


def grade_all(history):
    """Settle each day's ticket (its newest row) in place; returns them."""
    res = _results()
    days = {}
    for r in history.get("tickets", []):
        days[r["day"]] = r
    graded = []
    for day, r in sorted(days.items()):
        if r.get("status") == "none":
            continue
        legs = []
        for l in r["legs"]:
            result = leg_result(l, res)
            l["outcome"] = sdy.pp.leg_outcome({"side": "pick", "market": "moneyline", "result": result})
            legs.append({"side": "pick", "market": "moneyline", "decimal": float(l["decimal"]), "result": result})
        s = sdy.settle({"legs": legs})
        r["status"] = "pending" if s["status"] == "open" else s["status"]
        r["paid"], r["profit"] = s["paid"], s["profit"]
        graded.append(r)
    for r in history.get("tickets", []):
        if days.get(r["day"]) is not r and r.get("status") not in ("none",):
            r["status"] = "superseded"
    return graded


def record(graded):
    done = [r for r in graded if r.get("status") in ("won", "lost", "void")]
    won = sum(r["status"] == "won" for r in done)
    units = round(sum(r.get("profit") or 0.0 for r in done), 2)
    return {"settled": len(done), "won": won, "units": units,
            "roi": round(units / len(done), 4) if done else None,
            "pending": sum(r.get("status") == "pending" for r in graded),
            "last": [{k: r.get(k) for k in ("day", "status", "multiplier", "paid", "profit", "sports")}
                     for r in sorted(done, key=lambda r: r["day"])[-10:]]}


# ---------------------------------------------------------------- the page

def ticket_payload(day, row, now):
    if row is None or row.get("status") == "none":
        return {"day": day, "status": "none", "legs": [], "multiplier": None, "p": None, "p_market": None,
                "missing_reason": "No combination of this day's priced picks (each at least a 50% chance) "
                                  "pays 3-5x with six legs or fewer."}
    legs = []
    for l in row["legs"]:
        legs.append(dict({k: l.get(k) for k in ("sport", "event", "pick", "opponent", "kickoff", "decimal",
                                                 "p", "p_model", "p_market")},
                         american=sdy.american(l["decimal"]), reason=sdy.reason(l), outcome=l.get("outcome")))
    status = row.get("status", "pending")
    if status == "pending":
        status = "locked" if locked(row, now) else "open"
    return {"day": day, "status": status, "legs": legs, "multiplier": row.get("multiplier"),
            "p": row.get("p"), "p_market": row.get("p_market"), "sports": row.get("sports"),
            "first_start": row.get("first_start"), "last_start": row.get("last_start"),
            "built_at": row.get("built_at"), "paid": row.get("paid"), "profit": row.get("profit"),
            "missing_reason": None}


def payload(history, show, now, graded):
    bt = _load(BACKTEST)
    measured = None
    caveats = []
    if bt:
        measured = {"rule": bt["results"]["same_day"], "market_only": bt["results"]["market_only"],
                    "periods": bt["periods"], "multipliers": bt["splits"]["same_day"]["multipliers"]}
        caveats = list(bt.get("caveats") or [])
    else:
        caveats = ["Not measured yet: the backtest (parlays/engine/experiments/sameday_backtest.py) "
                   "has not been run, so no record is shown. The market is hard to beat."]
    caveats.append("Prices are the ones on file when the sports' runners last ran: UFC the best price seen "
                   "across books, NFL the consensus moneyline, soccer the average posted price. One book "
                   "may pay less. Check every price before betting; a leg whose game has started is never used.")
    return {"generated": _iso(now), "day_tz": sdy.DAY_TZ,
            "rule": {"band": list(sdy.BAND), "max_legs": sdy.MAX_LEGS, "pool": sdy.POOL, "min_p": sdy.MIN_P},
            "tickets": [ticket_payload(d, latest(history, d), now) for d in show],
            "record": record(graded), "measured": measured, "caveats": caveats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", help="ISO time to build as of (testing)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    now = _parse(args.now) if args.now else datetime.now(timezone.utc)
    history = _load(HISTORY, None) or {"schema": 1, "tickets": []}
    legs = live_legs(now)
    show = update(history, now, legs)
    graded = grade_all(history)
    out = payload(history, show, now, graded)
    text = json.dumps(out, indent=1, allow_nan=False)
    if args.no_write:
        print(text)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    HISTORY.write_text(json.dumps(history, indent=1, allow_nan=False))
    for t in out["tickets"]:
        legs = ", ".join(f"{l['sport']} {l['pick']} {l['decimal']:.2f}" for l in t["legs"])
        print(f"{t['day']}: {t['status']} {t['multiplier'] or ''} {legs or t['missing_reason']}")


if __name__ == "__main__":
    main()
