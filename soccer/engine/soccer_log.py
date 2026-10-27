"""The soccer prediction log: every pick written down before kickoff, and
graded once the result is in.

A PICK IS LOGGED ONLY BEFORE KICKOFF. Re-running before a match updates its
entry (the last line before kickoff is the one that counts); once the match
has started the entry is frozen, and a match that started before any run
logged it shows its result with no pick rather than a pick made afterwards.

A soccer pick is one of three results - home win, draw, away win - so a
graded pick is simply right or wrong; there is no push. The over/under 2.5
goals call is graded beside it, and so is both teams to score (the side its
chance favoured, from the score matrix soccer_scores builds) - logged with
over 1.5 / 3.5, the three likeliest scores, where those numbers came from
and the pick's band, so their live record starts with the first pick.

WHEN A PICK FREEZES. At `freeze_at` when the entry has one, else at
kickoff. soccer_schedule sets it: the kickoff where the time is confirmed,
00:00 UK time on the match's UK date where it is not (a feed's time that the
two feeds do not agree on), so a pick is never logged after an earlier
kickoff than the one listed. Once a logged pick's own freeze time has
passed it never changes, whatever the match's time becomes later.

NEVER GRADED IF LOGGED AFTER KICKOFF. A result row that carries its kickoff
(football-data.co.uk's, or the feeds') is checked against the pick's last
log: a pick written at or after the match's real kickoff (a match brought
forward that no source had moved yet) is void, "logged after kickoff" -
point in time, whatever the reason.

A MOVED MATCH. A match id carries its date, so a postponed match comes back
under a new one. A pick is graded against its own match found by id or,
failing that, by its league, season and clubs (a pairing that is unique in a
league season). When the moved match is picked again under its new id, the
older pick is marked void - one match, one pick in the record - and a pick
whose match never produced a result is voided once it is STALE days past
kickoff. Void picks are neither settled nor pending. An older pick for the
same pairing that has NOT frozen yet is the same pick under an id that has
since changed (a feed's date before football-data.co.uk posted its own): it
moves to the new id ("id_was"), keeping when it was first logged; it is
not voided.

PROVISIONAL RESULTS. A result from the free feeds (soccer_schedule) for a
match football-data.co.uk has not posted grades its pick PROVISIONALLY:
final.provisional is true, with the feeds' names and when they showed it.
When football-data.co.uk posts the result, it is the result: the same score
confirms the grade (the provisional flag goes; final.provisional_was keeps
the record of it), a different one regrades the pick and is written down in
`corrections`. A provisional grade the feeds later contradict each other on
is withdrawn - the pick is pending again - and that is written down too.
"""

import json
import math
import os
from pathlib import Path

import pandas as pd

import soccer_data as sd

LOG = Path(os.environ.get("SOCCER_PREDICTIONS_LOG") or (sd.DATA / "prediction_history.json"))
SCHEMA = 1
STALE_DAYS = 21
FIELDS = ("match_id", "league", "season", "kickoff", "home", "away",
          "pick", "pick_prob", "p_h", "p_d", "p_a", "model", "market", "prices",
          "p_o25", "model_o25", "market_o25", "tier", "price_source", "prices_at",
          "freeze_at", "time_confirmed", "fixture_source",
          # the numbers beside the pick (soccer_scores): both teams to score,
          # over 1.5 / 3.5, the three likeliest scores with the matrix's own
          # 1X2 and why it is the model's where it is, its source, the band
          "p_btts", "p_o15", "p_o35", "scores", "score_source", "band", "band_basis")
GRADES = ("final", "correct", "o25_correct", "btts_correct")


def naive_utc(t):
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def stamp(t):
    return naive_utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(x):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def load(path=LOG):
    path = Path(path)
    if not path.exists():
        return {"schema": SCHEMA, "picks": {}}
    log = json.loads(path.read_text())
    if log.get("schema") != SCHEMA:
        raise ValueError(f"prediction log is schema {log.get('schema')}, expected {SCHEMA}")
    return log


def save(log, path=LOG):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_clean(log), indent=1, sort_keys=True, allow_nan=False) + "\n")


def pairing(p):
    return (p.get("league"), int(p["season"]) if p.get("season") is not None else None, p.get("home"), p.get("away"))


def frozen(p, now):
    """Whether a pick (or a match about to be logged) can no longer change."""
    t = p.get("freeze_at") or p.get("kickoff")
    return t is None or naive_utc(t) <= naive_utc(now)


def log_pick(log, match, now):
    """Upsert one pick while it has not frozen. Returns True if written. A
    pick on file whose own freeze time has passed stays as it is, even if the
    match's time has since moved later (an unconfirmed time that froze it at
    midnight, then confirmed for the afternoon): frozen is frozen. A pending
    pick for the same pairing under another id: moved to this id if it has
    not frozen either (the same match, its id since changed), voided if it
    has (the match was moved)."""
    if match.get("kickoff") is None or frozen(match, now):
        return False
    mid = match["match_id"]
    entry = log["picks"].get(mid, {})
    if entry and frozen(entry, now):
        return False
    same = [k for k, p in log["picks"].items() if k != mid and not p.get("final") and not p.get("void")
            and pairing(p) == pairing(match)]
    moved = [k for k in same if not frozen(log["picks"][k], now)]
    keep = {"first_logged": entry.get("first_logged")}
    for k in moved:
        old = log["picks"].pop(k)
        firsts = [t for t in (keep["first_logged"], old.get("first_logged")) if t]
        keep["first_logged"] = min(firsts) if firsts else None
        keep["id_was"] = entry.get("id_was") or k
    if entry.get("id_was") and "id_was" not in keep:
        keep["id_was"] = entry["id_was"]
    keep["first_logged"] = keep["first_logged"] or stamp(now)
    log["picks"][mid] = dict({k: match.get(k) for k in FIELDS}, **keep, last_logged=stamp(now))
    for k in same:
        if k not in moved:
            log["picks"][k]["void"] = "rescheduled"
    return True


def _index(results):
    """{match_id: row}, {pairing: row} for the rows of `results` with a result."""
    if results is None or not len(results):
        return {}, {}
    done = results[results["result"].isin(("H", "D", "A"))]
    by_id = {r.match_id: r for r in done.itertuples()}
    cols = {"league", "season", "home", "away"}
    by_pair = ({(r.league, int(r.season), r.home, r.away): r for r in done.itertuples() if r.league is not None}
               if cols <= set(done.columns) else {})
    return by_id, by_pair


def _final(r, p):
    out = {"home": int(r.hg), "away": int(r.ag), "result": r.result}
    played_on = getattr(r, "date", None)
    if played_on is not None and not (isinstance(played_on, float) and math.isnan(played_on)) \
            and str(played_on) != str(p.get("kickoff", ""))[:10]:
        out["played_on"] = str(played_on)
    return out


def _settle(p, final):
    p["final"] = final
    p["correct"] = p.get("pick") == final["result"] if p.get("pick") else None
    if p.get("p_o25") is not None:
        p["o25_correct"] = (p["p_o25"] >= 0.5) == (final["home"] + final["away"] > 2.5)
    else:
        p.pop("o25_correct", None)
    if p.get("p_btts") is not None:
        p["btts_correct"] = (p["p_btts"] >= 0.5) == (final["home"] > 0 and final["away"] > 0)
    else:
        p.pop("btts_correct", None)


def _was(p):
    f = p.get("final") or {}
    return {"home": f.get("home"), "away": f.get("away"), "result": f.get("result"), "correct": p.get("correct"),
            "provisional": bool(f.get("provisional"))}


def _logged_late(p, r):
    """Whether the pick's last log is at or after the match's real kickoff,
    where the result row says when that was."""
    kick, last = getattr(r, "kickoff", None), p.get("last_logged")
    if kick is None or last is None or (not isinstance(kick, str) and pd.isna(kick)):
        return False
    return naive_utc(last) >= naive_utc(kick)


def _void_late(p, at, by):
    """Void a pick logged after its match kicked off; a grade it had (a
    provisional one) is written down as withdrawn."""
    if p.get("final"):
        before = _was(p)
        for k in GRADES:
            p.pop(k, None)
        p.setdefault("corrections", []).append({"at": at, "by": by, "was": before, "now": None})
    p["void"] = "logged after kickoff"


def grade(log, matches, now=None, provisional=None, disputed=(), stats=None):
    """Fill in the result of every logged pick whose match has one - found by
    id, or by pairing when the match was moved - and void a pick whose match
    has produced no result STALE_DAYS after its kickoff.

    `matches` are football-data.co.uk's results, the record; `provisional`
    the feeds' results for matches it has not posted (soccer_schedule), which
    grade a pick provisionally until it does; `disputed` the pairings whose
    feed scores now contradict each other. Returns how many picks were
    graded for the first time; `stats`, if given, also counts the provisional
    grades confirmed, corrected and withdrawn, and the picks voided as logged
    after kickoff ("late")."""
    by_id, by_pair = _index(matches)
    pv_id, pv_pair = _index(provisional)
    at = stamp(now) if now is not None else None
    count = {"graded": 0, "provisional": 0, "confirmed": 0, "corrected": 0, "withdrawn": 0, "late": 0}
    for mid, p in log["picks"].items():
        fin = p.get("final")
        if p.get("void") or (fin and not fin.get("provisional")):
            continue
        r = by_id.get(mid)
        if r is None:
            r = by_pair.get(pairing(p))
        if r is not None and _logged_late(p, r):
            _void_late(p, at, "football-data.co.uk")
            count["late"] += 1
            continue
        if r is not None:                                   # football-data.co.uk: the record
            new = _final(r, p)
            if fin:
                same = (fin["home"], fin["away"]) == (new["home"], new["away"])
                new["provisional_was"] = {k: fin.get(k) for k in ("home", "away", "source", "seen_at", "graded_at")}
                new["provisional_was"]["confirmed"] = same
                if not same:
                    before = _was(p)
                    _settle(p, new)
                    p.setdefault("corrections", []).append({"at": at, "by": "football-data.co.uk",
                                                            "was": before, "now": _was(p)})
                    count["corrected"] += 1
                else:
                    _settle(p, new)
                    count["confirmed"] += 1
            else:
                _settle(p, new)
                count["graded"] += 1
            continue
        r = pv_id.get(mid)
        if r is None:
            r = pv_pair.get(pairing(p))
        if r is not None and _logged_late(p, r):
            _void_late(p, at, getattr(r, "source", None) or "the feeds")
            count["late"] += 1
            continue
        if r is not None:                                   # the feeds: provisional
            new = dict(_final(r, p), provisional=True, source=getattr(r, "source", None),
                       seen_at=getattr(r, "seen_at", None), graded_at=at)
            if not fin:
                _settle(p, new)
                count["graded"] += 1
                count["provisional"] += 1
            elif (fin["home"], fin["away"]) != (new["home"], new["away"]):
                before = _was(p)
                _settle(p, new)
                p.setdefault("corrections", []).append({"at": at, "by": new["source"], "was": before, "now": _was(p)})
                count["corrected"] += 1
            continue
        if fin and pairing(p) in disputed:                  # the feeds now contradict each other
            before = _was(p)
            for k in GRADES:
                p.pop(k, None)
            p.setdefault("corrections", []).append({"at": at, "by": "the feeds disagree", "was": before,
                                                    "now": None})
            count["withdrawn"] += 1
            continue
        if not fin and now is not None and p.get("kickoff") is not None and \
                naive_utc(p["kickoff"]) < naive_utc(now) - pd.Timedelta(days=STALE_DAYS):
            p["void"] = "no result"
    if stats is not None:
        stats.update(count)
    return count["graded"]


def record(log):
    """Live record: settled picks and how many were right, plus over/under
    2.5 and both teams to score (the side each chance favoured). Void picks
    are neither settled nor pending. A provisional grade counts as settled,
    and how many settled picks are provisional is said beside it."""
    live = [p for p in log["picks"].values() if not p.get("void")]
    settled = [p for p in live if p.get("final") and p.get("correct") is not None]
    ou = [p for p in live if p.get("final") and p.get("o25_correct") is not None]
    bt = [p for p in live if p.get("final") and p.get("btts_correct") is not None]
    return {"settled": len(settled), "correct": sum(1 for p in settled if p["correct"]),
            "ou_settled": len(ou), "ou_correct": sum(1 for p in ou if p["o25_correct"]),
            "btts_settled": len(bt), "btts_correct": sum(1 for p in bt if p["btts_correct"]),
            "pending": sum(1 for p in live if not p.get("final")),
            # graded on the feeds' result, football-data.co.uk's not yet in
            "provisional": sum(1 for p in settled if p["final"].get("provisional")),
            "corrected": sum(1 for p in live if p.get("corrections"))}
