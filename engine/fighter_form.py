"""A fighter's pro MMA record and last five results, for the card.

Shown next to each name in the app, for the reader - none of it is a model
input. Two sources, in order:

  Sherdog   the fighter's whole professional record (UFC and everything
            before or outside it), harvested by harvest_sherdog.py into
            data/sherdog_ufc_records.jsonl.gz, keyed by the dataset's name.
            A record found by name alone (no known bout to check it
            against - a UFC debutant) is marked unverified and the app says
            so, because a namesake's record is worse than no record.
  Archive   the UFC bouts in the dataset, used to extend a Sherdog record
            with bouts fought after it was looked up, and on their own
            (marked UFC-only) when Sherdog has nothing.

POINT IN TIME. Only bouts dated strictly before the card count, so a record
harvested after a fight can never show its result on that fight's card.

Each of the last five is a result (W, L, D, NC) and a two-character code:
K<round> knockout or TKO, S<round> submission, DC decision, DQ
disqualification, NC no contest, ?? when the method is not stated.
"""

import gzip
import json
from datetime import date
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
SHERDOG = ENGINE / "data" / "sherdog_ufc_records.jsonl.gz"
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"

LAST_N = 5
# Two bouts this close on the same pairing are one bout reported twice (the
# archive and Sherdog can disagree by a day across time zones).
SAME_BOUT_DAYS = 2


def method_code(method, rnd=None):
    """Two characters for how a bout ended: K1, S2, DC, DQ, NC or ??.

    Classified on the words before any parenthesis first: Sherdog's
    "TKO (Submission to Punches)" is a TKO - the opponent tapped to strikes -
    and must not read as a submission.
    """
    m = str(method or "").strip().lower()
    head = m.split("(")[0].strip()
    r = str(rnd or "").strip()
    r = r if r.isdigit() and r != "0" else ""
    if not m or m in ("n/a", "nd"):
        return "??"
    if (head.startswith(("no contest", "no decision", "overturned",
                         "could not continue", "cnc"))
            or head in ("nc", "nd") or m.startswith(("nd ", "nd(", "nc ", "nc("))):
        return "NC"
    if head.startswith(("dq", "disqualification")):
        return "DQ"
    if (head.startswith(("ko", "tko", "k.o", "technical knockout"))
            or "ko/tko" in head or "knockout" in head):
        return f"K{r}" if r else "K?"
    if head.startswith(("submission", "sub", "technical submission",
                        "techinal sub")):
        return f"S{r}" if r else "S?"
    if ("decision" in head or "draw" in head
            or head.startswith(("u-dec", "s-dec", "m-dec", "dec"))):
        return "DC"
    # Nothing recognisable before a parenthesis: fall back to the whole text.
    if "submission" in m:
        return f"S{r}" if r else "S?"
    if "decision" in m:
        return "DC"
    return "??"


_RESULT = {"win": "W", "loss": "L", "draw": "D", "nc": "NC"}


def _day(text):
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _surname(name):
    parts = str(name or "").lower().replace(".", "").split()
    while len(parts) > 1 and parts[-1] in ("jr", "sr", "ii", "iii", "iv"):
        parts.pop()
    return parts[-1] if parts else ""


def load_sherdog(path=SHERDOG):
    """{dataset name: harvested row} for every fighter found on Sherdog."""
    path = Path(path)
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["target"]: row for row in rows if row.get("id")}


def archive_bouts(archive, name):
    """The fighter's UFC bouts from the dataset, in Sherdog's shape."""
    out = []
    for side, other in (("r", "b"), ("b", "r")):
        mine = archive[archive[f"{side}_name"] == name]
        for row in mine.itertuples(index=False):
            winner = getattr(row, "winner")
            opponent = getattr(row, f"{other}_name")
            if isinstance(winner, str) and winner.strip():
                result = ("win" if winner == name else
                          "loss" if winner == opponent else "nc")
            else:
                method = str(getattr(row, "method") or "").lower()
                result = "draw" if "draw" in method or "dec" in method else "nc"
            out.append({"date": str(getattr(row, "date"))[:10],
                        "opponent": opponent,
                        "event": getattr(row, "event_name", "") or "",
                        "method": getattr(row, "method"),
                        "round": getattr(row, "finish_round", None),
                        "result": result,
                        "source": "ufc"})
    return out


def _merge(sherdog, ufc, looked_up):
    """Sherdog's record plus the UFC bouts fought AFTER it was read.

    The Sherdog record was complete on the day it was looked up, so a UFC
    bout dated before that is either already in it - under another spelling
    of the opponent ("Sopai"/"Sopaj", "Darya"/"Daria") - or belongs to a
    namesake sharing the dataset's name (the archive's "Bruno Silva" is two
    men). Adding either would put a bout in the record that the fighter did
    not have, so neither is added. Only bouts after the lookup are, once
    each: nobody fights twice in two days.
    """
    out = [dict(b, source="sherdog") for b in sherdog]
    read = _day(looked_up)
    for bout in ufc:
        day = _day(bout["date"])
        if day is None or read is None or day <= read:
            continue
        if any(_day(s.get("date")) is not None
               and abs((_day(s["date"]) - day).days) <= SAME_BOUT_DAYS
               for s in out):
            continue
        out.append(bout)
    return out


def form(name, before, sherdog=None, archive=None):
    """{record, last5, source, verified, as_of} for one fighter, using only
    bouts dated strictly before `before`. None when nothing is known."""
    cutoff = _day(before)
    if cutoff is None:
        return None
    row = (sherdog or {}).get(name)
    ufc = archive_bouts(archive, name) if archive is not None else []
    if row:
        bouts, source = _merge(row.get("record") or [], ufc, row.get("looked_up")), "sherdog"
    elif ufc:
        bouts, source = ufc, "ufc"
    else:
        return None
    bouts = [b for b in bouts
             if _day(b.get("date")) is not None and _day(b["date"]) < cutoff]
    if not bouts:
        return None
    bouts.sort(key=lambda b: b["date"], reverse=True)
    tally = {"W": 0, "L": 0, "D": 0, "NC": 0}
    for b in bouts:
        tally[_RESULT.get(str(b.get("result")).lower(), "NC")] += 1
    last = [{"result": _RESULT.get(str(b.get("result")).lower(), "NC"),
             "code": method_code(b.get("method"), b.get("round")),
             "date": str(b["date"])[:10],
             "opponent": b.get("opponent"),
             "event": b.get("event"),
             "method": None if b.get("method") is None else str(b.get("method"))}
            for b in bouts[:LAST_N]]
    return {"record": {"wins": tally["W"], "losses": tally["L"],
                       "draws": tally["D"], "nc": tally["NC"]},
            "last5": last,
            "source": source,
            "verified": bool(row.get("verified")) if row else True,
            "as_of": (row or {}).get("looked_up")}


def load_archive(path=ARCHIVE):
    cols = ["date", "r_name", "b_name", "winner", "method", "finish_round",
            "event_name"]
    return pd.read_csv(path, usecols=cols, low_memory=False)
