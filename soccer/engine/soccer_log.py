"""The soccer prediction log: every pick written down before kickoff, and
graded once the result is in.

A PICK IS LOGGED ONLY BEFORE KICKOFF. Re-running before a match updates its
entry (the last line before kickoff is the one that counts); once the match
has started the entry is frozen, and a match that started before any run
logged it shows its result with no pick rather than a pick made afterwards.

A soccer pick is one of three results - home win, draw, away win - so a
graded pick is simply right or wrong; there is no push. The over/under 2.5
goals call is graded beside it.
"""

import json
import math
import os
from pathlib import Path

import pandas as pd

import soccer_data as sd

LOG = Path(os.environ.get("SOCCER_PREDICTIONS_LOG") or (sd.DATA / "prediction_history.json"))
SCHEMA = 1
FIELDS = ("match_id", "league", "season", "kickoff", "home", "away",
          "pick", "pick_prob", "p_h", "p_d", "p_a", "model", "market", "prices",
          "p_o25", "model_o25", "market_o25", "tier", "price_source", "prices_at")


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


def log_pick(log, match, now):
    """Upsert one pick while its match has not started. Returns True if
    written."""
    if match.get("kickoff") is None or naive_utc(match["kickoff"]) <= naive_utc(now):
        return False
    entry = log["picks"].get(match["match_id"], {})
    keep = {"first_logged": entry.get("first_logged") or stamp(now)}
    log["picks"][match["match_id"]] = dict({k: match.get(k) for k in FIELDS}, **keep, last_logged=stamp(now))
    return True


def grade(log, matches):
    """Fill in the result of every logged pick whose match has one."""
    done = matches[matches["result"].isin(("H", "D", "A"))].set_index("match_id")
    graded = 0
    for mid, p in log["picks"].items():
        if p.get("final") or mid not in done.index:
            continue
        m = done.loc[mid]
        hg, ag = int(m["hg"]), int(m["ag"])
        p["final"] = {"home": hg, "away": ag, "result": m["result"]}
        p["correct"] = p.get("pick") == m["result"] if p.get("pick") else None
        if p.get("p_o25") is not None:
            p["o25_correct"] = (p["p_o25"] >= 0.5) == (hg + ag > 2.5)
        graded += 1
    return graded


def record(log):
    """Live record: settled picks and how many were right, plus over/under."""
    settled = [p for p in log["picks"].values() if p.get("final") and p.get("correct") is not None]
    ou = [p for p in log["picks"].values() if p.get("final") and p.get("o25_correct") is not None]
    return {"settled": len(settled), "correct": sum(1 for p in settled if p["correct"]),
            "ou_settled": len(ou), "ou_correct": sum(1 for p in ou if p["o25_correct"]),
            "pending": sum(1 for p in log["picks"].values() if not p.get("final"))}
