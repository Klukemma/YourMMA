"""The NFL app's data files: week.json, teams.json, performance.json.

Shapes are the contract with nfl/app/app.js; tests/test_nfl_export.py pins
them. The MMA export's rules carry over unchanged:

  * NaN and infinity become null (allow_nan=False would otherwise crash the
    build, and a bare NaN makes JSON.parse throw on the phone - a blank
    screen). A missing number prints as a dash, never as 0 or 50%.
  * Every payload carries `schema` and `generated`.
  * Every caveat the reader sees is written here, in one place.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_data as nd

APP = nd.NFL / "app"
APP_DATA = APP / "data"
SCHEMA = 1

CAVEATS = {
    "model": (
        "The pick is the model blended with the betting market. The market is "
        "hard to beat in the NFL: across 2016-2025 the blend picked {acc} of "
        "winners against the closing favourite's {mkt}, and betting the model "
        "where it disagreed with the price lost money. Treat a pick as an "
        "opinion with a measured hit rate, not an edge."),
    "tiers": (
        "STRONG, SOLID, LEAN and TOSS-UP are bands of the pick's probability. "
        "The percentage beside each is how often picks in that band won in "
        "2016-2025 - measured, not promised."),
    "value_off": (
        "No game is labelled a bet: backing the model where it disagreed with "
        "the closing moneyline did not make money in 2016-2025 at any edge "
        "tested."),
    "lines": (
        "Model line and model total are the model's own numbers beside the "
        "market's. Taking the model's side against the closing spread won "
        "{ats} of 2016-2025 games and against the total {tot}; -110 needs "
        "52.4% to break even."),
    "injuries": (
        "Injury report from the league's official practice reports via "
        "nflverse: shown, not applied. The model reads only who starts at "
        "quarterback."),
    "qb": (
        "The quarterback the model assumes is the one the schedule projects "
        "to start. If the injury report says he is out, the number above does "
        "not know yet."),
}


def _clean(value):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python, floats rounded."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if value is pd.NaT:
        return None
    return value


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(name, payload, out_dir=APP_DATA):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _clean(dict(payload, schema=SCHEMA, generated=_now()))
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def pct(x):
    return "—" if x is None or x != x else f"{x * 100:.1f}%"


def caveats(backtest):
    confirm = f"{backtest['periods']['confirm'][0]}-{backtest['periods']['confirm'][1]}"
    blend = backtest["winner"]["blend"][confirm]
    spread = backtest["spread"][confirm]["0"]["hit_rate"]
    total = backtest["total"][confirm]["0"]["hit_rate"]
    out = dict(CAVEATS)
    out["model"] = out["model"].format(acc=pct(blend["accuracy"]), mkt=pct(blend["market_accuracy"]))
    out["lines"] = out["lines"].format(ats=pct(spread), tot=pct(total))
    if backtest["value"]["enabled_at"] is not None:
        out.pop("value_off")
    return out


def team_entry(code, state, record, rank, qb_names):
    name, conf, div, _ = nd.TEAMS[code]
    qb = state["last_qb"].get(code)
    return {
        "team": code, "name": name, "conference": conf, "division": f"{conf} {div}",
        "elo": state["elo"].get(code), "rank": rank,
        "off_epa": state["off_epa"].get(code), "def_epa": state["def_epa"].get(code),
        "net_epa": (state["off_epa"].get(code, 0) - state["def_epa"].get(code, 0)),
        "points_for": state["points_for"].get(code), "points_against": state["points_against"].get(code),
        "qb": qb_names.get(qb), "qb_rating": state["qb_rating"].get(qb),
        "qb_starts": state["qb_starts"].get(qb),
        "record": record,
    }


def track_record(log):
    """The live record of picks logged before kickoff, graded."""
    picks = list(log.get("picks", {}).values())
    graded = [p for p in picks if p.get("correct") is not None]
    ats = [p for p in picks if p.get("spread_correct") is not None]
    tot = [p for p in picks if p.get("total_correct") is not None]
    seasons = {}
    for p in graded:
        s = seasons.setdefault(int(p["season"]), {"season": int(p["season"]), "picks": 0, "correct": 0})
        s["picks"] += 1
        s["correct"] += int(bool(p["correct"]))
    weeks = {}
    for p in graded:
        key = (int(p["season"]), int(p["week"]))
        w = weeks.setdefault(key, {"season": key[0], "week": key[1], "picks": 0, "correct": 0})
        w["picks"] += 1
        w["correct"] += int(bool(p["correct"]))
    ll = [-(math.log(p["pick_prob"]) if p["correct"] else math.log(1 - p["pick_prob"]))
          for p in graded if p.get("pick_prob") and 0 < p["pick_prob"] < 1]
    return {
        "logged": len(picks),
        "graded": len(graded),
        "correct": sum(1 for p in graded if p["correct"]),
        "accuracy": (sum(1 for p in graded if p["correct"]) / len(graded)) if graded else None,
        "log_loss": (sum(ll) / len(ll)) if ll else None,
        "pending": sum(1 for p in picks if p.get("correct") is None and not p.get("void")),
        "ats": {"picks": len(ats), "correct": sum(1 for p in ats if p["spread_correct"])},
        "totals": {"picks": len(tot), "correct": sum(1 for p in tot if p["total_correct"])},
        "by_season": sorted(seasons.values(), key=lambda r: r["season"]),
        "by_week": sorted(weeks.values(), key=lambda r: (r["season"], r["week"])),
        "basis": "one pick per game, the last one logged before kickoff",
    }
