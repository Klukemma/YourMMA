"""The soccer app's data files: matches.json, tables.json, performance.json.

Shapes are the contract with soccer/app/app.js; tests/test_soccer_export.py
pins them. The other apps' export rules carry over unchanged:

  * NaN and infinity become null (a bare NaN makes JSON.parse throw on the
    phone - a blank screen). A missing number prints as a dash, never as 0.
  * Every payload carries `schema` and `generated`.
  * Every caveat the reader sees is written here, in one place, and every
    number in one is read from the measured backtest - never typed in.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import soccer_data as sd
import soccer_form as sf
import soccer_teams as st

APP = sd.SOCCER / "app"
APP_DATA = APP / "data"
SCHEMA = 1

# code: (short label for the filter, the name)
LEAGUE_SHORT = {"E0": "ENG", "SP1": "ESP", "I1": "ITA", "D1": "GER", "F1": "FRA"}


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write(name, payload, out_dir=APP_DATA):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = dict(payload, schema=SCHEMA, generated=payload.get("generated") or now_stamp())
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(_clean(body), indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def now_stamp(now=None):
    t = now or datetime.now(timezone.utc)
    if getattr(t, "tzinfo", None) is not None:
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def leagues():
    return {code: {"name": sd.LEAGUES[code][0], "country": sd.LEAGUES[code][1], "short": LEAGUE_SHORT[code]}
            for code in sd.TOP}


def pct(x, d=1):
    return "—" if x is None or x != x else f"{x * 100:.{d}f}%"


def caveats(bt, basis="blend"):
    """What the reader is told, with every number read from the backtest.

    basis "market": the pooling gave the model no weight, so the pick is
    the market's chances recalibrated on earlier seasons, and is called
    that - the model's own chances are shown beside it, labelled."""
    res = (bt or {}).get("result", {})
    conf = (bt or {}).get("periods", {}).get("confirm", [2016, 2025])
    key = f"{conf[0]}-{conf[1]}"
    blend = (res.get("blend") or {}).get(key, {})
    model = (res.get("model") or {}).get(key, {})
    value = (bt or {}).get("value", {})
    lead = ("The pick is the betting market's chances, recalibrated on earlier seasons: across "
            f"{key} the model alone picked {pct(model.get('accuracy'))} of results and the market "
            f"{pct(blend.get('market_accuracy'))}, and blending the model in added nothing the market "
            "did not already know. The model's own chances are shown beside each pick, labelled"
            if basis == "market" else
            f"The pick is the model blended with the betting market, which is hard to beat at soccer: "
            f"across {key} the blend picked {pct(blend.get('accuracy'))} of results against the betting "
            f"favourite's {pct(blend.get('market_accuracy'))}")
    out = {
        "model": (
            lead
            + (". Betting the model where it disagreed with the price did not make money. "
               if value.get("enabled_at") is None else ". ")
            + "Treat a pick as an opinion with a measured hit rate, not an edge."),
        "draws": (
            "A draw is rarely the single likeliest result, so it is rarely the pick - even though about "
            "a quarter of matches end level. Its chance is always shown."),
        "news": (
            "The model does not read team news - injuries, suspensions, rotation. The betting market "
            "does, which is part of why the pick leans on it."),
    }
    out["rating_short"] = ("The goal difference a match each club would expect against an average club "
                           "of its league on a neutral pitch, by the model's ratings, strongest first.")
    out["rating"] = ("The model rates each club's attack and defence from its goals, shots on target and "
                     "shots, weighted toward recent matches; a promoted club carries its rating up with a "
                     "measured step. The rating is the model's own view - the pick on each match is the "
                     "market's chances, recalibrated, because the market forecasts better.")
    return out


def club(key):
    return st.display(key)


def form_payload(matches, team, before, n=5):
    rows = sf.form(matches, team, before, n)
    for r in rows:
        r["opponent_name"] = st.display(r["opponent"])["name"]
        r["league_name"] = sd.LEAGUES.get(r["league"], (r["league"],))[0]
    return rows


def positions(matches, league, season, before):
    return {r["team"]: r["position"] for r in sf.table(matches, league, season, before)
            if r["played"] > 0}


def tables_payload(matches, season, ratings=None, generated=None, caveat=None, season_label=None):
    """Each top league's table as it stands, with each club's last five and
    the model's rating where it has one."""
    out = []
    for code in sd.TOP:
        rows = sf.table(matches, code, season)
        if not rows:
            continue
        for r in rows:
            r["club"] = club(r["team"])
            r["form"] = sf.form(matches, r["team"], "9999-12-31")
            r["rating"] = (ratings or {}).get(r["team"])
        out.append({"code": code, "name": sd.LEAGUES[code][0], "short": LEAGUE_SHORT[code], "rows": rows})
    return {"season": season, "season_label": season_label, "leagues": out, "generated": generated,
            "caveats": caveat or {}}
