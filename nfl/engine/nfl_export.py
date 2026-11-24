"""The NFL app's data files: week.json, teams.json, performance.json.

Shapes are the contract with nfl/app/app.js; tests/test_nfl_export.py pins
them. The MMA export's rules carry over unchanged:

  * NaN and infinity become null (allow_nan=False would otherwise crash the
    build, and a bare NaN makes JSON.parse throw on the phone - a blank
    screen). A missing number prints as a dash, never as 0 or 50%.
  * Every payload carries `schema` and `generated`.
  * Every caveat the reader sees is written here, in one place - the
    same-game tickets' words are assembled here (sgp_block) from the rule's
    own wording in nfl_sgp and nfl_sgp2, so the app and the rule cannot
    drift apart.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_data as nd
import nfl_odds as no
import nfl_sgp as sgp
import nfl_sgp2 as sgp2

APP = nd.NFL / "app"
APP_DATA = APP / "data"
SCHEMA = 1
SGP_HISTORY = nd.DATA / "sgp_backtest.json"      # experiments/nfl_sgp_backtest.py writes it
SGP2_HISTORY = nd.DATA / "sgp2_backtest.json"    # experiments/nfl_sgp2_backtest.py writes it

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
    "players": (
        "Most valuable on offence is ranked by expected points added (EPA) this "
        "season - the points a player's plays added above an average play in "
        "the same situation; defenders by sacks, takeaways, passes defended and "
        "tackles for loss. A listed player who is out or doubtful - or the most "
        "valuable player questionable - is flagged at the top of the game, from "
        "the official injury report and the weekly roster. Shown, not applied: "
        "only the starting quarterback moves the pick."),
    "news": (
        "Headlines from public NFL news feeds that name one of the two teams or "
        "their players, published before kickoff. Shown, not applied."),
    "qb": (
        "The quarterback the model assumes is the one the schedule projects "
        "to start. If the injury report says he is out, the number above does "
        "not know yet."),
    # Beside each scorer's odds in the TD lists (nfl_odds.td_price).
    "td_odds": no.TD_ODDS_NOTE,
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


def caveats(backtest, td=None, props=None):
    confirm = f"{backtest['periods']['confirm'][0]}-{backtest['periods']['confirm'][1]}"
    blend = backtest["winner"]["blend"][confirm]
    spread = backtest["spread"][confirm]["0"]["hit_rate"]
    total = backtest["total"][confirm]["0"]["hit_rate"]
    out = dict(CAVEATS)
    out["model"] = out["model"].format(acc=pct(blend["accuracy"]), mkt=pct(blend["market_accuracy"]))
    out["lines"] = out["lines"].format(ats=pct(spread), tot=pct(total))
    if backtest["value"]["enabled_at"] is not None:
        out.pop("value_off")
    if td:
        period = [k for k in td if k[:2] == "20"][-1]
        top = td[period]["top_pick"][td["model"]]
        out["td"] = (
            f"Chance of a rushing or receiving touchdown, if he plays - from his recent carries, "
            f"targets and snaps and his team's expected points from the betting line. In {period} "
            f"each team's most likely scorer scored {pct(top['hit_rate'])} of the time "
            f"(the model said {pct(top['mean_prob'])}). Measured, and shown, not applied.")
        groups = td.get("by_usage") or {}
        hidden = [k for k, ok in (groups.get("ships") or {}).items() if not ok]
        dev = "-".join(str(y) for y in (td.get("periods") or {}).get("dev", ("", "")))
        if hidden:
            said = []
            for name in hidden:
                # The period where the group missed most is the one that keeps it off.
                measured = [(k, r) for k, r in ((k, (groups.get(k) or {}).get(name)) for k in (dev, period))
                            if r and r.get("ratio") is not None]
                if measured:
                    k, r = max(measured, key=lambda kr: abs(kr[1]["ratio"] - 1))
                    word = "no NFL game" if name == "no history" else f"{name} weighted carries + targets a game"
                    said.append(f"{word}: it said {pct(r['predicted'])} where {pct(r['observed'])} scored ({k})")
            out["td_list"] = (
                "Every active QB, RB, FB, WR and TE is listed - everyone you can back to score. A chance "
                "is shown only where the model was measured as well calibrated: within "
                f"{groups.get('tolerance', 0.2):.0%} of what happened in both {dev} and {period}. "
                "Players with little use or no NFL game are listed with a dash instead"
                + (". For players with " + "; ".join(said) if said else "") + ".")
    out["td_meters"] = TD_METERS_DEFINITION
    if props:
        period = [k for k in props if k[:2] == "20"][-1]
        parts = []
        for stat, word in (("pass", "passing"), ("rush", "rushing"), ("rec", "receiving")):
            mae = (props[period].get(stat) or {}).get("mae") or {}
            chosen, own = mae.get(props["model"][stat]), mae.get("mean")
            if chosen is not None and own is not None:
                parts.append(f"{word} {chosen:.0f} yards off on average (his own average: {own:.0f})")
        out["props"] = (
            "Projected yards are the middle of what the player's recent usage and his team's "
            f"expected points point to, if he plays. In {period}: " + "; ".join(parts) +
            ". Players went over about half the time, as a middle number should. Shown, not applied.")
    return out


def team_entry(code, state, record, rank, qb_names):
    name, conf, div, _ = nd.TEAMS[code]
    qb = state["last_qb"].get(code)
    off, dfn = state["off_epa"].get(code), state["def_epa"].get(code)
    return {
        "team": code, "name": name, "conference": conf, "division": f"{conf} {div}",
        "elo": state["elo"].get(code), "rank": rank,
        "off_epa": off, "def_epa": dfn,
        # Missing is missing: a team with no measured offence or defence has
        # no net figure, not a made-up 0 or its offence alone.
        "net_epa": None if off is None or dfn is None else off - dfn,
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
        "yards": {"picks": sum(1 for p in picks for side in ("home_leaders", "away_leaders")
                               for l in (p.get(side) or {}).values() if l.get("over") is not None),
                  "over": sum(1 for p in picks for side in ("home_leaders", "away_leaders")
                              for l in (p.get(side) or {}).values() if l.get("over"))},
        "td": {"picks": sum(1 for p in picks for side in ("home_td", "away_td") for t in (p.get(side) or [])
                            if t.get("scored") is not None),
               "scored": sum(1 for p in picks for side in ("home_td", "away_td") for t in (p.get(side) or [])
                             if t.get("scored"))},
        "totals": {"picks": len(tot), "correct": sum(1 for p in tot if p["total_correct"])},
        "by_season": sorted(seasons.values(), key=lambda r: r["season"]),
        "by_week": sorted(weeks.values(), key=lambda r: (r["season"], r["week"])),
        "basis": "one pick per game, the last one logged before kickoff",
    }


# ------------------------------------------------------------ touchdown meters

TD_METERS_DEFINITION = (
    "Two meters from the live record - touchdown picks logged before kickoff, games graded "
    "from the box score. TOP 3 -> SCORED: of the times a player was in his team's top three "
    "most likely touchdown scorers, how often he scored a rushing or receiving touchdown; a "
    "pick who did not play is left out, as every chance was if he plays. NOT TOP 3 -> NO TD: "
    "of the times a player played without being in that top three, how often he did not "
    "score. Where the whole list a bettor could back (every active QB, RB, FB, WR and TE) was "
    "logged before kickoff, 'not top 3' is the rest of that list who played. Games logged "
    "before that list was, with only the top three, rebuild 'not top 3' from the snap counts: "
    "every other player of the team who took an offensive snap at QB, RB, FB, WR or TE, his "
    "touchdowns from the box score (none if he has no box-score line).")


def _meter(n, hits, key):
    return {"n": n, key: hits, "rate": hits / n if n else None}


def _blank():
    return {"top3": [0, 0], "rest": [0, 0]}


def _meters(acc):
    return {"top3": _meter(*acc["top3"], "scored"), "rest": _meter(*acc["rest"], "no_td")}


def td_meters(log, lines):
    """For every player and team, and overall: how the touchdown picks did.

    top3 - each time a player was in his team's logged top three and played:
      n, scored (he scored a rushing or receiving TD), rate.
    rest - each time a player played without being in it: n, no_td (he did
      not score), rate. A game whose whole list was logged before kickoff
      (home_td_all / away_td_all) counts the rest of that list who played,
      once it is graded (td_all_graded); an older game, with only the top
      three logged, counts every other player of the team who took an
      offensive snap at QB, RB, FB, WR or TE (`lines`, from
      nfl_predict.td_record_lines), with his touchdowns from the box score.
      A side whose snap counts are not in yet is pending, not counted.

    Only games whose top three are graded (td_graded) count, and only sides
    with touchdown picks logged: nothing here was decided after kickoff but
    who played and who scored. Returns overall, by_week, teams, players,
    pending (sides waiting for snap counts) and the definition. A player's
    team is his latest; one counted for more than one team also carries
    by_team (_split())."""
    snapped = lines[lines["in_snaps"].astype(bool)] if len(lines) else lines
    by_side = {key: g for key, g in snapped.groupby(["game_id", "team"])} if len(snapped) else {}
    events, pending = [], 0
    picks = sorted(log.get("picks", {}).items(), key=lambda kv: (str(kv[1].get("kickoff")), kv[0]))
    for gid, p in picks:
        if not p.get("td_graded"):
            continue
        for side in ("home", "away"):
            top = p.get(f"{side}_td") or []
            if not top:
                continue
            team = p.get(side)
            base = {"season": int(p["season"]), "week": int(p["week"]), "team": team}
            for t in top:
                if t.get("scored") is None:       # did not play: not a miss
                    continue
                events.append(dict(base, gsis_id=t.get("gsis_id"), name=t.get("name"),
                                   position=t.get("position"), kind="top3", hit=bool(t["scored"])))
            top_ids = {t.get("gsis_id") for t in top}
            top_names = {t.get("name") for t in top}
            full = p.get(f"{side}_td_all")
            if full is not None:
                if not p.get("td_all_graded"):
                    pending += 1
                    continue
                rest = [(t.get("gsis_id"), t.get("name"), t.get("position"), not t["scored"])
                        for t in full if t.get("gsis_id") not in top_ids and t.get("scored") is not None]
                source = "list"
            else:
                g = by_side.get((gid, team))
                if g is None:
                    pending += 1
                    continue
                rest = [(i, n, pos, not (tds >= 1)) for i, n, pos, tds in
                        zip(g["gsis_id"], g["name"], g["position"], g["tds"].astype(float))
                        if i not in top_ids and n not in top_names]
                source = "snaps"
            for gsis, name, pos, no_td in rest:
                events.append(dict(base, gsis_id=gsis, name=name, position=pos, kind="rest", hit=no_td,
                                   source=source))

    overall, weeks, teams, players = _blank(), {}, {}, {}
    sources = {}
    for e in events:
        accs = [overall, weeks.setdefault((e["season"], e["week"]), _blank()),
                teams.setdefault(e["team"], _blank())]
        if e["gsis_id"]:
            pl = players.setdefault(e["gsis_id"], dict(_blank(), info={}, by_team={}))
            pl["info"] = {"name": e["name"], "team": e["team"], "position": e["position"]}   # latest wins
            accs += [pl, pl["by_team"].setdefault(e["team"], _blank())]
        for acc in accs:
            acc[e["kind"]][0] += 1
            acc[e["kind"]][1] += int(e["hit"])
        if e["kind"] == "rest":
            sources.setdefault((e["season"], e["week"]), set()).add(e["source"])
    return {
        "overall": _meters(overall),
        "by_week": [dict(_meters(acc), season=s, week=w, rest_from=sorted(sources.get((s, w), ())))
                    for (s, w), acc in sorted(weeks.items())],
        "teams": sorted(({"team": code, "name": nd.TEAMS.get(code, (code,))[0], **_meters(acc)}
                         for code, acc in teams.items()),
                        key=lambda t: (-(t["top3"]["n"] + t["rest"]["n"]), t["team"])),
        "players": sorted(({"gsis_id": gsis, **acc["info"], **_meters(acc), **_split(acc["by_team"])}
                           for gsis, acc in players.items()),
                          key=lambda r: (str(r["team"]), -r["top3"]["n"], -r["rest"]["n"], str(r["name"]))),
        "pending": pending,
        "list_from": min((k for k, v in sources.items() if "list" in v), default=None),
        "definition": TD_METERS_DEFINITION,
    }


def _split(by_team):
    """A player counted for more than one team (traded, or signed from
    another) carries by_team: {team: his two meters for that team}, so a
    team's players add up to the team's meters; one team, nothing extra."""
    if len(by_team) < 2:
        return {}
    return {"by_team": {team: _meters(acc) for team, acc in sorted(by_team.items())}}


def _no_meters():
    return {"top3": _meter(0, 0, "scored"), "rest": _meter(0, 0, "no_td")}


def team_td_meters(meters, team):
    """One team's two meters; n = 0 and rate None when it has none yet."""
    for t in (meters or {}).get("teams", []):
        if t["team"] == team:
            return {"top3": t["top3"], "rest": t["rest"]}
    return _no_meters()


def attach_td_meters(game, meters):
    """Each listed touchdown player's two meters, and each side's team
    meters, on a week.json game. Copies the lists: a logged list stays as
    it was logged."""
    by_id = {p["gsis_id"]: {"top3": p["top3"], "rest": p["rest"]} for p in (meters or {}).get("players", [])}
    for key in ("home_td", "away_td", "home_td_all", "away_td_all"):
        if game.get(key):
            game[key] = [dict(e, meters=by_id.get(e.get("gsis_id")) or _no_meters()) for e in game[key]]
    game["td_meters"] = {side: team_td_meters(meters, game[side]) for side in ("home", "away")}
    return game


# ---------------------------------------------------------------- same-game tickets

def load_sgp_history(path=SGP_HISTORY):
    """A same-game tickets' backtest payload (v1's by default, v2's at
    SGP2_HISTORY), or None when none is on file."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# What week.json's sgp.history keeps of the backtest's payload.
SGP_HISTORY_KEYS = ("generated", "through", "history_from", "current_season", "periods", "games",
                    "left_out", "roi_reason", "td_prices", "td_groups", "inputs", "kinds")
# ...and of v2's, which also carries its choice and what it reports beside it:
# the confident games per period, the chosen pair and the objective that
# chose it, every pair of the grid, SMALL with the game's best TD, and v1 on
# the same games (all of it report-only but the choice).
SGP2_HISTORY_KEYS = SGP_HISTORY_KEYS + ("confident_games", "chosen", "grid", "alt_small", "v1_same_games")

# v1, retired: said beside its meters.
SGP_V1_RETIRED = (f"{sgp.VERSION} is retired: each game's tickets are built by {sgp2.VERSION} now. The "
                  f"{sgp.VERSION} tickets logged before kickoff stay logged and are graded by its own rules, "
                  "and its history and live meters are kept here, measured as before.")


def _sgp_history(payload, rule=sgp, keys=SGP_HISTORY_KEYS, script="experiments/nfl_sgp_backtest.py"):
    """The backtest's meters as the app shows them, or why there are none:
    no file, or one built by another version of the rule."""
    if payload is None:
        return {"available": False,
                "reason": f"no same-game ticket backtest on file ({script})"}
    if payload.get("version") != rule.VERSION:
        return {"available": False,
                "reason": f"the backtest on file is {payload.get('version')}, the rule is {rule.VERSION}"}
    return dict({k: payload.get(k) for k in keys}, available=True, reason=None)


def _sgp2_history(payload):
    """v2's history, or why there is none - also when the pair it replayed
    is not the one the rule holds (a file written before --choose)."""
    out = _sgp_history(payload, sgp2, SGP2_HISTORY_KEYS, "experiments/nfl_sgp2_backtest.py")
    if not out["available"]:
        return out
    if sgp2.CONFIDENT is None or sgp2.TD_FLOOR is None:
        return {"available": False, "reason": sgp2.NOT_CHOSEN}
    chosen = payload.get("chosen") or {}
    if (chosen.get("confident"), chosen.get("td_floor")) != (sgp2.CONFIDENT, sgp2.TD_FLOOR):
        return {"available": False,
                "reason": (f"the backtest on file replayed CONFIDENT {chosen.get('confident')} and TD_FLOOR "
                           f"{chosen.get('td_floor')}; the rule holds {sgp2.CONFIDENT} and {sgp2.TD_FLOOR}")}
    return out


def _live(log, rule):
    """(the logged games, the live meters) of one version: every ticket of
    `rule.VERSION` logged before kickoff (prediction_history.json), graded
    or pending, metered by that version's summarise."""
    logged = [p for p in (log.get("picks") or {}).values()
              if any(t.get("version") == rule.VERSION for t in p.get("tickets") or [])]
    tickets = [t for p in logged for t in p["tickets"] if t.get("version") == rule.VERSION]
    return logged, {"games": len(logged), "graded_games": sum(bool(p.get("tickets_graded")) for p in logged),
                    "first_kickoff": min((p["kickoff"] for p in logged), default=None),
                    "kinds": rule.summarise(tickets)}


def _v1_caveats():
    """v1's sentences (nfl_sgp), one wording live and history."""
    return {
        "edge": sgp.NO_EDGE, "chance": sgp.CHANCE_NOTE, "est": sgp.EST_PAYOUT,
        "est_erratum": sgp.EST_ERRATUM, "roi_priced": sgp.ROI_PRICED, "prices": sgp.PRICE_NOTE, "exact": sgp.EXACT_PAYOUT, "bias": sgp.KNOWN_BIAS,
        "meters": sgp.METER_NOTE, "live": sgp.LIVE_ROI, "slots": sgp.SLOT_RULE,
        "rookie": sgp.ROOKIE_RULE, "not_in_v1": sgp.NOT_IN_V1,
        "legs": dict(sgp.LEG_RULES), "tickets": dict(sgp.TICKET_RULES)}


def sgp_v1_block(log, history=None):
    """v1's (sgp-v1, retired) meters, history and live, and the rule's words
    beside them - week.json's "sgp" before v2, under sgp.v1 now, labelled
    retired (SGP_V1_RETIRED).

      history  the backtest's (nfl/data/sgp_backtest.json, `history`): per
               kind, per period, its meters and chance bins, when it was
               generated and the last game it replayed
      live     nfl_sgp.summarise of every ticket of this rule's version logged
               before kickoff (prediction_history.json), graded or pending -
               return at the logged est. payout (an estimate, not a bound:
               est_erratum)
      caveats  the rule's own sentences (nfl_sgp): no edge claimed, the
               product chance, est. prices and the exact payout, the
               backtest's known bias, the meters, what v1 leaves out - and
               est_erratum, the correction found after the rule was fixed,
               shown beside v1's words on the est. payout and live return
    """
    return {
        "version": sgp.VERSION, "rule": "nfl/engine/nfl_sgp.py", "retired": True,
        "retired_note": SGP_V1_RETIRED, "kinds": list(sgp.KINDS),
        "legs": dict(sgp.LEGS), "stake": sgp.STAKE_UNIT, "use_cap": sgp.USE_CAP,
        "rookie_floor": sgp.ROOKIE_FLOOR,
        "history": _sgp_history(history),
        "live": _live(log, sgp)[1],
        "caveats": _v1_caveats(),
    }


def sgp_block(log, history=None, v1_history=None):
    """week.json's "sgp": v2's (sgp-v2, nfl_sgp2) ticket meters, history and
    live, the pair it was chosen with and the grid, and its rule's words -
    with v1's whole block, retired, under "v1". `history` is v2's backtest
    payload, `v1_history` v1's.

      confident, td_floor, chosen_on, grids  the rule's two numbers
               (nfl_sgp2), the day --choose wrote them, and the grids,
               coverage floor and seasons they were chosen from
      history  v2's backtest (nfl/data/sgp2_backtest.json): per kind, per
               period, its meters, coverage and chance bins; the confident
               games per period; `chosen` (the pair, the objective, its DEV
               figures, how many pairs cleared the coverage floor, and
               `rises` - the rule's "hit rate always rises" claim counted
               against DEV's grid, found after the rule was fixed) and
               `grid` (every pair, per period - report-only); SMALL with the
               game's best TD and v1 on the same games (report-only) - or
               missing, and why, when the file is not this rule's or
               replayed another pair
      live     every sgp-v2 ticket logged before kickoff, graded or pending,
               and `confident_games`, how many of those games were
               confident (each v2 ticket logs whether its game was)
      caveats  v1's sentences, which v2 keeps (its legs, slots, grading,
               meters and est. prices are v1's), the market-is-hard-to-beat
               edge line among them; edge_v2, v2's own no-edge words;
               `rules`, v2's narrowing (confident games only, the pick's
               side only, the TD floor, SMALL's preference, how the two
               numbers were chosen on DEV and measured on the seasons
               after); `objective`; `tickets`, each ticket's slots as v2
               fills them; and `untouched` (nfl_sgp2.UNTOUCHED_NOTE, found
               after the rule was fixed): "untouched" covers the two
               numbers, not SMALL's pick's-team clause, whose motivating
               seasons overlap CONFIRM
      v1       sgp_v1_block: v1's history and live meters, retired
    """
    logged, live = _live(log, sgp2)
    live["confident_games"] = sum(
        1 for p in logged if any(t.get("confident_game") for t in p["tickets"] if t.get("version") == sgp2.VERSION))
    caveats = _v1_caveats()
    caveats.update(edge_v2=sgp2.NO_EDGE, rules=sgp2.rules(), objective=sgp2.OBJECTIVE,
                   tickets=sgp2.ticket_rules(), untouched=sgp2.UNTOUCHED_NOTE)
    return {
        "version": sgp2.VERSION, "rule": "nfl/engine/nfl_sgp2.py", "kinds": list(sgp2.KINDS),
        "legs": dict(sgp2.LEGS), "stake": sgp.STAKE_UNIT, "use_cap": sgp.USE_CAP,
        "rookie_floor": sgp.ROOKIE_FLOOR,
        "confident": sgp2.CONFIDENT, "td_floor": sgp2.TD_FLOOR, "chosen_on": sgp2.CHOSEN_ON,
        "grids": {"confident": list(sgp2.CONFIDENT_GRID), "td_floor": list(sgp2.TD_FLOOR_GRID),
                  "coverage_min": sgp2.COVERAGE_MIN, "dev": list(sgp2.DEV), "confirm": list(sgp2.CONFIRM)},
        "history": _sgp2_history(history),
        "live": live,
        "caveats": caveats,
        "v1": sgp_v1_block(log, v1_history),
    }
