"""Predict this NFL week and write everything the app shows.

    python nfl/engine/nfl_predict.py            # sync the current season first
    python nfl/engine/nfl_predict.py --no-sync  # use the snapshot as it is

One run:
  1. refreshes the current season from nflverse (results, the schedule's
     projected starters, injury reports);
  2. rebuilds every pre-game feature and refits the models season by season,
     exactly as the backtest does, so the record the app prints is the
     record of this procedure;
  3. finds THE WEEK: the one holding the next game still to kick off;
  4. buys prices for it if the cache has none fresh (nfl_odds);
  5. predicts every game of the week that has not started, logs each pick
     (prediction_history.json), grades every logged pick whose game is over;
  6. writes nfl/app/data/{week,teams,performance}.json and the standalone
     page.

A PICK IS LOGGED ONLY BEFORE KICKOFF. Re-running before a game updates its
entry (the last line before kickoff is the one that counts); once the game
has started the entry is frozen, and a game that started before any run
logged it shows its result with no pick rather than a pick made afterwards.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_backtest as bt
import nfl_data as nd
import nfl_export as nx
import nfl_features as nf
import nfl_form as nform
import nfl_model as nm
import nfl_news as nnews
import nfl_odds as no
import nfl_players as nplayers
import nfl_props as nprops
import nfl_td as ntd

LOG = Path(os.environ.get("NFL_PREDICTIONS_LOG") or (nd.DATA / "prediction_history.json"))
LOG_SCHEMA = 1

SHOWN_STATUSES = ("Out", "Doubtful", "Questionable")
STATUS_ORDER = {s: i for i, s in enumerate(SHOWN_STATUSES)}
POSITION_ORDER = ["QB", "RB", "FB", "WR", "TE", "T", "OT", "G", "OG", "C", "OL",
                  "DE", "EDGE", "DT", "NT", "DL", "OLB", "LB", "ILB", "MLB",
                  "CB", "S", "FS", "SS", "DB", "K", "P", "LS"]
INJURIES_PER_TEAM = 8
TD_SHOWN = 3            # most likely scorers shown per team
WEEK_NAMES = {"WC": "Wild Card", "DIV": "Divisional round",
              "CON": "Conference championships", "SB": "Super Bowl"}


def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0))


def stamp(t):
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def naive_utc(t):
    """A timestamp in naive UTC, whether it arrived naive, aware or as text."""
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


# ---------------------------------------------------------------- the week

def current_week(games, now):
    """(season, week, game_type) of the next game to kick off, or of the
    last game played once a season is over."""
    upcoming = games[~games["played"] & (games["kickoff"] > now)]
    row = upcoming.iloc[0] if len(upcoming) else games[games["played"]].iloc[-1]
    return int(row["season"]), int(row["week"]), row["game_type"]


def week_label(week, game_type):
    return WEEK_NAMES.get(game_type, f"Week {week}")


# ---------------------------------------------------------------- the log

def load_log(path=LOG):
    path = Path(path)
    if not path.exists():
        return {"schema": LOG_SCHEMA, "picks": {}}
    log = json.loads(path.read_text())
    if log.get("schema") != LOG_SCHEMA:
        raise ValueError(f"prediction log is schema {log.get('schema')}, expected {LOG_SCHEMA}")
    return log


def save_log(log, path=LOG):
    Path(path).write_text(json.dumps(nx._clean(log), indent=1, sort_keys=True, allow_nan=False) + "\n")


def log_pick(log, game, now):
    """Upsert one pick, only while its game has not started. Returns True if
    written."""
    if naive_utc(game["kickoff"]) <= naive_utc(now):
        return False
    entry = log["picks"].get(game["game_id"], {})
    keep = {"first_logged": entry.get("first_logged") or stamp(now)}
    fields = ("game_id", "season", "week", "game_type", "kickoff", "home", "away",
              "pick", "pick_prob", "home_prob", "model_prob", "market_prob", "market_source",
              "tier", "model_spread", "market_spread", "spread_pick", "model_total",
              "market_total", "total_pick", "home_qb", "away_qb", "market", "home_td", "away_td",
              "home_leaders", "away_leaders")
    log["picks"][game["game_id"]] = dict(
        {k: game.get(k) for k in fields}, **keep, last_logged=stamp(now))
    return True


def _snap_index(snaps):
    """{game_id: {player name: snaps taken}} from the snap counts."""
    if snaps is None or not len(snaps):
        return {}
    taken = snaps["offense_snaps"].fillna(0) + snaps.get("defense_snaps", 0 * snaps["offense_snaps"]).fillna(0)
    out = {}
    for gid, name, n in zip(snaps["game_id"], snaps["player"], taken):
        out.setdefault(gid, {})[name] = out.get(gid, {}).get(name, 0) + float(n)
    return out


def _played(snap_index, gid, pick):
    """True / False once the game's snap counts are in, None before."""
    game = snap_index.get(gid)
    if game is None:
        return None
    return game.get(pick.get("name"), 0) > 0


def grade_touchdowns(log, player_games, snaps=None):
    """Mark each logged touchdown pick scored or not, once its game's box
    score and snap counts are in (results arrive hours before player
    statistics). Every probability was IF HE PLAYS: a pick who took no
    snap is marked played=False and left out of the record, not counted
    as a miss."""
    if player_games is None or not len(player_games):
        return 0
    by_game = player_games.groupby("game_id")
    snap_index = _snap_index(snaps)
    graded = 0
    for gid, p in log["picks"].items():
        if not p.get("final") or p.get("td_graded") or gid not in by_game.groups:
            continue
        box = by_game.get_group(gid)
        tds = dict(zip(box["player_id"], box["rushing_tds"].fillna(0) + box["receiving_tds"].fillna(0)))
        picks = [t for side in ("home_td", "away_td") for t in (p.get(side) or [])]
        if any(t.get("gsis_id") not in tds and _played(snap_index, gid, t) is None for t in picks):
            continue                  # no box line and no snap counts yet: wait
        for pick in picks:
            if pick.get("gsis_id") in tds:
                pick["scored"] = bool(tds[pick["gsis_id"]] >= 1)
            elif _played(snap_index, gid, pick):
                pick["scored"] = False
            else:
                pick["played"], pick["scored"] = False, None
        p["td_graded"] = True
        graded += 1
    return graded


def grade(log, games):
    """Fill in the result of every logged pick whose game is final."""
    finals = games[games["played"]].set_index("game_id")
    graded = 0
    for gid, p in log["picks"].items():
        if gid not in finals.index or p.get("final"):
            continue
        g = finals.loc[gid]
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        p["final"] = {"home": hs, "away": as_}
        margin = hs - as_
        if margin == 0:
            p["correct"], p["void"] = None, True
        else:
            winner = p["home"] if margin > 0 else p["away"]
            p["correct"] = winner == p["pick"]
        sp = p.get("spread_pick")
        if sp and p.get("market_spread") is not None:
            cover = margin - p["market_spread"]     # > 0: home covered
            p["spread_correct"] = None if cover == 0 else ((cover > 0) == (sp["team"] == p["home"]))
        tp = p.get("total_pick")
        if tp and p.get("market_total") is not None:
            over = (hs + as_) - p["market_total"]
            p["total_correct"] = None if over == 0 else ((over > 0) == (tp["side"] == "OVER"))
        graded += 1
    return graded


# ---------------------------------------------------------------- one game

def _num(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def injuries_for(injuries, season, week, team):
    if injuries is None or injuries.empty:
        return []
    rows = injuries[(injuries["season"] == season) & (injuries["week"] == week)
                    & (injuries["team"] == team) & injuries["report_status"].isin(SHOWN_STATUSES)]
    out = []
    for r in rows.itertuples(index=False):
        pos = r.position if isinstance(r.position, str) else ""
        out.append({"name": r.full_name, "position": pos, "status": r.report_status,
                    "injury": r.report_primary_injury if isinstance(r.report_primary_injury, str) else None,
                    "gsis_id": r.gsis_id if isinstance(r.gsis_id, str) else None})
    out.sort(key=lambda i: (STATUS_ORDER.get(i["status"], 9),
                            POSITION_ORDER.index(i["position"]) if i["position"] in POSITION_ORDER else 99,
                            i["name"]))
    return out


def qb_info(row, side, state, names):
    qb_id = row[f"{side}_qb_used"]
    if not isinstance(qb_id, str):
        if row.get(f"{side}_qb_source"):   # a starter ruled out with no one to name
            return {"id": None, "name": "unnamed backup", "rating": _num(row[f"{side}_qb_rating"]),
                    "starts": 0, "source": row[f"{side}_qb_source"], "change": True}
        return None
    name = names.get(qb_id) or (row.get(f"{side}_qb") if row.get(f"{side}_qb_source") != "replaced" else None)
    return {"id": qb_id, "name": name, "rating": _num(row[f"{side}_qb_rating"]),
            "starts": int(row[f"{side}_qb_starts"]), "source": row[f"{side}_qb_source"],
            "change": bool(row[f"{side}_qb_change"])}


def ruled_out_qbs(injuries, season, week, team):
    """gsis ids and names of the team's quarterbacks listed OUT this week."""
    return {(i["gsis_id"], i["name"]) for i in injuries_for(injuries, season, week, team)
            if i["position"] == "QB" and i["status"] == "Out"}


def off_roster(rosters, season, week):
    """gsis ids whose latest weekly roster row this season, up to this week,
    is anything but active - injured reserve and the other reserve lists
    are not on the injury report, so this is where they show."""
    if rosters is None or not len(rosters):
        return set()
    r = rosters[(rosters["season"] == season) & (rosters["week"] <= week)].sort_values("week")
    latest = r.groupby("gsis_id").tail(1)
    # A game-day inactive from an earlier week says nothing about this one.
    gone = latest[(latest["status"] != "ACT") & ~((latest["status"] == "INA") & (latest["week"] != week))]
    return set(gone["gsis_id"].dropna())


def stand_in(state, team, season, ruled_out, elsewhere, names):
    """Who starts when the projected starter cannot: the quarterback who most
    recently threw for THIS team THIS season (a last starter from another
    season may have moved on - Kirk Cousins was Atlanta's last starter of
    2025 and Las Vegas's starter in 2026), is not ruled out himself and is
    not another team's starter this week. None when nobody qualifies: the
    model then reads a replacement-level unknown."""
    out_ids = {i for i, _ in ruled_out if i}
    out_names = {n for _, n in ruled_out if n}
    for s, _kick, qb, dropbacks in reversed(state.get("team_qbs", {}).get(team, [])):
        if s != season:
            break
        if dropbacks < 5 or qb in out_ids or names.get(qb) in out_names or qb in elsewhere:
            continue
        return qb
    return None


def replace_ruled_out(row, state, injuries, names, season, week, elsewhere=frozenset(),
                      unavailable=frozenset()):
    """If a projected starter is listed Out - or is off the active roster
    (`unavailable`: injured reserve and the like) - start stand_in()
    instead. `elsewhere` is every other game's projected starters this week.
    Returns (row, {side: name of the quarterback replaced})."""
    replaced = {}
    for side in ("home", "away"):
        qb_id = row[f"{side}_qb_used"]
        out = ruled_out_qbs(injuries, season, week, row[side])
        out |= {(q, names.get(q)) for q in unavailable}
        name = names.get(qb_id)
        if not isinstance(qb_id, str) or not (qb_id in {i for i, _ in out} or (name and name in {n for _, n in out if n})):
            continue
        alt = stand_in(state, row[side], season, out | {(qb_id, name)}, elsewhere, names)
        row = nf.with_quarterback(row, side, alt, state, "replaced")
        replaced[side] = name or qb_id
    return row, replaced


def predict_game(row, models, market, tiers):
    """The blended numbers for one game that has not started."""
    market_p = market.get("home_prob") if market else None
    spread_p = (float(nm.spread_prob(market["spread"]))
                if market and market.get("spread") is not None else None)
    if market_p is None:
        market_p = spread_p
    frame = row.to_frame().T.infer_objects()
    p = models.predict(frame, market_p=[market_p if market_p is not None else np.nan],
                       spread_p=[spread_p if spread_p is not None else np.nan]).iloc[0]
    home_prob = float(p["blend_prob"])
    pick_home = home_prob >= 0.5
    pick_prob = home_prob if pick_home else 1 - home_prob
    tier = bt.tier_of(pick_prob)
    out = {
        "home_prob": home_prob, "model_prob": float(p["model_prob"]), "market_prob": market_p,
        "market_source": market.get("source") if market else None,
        "pick": row["home"] if pick_home else row["away"], "pick_prob": pick_prob,
        "tier": tier, "tier_hit_rate": (tiers.get(tier) or {}).get("hit_rate"),
        "model_spread": float(p["margin"]), "model_total": float(p["total_pred"]),
        "market_spread": market.get("spread") if market else None,
        "market_total": market.get("total") if market else None,
        "spread_pick": None, "total_pick": None,
    }
    if out["market_spread"] is not None:
        gap = out["model_spread"] - out["market_spread"]
        team = row["home"] if gap > 0 else row["away"]
        line = -out["market_spread"] if team == row["home"] else out["market_spread"]
        out["spread_pick"] = {"team": team, "line": line, "edge": abs(gap)}
    if out["market_total"] is not None:
        gap = out["model_total"] - out["market_total"]
        out["total_pick"] = {"side": "OVER" if gap > 0 else "UNDER", "line": out["market_total"], "edge": abs(gap)}
    return out


# ---------------------------------------------------------------- the run

def run(sync=True, now=None, out_dir=nx.APP_DATA, log_path=LOG, log=print,
        cache_path=no.CACHE, api_key=None, news_path=nnews.CACHE):
    now = naive_utc(now) if now is not None else now_utc()
    if sync:
        try:
            nd.sync(log=log)
        except Exception as err:   # an outage must not stop a card going out
            log(f"   [sync] failed, using the snapshot on file: {err}")
    games = nd.load_games()
    frame, state = nf.build_with_state(games, nd.load_team_games(), nd.load_qb_games(), nm.ELO)
    frame["market_prob"] = bt.market_prob(frame)
    preds, models = nm.walk_forward(frame, keep_models=True)
    backtest = nm.backtest(frame, preds)
    confirm = f"{bt.CONFIRM[0]}-{bt.CONFIRM[1]}"
    tiers = backtest["tiers"][confirm]

    season, week, game_type = current_week(games, now)
    slate = frame[(frame["season"] == season) & (frame["week"] == week)].sort_values(["kickoff", "game_id"])
    log(f"   {season} {week_label(week, game_type)}: {len(slate)} games")

    cache = no.load(cache_path)
    upcoming = frame[~frame["played"] & (frame["kickoff"] > now)][["away", "home", "kickoff"]]
    clock = now.tz_localize("UTC").to_pydatetime()
    no.refresh(cache, upcoming, api_key=api_key, log=log, now=clock)
    no.save(cache, cache_path)

    qb_names = _qb_names(games)
    injuries = nd.load_injuries()
    players = nplayers.PlayerData(injuries=injuries)
    players.set_schedule(games)
    td_built = ntd.build_with_state(games, ntd.load_lines())
    td_report = ntd.report(td_built[0], ntd.walk_forward(td_built[0]))
    td_live = ntd.fit_live(season, games=games, built=td_built)
    props_built = nprops.build_with_state(games, nprops.load_lines())
    props_report = nprops.report(props_built[0], nprops.walk_forward(props_built[0]))
    props_live = nprops.fit_live(season, games=games, built=props_built)
    news = nnews.load(news_path)
    if sync:   # the feeds are read where the internet is open - the runner
        nnews.refresh(news, log=log, now=clock)
        nnews.save(news, news_path)
    history = load_log(log_path)
    out_games = []
    for _, row in slate.iterrows():
        kickoff = pd.Timestamp(row["kickoff"])
        status = "final" if row["played"] else ("in progress" if kickoff <= now else "upcoming")
        market = no.market_for(cache, row, now=clock)
        # The whole list: the QB check below must see a Doubtful starter even
        # when eight teammates are listed Out ahead of him; only the display
        # is cut to INJURIES_PER_TEAM.
        full_injuries = {s: injuries_for(injuries, season, week, row[s]) for s in ("home", "away")}
        game = {
            "game_id": row["game_id"], "season": season, "week": week, "game_type": row["game_type"],
            "kickoff": stamp(kickoff), "status": status,
            "home": row["home"], "away": row["away"],
            "home_name": nd.TEAMS[row["home"]][0], "away_name": nd.TEAMS[row["away"]][0],
            "neutral": bool(row["neutral"]), "div_game": bool(row["div_game"]),
            "roof": row["roof"] if isinstance(row["roof"], str) else row["roof_class"],
            "stadium": row["stadium"] if isinstance(row["stadium"], str) else None,
            "home_rest": _num(row["home_rest_days"]), "away_rest": _num(row["away_rest_days"]),
            "elo": {"home": _num(row["home_elo"]), "away": _num(row["away_elo"])},
            "home_qb": qb_info(row, "home", state, qb_names),
            "away_qb": qb_info(row, "away", state, qb_names),
            "home_form": nform.form(games, row["home"], kickoff, season),
            "away_form": nform.form(games, row["away"], kickoff, season),
            "home_injuries": full_injuries["home"][:INJURIES_PER_TEAM],
            "away_injuries": full_injuries["away"][:INJURIES_PER_TEAM],
            "home_injuries_total": len(full_injuries["home"]),
            "away_injuries_total": len(full_injuries["away"]),
            "market": market,
        }
        key = nplayers.for_game(players, row, season, week, kickoff)
        game["home_players"], game["away_players"] = key["home"], key["away"]
        game["alerts"] = key["alerts"]
        named = {p["name"] for side in ("home", "away") for grp in ("offense", "defense")
                 for p in key[side][grp]}
        named |= _roster_names(players, season, (row["home"], row["away"]))
        game["news"] = nnews.for_game(news, (row["home"], row["away"]), named,
                                      kickoff.tz_localize("UTC").to_pydatetime())
        if status == "final":
            game["score"] = {"home": int(row["home_score"]), "away": int(row["away_score"])}
        if status == "upcoming":
            elsewhere = {q for _, r in slate.iterrows() if r["game_id"] != row["game_id"]
                         for q in (r["home_qb_used"], r["away_qb_used"]) if isinstance(q, str)}
            row, replaced = replace_ruled_out(row, state, injuries, qb_names, season, week, elsewhere,
                                              off_roster(players.rosters, season, week))
            for side, name in replaced.items():
                game[f"{side}_qb"] = qb_info(row, side, state, qb_names)
                game[f"{side}_qb"]["replaced"] = name
            game.update(predict_game(row, models[season], market, tiers))
            out_now = off_roster(players.rosters, season, week)
            for side in ("home", "away"):
                game[f"{side}_td"] = touchdown_picks(td_live, players, row, side, game, market,
                                                     full_injuries[side], out_now, season, week)
                proj, game[f"{side}_leaders"] = yardage(props_live, players, row, side, game, market,
                                                        full_injuries[side], out_now, season, week)
                for p in game[f"{side}_players"]["offense"] + game[f"{side}_td"]:
                    if p.get("gsis_id") in proj:
                        p["proj"] = proj[p["gsis_id"]]
            log_pick(history, game, now)
        else:
            logged = history["picks"].get(row["game_id"])
            if logged:
                game.update({k: logged.get(k) for k in (
                    "pick", "pick_prob", "home_prob", "model_prob", "market_prob", "market_source",
                    "tier", "model_spread", "market_spread", "spread_pick", "model_total",
                    "market_total", "total_pick")})
                # The price the pick was made against, not today's.
                if logged.get("market"):
                    game["market"] = logged["market"]
                for side in ("home_td", "away_td", "home_leaders", "away_leaders"):
                    if logged.get(side):
                        game[side] = logged[side]
                # The quarterbacks the logged pick was computed with - a
                # replaced starter must not reappear once the game is on.
                for side in ("home", "away"):
                    if logged.get(f"{side}_qb"):
                        game[f"{side}_qb"] = logged[f"{side}_qb"]
                game["tier_hit_rate"] = (tiers.get(logged.get("tier")) or {}).get("hit_rate")
                game["logged_at"] = logged.get("last_logged")
            else:
                game["pick"] = None
                game["no_pick_reason"] = "No pick was logged before kickoff."
        for side in ("home", "away"):
            qb = game[f"{side}_qb"]
            if qb:
                hurt = [i for i in full_injuries[side]
                        if i["position"] == "QB" and (i["gsis_id"] == qb["id"] or i["name"] == qb["name"])
                        and i["status"] in ("Out", "Doubtful")]
                qb["injury_status"] = hurt[0]["status"] if hurt else None
        out_games.append(game)

    graded = grade(history, games)
    grade_touchdowns(history, players.offense, players.snaps)
    grade_yards(history, players.offense, players.snaps)
    save_log(history, log_path)
    log(f"   logged {sum(1 for g in out_games if g['status'] == 'upcoming')} picks, graded {graded}")

    last_played = games[games["played"]]["kickoff"].max()
    trained_on = models[season].trained_on
    cav = nx.caveats(backtest, td_report, props_report)
    nx.write("week", {
        "season": season, "week": week, "game_type": game_type, "label": week_label(week, game_type),
        "games": out_games, "data_through": nd.eastern_day(last_played), "trained_on": trained_on,
        "byes": byes(slate, game_type),
        "tiers": tiers, "value_enabled_at": backtest["value"]["enabled_at"],
        "caveats": cav, "as_of": stamp(now),
    }, out_dir)
    nx.write("teams", teams_payload(games, state, qb_names, now, season, players), out_dir)
    nx.write("performance", {
        "backtest": backtest, "track_record": nx.track_record(history), "td": td_report,
        "props": props_report,
        "trained_on": trained_on, "caveats": cav, "data_through": nd.eastern_day(last_played),
    }, out_dir)
    return out_games


def touchdown_picks(live, players, row, side, game, market, injuries, off, season, week):
    """The team's TD_SHOWN most likely scorers for a game still to come.

    The team's expected points are the market's (its total and spread) when
    there is a price, else the model's own; players listed Out or off the
    active roster are left out, Doubtful and Questionable ones kept and
    marked - every probability is conditional on the player playing."""
    home = side == "home"
    implied, _against = _implied(game, market, home)
    status, exclude, roster = _who_plays(players, row, side, injuries, off, season, week)
    picks = ntd.predict_team(live, row[side], implied, home, exclude=exclude, roster=roster,
                             playoff=row["game_type"] != "REG")[:TD_SHOWN]
    return [{"gsis_id": p["gsis_id"], "name": p["name"], "position": p["position"],
             "prob": p["prob"], "tds_season": p["tds_season"], "games_season": p["games_season"],
             "status": status.get(p["gsis_id"], "active")} for p in picks]


def byes(slate, game_type):
    """Teams with no game in a regular-season week, as the week's header
    lists them."""
    if game_type != "REG":
        return []
    playing = set(slate["home"]) | set(slate["away"])
    return sorted(set(nd.TEAMS) - playing)


def _implied(game, market, home):
    """(this side's, the other side's) expected points: the market's total
    and spread when priced, else the model's own."""
    total, spread = (market or {}).get("total"), (market or {}).get("spread")
    if total is None or spread is None:
        total, spread = game.get("model_total"), game.get("model_spread")
    return ntd.implied_points(total, spread, home), ntd.implied_points(total, spread, not home)


def _who_plays(players, row, side, injuries, off, season, week):
    """({gsis: injury status}, ids not playing, the active roster or None)."""
    status = {i["gsis_id"]: i["status"].lower() for i in injuries if i.get("gsis_id")}
    exclude = {g for g, st in status.items() if st == "out"} | set(off)
    roster = None
    r = players.rosters
    if len(r):
        mine = r[(r["season"] == season) & (r["team"] == row[side]) & (r["week"] <= week)]
        if len(mine):
            latest = mine[mine["week"] == mine["week"].max()]
            roster = list(latest[latest["status"] == "ACT"]["gsis_id"].dropna())
    return status, exclude, roster


def yardage(live, players, row, side, game, market, injuries, off, season, week):
    """Every likely player's projected (median) yards for a game still to
    come, if he plays: {gsis: {pass_yds, rush_yds, rec_yds}}, and the
    team's projected leaders - passer, rusher, receiver."""
    home = side == "home"
    implied_for, implied_against = _implied(game, market, home)
    status, exclude, roster = _who_plays(players, row, side, injuries, off, season, week)
    starter = row[f"{side}_qb_used"] if isinstance(row[f"{side}_qb_used"], str) else None
    named = (game.get(f"{side}_qb") or {}).get("name")
    proj = nprops.project_team(live, row[side], implied_for, implied_against, home, exclude=exclude,
                               roster=roster, playoff=row["game_type"] != "REG", starter=starter,
                               starter_name=named)
    by_id = {p["gsis_id"]: {k: p[k] for k in ("pass_yds", "rush_yds", "rec_yds")} for p in proj}
    leaders = {}
    for key, word in (("pass_yds", "passing"), ("rush_yds", "rushing"), ("rec_yds", "receiving")):
        best = max((p for p in proj if p[key] is not None), key=lambda p: p[key], default=None)
        if best is not None:
            leaders[word] = {"gsis_id": best["gsis_id"], "name": best["name"], "position": best["position"],
                             "yards": best[key], "status": status.get(best["gsis_id"], "active")}
    return by_id, leaders


def grade_yards(log, player_games, snaps=None):
    """Each logged yardage leader's actual yards and whether he went over
    his projection, once the box score and snap counts are in. A leader who
    took no snap is marked played=False and left out - the projection was
    IF HE PLAYS - rather than graded a zero-yard under."""
    if player_games is None or not len(player_games):
        return 0
    by_game = player_games.groupby("game_id")
    snap_index = _snap_index(snaps)
    cols = {"passing": "passing_yards", "rushing": "rushing_yards", "receiving": "receiving_yards"}
    graded = 0
    for gid, p in log["picks"].items():
        if not p.get("final") or p.get("yards_graded") or gid not in by_game.groups:
            continue
        box = by_game.get_group(gid).groupby("player_id")[list(cols.values())].sum(min_count=1)
        leaders = [l for side in ("home_leaders", "away_leaders") for l in (p.get(side) or {}).values()]
        if any(l.get("gsis_id") not in box.index and _played(snap_index, gid, l) is None for l in leaders):
            continue                  # no box line and no snap counts yet: wait
        for side in ("home_leaders", "away_leaders"):
            for stat, leader in (p.get(side) or {}).items():
                if leader.get("gsis_id") in box.index:
                    actual = box.loc[leader["gsis_id"], cols[stat]]
                    actual = 0.0 if actual != actual else float(actual)
                elif _played(snap_index, gid, leader):
                    actual = 0.0
                else:
                    leader["played"], leader["actual"], leader["over"] = False, None, None
                    continue
                leader["actual"] = actual
                leader["over"] = None if actual == leader["yards"] else actual > leader["yards"]
        p["yards_graded"] = True
        graded += 1
    return graded


def _roster_names(players, season, teams):
    """Full names of everyone on either team's latest weekly roster (active,
    reserve or inactive) - the names news is matched against."""
    r = players.rosters
    if not len(r):
        return set()
    r = r[(r["season"] == season) & r["team"].isin(teams) & r["status"].isin(["ACT", "RES", "INA"])]
    if not len(r):
        return set()
    latest = r[r["week"] == r["week"].max()]
    return {n for n in latest["full_name"].dropna() if len(n.split()) >= 2}


def _qb_names(games):
    names = {}
    for side in ("home", "away"):
        ids, ns = games[f"{side}_qb_id"], games[f"{side}_qb"]
        for i, n in zip(ids, ns):
            if isinstance(i, str) and isinstance(n, str):
                names[i] = n
    return names


def teams_payload(games, state, qb_names, now, season, players=None):
    order = sorted(nd.TEAMS, key=lambda t: -state["elo"].get(t, 0))
    teams = []
    for rank, code in enumerate(order, 1):
        f = nform.form(games, code, now, season)
        entry = nx.team_entry(code, state, f["record"], rank, qb_names)
        entry["last5"] = f["last5"]
        if players is not None:
            kp = nplayers.key_players(players, code, season, now)
            entry["key_players"] = {grp: [{k: p[k] for k in ("name", "position", "value", "line", "rank")}
                                          for p in kp[grp]] for grp in ("offense", "defense")}
        teams.append(entry)
    return {"season": season, "teams": teams, "as_of": stamp(now),
            "elo_scale": {"hfa": nm.ELO.hfa, "points_per_elo": nm.ELO.points_per_elo}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args(argv)
    print("YourNFL weekly run")
    games = run(sync=not args.no_sync)
    # The phone's offline copy carries the data inside it, so it is stale the
    # moment the data files change - rebuilt here, after them, every run.
    import nfl_standalone
    nfl_standalone.build()
    for g in games:
        if g.get("pick"):
            print(f"   {g['away']:>3} @ {g['home']:<3}  {g['pick']:<3} {g['pick_prob']*100:5.1f}%  "
                  f"{g.get('tier') or '':8s} {g['status']}")
        else:
            print(f"   {g['away']:>3} @ {g['home']:<3}  -- {g['status']}")


if __name__ == "__main__":
    main()
