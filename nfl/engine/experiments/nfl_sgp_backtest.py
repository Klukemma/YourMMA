"""The history behind nfl_sgp's ROI meters (sgp-v1): every NFL game from
nfl_sgp.HISTORY_FROM to the last one played, its four same-game tickets
built exactly as the rule builds them, from inputs each known before the
game's kickoff, and graded from the final score and the box score.

    python nfl/engine/experiments/nfl_sgp_backtest.py
        -> nfl/data/sgp_backtest.json           the meters, per kind and period
           nfl/data/sgp_backtest_tickets.csv.gz  every ticket, for audit
    python nfl/engine/experiments/nfl_sgp_backtest.py --if-stale
        the same, only when stale() says the file on disk is behind

The NFL runner calls it with --if-stale before each predict run (about 30
seconds when it replays), so the app's history meter (week.json sgp.history,
which shows `generated` and `through`) holds every game whose box score the
snapshot has.

THE INPUTS, point in time:

  WIN     nfl_model.walk_forward's blend_prob: season S from models fitted
          on the seasons before S, reading the game's pre-kickoff features
          and closing market (as the app's pick does).
  SPREAD  the closing spread_line and its two prices (games.csv.gz), the
  TOTAL   closing total_line and its two prices, each de-vigged in
          proportion (nfl_sgp.no_vig) - the market's chance, as the rule says.
  TD      nfl_td.walk_forward's prob: season S fitted on the labelled rows of
          nfl_td.TRAIN_FROM to S - 1, each row's features read from the
          player's state before kickoff. A chance only in the usage groups
          nfl_td.shipped_groups() ships - the live rule (predict_roster gives
          the others None). Which groups ship was measured on the
          nfl_td.DEV and CONFIRM walk-forward rows, so it is the one input
          chosen with hindsight; `td_groups` in the output measures how many
          tickets it changes.
  rookie  no player_games row in any season before S (nfl_sgp.is_rookie).
  prices  the closing moneylines and spread/total prices. No anytime-TD
          price is on file, so a TD leg has no price, every ticket holds one,
          and return is shown as missing with nfl_sgp.NO_TD_PRICE - until
          TD_PRICES (game_id, gsis_id, decimal) is added, when the legs it
          prices are read and the return fills in where every leg is priced.

THE GRADE. WIN, SPREAD and TOTAL from the final score, overtime included
(a tie or a push voids). TD from the box score: the player's rushing plus
receiving touchdowns, nfl_td's label. Every TD candidate here took a snap,
so none is voided for not playing - nfl_sgp.KNOWN_BIAS, carried into the
output beside the meters.

The current season's games are replayed only once final with their snap
counts and box score on file; the rest are counted, not built.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import nfl_data as nd       # noqa: E402
import nfl_export as ne     # noqa: E402
import nfl_model as nm      # noqa: E402
import nfl_sgp as sgp       # noqa: E402
import nfl_td as nt         # noqa: E402

OUT_NAME = "sgp_backtest"
TICKETS_OUT = nd.DATA / "sgp_backtest_tickets.csv.gz"
TD_PRICES = nd.DATA / "td_price_history.csv.gz"
CHANCE_EDGES = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0)
TD_KINDS = (nt.MODEL, "base")     # the model and the base rate shipped_groups() is measured against


# ---------------------------------------------------------------- inputs

def blend_history(frame, end=None):
    """{game_id: walk-forward blend_prob} - `frame` is nm.feature_frame()."""
    wf = nm.walk_forward(frame, end=end)
    return pd.Series(wf["blend_prob"].astype(float).to_numpy(), index=frame["game_id"].to_numpy())


def td_history(td_frame, ships, player_games, preds=None, start=sgp.HISTORY_FROM, end=None):
    """The TD candidates of every game from `start` to `end`: td_frame's rows (nfl_td.build)
    with `prob` - the walk-forward chance where the row's usage group ships,
    else NaN - `group`, `rookie` (from `player_games`) and `tds` (the box
    score's label). `preds` is nt.walk_forward(td_frame) when the caller has it."""
    end = end or int(td_frame["season"].max())
    if preds is None:
        preds = nt.walk_forward(td_frame, start=start, end=end, kinds=(nt.MODEL,))
    rows = td_frame[td_frame["season"].between(start, end)].copy()
    rows["group"] = nt.frame_groups(rows)
    rows["prob"] = preds.loc[rows.index, "prob"].where(rows["group"].isin(set(ships)))
    first = sgp.first_seasons(player_games)
    rows["rookie"] = [sgp.is_rookie(first, g, s) for g, s in zip(rows["gsis_id"], rows["season"])]
    return rows[["game_id", "season", "team", "gsis_id", "name", "position", "group", "prob", "rookie", "tds"]]


def td_prices(path=TD_PRICES):
    """{(game_id, gsis_id): decimal} from a historical anytime-TD price file,
    or None when there is none on file."""
    path = Path(path)
    if not path.exists():
        return None
    p = pd.read_csv(path)
    p = p[pd.to_numeric(p["decimal"], errors="coerce") > 1.0]
    return {(g, s): float(d) for g, s, d in zip(p["game_id"], p["gsis_id"], p["decimal"])}


def game_legs(game, blend, candidates, prices=None):
    """nfl_sgp.game_legs() of one schedule row: the blend, the closing lines
    and prices (American, converted), and its TD candidates. Reads nothing
    from the game's result."""
    dec = sgp.decimal_from_american
    td = [{"gsis_id": r.gsis_id, "name": r.name, "position": r.position, "team": r.team,
           "prob": None if pd.isna(r.prob) else float(r.prob), "rookie": bool(r.rookie),
           "decimal": (prices or {}).get((game.game_id, r.gsis_id))}
          for r in candidates.itertuples(index=False)]
    return sgp.game_legs(game.home, game.away, p_home=blend, spread_line=game.spread_line,
                         home_spread_price=dec(game.home_spread_odds),
                         away_spread_price=dec(game.away_spread_odds),
                         total_line=game.total_line, over_price=dec(game.over_odds),
                         under_price=dec(game.under_odds), home_ml=dec(game.home_ml),
                         away_ml=dec(game.away_ml), td=td)


def game_outcome(game, candidates):
    """nfl_sgp.outcome() of one game: the final score, and every candidate's
    rushing + receiving TDs once the box score is in (None before)."""
    tds = candidates["tds"]
    players = None if tds.isna().any() else dict(zip(candidates["gsis_id"], tds.astype(float)))
    return sgp.outcome(game.home, game.away, game.home_score if game.played else None,
                       game.away_score if game.played else None, players)


# ---------------------------------------------------------------- the replay

def replay(games, blend, td_rows, prices=None, start=sgp.HISTORY_FROM):
    """Every graded ticket of every game from `start` that is final with its
    box score on file, plus the count of games left out and why."""
    by_game = {gid: rows for gid, rows in td_rows.groupby("game_id", sort=False)}
    tickets, left_out = [], {}
    for g in games[games["season"] >= start].itertuples(index=False):
        candidates = by_game.get(g.game_id)
        if not g.played:
            why = "not played yet"
        elif candidates is None or candidates["tds"].isna().any():
            why = "snap counts or box score not on file"
        else:
            why = None
        if why:
            left_out[why] = left_out.get(why, 0) + 1
            continue
        legs = game_legs(g, blend.get(g.game_id), candidates, prices)
        result = game_outcome(g, candidates)
        meta = {"game_id": g.game_id, "season": int(g.season), "week": int(g.week),
                "game_type": g.game_type, "home": g.home, "away": g.away}
        for t in sgp.build_tickets(legs):
            tickets.append(dict(sgp.grade_ticket(t, result), **meta))
    return tickets, left_out


def _bins(tickets, edges=CHANCE_EDGES):
    """Product chance against how often the decided tickets hit, by chance bin."""
    decided = [t for t in tickets if t["result"] in (sgp.WON, sgp.LOST)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = [t for t in decided if lo <= t["chance"] < hi or (hi == edges[-1] and t["chance"] == hi)]
        if rows:
            out.append({"lo": lo, "hi": hi, "tickets": len(rows),
                        "mean_chance": float(np.mean([t["chance"] for t in rows])),
                        "mean_chance_graded": float(np.mean([t["chance_graded"] for t in rows])),
                        "hit_rate": float(np.mean([t["result"] == sgp.WON for t in rows]))})
    return out


def periods(current):
    out = {k: list(v) for k, v in sgp.PERIODS.items()}
    out[str(current)] = [current, current]
    out["all"] = [sgp.HISTORY_FROM, current]
    return out


def measure(tickets, current, no_price_reason=None):
    """The payload's meters: per kind, per period, and the chance bins."""
    spans = periods(current)
    by_period = {name: sgp.summarise([t for t in tickets if lo <= t["season"] <= hi], no_price_reason)
                 for name, (lo, hi) in spans.items()}
    games = {name: len({t["game_id"] for t in tickets if lo <= t["season"] <= hi})
             for name, (lo, hi) in spans.items()}
    kinds = {kind: {"meters": {name: by_period[name][kind] for name in spans},
                    "chance_bins": _bins([t for t in tickets if t["kind"] == kind])}
             for kind in sgp.KINDS}
    return {"periods": spans, "games": games, "kinds": kinds}


def changed_by_groups(tickets, every_group):
    """How many built tickets the shipped-groups restriction changes: the
    replay's `tickets` beside the same games' replayed with every group's
    chance in (`every_group`)."""
    key = lambda t: (t["game_id"], t["kind"])                                  # noqa: E731
    legs = lambda t: tuple(leg["key"] for leg in t["legs"])                    # noqa: E731
    other = {key(t): legs(t) for t in every_group}
    built = [t for t in tickets if t["built"]]
    return {"tickets": len(built), "changed": sum(other.get(key(t)) != legs(t) for t in built)}


def ticket_rows(tickets):
    """One flat row per ticket, for the audit file."""
    rows = []
    for t in tickets:
        legs = t["legs"]
        rows.append({
            "game_id": t["game_id"], "season": t["season"], "week": t["week"], "game_type": t["game_type"],
            "home": t["home"], "away": t["away"], "kind": t["kind"], "built": t["built"],
            "reason": t["reason"], "legs": " + ".join(leg["label"] for leg in legs),
            "keys": "|".join(leg["key"] for leg in legs),
            "leg_chances": "|".join(f"{leg['chance']:.4f}" for leg in legs),
            "leg_results": "|".join(f"{k}={v}" for k, v in t["leg_results"].items()),
            "rookie": any(leg.get("rookie") for leg in legs),
            "chance": t["chance"], "chance_graded": t["chance_graded"], "est_payout": t["est_payout"],
            "payout": t["payout"], "result": t["result"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- run

def run(out_dir=nd.DATA, tickets_out=TICKETS_OUT, price_path=TD_PRICES):
    t0 = time.time()
    games, player_games = nd.load_games(), nd.load_player_games()
    frame = nm.feature_frame(games)
    blend = blend_history(frame)
    td_frame = nt.build(games, nt.load_lines())
    preds = nt.walk_forward(td_frame, start=sgp.HISTORY_FROM, kinds=TD_KINDS)
    ships = sorted(nt.shipped_groups(nt.report(td_frame, preds)))
    td_rows = td_history(td_frame, ships, player_games, preds)
    prices = td_prices(price_path)
    print(f"inputs in {time.time() - t0:.0f}s; TD groups shipped: {', '.join(ships)}; "
          f"TD prices on file: {'yes' if prices is not None else 'no'}")

    tickets, left_out = replay(games, blend, td_rows, prices)
    current = int(games["season"].max())
    no_price = sgp.NO_TD_PRICE if prices is None else None
    every_group, _ = replay(games, blend, td_history(td_frame, nt.GROUPS, player_games, preds), prices)
    played = [t for t in tickets if t["result"] is not None]
    payload = dict(
        measure(tickets, current, no_price),
        version=sgp.VERSION, rule="nfl/engine/nfl_sgp.py", history_from=sgp.HISTORY_FROM,
        current_season=current, stake=sgp.STAKE_UNIT,
        through=games.loc[games["game_id"].isin({t["game_id"] for t in played}), "kickoff"].max(),
        left_out=left_out, roi_reason=no_price, known_bias=sgp.KNOWN_BIAS, no_edge=sgp.NO_EDGE,
        payout_basis=sgp.EST_PAYOUT,
        td_prices={"file": str(TD_PRICES.relative_to(nd.NFL.parent)), "on_file": prices is not None,
                   "rows": 0 if prices is None else len(prices)},
        td_groups=dict(changed_by_groups(tickets, every_group), shipped=ships,
                       note=f"the usage groups given a chance were chosen on {nt.DEV[0]}-{nt.CONFIRM[1]} "
                            "walk-forward calibration (nfl_td.shipped_groups); `changed` is how many built "
                            "tickets would differ with every group's chance in"),
        inputs={"WIN": "nfl_model walk-forward blend_prob (season S fitted on seasons before S)",
                "SPREAD": "closing spread_line and its prices, games.csv.gz, proportional no-vig",
                "TOTAL": "closing total_line and its prices, games.csv.gz, proportional no-vig",
                "TD": "nfl_td walk-forward prob (season S fitted on seasons before S), shipped usage "
                      "groups only; graded on box-score rushing + receiving TDs",
                "rookie": "no player_games row in an earlier season"})
    path = ne.write(OUT_NAME, payload, out_dir=out_dir)
    ticket_rows(tickets).to_csv(tickets_out, index=False, compression="gzip")
    print(f"wrote {path} and {tickets_out} in {time.time() - t0:.0f}s")
    return payload


def stale(path=None, games=None, player_games=None, snaps=None):
    """Why the history on file needs replaying, or None when it is current:
    no file, another version of the rule, or a game final after its
    `through` that the snapshot holds the box score and snap counts of (a
    game without them would replay to the same file)."""
    path = Path(path) if path is not None else nd.DATA / f"{OUT_NAME}.json"
    if not path.exists():
        return "no history on file"
    payload = json.loads(path.read_text())
    if payload.get("version") != sgp.VERSION:
        return f"the history on file is {payload.get('version')}, the rule is {sgp.VERSION}"
    games = nd.load_games() if games is None else games
    player_games = nd.load_player_games() if player_games is None else player_games
    snaps = nd.load_snaps() if snaps is None else snaps
    through = pd.Timestamp(payload["through"]).tz_localize(None) if payload.get("through") else None
    new = games[games["played"] & (games["season"] >= sgp.HISTORY_FROM)
                & games["game_id"].isin(set(player_games["game_id"]))
                & games["game_id"].isin(set(snaps["game_id"]))]
    if through is not None:
        new = new[new["kickoff"] > through]
    return f"games final since {payload.get('through')} not in it: {len(new)}" if len(new) else None


def headline(payload):
    lines = [f"{payload['version']}: {payload['games']['all']:,} games, {payload['history_from']}-"
             f"{payload['current_season']}; left out: {payload['left_out']}"]
    for kind, k in payload["kinds"].items():
        for name, m in k["meters"].items():
            hit = "-" if m["hit_rate"] is None else f"{m['hit_rate']:.1%}"
            chance = "-" if m["mean_chance"] is None else f"{m['mean_chance']:.1%}"
            lines.append(f"  {kind:7s} {name:10s} tickets {m['tickets']:5d}  won {m['won']:5d}  "
                         f"lost {m['lost']:5d}  void {m['void']:3d}  hit {hit:>6s}  chance {chance:>6s}  "
                         f"not built {m['not_built']:4d}  roi {m['roi'] if m['roi'] is not None else m['roi_reason']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--if-stale", action="store_true",
                        help="replay only when the history on file is behind (stale())")
    args = parser.parse_args(argv)
    if args.if_stale:
        why = stale()
        if why is None:
            print(f"{sgp.VERSION}: the history on file is current; not replayed")
            return None
        print(f"{sgp.VERSION}: replaying - {why}")
    payload = run()
    print(headline(payload))
    return payload


if __name__ == "__main__":
    main()
