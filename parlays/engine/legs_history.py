"""The three sports' historical legs, one row per event, for the backtest.

Each row is that sport's SHIPPED pick, point in time, at the price on file:

  UFC     engine/experiments/parlay_legs.csv.gz - the walk-forward model
          (the shipped recipe) and each card's single price snapshot; the
          pick is the blended side (market_blend at the shipped weight).
          That file holds decided fights only: a draw, a no contest or an
          overturned result has no row there, so it is no leg and no
          underdog here (no UFC row is "void"), although live such a fight
          is on the card and its leg voids. The backtest counts them from
          the UFC archive in a caveat (sameday_backtest.ufc_no_winner).
  NFL     nfl_model.walk_forward over nfl_model.feature_frame - each season
          fitted on earlier seasons; the pick is the blend's (home at 0.5 or
          more), priced at the pick side's closing moneyline (nflverse).
  Soccer  soccer_model.shipped_pick over the decided leagues
          (soccer_leagues.in_app(), as soccer_predict's measured_leagues),
          kept for the top five only (build_sameday.SOCCER_MEASURED) - each
          league-season's pick from matches dated before its first match:
          the pre-match market recalibrated with the weights pooled over
          every European division (PICK_WEIGHTS "per_price"), the raw
          no-vig price for a league whose recalibration lost to it (D1).
          The pick, its chance and its price are the app's (soccer_predict.
          match_entry): the likeliest of the three, at the pre-match average
          for it (never the closing price), the market the no-vig pre-match
          price. A match with no pre-match price is no leg (the live
          builder takes only "posted" ones).
          WHY THIS ENGINE: the live soccer legs come from it. The legs were
          first built from main's soccer_model.pick (one recalibration for
          the five, D1 included), and once the pipeline switched the live
          leg chances moved (D1 2-3 points on six of eight matches), so the
          record shown was no longer this procedure's. `--soccer-engine`
          builds from another soccer/engine (e.g. main's, to compare).

Writes parlays/data/legs_{ufc,nfl,soccer}.csv.gz with: day (ET), key, event,
pick, opponent, p, p_model, p_market, decimal, price_source, result
(won / lost / void - void is an NFL tie; UFC has none, see above).

THE UNDERDOG SIDE (sameday.py's underdog ticket, v1) goes in its own files,
parlays/data/legs_dogs_{ufc,nfl,soccer}.csv.gz, so the pick-side files and
everything that reads them stay as they were. Per event of the pick-side
file, the market's underdog by the docstring's DOG: the side whose no-vig
chance q is under 0.5 at the same single snapshot that pays (UFC the card's
dec_red / dec_blue, NFL the closing moneylines, soccer the pre-match
average) - in soccer the less likely of the two TEAMS, never the draw - and
no row when the two sides are level (equal no-vig chances, i.e. equal
prices). Columns as the pick side's plus lot_key (sameday.dog_lot_key: NFL
the game id, soccer the match id, UFC the corners' normalised names), with
key the SAME as the event's pick-side row (so the fill leaves the dog's
event out) and pick the dog: p is that side's chance as its sport ships it
(UFC and NFL the blend's, soccer the shipped three-way chance), p_model the
model's for that side, p_market q, decimal that side's price at the same
snapshot. Every underdog is kept, whatever its price or chance: sameday.dog()
applies the 2.00-4.00 band and build_dog the floor, and the backtest's
descriptive splits can then show what the band leaves out.

Runs offline in under a minute a sport (each in its own process: --only).

TICKETS v2 (sameday.py's "TICKETS v2", LEGS v2) - `--v2` writes NEW files
only and never touches the v1 files above, which stay byte for byte what v1
was measured on (rebuilding them now would move them: the NFL schedule has
grown since). One row per event and market side, key the event's (the same
key the v1 files give that match, game or fight, so a ticket holds at most
one leg per event across markets), each with bet (sameday.LEG_MARKETS) and
bet_detail:

  legs_v2_soccer      1X2: the shipped pick of EVERY league in the app
                      (soccer_leagues.in_app()), by the same procedure as the
                      v1 soccer file - whose top-five rows these reproduce (a
                      test holds them equal) - priced where the pick is
                      priced: the pre-match average for the European
                      divisions ("pre"), the CLOSING average for the 15 extra
                      leagues in the app, whose history has no pre-match
                      price ("close"; live they pay the pre-match price).
  legs_dogs_v2_soccer the market's underdog of every such match, as the v1
                      dog file (the less likely TEAM, never the draw; none
                      when level) at the same snapshot that prices the pick.
  legs_v2_ou25        OU25: over/under 2.5 goals in each league that ships
                      over 2.5 from the market (soccer_leagues.withheld(code,
                      "market") leaves o25 in) - the side the pre-match
                      no-vig over/under price favours, its chance that no-vig
                      chance, its price the pre-match average (avg_o25 /
                      avg_u25); level prices give no leg. Graded on the total
                      goals.
  legs_v2_nfl_lines   SPREAD and TOTAL from nfl/data/games.csv.gz read raw:
                      the closing line and its closing prices (nflverse
                      home/away_spread_odds, over/under_odds). The side is
                      the one whose no-vig chance at those prices is at least
                      0.5; at level prices the shipped model's side (nfl_
                      predict.predict_game: home when the walk-forward margin
                      exceeds the line, over when the walk-forward total
                      does) at chance 0.5 - side_source "model" - and no leg
                      where that model number is missing (counted). A zero
                      spread gives no spread leg (a pick'em is the moneyline).
                      A push voids the leg.
  legs_v2_ufc_method  METHOD: each corner's "wins by KO/TKO", "by submission"
                      and "by decision" on the fights of the v1 UFC file
                      (same believable moneyline) whose six props are all
                      priced and make a believable book (parlay_profiles.
                      _method_market, overround 0-40%), 2016-08-01 to
                      sameday.PROPS_HISTORY_END. Price: the real prop
                      (parlay_legs.csv.gz's method columns). Chance: the
                      corner's shipped win chance (the blend, as the v1 leg)
                      times the six-way no-vig share of that corner's wins by
                      that type (parlay_profiles._as_method's reading);
                      p_market the six-way no-vig chance itself. est_decimal:
                      the METHOD ESTIMATOR's price (engine/method_price.py)
                      refitted on fights dated before the card's year only
                      (fit_before) - missing for 2016, which has no earlier
                      props to fit on, with the reason in est_missing.
                      Result from the archive's method class (parlay_
                      profiles.method_class of the archive's method): won
                      only when the corner won by that type - a win by DQ
                      wins no prop - and void when the fight had no winner.

WHY THE DAY IS THE V1 FILES' DAY: a v2 leg sits on the day its event's v1 leg
does (soccer the ET day of the kickoff, else the date; NFL the schedule day;
UFC the event date), so every market of one event falls on one day.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "parlays" / "data"
COLS = ["day", "key", "event", "pick", "opponent", "p", "p_model", "p_market",
        "decimal", "price_source", "result"]
DOG_COLS = ["day", "key", "lot_key", "event", "pick", "opponent", "p", "p_model", "p_market",
            "decimal", "price_source", "result"]


# build_sameday.SOCCER_MEASURED (a test holds them equal): not imported, since
# sameday puts the UFC engine on the path and it has a package named
# experiments, as the soccer engine does.
SOCCER_MEASURED = ("E0", "SP1", "I1", "D1", "F1")
DAY_TZ = ZoneInfo("America/New_York")      # sameday.DAY_TZ


def _et_day(kickoff):
    """sameday.et_day of a naive-UTC kickoff: the live builder's day."""
    t = pd.Timestamp(kickoff)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(DAY_TZ).strftime("%Y-%m-%d")


def _devig(a, b):
    ia, ib = 1.0 / a, 1.0 / b
    return ia / (ia + ib)


def _dec(american):
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def _sameday():
    """sameday, for dog_lot_key - the one lot key live and here. Imported
    after the sport's engine has loaded and with sys.path put back: sameday
    puts the UFC engine first on the path, and the soccer engine has a
    package of the same name (experiments)."""
    before = list(sys.path)
    sys.path.insert(0, str(ROOT / "parlays" / "engine"))
    try:
        import sameday
    finally:
        sys.path[:] = before
    return sameday


def _side(a_chance, b_chance):
    """Which of two sides the market makes the underdog: "a", "b", or None
    when they are level (equal no-vig chances - equal prices)."""
    if a_chance < b_chance:
        return "a"
    if b_chance < a_chance:
        return "b"
    return None


def ufc():
    sys.path.insert(0, str(ROOT / "engine"))
    import market_blend
    d = pd.read_csv(ROOT / "engine" / "experiments" / "parlay_legs.csv.gz")
    d = d[d["dec_red"].notna() & d["dec_blue"].notna() & d["winner"].ne("unresolved")]
    sdy = _sameday()
    rows, dogs = [], []
    for r in d.itertuples():
        over = 1.0 / r.dec_red + 1.0 / r.dec_blue - 1.0
        if over < 0 or over > 0.15:            # the same believable-book check parlay_profiles uses
            continue
        mk_red = _devig(r.dec_red, r.dec_blue)
        pb_red = market_blend.blend(r.p_model, mk_red)
        red = pb_red >= 0.5
        pick, opp = (r.red, r.blue) if red else (r.blue, r.red)
        # never None today (parlay_legs keeps decided fights), kept so a
        # file that does carry a fight without a winner voids its leg
        winner = None if pd.isna(r.winner) else r.winner
        result = "void" if winner is None else ("won" if winner == ("red" if red else "blue") else "lost")
        rows.append({"day": r.date, "key": r.fight_id, "event": r.event, "pick": pick, "opponent": opp,
                     "p": pb_red if red else 1 - pb_red,
                     "p_model": r.p_model if red else 1 - r.p_model,
                     "p_market": mk_red if red else 1 - mk_red,
                     "decimal": r.dec_red if red else r.dec_blue,
                     "price_source": r.price_source, "result": result})
        # the market's underdog at the same snapshot: possibly the blend's own pick
        side = _side(mk_red, 1 - mk_red)
        if side is None:
            continue
        dr = side == "a"
        dogs.append({"day": r.date, "key": r.fight_id,
                     "lot_key": sdy.dog_lot_key("UFC", corners=[r.red, r.blue]), "event": r.event,
                     "pick": r.red if dr else r.blue, "opponent": r.blue if dr else r.red,
                     "p": pb_red if dr else 1 - pb_red,
                     "p_model": r.p_model if dr else 1 - r.p_model,
                     "p_market": mk_red if dr else 1 - mk_red,
                     "decimal": r.dec_red if dr else r.dec_blue, "price_source": r.price_source,
                     "result": "void" if winner is None else ("won" if winner == ("red" if dr else "blue")
                                                              else "lost")})
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(dogs, columns=DOG_COLS)


def nfl():
    sys.path.insert(0, str(ROOT / "nfl" / "engine"))
    import nfl_data as nd
    import nfl_model as nm
    frame = nm.feature_frame()
    wf = nm.walk_forward(frame)
    f = frame.join(wf)
    f = f[(f["season"] >= 2016) & f["played"].astype(bool) & f["home_ml"].notna() & f["away_ml"].notna()
          & f["blend_prob"].notna()]
    sdy = _sameday()
    rows, dogs = [], []
    for r in f.itertuples():
        dh, da = _dec(r.home_ml), _dec(r.away_ml)
        mk_home = _devig(dh, da)
        home = r.blend_prob >= 0.5
        res = r.result
        result = "void" if res == 0 else ("won" if (res > 0) == home else "lost")
        day = (nd.eastern_day(r.kickoff).strftime("%Y-%m-%d") if hasattr(nd.eastern_day(r.kickoff), "strftime")
               else str(nd.eastern_day(r.kickoff))[:10])
        rows.append({"day": day,
                     "key": r.game_id, "event": f"{r.away} @ {r.home}",
                     "pick": r.home if home else r.away, "opponent": r.away if home else r.home,
                     "p": r.blend_prob if home else 1 - r.blend_prob,
                     "p_model": r.model_prob if home else 1 - r.model_prob,
                     "p_market": mk_home if home else 1 - mk_home,
                     "decimal": dh if home else da, "price_source": "close", "result": result})
        # the market's underdog at the closing moneylines: possibly the blend's own pick
        side = _side(mk_home, 1 - mk_home)
        if side is None:
            continue
        dh_ = side == "a"
        dogs.append({"day": day, "key": r.game_id, "lot_key": sdy.dog_lot_key("NFL", game_id=r.game_id),
                     "event": f"{r.away} @ {r.home}",
                     "pick": r.home if dh_ else r.away, "opponent": r.away if dh_ else r.home,
                     "p": r.blend_prob if dh_ else 1 - r.blend_prob,
                     "p_model": r.model_prob if dh_ else 1 - r.model_prob,
                     "p_market": mk_home if dh_ else 1 - mk_home,
                     "decimal": dh if dh_ else da, "price_source": "close",
                     "result": "void" if res == 0 else ("won" if (res > 0) == dh_ else "lost")})
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(dogs, columns=DOG_COLS)


def soccer(engine_dir):
    """The top five's legs by the shipped pick of the soccer engine at
    `engine_dir` (its own soccer_data reads the data directory beside it):
    soccer_model.shipped_pick over every league the app carries by decision
    (soccer_leagues.in_app(), the leagues soccer_predict computes the pick
    over), on a frame of every measured league (the weights are pooled
    over the leagues priced the same way, shown or not), kept for the top
    five. The model's own chances are not needed: a leg is a priced match,
    and a priced match's pick never reads them (pick_leagues), so they are
    NaN here as in main's build. The day is the live builder's: the ET day
    of the kickoff."""
    sys.path.insert(0, str(engine_dir))
    import soccer_backtest as bt
    import soccer_data as sd
    import soccer_leagues as slg
    import soccer_model as smod
    m = sd.load_matches(leagues="all")
    nan = pd.DataFrame(np.nan, index=m.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    leagues = slg.in_app()
    p, _ = smod.shipped_pick(m, nan, leagues)
    q = bt.no_vig(m, "pre")
    keep = m["league"].isin(SOCCER_MEASURED) & ~p["model_only"].astype(bool) & m["result"].notna() \
        & (m["season"] >= 2016)
    sdy = _sameday()
    rows, dogs = [], []
    for i in m.index[keep]:
        r = m.loc[i]
        probs = {"H": p.at[i, "p_h"], "D": p.at[i, "p_d"], "A": p.at[i, "p_a"]}
        if any(pd.isna(v) for v in probs.values()) or p.at[i, "price"] != "pre":
            continue
        k = max(probs, key=probs.get)
        dec = r[f"avg_{k.lower()}"]
        if pd.isna(dec):
            continue
        day = _et_day(r["kickoff"]) if pd.notna(r["kickoff"]) else str(r["date"])[:10]
        rows.append({"day": day, "key": r["match_id"], "event": f"{r['home']} v {r['away']}",
                     "pick": {"H": r["home"], "D": "Draw", "A": r["away"]}[k],
                     "opponent": {"H": r["away"], "D": f"{r['home']} v {r['away']}", "A": r["home"]}[k],
                     "p": probs[k], "p_model": np.nan,
                     "p_market": q.at[i, f"p_{k.lower()}"],
                     "decimal": float(dec), "price_source": "pre",
                     "result": "won" if r["result"] == k else "lost"})
        # the less likely of the two TEAMS by the pre-match no-vig - never the draw
        side = _side(q.at[i, "p_h"], q.at[i, "p_a"])
        if side is None:
            continue
        j = "H" if side == "a" else "A"
        dogs.append({"day": day, "key": r["match_id"], "lot_key": sdy.dog_lot_key("Soccer", match_id=r["match_id"]),
                     "event": f"{r['home']} v {r['away']}",
                     "pick": r["home"] if j == "H" else r["away"], "opponent": r["away"] if j == "H" else r["home"],
                     "p": probs[j], "p_model": np.nan, "p_market": q.at[i, f"p_{j.lower()}"],
                     "decimal": float(r[f"avg_{j.lower()}"]), "price_source": "pre",
                     "result": "won" if r["result"] == j else "lost"})
    basis = p.loc[keep, "basis"].groupby(m.loc[keep, "league"]).agg(lambda b: ",".join(sorted(set(b))))
    print("soccer  basis by league:", dict(basis))
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(dogs, columns=DOG_COLS)


# ---------------------------------------------------------------- tickets v2

V2_COLS = ["day", "key", "lot_key", "event", "bet", "bet_detail", "pick", "opponent", "p", "p_model",
           "p_market", "decimal", "price_source", "result"]
SOCCER_V2_COLS = V2_COLS + ["league", "group", "day_source"]
NFL_LINE_COLS = V2_COLS + ["line", "side_source", "overround"]
METHOD_COLS = V2_COLS + ["corner", "p_win", "share", "est_decimal", "est_missing", "est_fit_last",
                         "method_overround"]
# The 2016 cards have no earlier props in the fit rows (method_price.fit_before):
# said on every such row rather than left blank.
NO_FIT_2016 = ("no earlier props to fit the estimator on: the fit rows start 2016-01-02, so "
               "method_price.fit_before(2016) is None")


def _goal_result(side, goals, line=2.5):
    """"won" / "lost" for an over or under at a half line (no push at 2.5)."""
    over = goals > line
    return "won" if over == (side == "Over") else "lost"


def soccer_v2(engine_dir):
    """The three soccer v2 files (module docstring): 1X2 picks of every
    league in the app, their underdogs, and the over/under 2.5 legs."""
    sys.path.insert(0, str(engine_dir))
    import soccer_backtest as bt
    import soccer_data as sd
    import soccer_leagues as slg
    import soccer_model as smod
    m = sd.load_matches(leagues="all")
    nan = pd.DataFrame(np.nan, index=m.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    leagues = slg.in_app()
    p, _ = smod.shipped_pick(m, nan, leagues)
    q = {"pre": bt.no_vig(m, "pre"), "close": bt.no_vig(m, "close")}
    cols = {"pre": "avg_", "close": "avgc_"}
    keep = m["league"].isin(leagues) & ~p["model_only"].astype(bool) & m["result"].notna() \
        & (m["season"] >= 2016)
    ou_leagues = [c for c in leagues if "o25" not in slg.withheld(c, "market")]
    qo = bt.no_vig_total(m)
    sdy = _sameday()
    rows, dogs, ous = [], [], []
    for i in m.index[keep]:
        r = m.loc[i]
        src = p.at[i, "price"]
        probs = {"H": p.at[i, "p_h"], "D": p.at[i, "p_d"], "A": p.at[i, "p_a"]}
        if any(pd.isna(v) for v in probs.values()) or src not in cols:
            continue
        k = max(probs, key=probs.get)
        dec = r[f"{cols[src]}{k.lower()}"]
        if pd.isna(dec):
            continue
        timed = pd.notna(r["kickoff"])
        day = _et_day(r["kickoff"]) if timed else str(r["date"])[:10]
        base = {"day": day, "key": r["match_id"], "event": f"{r['home']} v {r['away']}", "league": r["league"],
                "group": slg.group_of(r["league"]), "day_source": "kickoff" if timed else "date"}
        rows.append(dict(base, lot_key=None, bet="1X2", bet_detail=k,
                         pick={"H": r["home"], "D": "Draw", "A": r["away"]}[k],
                         opponent={"H": r["away"], "D": f"{r['home']} v {r['away']}", "A": r["home"]}[k],
                         p=probs[k], p_model=np.nan, p_market=q[src].at[i, f"p_{k.lower()}"], decimal=float(dec),
                         price_source=src, result="won" if r["result"] == k else "lost"))
        # the less likely of the two TEAMS at the snapshot that prices the pick
        side = _side(q[src].at[i, "p_h"], q[src].at[i, "p_a"])
        if side is not None:
            j = "H" if side == "a" else "A"
            dogs.append(dict(base, lot_key=sdy.dog_lot_key("Soccer", match_id=r["match_id"]),
                             pick=r["home"] if j == "H" else r["away"],
                             opponent=r["away"] if j == "H" else r["home"],
                             p=probs[j], p_model=np.nan, p_market=q[src].at[i, f"p_{j.lower()}"],
                             decimal=float(r[f"{cols[src]}{j.lower()}"]), price_source=src,
                             result="won" if r["result"] == j else "lost"))
        # over/under 2.5 where the league ships it from the market, at the pre-match average
        qo_i = qo.at[i]
        if r["league"] in ou_leagues and pd.notna(qo_i) and qo_i != 0.5 \
                and pd.notna(r["hg"]) and pd.notna(r["ag"]):
            o = "Over" if qo_i > 0.5 else "Under"
            ous.append(dict(base, lot_key=None, bet="OU25", bet_detail=f"{o} 2.5", pick=o,
                            opponent="Under" if o == "Over" else "Over",
                            p=qo_i if o == "Over" else 1 - qo_i, p_model=np.nan,
                            p_market=qo_i if o == "Over" else 1 - qo_i,
                            decimal=float(r["avg_o25" if o == "Over" else "avg_u25"]), price_source="pre",
                            result=_goal_result(o, float(r["hg"]) + float(r["ag"]))))
    legs = pd.DataFrame(rows, columns=SOCCER_V2_COLS)
    dog_cols = [c for c in SOCCER_V2_COLS if c not in ("bet", "bet_detail")]
    print("soccer v2  leagues:", len(leagues), " over/under 2.5 from the market:", len(ou_leagues),
          " priced by:", dict(legs["price_source"].value_counts()),
          " day from:", dict(legs["day_source"].value_counts()))
    return {"legs_v2_soccer": legs, "legs_dogs_v2_soccer": pd.DataFrame(dogs, columns=dog_cols),
            "legs_v2_ou25": pd.DataFrame(ous, columns=SOCCER_V2_COLS)}


def nfl_v2():
    """legs_v2_nfl_lines (module docstring): the closing spread and total,
    read raw from nfl/data/games.csv.gz, the side by the no-vig prices and at
    level prices the shipped model's side from the walk-forward (each season
    fitted on earlier seasons) - point in time."""
    sys.path.insert(0, str(ROOT / "nfl" / "engine"))
    import nfl_data as nd
    import nfl_model as nm
    frame = nm.feature_frame()
    wf = frame[["game_id"]].join(nm.walk_forward(frame)).set_index("game_id")
    g = pd.read_csv(ROOT / "nfl" / "data" / "games.csv.gz", low_memory=False)
    g = g[(g["season"] >= 2016) & g["result"].notna() & g["total"].notna()]
    rows, counts = [], Counter()
    for r in g.itertuples():
        home, away = nd.franchise(r.home_team), nd.franchise(r.away_team)
        day = nd.eastern_day(nd.kickoff_utc(r.gameday, r.gametime))
        base = {"day": day, "key": r.game_id, "lot_key": None, "event": f"{away} @ {home}",
                "p_model": np.nan, "price_source": "close"}
        model = wf.loc[r.game_id] if r.game_id in wf.index else None
        for bet in ("SPREAD", "TOTAL"):
            if bet == "SPREAD":
                line, a_odds, b_odds = r.spread_line, r.home_spread_odds, r.away_spread_odds
                if pd.isna(line) or float(line) == 0.0:
                    counts["spread: zero or no line"] += 1
                    continue
            else:
                line, a_odds, b_odds = r.total_line, r.over_odds, r.under_odds
                if pd.isna(line):
                    counts["total: no line"] += 1
                    continue
            if pd.isna(a_odds) or pd.isna(b_odds):
                counts[f"{bet.lower()}: no price"] += 1
                continue
            da, db = _dec(a_odds), _dec(b_odds)
            qa = _devig(da, db)
            if da == db:
                # level prices: the shipped model's side at chance 0.5, where it exists
                if bet == "SPREAD":
                    m_ = None if model is None or pd.isna(model["margin"]) else float(model["margin"])
                    first = None if m_ is None else (m_ - float(line) > 0)
                else:
                    m_ = None if model is None or pd.isna(model["total_pred"]) else float(model["total_pred"])
                    first = None if m_ is None else (m_ - float(line) > 0)
                if first is None:
                    counts[f"{bet.lower()}: level, no model number"] += 1
                    continue
                counts[f"{bet.lower()}: level, model side"] += 1
                chance, source = 0.5, "model"
            else:
                first, chance, source = qa > 0.5, max(qa, 1 - qa), "market"
                counts[f"{bet.lower()}: market side"] += 1
            if bet == "SPREAD":
                margin = float(r.result) - float(line)              # > 0: home covered
                team, opp = (home, away) if first else (away, home)
                hcap = -float(line) if first else float(line)
                pick, detail = team, f"{team} {hcap:+g}"
                result = "void" if margin == 0 else ("won" if (margin > 0) == first else "lost")
            else:
                over = float(r.total) - float(line)
                pick, opp = ("Over", "Under") if first else ("Under", "Over")
                detail = f"{pick} {float(line):g}"
                result = "void" if over == 0 else ("won" if (over > 0) == first else "lost")
            rows.append(dict(base, bet=bet, bet_detail=detail, pick=pick, opponent=opp, p=chance,
                             p_market=chance, decimal=da if first else db, result=result,
                             line=float(line) if bet == "TOTAL" else hcap, side_source=source,
                             overround=1 / da + 1 / db - 1))
    print("nfl v2 lines:", dict(sorted(counts.items())))
    return {"legs_v2_nfl_lines": pd.DataFrame(rows, columns=NFL_LINE_COLS)}


def ufc_v2():
    """legs_v2_ufc_method (module docstring): six props a fight, on the v1
    UFC file's fights, priced, chanced and graded as the docstring says; the
    estimate from method_price.fit_before(year) - one refit per year, on
    earlier years' fights only."""
    sys.path.insert(0, str(ROOT / "engine"))
    import market_blend
    import method_price as mpr
    import parlay_profiles as pp
    sdy = _sameday()
    d = pd.read_csv(ROOT / "engine" / "experiments" / "parlay_legs.csv.gz")
    d = d[d["dec_red"].notna() & d["dec_blue"].notna() & d["winner"].ne("unresolved")
          & (d["date"] >= "2016-08-01") & (d["date"] <= sdy.PROPS_HISTORY_END)]
    arch = pd.read_csv(ROOT / "engine" / "data" / "UFC_with_mmr_rebuilt_dedup.csv",
                       usecols=["fight_id", "method"], low_memory=False).drop_duplicates("fight_id")
    # a plain dict comprehension: Series.map turns method_class's None into NaN
    method_of = {f: pp.method_class(t) for f, t in zip(arch["fight_id"], arch["method"])}
    fit = {r["fight_id"]: r for r in mpr.fit_rows()}
    rows_all = list(fit.values())
    models = {}
    rows, counts = [], Counter()
    for r in d.itertuples():
        over = 1.0 / r.dec_red + 1.0 / r.dec_blue - 1.0
        if over < 0 or over > 0.15:            # the v1 UFC file's believable-book check
            continue
        mm = pp._method_market({c: getattr(r, c) for c in ("ko_red", "ko_blue", "sub_red", "sub_blue",
                                                            "decision_red", "decision_blue")})
        if mm is None:
            counts["fights without a believable six-way book"] += 1
            continue
        counts["fights with props"] += 1
        mk_red = _devig(r.dec_red, r.dec_blue)
        pb_red = market_blend.blend(r.p_model, mk_red)
        year = int(str(r.date)[:4])
        if year not in models:
            models[year] = mpr.fit_before(year, rows_all)
        model, est, why = models[year], None, None
        if model is None:
            why = NO_FIT_2016 if year == 2016 else f"method_price.fit_before({year}) is None"
        elif r.fight_id not in fit:
            why = "no closing moneyline pair or six-way book in method_price.fit_rows for this fight"
        else:
            est = mpr.predict(mpr.features_of(fit[r.fight_id]), model)
        winner = None if pd.isna(r.winner) else r.winner
        how = method_of.get(r.fight_id)
        if winner is not None and how is None:
            counts["wins by DQ or another unclassed method (every prop of the fight lost)"] += 1
        book = sum(1.0 / getattr(r, f"{c}_{s}") for c in ("ko", "sub", "decision") for s in ("red", "blue")) - 1
        for side, other in (("red", "blue"), ("blue", "red")):
            p_win = pb_red if side == "red" else 1 - pb_red
            pm_win = r.p_model if side == "red" else 1 - r.p_model
            tot = sum(mm[side].values())
            for t, col in (("KO/TKO", "ko"), ("Submission", "sub"), ("Decision", "decision")):
                share = mm[side][t] / tot
                result = "void" if winner is None else (
                    "won" if winner == side and how == t else "lost")
                rows.append({"day": r.date, "key": r.fight_id,
                             "lot_key": sdy.method_lot_key([r.red, r.blue], getattr(r, side), t),
                             "event": r.event, "bet": "METHOD", "bet_detail": t,
                             "pick": getattr(r, side), "opponent": getattr(r, other),
                             "p": p_win * share, "p_model": pm_win * share, "p_market": mm[side][t],
                             "decimal": float(getattr(r, f"{col}_{side}")), "price_source": "close",
                             "result": result, "corner": side, "p_win": p_win, "share": share,
                             "est_decimal": None if est is None else est[side][t]["decimal"],
                             "est_missing": why, "est_fit_last": None if model is None else model["fit_last"],
                             "method_overround": book})
    out = pd.DataFrame(rows, columns=METHOD_COLS)
    counts["props without an estimate"] = int(out["est_decimal"].isna().sum())
    print("ufc v2 method:", dict(counts))
    return {"legs_v2_ufc_method": out}


def write_v2(frames):
    """Each v2 frame sorted and gzipped with mtime 0 (the same rows give the
    same bytes)."""
    for name, df in frames.items():
        sort = ["day", "key"] + [c for c in ("bet", "corner", "bet_detail") if c in df.columns]
        df = df.sort_values(sort, kind="mergesort").reset_index(drop=True)
        df.to_csv(DATA / f"{name}.csv.gz", index=False, float_format="%.6f",
                  compression={"method": "gzip", "mtime": 0})
        won = np.mean(df["result"] == "won") if len(df) else float("nan")
        print(f"{name:20s} {len(df):7,d} rows  {df['day'].min()} .. {df['day'].max()}  won {won:.3f}  "
              f"mean p {df['p'].mean():.3f}  mean p_market {df['p_market'].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soccer-engine", default=str(ROOT / "soccer" / "engine"),
                    help="the soccer/engine to build the soccer legs from (default this checkout's, "
                         "whose pick the live legs take)")
    ap.add_argument("--only", choices=("ufc", "nfl", "soccer"),
                    help="build one sport in this process (each sport runs in its own process: "
                         "the UFC and soccer engines both have a package named experiments)")
    ap.add_argument("--v2", action="store_true",
                    help="write the tickets v2 files (legs_v2_*, legs_dogs_v2_soccer) only; the v1 files "
                         "are left untouched")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.only is None:
        import subprocess
        for name in ("ufc", "nfl", "soccer"):
            subprocess.run([sys.executable, __file__, "--soccer-engine", args.soccer_engine,
                            "--only", name] + (["--v2"] if args.v2 else []), check=True)
        return
    if args.v2:
        build = {"ufc": ufc_v2, "nfl": nfl_v2, "soccer": lambda: soccer_v2(Path(args.soccer_engine))}[args.only]
        write_v2(build())
        return
    build = {"ufc": ufc, "nfl": nfl, "soccer": lambda: soccer(Path(args.soccer_engine))}[args.only]
    picks, dogs = build()
    for name, df in ((f"legs_{args.only}", picks), (f"legs_dogs_{args.only}", dogs)):
        df = df.sort_values(["day", "key"]).reset_index(drop=True)
        # mtime 0: the same rows give the same bytes, so a rebuild that changes
        # nothing shows no change
        df.to_csv(DATA / f"{name}.csv.gz", index=False, float_format="%.6f",
                  compression={"method": "gzip", "mtime": 0})
        print(f"{name:17s} {len(df):6,d} rows  {df['day'].min()} .. {df['day'].max()}  "
              f"won {np.mean(df['result'] == 'won'):.3f}  mean p {df['p'].mean():.3f}  "
              f"mean p_market {df['p_market'].mean():.3f}")


if __name__ == "__main__":
    main()
