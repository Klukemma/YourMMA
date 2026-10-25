"""Predict the soccer matches still to play and write everything the app shows.

    python soccer/engine/soccer_predict.py            # sync this season first
    python soccer/engine/soccer_predict.py --no-sync  # use the snapshot as it is

One run:
  1. refreshes this season's results and the fixtures file - the matches
     still to play, with the bookmakers' Friday/Tuesday prices - from
     football-data.co.uk;
  2. recalibrates the market's chances season by season and runs the model
     over every match since 2005-06, finished or not, each predicted only
     from matches on earlier days - exactly as the backtest does, so the
     record the app prints is the record of this procedure (soccer_model
     says what is shown and why);
  3. logs a pick for every top-league match in the next week that has not
     kicked off (prediction_history.json) and grades every logged pick
     whose result is in;
  4. writes soccer/app/data/{matches,tables,performance}.json and the
     standalone page.

A PICK IS LOGGED ONLY BEFORE KICKOFF. Re-running before a match updates its
entry; once the match has started the entry is frozen.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import soccer_backtest as bt
import soccer_data as sd
import soccer_export as sx
import soccer_form as sf
import soccer_log as sl
import soccer_model as smod
import soccer_teams as st

AHEAD = timedelta(days=8)       # how far ahead the app shows fixtures
BEHIND = timedelta(days=4)      # how far back it shows full-time results
VALUE_EDGES = (0.02, 0.05, 0.08)
MIN_GATE_BETS = 100
CAL_BINS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]


def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def combined(matches, fixtures, now):
    """History plus the fixtures still to play, in date order. A fixture
    already in the results file is a played match and comes from there."""
    up = fixtures[fixtures["kickoff"] > now].drop(columns=["fetched_at"], errors="ignore")
    up = up[~up["match_id"].isin(matches["match_id"])]
    # A match already played - perhaps earlier than the fixtures file says -
    # is in the results under its own date; never pick it again.
    played = set(zip(matches.loc[matches["result"].notna(), "league"], matches.loc[matches["result"].notna(), "season"],
                     matches.loc[matches["result"].notna(), "home"], matches.loc[matches["result"].notna(), "away"]))
    up = up[[k not in played for k in zip(up["league"], up["season"], up["home"], up["away"])]]
    frame = pd.concat([matches, up], ignore_index=True)
    return sd.order(frame)


def stamp(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------- the record

def _row(frame, p, sel):
    if not sel.any():
        return {"matches": 0}
    m, q = frame[sel], p[sel]
    mkt = bt.no_vig(m, "pre")
    return {"matches": int(sel.sum()),
            "accuracy": bt.accuracy(q, m["result"]), "log_loss": bt.log_loss(q, m["result"]),
            "brier": bt.brier(q, m["result"]),
            "market_accuracy": bt.accuracy(mkt, m["result"]), "market_log_loss": bt.log_loss(mkt, m["result"])}


def backtest_summary(frame, preds, blended, weights, goals_used):
    """What the Record tab shows: the model alone and the blend against the
    market, DEV and CONFIRM never pooled, plus by league, goals, betting
    the disagreements and calibration."""
    periods = {"dev": list(bt.DEV), "confirm": list(bt.CONFIRM)}
    has = bt.scored(frame) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1) & ~blended["model_only"]
    result = {"model": {}, "blend": {}}
    for lo, hi in (bt.DEV, bt.CONFIRM):
        sel = has & frame["season"].between(lo, hi)
        key = f"{lo}-{hi}"
        result["model"][key] = _row(frame, preds, sel)
        result["blend"][key] = _row(frame, blended, sel)
    conf = has & frame["season"].between(*bt.CONFIRM)
    by_league = []
    for code in sd.TOP:
        sel = conf & (frame["league"] == code)
        if sel.any():
            r = _row(frame, blended, sel)
            by_league.append(dict(r, code=code, name=sd.LEAGUES[code][0]))
    total = frame["hg"] + frame["ag"]
    over = {}
    for lo, hi in (bt.DEV, bt.CONFIRM):
        # The model's own chance against the market's (which is the one shown).
        sel = (has & frame["season"].between(lo, hi) & preds["p_o25"].notna()
               & bt.no_vig_total(frame).notna() & total.notna())
        if not sel.any():
            continue
        y = (total[sel] > 2.5).astype(float)
        over[f"{lo}-{hi}"] = {"matches": int(sel.sum()),
                              "log_loss": bt.binary_log_loss(preds.loc[sel, "p_o25"], y),
                              "market_log_loss": bt.binary_log_loss(bt.no_vig_total(frame[sel]), y)}
    # Betting the disagreements: the RAW model (no market in it) against the
    # price - an edge measured against a number that already contains the
    # price is the price measured against itself.
    edges, enabled = {}, None
    for e in VALUE_EDGES:
        per = {}
        for lo, hi in (bt.DEV, bt.CONFIRM):
            sel = has & frame["season"].between(lo, hi)
            per[f"{lo}-{hi}"] = bt.flat_roi(frame[sel], preds[sel], prices="best", min_edge=e)
        edges[str(e)] = per
        if enabled is None and all(r["bets"] >= MIN_GATE_BETS and r["roi"] is not None and r["roi"] > 0
                                   for r in per.values()):
            enabled = e
    pick_p = blended.loc[conf, ["p_h", "p_d", "p_a"]].max(axis=1)
    pick = blended.loc[conf, ["p_h", "p_d", "p_a"]].to_numpy(float).argmax(axis=1)
    hit = (np.asarray(bt.RESULTS)[pick] == frame.loc[conf, "result"].to_numpy())
    cal = []
    for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
        b = (pick_p >= lo) & (pick_p < hi)
        if b.sum() >= 50:
            cal.append({"lo": lo, "hi": min(hi, 1.0), "picks": int(b.sum()),
                        "predicted": float(pick_p[b].mean()), "observed": float(hit[b.to_numpy()].mean())})
    last = max(weights) if weights else None
    return {"periods": periods, "result": result, "by_league": by_league, "over_2_5": over,
            "value": {"edges": edges, "enabled_at": enabled}, "calibration": cal,
            "model": smod.MODEL, "weights": {"season": last, "w": weights.get(last)} if last else None}


def live_record(log):
    rec = sl.record(log)
    weeks = {}
    for p in log["picks"].values():
        if not p.get("final") or p.get("correct") is None:
            continue
        k = pd.Timestamp(p["kickoff"]).tz_localize(None)
        monday = (k - pd.Timedelta(days=k.weekday())).normalize()
        w = weeks.setdefault(monday, {"label": f"Week of {monday:%b} {monday.day}", "settled": 0, "correct": 0})
        w["settled"] += 1
        w["correct"] += int(bool(p["correct"]))
    rec["by_week"] = [weeks[k] for k in sorted(weeks)]
    return rec


# ------------------------------------------------------------- one match

def three(df_row):
    return {"p_h": float(df_row["p_h"]), "p_d": float(df_row["p_d"]), "p_a": float(df_row["p_a"])}


def match_entry(row, model_p, blend_p, model_only, p_o25, market_o25, matches, now, priced_at):
    probs = {"H": blend_p["p_h"], "D": blend_p["p_d"], "A": blend_p["p_a"]}
    has = all(v == v and v is not None for v in probs.values())
    pick = max(probs, key=probs.get) if has else None
    mkt = bt.no_vig(pd.DataFrame([row]), "pre").iloc[0]
    has_mkt = mkt.notna().all()
    entry = {
        "match_id": row["match_id"], "league": row["league"], "season": int(row["season"]),
        "kickoff": stamp(row["kickoff"]), "home_key": row["home"], "away_key": row["away"],
        "home": st.display(row["home"]), "away": st.display(row["away"]),
        "p_h": probs["H"] if has else None, "p_d": probs["D"] if has else None, "p_a": probs["A"] if has else None,
        "pick": pick, "pick_prob": probs[pick] if pick else None,
        "model": three(model_p) if has else None,
        "market": {"p_h": float(mkt.p_h), "p_d": float(mkt.p_d), "p_a": float(mkt.p_a)} if has_mkt and not model_only else None,
        "prices": {"avg": {"h": row.get("avg_h"), "d": row.get("avg_d"), "a": row.get("avg_a")},
                   "best": {"h": row.get("max_h"), "d": row.get("max_d"), "a": row.get("max_a")}},
        "p_o25": p_o25, "model_o25": float(model_p["p_o25"]) if model_p["p_o25"] == model_p["p_o25"] else None,
        "market_o25": market_o25,
        # When these prices were first fetched (soccer_data.first_seen), and
        # only where there are prices.
        "prices_at": stamp(priced_at) if has_mkt else None,
        "home_form": sx.form_payload(matches, row["home"], row["date"]),
        "away_form": sx.form_payload(matches, row["away"], row["date"]),
    }
    pos = sx.positions(matches, row["league"], int(row["season"]), row["date"])
    entry["home_position"], entry["away_position"] = pos.get(row["home"]), pos.get(row["away"])
    if not has:
        entry["no_pick_reason"] = "The model has not seen enough of these clubs yet."
    return entry


def log_fields(e):
    """What the log keeps of a match entry."""
    return dict(match_id=e["match_id"], league=e["league"], season=e["season"], kickoff=e["kickoff"],
                home=e["home_key"], away=e["away_key"], pick=e["pick"], pick_prob=e["pick_prob"],
                p_h=e["p_h"], p_d=e["p_d"], p_a=e["p_a"], model=e["model"], market=e["market"],
                prices=e["prices"], p_o25=e["p_o25"], model_o25=e["model_o25"], market_o25=e["market_o25"],
                prices_at=e["prices_at"], price_source="football-data.co.uk")


def logged_entry(p, matches):
    """A logged pick whose match has kicked off, for the Matches tab: the
    pick as it was logged before kickoff, beside the result once it is in
    ("final") and until then as "started" - never dropped in between."""
    date = str(p["kickoff"])[:10]
    out = {
        "match_id": p["match_id"], "league": p["league"], "season": p["season"], "kickoff": p["kickoff"],
        "status": "final" if p.get("final") else "started",
        "home": st.display(p["home"]), "away": st.display(p["away"]),
        "p_h": p.get("p_h"), "p_d": p.get("p_d"), "p_a": p.get("p_a"), "pick": p.get("pick"),
        "pick_prob": p.get("pick_prob"), "model": p.get("model"), "market": p.get("market"),
        "prices": p.get("prices"), "p_o25": p.get("p_o25"), "model_o25": p.get("model_o25"),
        "market_o25": p.get("market_o25"), "prices_at": p.get("prices_at"), "logged_at": p.get("last_logged"),
        "home_form": sx.form_payload(matches, p["home"], date),
        "away_form": sx.form_payload(matches, p["away"], date),
    }
    pos = sx.positions(matches, p["league"], int(p["season"]), date)
    out["home_position"], out["away_position"] = pos.get(p["home"]), pos.get(p["away"])
    if p.get("final"):
        out["score"] = {"home": p["final"]["home"], "away": p["final"]["away"]}
        out["correct"] = p.get("correct")
    return out


# ------------------------------------------------------------- the run

def run(sync=True, now=None, out_dir=sx.APP_DATA, log_path=sl.LOG, log=print):
    now = pd.Timestamp(now) if now is not None else now_utc()
    if sync:
        log("1. sync")
        sd.sync(log=lambda m: log(m))
    matches = sd.load_matches()
    fixtures = sd.load_fixtures()
    fetched_at = fixtures["fetched_at"].max() if len(fixtures) else None
    priced_at = dict(zip(fixtures["match_id"], fixtures["fetched_at"])) if len(fixtures) else {}
    frame = combined(matches, fixtures, now)
    log(f"2. model ({smod.MODEL}) over {len(frame):,} matches")
    preds = smod.predict(frame)
    blended, weights = smod.pick(frame, preds)
    goals_used = smod.goals(frame, preds)
    ratings = {team: r["egd"] for team, r in (preds.attrs.get("ratings") or {}).items()}
    mkt_o25 = bt.no_vig_total(frame)

    upcoming = frame["league"].isin(sd.TOP) & frame["result"].isna() & (frame["kickoff"] > now) \
        & (frame["kickoff"] <= now + AHEAD)
    entries = []
    for i in frame.index[upcoming]:
        row = frame.loc[i]
        e = match_entry(row, preds.loc[i], blended.loc[i], bool(blended.loc[i, "model_only"]),
                        None if pd.isna(goals_used[i]) else float(goals_used[i]),
                        None if pd.isna(mkt_o25[i]) else float(mkt_o25[i]),
                        matches, now, priced_at.get(row["match_id"]))
        e["status"] = "upcoming"
        entries.append(e)

    log("3. log and grade")
    history = sl.load(log_path)
    logged = sum(sl.log_pick(history, log_fields(e), now) for e in entries if e["pick"])
    graded = sl.grade(history, matches, now)
    sl.save(history, log_path)
    for e in entries:
        p = history["picks"].get(e["match_id"])
        if p:
            e["logged_at"] = p.get("last_logged")
    # Top-league results only: a second-tier match must not set "results to".
    top_done = matches["league"].isin(sd.TOP) & matches["result"].notna()
    data_through = matches.loc[top_done, "date"].max() if top_done.any() else None
    window = min(now - BEHIND, pd.Timestamp(data_through) - BEHIND) if isinstance(data_through, str) else now - BEHIND
    # Every logged pick that has kicked off in the window: graded beside its
    # result, or "started" until the result is in.
    shown_picks = [logged_entry(p, matches) for p in history["picks"].values()
                   if not p.get("void") and sl.naive_utc(p["kickoff"]) <= now
                   and sl.naive_utc(p["kickoff"]) >= window]
    log(f"   {len(entries)} upcoming, {logged} logged, {graded} graded, {len(shown_picks)} kicked off or over")

    log("4. write the app data")
    bts = backtest_summary(frame, preds, blended, weights, goals_used)
    season = sd.current_season(now)
    season_label = f"{season}-{(season + 1) % 100:02d} season"
    # The pick is the recalibrated market (soccer_model says why); the
    # model's own view is shown beside it.
    basis = "market"
    caveats = sx.caveats(bts, basis)
    # The latest round's results, as facts, so the tab is never empty between
    # rounds: only matches no pick was ever logged for (a logged one is shown
    # with its pick above, whatever its id became).
    logged_pairs = {sl.pairing(p) for p in history["picks"].values()}
    last = matches[top_done & (matches["date"] >= (pd.Timestamp(data_through) - BEHIND).strftime("%Y-%m-%d"))] \
        if isinstance(data_through, str) else matches.iloc[:0]
    last = last[[(l, int(se), h, a) not in logged_pairs
                 for l, se, h, a in zip(last["league"], last["season"], last["home"], last["away"])]]
    results = [{"match_id": r.match_id, "league": r.league, "date": r.date, "kickoff": stamp(r.kickoff),
                "home": st.display(r.home), "away": st.display(r.away), "hg": int(r.hg), "ag": int(r.ag)}
               for r in last.itertuples()]
    sx.write("matches", {
        "season": season, "season_label": season_label, "leagues": sx.leagues(), "league_order": list(sd.TOP),
        "matches": shown_picks + entries, "results": results, "data_through": data_through,
        "prices_at": stamp(fetched_at),
        "caveats": caveats, "basis": basis, "generated": sx.now_stamp(now),
    }, out_dir)
    # The combined frame, so a league whose season has fixtures but no
    # results yet still has its clubs (and their ratings) listed.
    sx.write("tables", sx.tables_payload(frame, season, ratings=ratings or None, generated=sx.now_stamp(now),
                                         caveat=caveats, season_label=season_label), out_dir)
    sx.write("performance", {"backtest": bts, "basis": basis, "track_record": live_record(history),
                             "generated": sx.now_stamp(now)}, out_dir)
    return {"entries": entries, "shown": shown_picks, "backtest": bts, "weights": weights}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args(argv)
    run(sync=not a.no_sync)
    import soccer_standalone
    soccer_standalone.build()


if __name__ == "__main__":
    sys.exit(main())
