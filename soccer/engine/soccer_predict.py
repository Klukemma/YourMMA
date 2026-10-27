"""Predict the soccer matches still to play and write everything the app shows.

    python soccer/engine/soccer_predict.py            # sync this season first
    python soccer/engine/soccer_predict.py --no-sync  # use the snapshot as it is

One run:
  1. refreshes this season's results and the fixtures file - the matches
     still to play, with the bookmakers' Friday/Tuesday prices - from
     football-data.co.uk, and the free fixture feeds (soccer_feed: ESPN's
     scoreboard and fixturedownload.com) into soccer/data/feed.json; a feed
     that fails keeps the snapshot on file;
  2. joins them (soccer_schedule): football-data.co.uk's fixtures, plus the
     feeds' matches it has not posted yet, with no price; recalibrates the
     market's chances season by season and runs the model over every match
     since 2005-06, finished or not, each predicted only from matches on
     earlier days - exactly as the backtest does, so the record the app
     prints is the record of this procedure (soccer_model says what is shown
     and why). The model reads football-data.co.uk's results only. Beside
     each pick: both teams to score, over 1.5 / 3.5, the likeliest scores and
     the pick's band (soccer_scores), with the measured table they ship on;
  3. logs a pick for every top-league match in the next ten days that has
     not frozen (prediction_history.json) and grades every logged pick whose
     result is in - provisionally on the feeds' result until
     football-data.co.uk posts its own;
  4. writes soccer/app/data/{matches,tables,performance}.json and the
     standalone page.

A PICK IS LOGGED ONLY BEFORE IT FREEZES: at kickoff, or at 00:00 UK time on
match day where the kickoff time is not confirmed (soccer_schedule).
Re-running before then updates its entry; after, the entry is frozen, and a
pick logged after the match's real kickoff is never graded (soccer_log).
A pick logged for a match the feeds then postpone, or stop dating, is shown
on that match's card as held: kept in the log, not graded, replaced if the
match is picked on its new date, void after three weeks with no result.
A feed that fails, or a snapshot that cannot be read or joined, never stops
the run: football-data.co.uk alone, said so in the log.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import soccer_backtest as bt
import soccer_data as sd
import soccer_export as sx
import soccer_feed as sfeed
import soccer_form as sf
import soccer_log as sl
import soccer_model as smod
import soccer_schedule as ssch
import soccer_scores as ssc
import soccer_teams as st

AHEAD = timedelta(days=10)      # how far ahead the app shows fixtures (the feeds ask ESPN 10 days ahead)
BEHIND = timedelta(days=4)      # how far back it shows full-time results
VALUE_EDGES = (0.02, 0.05, 0.08)
MIN_GATE_BETS = 100
CAL_BINS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]


def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def combined(matches, fixtures, now):
    """History plus football-data.co.uk's fixtures still to play, in date
    order (soccer_schedule.schedule is this, with the feeds joined in). A
    fixture already in the results file is a played match and comes from
    there."""
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
        w = weeks.setdefault(monday, {"label": f"Week of {monday:%b} {monday.day}", "settled": 0, "correct": 0,
                                      "provisional": 0})
        w["settled"] += 1
        w["correct"] += int(bool(p["correct"]))
        w["provisional"] += int(bool(p["final"].get("provisional")))
    rec["by_week"] = [weeks[k] for k in sorted(weeks)]
    return rec


# ------------------------------------------------------------- one match

def three(df_row):
    return {"p_h": float(df_row["p_h"]), "p_d": float(df_row["p_d"]), "p_a": float(df_row["p_a"])}


# The numbers beside a pick (soccer_scores), on every match entry and in the
# log: None where there are none.
SCORE_KEYS = ("p_btts", "p_o15", "p_o35", "scores", "score_source", "band", "band_basis")


def score_fields(card, pick_prob, model_only):
    """A match's goals and scores numbers and the pick's band, as the entry
    and the log carry them. `card` is soccer_scores.card (None: none); the
    band is read off the pick's own chance and measured on the market's
    picks where the pick is the market's, the model's own where it is not."""
    out = dict.fromkeys(SCORE_KEYS)
    if card is not None:
        out.update(p_btts=card["btts"], p_o15=card["o15"], p_o35=card["o35"], score_source=card["source"],
                   scores={"top": card["scores"], "other": card["other"], "matrix": card["matrix"],
                           "o25": card["o25"], "reason": card["reason"], "miss": card["miss"],
                           "miss_on": card["miss_on"]})
    if pick_prob is not None:
        out.update(band=ssc.band_of(pick_prob), band_basis="model" if model_only else "pick")
    return out


def match_entry(row, model_p, blend_p, model_only, p_o25, market_o25, matches, now, priced_at, card=None):
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
    entry.update(score_fields(card if has else None, entry["pick_prob"], model_only))
    if not has:
        entry["no_pick_reason"] = "The model has not seen enough of these clubs yet."
    return entry


def log_fields(e):
    """What the log keeps of a match entry."""
    return dict(match_id=e["match_id"], league=e["league"], season=e["season"], kickoff=e["kickoff"],
                home=e["home_key"], away=e["away_key"], pick=e["pick"], pick_prob=e["pick_prob"],
                p_h=e["p_h"], p_d=e["p_d"], p_a=e["p_a"], model=e["model"], market=e["market"],
                prices=e["prices"], p_o25=e["p_o25"], model_o25=e["model_o25"], market_o25=e["market_o25"],
                prices_at=e["prices_at"], price_source="football-data.co.uk" if e["market"] else None,
                freeze_at=e["freeze_at"], time_confirmed=e["time_confirmed"], fixture_source=e["fixture_source"],
                **{k: e.get(k) for k in SCORE_KEYS})


# What a frozen pick shows on its card: the line that was logged.
PICK_KEYS = ("p_h", "p_d", "p_a", "pick", "pick_prob", "model", "market", "prices", "p_o25", "model_o25",
             "market_o25") + SCORE_KEYS + ("prices_at",)


def lock(e, p, now):
    """An entry whose pick has frozen before kickoff (a time not confirmed
    closes at midnight UK): its card shows the pick as logged - or, with
    none logged in time, no pick, never one made after it froze."""
    e["locked"] = True
    if p and not p.get("void"):
        # A pick on file froze at its own time, which may be earlier than the
        # match's time now (an unconfirmed time, since confirmed later).
        times = [t for t in (e.get("freeze_at"), p.get("freeze_at")) if t]
        e["freeze_at"] = min(times) if times else None
        e.update({k: p.get(k) for k in PICK_KEYS})
        e["price_status"] = ("posted" if p.get("market") else
                             "not_yet" if p.get("fixture_source") == ssch.FEED_SOURCE else "none")
        return e
    e.update(dict.fromkeys(PICK_KEYS[:-1]), no_pick_reason=(
        "No pick: the kickoff time is not confirmed, so picks closed at midnight UK time on match day, "
        "before any run logged one."))
    return e


def logged_entry(p, matches, feed_status=None, feed_kickoff=None):
    """A logged pick whose match has kicked off, for the Matches tab: the
    pick as it was logged before kickoff, beside the result once it is in
    ("final", flagged provisional while only the feeds have it), until then
    "started" - or postponed, cancelled or abandoned as the feeds say, or
    postponed where the feeds now list the match on a later date
    ("moved_to") - never dropped in between."""
    date = str(p["kickoff"])[:10]
    fin = p.get("final")
    off = (feed_status or {}).get(sl.pairing(p))
    later = (feed_kickoff or {}).get(sl.pairing(p))
    moved = (not fin and later is not None
             and pd.Timestamp(later) > sl.naive_utc(p["kickoff"]) + ssch.MOVED)
    out = {
        "match_id": p["match_id"], "league": p["league"], "season": p["season"], "kickoff": p["kickoff"],
        "status": ("final" if fin else off if off in ssch.OFF + ("abandoned",) else "postponed" if moved
                   else "started"),
        "time_confirmed": p.get("time_confirmed", True), "fixture_source": p.get("fixture_source"),
        "home": st.display(p["home"]), "away": st.display(p["away"]),
        "p_h": p.get("p_h"), "p_d": p.get("p_d"), "p_a": p.get("p_a"), "pick": p.get("pick"),
        "pick_prob": p.get("pick_prob"), "model": p.get("model"), "market": p.get("market"),
        "prices": p.get("prices"), "p_o25": p.get("p_o25"), "model_o25": p.get("model_o25"),
        "market_o25": p.get("market_o25"), "prices_at": p.get("prices_at"), "logged_at": p.get("last_logged"),
        **{k: p.get(k) for k in SCORE_KEYS},
        # Kicked off with no price: the model alone, for good ("Model only").
        "price_status": "posted" if p.get("market") else "none",
        "home_form": sx.form_payload(matches, p["home"], date),
        "away_form": sx.form_payload(matches, p["away"], date),
    }
    pos = sx.positions(matches, p["league"], int(p["season"]), date)
    out["home_position"], out["away_position"] = pos.get(p["home"]), pos.get(p["away"])
    if fin:
        out["score"] = {"home": fin["home"], "away": fin["away"]}
        out["correct"] = p.get("correct")
        out["o25_correct"], out["btts_correct"] = p.get("o25_correct"), p.get("btts_correct")
        out["provisional"] = bool(fin.get("provisional"))
        out["result_source"] = fin.get("source") if fin.get("provisional") else "football-data.co.uk"
    if p.get("corrections"):
        out["corrections"] = p["corrections"]
    if moved:
        out["moved_to"] = stamp(later)
    if not fin and out["status"] != "started":
        # Postponed after it was logged: held, and a price may yet come.
        out["held"] = True
        if not p.get("market") and p.get("fixture_source") == ssch.FEED_SOURCE:
            out["price_status"] = "not_yet"
    return out


def hold(e, p):
    """A postponed or undated match whose pick was logged before the feeds
    said so: the card shows that pick, kept in the log and not graded until
    the match is played (soccer_log voids it when the match is picked again
    on its new date, or STALE_DAYS after kickoff with no result)."""
    e.update({k: p.get(k) for k in PICK_KEYS})
    e.update(held=True, logged_at=p.get("last_logged"), logged_for=p.get("kickoff"), logged_id=p.get("match_id"),
             fixture_source=p.get("fixture_source") or e.get("fixture_source"),
             price_status="posted" if p.get("market") else
             "not_yet" if p.get("fixture_source") == ssch.FEED_SOURCE else "none")
    e.pop("no_pick_reason", None)
    return e


def off_entry(o, matches):
    """A postponed, cancelled or abandoned match with no pick logged: shown
    as such, never picked."""
    kick = o["kickoff"]
    out = {"match_id": o["match_id"], "league": o["league"], "season": o["season"],
           "kickoff": stamp(kick) if kick is not None else f"{o['date']}T12:00:00Z", "time_confirmed": kick is not None,
           "status": o["status"], "home": st.display(o["home"]), "away": st.display(o["away"]),
           "p_h": None, "p_d": None, "p_a": None, "pick": None, "pick_prob": None, "model": None, "market": None,
           "prices": None, "p_o25": None, "model_o25": None, "market_o25": None, "fixture_source": o["source"],
           **dict.fromkeys(SCORE_KEYS),
           "no_pick_reason": f"{o['status'].capitalize()}, as the feeds report it - not picked.",
           "home_form": sx.form_payload(matches, o["home"], o["date"]),
           "away_form": sx.form_payload(matches, o["away"], o["date"])}
    pos = sx.positions(matches, o["league"], int(o["season"]), o["date"])
    out["home_position"], out["away_position"] = pos.get(o["home"]), pos.get(o["away"])
    return out


def tbc_entry(t, matches, now):
    """A match the feeds list with no kickoff time from either source: its
    round's nominal day only, so no id and no pick. Form and positions as
    they stand now."""
    today = pd.Timestamp(now).strftime("%Y-%m-%d")
    out = {"match_id": None, "league": t["league"], "season": t["season"], "kickoff": None,
           "date_hint": t["date_hint"], "round": t["round"], "status": "tbc", "time_confirmed": False,
           "home": st.display(t["home"]), "away": st.display(t["away"]), "fixture_source": t["source"],
           "p_h": None, "p_d": None, "p_a": None, "pick": None, "pick_prob": None, "model": None, "market": None,
           **dict.fromkeys(SCORE_KEYS),
           "no_pick_reason": "Date and time to be confirmed - no pick until the kickoff is set.",
           # through today: the match is on a day after it
           "home_form": sx.form_payload(matches, t["home"], "9999-12-31"),
           "away_form": sx.form_payload(matches, t["away"], "9999-12-31")}
    pos = sx.positions(matches, t["league"], int(t["season"]), "9999-12-31")
    out["home_position"], out["away_position"] = pos.get(t["home"]), pos.get(t["away"])
    out["form_through"] = today
    return out


# ------------------------------------------------------------- the run

def sync_all(now, feed_path, log=print):
    """football-data.co.uk, then the feeds. A feed sync that fails outright
    (it catches each request's failure itself) keeps the snapshot on file:
    the run goes on."""
    sd.sync(log=lambda m: log(m))
    try:
        sfeed.sync(now=now, path=feed_path, log=log)
    except Exception as err:                       # noqa: BLE001 - logged; the snapshot on file is used
        log(f"   feeds: sync failed ({err.__class__.__name__}: {err}); the snapshot on file is used")


def run(sync=True, now=None, out_dir=sx.APP_DATA, log_path=sl.LOG, log=print, feed_path=None):
    now = pd.Timestamp(now) if now is not None else now_utc()
    feed_path = feed_path if feed_path is not None else sfeed.FEED
    if sync:
        log("1. sync")
        sync_all(now, feed_path, log)
    matches = sd.load_matches()
    fixtures = sd.load_fixtures()
    snap, feed = ssch.feed_matches(feed_path, log=log)
    fetched_at = fixtures["fetched_at"].max() if len(fixtures) else None
    priced_at = dict(zip(fixtures["match_id"], fixtures["fetched_at"])) if len(fixtures) else {}
    try:
        sched = ssch.schedule(matches, fixtures, feed, now, ahead=pd.Timedelta(AHEAD),
                              behind=pd.Timedelta(BEHIND), log=log)
    except Exception as err:                       # noqa: BLE001 - a feed problem never fails the run
        log(f"   feeds: could not be joined ({err.__class__.__name__}: {err}); football-data.co.uk alone")
        feed = []
        sched = ssch.schedule(matches, fixtures, feed, now, ahead=pd.Timedelta(AHEAD),
                              behind=pd.Timedelta(BEHIND), log=log)
    # The model's input: football-data.co.uk's results and the matches still
    # to play - never a provisional result.
    frame = sched.frame
    from_feed = sum(1 for v in sched.info.values() if v["source"] == ssch.FEED_SOURCE)
    log(f"   feeds: {len(feed)} matches on file, {from_feed} to play that football-data.co.uk has not posted, "
        f"{len(sched.provisional)} provisional results, {len(sched.off)} postponed/cancelled/abandoned")
    log(f"2. model ({smod.MODEL}) over {len(frame):,} matches")
    preds = smod.predict(frame)
    blended, weights = smod.pick(frame, preds)
    goals_used = smod.goals(frame, preds)
    ratings = {team: r["egd"] for team, r in (preds.attrs.get("ratings") or {}).items()}
    mkt_o25 = bt.no_vig_total(frame)
    # The score matrix behind every top-league match the model predicts -
    # by the rule soccer_scores ships - and the table it is measured by.
    walked = ssc.walk(frame, preds, blended, mkt_o25)
    row_of = {i: k for k, i in enumerate(walked[0])}

    def card(i):
        k = row_of.get(i)
        if k is None:
            return None
        r = walked[2].iloc[k]
        return ssc.card(walked[1][k], r["source"], r["reason"], r["miss"], r["miss_on"])
    scores_measured = ssc.measure(frame, preds, blended, walked)
    # What the app's form lines and tables read: the results with the
    # provisional ones, flagged.
    shown = ssch.shown(frame, sched.provisional)

    upcoming = frame["league"].isin(sd.TOP) & frame["result"].isna() & (frame["kickoff"] > now) \
        & (frame["kickoff"] <= now + AHEAD)
    entries = []
    for i in frame.index[upcoming]:
        row = frame.loc[i]
        e = match_entry(row, preds.loc[i], blended.loc[i], bool(blended.loc[i, "model_only"]),
                        None if pd.isna(goals_used[i]) else float(goals_used[i]),
                        None if pd.isna(mkt_o25[i]) else float(mkt_o25[i]),
                        shown, now, priced_at.get(row["match_id"]), card=card(i))
        meta = sched.info.get(row["match_id"]) or {"source": ssch.FD_SOURCE, "time_confirmed": True,
                                                   "freeze_at": row["kickoff"]}
        e["status"] = "upcoming"
        e["time_confirmed"] = bool(meta["time_confirmed"])
        e["freeze_at"] = stamp(meta["freeze_at"])
        e["fixture_source"] = meta["source"]
        e["price_status"] = ("posted" if e["market"] else
                             "not_yet" if meta["source"] == ssch.FEED_SOURCE else "none")
        kf, kd = meta.get("kickoff_feed"), meta.get("kickoff_fd")
        if kf is not None and kd is not None and kf != kd:
            e["kickoff_sources"] = {"football-data.co.uk": stamp(kd), "feeds": stamp(kf)}
        entries.append(e)

    log("3. log and grade")
    history = sl.load(log_path)
    logged = sum(sl.log_pick(history, log_fields(e), now) for e in entries if e["pick"])
    stats = {}
    graded = sl.grade(history, matches, now, provisional=sched.provisional, disputed=sched.disputed, stats=stats)
    sl.save(history, log_path)
    for e in entries:
        p = history["picks"].get(e["match_id"])
        if sl.naive_utc(e["freeze_at"]) <= now or (p and not p.get("void") and sl.frozen(p, now)):
            lock(e, p, now)
        if p and not p.get("void"):
            e["logged_at"] = p.get("last_logged")
    # Top-league results only: a second-tier match must not set "results to".
    top_done = matches["league"].isin(sd.TOP) & matches["result"].notna()
    data_through = matches.loc[top_done, "date"].max() if top_done.any() else None
    prov = sched.provisional
    results_through = max([d for d in (data_through, prov["date"].max() if len(prov) else None)
                           if isinstance(d, str)], default=None)
    window = min(now - BEHIND, pd.Timestamp(data_through) - BEHIND) if isinstance(data_through, str) else now - BEHIND
    # Every logged pick that has kicked off in the window - by its logged
    # time, or because it is graded or the feeds say it is under way or over
    # (a match brought forward): graded beside its result, or "started"
    # (postponed, ...) until the result is in. A match on the list to play
    # is shown there, never twice.
    started = {k for k, s in sched.status.items() if s in ssch.STARTED}
    to_play = {(e["league"], e["season"], e["home_key"], e["away_key"]) for e in entries}
    kicked = [p for p in history["picks"].values()
              if not p.get("void") and sl.pairing(p) not in to_play
              and window <= sl.naive_utc(p["kickoff"])
              and (sl.naive_utc(p["kickoff"]) <= now
                   or (sl.naive_utc(p["kickoff"]) <= now + AHEAD and (p.get("final") or sl.pairing(p) in started)))]
    shown_picks = [logged_entry(p, shown, sched.status, sched.feed_kickoff) for p in kicked]
    # A postponed or undated match whose pick is on screen already is not
    # shown twice; one with a pick logged before the feeds said so shows it.
    shown_pairs = {sl.pairing(p) for p in kicked}
    pending = {}
    for p in history["picks"].values():
        if not p.get("void") and not p.get("final"):
            pending.setdefault(sl.pairing(p), []).append(p)
    held = {k: max(v, key=lambda p: p["kickoff"]) for k, v in pending.items()}
    off, tbc = [], []
    for o in sched.off:
        key = (o["league"], o["season"], o["home"], o["away"])
        if key not in shown_pairs:
            e = off_entry(o, shown)
            off.append(hold(e, held[key]) if key in held else e)
    for t in sched.tbc:
        key = (t["league"], t["season"], t["home"], t["away"])
        if key not in shown_pairs:
            e = tbc_entry(t, shown, now)
            tbc.append(hold(e, held[key]) if key in held else e)
    log(f"   {len(entries)} upcoming ({sum(1 for e in entries if e['price_status'] == 'posted')} priced, "
        f"{sum(1 for e in entries if e['pick'] and not e['market'])} the model alone), {logged} logged, "
        f"{graded} graded ({stats.get('provisional', 0)} provisionally), {stats.get('confirmed', 0)} provisional "
        f"confirmed, {stats.get('corrected', 0)} corrected, {stats.get('withdrawn', 0)} withdrawn, "
        f"{stats.get('late', 0)} void as logged after kickoff, "
        f"{len(shown_picks)} kicked off or over, {len(off)} postponed/cancelled/abandoned, {len(tbc)} date to come")

    log("4. write the app data")
    bts = backtest_summary(frame, preds, blended, weights, goals_used)
    season = sd.current_season(now)
    season_label = f"{season}-{(season + 1) % 100:02d} season"
    # The pick is the recalibrated market (soccer_model says why); the
    # model's own view is shown beside it.
    basis = "market"
    caveats = sx.caveats(bts, basis, model_through=data_through, scores=scores_measured)
    # The latest round's results, as facts, so the tab is never empty between
    # rounds: only matches with no pick in the record (a logged one is shown
    # with its pick above, whatever its id became; one whose only pick is
    # void - logged after kickoff - is a fact like any other).
    # football-data.co.uk's, and the feeds' provisional ones it has not
    # posted yet, flagged.
    logged_pairs = {sl.pairing(p) for p in history["picks"].values() if not p.get("void")}
    since = (pd.Timestamp(results_through) - BEHIND).strftime("%Y-%m-%d") if results_through else None
    last = matches[top_done & (matches["date"] >= since)] if since else matches.iloc[:0]
    results = [{"match_id": r.match_id, "league": r.league, "date": r.date, "kickoff": stamp(r.kickoff),
                "home": st.display(r.home), "away": st.display(r.away), "hg": int(r.hg), "ag": int(r.ag),
                "provisional": False}
               for r in last.itertuples() if (r.league, int(r.season), r.home, r.away) not in logged_pairs]
    recent = prov[prov["date"] >= since] if since and len(prov) else prov.iloc[:0]
    results += [{"match_id": r.match_id, "league": r.league, "date": r.date, "kickoff": stamp(r.kickoff),
                 "home": st.display(r.home), "away": st.display(r.away), "hg": int(r.hg), "ag": int(r.ag),
                 "provisional": True, "source": r.source}
                for r in recent.itertuples() if (r.league, int(r.season), r.home, r.away) not in logged_pairs]
    feed_run = (snap or {}).get("last_run") or {}
    sx.write("matches", {
        "season": season, "season_label": season_label, "leagues": sx.leagues(), "league_order": list(sd.TOP),
        "matches": shown_picks + entries + off + tbc, "results": results, "data_through": data_through,
        # The date of football-data.co.uk's latest top-league result: the
        # last the model's ratings include. Results after it are provisional.
        "model_through": data_through, "results_through": results_through,
        "provisional_results": int(len(prov)),
        "feeds_at": feed_run.get("at"), "prices_at": stamp(fetched_at),
        "caveats": caveats, "basis": basis, "generated": sx.now_stamp(now),
        # How the goals and scores numbers and the pick's bands were measured,
        # by this run's own walk-forward (the Method tab's table, the bands'
        # hit rates on the cards).
        "scores_measured": scores_measured,
    }, out_dir)
    # The combined frame, so a league whose season has fixtures but no
    # results yet still has its clubs (and their ratings) listed; with the
    # provisional results, flagged.
    sx.write("tables", dict(sx.tables_payload(shown, season, ratings=ratings or None, generated=sx.now_stamp(now),
                                              caveat=caveats, season_label=season_label),
                            model_through=data_through), out_dir)
    sx.write("performance", {"backtest": bts, "basis": basis, "track_record": live_record(history),
                             "model_through": data_through, "generated": sx.now_stamp(now)}, out_dir)
    return {"entries": entries, "shown": shown_picks, "off": off, "tbc": tbc, "results": results,
            "backtest": bts, "weights": weights, "frame": frame, "schedule": sched,
            "scores_measured": scores_measured}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args(argv)
    run(sync=not a.no_sync)
    import soccer_standalone
    soccer_standalone.build()


if __name__ == "__main__":
    sys.exit(main())
