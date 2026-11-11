"""Predict the soccer matches still to play and write everything the app shows.

    python soccer/engine/soccer_predict.py            # sync this season first
    python soccer/engine/soccer_predict.py --no-sync  # use the snapshot as it is

One run:
  1. refreshes this season's results and the fixtures file - the matches
     still to play, with the bookmakers' Friday/Tuesday prices - from
     football-data.co.uk (all 38 leagues), and the free fixture feeds
     (soccer_feed: ESPN's scoreboard and fixturedownload.com) for every
     league of the app's a feed covers (sfeed.leagues_for(in_app()): 30
     leagues, 462 requests, never RUS) into soccer/data/feed.json; a feed
     that fails keeps the snapshot on file;
  2. loads every league (38: the model and the pick's weights learn from
     every league that was measured, RUS's matches included) and takes the
     app's leagues from the data now (app_leagues: soccer_leagues.in_app
     with the feeds' matches to play counted as posted - rule (a), a current
     season, which a winter break does not end - 37 today, RUS out; each
     league out says why); joins football-data.co.uk's fixtures of those
     leagues with the feeds' matches it has not posted yet, with no price
     (soccer_schedule); runs
     the model over every match since 2005-06 of every league, finished or
     not, each predicted only from matches on earlier days, and picks each
     league as experiments/leagues.py decided (soccer_model.shipped_pick:
     the market recalibrated, or raw where recalibrating lost; the model
     alone where no price is posted, only in a league whose model beat its
     base rates) - exactly as the backtest does, so the record the app
     prints is the record of this procedure. The model reads
     football-data.co.uk's results only. Beside each pick: both teams to
     score, over 1.5 / 2.5 / 3.5, the likeliest scores - each only where
     its league shipped it from that source (soccer_leagues.withheld; a
     number withheld is null, and listed as withheld) - and the pick's band
     (soccer_scores), read against its league's group's measured rates.
     The pick, the score matrices and the Record are computed over the
     decided leagues and the day's (measured_leagues): the top five's
     pooled Record blocks and the global caveats do not empty in summer;
  3. logs a pick for every match of the app's leagues in the next ten days
     that has not frozen (prediction_history.json, with its basis and the
     price it was read from) and grades every logged pick whose result is
     in - provisionally on the feeds' result until football-data.co.uk
     posts its own;
  4. writes soccer/app/data/{matches,tables,performance}.json - every league
     with its own season, dates, decisions, caveats and record, and
     football-data.co.uk's download problems of this run per league
     (feeds.data_problems: that league's data may be stale) - and the
     standalone page.

WHERE A PAIRING REPEATS (soccer_feed.REPEATS) a pairing is not a match: the
feeds' status, kickoff and score of a match are looked up by
soccer_schedule's key (the pairing and its day there, find_key), a graded
pick is hidden only by an upcoming entry of the same meeting, and a logged
pick never stands for another meeting (soccer_log). Where a pairing is played
once a season, the key is the pairing, as before.

EACH LEAGUE ON ITS OWN DATES. A league's season is the latest on file no
later than its own current one (calendar-year in ARG BRA CHN FIN IRL NOR SWE
USA: in January to June the August-May label would be last season's); its
"model_through" is its own latest football-data.co.uk result (CHN's file
lags BRA's by two weeks); its latest results are its own last round. The
window of kicked-off picks runs from the latest result in any league - never
the earliest, or a league between seasons would drag it back months - and,
for each league, from its own last round too, so a league whose results come
later keeps its picks on screen until they are graded; a result is hidden
only behind its pick on screen.

A PICK IS LOGGED ONLY BEFORE IT FREEZES: at kickoff, or at 00:00 UK time on
match day where the kickoff time is not confirmed - or earlier, at any
source's earlier time (soccer_schedule). Re-running before then updates its
entry (never a priced pick with an unpriced one); after, the entry is
frozen - it follows its match to a new id unchanged - and a pick logged
after the match's real kickoff is never graded (soccer_log).
A pick logged for a match the feeds then postpone, or stop dating, is shown
on that match's card as held: kept in the log, not graded, replaced if the
match is picked on its new date, void after three weeks with no result.
A feed that fails, or a snapshot that cannot be read or joined, never stops
the run: football-data.co.uk alone, said so in the log.
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import soccer_backtest as bt
import soccer_data as sd
import soccer_export as sx
import soccer_feed as sfeed
import soccer_leagues as slg
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
TOP_FIVE = "top five"           # what the pooled backtest blocks are measured on
NO_PICK = {
    # shipped_pick's basis "unpriced": rule (b) - the model alone did not beat
    # the league's base rates in both periods, so a match with no price waits for one
    "unpriced": "No pick until the price is posted: in this league the model alone did not beat the league's own "
                "base rates in both of its periods.",
    "model": "The model has not seen enough of these clubs yet.",
}


def now_utc():
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def stamp(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


class ByKey:
    """Items filed under soccer_schedule's key (the pairing; its day too where
    pairings repeat) and read with find_key - which only ever looks among
    the keys of the item's own pairing, so each lookup is handed those
    alone: the log grows by thousands of picks a season."""

    def __init__(self, items=(), key=ssch.key_of):
        self._items, self._pairs = defaultdict(list), defaultdict(set)
        for it in items:
            self.add(key(it), it)

    def add(self, k, item):
        self._items[k].append(item)
        self._pairs[k[:4]].add(k)

    def get(self, k):
        found = ssch.find_key(self._pairs.get(k[:4], set()), k)
        return self._items[found] if found is not None else []

    def __contains__(self, k):
        return bool(self.get(k))


def entry_key(e):
    """An entry's key (soccer_schedule.key): its clubs are home_key/away_key."""
    return ssch.key(e["league"], e["season"], e["home_key"], e["away_key"], ssch.uk_day(e["kickoff"]))


def league_seasons(frame, leagues, now):
    """{league: its season on show}: the latest on file no later than its own
    current season (soccer_data.season_for - the calendar year for ARG BRA
    CHN FIN IRL NOR SWE USA), so a league between seasons shows the one it
    finished, under its own label."""
    on_file = frame.groupby("league")["season"].unique().to_dict()
    out = {}
    for c in leagues:
        now_s = sd.season_for(c, now)
        mine = [int(s) for s in on_file.get(c, ()) if pd.notna(s) and int(s) <= now_s]
        out[c] = max(mine) if mine else now_s
    return out


# ------------------------------------------------------------- the leagues

def feed_to_play(feed, now):
    """The feeds' matches still to play (status "scheduled", on a UK day no
    earlier than today's), as fixture rows {league, date}: what rule (a)
    counts beside football-data.co.uk's posted fixtures (app_leagues)."""
    today = pd.Timestamp(now).strftime("%Y-%m-%d")
    rows = [{"league": m["league"], "date": m.get("uk_date") or m.get("date_hint")} for m in feed or ()
            if m.get("status") == "scheduled" and (m.get("uk_date") or m.get("date_hint") or "") >= today]
    return pd.DataFrame(rows, columns=["league", "date"])


def app_leagues(matches, fixtures, feed, now):
    """(leagues, out): the app's leagues by rule (a) on the data of the day
    (soccer_leagues.in_app) - with the feeds' matches still to play counted
    as fixtures posted, beside football-data.co.uk's - and {code:
    soccer_leagues.season_check facts} for each measured league out of it,
    which not_in_app shows as the reason.

    WHY THE FEEDS COUNT: football-data.co.uk posts a round's fixtures only
    days before it, so a winter or mid-year break longer than RECENT_DAYS
    (45) would drop a league in the middle of its season - DNK (Dec 8 to
    Feb 8), AUT (54 days), POL (47), ARG (60 in mid-2026) - and with it its
    card, table and Record row, and its first round back, which the feeds
    list ten days ahead, would get no pick. A league the feeds list a match
    to play for has a current season."""
    listed = feed_to_play(feed, now)
    posted = fixtures[["league", "date"]] if len(fixtures) else pd.DataFrame(columns=["league", "date"])
    if len(listed):
        posted = pd.concat([posted, listed], ignore_index=True)
    leagues = slg.in_app(matches, posted, now)
    out = {c: slg.season_check(matches, posted, c, now) for c in slg.SHIPPED if c not in leagues}
    return leagues, out


def measured_leagues(leagues):
    """The leagues the pick, the score matrices and the Record are computed
    over: the decided list (soccer_leagues.in_app()) and the app's leagues of
    the day, in soccer_data's order.

    WHY NOT THE DAY'S ALONE: every summer the top five leave the app for
    weeks (no result in 45 days, no fixture posted: D1 and F1 from about Jul
    1 until their fixtures appear), and the Record tab's pooled top-five
    blocks, the calibration and the global scores caveat are measured on the
    top five - over the day's leagues they went empty (18,259 matches to 0)
    and a caveat was deleted, and while only some of the five were back they
    were measured on that subset under the "top five" label. What was
    measured does not change with the calendar; what is shown does (the
    upcoming entries, the log and the schedule keep the day's leagues)."""
    keep = set(slg.in_app()) | set(leagues)
    return [c for c in sd.ALL_LEAGUES if c in keep]


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


def by_league(frame, preds, blended, leagues):
    """The Record tab's table for every league in the app, each on its own
    periods against the price it was measured on (soccer_backtest.record:
    soccer_leagues.SHIPPED[code]["measured"] to the digit), with the
    confirmation period's numbers also at the top level of each row, where
    the app has always read them. Never the top five's `has` (top-five
    only) nor the pre-match no-vig price (the extra leagues' history has
    only the close: argmax of NaN would make "the favourite" the home win)."""
    rec = bt.record(frame, blended, preds, leagues)
    rows = []
    for r in rec["by_league"]:
        c = r.get("confirm") or {}
        rows.append(dict(r, label=sx.league_name(r["code"]), short=sx.LEAGUE_SHORT.get(r["code"], r["code"]),
                         matches=c.get("matches", 0), accuracy=c.get("accuracy"),
                         market_accuracy=c.get("market_accuracy"), log_loss=c.get("log_loss"),
                         market_log_loss=c.get("market_log_loss")))
    return rows, rec.get("by_group"), rec.get("pooled")


def latest_weights(weights, leagues):
    """{league: {season, w} or None} - each league's recalibration for its
    latest season: soccer_model.latest_weights, the one function that makes
    this shape, so the Record tab prints what soccer_leagues.SHIPPED stores
    ("weights": None for a league picked on the raw price, never {season, w:
    None}) - a reader comparing the two sees the same thing."""
    return smod.latest_weights(weights, leagues)


def backtest_summary(frame, preds, blended, weights, goals_used, leagues=None, in_app=None):
    """What the Record tab shows: the model alone and the pick against the
    market, DEV and CONFIRM never pooled, plus goals, betting the
    disagreements and calibration - those over the top five, as measured,
    and labelled so ("scope") - and by league (by_league), group and pooled
    for every league of `leagues` (run: measured_leagues - the decided ones
    and the day's), each on its own periods. in_app: the app's leagues of the
    day (default `leagues`): each by_league row says whether it is one, so a
    league between seasons keeps its measured row, marked out of the app."""
    leagues = list(sd.TOP) if leagues is None else list(leagues)
    periods = {"dev": list(bt.DEV), "confirm": list(bt.CONFIRM)}
    has = bt.scored(frame) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1) & ~blended["model_only"]
    result = {"model": {}, "blend": {}}
    for lo, hi in (bt.DEV, bt.CONFIRM):
        sel = has & frame["season"].between(lo, hi)
        key = f"{lo}-{hi}"
        result["model"][key] = _row(frame, preds, sel)
        result["blend"][key] = _row(frame, blended, sel)
    result["scope"] = TOP_FIVE
    conf = has & frame["season"].between(*bt.CONFIRM)
    leagues_rows, by_group, pooled = by_league(frame, preds, blended, leagues)
    app = set(leagues if in_app is None else in_app)
    for r in leagues_rows:
        r["in_app"] = r["code"] in app
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
    over["scope"] = TOP_FIVE
    return {"periods": periods, "result": result, "by_league": leagues_rows, "by_group": by_group,
            "pooled": pooled, "over_2_5": over,
            "value": {"edges": edges, "enabled_at": enabled, "scope": TOP_FIVE}, "calibration": cal,
            "calibration_scope": TOP_FIVE, "model": smod.MODEL, "weights": latest_weights(weights, leagues)}


# What the record calls a pick logged before every league was switched on:
# no basis was written, and its pick was the top five's recalibration of then.
BEFORE_LEAGUES = "before every league"


def live_record(log):
    """The live record (soccer_log.record), pooled as the app has always read
    it, and per league, per group and per basis of the pick - a league's
    own record is never read off the pooled one, and the picks of the
    procedure before (no basis logged) are counted apart."""
    rec = sl.record(log)
    rec["by_league"] = sl.record(log, by=lambda p: p.get("league") or "")
    rec["by_group"] = sl.record(log, by=lambda p: p.get("group") or (
        slg.group_of(p["league"]) if p.get("league") in sd.ALL_LEAGUES else ""))
    rec["by_basis"] = sl.record(log, by=lambda p: p.get("basis") or BEFORE_LEAGUES)
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


def score_fields(card, pick_prob, model_only, withheld=()):
    """A match's goals and scores numbers and the pick's band, as the entry
    and the log carry them. `card` is soccer_scores.card (None: none), which
    has already nulled the numbers its league did not ship from its source;
    scores.withheld lists them (with `withheld`: the over 2.5 numbers beside
    the card the entry nulled), present only when there are any - a number
    withheld is shown as missing, never as another source's. The band is
    read off the pick's own chance and measured on the market's picks where
    the pick is the market's, the model's own where it is not."""
    out = dict.fromkeys(SCORE_KEYS)
    if card is not None:
        out.update(p_btts=card["btts"], p_o15=card["o15"], p_o35=card["o35"], score_source=card["source"],
                   scores={"top": card["scores"], "other": card["other"], "matrix": card["matrix"],
                           "o25": card["o25"], "reason": card["reason"], "miss": card["miss"],
                           "miss_on": card["miss_on"], "flagged": card.get("flagged", False)})
        held = [k for k in ssc.NUMBERS if k in set(card.get("withheld") or ()) | set(withheld)]
        if held:
            out["scores"]["withheld"] = held
    if pick_prob is not None:
        out.update(band=ssc.band_of(pick_prob), band_basis="model" if model_only else "pick")
    return out


def _num(x):
    return None if x is None or x != x else float(x)


def match_entry(row, model_p, blend_p, model_only, p_o25, market_o25, matches, now, priced_at, card=None,
                basis=None, price_kind=None):
    """One match still to play, as the app shows it. basis / price_kind:
    shipped_pick's ("recalibrated" / "raw" / "model" / "unpriced"; "pre" /
    "close"). The over 2.5 numbers are nulled where the league did not ship
    over 2.5 from their source (soccer_leagues.withheld - the shown one's
    source: the market where the over/under is posted, else the model's), as
    the card nulls its own. `matches`: soccer_export.View (or a frame)."""
    league = row["league"]
    probs = {"H": blend_p["p_h"], "D": blend_p["p_d"], "A": blend_p["p_a"]}
    has = all(v == v and v is not None for v in probs.values())
    pick = max(probs, key=probs.get) if has else None
    mkt = bt.no_vig(pd.DataFrame([row]), "pre").iloc[0]
    has_mkt = mkt.notna().all()
    model_o25 = _num(model_p["p_o25"])
    held = set()
    if p_o25 is not None and "o25" in slg.withheld(league, "market" if market_o25 is not None else "model"):
        p_o25 = None
        held.add("o25")
    if model_o25 is not None and "o25" in slg.withheld(league, "model"):
        model_o25 = None
        held.add("o25")
    if market_o25 is not None and "o25" in slg.withheld(league, "market"):
        market_o25 = None
        held.add("o25")
    chip = matches.chip if isinstance(matches, sx.View) else (lambda lg, k: st.display(k))
    entry = {
        "match_id": row["match_id"], "league": league, "season": int(row["season"]),
        "kickoff": stamp(row["kickoff"]), "home_key": row["home"], "away_key": row["away"],
        "home": chip(league, row["home"]), "away": chip(league, row["away"]),
        "p_h": probs["H"] if has else None, "p_d": probs["D"] if has else None, "p_a": probs["A"] if has else None,
        "pick": pick, "pick_prob": probs[pick] if pick else None,
        "model": three(model_p) if has else None,
        "market": {"p_h": float(mkt.p_h), "p_d": float(mkt.p_d), "p_a": float(mkt.p_a)} if has_mkt and not model_only else None,
        "prices": {"avg": {"h": row.get("avg_h"), "d": row.get("avg_d"), "a": row.get("avg_a")},
                   "best": {"h": row.get("max_h"), "d": row.get("max_d"), "a": row.get("max_a")}},
        "p_o25": p_o25, "model_o25": model_o25, "market_o25": market_o25,
        # When these prices were first fetched (soccer_data.first_seen), and
        # only where there are prices.
        "prices_at": stamp(priced_at) if has_mkt else None,
        "basis": basis if isinstance(basis, str) else None,
        "price_kind": price_kind if isinstance(price_kind, str) else None,
        "home_form": sx.form_payload(matches, row["home"], row["date"]),
        "away_form": sx.form_payload(matches, row["away"], row["date"]),
    }
    pos = sx.positions(matches, league, int(row["season"]), row["date"])
    entry["home_position"], entry["away_position"] = pos.get(row["home"]), pos.get(row["away"])
    entry.update(score_fields(card if has else None, entry["pick_prob"], model_only, held))
    if not has:
        entry["no_pick_reason"] = NO_PICK.get(entry["basis"], NO_PICK["model"])
    return entry


def log_fields(e):
    """What the log keeps of a match entry - with the league's tier and
    group, and the pick's basis and the price it was read from, so the live
    record can be told per league, group and procedure."""
    meta = sd.ALL_LEAGUES.get(e["league"])
    return dict(match_id=e["match_id"], league=e["league"], season=e["season"], kickoff=e["kickoff"],
                home=e["home_key"], away=e["away_key"], pick=e["pick"], pick_prob=e["pick_prob"],
                p_h=e["p_h"], p_d=e["p_d"], p_a=e["p_a"], model=e["model"], market=e["market"],
                prices=e["prices"], p_o25=e["p_o25"], model_o25=e["model_o25"], market_o25=e["market_o25"],
                prices_at=e["prices_at"], price_source="football-data.co.uk" if e["market"] else None,
                tier=meta.tier if meta else None, group=slg.group_of(e["league"]) if meta else None,
                basis=e.get("basis"), price_kind=e.get("price_kind"),
                freeze_at=e["freeze_at"], time_confirmed=e["time_confirmed"], fixture_source=e["fixture_source"],
                kickoff_sources=e.get("kickoff_sources"), **{k: e.get(k) for k in SCORE_KEYS})


# What a frozen pick shows on its card: the line that was logged (a pick
# logged before every league has no basis: None).
PICK_KEYS = ("p_h", "p_d", "p_a", "pick", "pick_prob", "model", "market", "prices", "p_o25", "model_o25",
             "market_o25") + SCORE_KEYS + ("basis", "price_kind", "prices_at")


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
    midnight = e.get("freeze_at") and e.get("kickoff") and not e.get("time_confirmed") and \
        sl.naive_utc(e["freeze_at"]) == ssch.midnight_uk(e["kickoff"])
    e.update(dict.fromkeys(PICK_KEYS[:-1]), no_pick_reason=(
        "No pick: the kickoff time is not confirmed, so picks closed at midnight UK time on match day, "
        "before any run logged one." if midnight else
        "No pick: a source lists an earlier kickoff, so picks closed then, before any run logged one."))
    return e


def logged_entry(p, matches, feed_status=None, feed_kickoff=None, disputes=None):
    """A logged pick whose match has kicked off, for the Matches tab: the
    pick as it was logged before kickoff, beside the result once it is in
    ("final", flagged provisional while only the feeds have it), until then
    "started" - with the feeds' two scores where they contradict each other
    ("disputed": no result until football-data.co.uk's) - or postponed,
    cancelled or abandoned as the feeds say, or postponed where ESPN now
    lists the match on a later date ("moved_to") - never dropped in
    between. The feeds' tables are read by the pick's key (soccer_schedule:
    its pairing, and its day where pairings repeat), never another meeting's.
    `matches`: soccer_export.View (or a frame)."""
    date = str(p["kickoff"])[:10]
    fin = p.get("final")
    k = ssch.key_of(p)
    off = ssch.lookup(feed_status or {}, k)
    later = ssch.lookup(feed_kickoff or {}, k)
    chip = matches.chip if isinstance(matches, sx.View) else (lambda lg, key: st.display(key))
    moved = (not fin and later is not None
             and pd.Timestamp(later) > sl.naive_utc(p["kickoff"]) + ssch.MOVED)
    out = {
        "match_id": p["match_id"], "league": p["league"], "season": p["season"], "kickoff": p["kickoff"],
        "status": ("final" if fin else off if off in ssch.OFF + ("abandoned",) else "postponed" if moved
                   else "started"),
        "time_confirmed": p.get("time_confirmed", True), "fixture_source": p.get("fixture_source"),
        "kickoff_sources": p.get("kickoff_sources"),
        "home": chip(p["league"], p["home"]), "away": chip(p["league"], p["away"]),
        "p_h": p.get("p_h"), "p_d": p.get("p_d"), "p_a": p.get("p_a"), "pick": p.get("pick"),
        "pick_prob": p.get("pick_prob"), "model": p.get("model"), "market": p.get("market"),
        "prices": p.get("prices"), "p_o25": p.get("p_o25"), "model_o25": p.get("model_o25"),
        "market_o25": p.get("market_o25"), "prices_at": p.get("prices_at"), "logged_at": p.get("last_logged"),
        "basis": p.get("basis"), "price_kind": p.get("price_kind"),
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
    if not fin and off == "disputed" and ssch.lookup(disputes or {}, k):
        out["disputed"] = ssch.lookup(disputes, k)
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
    chip = matches.chip if isinstance(matches, sx.View) else (lambda lg, key: st.display(key))
    out = {"match_id": o["match_id"], "league": o["league"], "season": o["season"],
           "kickoff": stamp(kick) if kick is not None else f"{o['date']}T12:00:00Z", "time_confirmed": kick is not None,
           "status": o["status"], "home": chip(o["league"], o["home"]), "away": chip(o["league"], o["away"]),
           "p_h": None, "p_d": None, "p_a": None, "pick": None, "pick_prob": None, "model": None, "market": None,
           "prices": None, "p_o25": None, "model_o25": None, "market_o25": None, "fixture_source": o["source"],
           "basis": None, "price_kind": None,
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
    chip = matches.chip if isinstance(matches, sx.View) else (lambda lg, key: st.display(key))
    out = {"match_id": None, "league": t["league"], "season": t["season"], "kickoff": None,
           "date_hint": t["date_hint"], "round": t["round"], "status": "tbc", "time_confirmed": False,
           "home": chip(t["league"], t["home"]), "away": chip(t["league"], t["away"]), "fixture_source": t["source"],
           "p_h": None, "p_d": None, "p_a": None, "pick": None, "pick_prob": None, "model": None, "market": None,
           "basis": None, "price_kind": None,
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

# football-data.co.uk's download problems, as soccer_data.sync logs them. Its
# sync keeps its problems list to itself and returns the frame; its log lines
# are the record, so they are read here (this module does not change
# soccer_data) - and a test runs the real sync against a made-up site failing
# in each way, so a changed message fails a test, never the app silently.
_PROBLEM_LINES = (
    (re.compile(r"^ {3}(?P<what>.+?): download failed \((?P<err>[^)]*)\); kept what is on file$"),
     lambda m: f"download failed ({m['err']})"),
    (re.compile(r"^ {3}(?P<what>.+?): NOT A football-data\.co\.uk CSV; kept what is on file$"),
     lambda m: "not a football-data.co.uk CSV"),
    (re.compile(r"^ {3}(?P<what>.+?): CUT OFF PART-WAY THROUGH ITS LAST ROW; kept what is on file$"),
     lambda m: "cut off part-way through its last row"),
    (re.compile(r"^ {3}(?P<what>.+?): NO ROWS PARSED \(format change\?\); kept what is on file$"),
     lambda m: "no rows parsed (format change?)"),
    (re.compile(r"^ {3}(?P<what>.+?): \d+ matches, FEWER than the \d+ on file; kept what is on file$"),
     lambda m: "fewer matches than on file"),
    (re.compile(r"^ {3}(?P<code>[A-Z0-9]+): no fixtures while the file holds rows of .+ \(a renamed league\?\); "
                r"kept what is on file$"), lambda m: "no fixtures for it in the file (a renamed league?)"),
)
_BUDGET_LINE = re.compile(r"^ {3}football-data\.co\.uk: (?P<why>.+); every file not yet asked for keeps what is "
                          r"on file$")
_KEPT_LINE = re.compile(r"^ {3}KEPT ON FILE after a failed or unreadable download: (?P<items>.+)$")
_NOT_PUBLISHED = re.compile(r"^ {3}(?P<what>.+?): not published$")


def _leagues_of(what, asked):
    """The leagues a football-data.co.uk file holds, by the name sync gives
    it: "fixtures" (every main division's), "extra-league fixtures" (every
    extra league's), "SC0 fixtures", "E0 2026", "ARG"."""
    if what == "fixtures":
        return [c for c in asked if sd.ALL_LEAGUES[c].kind == "main"]
    if what == "extra-league fixtures":
        return [c for c in asked if sd.ALL_LEAGUES[c].kind != "main"]
    code = what.split(" ")[0]
    return [code] if code in sd.ALL_LEAGUES else []


def data_problems(lines, at, asked="all", root=None):
    """[{league, what, why, at}] from one football-data.co.uk sync's log
    lines: every file it kept on file after a failed, unreadable, cut-off or
    short download, or because its failure budget was spent before asking
    (why: the budget's reason), and every file the site no longer publishes
    that is on file (a 404 or an empty body: "not published") - one entry
    per league the file holds, so the app can say that league's data may be
    stale. A season the site has not started publishing (no file on file
    yet, as in July) is not a problem. `asked`: the leagues synced; root:
    the results snapshot (soccer_data.MATCHES)."""
    root = sd.MATCHES if root is None else root
    asked = sd.league_codes(asked)
    why, kept, budget = {}, [], None
    gone = []
    for line in lines:
        line = str(line)
        for rx, reason in _PROBLEM_LINES:
            m = rx.match(line)
            if m:
                what = m.groupdict().get("what") or f"{m['code']} fixtures"
                why.setdefault(what, reason(m))
                break
        else:
            if (m := _BUDGET_LINE.match(line)):
                budget = m["why"]
            elif (m := _KEPT_LINE.match(line)):
                kept += [w.strip() for w in m["items"].split(",") if w.strip()]
            elif (m := _NOT_PUBLISHED.match(line)):
                gone.append(m["what"])
    out = []
    for what in dict.fromkeys(kept):
        reason = why.get(what) or (f"not asked: {budget}" if budget else "kept on file")
        out += [{"league": c, "what": what, "why": reason, "at": at} for c in _leagues_of(what, asked)]
    for what in dict.fromkeys(gone):
        parts = what.split(" ")
        if what.endswith("fixtures"):
            on_file = True                        # the fixtures file keeps the rows it had
        elif len(parts) == 2 and parts[1].isdigit():
            on_file = int(parts[1]) in sd.seasons_on_file(root, parts[0])
        else:
            on_file = bool(sd.seasons_on_file(root, parts[0]))
        if on_file:
            out += [{"league": c, "what": what, "why": "not published (kept what is on file)", "at": at}
                    for c in _leagues_of(what, asked)]
    return out


def sync_all(now, feed_path, log=print, leagues=None, data=None):
    """football-data.co.uk (all 38 leagues), then the feeds (`now` None: a
    real run, each feed row stamped when its own request answered) for the
    leagues of `leagues` - default every league the app may carry,
    soccer_leagues.in_app() - that a feed covers (soccer_feed.leagues_for: 30
    leagues and 462 requests today; never RUS, which has a feed and is out of
    the app; the seven with no feed ask nothing). The sync comes before the
    data is read, so it asks about the leagues as decided rather than as
    the data will say after it; the schedule keeps the run's own (run). A
    feed sync that fails outright (it catches each request's failure itself)
    keeps the snapshot on file: the run goes on.

    -> football-data.co.uk's download problems (data_problems): a run that
    stays green on files kept on file says so in matches.json, not only in
    the Actions log - a budget that ran out every run would otherwise starve
    the same leagues (asked last) unseen. data: soccer_data.sync's own
    arguments (a test's made-up site and folders)."""
    data = dict(data or {})
    lines = []

    def heard(m):
        lines.append(m)
        log(m)
    sd.sync(log=heard, **data)
    # at: `now`'s; a real run (now None) stamps them when it reads the clock again (run)
    problems = data_problems(lines, stamp(now), data.get("leagues", "all"), data.get("matches_path"))
    try:
        sfeed.sync(now=now, path=feed_path, log=log,
                   leagues=sfeed.leagues_for(slg.in_app() if leagues is None else leagues))
    except Exception as err:                       # noqa: BLE001 - logged; the snapshot on file is used
        log(f"   feeds: sync failed ({err.__class__.__name__}: {err}); the snapshot on file is used")
    return problems


def measure_groups(frame, preds, pick, walked, leagues):
    """{group: soccer_scores.measure} for each group of the app's leagues
    (soccer_leagues.GROUPS), on the group's own periods (a group's leagues
    share them: the extra leagues' are 2013-2018 / 2019-2025) and its
    leagues' own base rates - the Method tab's table and the scores caveat
    of each group, never the top five's read for another league's card."""
    idx, mats, info = walked
    lg = frame.loc[idx, "league"].to_numpy()
    out = {}
    for g, members in slg.GROUPS.items():
        codes = [c for c in leagues if c in members]
        keep = np.isin(lg, codes)
        if not codes or not keep.any():
            continue
        out[g] = ssc.measure(frame, preds, pick, (idx[keep], mats[keep], info[keep]),
                             periods=bt.periods_for(codes[0]), leagues=codes)
    return out


def run(sync=True, now=None, out_dir=sx.APP_DATA, log_path=sl.LOG, log=print, feed_path=None):
    """One run (the module's notes). `now` None is a real run: the feeds stamp
    each row when its own request was answered, and the clock is read again
    after the sync, so every pick is logged - and checked for freezing - at a
    moment no earlier than the data it was built from. A `now` given (a
    replay, a test) is used throughout."""
    real = now is None
    now = pd.Timestamp(now) if now is not None else now_utc()
    feed_path = feed_path if feed_path is not None else sfeed.FEED
    problems, synced_at = [], None
    if sync:
        log("1. sync")
        problems = sync_all(None if real else now, feed_path, log)
        if real:
            now = now_utc()
        synced_at = stamp(now)
        problems = [dict(p, at=p.get("at") or synced_at) for p in problems]
    # Every league: the model and the pick's weights learn from every league
    # that was measured, shown or not (soccer_model refuses a frame without
    # one); the app's leagues are rule (a)'s on the data as it is now - the
    # feeds' matches to play counted as posted (app_leagues) - and the pick,
    # the score matrices and the Record are computed over those and the
    # decided ones (measured_leagues).
    matches = sd.load_matches(leagues="all")
    fixtures = sd.load_fixtures(leagues="all")
    snap, feed = ssch.feed_matches(feed_path, log=log)
    leagues, out_of_app = app_leagues(matches, fixtures, feed, now)
    on_record = measured_leagues(leagues)
    log(f"   leagues: {len(leagues)} in the app ({' '.join(leagues)}); not in it: "
        f"{' '.join(out_of_app) or 'none'}; measured and recorded: {len(on_record)}")
    fetched_at = fixtures["fetched_at"].max() if len(fixtures) else None
    priced_at = dict(zip(fixtures["match_id"], fixtures["fetched_at"])) if len(fixtures) else {}
    try:
        sched = ssch.schedule(matches, fixtures, feed, now, ahead=pd.Timedelta(AHEAD),
                              behind=pd.Timedelta(BEHIND), log=log, leagues=leagues)
    except Exception as err:                       # noqa: BLE001 - a feed problem never fails the run
        log(f"   feeds: could not be joined ({err.__class__.__name__}: {err}); football-data.co.uk alone")
        feed = []
        sched = ssch.schedule(matches, fixtures, feed, now, ahead=pd.Timedelta(AHEAD),
                              behind=pd.Timedelta(BEHIND), log=log, leagues=leagues)
    # The model's input: football-data.co.uk's results and the matches still
    # to play - never a provisional result.
    frame = sched.frame
    from_feed = sum(1 for v in sched.info.values() if v["source"] == ssch.FEED_SOURCE)
    log(f"   feeds: {len(feed)} matches on file, {from_feed} to play that football-data.co.uk has not posted, "
        f"{len(sched.provisional)} provisional results, {len(sched.off)} postponed/cancelled/abandoned")
    feed_run = (snap or {}).get("last_run") or {}
    if feed_run.get("unmapped") or feed_run.get("skipped_phase"):
        log(f"   feeds: unmapped clubs {feed_run.get('unmapped') or []}; knock-out phases skipped "
            f"{feed_run.get('skipped_phase') or []}")
    log(f"2. model ({smod.MODEL}) over {len(frame):,} matches of {frame['league'].nunique()} leagues")
    preds = smod.predict(frame, leagues=slg.MODEL_LEAGUES)
    blended, weights = smod.shipped_pick(frame, preds, on_record)
    goals_used = smod.goals(frame, preds, leagues)
    ratings = {team: r["egd"] for team, r in (preds.attrs.get("ratings") or {}).items()}
    mkt_o25 = bt.no_vig_total(frame)
    # The score matrix behind every match of the measured leagues the model
    # predicts - by the rule soccer_scores ships - and the tables it is
    # measured by, one per group (the day's cards read their own rows).
    walked = ssc.walk(frame, preds, blended, mkt_o25, leagues=on_record)
    row_of = {i: k for k, i in enumerate(walked[0])}

    def card(i):
        k = row_of.get(i)
        if k is None:
            return None
        r = walked[2].iloc[k]
        src = r["source"] if isinstance(r["source"], str) else None
        return ssc.card(walked[1][k], src, r["reason"], r["miss"], r["miss_on"],
                        withhold=slg.withheld(frame.at[i, "league"], src) if src else ())
    measured = measure_groups(frame, preds, blended, walked, on_record)
    # What the app's form lines and tables read: the results with the
    # provisional ones, flagged - cut by club and league-season once (View).
    shown = ssch.shown(frame, sched.provisional)
    seasons = league_seasons(frame, leagues, now)
    view = sx.View(shown, seasons)

    upcoming = frame["league"].isin(leagues) & frame["result"].isna() & (frame["kickoff"] > now) \
        & (frame["kickoff"] <= now + AHEAD)
    entries = []
    for i in frame.index[upcoming]:
        row = frame.loc[i]
        e = match_entry(row, preds.loc[i], blended.loc[i], bool(blended.loc[i, "model_only"]),
                        None if pd.isna(goals_used[i]) else float(goals_used[i]),
                        None if pd.isna(mkt_o25[i]) else float(mkt_o25[i]),
                        view, now, priced_at.get(row["match_id"]), card=card(i),
                        basis=blended.at[i, "basis"], price_kind=blended.at[i, "price"])
        meta = sched.info.get(row["match_id"]) or {"source": ssch.FD_SOURCE, "time_confirmed": True,
                                                   "freeze_at": row["kickoff"]}
        e["status"] = "upcoming"
        e["time_confirmed"] = bool(meta["time_confirmed"])
        e["freeze_at"] = stamp(meta["freeze_at"])
        e["fixture_source"] = meta["source"]
        e["price_status"] = ("posted" if e["market"] else
                             "not_yet" if meta["source"] == ssch.FEED_SOURCE else "none")
        # Each source's kickoff where they differ: the card lists them.
        if meta.get("kickoff_sources"):
            e["kickoff_sources"] = {k: stamp(v) for k, v in meta["kickoff_sources"].items()}
        entries.append(e)

    log("3. log and grade")
    history = sl.load(log_path)
    off_ids = {o["match_id"] for o in sched.off}
    # Two meetings of a pairing both listed to play are two matches (soccer_log).
    listed = {e["match_id"] for e in entries}
    logged = sum(sl.log_pick(history, log_fields(e), now, off=off_ids, listed=listed) for e in entries if e["pick"])
    stats = {}
    graded = sl.grade(history, matches, now, provisional=sched.provisional, disputed=sched.disputed, stats=stats,
                      feed_status=sched.status)
    sl.save(history, log_path)
    pending = ByKey(p for p in history["picks"].values() if not p.get("void") and not p.get("final"))
    for e in entries:
        p = history["picks"].get(e["match_id"])
        if p is None or p.get("void"):
            # A pick for the match under another id (one log_pick has not
            # moved: its match closed after its old date passed) is the one shown.
            mine = pending.get(entry_key(e))
            p = max(mine, key=lambda x: x.get("last_logged") or "") if mine else p
        if sl.naive_utc(e["freeze_at"]) <= now or (p and not p.get("void") and sl.frozen(p, now)):
            lock(e, p, now)
        elif p and not p.get("void") and p.get("market") and not e["market"]:
            # The log never replaces a priced pick with an unpriced one
            # (football-data.co.uk's file no longer lists the match): the
            # card shows the pick that counts.
            e.update({k: p.get(k) for k in PICK_KEYS}, price_status="posted", priced_kept=True)
        if p and not p.get("void"):
            e["logged_at"] = p.get("last_logged")
    # Each league's own dates: football-data.co.uk's latest result in it (the
    # last its ratings include) and the latest shown, with the feeds'.
    done = matches["league"].isin(leagues) & matches["result"].notna()
    through = matches.loc[done].groupby("league")["date"].max().to_dict()
    prov = sched.provisional
    prov_through = prov.groupby("league")["date"].max().to_dict() if len(prov) else {}
    results_through_of = {c: max([d for d in (through.get(c), prov_through.get(c)) if isinstance(d, str)],
                                 default=None) for c in leagues}
    # The window runs from the latest result in any league - never the
    # earliest: a league between seasons would drag it back months - and,
    # for each league, from its own latest round (`since`: its results
    # through, less BEHIND - the same bound its results list uses).
    data_through = max(through.values()) if through else None
    results_through = max([d for d in results_through_of.values() if d], default=None)
    window = min(now - BEHIND, pd.Timestamp(data_through) - BEHIND) if isinstance(data_through, str) else now - BEHIND
    since = {c: (pd.Timestamp(d) - BEHIND).strftime("%Y-%m-%d") for c, d in results_through_of.items() if d}
    # Every logged pick that has kicked off in the window, or in its league's
    # own latest round - by its logged time, or because it is graded or the
    # feeds say it is under way or over (a match brought forward): graded
    # beside its result, or "started" (postponed, ...) until the result is
    # in. A match on the list to play is shown there, never twice - the same
    # meeting, where pairings repeat.
    #   WHY PER LEAGUE: the main leagues' results land within a day, so the
    # global window moves on while a league whose results come later (POL's
    # Nov 6-8 round was not on football-data.co.uk on Nov 9; seven leagues
    # have no feed to grade provisionally) still has its round to come: its
    # picks dropped off the Matches tab, and when the result landed the pick
    # was graded out of sight. Its own round keeps them, by the UK day the
    # results list reads.
    started = ByKey((k for k, s in sched.status.items() if s in ssch.STARTED), key=lambda k: k)
    to_play = ByKey(entries, key=entry_key)
    kicked = []
    for p in history["picks"].values():
        if p.get("void") or not p.get("kickoff"):
            continue
        kick = sl.naive_utc(p["kickoff"])
        own = since.get(p.get("league"))
        # (a day's margin before the UK-day test: the log holds thousands of old picks)
        mine = own and kick >= pd.Timestamp(own) - pd.Timedelta(days=1) and ssch.uk_day(kick) >= own
        if (kick < window and not mine) or kick > now + AHEAD:
            continue
        k = ssch.key_of(p)
        if k in to_play or not (kick <= now or p.get("final") or k in started):
            continue
        kicked.append(p)
    shown_picks = [logged_entry(p, view, sched.status, sched.feed_kickoff, sched.disputes) for p in kicked]
    # A postponed or undated match whose pick is on screen already is not
    # shown twice; one with a pick logged before the feeds said so shows it.
    on_screen = ByKey(kicked)
    off, tbc = [], []
    for o in sched.off:
        k = o.get("key") or ssch.key_of(o)
        if k not in on_screen:
            e = off_entry(o, view)
            mine = pending.get(k)
            off.append(hold(e, max(mine, key=lambda p: p["kickoff"])) if mine else e)
    for t in sched.tbc:
        k = t.get("key") or ssch.key_of(t)
        if k not in on_screen:
            e = tbc_entry(t, view, now)
            mine = pending.get(k)
            tbc.append(hold(e, max(mine, key=lambda p: p["kickoff"])) if mine else e)
    log(f"   {len(entries)} upcoming ({sum(1 for e in entries if e['price_status'] == 'posted')} priced, "
        f"{sum(1 for e in entries if e['pick'] and not e['market'])} the model alone, "
        f"{sum(1 for e in entries if not e['pick'])} no pick), {logged} logged, "
        f"{graded} graded ({stats.get('provisional', 0)} provisionally), {stats.get('confirmed', 0)} provisional "
        f"confirmed, {stats.get('corrected', 0)} corrected, {stats.get('withdrawn', 0)} withdrawn, "
        f"{stats.get('late', 0)} void as logged after kickoff ({stats.get('restored', 0)} restored), "
        f"{len(shown_picks)} kicked off or over, {len(off)} postponed/cancelled/abandoned, {len(tbc)} date to come")

    log("4. write the app data")
    # The Record over the measured leagues (measured_leagues says why), each
    # league's row flagged in_app or not.
    bts = backtest_summary(frame, preds, blended, weights, goals_used, on_record, in_app=leagues)
    # The top-level season and label are the August-May one, the five's, as
    # before; each league has its own in `leagues`.
    season = sd.current_season(now)
    season_label = f"{season}-{(season + 1) % 100:02d} season"
    # The pick is the market (soccer_model says why); the model's own view is
    # shown beside it.
    basis = "market"
    # The global caveats over the measured leagues too: their numbers are the
    # top five's whatever the day's league set is, and are never deleted.
    caveats = sx.caveats(bts, basis, model_through=data_through, scores=measured.get("top"), leagues=on_record)
    # The latest round's results, as facts, so the tab is never empty between
    # rounds: each league's own latest round (its results_through less
    # BEHIND), only matches with no pick ON SCREEN (a logged one is shown with
    # its pick above, whatever its id became; one whose pick is void - logged
    # after kickoff - or is not on screen is a fact like any other: a result
    # is never hidden behind a pick nobody can see) - the same meeting, where
    # pairings repeat. football-data.co.uk's, and the feeds' provisional ones
    # it has not posted yet, flagged.
    first = min(since.values(), default=None)
    last = matches[done & (matches["date"] >= first)] if first else matches.iloc[:0]
    results = [{"match_id": r.match_id, "league": r.league, "date": r.date, "kickoff": stamp(r.kickoff),
                "home": view.chip(r.league, r.home), "away": view.chip(r.league, r.away), "hg": int(r.hg),
                "ag": int(r.ag), "provisional": False}
               for r in last.itertuples() if r.date >= since.get(r.league, "9999")
               and ssch.key(r.league, r.season, r.home, r.away, r.date) not in on_screen]
    recent = prov[prov["date"] >= first] if first and len(prov) else prov.iloc[:0]
    results += [{"match_id": r.match_id, "league": r.league, "date": r.date, "kickoff": stamp(r.kickoff),
                 "home": view.chip(r.league, r.home), "away": view.chip(r.league, r.away), "hg": int(r.hg),
                 "ag": int(r.ag), "provisional": True, "source": r.source}
                for r in recent.itertuples() if r.league in set(leagues) and r.date >= since.get(r.league, "9999")
                and ssch.key(r.league, r.season, r.home, r.away, r.date) not in on_screen]
    cards = shown_picks + entries + off + tbc
    # Every league in the app, and any measured league a card or a result is
    # of (a league that has just left the app still has picks on screen).
    seen = {x["league"] for x in cards + results}
    listed_leagues = list(leagues) + [c for c in sd.ALL_LEAGUES if c in seen and c not in leagues and c in slg.SHIPPED]
    seasons_all = dict(league_seasons(frame, listed_leagues, now), **seasons)
    view.seasons.update(seasons_all)
    # Each group's scores caveat from its own measure (the global one is the top five's).
    groups = sx.groups_payload(leagues)
    for g in groups:
        groups[g]["scores_caveat"] = sx.scores_caveat(measured.get(g), scope=slg.GROUP_NAMES[g])
    sx.write("matches", {
        "season": season, "season_label": season_label,
        "leagues": sx.leagues(listed_leagues, seasons_all, through, results_through_of, in_app=leagues),
        # every league in the app, in soccer_data's order, even one with no match on screen
        "league_order": list(leagues), "groups": groups,
        "bands": sx.bands_payload(), "band_floors": sx.band_floors(), "not_in_app": sx.not_in_app(leagues, out_of_app),
        "matches": cards, "results": results, "data_through": data_through,
        # The date of football-data.co.uk's latest result in any of the app's
        # leagues: the last the model's ratings include (each league's own is
        # in `leagues`). Results after it are provisional.
        "model_through": data_through, "results_through": results_through,
        "provisional_results": int(len(prov)),
        "feeds_at": feed_run.get("at"), "prices_at": stamp(fetched_at),
        # data_problems: football-data.co.uk's files this run's sync kept on
        # file (failed, short, unreadable, budget spent) or found no longer
        # published, per league - that league's data may be stale
        # (sync_all); data_synced_at None: this run did not sync.
        "feeds": {"at": feed_run.get("at"), "unmapped": feed_run.get("unmapped") or [],
                  "skipped_phase": feed_run.get("skipped_phase") or [],
                  "data_problems": problems, "data_synced_at": synced_at},
        "caveats": caveats, "basis": basis, "generated": sx.now_stamp(now),
        # How the goals and scores numbers and the pick's bands were measured,
        # by this run's own walk-forward, per group (the Method tab's table);
        # at the top level the top five's, as the app has always read it.
        "scores_measured": dict(measured.get("top") or {}, **measured),
    }, out_dir)
    # The combined frame, so a league whose season has fixtures but no
    # results yet still has its clubs (and their ratings) listed; with the
    # provisional results, flagged. Each league on its own season.
    sx.write("tables", dict(sx.tables_payload(view, season, ratings=ratings or None, generated=sx.now_stamp(now),
                                              caveat=caveats, season_label=season_label, leagues=leagues,
                                              seasons=seasons),
                            model_through=data_through), out_dir)
    sx.write("performance", {"backtest": bts, "basis": basis, "track_record": live_record(history),
                             "model_through": data_through, "generated": sx.now_stamp(now)}, out_dir)
    return {"entries": entries, "shown": shown_picks, "off": off, "tbc": tbc, "results": results,
            "backtest": bts, "weights": weights, "frame": frame, "schedule": sched,
            "scores_measured": measured, "leagues": leagues, "on_record": on_record, "out_of_app": out_of_app,
            "data_problems": problems, "pick": blended, "preds": preds}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args(argv)
    run(sync=not a.no_sync)
    import soccer_standalone
    soccer_standalone.build()


if __name__ == "__main__":
    sys.exit(main())
