"""Every league football-data.co.uk carries - measured walk-forward, and
what of each ships.

THE IDEA. The app predicted Europe's top five leagues. football-data.co.uk
carries 38 leagues with results and prices (soccer_data.ALL_LEAGUES): the
top five and their second tiers; twelve more European divisions (E2 E3 EC
SC0-SC3 N1 B1 P1 T1 G1) with pre-match AND closing prices, over/under and -
in about half their seasons - shots, since 2005-06; and sixteen "extra"
leagues (ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE SWZ USA)
since 2012 with CLOSING 1X2 prices only - no over/under, no shots, no
pre-match price in history (their live fixtures carry pre-match prices).
Each league is measured by the procedure that ships for the top five, and a
league, a number or a kind of pick reaches the app only where its own
measurement says so.

THE PROTOCOL (fixed before any number was seen; the lead's, with the
details this module had to settle written down before the first run).
  periods   the 22 European divisions DEV 2006-2015, CONFIRM 2016-2025
            (2005-06 warms up); the 16 extra leagues 2012 warm-up only, DEV
            2013-2018, CONFIRM 2019-2025 (soccer_backtest.periods_for).
            Season 2026 is live and never measured. Everything walks forward
            on DATES: ARG files one 2013/14 match on 29/01/2015, after
            season 2014's start. A league's first season on file is a
            warm-up (it has no earlier season for the base rate): only CHN,
            whose file starts in 2014, is affected.
  prices    the European divisions' pre-match average (as the top five);
            the extra leagues' closing average (all their history has),
            while their live pick will have the pre-match price. The gap
            between the two is measured on the European divisions'
            2019-2025, where both exist (log loss of the pre-match no-vig
            price against the closing one, same matches), and stated beside
            every extra league. Pinnacle's prices are never read (stale
            after 2025-07-23); the market averages are not affected.
  the model shots_ratings walk-forward extended to every league
            (shots_ratings.predict_leagues): one rating pool per country -
            England E0-EC, Scotland SC0-SC3, Spain/Italy/Germany/France top
            two, every one-league country its own; goals only where there
            are no shots (the proxy renormalises). Settled here, before any
            number: (i) rating passes by calendar - the top five's
            countries; every other August-May league; the leagues whose
            seasons are, or were through 2025, calendar years (ARG BRA CHN
            FIN IRL JPN NOR SWE USA) - because rate() turns its tier-step
            measurement over at a pass's first new season; (ii) one Poisson
            regression per league group, a dummy per league, slopes and
            stakes terms shared: the top five's exactly as shipped (trained
            on the top five only), the other 17 European divisions', the 16
            extra leagues'; (iii) the table's stakes for every league, read
            only while a club has played fewer than a double round robin
            (split seasons and play-offs read nothing at stake); (iv) a
            match filed under an earlier season than the club's current one
            (ARG's) never restarts the club as new. On the ten leagues with
            the top five predicted, every number is the shipped model's bit
            for bit. England's pool: E0-EC against E0-E1 as today, judged by
            the top five's DEV model log loss; if E0-EC is worse, E0-E1
            stays (E2, E3, EC then a pool of their own).
  the pick  the league's market price recalibrated, p ~ q^beta exp(c), with
            weights fitted only on matches dated before the league-season's
            first match (soccer_model.pick_leagues). ONE of three weight
            options is chosen on DEV - the mean DEV log loss over every DEV
            match of every league, each match once: per league (its own
            earlier matches; fewer than 1,000: the raw price), pooled per
            price type (every European division's pre-match prices; every
            extra league's closing prices), or the top five's weights for
            all (pick()'s season fits; for a league-season the latest whose
            matches all precede its first match). A match with no price:
            the model alone.
  ship (a)  a league is in the app if it has a current season in the data
            (a result in the 45 days to 2026-10-29, or a fixture posted on
            or after it) - and its pick is the recalibrated market if that
            is not worse in log loss than the raw no-vig market on its DEV,
            else the raw no-vig market.
       (b)  model-alone picks (no price yet) are shown for a league only if
            the model beats the league's base rate - its home / draw / away
            frequencies over its earlier seasons - on log loss in BOTH
            periods; otherwise such a match shows "no pick until priced".
       (c)  each score number - both teams to score, over 1.5, over 3.5, the
            exact score and, added here under the same rule (nothing ships
            unmeasured), over 2.5 - is shown for a league only if the source
            it would come from beats the league's base rate on log loss in
            BOTH periods. Two sources, judged apart: the market-implied
            matrix K (where the 1X2 and the over/under are priced: the
            European divisions; over 2.5 is then the over/under price
            itself) and the model's own matrix M (a match priced 1X2 only -
            every extra league's - or not priced at all).
       (d)  bands keep the edges 0.40 / 0.50 / 0.60; their hit rates, the
            pick's and the model's own picks', are measured per league
            group (top five; other European divisions; extra leagues - the
            leagues in the app) on CONFIRM and shipped with them.
  same rows every comparison of sources is on the same matches: a result,
            the league's measurement price, a model number and a base rate.
  point in time  the whole pipeline (model, pick, base rates, score
            matrices) re-run at one cut per period with every result and
            match statistic from the cut on scrambled and prices after the
            cut shuffled - nothing dated on or before the cut may move - and
            again with prices from the cut on shuffled too - nothing dated
            before it may move.

MEASURED, DECISIONS, CAVEATS: filled in from the run below (main() prints
every table; the results JSON is OUT).
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_backtest as bt  # noqa: E402
import soccer_data as sd  # noqa: E402
import soccer_model as smod  # noqa: E402
import soccer_scores as ssc  # noqa: E402
from experiments import shots_ratings as sr  # noqa: E402

OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/leagues.json")
TODAY = "2026-10-29"
RECENT_DAYS = 45
ALL = list(sd.ALL_LEAGUES)
GROUPS = {"top": list(sd.TOP), "europe": list(sr.EUROPE_OTHER), "extra": list(sd.EXTRA)}
GROUP_NAMES = {"top": "top five", "europe": "other European divisions", "extra": "extra leagues"}
NUMBERS = ssc.NUMBERS                      # btts, o15, o25, o35, exact
CUTS = ("2014-04-12", "2021-10-16")        # one Saturday in each period, inside both kinds' periods
STATS = ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar")
PRICES = tuple(c for c in sd.SOURCES if c.startswith(("avg", "max", "ps", "b365")))
PIECE = ["p_h", "p_d", "p_a", "p_o25", "lam", "nu", "rho"]


def group_of(code):
    return next(g for g, codes in GROUPS.items() if code in codes)


# ----------------------------------------------------------- (a) a current season

def current_season(matches, fixtures, code, today=TODAY, days=RECENT_DAYS):
    """Whether a league has a current season in the data: a result within
    `days` before `today`, or a fixture on file dated `today` or later."""
    res = matches[(matches["league"] == code) & matches["result"].notna()]
    last = res["date"].max() if len(res) else None
    since = (pd.Timestamp(today) - pd.Timestamp(last)).days if last else None
    ahead = fixtures[(fixtures["league"] == code) & (fixtures["date"] >= today)]
    ok = (since is not None and since <= days) or len(ahead) > 0
    return {"ok": bool(ok), "last_result": last, "days_since": since, "fixtures_ahead": int(len(ahead)),
            "season": int(res["season"].max()) if len(res) else None}


# ----------------------------------------------------------- the pipeline

def numbers(frame, preds, pick, rows):
    """K (where shown_matrices gives the market's) and M for the given rows
    (a boolean mask): {source: {btts, o15, o25, o35, cells}} with NaN where
    the source does not apply, plus K's info. Row-local, computed in chunks."""
    idx = np.flatnonzero(rows)
    n = len(idx)
    out = {s: {k: np.full(n, np.nan) for k in ("btts", "o15", "o25", "o35")} for s in ("K", "M")}
    for s in out:
        out[s]["cells"] = np.full((n, ssc.N_CELLS), np.nan)
    src = np.full(n, None, dtype=object)
    flagged = np.zeros(n, bool)
    q_o = bt.no_vig_total(frame).to_numpy(float)
    p3 = pick[["p_h", "p_d", "p_a"]].to_numpy(float).copy()
    p3[pick["model_only"].to_numpy(bool)] = np.nan
    lam, nu, rho = (preds[c].to_numpy(float) for c in ("lam", "nu", "rho"))
    for a in range(0, n, 20000):
        j = idx[a:a + 20000]
        mats, info = ssc.shown_matrices(p3[j, 0], p3[j, 1], p3[j, 2], q_o[j], rho[j], lam[j], nu[j])
        mk = ssc.markets(np.nan_to_num(mats, nan=0.0))
        own = ssc.markets(ssc.dc_matrix(lam[j], nu[j], rho[j]))
        market = info["source"] == ssc.SOURCE_MARKET
        for k in ("btts", "o15", "o35"):
            out["K"][k][a:a + len(j)] = np.where(market, mk[k], np.nan)
            out["M"][k][a:a + len(j)] = own[k]
        out["K"]["o25"][a:a + len(j)] = np.where(market, q_o[j], np.nan)      # the card's over 2.5: the price
        out["M"]["o25"][a:a + len(j)] = preds["p_o25"].to_numpy(float)[j]
        out["K"]["cells"][a:a + len(j)] = np.where(market[:, None], mk["cells"], np.nan)
        out["M"]["cells"][a:a + len(j)] = own["cells"]
        src[a:a + len(j)] = info["source"]
        flagged[a:a + len(j)] = info["flagged"]
        # the whole shown matrix, for the point-in-time comparison
        if a == 0:
            shown = np.empty((n, mats.shape[1], mats.shape[2]))
        shown[a:a + len(j)] = mats
    return {"idx": idx, "K": out["K"], "M": out["M"], "source": src, "flagged": flagged,
            "shown": shown if n else np.empty((0, 11, 11))}


def pipeline(frame, england, option, raw=(), score_rows=None, preds=None):
    """Every number for every match of every league, as the app would build
    it: the model (predict_leagues, England's pool `england`), the pick
    (pick_leagues: weight `option`, the raw price for leagues in `raw`), the
    base rates and - for `score_rows` (a mask; default every row the model
    predicts) - the score matrices."""
    t0 = time.time()
    preds = sr.predict_leagues(frame, ALL, score_params=True, england=england) if preds is None else preds
    t_model = time.time() - t0
    pick, weights = smod.pick_leagues(frame, preds, ALL, option, raw=raw)
    base = ssc.baseline(frame, ALL)
    rows = preds["p_h"].notna().to_numpy() if score_rows is None else (score_rows & preds["p_h"].notna().to_numpy())
    nums = numbers(frame, preds, pick, rows)
    return {"preds": preds, "pick": pick, "weights": weights, "base": base, "numbers": nums,
            "runtime_model": t_model, "runtime": time.time() - t0}


# ----------------------------------------------------------- losses

def ll3(p, result):
    """Per-match three-way log loss (NaN where p or the result is missing)."""
    p = np.asarray(p, float)
    idx = pd.Series(result).map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ok = ~pd.isna(idx) & np.isfinite(p).all(axis=1)
    out = np.full(len(p), np.nan)
    i = idx[ok].astype(int)
    out[ok] = -np.log(np.clip(p[ok][np.arange(ok.sum()), i], 1e-12, 1))
    return out


def hit3(p, result):
    p = np.asarray(p, float)
    side = np.array(bt.RESULTS)[np.nan_to_num(p).argmax(axis=1)]
    return (side == np.asarray(result)).astype(float)


def bin_ll(p, y):
    return ssc._bin_loss(p, y)


def common(frame, q, preds, base, code, period):
    """The same rows for every source: league `code`, a season in `period`,
    a result, the league's measurement price, a model number and a base rate."""
    lo, hi = period
    return ((frame["league"] == code) & frame["season"].between(lo, hi) & frame["result"].isin(bt.RESULTS)
            & q.notna().all(axis=1) & preds["p_h"].notna() & base["p_h"].notna()).to_numpy()


def mean(x):
    x = np.asarray(x, float)
    return round(float(np.mean(x)), 5) if len(x) else None


# ----------------------------------------------------------- the steps

def england_choice(frame, q, preds_all, preds_two, base):
    """The top five's model log loss on the same matches, E0-EC against
    E0-E1 (the protocol: E0-EC only if not worse on DEV); England's other
    divisions beside it, for the record."""
    out = {}
    for name, codes in (("top five", sd.TOP), ("E1", ["E1"]), ("E2-EC", ["E2", "E3", "EC"])):
        out[name] = {}
        for key, per in (("dev", bt.DEV), ("confirm", bt.CONFIRM)):
            sel = np.zeros(len(frame), bool)
            for c in codes:
                sel |= common(frame, q, preds_all, base, c, per) & preds_two["p_h"].notna().to_numpy()
            res = frame["result"].to_numpy()[sel]
            a = ll3(preds_all[["p_h", "p_d", "p_a"]].to_numpy(float)[sel], res)
            b = ll3(preds_two[["p_h", "p_d", "p_a"]].to_numpy(float)[sel], res)
            d = a - b
            out[name][key] = {"matches": int(sel.sum()), "E0-EC": mean(a), "E0-E1": mean(b),
                              "diff": round(float(d.mean()), 6), "diff_se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 6)}
    choice = "all" if out["top five"]["dev"]["E0-EC"] <= out["top five"]["dev"]["E0-E1"] else "top-two"
    return choice, out


def weight_choice(frame, q, preds, base):
    """Each weight option's pick on every league's DEV rows (pooled, each
    match once), and per group; the lowest pooled DEV log loss is chosen.
    Returns (choice, table, {option: pick frame})."""
    picks, table = {}, {}
    dev = np.zeros(len(frame), bool)
    for code in ALL:
        dev |= common(frame, q, preds, base, code, bt.periods_for(code)[0])
    res = frame["result"].to_numpy()
    raw_ll = ll3(q.to_numpy(float), res)
    for opt in smod.WEIGHT_OPTIONS:
        t0 = time.time()
        picks[opt], _ = smod.pick_leagues(frame, preds, ALL, opt)
        ll = ll3(picks[opt][["p_h", "p_d", "p_a"]].to_numpy(float), res)
        row = {"dev_all": {"matches": int(dev.sum()), "log_loss": mean(ll[dev]), "raw": mean(raw_ll[dev])},
               "seconds": round(time.time() - t0, 1)}
        for g, codes in GROUPS.items():
            sel = dev & frame["league"].isin(codes).to_numpy()
            row[g] = {"matches": int(sel.sum()), "log_loss": mean(ll[sel]), "raw": mean(raw_ll[sel])}
        table[opt] = row
    choice = min(smod.WEIGHT_OPTIONS, key=lambda o: table[o]["dev_all"]["log_loss"])
    return choice, table, picks


def league_rows(frame, q, kind, preds, base, pick_opt, pick, decisions_season):
    """Per league and period: matches, log loss of the shipped pick, the
    chosen option's pick before rule (a), the raw no-vig market, the model,
    the base rate; the pick's accuracy (and the raw market's)."""
    res = frame["result"].to_numpy()
    srcs = {"pick": pick[["p_h", "p_d", "p_a"]].to_numpy(float),
            "pick_recalibrated": pick_opt[["p_h", "p_d", "p_a"]].to_numpy(float),
            "market": q.to_numpy(float), "model": preds[["p_h", "p_d", "p_a"]].to_numpy(float),
            "base": base[["p_h", "p_d", "p_a"]].to_numpy(float)}
    losses = {s: ll3(p, res) for s, p in srcs.items()}
    hits = {s: hit3(srcs[s], res) for s in ("pick", "market", "model")}
    out = {}
    for code in ALL:
        dev, conf = bt.periods_for(code)
        r = {"name": sd.ALL_LEAGUES[code].name, "country": sd.ALL_LEAGUES[code].country, "group": group_of(code),
             "price": "closing" if code in sd.EXTRA else "pre-match", "season_check": decisions_season[code]}
        for key, per in (("dev", dev), ("confirm", conf)):
            sel = common(frame, q, preds, base, code, per)
            r[key] = {"seasons": list(per), "matches": int(sel.sum())}
            if sel.any():
                r[key].update({s: mean(losses[s][sel]) for s in srcs})
                r[key].update({f"acc_{s}": round(float(hits[s][sel].mean()), 4) for s in hits})
                d = losses["model"][sel] - losses["base"][sel]
                r[key]["model_vs_base_se"] = round(float(d.std(ddof=1) / math.sqrt(len(d))), 5) if len(d) > 1 else None
                d = losses["pick_recalibrated"][sel] - losses["market"][sel]
                r[key]["recal_vs_market_se"] = round(float(d.std(ddof=1) / math.sqrt(len(d))), 5) if len(d) > 1 else None
        out[code] = r
    return out


def score_rows(frame, q, preds, base, nums):
    """Per league, period, source (K, M) and number: matches, the source's
    log loss and the base rate's on the same matches."""
    pos = {i: k for k, i in enumerate(nums["idx"])}
    take = np.array([pos.get(i, -1) for i in range(len(frame))])
    hg, ag = frame["hg"].to_numpy(float), frame["ag"].to_numpy(float)
    ys = {"btts": ((hg > 0) & (ag > 0)).astype(float), "o15": (hg + ag > 1.5).astype(float),
          "o25": (hg + ag > 2.5).astype(float), "o35": (hg + ag > 3.5).astype(float)}
    cat = ssc.score_index(np.nan_to_num(hg), np.nan_to_num(ag))
    bcells = base[[f"c{k}" for k in range(ssc.N_CELLS)]].to_numpy(float)
    out = {}
    for code in ALL:
        out[code] = {}
        for key, per in zip(("dev", "confirm"), bt.periods_for(code)):
            sel = common(frame, q, preds, base, code, per) & (take >= 0)
            out[code][key] = {}
            for s in ("K", "M"):
                k_rows = take[sel]
                has = np.isfinite(nums[s]["btts"][k_rows])
                rows = np.flatnonzero(sel)[has]
                kr = k_rows[has]
                rep = {"matches": int(len(rows))}
                if len(rows):
                    for t in ("btts", "o15", "o25", "o35"):
                        rep[t] = {"log_loss": mean(bin_ll(nums[s][t][kr], ys[t][rows])),
                                  "base": mean(bin_ll(base[t].to_numpy(float)[rows], ys[t][rows]))}
                    pe = nums[s]["cells"][kr, cat[rows]]
                    pb = bcells[rows, cat[rows]]
                    C = nums[s]["cells"][kr]
                    top1 = np.argsort(-C[:, :-1], axis=1, kind="stable")[:, 0]
                    rep["exact"] = {"log_loss": mean(-np.log(np.clip(pe, 1e-12, 1))),
                                    "base": mean(-np.log(np.clip(pb, 1e-12, 1))),
                                    "top1_hit": round(float((top1 == cat[rows]).mean()), 4)}
                    rep["btts"]["mean_p"] = round(float(np.mean(nums[s]["btts"][kr])), 4)
                    rep["btts"]["rate"] = round(float(ys["btts"][rows].mean()), 4)
                out[code][key][s] = rep
    return out


def decide(leagues, scores, season):
    """Rules (a)-(c) from the measured numbers."""
    out = {}
    for code in ALL:
        r = leagues[code]
        dev, conf = r["dev"], r["confirm"]
        in_app = season[code]["ok"]
        recal = dev.get("pick_recalibrated") is not None and dev["pick_recalibrated"] <= dev["market"]
        model_only = all(p.get("model") is not None and p["model"] < p["base"] for p in (dev, conf))
        shown = {}
        for s in ("K", "M"):
            if code in sd.EXTRA and s == "K":
                shown[s] = None                                    # no over/under price: no K
                continue
            sd_, sc_ = scores[code]["dev"][s], scores[code]["confirm"][s]
            shown[s] = [t for t in NUMBERS
                        if sd_.get("matches") and sc_.get("matches")
                        and sd_[t]["log_loss"] < sd_[t]["base"] and sc_[t]["log_loss"] < sc_[t]["base"]]
        out[code] = {"in_app": in_app, "pick": "recalibrated" if recal else "raw", "model_only": model_only,
                     "numbers": {"market": shown["K"], "model": shown["M"]}}
    return out


def bands(frame, q, preds, base, pick, decisions):
    """Rule (d): per league group (the leagues in the app), per period, the
    pick's band hit rates and the model's own picks' at the same edges."""
    res = frame["result"].to_numpy()
    pk = pick[["p_h", "p_d", "p_a"]].to_numpy(float)
    md = preds[["p_h", "p_d", "p_a"]].to_numpy(float)
    out = {}
    for g, codes in GROUPS.items():
        out[g] = {}
        for k, key in enumerate(("dev", "confirm")):
            sel = np.zeros(len(frame), bool)
            for c in codes:
                if decisions[c]["in_app"]:
                    sel |= common(frame, q, preds, base, c, bt.periods_for(c)[k])
            out[g][key] = {
                "pick": ssc._band_table(pk[sel].max(axis=1), hit3(pk[sel], res[sel]).astype(bool)),
                "model": ssc._band_table(md[sel].max(axis=1), hit3(md[sel], res[sel]).astype(bool))}
    return out


def price_gap(frame):
    """Pre-match against closing no-vig log loss where both exist: the
    European divisions, seasons 2019-2025, per group and per league."""
    pre, close = bt.no_vig(frame, "pre"), bt.no_vig(frame, "close")
    sel = (frame["league"].isin(sd.MAIN) & frame["season"].between(2019, 2025) & frame["result"].isin(bt.RESULTS)
           & pre.notna().all(axis=1) & close.notna().all(axis=1)).to_numpy()
    res = frame["result"].to_numpy()
    a, b = ll3(pre.to_numpy(float), res), ll3(close.to_numpy(float), res)
    out = {}
    for name, codes in [("all", sd.MAIN), ("top", sd.TOP), ("europe", sr.EUROPE_OTHER)] + [(c, [c]) for c in sd.MAIN]:
        s = sel & frame["league"].isin(codes).to_numpy()
        d = a[s] - b[s]
        out[name] = {"matches": int(s.sum()), "pre": mean(a[s]), "close": mean(b[s]),
                     "gap": round(float(d.mean()), 5), "gap_se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 5),
                     "acc_pre": round(float(hit3(pre.to_numpy(float)[s], res[s]).mean()), 4),
                     "acc_close": round(float(hit3(close.to_numpy(float)[s], res[s]).mean()), 4)}
    return out


# ----------------------------------------------------------- point in time

def scramble(frame, cut, prices_from_cut=False, seed=1):
    """Every result and match statistic from `cut` on replaced by random
    ones (as soccer_backtest.check_point_in_time); every price of the
    matches AFTER the cut - or, prices_from_cut, from the cut on - shuffled
    among those matches."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    later = (out["date"] >= cut).to_numpy()
    for c in STATS:
        out.loc[later, c] = rng.integers(0, 6, int(later.sum()))
    hg, ag = out.loc[later, "hg"].astype(float), out.loc[later, "ag"].astype(float)
    out.loc[later, "result"] = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    move = (out["date"] >= cut).to_numpy() if prices_from_cut else (out["date"] > cut).to_numpy()
    perm = rng.permutation(np.flatnonzero(move))
    out.loc[move, list(PRICES)] = out.iloc[perm][list(PRICES)].to_numpy()
    return out


def _flat(run, rows):
    """Every number of a pipeline run for the given rows, one matrix."""
    nums = run["numbers"]
    pos = {i: k for k, i in enumerate(nums["idx"])}
    k = np.array([pos.get(i, -1) for i in np.flatnonzero(rows)])
    have = k >= 0
    parts = [run["preds"][PIECE].to_numpy(float)[rows], run["pick"][["p_h", "p_d", "p_a"]].to_numpy(float)[rows],
             run["base"].to_numpy(float)[rows]]
    shown = np.full((int(rows.sum()), 121), np.nan)
    shown[have] = nums["shown"][k[have]].reshape(int(have.sum()), -1)
    parts.append(shown)
    return np.column_stack(parts)


def point_in_time(frame, cut, england, option, raw, reference=None):
    """The whole pipeline re-run with the future scrambled (module notes):
    nothing on or before the cut moves with results from the cut on and
    prices after it scrambled; nothing before the cut with prices from the
    cut on scrambled as well. Score matrices are compared for the rows up to
    the cut (a matrix is row-local: soccer_scores' notes). reference: the
    model's predictions on the unscrambled frame, when already computed by
    the same call."""
    upto = (frame["date"] <= cut).to_numpy()
    before = (frame["date"] < cut).to_numpy()
    ref = pipeline(frame, england, option, raw, score_rows=upto, preds=reference)
    a = _flat(ref, upto)
    b = _flat(pipeline(scramble(frame, cut), england, option, raw, score_rows=upto), upto)
    c_run = pipeline(scramble(frame, cut, prices_from_cut=True), england, option, raw, score_rows=before)
    c = _flat(c_run, before)
    a_before = _flat(ref, before)
    on_day = (frame["date"] == cut).to_numpy()
    return {"cut": cut, "rows_on_or_before": int(upto.sum()), "rows_on_cut_day": int(on_day.sum()),
            "leagues_on_cut_day": sorted(set(frame.loc[on_day, "league"])),
            "numbers_compared": int(np.isfinite(a).sum()),
            "unmoved_on_or_before_cut": bool(np.array_equal(a, b, equal_nan=True)),
            "unmoved_before_cut_prices_from_cut_scrambled": bool(np.array_equal(a_before, c, equal_nan=True))}


# ----------------------------------------------------------- the run

def run(pit=True, log=print, frame=None, fixtures=None, cuts=CUTS):
    """The whole measurement (module notes). frame / fixtures: what to
    measure on (default every league's snapshot; the tests and a DEV-only
    debugging pass give a slice)."""
    t_all = time.time()
    frame = sd.load_matches(leagues="all") if frame is None else frame
    fixtures = sd.load_fixtures(leagues="all") if fixtures is None else fixtures
    log(f"{len(frame):,} matches in {frame['league'].nunique()} leagues, {len(fixtures)} fixtures on file")
    q, kind = smod.market_price(frame)

    t0 = time.time()
    preds_all = sr.predict_leagues(frame, ALL, score_params=True, england="all")
    t_model = time.time() - t0
    log(f"model, every league, England E0-EC: {t_model:.1f}s")
    preds_two = sr.predict_leagues(frame, ALL, score_params=True, england="top-two")
    log("model, England E0-E1 as before: done")
    base = ssc.baseline(frame, ALL)
    england, eng_table = england_choice(frame, q, preds_all, preds_two, base)
    preds = preds_all if england == "all" else preds_two
    log(f"England's pool: {'E0-EC' if england == 'all' else 'E0-E1'} (top five DEV model log loss "
        f"E0-EC {eng_table['top five']['dev']['E0-EC']} vs E0-E1 {eng_table['top five']['dev']['E0-E1']})")

    option, w_table, picks = weight_choice(frame, q, preds, base)
    log(f"weights: {option}")
    season = {c: current_season(frame, fixtures, c) for c in ALL}
    first = league_rows(frame, q, kind, preds, base, picks[option], picks[option], season)
    raw = [c for c in ALL if first[c]["dev"].get("pick_recalibrated") is not None
           and first[c]["dev"]["pick_recalibrated"] > first[c]["dev"]["market"]]
    pick, weights = smod.pick_leagues(frame, preds, ALL, option, raw=raw)
    leagues = league_rows(frame, q, kind, preds, base, picks[option], pick, season)
    t0 = time.time()
    nums = numbers(frame, preds, pick, preds["p_h"].notna().to_numpy())
    t_scores = time.time() - t0
    scores = score_rows(frame, q, preds, base, nums)
    decisions = decide(leagues, scores, season)
    assert sorted(raw) == sorted(c for c in ALL if decisions[c]["pick"] == "raw")
    band = bands(frame, q, preds, base, pick, decisions)
    gap = price_gap(frame)
    g = gap["all"]
    if g["matches"]:
        for c in sd.EXTRA:
            leagues[c]["caveat"] = (
                f"Measured on closing prices, the only ones its history has; its live pick uses the pre-match "
                f"price. On the European divisions' 2019-2025, where both exist, the pre-match price scored "
                f"{g['gap']:.4f} worse in log loss than the closing one ({g['pre']:.4f} against {g['close']:.4f}, "
                f"{g['matches']:,} matches): its live picks should do about that much worse than measured.")
    latest = {}
    for c in ALL:
        ws = sorted((s, w) for (lg, s), w in weights.items() if lg == c and w is not None)
        latest[c] = {"season": ws[-1][0], "w": [round(v, 4) for v in ws[-1][1]]} if ws else None
    out = {"protocol": {"england": england, "weights": option, "cuts": list(CUTS), "today": TODAY,
                        "recent_days": RECENT_DAYS},
           "runtime": {"model_every_league_s": round(t_model, 1), "score_matrices_s": round(t_scores, 1),
                       "matches": int(len(frame))},
           "england": eng_table, "weight_options": w_table, "leagues": leagues, "scores": scores,
           "decisions": decisions, "bands": band, "price_gap": gap, "latest_weights": latest}
    if pit:
        out["point_in_time"] = []
        for cut in cuts:
            t0 = time.time()
            r = point_in_time(frame, cut, england, option, raw, reference=preds)
            r["seconds"] = round(time.time() - t0, 1)
            out["point_in_time"].append(r)
            log(f"point in time {cut}: {r}")
    out["runtime"]["experiment_s"] = round(time.time() - t_all, 1)
    return out


# ----------------------------------------------------------- printing

def _f(x, d=4):
    return "   -  " if x is None else f"{x:.{d}f}"


def tables(out):
    lines = []
    p = out["protocol"]
    lines.append(f"Runtime: the model over {out['runtime']['matches']:,} matches {out['runtime']['model_every_league_s']}s; "
                 f"score matrices {out['runtime']['score_matrices_s']}s")
    lines.append("\nENGLAND'S POOL (model log loss, same matches; diff = E0-EC minus E0-E1)")
    for name, r in out["england"].items():
        for key in ("dev", "confirm"):
            x = r[key]
            lines.append(f"  {name:>9} {key:>7}: {x['matches']:>6}  E0-EC {_f(x['E0-EC'], 5)}  E0-E1 {_f(x['E0-E1'], 5)}"
                         f"  diff {x['diff']:+.5f} (se {x['diff_se']:.5f})")
    lines.append(f"  -> {'E0-EC' if p['england'] == 'all' else 'E0-E1'}")
    lines.append("\nWEIGHT OPTIONS (DEV log loss of the pick; raw = the no-vig price, same matches)")
    for opt, r in out["weight_options"].items():
        lines.append(f"  {opt:>10}: all {r['dev_all']['log_loss']:.5f} (raw {r['dev_all']['raw']:.5f}, "
                     f"{r['dev_all']['matches']})  " + "  ".join(
                         f"{g} {r[g]['log_loss']:.5f}" for g in GROUPS))
    lines.append(f"  -> {p['weights']}")
    lines.append("\nPER LEAGUE: log loss of the pick as shipped (recal = recalibrated before rule a) / raw market / "
                 "model / base rate, accuracy of the pick")
    lines.append("  code  in  pick  |    DEV n   pick  recal  market  model   base   acc |  CONF n   pick  recal  market"
                 "  model   base   acc | model-only")
    for code, r in out["leagues"].items():
        d = out["decisions"][code]
        row = f"  {code:<4}  {'Y' if d['in_app'] else 'n'}  {d['pick'][:5]:<5} |"
        for key in ("dev", "confirm"):
            x = r[key]
            row += (f" {x['matches']:>6} {_f(x.get('pick'))} {_f(x.get('pick_recalibrated'))} {_f(x.get('market'))} "
                    f"{_f(x.get('model'))} {_f(x.get('base'))} {_f(x.get('acc_pick'), 3)} |")
        row += f" {'yes' if d['model_only'] else 'no'}"
        lines.append(row)
    lines.append("\nIN THE APP (rule a): " + ", ".join(c for c, d in out["decisions"].items() if d["in_app"]))
    for c, d in out["decisions"].items():
        if not d["in_app"]:
            sc = out["leagues"][c]["season_check"]
            lines.append(f"  NOT: {c} - newest result {sc['last_result']} ({sc['days_since']} days before {out['protocol']['today']}), "
                         f"{sc['fixtures_ahead']} fixtures on file")
    lines.append("\nSCORE NUMBERS (rule c), log loss minus the base rate's, DEV / CONFIRM; * = shown")
    for code in out["leagues"]:
        for s, key in (("K", "market"), ("M", "model")):
            shown = out["decisions"][code]["numbers"][key]
            if shown is None:
                continue
            dv, cf = out["scores"][code]["dev"][s], out["scores"][code]["confirm"][s]
            if not dv.get("matches") or not cf.get("matches"):
                lines.append(f"  {code:<4} {s}: no matches")
                continue
            lines.append(f"  {code:<4} {s} {dv['matches']:>5}/{cf['matches']:>5}: " + "  ".join(
                f"{t}{'*' if t in shown else ' '} {dv[t]['log_loss'] - dv[t]['base']:+.4f}/{cf[t]['log_loss'] - cf[t]['base']:+.4f}"
                for t in NUMBERS))
    lines.append("\nBANDS (rule d), CONFIRM: hit rate (picks) - the pick | the model's own")
    for g, r in out["bands"].items():
        c = r["confirm"]
        lines.append(f"  {GROUP_NAMES[g]:>24}: " + "  ".join(
            f"{b} {_f(c['pick'][b]['hit_rate'], 3)} ({c['pick'][b]['picks']}) | {_f(c['model'][b]['hit_rate'], 3)}"
            for b, _ in ssc.BANDS))
    lines.append("\nPRE-MATCH vs CLOSING (European divisions 2019-2025, same matches)")
    for k in ("all", "top", "europe"):
        g = out["price_gap"][k]
        if not g["matches"]:
            continue
        lines.append(f"  {k:>6}: {g['matches']} matches  pre {g['pre']:.5f}  close {g['close']:.5f}  "
                     f"gap {g['gap']:+.5f} (se {g['gap_se']:.5f})  accuracy {g['acc_pre']:.3f} vs {g['acc_close']:.3f}")
    for r in out.get("point_in_time", []):
        lines.append(f"\nPOINT IN TIME {r['cut']}: {r}")
    return "\n".join(lines)


def main():
    out = run()
    print(tables(out))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
