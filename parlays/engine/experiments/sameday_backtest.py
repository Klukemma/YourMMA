"""Backtest of the same-day rules (sameday.py's docstring), point in time.

THE LIKELIEST TICKET. Every ET day from 2016-08-01 to the last day all three
leg tables cover (the UFC walk-forward cache ends 2026-08-08), the rule
builds at most one ticket from that day's legs (parlays/data/legs_*.csv.gz,
written by legs_history.py), and the market-only control builds its own.
Each is settled at the legs' prices on file, a void leg (NFL tie, UFC draw or
no contest) paying 1.0. Reported for ALL days, the FIRST years and the LAST 3
YEARS before the end, and split by how many sports a ticket mixes, by
weekday and by year. Nothing here was tuned: the rule was fixed before this
ran.

THE UNDERDOG TICKET (v1, under "dog") is measured exactly as the docstring's
MEASURED fixes it, on the same days and legs, its candidates from
parlays/data/legs_dogs_*.csv.gz (every event's market underdog; sameday.dog
applies the 2.00-4.00 band, build_dog the floor): the live salt's record -
the headline, whatever the other rows show; the 1,000 simulation salts'
mean with an interval over days, and their 5th/50th/95th percentiles,
which only say how much the draw alone moves the record over the same games
(not a confidence interval); and, report-only, the largest model gap first,
the likeliest underdog first, the market-only lottery, the likeliest ticket
paired with it and both tickets held. Every row beside its price-implied
return (the product of its legs' no-vig chances times its multiplier, less
1), because the prices expected a loss before any result was seen. The
splits are descriptive: none can move the band, the floor, the weight or
the sports in v1. Its periods are the registered dates (DOG_PERIODS), not
the moving end the likeliest ticket's use.

WHY THE FILLS ARE BUILT ONCE. A dog's fill does not depend on the salt - it
is the likeliest in-band combination of that day's other legs with that dog
- so each day's candidates are filled once, each by sameday.build_dog itself
on that one candidate; a salt then only orders the candidates, and its
ticket is the first one with a fill (build_dog's own loop). The live salt's
tickets are ALSO built by build_dog over all of a day's candidates, and both
the per-candidate path and the 1,000-salt matrix must reproduce them exactly
(the run stops otherwise), so the salts are measured by the function that
ships. The day's work runs in worker processes (fork), and the lottery
numbers are hashed once per salt, day and event.

Writes parlays/data/sameday_backtest.json, parlays/data/sameday_tickets.csv.gz
(the likeliest ticket and its control) and parlays/data/sameday_dog_tickets.
csv.gz (the underdog ticket at the live salt and its report-only rows).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "parlays" / "engine"))
import sameday as sdy  # noqa: E402

DATA = ROOT / "parlays" / "data"
NFL_LOG = ROOT / "nfl" / "data" / "prediction_history.json"
START = "2016-08-01"
SPORT_OF = {"ufc": "UFC", "nfl": "NFL", "soccer": "Soccer"}
RESULT = {"won": {"winner": "pick"}, "lost": {"winner": "other"}, "void": {"winner": None}}

# The underdog ticket's registered periods (sameday.py, MEASURED): fixed
# dates, so a later UFC refit that extends the tables does not move them.
DOG_PERIODS = {"all": ("2016-08-01", "2026-08-08"), "first": ("2016-08-01", "2023-08-08"),
               "last_3_years": ("2023-08-09", "2026-08-08")}
N_BOOT, SEED = 4000, 7
SPREAD_LABEL = "how much the draw alone moves it, same games"
SPLIT_NOTE = "descriptive: cannot move the band, the floor, the weight or the sports in v1"
PRICE_BUCKETS = (("2.00-2.49", 2.00, 2.50), ("2.50-2.99", 2.50, 3.00), ("3.00-3.49", 3.00, 3.50),
                 ("3.50-4.00", 3.50, 4.00 + 1e-9))
ALL_PRICE_BUCKETS = (("under 2.00", 0.0, 2.00),) + PRICE_BUCKETS + (
    ("4.01-6.00", 4.00 + 1e-9, 6.00 + 1e-9), ("over 6.00", 6.00 + 1e-9, math.inf))
WORKERS = 3


# ---------------------------------------------------------------- loading

def _num(x):
    return None if pd.isna(x) else float(x)


def _read(prefix):
    frames = []
    for name, sport in SPORT_OF.items():
        d = pd.read_csv(DATA / f"{prefix}_{name}.csv.gz")
        d["sport"] = sport
        frames.append(d)
    return frames


def load():
    frames = _read("legs")
    ends = {f["sport"].iloc[0]: f["day"].max() for f in frames}
    end = min(ends.values())
    legs = pd.concat(frames, ignore_index=True)
    legs = legs[(legs["day"] >= START) & (legs["day"] <= end)]
    return legs, end, ends


def load_dogs(end):
    """Every event's market underdog (legs_dogs_*), on the days the legs
    cover. The band and the floor are applied later, by sameday."""
    dogs = pd.concat(_read("legs_dogs"), ignore_index=True)
    return dogs[(dogs["day"] >= START) & (dogs["day"] <= end)]


def to_leg(r):
    return sdy.leg(r.sport, r.key, r.day, r.event, r.pick, _num(r.p), _num(r.decimal),
                   p_market=_num(r.p_market), p_model=_num(r.p_model), opponent=r.opponent,
                   result=RESULT[r.result], price_source=r.price_source)


def to_dog(r):
    """A legs_dogs row as sameday.dog takes it live: None outside 2.00-4.00
    or when the market does not make it the underdog."""
    return sdy.dog(r.sport, r.key, r.day, r.event, r.pick, _num(r.p), _num(r.decimal), _num(r.p_market),
                   r.lot_key, p_model=_num(r.p_model), opponent=r.opponent, result=RESULT[r.result],
                   price_source=r.price_source)


# ---------------------------------------------------------------- the day's work

_LEGS, _DOGS = {}, {}


def _day(day):
    """Everything one day gives, by sameday's own functions: the likeliest
    ticket and its control; every candidate's fill under the rule (p) and
    the market-only control (p_market); and the live salt's ticket, the
    report-only orders' and the market-only lottery's, built by build_dog
    over all of the day's candidates."""
    L, D = _LEGS.get(day, []), _DOGS.get(day, [])
    out = {"day": day, "same_day": sdy.build(L, "p"), "market_only": sdy.build(L, "p_market"),
           "n_band": len(D)}
    for obj in ("p", "p_market"):
        cands = sorted((d for d in D if d[obj] >= sdy.DOG_FLOOR), key=lambda d: d["key"])
        out[f"fills_{obj}"] = [(d, sdy.build_dog(L, [d], objective=obj)) for d in cands]
    out["dog"] = sdy.build_dog(L, D)
    out["dog_gap"] = sdy.build_dog(L, D, order="gap")
    out["dog_likeliest"] = sdy.build_dog(L, D, order="likeliest")
    out["dog_market"] = sdy.build_dog(L, D, objective="p_market")
    return out


def run(legs, dogs):
    """Every day's work, in day order; the number of days with any leg."""
    _LEGS.clear()
    _DOGS.clear()
    for day, g in legs.groupby("day"):
        _LEGS[day] = [l for l in (to_leg(r) for r in g.itertuples()) if l is not None]
    for day, g in dogs.groupby("day"):
        _DOGS[day] = [d for d in (to_dog(r) for r in g.itertuples()) if d is not None]
    days = sorted(legs["day"].unique())
    with mp.get_context("fork").Pool(WORKERS) as pool:
        out = pool.map(_day, days, chunksize=16)
    for o in out:
        for k in ("same_day", "market_only"):
            if o[k] is not None:
                o[k]["day"] = o["day"]
    return out, len(days)


# ---------------------------------------------------------------- records

def implied(t):
    """The ticket's price-implied return: the product of its legs' no-vig
    chances times its multiplier, less 1 - what it returns if the prices'
    chances are the truth."""
    return t["p_market"] * t["multiplier"] - 1.0


def profit(t):
    return sdy.settle(t)["profit"]


def record(tickets):
    """summarise() and the price-implied return beside it."""
    s = dict(sdy.summarise(tickets))
    if tickets:
        s["implied_roi"] = round(float(np.mean([implied(t) for t in tickets])), 4)
    return s


def boot(x, n_boot=N_BOOT, seed=SEED):
    """The day bootstrap summarise() uses: 95% interval of the mean."""
    x = np.asarray(x, float)
    if not len(x):
        return None, None
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def dog_leg(t):
    return next(l for l in t["legs"] if l.get("dog"))


def in_period(day, period):
    lo, hi = period
    return lo <= day <= hi


def quantiles(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    return {f"p{q}": round(float(np.percentile(x, q)), 2) for q in qs} if len(x) else None


def mix(tickets):
    c = Counter(dog_leg(t)["sport"] for t in tickets)
    n = sum(c.values())
    return {k: {"tickets": v, "share": round(v / n, 4)} for k, v in sorted(c.items())} if n else {}


def price_bucket(dec, buckets=PRICE_BUCKETS):
    for name, lo, hi in buckets:
        if lo <= dec < hi:
            return name
    return None


def signature(t):
    return tuple(sorted((l["key"], l["pick"]) for l in t["legs"]))


# ---------------------------------------------------------------- the 1,000 salts

_PAIRS = []


def _neglog(salts):
    """-ln(u) for every (salt, day, lot key): sameday.lottery_u, math.log
    - the numbers dog_order scores, to the bit."""
    out = np.empty((len(salts), len(_PAIRS)))
    for i, s in enumerate(salts):
        out[i] = [-math.log(sdy.lottery_u(s, d, k)) for d, k in _PAIRS]
    return out


def lottery_matrix(days_out, salts):
    """{objective: (ticket-day list, chosen candidate per salt and day,
    candidate table)} for the lottery under `salts` - each salt's ticket is
    the feasible candidate with the lowest -ln(u) / weight, ties by key (the
    candidates are kept in key order, and argmin takes the first)."""
    tables = {}
    pairs = {}
    for obj in ("p", "p_market"):
        cands, segs = [], []
        for o in days_out:
            feas = [(d, t) for d, t in o[f"fills_{obj}"] if t is not None]
            if not feas:
                continue
            start = len(cands)
            for d, t in feas:
                pairs.setdefault((o["day"], d["lot_key"]), len(pairs))
                cands.append({"day": o["day"], "pair": pairs[(o["day"], d["lot_key"])], "w": d[obj],
                              "ticket": t})
            segs.append((o["day"], start, len(cands)))
        tables[obj] = (cands, segs)
    _PAIRS[:] = sorted(pairs, key=pairs.get)
    chunks = [list(salts[i:i + 50]) for i in range(0, len(salts), 50)]
    with mp.get_context("fork").Pool(WORKERS) as pool:
        E = np.vstack(pool.map(_neglog, chunks))
    out = {}
    for obj, (cands, segs) in tables.items():
        col = np.array([c["pair"] for c in cands])
        w = np.array([c["w"] for c in cands])
        scores = E[:, col] / w[None, :]
        chosen = np.empty((len(salts), len(segs)), dtype=np.int64)
        for k, (_, a, b) in enumerate(segs):
            chosen[:, k] = a + scores[:, a:b].argmin(axis=1)
        out[obj] = ([s[0] for s in segs], chosen, cands)
    return out


def salt_stats(days, chosen, cands, period, live_profit=None, live_hit=None):
    """The mean over the salts with a day-bootstrap interval of each day's
    mean-over-salts profit (and won share), the salts' percentiles, and the
    live salt's rank among them. Row 0 of `chosen` is the live salt and is
    left out of every number here but the rank."""
    keep = np.array([in_period(d, period) for d in days])
    if not keep.any():
        return {"tickets_per_salt": 0}
    C = chosen[1:, keep]
    prof = np.array([profit(c["ticket"]) for c in cands])[C]
    won = np.array([sdy.settle(c["ticket"])["status"] == "won" for c in cands], float)[C]
    imp = np.array([implied(c["ticket"]) for c in cands])[C]
    mult = np.array([c["ticket"]["multiplier"] for c in cands])[C]
    legs = np.array([c["ticket"]["n_legs"] for c in cands], float)[C]
    q = np.array([c["ticket"]["p_market"] for c in cands])[C]
    roi_s, hit_s = prof.mean(axis=1), won.mean(axis=1)
    lo, hi = boot(prof.mean(axis=0))
    hlo, hhi = boot(won.mean(axis=0))
    pct = lambda x: {f"p{p}": round(float(np.percentile(x, p)), 4) for p in (5, 50, 95)}
    out = {
        "salts": int(C.shape[0]), "tickets_per_salt": int(C.shape[1]),
        "roi": round(float(roi_s.mean()), 4), "roi_low": lo, "roi_high": hi,
        "hit_rate": round(float(hit_s.mean()), 4), "hit_low": hlo, "hit_high": hhi,
        "interval": "day bootstrap (4,000 resamples, seed 7) of each day's mean-over-salts profit / won share",
        "implied_roi": round(float(imp.mean()), 4), "market_hit_rate": round(float(q.mean()), 4),
        "mean_multiplier": round(float(mult.mean()), 2), "mean_legs": round(float(legs.mean()), 2),
        "spread": {"label": SPREAD_LABEL, "roi": pct(roi_s), "hit_rate": pct(hit_s),
                   "implied_roi": pct(imp.mean(axis=1))},
    }
    if live_profit is not None:
        out["live_rank"] = {
            "label": "the live salt among the 1,000: how many salts did worse / the same",
            "roi_below": int((roi_s < live_profit - 1e-12).sum()),
            "roi_equal": int((np.abs(roi_s - live_profit) <= 1e-12).sum()),
            "hit_below": int((hit_s < live_hit - 1e-12).sum()),
            "hit_equal": int((np.abs(hit_s - live_hit) <= 1e-12).sum()), "of": int(C.shape[0])}
    return out


# ---------------------------------------------------------------- the underdog measurement

def dog_measure(days_out, likeliest_by_day):
    """The underdog ticket's results and splits by DOG_PERIODS, and its
    tickets for the csv. Stops if the per-candidate fills or the salt
    matrix fail to reproduce build_dog's live-salt tickets."""
    win = [o for o in days_out if in_period(o["day"], DOG_PERIODS["all"])]
    # 1. the per-candidate fills give build_dog's own ticket at the live salt
    for o in win:
        order = sdy.dog_order([d for d, _ in o["fills_p"]], sdy.DOG_SALT)
        fills = {d["key"]: t for d, t in o["fills_p"]}
        first = next((fills[d["key"]] for d in order if fills[d["key"]] is not None), None)
        live = o["dog"]
        same = (first is None and live is None) or (
            first is not None and live is not None and signature(first) == signature(live))
        if not same:
            raise SystemExit(f"{o['day']}: the precomputed fill does not reproduce build_dog")
    # 2. the 1,000 salts (row 0 the live salt, checked against build_dog too)
    salts = (sdy.DOG_SALT,) + sdy.DOG_SIM_SALTS
    M = lottery_matrix(win, salts)
    by_day = {o["day"]: o for o in win}
    for obj, key in (("p", "dog"), ("p_market", "dog_market")):
        days, chosen, cands = M[obj]
        for k, d in enumerate(days):
            if signature(cands[chosen[0, k]]["ticket"]) != signature(by_day[d][key]):
                raise SystemExit(f"{d}: the salt matrix does not reproduce build_dog ({obj})")
        if sum(by_day[d][key] is not None for d in by_day) != len(days):
            raise SystemExit(f"the salt matrix misses days with a ticket ({obj})")

    live = [o["dog"] for o in win if o["dog"]]
    rows = {"live": live, "gap": [o["dog_gap"] for o in win if o["dog_gap"]],
            "likeliest_dog": [o["dog_likeliest"] for o in win if o["dog_likeliest"]],
            "market_only": [o["dog_market"] for o in win if o["dog_market"]]}
    n_fill = {o["day"]: sum(t is not None for _, t in o["fills_p"]) for o in win}
    n_cand = {o["day"]: len(o["fills_p"]) for o in win}

    results = {"live": {}, "salts": {}, "gap": {"label": "largest model gap"},
               "likeliest_dog": {"label": "likeliest underdog"},
               "market_only": {"label": "market-only lottery (weights, floor and fill by the no-vig chance, "
                                         "same salt)", "live": {}, "salts": {}, "overlap": {},
                               "paired_with_lottery": {}},
               "paired": {}, "both": {}}
    results["gap"]["note"] = ("shipped minus market chance first; in soccer that gap is the recalibration "
                              "curve, not a model opinion - never called value")
    for name, period in DOG_PERIODS.items():
        P = lambda ts: [t for t in ts if in_period(t["day"], period)]
        lv = P(live)
        n_days = sum(in_period(o["day"], period) for o in win)
        s = record(lv)
        prices = [dog_leg(t)["decimal"] for t in lv]
        s.update({
            "days": n_days, "days_with_ticket": len(lv),
            "days_one_candidate": sum(n_cand[t["day"]] == 1 for t in lv),
            "days_one_fillable": sum(n_fill[t["day"]] == 1 for t in lv),
            "dog_price": quantiles(prices), "dog_sports": mix(lv),
            "dog_chance": round(float(np.mean([dog_leg(t)["p"] for t in lv])), 4) if lv else None,
            "dog_market_chance": round(float(np.mean([dog_leg(t)["p_market"] for t in lv])), 4) if lv else None,
            "fallback": {str(k): v for k, v in sorted(Counter(t["dog_rank"] for t in lv).items())},
            "legs": {str(k): v for k, v in sorted(Counter(t["n_legs"] for t in lv).items())},
        })
        results["live"][name] = s
        live_roi = float(np.mean([profit(t) for t in lv])) if lv else None
        live_hit = float(np.mean([sdy.settle(t)["status"] == "won" for t in lv])) if lv else None
        days, chosen, cands = M["p"]
        results["salts"][name] = salt_stats(days, chosen, cands, period, live_roi, live_hit)
        for key in ("gap", "likeliest_dog"):
            ts = P(rows[key])
            results[key][name] = dict(record(ts), dog_sports=mix(ts),
                                      dog_price=quantiles([dog_leg(t)["decimal"] for t in ts]))
        mo = P(rows["market_only"])
        mo_roi = float(np.mean([profit(t) for t in mo])) if mo else None
        mo_hit = float(np.mean([sdy.settle(t)["status"] == "won" for t in mo])) if mo else None
        results["market_only"]["live"][name] = dict(record(mo), dog_sports=mix(mo))
        mdays, mchosen, mcands = M["p_market"]
        results["market_only"]["salts"][name] = salt_stats(mdays, mchosen, mcands, period, mo_roi, mo_hit)
        results["market_only"]["overlap"][name] = overlap(M, period)
        results["market_only"]["paired_with_lottery"][name] = paired_salts(M, period)
        results["paired"][name], results["both"][name] = paired(lv, likeliest_by_day, M, period)

    splits = {"note": SPLIT_NOTE}
    lv = rows["live"]
    for key, f in (("by_sport", lambda t: dog_leg(t)["sport"]),
                   ("by_price", lambda t: price_bucket(dog_leg(t)["decimal"])),
                   ("by_year", lambda t: t["day"][:4]),
                   ("by_legs", lambda t: f"{t['n_legs']} legs")):
        by = defaultdict(list)
        for t in lv:
            by[f(t)].append(t)
        splits[key] = {"note": SPLIT_NOTE, **{k: record(v) for k, v in sorted(by.items())}}
    return results, splits, rows, M


def overlap(M, period):
    """How often the market-only lottery picks the same dog and the same
    ticket as the lottery, on the days both have one: at the live salt and
    the mean over the 1,000."""
    pd_, pc, pcand = M["p"]
    md, mc, mcand = M["p_market"]
    ip = {d: k for k, d in enumerate(pd_) if in_period(d, period)}
    im = {d: k for k, d in enumerate(md) if in_period(d, period)}
    common = sorted(set(ip) & set(im))
    if not common:
        return {"days": 0}
    a, b = np.array([ip[d] for d in common]), np.array([im[d] for d in common])
    sig_p = np.array([hash(signature(c["ticket"])) for c in pcand])
    sig_m = np.array([hash(signature(c["ticket"])) for c in mcand])
    key_p = np.array([hash(c["ticket"]["dog_key"]) for c in pcand])
    key_m = np.array([hash(c["ticket"]["dog_key"]) for c in mcand])
    same_dog = key_p[pc[:, a]] == key_m[mc[:, b]]
    same_t = sig_p[pc[:, a]] == sig_m[mc[:, b]]
    return {"days": len(common),
            "same_dog": round(float(same_dog[0].mean()), 4), "same_ticket": round(float(same_t[0].mean()), 4),
            "salts_same_dog": round(float(same_dog[1:].mean()), 4),
            "salts_same_ticket": round(float(same_t[1:].mean()), 4)}


def paired_salts(M, period):
    """Descriptive: the lottery minus the market-only lottery, each day's
    mean over the 1,000 salts, on the days both have a ticket."""
    pd_, pc, pcand = M["p"]
    md, mc, mcand = M["p_market"]
    ip = {d: k for k, d in enumerate(pd_) if in_period(d, period)}
    im = {d: k for k, d in enumerate(md) if in_period(d, period)}
    common = sorted(set(ip) & set(im))
    if not common:
        return {"days": 0}
    a, b = np.array([ip[d] for d in common]), np.array([im[d] for d in common])
    pp_ = np.array([profit(c["ticket"]) for c in pcand])[pc[1:, a]].mean(axis=0)
    pm = np.array([profit(c["ticket"]) for c in mcand])[mc[1:, b]].mean(axis=0)
    lo, hi = boot(pp_ - pm)
    return {"days": len(common), "roi_diff": round(float((pp_ - pm).mean()), 4), "roi_diff_low": lo,
            "roi_diff_high": hi, "note": "lottery minus market-only, mean over the 1,000 salts; descriptive"}


def paired(live, likeliest_by_day, M, period):
    """The likeliest ticket beside the underdog ticket on the days both
    exist: the difference (dog minus likeliest) in hit rate and ROI with a
    paired day bootstrap, at the live salt and as each day's mean over the
    salts; and both tickets held, 2 units a day."""
    pairs = [(t, likeliest_by_day[t["day"]]) for t in live
             if in_period(t["day"], period) and t["day"] in likeliest_by_day]
    if not pairs:
        return {"days": 0}, {"days": 0}
    pd_ = np.array([profit(d) for d, _ in pairs])
    pl = np.array([profit(l) for _, l in pairs])
    wd = np.array([sdy.settle(d)["status"] == "won" for d, _ in pairs], float)
    wl = np.array([sdy.settle(l)["status"] == "won" for _, l in pairs], float)
    lo, hi = boot(pd_ - pl)
    hlo, hhi = boot(wd - wl)
    days, chosen, cands = M["p"]
    k = {d: i for i, d in enumerate(days)}
    salt_mean = np.array([profit(c["ticket"]) for c in cands])[chosen[1:, [k[d["day"]] for d, _ in pairs]]].mean(axis=0)
    slo, shi = boot(salt_mean - pl)
    p = {"days": len(pairs),
         "dog": record([d for d, _ in pairs]), "likeliest": record([l for _, l in pairs]),
         "hit_rate_diff": round(float((wd - wl).mean()), 4), "hit_diff_low": hlo, "hit_diff_high": hhi,
         "roi_diff": round(float((pd_ - pl).mean()), 4), "roi_diff_low": lo, "roi_diff_high": hi,
         "salts_roi_diff": round(float((salt_mean - pl).mean()), 4), "salts_roi_diff_low": slo,
         "salts_roi_diff_high": shi,
         "note": "underdog ticket minus likeliest ticket, paired by day (bootstrap 4,000, seed 7)"}
    tot = pd_ + pl
    blo, bhi = boot(tot / 2.0)
    opp = sum(sdy.opposes(d, l) for d, l in pairs)
    b = {"days": len(pairs), "units": 2 * len(pairs), "profit": round(float(tot.sum()), 2),
         "roi": round(float(tot.sum() / (2 * len(pairs))), 4), "roi_low": blo, "roi_high": bhi,
         "both_won": int((wd * wl).sum()), "one_won": int(((wd + wl) == 1).sum()),
         "neither_won": int(((wd + wl) == 0).sum()), "cannot_both_win": int(opp),
         "note": "both tickets held, 1 unit each a day; cannot_both_win counts the days the dog is the other "
                 "side of one of the likeliest ticket's legs (sameday.opposes)"}
    return p, b


def dogs_by_price(dogs):
    """Descriptive: every market underdog as a single bet at its price,
    in and out of the 2.00-4.00 band - what the band leaves out."""
    lo, hi = DOG_PERIODS["all"]
    d = dogs[(dogs["day"] >= lo) & (dogs["day"] <= hi)]
    out = {"note": SPLIT_NOTE + "; single bets on every underdog, not tickets"}
    for name, a, b in ALL_PRICE_BUCKETS:
        g = d[(d["decimal"] >= a) & (d["decimal"] < b)]
        ts = [{"legs": [l], "multiplier": l["decimal"], "p": l["p"], "p_market": l["p_market"], "n_legs": 1}
              for l in ({"decimal": float(r.decimal), "p": float(r.p), "p_market": float(r.p_market),
                         "side": "pick", "market": "moneyline", "result": RESULT[r.result]}
                        for r in g.itertuples())]
        s = record(ts)
        s["by_sport"] = {k: int(v) for k, v in g["sport"].value_counts().sort_index().items()}
        out[name] = s
    return out


# ---------------------------------------------------------------- caveat numbers

def _overround(legs, sport):
    """Each event's overround from its pick-side row: 1 / (decimal x no-vig
    chance) - 1 (the no-vig chance is the pick's share of the two inverse
    prices)."""
    g = legs[legs["sport"] == sport]
    return 1.0 / (g["decimal"] * g["p_market"]) - 1.0


def nfl_live_overround():
    """The median overround of the consensus moneylines the NFL app logged
    live, and how many games: (None, 0) when there is no log."""
    try:
        picks = json.loads(NFL_LOG.read_text()).get("picks") or {}
    except (OSError, ValueError):
        return None, 0
    picks = picks.values() if isinstance(picks, dict) else picks
    dec = lambda a: 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)
    o = []
    for p in picks:
        m = p.get("market") or {}
        h, a = m.get("home_ml"), m.get("away_ml")
        if h and a:
            o.append(1 / dec(float(h)) + 1 / dec(float(a)) - 1)
    return (float(np.median(o)) if o else None), len(o)


def equal_margin(legs, dogs, sport):
    """In-band dogs of a two-sided market: the mean return at the price if
    the margin is taken equally from each side (Shin's reading, for two
    sides) against the proportional reading these files use, and the mean
    chance each gives."""
    d = dogs[(dogs["sport"] == sport) & (dogs["decimal"] >= sdy.DOG_PRICE[0]) & (dogs["decimal"] <= sdy.DOG_PRICE[1])
             & (dogs["p_market"] < 0.5)]
    over = 1.0 / (d["decimal"] * d["p_market"]) - 1.0
    q_eq = 1.0 / d["decimal"] - over / 2.0
    return {"dogs": int(len(d)), "chance_proportional": float(d["p_market"].mean()),
            "chance_equal_margin": float(q_eq.mean()),
            "return_proportional": float((d["decimal"] * d["p_market"] - 1).mean()),
            "return_equal_margin": float((d["decimal"] * q_eq - 1).mean())}


def _pct(x, digits=0):
    return f"{'+' if x >= 0 else '-'}{abs(x) * 100:.{digits}f}%"


def caveats(results, extra, ends, end, legs):
    r3 = results["same_day"]["last_3_years"]
    mo = results["market_only"]["last_3_years"]
    close = float(np.median(_overround(legs, "NFL")))
    live, n_live = nfl_live_overround()
    live_txt = (f"{live:.1%} on the {n_live} games the NFL app logged live" if live is not None
                else "a live figure that is missing: the NFL app has logged no game with a moneyline yet")
    out = [
        f"The market is hard to beat. Over the last 3 years the rule's tickets won {r3.get('wins', 0)} of "
        f"{r3.get('tickets', 0)} ({_pct(r3.get('roi', 0))} a ticket, {_pct(r3.get('roi_low', 0))} to "
        f"{_pct(r3.get('roi_high', 0))}); the market-only control's won {mo.get('wins', 0)} of "
        f"{mo.get('tickets', 0)} ({_pct(mo.get('roi', 0))}). An interval that spans zero is no evidence of an edge.",
        f"Measured to {end}, the last day all three sports' leg tables cover (UFC {ends['UFC']}, "
        f"NFL {ends['NFL']}, soccer {ends['Soccer']}); the UFC walk-forward needs a runner refit to go further.",
        "Prices: UFC at each card's closing (or, where only that exists, opening) line, NFL at the closing "
        "moneyline, soccer at the pre-match average posted on Friday or Tuesday afternoon. A ticket built "
        "earlier in the day gets different prices; NFL closing lines know the day's news.",
        f"Payouts: these files' margins are lower than a single sportsbook's live prices (NFL closing "
        f"{close:.1%} a game, the median here, against {live_txt}), so a live ticket pays a little less "
        f"than the record assumes.",
        "Soccer here is the top five leagues only, which play no matches on international-break weekends "
        "and in June-July; on those days a ticket comes from UFC and NFL alone, or there is none.",
        "UFC days are the event date; a card held outside the Americas can fall on another ET date.",
        "The likeliest ticket in 3-5x is mostly favourites, and every leg is the same model's opinion; "
        "the legs are different events, so their outcomes are close to independent.",
    ]
    return out


def dog_caveats(dog, legs, dogs, rows):
    lv = dog["results"]["live"]["all"]
    sl = dog["results"]["salts"]["all"]
    fs, l3 = dog["results"]["salts"]["first"], dog["results"]["salts"]["last_3_years"]
    em = {s: equal_margin(legs, dogs, s) for s in ("UFC", "NFL")}
    live = [t for t in rows["live"] if in_period(t["day"], DOG_PERIODS["all"])]
    ufc = [dog_leg(t) for t in live if dog_leg(t)["sport"] == "UFC"]
    ufc25 = [l for t, l in ((t, dog_leg(t)) for t in live) if l["sport"] == "UFC" and t["day"][:4] == "2025"]
    nfl = sum(dog_leg(t)["sport"] == "NFL" for t in live)
    opened = sum(l["price_source"] == "open" for l in ufc)
    p = dog["results"]["paired"]["all"]
    out = [
        f"The market is hard to beat, and a draw cannot create an edge: every 3-5x ticket pays the bookmaker's "
        f"margin on every leg. This ticket is for variety and a bigger payout, never the better bet. At the "
        f"live salt it won {lv['wins']} of {lv['tickets']:,} ({_pct(lv['roi'])} a ticket, {_pct(lv['roi_low'])} "
        f"to {_pct(lv['roi_high'])}); an interval that spans zero is no evidence of an edge.",
        f"Expected before any result was seen (sameday.py): by the prices it lands about 1 time in 4 (a market "
        f"chance of about 0.25 at about 3.8x), the likeliest about 2 in 7 at about 3.1x, price-implied about -8% "
        f"a ticket, and somewhat worse than its prices imply. The prices on file give it a market chance of "
        f"{lv['market_hit_rate']:.3f} at {lv['mean_multiplier']:.2f}x, price-implied {_pct(lv['implied_roi'], 1)}.",
        "Prior exposure: the rule's own results were not seen before it was fixed, but underdog results over the "
        "same years had been - the UFC profiles' risker and risk-lover tickets and the NFL and soccer "
        "edge-betting checks. Its record is not an out-of-sample test of underdogs.",
        "Bookmakers load more margin on underdogs (about 5% on a dog leg against 3% on a favourite, as written "
        "before the results), so a dog ticket pays the heavier margin once on its dog and a lighter one on each "
        "favourite that fills it; the fill adds payout, not safety or value.",
        f"The market chances here take the margin off in proportion, which reads a dog's chance generously. "
        f"Taking it equally off each side (Shin's reading for two sides), the {em['UFC']['dogs']:,} UFC dogs "
        f"priced 2.00-4.00 average a chance of {em['UFC']['chance_equal_margin']:.3f} against "
        f"{em['UFC']['chance_proportional']:.3f} and return {_pct(em['UFC']['return_equal_margin'], 1)} at their "
        f"price against {_pct(em['UFC']['return_proportional'], 1)}; the {em['NFL']['dogs']:,} NFL dogs "
        f"{em['NFL']['chance_equal_margin']:.3f} against {em['NFL']['chance_proportional']:.3f}, "
        f"{_pct(em['NFL']['return_equal_margin'], 1)} against {_pct(em['NFL']['return_proportional'], 1)}. "
        f"Soccer's split is missing: these files keep the two teams' prices, not the draw's.",
        f"NFL underdogs are chosen and priced at the closing moneyline ({nfl} of the live salt's "
        f"{lv['tickets']:,} dogs): the close knows the day's news, and whether a dog is priced 2.00-4.00 is "
        f"decided at it; live, both are decided at the consensus price when the ticket is built.",
        f"UFC cards of 2025 are priced at their opening lines (no closing line is on file for them): "
        f"{len(ufc25)} of the live salt's UFC dogs that year, {opened} of its {len(ufc)} UFC dogs in all priced "
        f"at an opening line.",
        f"The live salt drew well: over all days its record is above {sl['live_rank']['roi_below']:,} of the "
        f"1,000 simulation salts' on the same games (the first years {fs['live_rank']['roi_below']:,}, the last 3 "
        f"years {l3['live_rank']['roi_below']:,}), at the same prices (price-implied {_pct(lv['implied_roi'], 1)} "
        f"against {_pct(sl['implied_roi'], 1)}). Its past draw is no forecast: the salts averaged "
        f"{_pct(sl['roi'], 1)} a ticket ({_pct(sl['roi_low'], 1)} to {_pct(sl['roi_high'], 1)}), and the prices "
        f"imply {_pct(lv['implied_roi'], 1)}.",
        f"The 5th-95th percentile across the 1,000 simulation salts ({_pct(sl['spread']['roi']['p5'], 1)} to "
        f"{_pct(sl['spread']['roi']['p95'], 1)} a ticket over all days) is how much the draw alone moves the "
        f"record over the same games; it is not a confidence interval. The interval over days is "
        f"({_pct(lv['roi_low'], 1)} to {_pct(lv['roi_high'], 1)} at the live salt).",
        "Live, a UFC leg is priced from the card the UFC app carries (the best price across books, or a "
        "median of books where the card has one); this history prices each fight at one line per card. "
        "A live UFC dog's price, and whether it falls in 2.00-4.00, can differ from what this record assumes.",
        f"Same days and events as the likeliest ticket: on the {p['days']:,} days both exist, the two cannot "
        f"both win on {dog['results']['both']['all']['cannot_both_win']} (the dog is the other side of one of "
        f"the likeliest ticket's legs).",
    ]
    return out


# ---------------------------------------------------------------- main

def split(tickets, key):
    by = defaultdict(list)
    for t in tickets:
        by[key(t)].append(t)
    return {k: sdy.summarise(v) for k, v in sorted(by.items())}


def multipliers(tickets):
    m = np.array([t["multiplier"] for t in tickets]) if tickets else np.array([])
    if not len(m):
        return None
    return {"median": round(float(np.median(m)), 2), "min": round(float(m.min()), 2),
            "max": round(float(m.max()), 2)}


def main():
    legs, end, ends = load()
    dogs = load_dogs(end)
    last3 = (date.fromisoformat(end) - timedelta(days=3 * 365)).isoformat()
    first_end = (date.fromisoformat(last3) - timedelta(days=1)).isoformat()
    periods = {"all": [START, end], "first": [START, first_end], "last_3_years": [last3, end]}
    days_out, days = run(legs, dogs)
    built = {name: [o[name] for o in days_out if o[name] is not None] for name in ("same_day", "market_only")}
    results, extra = {}, {}
    for name, tickets in built.items():
        recent = [t for t in tickets if t["day"] >= last3]
        early = [t for t in tickets if t["day"] <= first_end]
        results[name] = {"all": sdy.summarise(tickets), "last_3_years": sdy.summarise(recent),
                         "first": sdy.summarise(early)}
        extra[name] = {
            "by_sports": split(tickets, lambda t: "1 sport" if len(t["sports"]) == 1 else "2+ sports"),
            "by_mix": split(tickets, lambda t: "+".join(t["sports"])),
            "by_weekday": split(tickets, lambda t: date.fromisoformat(t["day"]).strftime("%a")),
            "by_year": split(tickets, lambda t: t["day"][:4]),
            "multipliers": {"all": multipliers(tickets), "last_3_years": multipliers(recent)},
            "legs_used": dict(pd.Series([l["sport"] for t in tickets for l in t["legs"]]).value_counts()),
        }
    likeliest_by_day = {t["day"]: t for t in built["same_day"]}
    dres, dsplits, drows, _ = dog_measure(days_out, likeliest_by_day)
    dsplits["dogs_by_price"] = dogs_by_price(dogs)
    lv, sl = dres["live"]["all"], dres["salts"]["all"]
    dog = {
        "version": sdy.DOG_VERSION, "salt": sdy.DOG_SALT,
        "rule": {"price": list(sdy.DOG_PRICE), "floor": sdy.DOG_FLOOR, "band": list(sdy.BAND),
                 "max_legs": sdy.MAX_LEGS, "pool": sdy.POOL, "weight": "shipped chance",
                 "sim_salts": len(sdy.DOG_SIM_SALTS)},
        "periods": {k: list(v) for k, v in DOG_PERIODS.items()},
        "headline": (f"At the live salt, {DOG_PERIODS['all'][0]} to {DOG_PERIODS['all'][1]}: {lv['tickets']:,} "
                     f"underdog tickets, {lv['wins']} won ({lv['hit_rate']:.1%}; the prices said "
                     f"{lv['market_hit_rate']:.1%}), {_pct(lv['roi'], 1)} a ticket ({_pct(lv['roi_low'], 1)} "
                     f"to {_pct(lv['roi_high'], 1)}), price-implied {_pct(lv['implied_roi'], 1)}. The same games "
                     f"drawn with the 1,000 simulation salts returned {_pct(sl['roi'], 1)} a ticket on average "
                     f"({_pct(sl['roi_low'], 1)} to {_pct(sl['roi_high'], 1)}), price-implied "
                     f"{_pct(sl['implied_roi'], 1)}; the live salt's record is above "
                     f"{sl['live_rank']['roi_below']:,} of the {sl['live_rank']['of']:,} - with the same prices, "
                     f"so the gap is which dogs happened to win."),
        "results": dres, "splits": dsplits,
    }
    if DOG_PERIODS["all"][1] > end:
        dog["missing"] = f"the leg tables end {end}, before the registered end {DOG_PERIODS['all'][1]}"
    dog["caveats"] = dog_caveats(dog, legs, dogs, drows)
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": {"day_tz": sdy.DAY_TZ, "band": list(sdy.BAND), "max_legs": sdy.MAX_LEGS,
                 "pool": sdy.POOL, "min_p": sdy.MIN_P},
        "periods": periods,
        "days": days, "leg_tables_end": ends,
        "results": results, "splits": extra,
        "caveats": caveats(results, extra, ends, end, legs),
        "dog": dog,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "sameday_backtest.json").write_text(json.dumps(payload, indent=1, default=_json, allow_nan=False))
    gz = {"method": "gzip", "mtime": 0}
    rows = [{"rule": name, "day": t["day"], "n_legs": t["n_legs"], "sports": "+".join(t["sports"]),
             "multiplier": round(t["multiplier"], 4), "p": round(t["p"], 6),
             "p_market": None if t["p_market"] is None else round(t["p_market"], 6),
             "status": sdy.settle(t)["status"], "profit": sdy.settle(t)["profit"],
             "legs": "|".join(f"{l['sport']}:{l['pick']}@{l['decimal']:.3f}" for l in t["legs"])}
            for name, ts in built.items() for t in ts]
    pd.DataFrame(rows).to_csv(DATA / "sameday_tickets.csv.gz", index=False, compression=gz)
    drows_out = []
    for name, rule in (("live", "dog"), ("gap", "dog_gap"), ("likeliest_dog", "dog_likeliest"),
                       ("market_only", "dog_market_only")):
        for t in drows[name]:
            d = dog_leg(t)
            st = sdy.settle(t)
            drows_out.append({
                "rule": rule, "day": t["day"], "dog_key": d["key"], "dog_sport": d["sport"], "dog": d["pick"],
                "dog_decimal": round(d["decimal"], 6), "dog_p": round(d["p"], 6),
                "dog_p_market": round(d["p_market"], 6), "dog_rank": t["dog_rank"], "candidates": t["candidates"],
                "n_legs": t["n_legs"], "sports": "+".join(t["sports"]), "multiplier": round(t["multiplier"], 4),
                "p": round(t["p"], 6), "p_market": round(t["p_market"], 6), "status": st["status"],
                "profit": st["profit"], "keys": "|".join(l["key"] for l in t["legs"]),
                "legs": "|".join(f"{l['sport']}:{l['pick']}@{l['decimal']:.3f}" for l in t["legs"])})
    pd.DataFrame(drows_out).to_csv(DATA / "sameday_dog_tickets.csv.gz", index=False, compression=gz)
    table(payload)


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


def _line(label, s):
    if not s.get("tickets"):
        return f"  {label:<34}       0"
    return (f"  {label:<34}{s['tickets']:>6}{s['wins']:>6}{s['hit_rate']:>7.1%}{s['expected_hit_rate']:>7.1%}"
            f"{(s['market_hit_rate'] or 0):>7.1%}{s['mean_multiplier']:>6.2f}x{s['mean_legs']:>6.2f}"
            f"{s['roi']:>+8.3f}  [{s['roi_low']:+.3f}, {s['roi_high']:+.3f}]"
            + (f"  implied {s['implied_roi']:+.3f}" if "implied_roi" in s else ""))


def table(payload):
    head = (f"  {'rule / period':<34}{'tick':>6}{'wins':>6}{'hit':>7}{'model':>7}{'market':>7}{'mult':>7}"
            f"{'legs':>6}{'ROI':>8}  [95% interval]")
    print(head)
    for name in ("same_day", "market_only"):
        for period in ("all", "first", "last_3_years"):
            print(_line(f"{name} {period}", payload["results"][name][period]))
    for k in ("by_sports", "by_mix", "by_weekday", "by_year"):
        print(f"\n  same_day {k}:")
        for g, s in payload["splits"]["same_day"][k].items():
            if s.get("tickets"):
                print(f"    {g:<20}{s['tickets']:>6}{s['wins']:>6}{s['hit_rate']:>7.1%}{s['expected_hit_rate']:>7.1%}"
                      f"{s['roi']:>+8.3f}  [{s['roi_low']:+.3f}, {s['roi_high']:+.3f}]")
    print("\n  multipliers:", payload["splits"]["same_day"]["multipliers"])
    print("  legs used:", payload["splits"]["same_day"]["legs_used"])
    print(f"  days {payload['days']}, periods {payload['periods']}")
    dog = payload["dog"]
    r = dog["results"]
    print(f"\n  UNDERDOG TICKET v{dog['version']} (salt {dog['salt']})\n{head}")
    for period in DOG_PERIODS:
        print(_line(f"live salt {period}", r["live"][period]))
        s = r["salts"][period]
        if s.get("tickets_per_salt"):
            sp = s["spread"]
            print(f"  {'1,000 salts ' + period:<34}{s['tickets_per_salt']:>6}{'':>6}{s['hit_rate']:>7.1%}"
                  f" [{s['hit_low']:.3f}, {s['hit_high']:.3f}]  ROI {s['roi']:+.3f} [{s['roi_low']:+.3f}, "
                  f"{s['roi_high']:+.3f}]  implied {s['implied_roi']:+.3f}  draw alone: ROI p5/p50/p95 "
                  f"{sp['roi']['p5']:+.3f}/{sp['roi']['p50']:+.3f}/{sp['roi']['p95']:+.3f}, hit "
                  f"{sp['hit_rate']['p5']:.3f}/{sp['hit_rate']['p50']:.3f}/{sp['hit_rate']['p95']:.3f}; live "
                  f"above {s['live_rank']['roi_below']} of {s['live_rank']['of']} (ROI)")
        print(_line(f"largest model gap {period}", r["gap"][period]))
        print(_line(f"likeliest underdog {period}", r["likeliest_dog"][period]))
        print(_line(f"market-only lottery {period}", r["market_only"]["live"][period]))
        ms = r["market_only"]["salts"][period]
        if ms.get("tickets_per_salt"):
            print(f"  {'  market-only 1,000 salts':<34}{ms['tickets_per_salt']:>6}{'':>6}{ms['hit_rate']:>7.1%}"
                  f"  ROI {ms['roi']:+.3f} [{ms['roi_low']:+.3f}, {ms['roi_high']:+.3f}]  overlap "
                  f"{r['market_only']['overlap'][period]}")
        p, b = r["paired"][period], r["both"][period]
        if p.get("days"):
            print(f"  paired with the likeliest ({p['days']} days): hit diff {p['hit_rate_diff']:+.3f} "
                  f"[{p['hit_diff_low']:+.3f}, {p['hit_diff_high']:+.3f}], ROI diff {p['roi_diff']:+.3f} "
                  f"[{p['roi_diff_low']:+.3f}, {p['roi_diff_high']:+.3f}]; salts mean {p['salts_roi_diff']:+.3f}")
            print(f"  both held: {b['units']} units, ROI {b['roi']:+.3f} [{b['roi_low']:+.3f}, {b['roi_high']:+.3f}],"
                  f" both won {b['both_won']}, one {b['one_won']}, neither {b['neither_won']}, cannot both win "
                  f"{b['cannot_both_win']}")
        print()
    for k in ("by_sport", "by_price", "by_year", "by_legs"):
        print(f"  dog {k} ({SPLIT_NOTE}):")
        for g, s in dog["splits"][k].items():
            if g != "note":
                print("  " + _line(g, s))
    print(f"\n  every underdog as a single bet, by price ({SPLIT_NOTE}):")
    for g, s in dog["splits"]["dogs_by_price"].items():
        if g != "note":
            print("  " + _line(g, s))
    print("\n  " + dog["headline"])


if __name__ == "__main__":
    main()
