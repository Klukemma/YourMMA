"""Backtest of the same-day rules (sameday.py's docstring), point in time.

THE LIKELIEST TICKET. Every ET day from 2016-08-01 to the last day all three
leg tables cover (the UFC walk-forward cache ends 2026-08-08), the rule
builds at most one ticket from that day's legs (parlays/data/legs_*.csv.gz,
written by legs_history.py), and the market-only control builds its own.
Each is settled at the legs' prices on file, a void leg (an NFL tie) paying
1.0. UFC draws and no contests are not void legs here but absent: the UFC
history (engine/experiments/parlay_legs.csv.gz) keeps decided fights only,
so such a fight is never a leg or an underdog, and the day's ticket comes
from its other events - an outcome the rule could not know live, where the
fight would stay in and its leg void. Both blocks carry a caveat that
counts them from the UFC archive (ufc_no_winner). Reported for ALL days, the FIRST years and the LAST 3
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
paired with it and both tickets held (each also as its mean over the 1,000
salts, since the live salt is one draw). Every row beside its price-implied
return (the product of its legs' no-vig chances times its multiplier, less
1), because the prices expected a loss before any result was seen. Every
word in the caveats that depends on a result (how the live salt drew,
whether live payouts are lower, whether a reading is generous) is chosen
from the numbers beside it, never typed. The
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
import re
import subprocess
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
    salts; and both tickets held, 2 units a day - each beside its
    price-implied return and its mean over the 1,000 salts.

    WHY BOTH THE SALTS AND THE PRICES BESIDE THE LIVE SALT: one salt is
    one draw, and its luck is not the rule's. When these were added
    (2026-11-13) the live salt ranked above 992 of the 1,000 and the page
    showed both held at +4% a unit alone - about -2% over the salts and
    -10% by the prices. A row with only the live salt's number lets the
    lottery's luck stand for the rule, and sameday.py's MEASURED puts
    every row beside its price-implied return.
    The salts' rows use every simulation salt's ticket on the same days:
    each salt has a ticket on exactly the live salt's days (dog_measure
    checks it), so the pairing is the same."""
    pairs = [(t, likeliest_by_day[t["day"]]) for t in live
             if in_period(t["day"], period) and t["day"] in likeliest_by_day]
    if not pairs:
        return {"days": 0}, {"days": 0}
    pd_ = np.array([profit(d) for d, _ in pairs])
    pl = np.array([profit(l) for _, l in pairs])
    wd = np.array([sdy.settle(d)["status"] == "won" for d, _ in pairs], float)
    wl = np.array([sdy.settle(l)["status"] == "won" for _, l in pairs], float)
    id_ = np.array([implied(d) for d, _ in pairs])
    il = np.array([implied(l) for _, l in pairs])
    lo, hi = boot(pd_ - pl)
    hlo, hhi = boot(wd - wl)
    days, chosen, cands = M["p"]
    k = {d: i for i, d in enumerate(days)}
    C = chosen[1:, [k[d["day"]] for d, _ in pairs]]
    salt_mean = np.array([profit(c["ticket"]) for c in cands])[C].mean(axis=0)
    salt_imp = np.array([implied(c["ticket"]) for c in cands])[C].mean(axis=0)
    slo, shi = boot(salt_mean - pl)
    p = {"days": len(pairs),
         "dog": record([d for d, _ in pairs]), "likeliest": record([l for _, l in pairs]),
         "hit_rate_diff": round(float((wd - wl).mean()), 4), "hit_diff_low": hlo, "hit_diff_high": hhi,
         "roi_diff": round(float((pd_ - pl).mean()), 4), "roi_diff_low": lo, "roi_diff_high": hi,
         "implied_roi_diff": round(float((id_ - il).mean()), 4),
         "salts_roi_diff": round(float((salt_mean - pl).mean()), 4), "salts_roi_diff_low": slo,
         "salts_roi_diff_high": shi, "salts_implied_roi_diff": round(float((salt_imp - il).mean()), 4),
         "note": "underdog ticket minus likeliest ticket, paired by day (bootstrap 4,000, seed 7): roi_diff "
                 "and hit_rate_diff at the live salt; implied_roi_diff the same two tickets by their prices; "
                 "salts_roi_diff each day's mean over the 1,000 simulation salts of the dog's profit, less the "
                 "likeliest's, day-bootstrapped; salts_implied_roi_diff the salts' tickets by their prices"}
    tot = pd_ + pl
    blo, bhi = boot(tot / 2.0)
    both_salts = (salt_mean + pl) / 2.0
    bslo, bshi = boot(both_salts)
    opp = sum(sdy.opposes(d, l) for d, l in pairs)
    b = {"days": len(pairs), "units": 2 * len(pairs), "profit": round(float(tot.sum()), 2),
         "roi": round(float(tot.sum() / (2 * len(pairs))), 4), "roi_low": blo, "roi_high": bhi,
         "implied_roi": round(float(((id_ + il) / 2.0).mean()), 4),
         "salts_roi": round(float(both_salts.mean()), 4), "salts_roi_low": bslo, "salts_roi_high": bshi,
         "salts_implied_roi": round(float(((salt_imp + il) / 2.0).mean()), 4),
         "both_won": int((wd * wl).sum()), "one_won": int(((wd + wl) == 1).sum()),
         "neither_won": int(((wd + wl) == 0).sum()), "cannot_both_win": int(opp),
         "note": "both tickets held, 1 unit each a day: roi at the live salt; implied_roi the same two tickets "
                 "by their prices; salts_roi each day's mean over the 1,000 simulation salts of the dog's "
                 "profit, plus the likeliest's, halved, day-bootstrapped (4,000, seed 7); salts_implied_roi "
                 "the salts' tickets by their prices; cannot_both_win counts the days the dog is the other "
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


OUTCOMES = {"UFC": 2, "NFL": 2, "Soccer": 3}


def equal_margin(legs, dogs, sport, lo=None, hi=None):
    """In-band dogs (2.00-4.00, between lo and hi when given): the mean
    return at the price if the margin is taken equally from each of the
    market's n outcomes (n = 2 for a fight or an NFL game, Shin's reading
    for two sides; n = 3 for soccer's home / draw / away) against the
    proportional reading these files use, and the mean chance each gives.

    WHY SOCCER CAN BE READ FROM THE DOG ROW ALONE: its p_market is the
    team's share of all THREE inverse prices, so 1 / (decimal x p_market)
    is the whole book (1 + overround), draw included, and the equal-margin
    chance is 1 / decimal - overround / 3. Shin's three-way reading is not
    the equal margin (it needs each price, the draw's too, which the dog
    files do not keep), so for soccer only the equal margin is given."""
    d = dogs[(dogs["sport"] == sport) & (dogs["decimal"] >= sdy.DOG_PRICE[0]) & (dogs["decimal"] <= sdy.DOG_PRICE[1])
             & (dogs["p_market"] < 0.5)]
    if lo is not None:
        d = d[(d["day"] >= lo) & (d["day"] <= hi)]
    n = OUTCOMES[sport]
    over = 1.0 / (d["decimal"] * d["p_market"]) - 1.0
    q_eq = 1.0 / d["decimal"] - over / n
    return {"dogs": int(len(d)), "outcomes": n, "chance_proportional": float(d["p_market"].mean()),
            "chance_equal_margin": float(q_eq.mean()),
            "return_proportional": float((d["decimal"] * d["p_market"] - 1).mean()),
            "return_equal_margin": float((d["decimal"] * q_eq - 1).mean())}


def generous_words(em):
    """Whether the proportional reading reads the dogs' chances generously,
    sport by sport, from the numbers: generous where the equal-margin
    chance comes out lower. WHY computed: for two sides it is so whenever
    there is a margin (the gap is overround x (1/2 - q)), but for three it
    is so only for a dog under 1/3 (overround x (1/3 - q)), and soccer's
    less likely team is above it in about 1 in 5 in-band matches (counted
    2026-11-13) - so the word is the data's."""
    name = lambda s: "soccer" if s == "Soccer" else s
    gen = [name(s) for s, e in em.items() if e["dogs"] and e["chance_equal_margin"] < e["chance_proportional"]]
    rest = [name(s) for s, e in em.items() if e["dogs"] and name(s) not in gen]
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    if gen and not rest:
        return "so the proportional reading these files use reads every sport's dogs generously"
    if gen:
        return (f"so the proportional reading these files use reads the {join(gen)} dogs generously and the "
                f"{join(rest)} dogs no more generously than the equal margin")
    return "so here the proportional reading does not read the dogs generously"


def draw_word(rank):
    """How the live salt drew, from its rank among the simulation salts
    (salt_stats' live_rank): its share of salts it beat, ties half. WHY
    bands, not above / below the middle: a draw in the middle half of the
    salts is a typical draw, and calling rank 510 of 1,000 "well" would be
    a verdict the number does not carry. The rank is always printed
    beside the word."""
    share = (rank["roi_below"] + rank["roi_equal"] / 2.0) / rank["of"]
    if share >= 0.75:
        return "drew well"
    if share <= 0.25:
        return "drew badly"
    return "drew about average"


def payout_words(close, live):
    """(the files' margins against live: "lower than" / "higher than" /
    "about the same as", a live leg pays: "less" / "more" / "about the
    same") from the two overrounds; None when the live one is missing.
    WHY computed: whether a live ticket pays less than the record depends
    on which way the logged live margins fall, and that can change as the
    NFL log grows. Within 0.1 point they are called the same."""
    if live is None:
        return None
    if abs(live - close) < 0.001:
        return "about the same as", "about the same"
    return ("lower than", "less") if live > close else ("higher than", "more")


ARCHIVE = ROOT / "engine" / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
UFC_LEGS = ROOT / "engine" / "experiments" / "parlay_legs.csv.gz"


def ufc_no_winner(lo, hi):
    """The UFC fights in lo..hi that ended without a winner, counted from
    the UFC archive: how many of how many, how many were draws (a decision
    with no winner) and how many no contests or overturned results, and how
    many of them are in the history's legs (parlay_legs) at all. None and
    the reason when either file cannot be read.

    WHY: parlay_legs keeps decided fights only, so these fights are never
    legs or underdogs here - the backtest draws and fills from the day's
    other fights - while live the rule cannot know them beforehand and such
    a leg voids. Which events a ticket can hold then depends on the
    outcome, against point in time; the count says how much."""
    try:
        a = pd.read_csv(ARCHIVE, usecols=["date", "fight_id", "method", "winner"], low_memory=False)
        legs = pd.read_csv(UFC_LEGS, usecols=["fight_id"])
    except (OSError, ValueError) as e:
        return None, f"the UFC archive or the history's legs could not be read ({type(e).__name__})"
    w = a[(a["date"] >= lo) & (a["date"] <= hi)]
    nw = w[w["winner"].isna() | (w["winner"].astype(str).str.strip() == "")]
    draws = nw["method"].fillna("").str.contains("dec", case=False)
    return {"fights": int(len(w)), "no_winner": int(len(nw)), "draws": int(draws.sum()),
            "no_contests": int((~draws).sum()), "in_legs": int(nw["fight_id"].isin(legs["fight_id"]).sum()),
            "lo": lo, "hi": hi}, None


def no_winner_caveat(lo, hi):
    """The caveat both blocks carry about UFC fights with no winner, its
    numbers counted by ufc_no_winner; a missing count is said with its
    reason."""
    c, why = ufc_no_winner(lo, hi)
    if c is None:
        return (f"UFC draws and no contests: the count is missing - {why}. The history's UFC legs hold decided "
                f"fights only, so such fights are neither legs nor underdogs here; live, such a fight voids its leg.")
    legs_txt = ("none of them is in the history's legs (engine/experiments/parlay_legs.csv.gz keeps decided "
                "fights only)" if c["in_legs"] == 0 else
                f"{c['in_legs']} of them are in the history's legs as void legs and {c['no_winner'] - c['in_legs']} "
                f"are not")
    share = c["no_winner"] / c["fights"] if c["fights"] else 0.0
    return (f"UFC draws and no contests: {c['no_winner']} of the {c['fights']:,} UFC fights from {c['lo']} to "
            f"{c['hi']} ({share:.1%}) ended without a winner - {c['draws']} draws, {c['no_contests']} no contests "
            f"or overturned results, counted from the UFC archive - and {legs_txt}, so here they are neither legs "
            f"nor underdogs and a ticket that day is drawn and filled from the other fights. Live the rule cannot "
            f"know them beforehand: such a fight stays in the draw and the fill, and its leg voids (pays 1.0).")


def _pct(x, digits=0):
    return f"{'+' if x >= 0 else '-'}{abs(x) * 100:.{digits}f}%"


def caveats(results, extra, ends, end, legs):
    r3 = results["same_day"]["last_3_years"]
    mo = results["market_only"]["last_3_years"]
    close = float(np.median(_overround(legs, "NFL")))
    live, n_live = nfl_live_overround()
    words = payout_words(close, live)
    if words is None:
        payouts = (f"Payouts: these files' margins (NFL closing {close:.1%} a game, the median here) cannot be "
                   f"set against live prices yet - the live figure is missing: the NFL app has logged no game with "
                   f"a moneyline. A live ticket can pay more or less than the record assumes.")
    else:
        ratio = (1.0 + close) / (1.0 + live)
        n_legs = results["same_day"]["all"].get("mean_legs") or 0
        payouts = (f"Payouts: these files' margins are {words[0]} those of the consensus moneylines the NFL app logged live "
                   f"(NFL closing {close:.1%} a game, the median here, against {live:.1%} on the {n_live} games "
                   f"logged live), so a live NFL leg at the same chance pays {words[1]}"
                   + ("" if words[1] == "about the same" else " than the record assumes")
                   + f": {abs(ratio - 1):.1%} a leg, {abs(ratio ** n_legs - 1):.1%} on a ticket of the record's "
                   f"{n_legs:.2f} legs on average were every leg priced so.")
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
        payouts,
        "Soccer here is the top five leagues only, which play no matches on international-break weekends "
        "and in June-July; on those days a ticket comes from UFC and NFL alone, or there is none.",
        "UFC days are the event date; a card held outside the Americas can fall on another ET date.",
        no_winner_caveat(START, end),
        "The likeliest ticket in 3-5x is mostly favourites, and every leg is the same model's opinion; "
        "the legs are different events, so their outcomes are close to independent.",
    ]
    return out


def dog_caveats(dog, legs, dogs, rows):
    """The underdog block's caveats, every number and every word that
    depends on one computed here. WHY the live draw is never alone: the
    live salt is one draw of the lottery, and when this was written it had
    drawn above 992 of the 1,000 simulation salts; so every line that shows
    the live draw's record shows beside it the 1,000 draws' mean and the
    price-implied return. The page calls the salt the draw, so these say
    "draw" and define the salt once."""
    lv = dog["results"]["live"]["all"]
    sl = dog["results"]["salts"]["all"]
    fs, l3 = dog["results"]["salts"]["first"], dog["results"]["salts"]["last_3_years"]
    em = {s: equal_margin(legs, dogs, s, *DOG_PERIODS["all"]) for s in ("UFC", "NFL", "Soccer")}
    live = [t for t in rows["live"] if in_period(t["day"], DOG_PERIODS["all"])]
    ufc = [dog_leg(t) for t in live if dog_leg(t)["sport"] == "UFC"]
    ufc25 = [l for t, l in ((t, dog_leg(t)) for t in live) if l["sport"] == "UFC" and t["day"][:4] == "2025"]
    nfl = sum(dog_leg(t)["sport"] == "NFL" for t in live)
    opened = sum(l["price_source"] == "open" for l in ufc)
    p, b = dog["results"]["paired"]["all"], dog["results"]["both"]["all"]
    beside = (f"over the 1,000 simulation draws {_pct(sl['roi'], 1)} ({_pct(sl['roi_low'], 1)} to "
              f"{_pct(sl['roi_high'], 1)}), by the prices {_pct(lv['implied_roi'], 1)}")
    e_txt = lambda s: (f"the {em[s]['dogs']:,} {s.replace('Soccer', 'soccer')} dogs {em[s]['chance_equal_margin']:.3f} against "
                       f"{em[s]['chance_proportional']:.3f}, {_pct(em[s]['return_equal_margin'], 1)} against "
                       f"{_pct(em[s]['return_proportional'], 1)}")
    out = [
        f"The market is hard to beat, and a draw cannot create an edge: every 3-5x ticket pays the bookmaker's "
        f"margin on every leg. This ticket is for variety and a bigger payout, never the better bet. At the "
        f"live draw it won {lv['wins']} of {lv['tickets']:,} ({_pct(lv['roi'], 1)} a ticket, "
        f"{_pct(lv['roi_low'], 1)} to {_pct(lv['roi_high'], 1)}; {beside}); an interval that spans zero is no "
        f"evidence of an edge.",
        f"Expected before any result was seen (sameday.py): by the prices it lands about 1 time in 4 (a market "
        f"chance of about 0.25 at about 3.8x), the likeliest about 2 in 7 at about 3.1x, price-implied about -8% "
        f"a ticket, and somewhat worse than its prices imply. The prices on file give the live draw's tickets a "
        f"market chance of {lv['market_hit_rate']:.3f} at {lv['mean_multiplier']:.2f}x, price-implied "
        f"{_pct(lv['implied_roi'], 1)}, and the 1,000 simulation draws' {sl['market_hit_rate']:.3f} at "
        f"{sl['mean_multiplier']:.2f}x, {_pct(sl['implied_roi'], 1)}.",
        "Prior exposure: the rule's own results were not seen before it was fixed, but underdog results over the "
        "same years had been - the UFC profiles' risker and risk-lover tickets and the NFL and soccer "
        "edge-betting checks. Its record is not an out-of-sample test of underdogs.",
        "Bookmakers load more margin on underdogs (about 5% on a dog leg against 3% on a favourite, as written "
        "before the results), so a dog ticket pays the heavier margin once on its dog and a lighter one on each "
        "favourite that fills it; the fill adds payout, not safety or value.",
        f"The market chances here take the margin off in proportion. Taking it equally off each outcome instead "
        f"(Shin's reading for a fight's or a game's two sides; for soccer's three outcomes the equal margin - "
        f"Shin's three-way reading needs the draw's price, which the dog files do not keep), the "
        f"{em['UFC']['dogs']:,} UFC dogs priced 2.00-4.00 average a chance of {em['UFC']['chance_equal_margin']:.3f} "
        f"against {em['UFC']['chance_proportional']:.3f} and return {_pct(em['UFC']['return_equal_margin'], 1)} "
        f"at their price against {_pct(em['UFC']['return_proportional'], 1)}; {e_txt('NFL')}; {e_txt('Soccer')} - "
        f"{generous_words(em)}.",
        f"NFL underdogs are chosen and priced at the closing moneyline ({nfl} of the live draw's "
        f"{lv['tickets']:,} dogs): the close knows the day's news, and whether a dog is priced 2.00-4.00 is "
        f"decided at it; live, both are decided at the consensus price when the ticket is built.",
        f"UFC cards of 2025 are priced at their opening lines (no closing line is on file for them): "
        f"{len(ufc25)} of the live draw's UFC dogs that year, {opened} of its {len(ufc)} UFC dogs in all priced "
        f"at an opening line.",
        f"The live draw is the lottery with the live salt ({dog['salt']}, its seed); the simulation draws are "
        f"the same lottery with 1,000 other salts. The live draw {draw_word(sl['live_rank'])}: over all days its "
        f"record is above {sl['live_rank']['roi_below']:,} of the {sl['live_rank']['of']:,} simulation draws' on "
        f"the same games (the first years {fs['live_rank']['roi_below']:,}, the last 3 years "
        f"{l3['live_rank']['roi_below']:,}), at prices implying {_pct(lv['implied_roi'], 1)} against "
        f"{_pct(sl['implied_roi'], 1)}. Its past draw is no forecast: the simulation draws averaged "
        f"{_pct(sl['roi'], 1)} a ticket ({_pct(sl['roi_low'], 1)} to {_pct(sl['roi_high'], 1)}), and the prices "
        f"imply {_pct(lv['implied_roi'], 1)}.",
        f"The 5th-95th percentile across the 1,000 simulation draws ({_pct(sl['spread']['roi']['p5'], 1)} to "
        f"{_pct(sl['spread']['roi']['p95'], 1)} a ticket over all days) is how much the draw alone moves the "
        f"record over the same games; it is not a confidence interval. The interval over days is "
        f"({_pct(lv['roi_low'], 1)} to {_pct(lv['roi_high'], 1)} at the live draw; {beside}).",
        "Live, a UFC leg is priced from the card the UFC app carries - the best price across books unless the "
        "card carries a median from two books or more - and this history prices each fight at one line per "
        "card. A best price pays at least a typical book's, so a live UFC dog's price, and whether it falls in "
        "2.00-4.00, can differ from what this record assumes.",
        f"Same days and events as the likeliest ticket: on the {p['days']:,} days both exist, the two cannot "
        f"both win on {b['cannot_both_win']} (the dog is the other side of one of the likeliest ticket's legs). "
        f"Held together, 1 unit each, they returned {_pct(b['roi'], 1)} a unit at the live draw "
        f"({_pct(b['roi_low'], 1)} to {_pct(b['roi_high'], 1)}), {_pct(b['salts_roi'], 1)} over the 1,000 "
        f"simulation draws ({_pct(b['salts_roi_low'], 1)} to {_pct(b['salts_roi_high'], 1)}), and "
        f"{_pct(b['implied_roi'], 1)} by the prices; the underdog ticket minus the likeliest was "
        f"{_pct(p['roi_diff'], 1)} a ticket at the live draw, {_pct(p['salts_roi_diff'], 1)} over the draws "
        f"({_pct(p['salts_roi_diff_low'], 1)} to {_pct(p['salts_roi_diff_high'], 1)}), and "
        f"{_pct(p['implied_roi_diff'], 1)} by the prices.",
        no_winner_caveat(*DOG_PERIODS["all"]),
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
                     f"{sl['live_rank']['roi_below']:,} of the {sl['live_rank']['of']:,} - "
                     + (f"at about the same prices, so the gap is which dogs happened to win."
                        if abs(lv["implied_roi"] - sl["implied_roi"]) < 0.005 else
                        f"at prices implying {_pct(lv['implied_roi'], 1)} against {_pct(sl['implied_roi'], 1)}, "
                        f"so part of the gap is the prices and the rest which dogs happened to win.")),
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
    payload["v2"] = measure_v2(payload)
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
    table_v2(payload["v2"])


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
                  f"[{p['roi_diff_low']:+.3f}, {p['roi_diff_high']:+.3f}], implied {p['implied_roi_diff']:+.3f}; "
                  f"salts mean {p['salts_roi_diff']:+.3f} [{p['salts_roi_diff_low']:+.3f}, "
                  f"{p['salts_roi_diff_high']:+.3f}], implied {p['salts_implied_roi_diff']:+.3f}")
            print(f"  both held: {b['units']} units, ROI {b['roi']:+.3f} [{b['roi_low']:+.3f}, {b['roi_high']:+.3f}],"
                  f" implied {b['implied_roi']:+.3f}; salts mean {b['salts_roi']:+.3f} [{b['salts_roi_low']:+.3f}, "
                  f"{b['salts_roi_high']:+.3f}], implied {b['salts_implied_roi']:+.3f}; both won {b['both_won']}, "
                  f"one {b['one_won']}, neither {b['neither_won']}, cannot both win {b['cannot_both_win']}")
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


# ================================================================ tickets v2
#
# sameday.py's TICKETS v2, MEASURED: LIKELIEST v2, UNDERDOG v2 (salt
# DOG_SALT_V2, the 1,000 DOG_SIM_SALTS_V2), MIXED and PROPS - "as priced"
# (chosen and paid at the real props, BAND) and "as live" (chosen by the
# yearly-refitted estimator's price in PROPS_LIVE_BAND, paid at the real
# props; the headline), each over PROPS_SALT and the 1,000 PROPS_SIM_SALTS -
# every one built by sameday's own v2 functions, over every day of the
# registered periods (v1's DOG_PERIODS; PROPS to PROPS_HISTORY_END, so its
# last-3-years period is the overlap), with every league in the app and
# again WITHOUT the extra leagues (their history is priced at the close, live
# at the pre-match price), every record beside its price-implied return.
#
# WHY THE FILLS ARE BUILT ONCE (as v1's): a dog's or a prop's fill does not
# depend on the salt, so each candidate is filled once, by build_underdog_v2
# or build_props itself on that one candidate, and a salt only orders the
# candidates - the first with a fill is its ticket. The live salt's tickets
# are built by the same builders over all of a day's candidates, and the
# 1,000-salt matrix must reproduce them exactly or the run stops.
#
# Nothing here can move a rule: v2 was fixed in sameday.py before any of
# these numbers was computed, and the splits are descriptive.

V2_PERIODS = DOG_PERIODS
PROPS_PERIODS = {k: (lo, min(hi, sdy.PROPS_HISTORY_END)) for k, (lo, hi) in DOG_PERIODS.items()}
V2_VARIANTS = ("all_leagues", "without_extra_leagues")
V2_SPLIT_NOTE = "descriptive: none can move a rule in v2"
# The PROPS readings. "as live" (the headline) draws and chooses as live does:
# by the estimator's chance (the shipped win chance x the ESTIMATOR's share)
# and price, in the live band, paid at the real prop. Beside it, report-only:
# the same weighted by the real book's split (the first build's reading - an
# input live never has, kept so the change is visible), "as priced" on every
# props day, and "as priced" on the days as live can draw (a prop with an
# estimate) - the comparison row, since the 2016 cards have no estimate and
# a row over other days would read as the estimate's cost.
PROPS_KINDS = ("props_as_live", "props_as_live_book_split", "props_as_priced", "props_as_priced_estimated_days")
V2_KINDS = ("likeliest", "underdog", "mixed") + PROPS_KINDS
V2_LOTTERIES = {"underdog": (sdy.DOG_SALT_V2, sdy.DOG_SIM_SALTS_V2),
                **{k: (sdy.PROPS_SALT, sdy.PROPS_SIM_SALTS) for k in PROPS_KINDS}}
# The four kinds that ship (PROPS ships as live); the other props readings are reported beside it.
V2_SHIPPED = ("likeliest", "underdog", "mixed", "props_as_live")
PROPS_LABELS = {
    "props_as_live": "as live: drawn and chosen as live does - each prop weighted by the shipped win chance times the "
                     "estimator's share and priced at the estimate, in the live band - and paid at the real price",
    "props_as_live_book_split": "as live but weighted by the real book's split: chosen at the estimated price in the "
                                "live band, the draw weighting each prop by the shipped win chance times the real "
                                "six-way book's share, which live never has; paid at the real price",
    "props_as_priced": "as priced: chosen and paid at the real prop price, band 3-5x, on every day with props",
    "props_as_priced_estimated_days": "as priced on the days as live can draw (a prop with an estimate): chosen and "
                                      "paid at the real prop price, band 3-5x - the row to set beside the headline",
}
LEFT_OUT = DATA / "legs_v2_ufc_method_left_out.csv.gz"
V2_SOURCES = {
    "winners": (("legs_ufc", "UFC"), ("legs_nfl", "NFL"), ("legs_v2_soccer", "Soccer")),
    "dogs": (("legs_dogs_ufc", "UFC"), ("legs_dogs_nfl", "NFL"), ("legs_dogs_v2_soccer", "Soccer")),
    "others": (("legs_v2_ou25", "Soccer"), ("legs_v2_nfl_lines", "NFL"), ("legs_v2_ufc_method", "UFC")),
}
V1_LEGS_FILES = tuple(f"{p}_{s}.csv.gz" for p in ("legs", "legs_dogs") for s in ("ufc", "nfl", "soccer"))
METHOD_JSON = ROOT / "engine" / "data" / "method_price.json"
WORKERS_V2 = 4


def _str(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def load_v2(lo=None, hi=None):
    """The v2 pool's files (V2_SOURCES: the v1 UFC and NFL files, the v2
    soccer, over/under, NFL lines and UFC props files) on the days lo..hi
    (default V2_PERIODS["all"]), each row tagged with its sport and file."""
    lo, hi = lo or V2_PERIODS["all"][0], hi or V2_PERIODS["all"][1]
    out = {}
    for part, files in V2_SOURCES.items():
        frames = []
        for name, sport in files:
            d = pd.read_csv(DATA / f"{name}.csv.gz")
            d["sport"], d["file"] = sport, name
            frames.append(d[(d["day"] >= lo) & (d["day"] <= hi)])
        out[part] = pd.concat(frames, ignore_index=True)
    return out


def to_v2_leg(r):
    """A v2 file row as sameday.market_leg takes it live (a v1 row is a
    winner leg, bet "1X2"), tagged with its soccer league and group."""
    bet = _str(getattr(r, "bet", None)) or "1X2"
    l = sdy.market_leg(r.sport, r.key, r.day, r.event, r.pick, _num(r.p), _num(r.decimal), bet,
                       bet_detail=_str(getattr(r, "bet_detail", None)), p_market=_num(r.p_market),
                       p_model=_num(r.p_model), opponent=r.opponent, result=RESULT[r.result],
                       price_source=r.price_source,
                       lot_key=_str(getattr(r, "lot_key", None)) if bet == "METHOD" else None,
                       est_decimal=_num(getattr(r, "est_decimal", None)))
    if l is not None:
        l["group"], l["league"] = _str(getattr(r, "group", None)), _str(getattr(r, "league", None))
        if bet == "METHOD":
            # the estimator's chance beside the book's (legs_history: the as-live draw reads it)
            l["est_p"] = _num(getattr(r, "est_p", None))
    return l


def to_v2_dog(r):
    d = to_dog(r)
    if d is not None:
        d["group"], d["league"] = _str(getattr(r, "group", None)), _str(getattr(r, "league", None))
    return d


def v2_days(frames, days=None):
    """{day: (winner legs, other-market legs, underdog candidates)} - each
    row through sameday's own constructors (market_leg, dog). days: only
    these (the tests rebuild a sample)."""
    keep = None if days is None else set(days)
    out = defaultdict(lambda: ([], [], []))
    for part, conv, slot in (("winners", to_v2_leg, 0), ("others", to_v2_leg, 1), ("dogs", to_v2_dog, 2)):
        f = frames[part]
        if keep is not None:
            f = f[f["day"].isin(keep)]
        for r in f.itertuples():
            x = conv(r)
            if x is not None:
                out[r.day][slot].append(x)
    return dict(out)


def v2_kinds(W, O, D):
    """One day's v2 work for one league set, by sameday's v2 builders: the
    LIKELIEST v2 ticket and its market-only control, MIXED and its control,
    every underdog candidate's fill and the live salt's UNDERDOG v2 ticket,
    and for each PROPS reading (PROPS_KINDS) every in-window prop's fill and
    the live salt's ticket. Only fillable candidates are kept (a salt's
    ticket is its first fillable candidate). A fill's candidate is the prop
    as the draw saw it (as live: its estimated chance), so the salt matrix
    weights what the builder weighted."""
    M = sorted((l for l in O if l["bet"] == "METHOD"), key=lambda l: (l["key"], l["lot_key"]))
    M_est = [m for m in M if m.get("est_decimal") is not None and m.get("est_p") is not None]
    out = {"likeliest": sdy.build_likeliest_v2(W), "likeliest_market_only": sdy.build_likeliest_v2(W, "p_market"),
           "mixed": sdy.build_mixed(W + O), "mixed_market_only": sdy.build_mixed(W + O, "p_market")}
    cands = sorted((d for d in D if d["p"] >= sdy.DOG_FLOOR), key=lambda d: d["key"])
    out["underdog_candidates"] = len(cands)
    out["underdog_fills"] = [(d, t) for d, t in ((d, sdy.build_underdog_v2(W, [d])) for d in cands) if t is not None]
    out["underdog"] = sdy.build_underdog_v2(W, D)
    live = {"band": sdy.PROPS_LIVE_BAND, "price_key": "est_decimal"}
    for kind, ms, price, kw in (("props_as_priced", M, "decimal", {}),
                                ("props_as_live", M_est, "est_decimal", dict(live, chance_key="est_p")),
                                ("props_as_live_book_split", M_est, "est_decimal", live)):
        inwin = [m for m in ms if sdy.PROPS_PRICE[0] <= m[price] <= sdy.PROPS_PRICE[1]]
        seen = (lambda m: sdy._priced_as(m, "decimal", kw["chance_key"])) if "chance_key" in kw else (lambda m: m)
        out[f"{kind}_candidates"] = len(inwin)
        out[f"{kind}_fills"] = [(seen(m), t) for m, t in ((m, sdy.build_props(W, [m], **kw)) for m in inwin)
                                if t is not None]
        out[kind] = sdy.build_props(W, ms, **kw) if ms else None
    # "as priced" on the days as live can draw: the same tickets, on days with an estimated prop only
    k, on = "props_as_priced", bool(M_est)
    out["props_as_priced_estimated_days"] = out[k] if on else None
    out["props_as_priced_estimated_days_fills"] = out[f"{k}_fills"] if on else []
    out["props_as_priced_estimated_days_candidates"] = out[f"{k}_candidates"] if on else 0
    out["props_days"] = bool(M)
    out["props_without_estimate"] = len(M) - len(M_est)
    return out


_V2 = {}


def _not_extra(ls):
    return [l for l in ls if l.get("group") != "extra"]


def _v2_day(day):
    """Both league sets' work for one day; the same object for both when
    the day has no extra-league leg."""
    W, O, D = _V2[day]
    base = v2_kinds(W, O, D)
    out = {"day": day, "all_leagues": base}
    if any(l.get("group") == "extra" for l in W + O + D):
        out["without_extra_leagues"] = v2_kinds(_not_extra(W), _not_extra(O), _not_extra(D))
    else:
        out["without_extra_leagues"] = base
    return out


def run_v2(frames):
    _V2.clear()
    _V2.update(v2_days(frames))
    with mp.get_context("fork").Pool(WORKERS_V2) as pool:
        return pool.map(_v2_day, sorted(_V2), chunksize=8)


def bets_of(t):
    return sorted({l.get("bet") or "1X2" for l in t["legs"]})


def v2_signature(t):
    """A ticket's legs by event, market, detail and selection (a prop and
    its fighter's moneyline share key and pick, so the market is in it)."""
    return None if t is None else tuple(sorted(
        (l["key"], l.get("bet") or "1X2", str(l.get("bet_detail")), str(l["pick"])) for l in t["legs"]))


def salt_tables(tables, salts):
    """{name: (ticket-days, chosen candidate per salt and day, candidates)}
    for tables {name: [(day, [(candidate, ticket), ...]), ...]} drawn with
    `salts` - each salt's ticket the fillable candidate with the lowest
    -ln(u) / chance (sameday.dog_order's lottery; candidates in key order,
    argmin taking the first, as dog_order's tie by key). One -ln(u) matrix
    over every (day, lot key) any table holds."""
    pairs, built = {}, {}
    for name, segs_in in tables.items():
        cands, segs = [], []
        for day, feas in segs_in:
            if not feas:
                continue
            start = len(cands)
            for d, t in feas:
                k = pairs.setdefault((day, d["lot_key"]), len(pairs))
                cands.append({"day": day, "pair": k, "w": d["p"], "ticket": t})
            segs.append((day, start, len(cands)))
        built[name] = (cands, segs)
    _PAIRS[:] = sorted(pairs, key=pairs.get)
    chunks = [list(salts[i:i + 50]) for i in range(0, len(salts), 50)]
    with mp.get_context("fork").Pool(WORKERS_V2) as pool:
        E = np.vstack(pool.map(_neglog, chunks))
    out = {}
    for name, (cands, segs) in built.items():
        col = np.array([c["pair"] for c in cands], dtype=np.int64)
        w = np.array([c["w"] for c in cands])
        chosen = np.empty((len(salts), len(segs)), dtype=np.int64)
        for k, (_, a, b) in enumerate(segs):
            chosen[:, k] = a + (E[:, col[a:b]] / w[None, a:b]).argmin(axis=1)
        out[name] = ([s[0] for s in segs], chosen, cands)
    return out


def v2_lotteries(days_out):
    """The salt matrices of the three lotteries in both league sets, each
    checked against the live builder: row 0 (the live salt) must give the
    very ticket build_underdog_v2 / build_props gave over all of the day's
    candidates, on exactly the days it gave one - or the run stops."""
    out = {}
    # the props readings share their salts, so one matrix draws them all
    groups = {"underdog": ("underdog",), "props": PROPS_KINDS}
    for group, kinds in groups.items():
        live, sims = V2_LOTTERIES[kinds[0]]
        tables = {(k, v): [(o["day"], o[v][f"{k}_fills"]) for o in days_out
                           if in_period(o["day"], (PROPS_PERIODS if k.startswith("props") else V2_PERIODS)["all"])]
                  for k in kinds for v in V2_VARIANTS}
        M = salt_tables(tables, (live,) + sims)
        for (k, v), (days, chosen, cands) in M.items():
            per = (PROPS_PERIODS if k.startswith("props") else V2_PERIODS)["all"]
            by_day = {o["day"]: o[v][k] for o in days_out if in_period(o["day"], per)}
            for j, d in enumerate(days):
                if v2_signature(cands[chosen[0, j]]["ticket"]) != v2_signature(by_day[d]):
                    raise SystemExit(f"{d}: the salt matrix does not reproduce the live builder ({k}, {v})")
            if sum(t is not None for t in by_day.values()) != len(days):
                raise SystemExit(f"the salt matrix misses days with a ticket ({k}, {v})")
            out[(k, v)] = (days, chosen, cands)
    return out


def calendar_days(period):
    lo, hi = period
    return (date.fromisoformat(hi) - date.fromisoformat(lo)).days + 1


def period_label(period):
    """A period named by its own dates and length (the props periods end
    with the props history, so "last 3 years" is not three years there)."""
    lo, hi = period
    n = calendar_days(period)
    span = f"{n / 365.25:.1f} years" if n >= 2 * 365 else f"{round(n / 30.4375)} months"
    return f"{lo[:7]} to {hi[:7]}, {span}"


def band_years(ts):
    """How often the as-live tickets' REAL multiplier lands in BAND, in the
    years the live band was chosen on and outside them. WHY: the scouts chose
    PROPS_LIVE_BAND on the estimator's test cards (sameday.py PROPS; the test
    years read from engine/data/method_price.json), so the share over those
    years is partly in-sample; the years before them are not."""
    try:
        lo, hi = json.loads(METHOD_JSON.read_text())["test"]["test"]
    except (OSError, ValueError, KeyError) as e:
        return {"missing": f"the band's choice years are missing - engine/data/method_price.json "
                           f"({type(e).__name__})"}

    def share(xs):
        return {"tickets": len(xs), "real_in_band": round(float(np.mean([sdy.BAND[0] <= t["multiplier"] <= sdy.BAND[1]
                                                                          for t in xs])), 4) if xs else None}
    return {"band_chosen_on": [lo, hi],
            "source": "the estimator's test years (engine/data/method_price.json test.test), on whose cards the "
                      "scouts chose the live band (sameday.py PROPS)",
            "in_choice_years": share([t for t in ts if lo <= t["day"] <= hi]),
            "before_choice_years": share([t for t in ts if t["day"] < lo]),
            "after_choice_years": share([t for t in ts if t["day"] > hi])}


def left_out_info():
    """The fights legs_history.py left out of the props file, by reason
    (legs_v2_ufc_method_left_out: a card priced from the opening line), or
    why that list is missing."""
    try:
        d = pd.read_csv(LEFT_OUT)
    except (OSError, ValueError) as e:
        return {"left_out_missing": f"{LEFT_OUT.name} could not be read ({type(e).__name__}): run "
                                    f"legs_history.py --v2 --only ufc"}
    return {"left_out": {why: int(n) for why, n in d["reason"].value_counts().items()},
            "left_out_days": int(d["day"].nunique()),
            "left_out_span": [d["day"].min(), d["day"].max()] if len(d) else None}


def v2_record(tickets, period, days_with_legs=None):
    """record() of the tickets in `period` (price-implied return always
    beside it, None when there is no ticket), with the activity: calendar
    days, days with a ticket and their share."""
    ts = [t for t in tickets if in_period(t["day"], period)]
    s = record(ts)
    s.setdefault("implied_roi", None)
    cal = calendar_days(period)
    s.update({"days": cal, "days_with_ticket": len(ts), "share_of_days": round(len(ts) / cal, 4),
              "tickets_per_day": round(len(ts) / cal, 4)})
    if days_with_legs is not None:
        n = sum(in_period(d, period) for d in days_with_legs)
        s["days_with_legs"] = n
        s["share_of_days_with_legs"] = round(len(ts) / n, 4) if n else None
    return s


def _leg_profit(l):
    o = pp_leg_outcome(l)
    return 0.0 if o == "void" else (l["decimal"] - 1.0 if o == "won" else -1.0)


def pp_leg_outcome(l):
    return sdy.pp.leg_outcome(l)


def leg_table(tickets, key):
    """Descriptive, per group of legs held by the tickets: how many, how
    often they landed against their chances and the prices', and each leg
    as a 1-unit single bet at its price (return and price-implied return)."""
    by = defaultdict(list)
    for t in tickets:
        for l in t["legs"]:
            by[key(l)].append(l)
    out = {}
    for k, ls in sorted(by.items()):
        outs = [pp_leg_outcome(l) for l in ls]
        decided = [o for o in outs if o != "void"]
        out[k] = {"legs": len(ls), "won": int(sum(o == "won" for o in outs)),
                  "void": int(sum(o == "void" for o in outs)),
                  "hit_rate": round(float(np.mean([o == "won" for o in decided])), 4) if decided else None,
                  "mean_p": round(float(np.mean([l["p"] for l in ls])), 4),
                  "mean_p_market": round(float(np.mean([l["p_market"] for l in ls])), 4),
                  "mean_decimal": round(float(np.mean([l["decimal"] for l in ls])), 3),
                  "single_roi": round(float(np.mean([_leg_profit(l) for l in ls])), 4),
                  "single_implied_roi": round(float(np.mean([l["p_market"] * l["decimal"] - 1 for l in ls])), 4)}
    return out


def v2_splits(tickets):
    """Descriptive splits of one kind's tickets: by the markets a ticket
    holds, by its sports and by its legs, and its legs by market and by
    sport."""
    out = {"note": V2_SPLIT_NOTE}
    for name, f in (("by_bets", lambda t: "+".join(bets_of(t))), ("by_sports", lambda t: "+".join(t["sports"])),
                    ("by_legs", lambda t: f"{t['n_legs']} legs")):
        by = defaultdict(list)
        for t in tickets:
            by[f(t)].append(t)
        out[name] = {"note": V2_SPLIT_NOTE, **{k: dict(record(v), share=round(len(v) / len(tickets), 4))
                                               for k, v in sorted(by.items())}}
    out["legs_by_bet"] = {"note": V2_SPLIT_NOTE, **leg_table(tickets, lambda l: l.get("bet") or "1X2")}
    out["legs_by_sport"] = {"note": V2_SPLIT_NOTE, **leg_table(tickets, lambda l: l["sport"])}
    return out


def prop_of(t):
    return next(l for l in t["legs"] if l.get("bet") == "METHOD")


def props_detail(ts):
    """Outcome-free facts of the props tickets: the drawn prop's price and
    type, its estimate's error against the real price, how often the real
    multiplier lands in BAND and the multiplier the search read."""
    if not ts:
        return {}
    props = [prop_of(t) for t in ts]
    est = [(l["est_decimal"], l["decimal"]) for l in props if l.get("est_decimal") is not None]
    return {"prop_price": quantiles([l["decimal"] for l in props]),
            "prop_type": {k: v for k, v in sorted(Counter(l["bet_detail"] for l in props).items())},
            "prop_est_error_median": round(float(np.median([abs(e - d) / d for e, d in est])), 4) if est else None,
            "real_in_band": round(float(np.mean([sdy.BAND[0] <= t["multiplier"] <= sdy.BAND[1] for t in ts])), 4),
            "searched_multiplier": round(float(np.mean([t["searched_multiplier"] for t in ts])), 2),
            "fallback": {str(k): v for k, v in sorted(Counter(t["method_rank"] for t in ts).items())}}


def mixed_detail(ts):
    """Which other markets the MIXED tickets hold (outcome-free)."""
    if not ts:
        return {}
    n = len(ts)
    other = [[l for l in t["legs"] if not sdy.is_winner(l)] for t in ts]
    hold = Counter(b for o in other for b in {l["bet"] for l in o})
    return {"holds": {b: round(hold[b] / n, 4) for b in sorted(hold)},
            "mean_other_legs": round(float(np.mean([len(o) for o in other])), 2)}


def measure_v2(payload):
    """The "v2" block of sameday_backtest.json (module docstring above)."""
    frames = load_v2()
    days_out = run_v2(frames)
    M = v2_lotteries(days_out)
    # days with legs: "props" any prop (as priced can draw), "props_estimated" a prop with an
    # estimate (as live can draw: the 2016 cards have none)
    has_legs = {v: {"winners": set(), "props": set(), "props_estimated": set(), "any": set()} for v in V2_VARIANTS}
    for part in ("winners", "others", "dogs"):
        f = frames[part]
        for v in V2_VARIANTS:
            g = f if v == "all_leagues" else f[f["group"].ne("extra")]
            has_legs[v]["any"] |= set(g["day"])
            if part == "winners":
                has_legs[v]["winners"] |= set(g["day"])
            if part == "others":
                me = g[g["bet"].eq("METHOD")]
                has_legs[v]["props"] |= set(me["day"])
                has_legs[v]["props_estimated"] |= set(me.loc[me["est_decimal"].notna() & me["est_p"].notna(), "day"])
    tickets = {(k, v): [o[v][k] for o in days_out if o[v][k] is not None]
               for k in V2_KINDS + ("likeliest_market_only", "mixed_market_only") for v in V2_VARIANTS}
    kinds = {}
    for k in V2_KINDS:
        periods = PROPS_PERIODS if k.startswith("props") else V2_PERIODS
        legs_days = {"props_as_priced": "props"}.get(k, "props_estimated" if k.startswith("props") else "winners")
        block = {"periods": {p: list(x) for p, x in periods.items()},
                 "period_labels": {p: period_label(x) for p, x in periods.items()}, "live": {}}
        if k.startswith("props"):
            block["days_with_legs_are"] = ("days with a prop" if legs_days == "props" else
                                           "days with a prop that has an estimate (as live can draw)")
        if k in V2_LOTTERIES:
            block["salt"], block["salts"] = V2_LOTTERIES[k][0], {}
        if k in ("likeliest", "mixed"):
            block["market_only"] = {}
        for v in V2_VARIANTS:
            ts = tickets[(k, v)]
            block["live"][v] = {p: v2_record(ts, x, has_legs[v][legs_days]) for p, x in periods.items()}
            if k.startswith("props"):
                for p, x in periods.items():
                    block["live"][v][p].update(props_detail([t for t in ts if in_period(t["day"], x)]))
            if k == "mixed":
                for p, x in periods.items():
                    block["live"][v][p].update(mixed_detail([t for t in ts if in_period(t["day"], x)]))
            if k in V2_LOTTERIES:
                days, chosen, cands = M[(k, v)]
                block["salts"][v] = {}
                for p, x in periods.items():
                    lv = [t for t in ts if in_period(t["day"], x)]
                    roi = float(np.mean([profit(t) for t in lv])) if lv else None
                    hit = float(np.mean([sdy.settle(t)["status"] == "won" for t in lv])) if lv else None
                    st = salt_stats(days, chosen, cands, x, roi, hit)
                    st.setdefault("implied_roi", None)
                    block["salts"][v][p] = st
            if "market_only" in block:
                mts = tickets[(f"{k}_market_only", v)]
                block["market_only"][v] = {p: v2_record(mts, x) for p, x in periods.items()}
        block["splits"] = v2_splits([t for t in tickets[(k, "all_leagues")]
                                     if in_period(t["day"], periods["all"])])
        kinds[k] = block
    for k in PROPS_KINDS:
        kinds[k]["label"] = PROPS_LABELS[k]
    kinds["props_as_live"]["real_in_band_by_years"] = band_years(
        [t for t in tickets[("props_as_live", "all_leagues")] if in_period(t["day"], PROPS_PERIODS["all"])])
    kinds["underdog"]["spread_label"] = SPREAD_LABEL

    activity = {}
    for v in V2_VARIANTS:
        activity[v] = {}
        for p, x in V2_PERIODS.items():
            per_kind = {k: [t for t in tickets[(k, v)] if in_period(t["day"], x)] for k in V2_SHIPPED}
            n = sum(len(ts) for ts in per_kind.values())
            any_days = {t["day"] for ts in per_kind.values() for t in ts}
            cal = calendar_days(x)
            activity[v][p] = {"days": cal, "tickets": n, "tickets_per_day": round(n / cal, 4),
                              "days_with_a_ticket": len(any_days), "share_of_days": round(len(any_days) / cal, 4),
                              "by_kind": {k: len(ts) for k, ts in per_kind.items()}}
    v1 = {"likeliest": payload["results"]["same_day"], "underdog": payload["dog"]["results"]["live"]}
    v1_n = {p: sum(v1[k].get(p, {}).get("tickets", 0) for k in v1) for p in V2_PERIODS}
    for p in V2_PERIODS:
        cal = calendar_days(V2_PERIODS[p])
        activity.setdefault("v1", {})[p] = {
            "days": cal, "tickets": v1_n[p], "tickets_per_day": round(v1_n[p] / cal, 4),
            "note": "v1's likeliest and underdog tickets (this file's results.same_day and dog.results.live)"}
    activity["note"] = ("tickets of the four kinds that ship (LIKELIEST v2, UNDERDOG v2 at the live salt, MIXED, "
                        "PROPS as live) per calendar day; PROPS has history to " + sdy.PROPS_HISTORY_END + " only")

    legs_info = v2_legs_info(frames)
    legs_info["legs_v2_ufc_method"].update(left_out_info())
    margins = v2_margins(frames)
    v2 = {
        "version": sdy.V2_VERSION,
        "rule": {"band": list(sdy.BAND), "max_legs": sdy.MAX_LEGS, "pool": sdy.POOL, "min_p": sdy.MIN_P,
                 "mixed_pool_other": sdy.MIXED_POOL_OTHER, "dog_salt": sdy.DOG_SALT_V2,
                 "props_salt": sdy.PROPS_SALT, "props_price": list(sdy.PROPS_PRICE),
                 "props_live_band": list(sdy.PROPS_LIVE_BAND), "props_history_end": sdy.PROPS_HISTORY_END,
                 "dog_price": list(sdy.DOG_PRICE), "dog_floor": sdy.DOG_FLOOR, "leg_markets": list(sdy.LEG_MARKETS),
                 "sim_salts": {"underdog": len(sdy.DOG_SIM_SALTS_V2), "props": len(sdy.PROPS_SIM_SALTS)}},
        "periods": {p: list(x) for p, x in V2_PERIODS.items()},
        "props_periods": {p: list(x) for p, x in PROPS_PERIODS.items()},
        "variants": {"all_leagues": "every league in the app",
                     "without_extra_leagues": "the same days without the extra leagues' legs and underdogs "
                                              "(their history is priced at the close)"},
        "v1_files_sha256": v1_hashes(),
        "v1_sha256": v1_fingerprint(payload),
        "legs": legs_info,
        "margins": margins,
        "kinds": kinds,
        "activity": activity,
    }
    v2["headline"] = v2_headline(v2)
    v2["caveats"] = v2_caveats(v2, frames, tickets)
    write_v2_tickets(tickets)
    return v2


def v2_margins(frames):
    """Outcome-free: the median overround of the six props' book on the
    fights with props, against the same fights' moneyline pair (each from
    its row: 1 / (decimal x no-vig chance) - 1), and the estimator file's
    own medians (engine/data/method_price.json coverage) beside them."""
    me = frames["others"][frames["others"]["file"] == "legs_v2_ufc_method"]
    ufc = frames["winners"][frames["winners"]["file"] == "legs_ufc"]
    ufc = ufc[ufc["key"].isin(set(me["key"]))]
    out = {"fights": int(me["key"].nunique()),
           "six_way_median": round(float(me.drop_duplicates("key")["method_overround"].median()), 4)
           if len(me) else None,
           "moneyline_median": round(float((1.0 / (ufc["decimal"] * ufc["p_market"]) - 1.0).median()), 4)
           if len(ufc) else None}
    try:
        cov = json.loads(METHOD_JSON.read_text())["coverage"]
        out["method_price_json"] = {"six_way_median": round(float(cov["six_way_overround_median"]), 4),
                                    "moneyline_median": round(float(cov["moneyline_overround_median"]), 4),
                                    "first": cov["first"], "last": cov["last"]}
    except (OSError, ValueError, KeyError) as e:
        out["method_price_json"] = {"missing": f"engine/data/method_price.json could not be read ({type(e).__name__})"}
    lo, hi = PROPS_PERIODS["all"]
    w = frames["winners"]
    card_days = set(w.loc[(w["file"] == "legs_ufc") & w["day"].between(lo, hi), "day"])
    props_days = set(me["day"])
    out["ufc_card_days"] = len(card_days)
    out["card_days_with_props"] = len(card_days & props_days)
    return out


def v1_fingerprint(payload):
    """sha256 of every key of the json but "v2" (canonical JSON): v2 is
    measured beside v1's published numbers and must leave them as they are;
    a test recomputes it from the published file."""
    import hashlib
    v1 = {k: v for k, v in payload.items() if k != "v2"}
    text = json.dumps(v1, sort_keys=True, separators=(",", ":"), default=_json, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def v1_hashes():
    """sha256 of the v1 legs files beside which v2 was measured: v2 reads the
    UFC and NFL ones and must leave every one byte for byte as v1 had it."""
    import hashlib
    return {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in V1_LEGS_FILES}


def v2_legs_info(frames):
    """Outcome-free counts of the v2 pool: legs by file and price source,
    the NFL lines' side source, the props with and without an estimate (the
    reason), and the leagues."""
    out = {}
    for part in ("winners", "others", "dogs"):
        f = frames[part]
        for name, g in f.groupby("file"):
            row = {"rows": int(len(g)), "first": g["day"].min(), "last": g["day"].max(),
                   "price_source": {k: int(v) for k, v in g["price_source"].value_counts().sort_index().items()}}
            if "bet" in g and g["bet"].notna().any():
                row["bet"] = {k: int(v) for k, v in g["bet"].value_counts().sort_index().items()}
            out[name] = row
    w = frames["winners"]
    soc = w[w["file"] == "legs_v2_soccer"]
    out["legs_v2_soccer"]["leagues"] = {g: sorted(x["league"].unique()) for g, x in soc.groupby("group")}
    out["legs_v2_soccer"]["day_source"] = {k: int(v) for k, v in soc["day_source"].value_counts().items()}
    ou = frames["others"][frames["others"]["file"] == "legs_v2_ou25"]
    out["legs_v2_ou25"]["leagues"] = sorted(ou["league"].unique())
    nl = frames["others"][frames["others"]["file"] == "legs_v2_nfl_lines"]
    out["legs_v2_nfl_lines"]["side_source"] = {f"{b} {s}": int(n) for (b, s), n in
                                               nl.groupby(["bet", "side_source"]).size().items()}
    out["legs_v2_nfl_lines"]["void"] = {b: int(n) for b, n in nl[nl["result"] == "void"].groupby("bet").size().items()}
    tot_keys, spr_keys = set(nl.loc[nl["bet"] == "TOTAL", "key"]), set(nl.loc[nl["bet"] == "SPREAD", "key"])
    out["legs_v2_nfl_lines"]["games_with_a_total_but_no_spread_leg"] = len(tot_keys - spr_keys)
    me = frames["others"][frames["others"]["file"] == "legs_v2_ufc_method"]
    out["legs_v2_ufc_method"]["fights"] = int(me["key"].nunique())
    out["legs_v2_ufc_method"]["est_missing"] = {k: int(v) for k, v in me["est_missing"].value_counts().items()}
    out["legs_v2_ufc_method"]["est_missing_by_year"] = {
        y: int(v) for y, v in me[me["est_decimal"].isna()]["day"].str[:4].value_counts().sort_index().items()}
    won = me.groupby("key")["result"].apply(lambda r: int((r == "won").sum()))
    out["legs_v2_ufc_method"]["fights_no_prop_won"] = int((won == 0).sum())
    return out


def _doc_block(label):
    """The text of one labelled block of sameday.py's TICKETS v2 docstring
    (e.g. "PRIOR EXPOSURE"), joined into one line - quoted, never retyped."""
    doc = sdy.__doc__ or ""
    sec = doc[doc.index("TICKETS v2"):]
    lines = sec.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"  {label} "))
    out = [lines[start][len(f"  {label} "):].strip()]
    for ln in lines[start + 1:]:
        if not ln.strip() or re.match(r"^  [A-Z]", ln):
            break
        out.append(ln.strip())
    return " ".join(out)


def _method_error_text():
    """The estimator's measured error, read from engine/data/method_price.json
    (written by engine/experiments/method_price_fit.py), or the reason it is
    missing."""
    try:
        doc = json.loads(METHOD_JSON.read_text())
        t = doc["test"]
    except (OSError, ValueError, KeyError) as e:
        return None, f"the estimator's measured error is missing - engine/data/method_price.json ({type(e).__name__})"
    by = t["by_leg"]
    name = {"KO/TKO": "KO/TKO", "Submission": "submission", "Decision": "decision"}
    parts = [f"{name[k]} {by[k]['fav']:.1%} as the favourite and {by[k]['dog']:.1%} as the underdog"
             for k in ("Decision", "KO/TKO", "Submission")]
    return doc, (f"fitted {t['fit'][0][:4]}-{t['fit'][1][:4]} and tested {t['test'][0][:4]}-{t['test'][1][:4]}, "
                 f"its median miss against the real prop price is {'; '.join(parts)}; {t['all_single_props']:.1%} "
                 f"over all {t['props']:,} single props")


def v2_caveats(v2, frames, tickets):
    """The v2 block's caveats, every number and every result-dependent word
    computed here or read from the file that measured it."""
    K = v2["kinds"]
    li = lambda k, v="all_leagues", p="all": K[k]["live"][v][p]
    pc = lambda x: "missing" if x is None else _pct(x, 1)
    sh = lambda x: "missing" if x is None else f"{x:.1%}"
    L, U, X, PL, PP = li("likeliest"), li("underdog"), li("mixed"), li("props_as_live"), li("props_as_priced")
    US, PLS = K["underdog"]["salts"]["all_leagues"]["all"], K["props_as_live"]["salts"]["all_leagues"]["all"]
    out = []

    def rec(name, s):
        if not s.get("tickets"):
            return f"{name} made no ticket"
        return (f"{name} won {s['wins']:,} of {s['tickets']:,} ({pc(s['roi'])} a ticket, {pc(s['roi_low'])} to "
                f"{pc(s['roi_high'])}; by the prices {pc(s['implied_roi'])})")
    out.append(
        "The market is hard to beat, and no v2 ticket is an edge: every leg pays the bookmaker's margin. Over "
        f"{v2['periods']['all'][0]} to {v2['periods']['all'][1]}, " + rec("LIKELIEST v2", L) + "; "
        + rec("UNDERDOG v2 at the live draw", U)
        + f", over the 1,000 simulation draws {pc(US.get('roi'))} ({pc(US.get('roi_low'))} to {pc(US.get('roi_high'))})"
        + "; " + rec("MIXED", X) + f"; and to {PROPS_PERIODS['all'][1]}, " + rec("PROPS as live", PL)
        + f", over the 1,000 simulation draws {pc(PLS.get('roi'))} ({pc(PLS.get('roi_low'))} to "
        f"{pc(PLS.get('roi_high'))}). " + edge_words(K) + " An interval that spans zero is no evidence of an edge.")

    doc, err = _method_error_text()
    me = frames["others"][frames["others"]["file"] == "legs_v2_ufc_method"]
    e = me[me["est_decimal"].notna()]
    own = float(((e["est_decimal"] - e["decimal"]).abs() / e["decimal"]).median()) if len(e) else None
    drawn = PL.get("prop_est_error_median")
    out.append(
        "Live there is no prop price on the odds feed: a live props ticket - and a live mixed ticket holding a prop "
        f"(here {sh((X.get('holds') or {}).get('METHOD', 0.0))} of MIXED's tickets held one, at the real price) - "
        "is chosen and shown at the METHOD ESTIMATOR's price, an estimate. " + (err[0].upper() + err[1:] if doc is None
                                                                                 else "Measured in engine/data/method_price.json: " + err)
        + (f". Refitted each year on earlier years only, as here, it missed the real price by a median "
           f"{own:.1%} over the history's {len(e):,} estimated props" if own is not None else "")
        + (f" and {drawn:.1%} on the props the as-live tickets drew" if drawn is not None else "")
        + f". Chosen by the estimate in {sdy.PROPS_LIVE_BAND[0]}-{sdy.PROPS_LIVE_BAND[1]}x, the as-live tickets' real "
        f"multiplier lay in {sdy.BAND[0]:g}-{sdy.BAND[1]:g}x on {sh(PL.get('real_in_band'))} of them (searched "
        f"{PL.get('searched_multiplier')}x on average, paid {PL.get('mean_multiplier')}x); a live ticket's multiplier is an "
        "estimate until a book prices the prop. " + band_years_text(K["props_as_live"].get("real_in_band_by_years")))
    PB, PBS = li("props_as_live_book_split"), K["props_as_live_book_split"]["salts"]["all_leagues"]["all"]
    out.append(
        "As live, the draw weights each prop by the shipped win chance times the ESTIMATOR's share of that corner's "
        "wins, and the search reads the estimated price - as live does, where no book splits a win chance by "
        "method. Weighted instead by the real six-way book's split, an input live never has (the first build's "
        "reading, kept beside it): " + rec("PROPS as live by the book's split", PB)
        + f", over the 1,000 simulation draws {pc(PBS.get('roi'))}; the two readings drew a different prop on "
        f"{props_differ(tickets)} both have a ticket.")

    mg = v2["margins"]
    out.append(
        f"A prop costs more than a moneyline: on the {mg['fights']:,} fights with props here the six props' "
        f"book runs a median {sh(mg['six_way_median'])} over across its six outcomes against "
        f"{sh(mg['moneyline_median'])} for the same fights' moneyline pair, so a ticket holding a prop pays the "
        f"heavier margin; PROPS as live is price-implied {pc(PL.get('implied_roi'))} a ticket against LIKELIEST "
        f"v2's {pc(L.get('implied_roi'))}. " + mixed_vs_likeliest(K)
        + f" In history a card has props only where all six were priced, make a believable book and the card is "
        f"priced at the close ({mg['card_days_with_props']:,} of {mg['ufc_card_days']:,} UFC card days "
        f"{PROPS_PERIODS['all'][0]} to {PROPS_PERIODS['all'][1]}); live every fight on a card is priced by the "
        f"estimate, so live PROPS can come on more card days than here.")

    lo_info = v2["legs"]["legs_v2_ufc_method"]
    if lo_info.get("left_out"):
        n_lo = sum(lo_info["left_out"].values())
        out.append(
            f"One price snapshot a card: the props history is the closing snapshot (method_odds.csv, the same "
            f"upstream snapshot as odds.csv's close), so the {n_lo:,} fights with a six-way book on the "
            f"{lo_info['left_out_days']:,} UFC cards priced from the opening line ({lo_info['left_out_span'][0]} to "
            f"{lo_info['left_out_span'][1]}) give no prop here (parlays/data/legs_v2_ufc_method_left_out.csv.gz): "
            f"a ticket pairing a closing prop with that card's opening moneylines could never have been bought, "
            f"and the prop's chance would mix the two snapshots.")
    elif lo_info.get("left_out_missing"):
        out.append("One price snapshot a card: the list of props left out is missing - "
                   + lo_info["left_out_missing"] + ".")

    wl = frames["winners"]
    soc = wl[wl["file"] == "legs_v2_soccer"]
    n_extra = soc.loc[soc["group"] == "extra", "league"].nunique()
    lt = [t for t in tickets[("likeliest", "all_leagues")] if in_period(t["day"], V2_PERIODS["all"])]
    with_extra = sum(any(l.get("group") == "extra" for l in t["legs"]) for t in lt)
    nfl_lines = v2["legs"]["legs_v2_nfl_lines"]
    Lx = li("likeliest", "without_extra_leagues")
    out.append(
        f"Closing prices: the {n_extra} extra leagues' history has no pre-match price, so their legs are chosen and "
        f"priced at the closing average while live they are priced pre-match ({with_extra:,} of LIKELIEST v2's "
        f"{len(lt):,} tickets hold one); NFL moneyline, spread and total legs are chosen and priced at the close, "
        f"which knows the day's news, while live they are priced when the ticket is built. Without the extra leagues "
        + rec("LIKELIEST v2", Lx) + f", on {sh(Lx.get('share_of_days'))} of days against {sh(L.get('share_of_days'))}.")

    lv = v2["legs"]["legs_v2_ufc_method"]
    no_est = sum(lv.get("est_missing", {}).values())
    first_live = min((t["day"] for t in tickets[("props_as_live", "all_leagues")]), default=None)
    me_all = frames["others"][frames["others"]["file"] == "legs_v2_ufc_method"]
    years = sorted(y for y, g in me_all.groupby(me_all["day"].str[:4]) if g["est_decimal"].isna().all())
    PE = li("props_as_priced_estimated_days")
    n_props_days, n_est_days = PP.get("days_with_legs"), PL.get("days_with_legs")
    missing = (f"{no_est:,} props have no estimate ("
               + "; ".join(f"{n:,}: {why}" for why, n in lv.get("est_missing", {}).items())
               + f"), so PROPS as live has its first ticket on {first_live}"
               + (f" and as priced is the only reading of {', '.join(years)}" if years else "")
               + (f"; {n_props_days - n_est_days:,} of the {n_props_days:,} days with props have no estimated prop, "
                  f"so the row beside the headline is as priced on the {n_est_days:,} days as live can draw - "
                  + rec("as priced on those days", PE) + " - and as priced on every props day is its own row: "
                  + rec("as priced on every props day", PP) if n_props_days is not None and n_est_days is not None
                  else "")
               + "." if no_est else f"Every prop has an estimate; PROPS as live has its first ticket on {first_live}.")
    out.append(
        f"Props history ends {sdy.PROPS_HISTORY_END} (engine/data/method_odds.csv), so PROPS is measured "
        f"{PROPS_PERIODS['all'][0]} to {PROPS_PERIODS['all'][1]} and its last-3-years period is "
        f"{PROPS_PERIODS['last_3_years'][0]} to {PROPS_PERIODS['last_3_years'][1]}; after it no history ticket "
        f"holds a prop (MIXED included), while live MIXED can take one at the estimate. " + missing)

    nl = frames["others"][frames["others"]["file"] == "legs_v2_nfl_lines"]
    ss = nfl_lines.get("side_source", {})
    out.append(
        f"NFL lines: where the two prices at a line are level the leg takes the shipped model's side (the "
        f"walk-forward's margin or total against the line) at chance 0.5 - {ss.get('SPREAD model', 0):,} spreads and "
        f"{ss.get('TOTAL model', 0):,} totals of {int((nl['bet'] == 'SPREAD').sum()):,} and "
        f"{int((nl['bet'] == 'TOTAL').sum()):,}; "
        f"{nfl_lines.get('games_with_a_total_but_no_spread_leg', 0)} games have no spread leg (a zero spread or no "
        f"price), and a push voids the leg ({nfl_lines.get('void', {}).get('SPREAD', 0)} spreads, "
        f"{nfl_lines.get('void', {}).get('TOTAL', 0)} totals).")

    out.append(
        f"UFC props are graded by the archive's method class: a prop wins only when its corner wins that way, so "
        f"on the {lv.get('fights_no_prop_won', 0)} fights won by disqualification or another unclassed method every "
        f"prop lost (some books void such a prop). " + no_winner_caveat(*V2_PERIODS["all"]))

    day_src = v2["legs"]["legs_v2_soccer"].get("day_source", {})
    timed = soc[(soc["day_source"] == "kickoff") & soc["group"].ne("extra")]
    listed = timed["key"].str.split("-").str[1].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True)
    same = float((listed == timed["day"]).mean()) if len(timed) else None
    undated = soc[soc["day_source"] == "date"]
    out.append(
        f"Soccer days: {day_src.get('kickoff', 0):,} of the {sum(day_src.values()):,} soccer winner legs take the ET "
        f"day of their kickoff and {day_src.get('date', 0):,} ("
        + ", ".join(f"{g} {n:,}" for g, n in undated["group"].value_counts().sort_index().items())
        + (f", {undated['day'].min()} to {undated['day'].max()}" if len(undated) else "")
        + ") their listed date, the kickoff time not being on file"
        + (f"; where it is, a European division's kickoff falls on its listed date in ET for {sh(same)} of "
           f"{len(timed):,} legs." if len(timed) else "."))

    rank = US.get("live_rank")
    prank = PLS.get("live_rank")
    if rank and prank:
        out.append(
            f"The live draws are the lotteries with their live salts ({sdy.DOG_SALT_V2} for UNDERDOG v2, "
            f"{sdy.PROPS_SALT} for PROPS); the simulation draws are the same lotteries with 1,000 other salts. "
            f"UNDERDOG v2's live draw {draw_word(rank)} (above {rank['roi_below']:,} of {rank['of']:,}) and PROPS as "
            f"live's {draw_word(prank)} (above {prank['roi_below']:,} of {prank['of']:,}); the 5th-95th percentile "
            f"across the draws (UNDERDOG v2 {pc(US['spread']['roi']['p5'])} to {pc(US['spread']['roi']['p95'])}, PROPS "
            f"{pc(PLS['spread']['roi']['p5'])} to {pc(PLS['spread']['roi']['p95'])} a ticket) is how much the draw "
            f"alone moves the record over the same games; it is not a confidence interval.")

    a, a1 = v2["activity"]["all_leagues"]["all"], v2["activity"]["v1"]["all"]
    out.append(
        "Expected (sameday.py) " + _doc_block("EXPECTED") + f" Measured: the four "
        f"kinds that ship made {a['tickets']:,} tickets ({a['tickets_per_day']:.2f} a day, a ticket on "
        f"{sh(a['share_of_days'])} of days) against v1's {a1['tickets']:,} ({a1['tickets_per_day']:.2f} a day); "
        f"LIKELIEST v2 on {sh(L.get('share_of_days'))} of days, MIXED on {sh(X.get('share_of_days'))}, UNDERDOG v2 on "
        f"{sh(U.get('share_of_days'))}, PROPS as live on {sh(PL.get('share_of_days_with_legs'))} of the "
        f"{PL.get('days_with_legs', 0):,} days with an estimated prop ({sh(PL.get('share_of_days'))} of all its days; "
        + no_estimate_days_text(frames) + "); by the prices "
        f"LIKELIEST v2 {pc(L.get('implied_roi'))}, MIXED {pc(X.get('implied_roi'))}, UNDERDOG v2 "
        f"{pc(U.get('implied_roi'))}, PROPS as live {pc(PL.get('implied_roi'))} and as priced "
        f"{pc(PP.get('implied_roi'))} a ticket.")
    out.append("Prior exposure (sameday.py): " + _doc_block("PRIOR EXPOSURE")
               + " The v2 records are not an out-of-sample test of these markets.")
    out.append(ou_rule_text(frames))
    out.append(
        "The PROPS lottery's key is the fight's two corners, the prop's own corner and its type (sameday."
        "method_lot_key, e.g. '" + sdy.method_lot_key(["Song Yadong", "Marlon Vera"], "Song Yadong", "Decision")
        + "'): the registered PROPS text names the corners and the type; the corner is in it so that both "
        "corners' props of one type do not share one lottery number. The draws measured here and the live draw "
        "are that function's.")
    out.append(method_json_caveats_text())
    out.append("The splits by market, sport and legs are descriptive: none can move a rule in v2.")
    return out


def band_years_text(by):
    """The as-live real-in-band share in the band's choice years and before
    them (band_years), or why it is missing."""
    if not by:
        return ""
    if by.get("missing"):
        return by["missing"][0].upper() + by["missing"][1:] + "."
    lo, hi = by["band_chosen_on"]
    a, b = by["in_choice_years"], by["before_choice_years"]
    f = lambda x: "missing (no ticket)" if x["real_in_band"] is None else f"{x['real_in_band']:.1%} of {x['tickets']:,}"
    return (f"The live band was chosen on the estimator's test cards, {lo[:4]}-{hi[:4]} (sameday.py PROPS), so the "
            f"share in {sdy.BAND[0]:g}-{sdy.BAND[1]:g}x is partly in-sample there: {f(a)} as-live tickets in "
            f"{lo[:4]}-{hi[:4]}, {f(b)} before {lo[:4]}.")


def props_differ(tickets):
    """"N of the M days": the days on which PROPS as live and the same
    lottery weighted by the real book's split both made a ticket, and how
    many of them drew a different prop."""
    a = {t["day"]: t["method_lot_key"] for t in tickets[("props_as_live", "all_leagues")]}
    b = {t["day"]: t["method_lot_key"] for t in tickets[("props_as_live_book_split", "all_leagues")]}
    both = sorted(set(a) & set(b))
    return f"{sum(a[d] != b[d] for d in both):,} of the {len(both):,} days"


def mixed_vs_likeliest(K):
    """MIXED's price-implied return against LIKELIEST v2's in every period
    and league set - the comparison word chosen from the numbers, never
    typed - with the share of MIXED's tickets that held a prop."""
    cells = [(K["mixed"]["live"][v][p].get("implied_roi"), K["likeliest"]["live"][v][p].get("implied_roi"))
             for v in V2_VARIANTS for p in V2_PERIODS]
    cells = [(x, l) for x, l in cells if x is not None and l is not None]
    if not cells:
        return "MIXED's and LIKELIEST v2's price-implied returns are missing (no ticket)."
    above, below = sum(x > l for x, l in cells), sum(x < l for x, l in cells)
    n = len(cells)
    if above == n:
        word = f"above LIKELIEST v2's in all {n} periods and league sets"
    elif below == n:
        word = f"below LIKELIEST v2's in all {n} periods and league sets"
    else:
        word = f"above LIKELIEST v2's in {above} of {n} periods and league sets and below it in {below}"
    X, L = K["mixed"]["live"]["all_leagues"]["all"], K["likeliest"]["live"]["all_leagues"]["all"]
    held = (X.get("holds") or {}).get("METHOD", 0.0)
    return (f"MIXED's price-implied return ({_pct(X['implied_roi'], 1)} over all days, every league, against "
            f"{_pct(L['implied_roi'], 1)}) is {word}: only {held:.1%} of its tickets held a prop, so most MIXED "
            f"tickets do not pay the props' margin.")


def no_estimate_days_text(frames):
    """The props days with no estimated prop, counted with the reason."""
    me = frames["others"][frames["others"]["file"] == "legs_v2_ufc_method"]
    est = me["est_decimal"].notna() & me["est_p"].notna()
    days = me.groupby("day").apply(lambda g: bool((g["est_decimal"].notna() & g["est_p"].notna()).any()),
                                   include_groups=False)
    none = sorted(days.index[~days.values])
    if not none:
        return "every day with props has an estimated prop"
    why = me[me["day"].isin(none) & ~est]["est_missing"].value_counts()
    return (f"the other {len(none):,} days with props, {none[0]} to {none[-1]}, have none: "
            + "; ".join(f"{why_}" for why_ in why.index))


SOCCER_ENGINE = ROOT / "soccer" / "engine"


def ou_rules():
    """The soccer leagues' rule (c) facts, read from soccer_leagues in its own
    process (the soccer engine has a package named experiments, as the UFC
    engine on this process's path does): per league in the app its group,
    whether it ships over 2.5 from the market, whether the market numbers
    were measured, the periods and the over-2.5 caveat. (None, reason) when
    it cannot be read."""
    code = ("import json, soccer_leagues as slg\n"
            "S = slg.SHIPPED\n"
            "print(json.dumps({c: {'group': S[c]['group'], 'o25': 'o25' not in slg.withheld(c, 'market'),\n"
            "    'market_numbers': (S[c].get('numbers') or {}).get('market') is not None,\n"
            "    'dev': S[c]['measured']['dev']['seasons'], 'confirm': S[c]['measured']['confirm']['seasons'],\n"
            "    'caveats': [x for x in slg.caveats(c) if 'over 2.5' in x and 'market' in x]}\n"
            "    for c in slg.in_app()}))")
    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=SOCCER_ENGINE, check=True, capture_output=True,
                             text=True, timeout=300)
        return json.loads(run.stdout), None
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return None, f"soccer_leagues could not be read ({type(e).__name__})"


def ou_rule_text(frames):
    """Caveat: the OU25 league set is soccer_leagues rule (c)'s, which read
    results over the years measured here - so, beside the registered prior-
    exposure text, the over/under legs' leagues are not outcome-free."""
    rules, why = ou_rules()
    if rules is None:
        return f"OU25's leagues are soccer_leagues rule (c)'s; its facts are missing here - {why}."
    ou = frames["others"][frames["others"]["file"] == "legs_v2_ou25"]
    on = sorted(c for c, r in rules.items() if r["o25"])
    if set(on) != set(ou["league"]):
        return ("OU25's leagues are soccer_leagues rule (c)'s, but the over/under legs file's leagues differ from "
                "it now: rebuild legs_history.py --v2 --only soccer.")
    periods = sorted({(tuple(rules[c]["dev"]), tuple(rules[c]["confirm"])) for c in on})
    per = "; ".join(f"{d[0]}-{d[1]} and {c[0]}-{c[1]}" for d, c in periods)
    rule_out = [c for c, r in rules.items() if not r["o25"] and r["market_numbers"]]
    unmeasured = [c for c, r in rules.items() if not r["o25"] and not r["market_numbers"]]
    return (f"OU25's leagues were chosen on results: a league offers the over/under leg only where it ships over "
            f"2.5 from the market (soccer_leagues rule (c)), and it does so only where the market's over/under beat "
            f"the league's base rate in log loss in both {per} - a measure that read the results of the years "
            f"measured here. {len(on)} of the {len(rules)} leagues in the app pass. Rule (c) left out "
            + (", ".join(f"{c} ({' '.join(rules[c]['caveats']) or 'withheld by rule (c)'})" for c in rule_out)
               if rule_out else "no measured league")
            + (f"; the {len(unmeasured)} " + ("extra leagues" if all(rules[c]['group'] == 'extra' for c in unmeasured)
                                              else "other leagues")
               + " ship no number from the market's prices" if unmeasured else "")
            + ". The prior-exposure text above says the new markets were chosen on the scouts' outcome-free counts; "
              "OU25's league set was not.")


def method_json_caveats_text():
    """engine/data/method_price.json's own caveats, quoted (they reach the
    v2 page only through here), or why they are missing."""
    try:
        cs = json.loads(METHOD_JSON.read_text())["caveats"]
    except (OSError, ValueError, KeyError) as e:
        return f"The estimator's own caveats are missing - engine/data/method_price.json ({type(e).__name__})."
    return "The estimator's own caveats (engine/data/method_price.json): " + " ".join(
        c if c.endswith(".") else c + "." for c in cs)


def edge_words(K):
    """Whether any shipped kind's 95% interval over days lies wholly above
    zero (all leagues, all days) - the sentence is the numbers', never
    typed: "None of ..." only when none does."""
    names = {"likeliest": "LIKELIEST v2", "underdog": "UNDERDOG v2", "mixed": "MIXED", "props_as_live": "PROPS as live"}
    above, below = [], []
    for k in V2_SHIPPED:
        s = K[k]["live"]["all_leagues"]["all"]
        if s.get("tickets") and s["roi_low"] > 0:
            above.append(names[k])
        if s.get("tickets") and s["roi_high"] < 0:
            below.append(names[k])
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    head = ("None of the four kinds' 95% intervals lies above zero" if not above else
            f"{join(above)}'s 95% interval lies above zero over these days; with the caveats below that is not "
            f"an established edge")
    return head + (f"; {join(below)}'s lies wholly below it." if below else ".")


def v2_headline(v2):
    """PROPS as live, and beside it the rows that compare with it on the same
    days (as priced, and the same lottery weighted by the real book's split),
    then as priced over every props day as its own labelled row - a row over
    other days (the 2016 cards have no estimate) would read as the
    estimate's cost."""
    K = v2["kinds"]
    rd = lambda k: K[k]["live"]["all_leagues"]["all"]
    sa = lambda k: K[k]["salts"]["all_leagues"]["all"]
    pl, ps = rd("props_as_live"), sa("props_as_live")
    pe, pes = rd("props_as_priced_estimated_days"), sa("props_as_priced_estimated_days")
    pb, pbs = rd("props_as_live_book_split"), sa("props_as_live_book_split")
    pr, prs = rd("props_as_priced"), sa("props_as_priced")
    pc = lambda x: "missing" if x is None else _pct(x, 1)
    if not pl.get("tickets"):
        return "PROPS as live made no ticket."

    def row(name, s, st):
        body = ("no ticket" if not s.get("tickets") else
                f"{s['tickets']:,} tickets, {s['wins']:,} won, {pc(s['roi'])} a ticket ({pc(s['roi_low'])} to "
                f"{pc(s['roi_high'])}), over the 1,000 simulation salts {pc(st.get('roi'))}, price-implied "
                f"{pc(s['implied_roi'])}")
        return f"{name}: {body}" if name else body
    return (f"PROPS as live (the headline), {K['props_as_live']['periods']['all'][0]} to "
            f"{K['props_as_live']['periods']['all'][1]}, on the {pl.get('days_with_legs', 0):,} days with an "
            f"estimated prop: {pl['tickets']:,} tickets at the live salt, {pl['wins']:,} "
            f"won ({pl['hit_rate']:.1%}; the prices said {pl['market_hit_rate']:.1%}), {pc(pl['roi'])} a ticket "
            f"({pc(pl['roi_low'])} to {pc(pl['roi_high'])}), price-implied {pc(pl['implied_roi'])}; over the 1,000 "
            f"simulation salts {pc(ps.get('roi'))} ({pc(ps.get('roi_low'))} to {pc(ps.get('roi_high'))}). On the same "
            f"days, " + row("as priced", pe, pes) + "; " + row("as live weighted by the real book's split (an input "
                                                              "live never has)", pb, pbs)
            + f". As priced on every one of the {pr.get('days_with_legs', 0):,} days with props, "
              f"{K['props_as_priced']['periods']['all'][0]} to {K['props_as_priced']['periods']['all'][1]} (the "
              f"cards without an estimate included): " + row(None, pr, prs) + ".")


def _leg_text(l):
    bet = l.get("bet") or "1X2"
    what = l["pick"] if bet == "1X2" else (f"{l['pick']} by {l['bet_detail']}" if bet == "METHOD"
                                           else l.get("bet_detail") or l["pick"])
    est = f"~{l['est_decimal']:.3f}" if l.get("est_decimal") is not None else ""
    return f"{l['sport']}:{bet}:{what}@{l['decimal']:.3f}{est}"


def write_v2_tickets(tickets):
    """parlays/data/sameday_v2_tickets.csv.gz: every v2 ticket (live salts),
    kind and league set, with its legs' keys and markets - what the tests
    rebuild a sample of."""
    rows = []
    for (k, v), ts in sorted(tickets.items()):
        for t in ts:
            st = sdy.settle(t)
            rows.append({"kind": k, "variant": v, "day": t["day"], "n_legs": t["n_legs"],
                         "sports": "+".join(t["sports"]), "bets": "+".join(bets_of(t)),
                         "multiplier": round(t["multiplier"], 4),
                         "searched_multiplier": round(t["searched_multiplier"], 4) if "searched_multiplier" in t else None,
                         "p": round(t["p"], 6), "p_market": round(t["p_market"], 6), "status": st["status"],
                         "profit": st["profit"], "keys": "|".join(l["key"] for l in t["legs"]),
                         "markets": "|".join(f"{l.get('bet') or '1X2'}:{l.get('bet_detail')}:{l['pick']}"
                                             for l in t["legs"]),
                         "legs": "|".join(_leg_text(l) for l in t["legs"])})
    rows.sort(key=lambda r: (r["kind"], r["variant"], r["day"]))
    pd.DataFrame(rows).to_csv(DATA / "sameday_v2_tickets.csv.gz", index=False,
                              compression={"method": "gzip", "mtime": 0})


def table_v2(v2):
    head = (f"  {'kind / league set / period':<44}{'tick':>6}{'wins':>6}{'hit':>7}{'model':>7}{'market':>7}"
            f"{'mult':>7}{'legs':>6}{'ROI':>8}  [95% interval]  implied  days%")
    print(f"\n  TICKETS v2\n{head}")
    for k in V2_KINDS:
        b = v2["kinds"][k]
        for v in V2_VARIANTS:
            for p in b["live"][v]:
                s = b["live"][v][p]
                lbl = f"{k} {'all' if v == 'all_leagues' else 'no-extra'} {p}"
                if s.get("tickets"):
                    print(_line(lbl, s) + f"  {s['share_of_days']:.1%}")
                else:
                    print(f"  {lbl:<44}     0")
                if k in V2_LOTTERIES and b["salts"][v][p].get("tickets_per_salt"):
                    st = b["salts"][v][p]
                    sp = st["spread"]
                    print(f"  {'    1,000 salts':<44}{st['tickets_per_salt']:>6}{'':>6}{st['hit_rate']:>7.1%}"
                          f"  ROI {st['roi']:+.3f} [{st['roi_low']:+.3f}, {st['roi_high']:+.3f}]  implied "
                          f"{st['implied_roi']:+.3f}  draw alone p5/p50/p95 {sp['roi']['p5']:+.3f}/"
                          f"{sp['roi']['p50']:+.3f}/{sp['roi']['p95']:+.3f}; live above "
                          f"{st['live_rank']['roi_below']} of {st['live_rank']['of']}")
                if "market_only" in b:
                    print(_line(f"    market-only control", b["market_only"][v][p]))
        print()
    for v in V2_VARIANTS + ("v1",):
        for p, a in v2["activity"][v].items():
            print(f"  activity {v:<22} {p:<13} {a['tickets']:>6} tickets  {a['tickets_per_day']:.3f}/day"
                  + (f"  a ticket on {a['share_of_days']:.1%} of days  {a['by_kind']}" if "share_of_days" in a else ""))
    for k in V2_KINDS:
        sp = v2["kinds"][k]["splits"]
        print(f"\n  {k} splits ({V2_SPLIT_NOTE}):")
        for name in ("by_bets", "by_sports", "by_legs"):
            for g, s in sp[name].items():
                if g != "note":
                    print("  " + _line(f"{name[3:]} {g}", s)[:150])
        for name in ("legs_by_bet", "legs_by_sport"):
            for g, s in sp[name].items():
                if g != "note":
                    print(f"    legs {g:<10} {s['legs']:>6} hit {s['hit_rate'] if s['hit_rate'] is None else round(s['hit_rate'], 3)}"
                          f"  p {s['mean_p']:.3f}  market {s['mean_p_market']:.3f}  @{s['mean_decimal']:.2f}  single "
                          f"{s['single_roi']:+.3f}  implied {s['single_implied_roi']:+.3f}")
    print("\n  " + v2["headline"])
    for c in v2["caveats"]:
        print("  - " + c)


README = DATA.parent / "README.md"
README_V2_START = "<!-- v2 figures: written by engine/experiments/sameday_backtest.py --v2 from data/sameday_backtest.json"
README_V2_END = "<!-- end of the v2 figures -->"


def readme_v2(v2):
    """The README's v2 paragraphs, every figure and result-dependent word
    from the json's "v2" block. WHY GENERATED: a re-run of the backtest
    rewrites them, so the README cannot go stale against the json (a test
    holds them to it)."""
    pc = lambda x: "missing" if x is None else _pct(x, 1)
    K = v2["kinds"]
    rd = lambda k: K[k]["live"]["all_leagues"]["all"]
    L, U, X = rd("likeliest"), rd("underdog"), rd("mixed")
    PL, PE, PB, PP = rd("props_as_live"), rd("props_as_priced_estimated_days"), rd("props_as_live_book_split"), \
        rd("props_as_priced")
    US, PS = K["underdog"]["salts"]["all_leagues"]["all"], K["props_as_live"]["salts"]["all_leagues"]["all"]
    a, a1 = v2["activity"]["all_leagues"]["all"], v2["activity"]["v1"]["all"]
    mg = v2["margins"]
    lo, hi = v2["periods"]["all"]
    plo, phi = v2["props_periods"]["all"]
    nu, npr = (f"{v2['rule']['sim_salts'][k]:,}" for k in ("underdog", "props"))
    rank = US.get("live_rank")
    prank = PS.get("live_rank")
    draws = (f" The live draws are one draw each: UNDERDOG v2's {draw_word(rank)} (above {rank['roi_below']:,} of "
             f"{rank['of']:,} simulation draws), PROPS as live's {draw_word(prank)} (above {prank['roi_below']:,} of "
             f"{prank['of']:,})." if rank and prank else "")
    return "\n".join([
        README_V2_START + " (key \"v2\"); a re-run rewrites them - do not edit by hand -->",
        "**Tickets v2** (fixed in `engine/sameday.py`'s docstring, \"TICKETS v2\", on 2026-11-14 before any of "
        "their results; asked for to make the builder more active and more diverse): four tickets a day, each "
        "3-5x and logged and graded on its own, from a wider pool - soccer's shipped pick from every league in the "
        "app, over/under 2.5 goals where a league ships it from the market, NFL spreads and totals, and UFC method "
        "props (a corner wins by KO/TKO, by submission or by decision). LIKELIEST v2 is v1's rule over the wider "
        f"winner pool; UNDERDOG v2 is v1's underdog lottery with its own salt (`{sdy.DOG_SALT_V2}`); MIXED is the "
        "likeliest combination that holds at least one leg other than a winner pick; PROPS (UFC card days only) "
        f"is one method prop drawn by a lottery (`{sdy.PROPS_SALT}`) plus the likeliest fill. Live no odds feed "
        "prices a prop, so a prop leg carries the METHOD ESTIMATOR's price (`engine/method_price.py`), marked as "
        "an estimate, and a ticket holding one shows its multiplier as an estimate.",
        "",
        f"**Measured** (`data/sameday_backtest.json`, under `v2`) every day from {lo} to {hi}, point in time: the "
        f"four kinds that ship made {a['tickets']:,} tickets ({a['tickets_per_day']:.2f} a day, a ticket on "
        f"{a['share_of_days']:.1%} of days) against v1's {a1['tickets']:,} ({a1['tickets_per_day']:.2f} a day). "
        f"LIKELIEST v2 {L['tickets']:,} tickets, {L['wins']:,} won, {pc(L['roi'])} a ticket ({pc(L['roi_low'])} to "
        f"{pc(L['roi_high'])}), by the prices {pc(L['implied_roi'])}; "
        f"MIXED {X['tickets']:,}, {pc(X['roi'])} ({pc(X['roi_low'])} to {pc(X['roi_high'])}), by the prices "
        f"{pc(X['implied_roi'])}; "
        f"UNDERDOG v2 {U['tickets']:,} at the live draw, {pc(U['roi'])}; over the {nu} simulation draws "
        f"{pc(US['roi'])} ({pc(US['roi_low'])} to {pc(US['roi_high'])}), by the prices {pc(U['implied_roi'])}. "
        f"Props history ends {v2['rule']['props_history_end']}, so PROPS is measured {plo} to {phi}, \"as live\" - "
        f"drawn and chosen by the "
        f"estimator's chance and price in the live band, as live does, and paid at the real prop: "
        f"PROPS as live (the headline) {PL['tickets']:,} tickets, {PL['wins']:,} won, {pc(PL['roi'])} a ticket "
        f"({pc(PL['roi_low'])} to {pc(PL['roi_high'])}); over the {npr} simulation draws {pc(PS['roi'])}; by the "
        f"prices {pc(PL['implied_roi'])}; on the same days as priced (chosen and paid at the real props) "
        f"{pc(PE['roi'])}, by the prices {pc(PE['implied_roi'])}, and as live weighted by the real book's split "
        f"(an input live never has) {pc(PB.get('roi'))}; as priced on every one of the "
        f"{PP.get('days_with_legs', 0):,} days with props (the cards without an estimate included) "
        f"{pc(PP['roi'])}, by the prices {pc(PP['implied_roi'])}." + draws,
        "",
        edge_words(K) + f" No v2 ticket is an edge: every leg pays the bookmaker's margin - a prop's book a median "
        f"{mg['six_way_median']:.1%} across its six outcomes against {mg['moneyline_median']:.1%} for the same "
        f"fights' moneyline pair - and the market is hard to beat. The json's {len(v2['caveats'])} v2 caveats say "
        "what these records leave out or cannot know (the extra leagues and the NFL lines priced at the close, "
        "the estimate's error, the end of the props history and the cards without an estimate, the over/under "
        "league set chosen by a rule that read results, the prior exposure, among others); every one stays, the "
        "page shows them, and the figures here are that file's - `engine/tests/test_v2_measure.py` holds them "
        "to it.",
        README_V2_END])


def write_readme_v2(v2, path=README):
    """Replace the README's v2 block (between its markers) with readme_v2;
    the markers must be there - the paragraphs' place in the README is a
    person's choice, never the script's."""
    text = path.read_text()
    a, b = text.find(README_V2_START), text.find(README_V2_END)
    if a < 0 or b < a:
        raise SystemExit(f"{path}: no v2 block markers ({README_V2_START!r} ... {README_V2_END!r})")
    path.write_text(text[:a] + readme_v2(v2) + text[b + len(README_V2_END):])


def main_v2():
    """Measure v2 alone and put it under the json's "v2" key, every other
    key left as it is (they are v1's, measured before v2 existed); then the
    README's v2 paragraphs from it."""
    path = DATA / "sameday_backtest.json"
    payload = json.loads(path.read_text())
    payload["v2"] = measure_v2(payload)
    path.write_text(json.dumps(payload, indent=1, default=_json, allow_nan=False))
    write_readme_v2(payload["v2"])
    table_v2(payload["v2"])


if __name__ == "__main__":
    if "--v2" in sys.argv[1:]:
        main_v2()
    else:
        main()
