"""Parlay profiles: three tickets a card, each aimed at a payout band.

WHAT THIS IS. A parlay multiplies its legs' prices and needs every leg to
land. This module turns one card's priced fights into three tickets aimed
at the bands a bettor asked for - about 5x, about 10x and 50-100x - each by
its own PROFILE, plus a MIXED profile and a MARKET-ONLY control at every
band. The live card (app_export.parlay_profiles_payload) and the 3-year
backtest (experiments/parlay_profiles.py) both call build() below, so what
the app shows is what was measured.

THE PROFILES - fixed here, before any backtest number was seen, and never
tuned on the 3-year window they are scored on (the experiment's docstring
has the protocol):

  steady      ~5x  (band 4.5-6.5)   favourites only: every leg the market
                                    favourite, the model agreeing, blended
                                    probability >= 0.60
  risker      ~10x (band 8-14)      one or two RISKY legs - a price of 2.0 or
                                    longer that the model rates at least 3
                                    points above the market (blended >= 0.35)
                                    - filled with steady legs
  risk_lover  50-100x (band 40-120) one or two LONG legs - an underdog at 2.5
                                    or longer the model rates at least as high
                                    as the market - on top of two or more
                                    confident favourites (blended >= 0.65).
                                    Each long leg carries its likeliest route
                                    (KO or submission) from the fighter's own
                                    UFC record as the reason.
  mixed       every band            any legs from all three pools, at least
                                    one the model rates above the market
  market_only every band            any legs, chosen by the market alone - the
                                    control that says whether the model adds
                                    anything

Within its pool and band, every profile takes the combination with the
highest combined probability (the product of its legs' blended chances):
the likeliest ticket that pays what the band asks. One leg per fight. A
profile with nothing in its pool, or no combination inside its band, gives
no ticket - a card that cannot honestly make 50x shows no 50x ticket.

WHY THE LONG LEG IS A MONEYLINE. Method props (KO / submission) are not on
the live odds plan and their history stops at 2024-12-07; their six-way
margin is about 22% against about 4% on a moneyline, and flat-betting every
underdog-KO prop 2012-2024 returned -24% (underdog submission -52%). So the
risk lover's long leg is priced as the underdog's moneyline - which pays
whatever route the upset takes - and the route is the reason, not the bet.
The backtest measures a method variant (risk_lover_method) where props were
priced, to say what swapping in the prop would have done.

NOTHING HERE IS AN EDGE. The market is hard to beat in MMA: flat-betting the
model's picks returned -0.2% (between -3.9% and +3.4%) from 2020. A ticket's
"combined probability" is the model's (blended) belief, and the measured
record beside it is what such tickets actually returned.
"""

from __future__ import annotations

import itertools
import math
import re

import numpy as np

# ---------------------------------------------------------------- profiles

PROFILES = {
    "steady": {
        "label": "Steady", "target": "~5x", "band": (4.5, 6.5),
        "blurb": "Favourites only, stacked until the payout reaches about 5x.",
    },
    "risker": {
        "label": "Risker", "target": "~10x", "band": (8.0, 14.0),
        "blurb": "One or two longer prices the model rates above the market, filled with safe favourites.",
    },
    "risk_lover": {
        "label": "Risk lover", "target": "50-100x", "band": (40.0, 120.0),
        "blurb": "Confident favourites plus one or two underdogs with a finishing route to an upset.",
    },
}
ORDER = ("steady", "risker", "risk_lover")
BANDS = {p: PROFILES[p]["band"] for p in ORDER}

SAFE_MIN_P = 0.60            # steady legs: blended probability
CORE_MIN_P = 0.65            # the risk lover's favourites
RISKY_MIN_DECIMAL = 2.0
RISKY_MIN_EDGE = 0.03        # model minus de-vigged market
RISKY_MIN_P = 0.35
LONG_MIN_DECIMAL = 2.5
LONG_MIN_EDGE = 0.0
MAX_LEGS = 7
POOL_CAP = 10                # legs kept per pool (by probability) for the search
MIXED_MIN_P = 0.30

STAKE = 1.0


# ---------------------------------------------------------------- prices

def decimal_from_american(american):
    """+150 -> 2.5, -200 -> 1.5; None/NaN/0 -> None."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or a == 0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def american_from_decimal(decimal):
    """2.5 -> +150, 1.5 -> -200 (rounded); None when not a price."""
    if decimal is None or not math.isfinite(decimal) or decimal <= 1.0:
        return None
    return int(round((decimal - 1.0) * 100)) if decimal >= 2.0 else int(round(-100.0 / (decimal - 1.0)))


def devig(dec_a, dec_b, max_margin=0.15):
    """The two sides' chances with the margin removed in proportion; None
    when either price is missing or the pair is not a believable book
    (an overround below 0 or above max_margin)."""
    if not dec_a or not dec_b or dec_a <= 1.0 or dec_b <= 1.0:
        return None
    ia, ib = 1.0 / dec_a, 1.0 / dec_b
    over = ia + ib - 1.0
    if over < 0.0 or over > max_margin:
        return None
    return ia / (ia + ib), ib / (ia + ib)


# ---------------------------------------------------------------- the legs

def fight_legs(fight):
    """Both moneyline sides of one fight as legs.

    fight: {"key", "event", "date", "red", "blue", "p_model" (red wins, the
    model), "p_blend" (red wins, blended; optional - rebuilt from the market
    at the shipped weight when absent), "dec_red", "dec_blue" (decimal
    prices), optional "facts": {"red": {...}, "blue": {...}} (see facts()),
    optional "ko_red", "ko_blue", "sub_red", "sub_blue" (decimal method
    prices), optional "result": {"winner": "red"|"blue"|None, "method":
    "KO/TKO"|"Submission"|"Decision"|None}}.
    A side without a price gives no leg."""
    pm_red = fight.get("p_model")
    if pm_red is None or not np.isfinite(pm_red):
        return []
    mk = devig(fight.get("dec_red"), fight.get("dec_blue"))
    pb_red = fight.get("p_blend")
    if pb_red is None or not np.isfinite(pb_red):
        if mk is None:
            return []
        import market_blend
        pb_red = market_blend.blend(pm_red, mk[0])
    legs = []
    for side, other in (("red", "blue"), ("blue", "red")):
        dec = fight.get(f"dec_{side}")
        if not dec or dec <= 1.0:
            continue
        pm = pm_red if side == "red" else 1.0 - pm_red
        pb = pb_red if side == "red" else 1.0 - pb_red
        pk = None if mk is None else (mk[0] if side == "red" else mk[1])
        legs.append({
            "key": fight["key"], "event": fight.get("event"), "date": fight.get("date"),
            "number": fight.get("number"),
            "side": side, "pick": fight[side], "opponent": fight[other],
            "market": "moneyline", "decimal": float(dec),
            "p": float(pb), "p_model": float(pm), "p_market": None if pk is None else float(pk),
            "edge": None if pk is None else float(pm - pk),
            "favourite": pk is not None and pk >= 0.5,
            "facts": (fight.get("facts") or {}).get(side) or {},
            "opp_facts": (fight.get("facts") or {}).get(other) or {},
            "method_prices": {"KO/TKO": fight.get(f"ko_{side}"), "Submission": fight.get(f"sub_{side}")},
            "result": fight.get("result"),
        })
    return legs


def _safe(l, min_p=SAFE_MIN_P):
    return (l["favourite"] and l["p_market"] is not None and l["p_model"] >= 0.5
            and l["p"] >= min_p)


def _risky(l):
    return (l["edge"] is not None and l["decimal"] >= RISKY_MIN_DECIMAL
            and l["edge"] >= RISKY_MIN_EDGE and l["p"] >= RISKY_MIN_P)


def _long(l):
    return (l["edge"] is not None and not l["favourite"] and l["decimal"] >= LONG_MIN_DECIMAL
            and l["edge"] >= LONG_MIN_EDGE)


def route(facts):
    """A fighter's likeliest finishing route from their own UFC wins:
    "KO/TKO" or "Submission", or None with no finish on record."""
    ko, sub = int(facts.get("ko_wins") or 0), int(facts.get("sub_wins") or 0)
    if ko == 0 and sub == 0:
        return None
    return "KO/TKO" if ko >= sub else "Submission"


# ---------------------------------------------------------------- the search

def _top(legs, n=POOL_CAP, key="p"):
    return sorted(legs, key=lambda l: -l[key])[:n]


def _search(fixed_pools, band, objective="p", require=None, max_legs=MAX_LEGS):
    """The combination with the highest product of `objective` whose
    multiplier lies inside `band`. fixed_pools: [(pool, min_count,
    max_count), ...] - legs are drawn from each pool within its counts, one
    leg per fight across all pools. require(legs) -> bool filters
    combinations. None when nothing qualifies."""
    lo, hi = band
    best, best_val = None, -1.0
    choices = []
    for pool, kmin, kmax in fixed_pools:
        opts = []
        for k in range(kmin, min(kmax, len(pool)) + 1):
            opts.extend(itertools.combinations(pool, k))
        choices.append(opts)
    for combo in itertools.product(*choices):
        legs = [l for part in combo for l in part]
        if not legs or len(legs) > max_legs:
            continue
        keys = [l["key"] for l in legs]
        if len(set(keys)) != len(keys):
            continue
        mult = float(np.prod([l["decimal"] for l in legs]))
        if not (lo <= mult <= hi):
            continue
        if require is not None and not require(legs):
            continue
        val = float(np.prod([l[objective] if l[objective] is not None else 0.0 for l in legs]))
        if val > best_val + 1e-15:
            best, best_val = legs, val
    return best


def build(card_legs, profile, band=None):
    """One ticket for `profile` from a card's legs (fight_legs over every
    fight), or None. Profiles: steady, risker, risk_lover, mixed, market_only
    (mixed and market_only need `band`); risk_lover_method is the backtest's
    variant with the long legs priced as KO/submission props."""
    legs = [l for l in card_legs if l["decimal"] and l["decimal"] > 1.0]
    if profile == "steady":
        pool = _top([l for l in legs if _safe(l)])
        picked = _search([(pool, 1, MAX_LEGS)], band or BANDS["steady"])
    elif profile == "risker":
        risky = _top([l for l in legs if _risky(l)], 4, key="edge")
        safe = _top([l for l in legs if _safe(l)])
        picked = _search([(risky, 1, 2), (safe, 0, MAX_LEGS - 1)], band or BANDS["risker"], max_legs=6)
    elif profile in ("risk_lover", "risk_lover_method"):
        longs = _top([l for l in legs if _long(l)], 4, key="edge")
        if profile == "risk_lover_method":
            longs = [m for m in (_as_method(l) for l in longs) if m is not None]
        core = _top([l for l in legs if _safe(l, CORE_MIN_P)])
        picked = _search([(longs, 1, 2), (core, 2, MAX_LEGS - 1)], band or BANDS["risk_lover"])
    elif profile == "mixed":
        pool = _top([l for l in legs if l["p"] >= MIXED_MIN_P and l["p_market"] is not None], 12)
        picked = _search([(pool, 1, MAX_LEGS)], band,
                         require=lambda ls: any((l["edge"] or 0) > 0 for l in ls))
    elif profile == "market_only":
        pool = _top([l for l in legs if l["p_market"] is not None and l["p_market"] >= MIXED_MIN_P], 12, key="p_market")
        picked = _search([(pool, 1, MAX_LEGS)], band, objective="p_market")
    else:
        raise ValueError(f"unknown profile {profile!r}")
    if not picked:
        return None
    return ticket(picked, profile, band or BANDS.get(profile))


def _as_method(leg):
    """The long leg as its likeliest-route prop, where that prop is priced."""
    r = route(leg["facts"])
    price = (leg.get("method_prices") or {}).get(r) if r else None
    if not price or price <= 1.0:
        return None
    m = dict(leg, market=r, decimal=float(price))
    return m


def ticket(legs, profile, band):
    legs = sorted(legs, key=lambda l: (l.get("number") is None, l.get("number") or 0, -l["p"]))
    p_mk = [l["p_market"] for l in legs]
    return {
        "profile": profile, "band": list(band) if band else None,
        "legs": legs, "n_legs": len(legs),
        "multiplier": float(np.prod([l["decimal"] for l in legs])),
        "p": float(np.prod([l["p"] for l in legs])),
        "p_market": None if any(x is None for x in p_mk) else float(np.prod(p_mk)),
        "event": legs[0].get("event"), "date": legs[0].get("date"),
    }


# ---------------------------------------------------------------- settlement

def leg_outcome(leg):
    """"won" / "lost" / "void" (draw, no contest, bout off) / None (not yet
    known). A method leg wins only when the pick wins by that method."""
    res = leg.get("result")
    if not res:
        return None
    w = res.get("winner")
    if w is None:
        return "void"
    if w != leg["side"]:
        return "lost"
    if leg["market"] == "moneyline":
        return "won"
    return "won" if res.get("method") == leg["market"] else "lost"


def settle(t):
    """{"status": won/lost/void/open, "paid": multiplier actually paid (void
    legs count 1.0), "profit": units on a 1-unit stake}. Any lost leg loses
    the ticket; every leg void returns the stake."""
    outs = [leg_outcome(l) for l in t["legs"]]
    if "lost" in outs:
        return {"status": "lost", "paid": 0.0, "profit": -STAKE}
    if None in outs:
        return {"status": "open", "paid": None, "profit": None}
    paid = float(np.prod([l["decimal"] for l, o in zip(t["legs"], outs) if o == "won"] or [1.0]))
    if all(o == "void" for o in outs):
        return {"status": "void", "paid": 1.0, "profit": 0.0}
    return {"status": "won", "paid": paid, "profit": STAKE * (paid - 1.0)}


def summarise(tickets, n_boot=4000, seed=7):
    """Flat 1-unit record of settled tickets: count, wins, hit rate, what the
    model expected (mean combined probability), mean multiplier, ROI and a
    bootstrap 95% interval over tickets (one ticket a card per profile, so
    this resamples cards)."""
    rows = [(t, settle(t)) for t in tickets]
    rows = [(t, s) for t, s in rows if s["status"] in ("won", "lost", "void")]
    n = len(rows)
    if n == 0:
        return {"tickets": 0}
    profit = np.array([s["profit"] for _, s in rows], dtype=float)
    wins = sum(s["status"] == "won" for _, s in rows)
    rng = np.random.default_rng(seed)
    boots = rng.choice(profit, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "tickets": n, "wins": int(wins), "hit_rate": round(wins / n, 4),
        "expected_hit_rate": round(float(np.mean([t["p"] for t, _ in rows])), 4),
        "market_hit_rate": (round(float(np.mean([t["p_market"] for t, _ in rows])), 4)
                            if all(t["p_market"] is not None for t, _ in rows) else None),
        "mean_multiplier": round(float(np.mean([t["multiplier"] for t, _ in rows])), 2),
        "mean_legs": round(float(np.mean([t["n_legs"] for t, _ in rows])), 2),
        "profit": round(float(profit.sum()), 2), "roi": round(float(profit.mean()), 4),
        "roi_low": round(float(lo), 4), "roi_high": round(float(hi), 4),
        "biggest_win": round(float(profit.max()), 2),
    }


# ---------------------------------------------------------------- reasons

def facts(history, name, before):
    """A fighter's UFC record strictly before `before`, from an archive frame
    with r_name, b_name, date, winner, method: wins, losses, KO/TKO and
    submission wins and losses, and the last five results ("W"/"L")."""
    d = history[(history["date"] < str(before)[:10])
                & ((history["r_name"] == name) | (history["b_name"] == name))]
    d = d.sort_values("date")
    out = {"fights": int(len(d)), "wins": 0, "losses": 0, "ko_wins": 0, "sub_wins": 0,
           "ko_losses": 0, "sub_losses": 0, "last5": []}
    for _, r in d.iterrows():
        m = method_class(r.get("method"))
        w = r.get("winner")
        if not isinstance(w, str) or not w:
            continue
        won = w == name
        out["wins" if won else "losses"] += 1
        if m == "KO/TKO":
            out["ko_wins" if won else "ko_losses"] += 1
        elif m == "Submission":
            out["sub_wins" if won else "sub_losses"] += 1
        out["last5"].append("W" if won else "L")
    out["last5"] = out["last5"][-5:]
    return out


def method_class(text):
    """Archive method text -> "KO/TKO" / "Submission" / "Decision" / None
    (overturned, no contest, DQ, could not continue). Unlike grade's, this
    reads the short labels too (U-DEC, S-DEC, M-DEC, SUB ..., KO/TKO ...)."""
    if not isinstance(text, str):
        return None
    t = text.lower()
    if "dec" in t:
        return "Decision"
    if re.search(r"\bsub", t):
        return "Submission"
    if "ko" in t:
        return "KO/TKO"
    return None


def _pct(x):
    return f"{100 * x:.0f}%"


def reason(leg):
    """One line, at most about 90 characters, built only from the leg's own
    numbers - never a claim the data does not hold. Gender-neutral."""
    f, o = leg.get("facts") or {}, leg.get("opp_facts") or {}
    pm, pk = leg["p_model"], leg["p_market"]
    form = "".join(f.get("last5") or [])
    if leg["market"] != "moneyline":
        r = leg["market"]
        n = f.get("ko_wins" if r == "KO/TKO" else "sub_wins", 0)
        return f"{n} of {f.get('wins', 0)} UFC wins by {'KO' if r == 'KO/TKO' else 'submission'}; priced as the prop"
    if not leg["favourite"]:
        r = route(f)
        bits = [f"Underdog the model rates {_pct(pm)} vs market {_pct(pk)}" if pk is not None else f"Underdog, model {_pct(pm)}"]
        if r == "KO/TKO":
            bits.append(f"{f.get('ko_wins', 0)} of {f.get('wins', 0)} UFC wins by KO")
            if o.get("ko_losses"):
                bits.append(f"opponent stopped by strikes {o['ko_losses']}x")
        elif r == "Submission":
            bits.append(f"{f.get('sub_wins', 0)} of {f.get('wins', 0)} UFC wins by submission")
            if o.get("sub_losses"):
                bits.append(f"opponent submitted {o['sub_losses']}x")
        return "; ".join(bits)
    line = f"Model {_pct(pm)}, market {_pct(pk)}" if pk is not None else f"Model {_pct(pm)}"
    if leg["edge"] is not None and leg["edge"] >= RISKY_MIN_EDGE:
        line += " - the model rates them higher"
    if form:
        line += f"; last {len(form)}: {form}"
    return line
