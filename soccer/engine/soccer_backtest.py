"""Walk-forward evaluation of the soccer models, against the market.

THE PROTOCOL. Every probability for a match comes only from matches played
on EARLIER DAYS (check_point_in_time enforces it on any predictor). Two
periods are reported and never pooled:

  DEV      2006-2015   what choices were made on (model, parameters)
  CONFIRM  2016-2025   untouched while choosing; the honest number

2005-06 only warms the ratings up. Only the five top leagues are scored;
the second tiers are rated, never predicted.

THE MARKET is football-data.co.uk's PRE-MATCH average price - collected on
Friday afternoon for weekend games, Tuesday afternoon for midweek ones -
with the bookmakers' margin removed. That is the price the app will have
when it picks (the fixtures file is published at the same moment), so it is
the fair bar. The closing price, which knows two more days of team news, is
reported beside it to show how much the market still moved.

A soccer match has three results. Every score here is three-way: log loss
over home/draw/away, the three-way Brier score, and accuracy of the most
likely result.
"""

import math
import sys

import numpy as np
import pandas as pd

import soccer_data as sd

DEV = (2006, 2015)
CONFIRM = (2016, 2025)
RESULTS = ("H", "D", "A")
PRICE_SETS = {
    "pre": ("avg_h", "avg_d", "avg_a"),
    "pinnacle": ("ps_h", "ps_d", "ps_a"),
    "close": ("avgc_h", "avgc_d", "avgc_a"),
    "pinnacle_close": ("psc_h", "psc_d", "psc_a"),
    "best": ("max_h", "max_d", "max_a"),
}


def no_vig(df, prices="pre"):
    """Decimal prices -> probabilities with the margin removed in proportion.
    NaN where any of the three prices is missing."""
    cols = PRICE_SETS[prices]
    inv = 1.0 / df[list(cols)].astype(float)
    inv = inv.where(np.isfinite(inv) & (inv > 0))
    total = inv.sum(axis=1, min_count=3)
    out = inv.div(total, axis=0)
    out.columns = ["p_h", "p_d", "p_a"]
    return out


def no_vig_total(df, over="avg_o25", under="avg_u25"):
    inv_o, inv_u = 1.0 / df[over].astype(float), 1.0 / df[under].astype(float)
    return inv_o / (inv_o + inv_u)


def outcome_index(result):
    return result.map({"H": 0, "D": 1, "A": 2})


def log_loss(p, result):
    """Mean three-way log loss. p: frame with p_h, p_d, p_a."""
    idx = outcome_index(result).to_numpy()
    probs = p[["p_h", "p_d", "p_a"]].to_numpy(float)
    chosen = probs[np.arange(len(probs)), idx]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1))))


def brier(p, result):
    idx = outcome_index(result).to_numpy()
    probs = p[["p_h", "p_d", "p_a"]].to_numpy(float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), idx] = 1.0
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))


def accuracy(p, result):
    pick = np.asarray(RESULTS)[p[["p_h", "p_d", "p_a"]].to_numpy(float).argmax(axis=1)]
    return float((pick == result.to_numpy()).mean())


def binary_log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scored(matches):
    """Top-league matches with a result and a full pre-match price."""
    top = matches["league"].isin(sd.TOP) & matches["result"].isin(RESULTS)
    mkt = no_vig(matches, "pre").notna().all(axis=1)
    return top & mkt


def report(matches, preds, label="model", periods=(DEV, CONFIRM)):
    """Three-way scores of `preds` (indexed like `matches`, columns p_h, p_d,
    p_a, optionally p_o25) beside the pre-match and closing markets ON THE
    SAME MATCHES, per period."""
    mkt = no_vig(matches, "pre")
    out = {}
    for lo, hi in periods:
        sel = scored(matches) & matches["season"].between(lo, hi) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1)
        if not sel.any():
            continue
        m, p, q = matches[sel], preds[sel], mkt[sel]
        row = {
            "matches": int(sel.sum()),
            "log_loss": log_loss(p, m["result"]), "market_log_loss": log_loss(q, m["result"]),
            "brier": brier(p, m["result"]), "market_brier": brier(q, m["result"]),
            "accuracy": accuracy(p, m["result"]), "market_accuracy": accuracy(q, m["result"]),
            "mean_draw": float(p["p_d"].mean()), "draw_rate": float((m["result"] == "D").mean()),
        }
        close = no_vig(m, "close")
        has_close = close.notna().all(axis=1)
        if has_close.sum() >= 200:
            row["closing"] = {"matches": int(has_close.sum()),
                              "log_loss": log_loss(p[has_close], m.loc[has_close, "result"]),
                              "market_pre": log_loss(q[has_close], m.loc[has_close, "result"]),
                              "market_close": log_loss(close[has_close], m.loc[has_close, "result"])}
        if "p_o25" in preds:
            ou = sel & preds["p_o25"].notna() & matches["avg_o25"].notna() & matches["avg_u25"].notna()
            y = (matches.loc[ou, "hg"] + matches.loc[ou, "ag"] > 2.5).astype(float)
            row["over_2_5"] = {"matches": int(ou.sum()),
                               "log_loss": binary_log_loss(preds.loc[ou, "p_o25"], y),
                               "market_log_loss": binary_log_loss(no_vig_total(matches[ou]), y)}
        out[f"{lo}-{hi}"] = row
    return {label: out}


# ------------------------------------------------------------- the blend

def _log3(p):
    return np.log(np.clip(p[["p_h", "p_d", "p_a"]].to_numpy(float), 1e-9, 1))


def pool(model, market, w):
    """Logarithmic pooling: p_k proportional to model_k^a * market_k^b *
    exp(c_k), with w = (a, b, c_draw, c_away) and c_home fixed at 0."""
    a, b, c_d, c_a = w
    z = a * _log3(model) + b * _log3(market) + np.array([0.0, c_d, c_a])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return pd.DataFrame(p, index=model.index, columns=["p_h", "p_d", "p_a"])


def fit_pool(model, market, result):
    """The pooling weights that minimise log loss on these matches."""
    from scipy.optimize import minimize
    idx = outcome_index(result).to_numpy()
    lm, lq = _log3(model), _log3(market)

    def loss(w):
        a, b, c_d, c_a = w
        z = a * lm + b * lq + np.array([0.0, c_d, c_a])
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(idx)), idx].mean()

    fit = minimize(loss, x0=np.array([0.2, 0.8, 0.0, 0.0]), method="L-BFGS-B")
    return tuple(float(v) for v in fit.x)


def blend_walk_forward(matches, preds, first=DEV[0] + 2):
    """Out-of-sample blended probabilities: each season's pooling weights are
    fitted on the model's out-of-sample predictions for EARLIER seasons only.
    Returns (blended frame, {season: weights})."""
    mkt = no_vig(matches, "pre")
    ok = scored(matches) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1)
    out = pd.DataFrame(np.nan, index=matches.index, columns=["p_h", "p_d", "p_a"])
    weights = {}
    for season in sorted(matches.loc[ok, "season"].unique()):
        if season < first:
            continue
        train = ok & (matches["season"] < season)
        test = ok & (matches["season"] == season)
        if train.sum() < 1000 or not test.any():
            continue
        w = fit_pool(preds[train], mkt[train], matches.loc[train, "result"])
        weights[int(season)] = w
        out.loc[test] = pool(preds[test], mkt[test], w).to_numpy()
    return out, weights


def flat_roi(matches, probs, prices="best", min_edge=0.0):
    """1 unit on every result `probs` rates at least `min_edge` above the
    no-vig pre-match market, at the `prices` set (best = the highest price
    any book posted). Returns {bets, roi}."""
    mkt = no_vig(matches, "pre")
    cols = PRICE_SETS[prices]
    staked = won = 0.0
    for k, res in enumerate(RESULTS):
        key = ["p_h", "p_d", "p_a"][k]
        price = matches[cols[k]].astype(float)
        bet = scored(matches) & (probs[key] - mkt[key] >= min_edge) & price.notna() & probs[key].notna()
        bet &= probs[key] - mkt[key] > 0
        staked += bet.sum()
        won += (price[bet] * (matches.loc[bet, "result"] == res)).sum()
    return {"bets": int(staked), "roi": (won - staked) / staked if staked else None}


# ------------------------------------------------------------- the guard

def check_point_in_time(predict, matches, cut_date):
    """Raise unless predictions for matches ON cut_date ignore every result
    from cut_date on. `predict(matches) -> frame indexed like matches`.
    Scrambles those results (and their match statistics) and compares."""
    cut = str(cut_date)
    later = matches["date"] >= cut
    scrambled = matches.copy()
    rng = np.random.default_rng(0)
    for c in ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"):
        if c in scrambled:
            scrambled.loc[later, c] = rng.integers(0, 6, int(later.sum()))
    scrambled.loc[later, "result"] = np.where(scrambled.loc[later, "hg"] > scrambled.loc[later, "ag"], "H",
                                              np.where(scrambled.loc[later, "hg"] < scrambled.loc[later, "ag"], "A", "D"))
    on_day = matches["date"] == cut
    a = predict(matches).loc[on_day, ["p_h", "p_d", "p_a"]]
    b = predict(scrambled).loc[on_day, ["p_h", "p_d", "p_a"]]
    if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
        raise AssertionError(f"predictions for {cut} change when results from {cut} on change")
    return True


def _fmt(rep):
    lines = []
    for label, periods in rep.items():
        for period, r in periods.items():
            lines.append(f"{label:>14} {period}: {r['matches']} matches  log loss {r['log_loss']:.4f} "
                         f"(market {r['market_log_loss']:.4f})  acc {r['accuracy']:.3f} "
                         f"(market {r['market_accuracy']:.3f})  draws {r['mean_draw']:.3f} vs {r['draw_rate']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    matches = sd.load_matches()
    mkt = no_vig(matches, "pre")
    print(_fmt(report(matches, mkt, "market")))
    sys.exit(0)
