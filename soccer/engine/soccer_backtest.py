"""Walk-forward evaluation of the soccer models, against the market.

THE PROTOCOL. Every probability for a match comes only from matches played
on EARLIER DAYS (check_point_in_time enforces it on any predictor). Two
periods are reported and never pooled:

  DEV      2006-2015   what choices were made on (model, parameters)
  CONFIRM  2016-2025   untouched while choosing; the honest number

2005-06 only warms the ratings up. Called without `leagues`, scored() and
report() score the five top leagues, as every experiment that chose the
model did; the second tiers are rated there, never predicted.

THE MARKET is football-data.co.uk's PRE-MATCH average price - collected on
Friday afternoon for weekend games, Tuesday afternoon for midweek ones -
with the bookmakers' margin removed. That is the price the app will have
when it picks (the fixtures file is published at the same moment), so it is
the fair bar. The closing price, which knows two more days of team news, is
reported beside it to show how much the market still moved.

EVERY LEAGUE IN THE APP (experiments/leagues.py measured them; soccer_leagues
holds what it decided). Each league is scored on its OWN periods
(periods_for: the European divisions 2006-2015 / 2016-2025, the extra
leagues 2013-2018 / 2019-2025, 2012 a warm-up) and against the price it was
measured on (market_price: the pre-match average; for an extra league,
whose history has only the close, the closing average). The close is the
better price (0.0035 lower log loss on the European divisions' 2019-2025),
so an extra league's record is measured against a harder bar than its live
picks, which use the pre-match price - its caveat says so. Periods of
different leagues are never pooled under one name: per league, per group
(soccer_leagues.GROUPS - leagues of one group share their periods) and
pooled over every league ("dev": each league's own DEV, "confirm" likewise),
each match once (scored(matches, leagues), report(..., leagues=...),
record()). record() is the Record tab's table: on the matches
experiments/leagues.py scored, it reproduces soccer_leagues.SHIPPED's
measured numbers to the digit.

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
# The extra leagues (soccer_data.EXTRA) start in 2012 with closing prices
# only: 2012 warms the ratings up, then their own two periods
# (experiments/leagues.py). Season 2026 is live and never measured.
EXTRA_WARMUP = 2012
EXTRA_DEV = (2013, 2018)
EXTRA_CONFIRM = (2019, 2025)
RESULTS = ("H", "D", "A")


PERIOD_KEYS = ("dev", "confirm")


def periods_for(league):
    """(DEV, CONFIRM) for a league: 2006-2015 / 2016-2025 for the European
    divisions, 2013-2018 / 2019-2025 for the extra leagues (EXTRA_WARMUP,
    2012, their first season on file, only warms the ratings and the base
    rates up). Read at call time, so a test that shortens the periods
    shortens every caller's."""
    return (EXTRA_DEV, EXTRA_CONFIRM) if league in sd.EXTRA else (DEV, CONFIRM)


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


def market_price(frame):
    """Each row's no-vig home/draw/away from the price its league is picked
    on and scored against: the pre-match average where all three were
    posted; for an extra league (soccer_data.EXTRA), whose history has only
    the closing price, the closing average where no pre-match price was
    (its live fixtures do carry the pre-match one). A European division's
    match with only a close has no price: its history always has the
    pre-match one, and mixing the two would score it on a better price
    than it is picked on. -> (frame p_h, p_d, p_a - NaN where neither;
    Series "pre" / "close" / None)."""
    pre = no_vig(frame, "pre")
    close = no_vig(frame, "close")
    has_pre = pre.notna().all(axis=1)
    use_close = frame["league"].isin(sd.EXTRA) & ~has_pre & close.notna().all(axis=1)
    q = pre.where(has_pre)
    q[use_close] = close[use_close]
    kind = pd.Series(np.where(has_pre, "pre", np.where(use_close, "close", None)), index=frame.index, dtype=object)
    return q, kind


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


def scored(matches, leagues=None, q=None):
    """The matches a forecast is scored on: a result and the price it is
    scored against. Without `leagues`, the top-league matches with a full
    pre-match price (what every experiment scored, bit for bit). With
    `leagues`, those leagues' matches with their own measurement price
    (market_price: the close for an extra league's history) - every season;
    league_period() cuts each league's own periods. q: market_price's
    prices, where already computed."""
    if leagues is None:
        top = matches["league"].isin(sd.TOP) & matches["result"].isin(RESULTS)
        mkt = no_vig(matches, "pre").notna().all(axis=1)
        return top & mkt
    q = market_price(matches)[0] if q is None else q
    return matches["league"].isin(list(leagues)) & matches["result"].isin(RESULTS) & q.notna().all(axis=1)


def league_period(matches, leagues, k):
    """Rows of `leagues` whose season lies in the league's own k-th period
    (0: DEV, 1: CONFIRM; periods_for) - so a match is never scored in a
    period its league was not measured on."""
    league = matches["league"]
    sel = pd.Series(False, index=matches.index)
    for code in leagues:
        lo, hi = periods_for(code)[k]
        sel |= (league == code) & matches["season"].between(lo, hi)
    return sel


def _scores(m, p, q):
    """The old report's columns for forecast p against the market q on
    matches m (all three aligned)."""
    return {"matches": int(len(m)),
            "log_loss": log_loss(p, m["result"]), "market_log_loss": log_loss(q, m["result"]),
            "brier": brier(p, m["result"]), "market_brier": brier(q, m["result"]),
            "accuracy": accuracy(p, m["result"]), "market_accuracy": accuracy(q, m["result"]),
            "mean_draw": float(p["p_d"].mean()), "draw_rate": float((m["result"] == "D").mean())}


def _over(matches, preds, sel, skip_empty=False):
    """Over 2.5: `preds`' p_o25 against the no-vig over/under, where posted.
    skip_empty: None where nothing is (the top five's report kept its empty
    row, NaN and all)."""
    ou = sel & preds["p_o25"].notna() & matches["avg_o25"].notna() & matches["avg_u25"].notna()
    if skip_empty and not ou.any():
        return None
    y = (matches.loc[ou, "hg"] + matches.loc[ou, "ag"] > 2.5).astype(float)
    return {"matches": int(ou.sum()),
            "log_loss": binary_log_loss(preds.loc[ou, "p_o25"], y),
            "market_log_loss": binary_log_loss(no_vig_total(matches[ou]), y)}


def report(matches, preds, label="model", periods=(DEV, CONFIRM), leagues=None):
    """Three-way scores of `preds` (indexed like `matches`, columns p_h, p_d,
    p_a, optionally p_o25) beside the pre-match and closing markets ON THE
    SAME MATCHES, per period - the top five's, without `leagues`.

    With `leagues`: every one of them on its own periods (periods_for, not
    `periods`) against the price it was measured on (market_price), as
    {label: {"dev": pooled, "confirm": pooled, "by_group": {group: {"codes",
    "periods", "dev", "confirm"}}, "by_league": {code: {"periods", "price",
    "dev", "confirm"}}}} - pooled over every league, each match once, under
    "dev" / "confirm" because the leagues' periods differ; a group's leagues
    share theirs. "over_2_5" where the over/under was posted (never an extra
    league's); no "closing" column, since an extra league's market already
    is the close."""
    if leagues is not None:
        return {label: _report_leagues(matches, preds, list(leagues))}
    mkt = no_vig(matches, "pre")
    out = {}
    for lo, hi in periods:
        sel = scored(matches) & matches["season"].between(lo, hi) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1)
        if not sel.any():
            continue
        m, p, q = matches[sel], preds[sel], mkt[sel]
        row = _scores(m, p, q)
        close = no_vig(m, "close")
        has_close = close.notna().all(axis=1)
        if has_close.sum() >= 200:
            row["closing"] = {"matches": int(has_close.sum()),
                              "log_loss": log_loss(p[has_close], m.loc[has_close, "result"]),
                              "market_pre": log_loss(q[has_close], m.loc[has_close, "result"]),
                              "market_close": log_loss(close[has_close], m.loc[has_close, "result"])}
        if "p_o25" in preds:
            row["over_2_5"] = _over(matches, preds, sel)
        out[f"{lo}-{hi}"] = row
    return {label: out}


def _groups(leagues):
    """[(group, its codes among `leagues`)] in soccer_leagues.GROUPS order."""
    import soccer_leagues as sl
    return [(g, [c for c in codes if c in leagues]) for g, codes in sl.GROUPS.items()
            if any(c in leagues for c in codes)]


def _report_leagues(matches, preds, leagues):
    q, _ = market_price(matches)
    ok = scored(matches, leagues, q) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1)

    def block(codes):
        out = {}
        for k, key in enumerate(PERIOD_KEYS):
            sel = ok & league_period(matches, codes, k)
            if sel.any():
                out[key] = _scores(matches[sel], preds[sel], q[sel])
                over = _over(matches, preds, sel, skip_empty=True) if "p_o25" in preds else None
                if over is not None:
                    out[key]["over_2_5"] = over
        return out
    rep = block(leagues)
    rep["by_group"] = {g: dict(codes=codes, periods={key: list(periods_for(codes[0])[k])
                                                     for k, key in enumerate(PERIOD_KEYS)}, **block(codes))
                       for g, codes in _groups(leagues)}
    rep["by_league"] = {c: dict(periods={key: list(periods_for(c)[k]) for k, key in enumerate(PERIOD_KEYS)},
                                price="closing" if c in sd.EXTRA else "pre-match", **block([c]))
                        for c in leagues}
    return rep


# ------------------------------------------------------------- the Record tab, every league

RECORD_SOURCES = ("pick", "market", "model", "base")


def _mean(x, digits):
    return round(float(np.mean(x)), digits) if len(x) else None


def record(frame, pick, preds, leagues, base=None):
    """The Record tab for every league in `leagues`: for each, on its own
    periods (periods_for) and against the price it was measured on
    (market_price), the three-way log loss and accuracy of the pick (`pick`:
    soccer_model.shipped_pick's), the raw no-vig market, the model's own
    (`preds`) and the league's base rate (its home / draw / away frequencies
    over earlier seasons: soccer_scores.baseline; `base` where computed) -
    all on the SAME matches: a result, the price, a model number and a base
    rate (experiments/leagues.py's `common` rows). So a league's row is
    soccer_leagues.SHIPPED[code]["measured"] to the digit (log loss to 5
    decimals, accuracy to 4, as it stores them): the record shown is the
    record that decided the league ships. A league with no such match in a
    period has {"matches": 0} there, and nothing else - a missing number is
    shown as missing.

    Returns {"by_league": [{code, name, country, group, price, periods, dev,
    confirm}] in `leagues` order, "by_group": {group: {name, codes, periods,
    dev, confirm}}, "pooled": {codes, dev, confirm}} - a group's leagues share
    their periods; "pooled" is every league on its own periods, each match
    once. Each period: {seasons (league and group only), matches, log_loss,
    market_log_loss, model_log_loss, base_log_loss, accuracy,
    market_accuracy, model_accuracy}."""
    import soccer_leagues as sl
    leagues = list(leagues)
    if base is None:
        import soccer_scores as ssc
        base = ssc.baseline(frame, leagues)
    q, _ = market_price(frame)
    res = frame["result"].to_numpy()
    lab = outcome_index(frame["result"]).to_numpy()
    probs = {"pick": pick, "market": q, "model": preds, "base": base}
    probs = {k: v[["p_h", "p_d", "p_a"]].to_numpy(float) for k, v in probs.items()}
    ok = (scored(frame, leagues, q) & preds["p_h"].notna() & base["p_h"].notna()).to_numpy()
    blank = ok & ~np.isfinite(probs["pick"]).all(axis=1)
    if blank.any():
        raise ValueError(f"the pick has no number on {int(blank.sum())} priced matches of "
                         f"{sorted(set(frame['league'].to_numpy()[blank]))}: pass the pick of every league scored")
    loss, hit = {}, {}
    for k, p in probs.items():
        loss[k] = np.full(len(frame), np.nan)
        loss[k][ok] = -np.log(np.clip(p[ok][np.arange(ok.sum()), lab[ok].astype(int)], 1e-12, 1))
        hit[k] = (np.array(RESULTS)[np.nan_to_num(p).argmax(axis=1)] == res).astype(float)
    names = {"pick": "log_loss", "market": "market_log_loss", "model": "model_log_loss", "base": "base_log_loss"}
    accs = {"pick": "accuracy", "market": "market_accuracy", "model": "model_accuracy"}

    def period(codes, k, seasons=True):
        sel = ok & league_period(frame, codes, k).to_numpy()
        out = {"seasons": list(periods_for(codes[0])[k])} if seasons else {}
        out["matches"] = int(sel.sum())
        if sel.any():
            out.update({names[s]: _mean(loss[s][sel], 5) for s in RECORD_SOURCES})
            out.update({accs[s]: round(float(hit[s][sel].mean()), 4) for s in accs})
        return out

    def periods(codes):
        return {key: list(periods_for(codes[0])[k]) for k, key in enumerate(PERIOD_KEYS)}
    by_league = []
    for c in leagues:
        info = sd.ALL_LEAGUES[c]
        by_league.append(dict(code=c, name=info.name, country=info.country, group=sl.group_of(c),
                              price="closing" if c in sd.EXTRA else "pre-match", periods=periods([c]),
                              **{key: period([c], k) for k, key in enumerate(PERIOD_KEYS)}))
    by_group = {g: dict(name=sl.GROUP_NAMES[g], codes=codes, periods=periods(codes),
                        **{key: period(codes, k) for k, key in enumerate(PERIOD_KEYS)})
                for g, codes in _groups(leagues)}
    pooled = dict(codes=leagues, **{key: period(leagues, k, seasons=False) for k, key in enumerate(PERIOD_KEYS)})
    return {"by_league": by_league, "by_group": by_group, "pooled": pooled}


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


def fit_pool(model, market, result, nonnegative=False):
    """The pooling weights that minimise log loss on these matches. With
    nonnegative=True the model's weight cannot go below zero: a negative
    weight bets against the model, a contrarian signal rather than a
    forecast."""
    from scipy.optimize import minimize
    idx = outcome_index(result).to_numpy()
    lm, lq = _log3(model), _log3(market)

    def loss(w):
        a, b, c_d, c_a = w
        z = a * lm + b * lq + np.array([0.0, c_d, c_a])
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(idx)), idx].mean()

    bounds = [(0.0, None) if nonnegative else (None, None), (None, None), (None, None), (None, None)]
    fit = minimize(loss, x0=np.array([0.2, 0.8, 0.0, 0.0]), method="L-BFGS-B", bounds=bounds)
    return tuple(float(v) for v in fit.x)


def blend_walk_forward(matches, preds, first=DEV[0] + 2, nonnegative=False):
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
        w = fit_pool(preds[train], mkt[train], matches.loc[train, "result"], nonnegative=nonnegative)
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
