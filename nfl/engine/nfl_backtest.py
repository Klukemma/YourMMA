"""Walk-forward evaluation of the NFL models, against the closing market.

THE PROTOCOL. Every prediction for season S comes from a model fitted only
on seasons before S, and every feature it reads was known before kickoff
(nfl_features). Two periods are reported and never pooled:

  DEV      2006-2015   what choices were made on (parameters, features)
  CONFIRM  2016-2025   untouched while choosing; the honest number

The market is the closing moneyline with the bookmaker's margin removed
(2006 onwards; earlier seasons have spreads only). The bar for "the model
knows something the market does not" is beating it on the SAME games -
and a flat-stake profit at closing prices in both periods. Anything that
fails that bar is shown to the reader as an opinion, not an edge.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DEV = (2006, 2015)
CONFIRM = (2016, 2025)
TRAIN_FROM = 2002   # 1999-2001 only warm the ratings up
# Point margin -> chance the home side wins, fitted on DEV games (closing
# spread against outcome). Used where a game has a spread but no moneyline.
SPREAD_SD = 12.2
# A game's result is about this many points from even a perfect forecast.
MARGIN_SD = 13.4
TOTAL_SD = 13.2


def american_to_prob(odds):
    """Implied probability of American odds (vig still in)."""
    o = np.asarray(odds, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def american_payout(odds):
    """Profit on a 1-unit stake that wins."""
    o = np.asarray(odds, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, 100.0 / -o, o / 100.0)


def market_prob(frame):
    """The market's chance the HOME side wins: vig-free moneyline, else the
    spread through a normal curve. NaN where the market said nothing."""
    ph, pa = american_to_prob(frame["home_ml"]), american_to_prob(frame["away_ml"])
    with np.errstate(invalid="ignore"):
        ml = ph / (ph + pa)
    spread = norm.cdf(np.asarray(frame["spread_line"], float) / SPREAD_SD)
    return pd.Series(np.where(np.isfinite(ml), ml, spread), index=frame.index)


def decided(frame):
    """Games with a winner (ties are scored by neither side's model)."""
    return frame["played"] & (frame["result"] != 0)


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def walk_forward(frame, fit_predict, start=DEV[0], end=CONFIRM[1], train_from=TRAIN_FROM):
    """Out-of-sample predictions, season by season.

    fit_predict(train, test) -> array for test rows. `train` is every
    decided game from train_from up to the season before; `test` is every
    game of the season (played or not).
    """
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    ok = decided(frame)
    for season in range(start, end + 1):
        train = frame[ok & frame["season"].between(train_from, season - 1)]
        test = frame[frame["season"] == season]
        if len(train) and len(test):
            out.loc[test.index] = np.asarray(fit_predict(train, test), float)
    return out


def winner_report(frame, p, label="model", periods=(DEV, CONFIRM)):
    """Log loss, Brier, accuracy and AUC for a home-win probability, beside
    the market's on the same games, plus flat-stake ROI at closing prices."""
    mkt = market_prob(frame)
    y = (frame["result"] > 0).astype(float)
    report = {}
    for lo, hi in periods:
        sel = decided(frame) & frame["season"].between(lo, hi) & p.notna() & mkt.notna()
        has_ml = sel & frame["home_ml"].notna() & frame["away_ml"].notna()
        row = {
            "games": int(sel.sum()),
            "log_loss": log_loss(p[sel], y[sel]),
            "brier": brier(p[sel], y[sel]),
            "accuracy": float(((p[sel] > 0.5) == (y[sel] == 1)).mean()),
            "auc": auc(p[sel], y[sel]),
            "market_log_loss": log_loss(mkt[sel], y[sel]),
            "market_brier": brier(mkt[sel], y[sel]),
            "market_accuracy": float(((mkt[sel] > 0.5) == (y[sel] == 1)).mean()),
            "roi": flat_roi(frame[has_ml], p[has_ml], y[has_ml]),
        }
        report[f"{lo}-{hi}"] = row
    return {label: report}


def flat_roi(frame, p, y, min_edge=0.0):
    """Bet 1 unit at the closing moneyline on whichever side the model
    rates above the market by more than `min_edge`. Returns
    {bets, won, roi}."""
    mkt = market_prob(frame)
    home_edge = p - mkt
    bet_home = home_edge > min_edge
    bet_away = (-home_edge) > min_edge
    pay_h, pay_a = american_payout(frame["home_ml"]), american_payout(frame["away_ml"])
    profit = np.where(bet_home, np.where(y == 1, pay_h, -1.0), 0.0)
    profit += np.where(bet_away, np.where(y == 0, pay_a, -1.0), 0.0)
    bets = int(bet_home.sum() + bet_away.sum())
    won = int(((bet_home) & (y == 1)).sum() + ((bet_away) & (y == 0)).sum())
    return {"bets": bets, "won": won,
            "roi": float(np.nansum(profit) / bets) if bets else float("nan")}


def line_report(frame, predicted, line, outcome, periods=(DEV, CONFIRM), edges=(0, 1.5, 3)):
    """Against the spread or the total: how often the side the model takes
    against the closing line wins, by how far the model is from the line.
    Pushes are dropped; -110 on both sides needs 52.4% to break even."""
    out = {}
    for lo, hi in periods:
        sel = frame["played"] & frame["season"].between(lo, hi) & predicted.notna() & line.notna()
        gap = (predicted - line)[sel]
        actual = (outcome - line)[sel]
        rows = {}
        for e in edges:
            pick = gap.abs() > e
            live = pick & (actual != 0)
            hit = (np.sign(gap[live]) == np.sign(actual[live]))
            rows[str(e)] = {"bets": int(live.sum()),
                            "hit_rate": float(hit.mean()) if live.any() else float("nan")}
        out[f"{lo}-{hi}"] = rows
    return out


def calibration(frame, p, bins=10, period=CONFIRM):
    """Predicted vs observed home-win rate by probability decile."""
    sel = decided(frame) & frame["season"].between(*period) & p.notna()
    y = (frame["result"] > 0).astype(float)[sel]
    q = p[sel]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q >= lo) & (q < hi if hi < 1 else q <= hi)
        if m.sum() >= 10:
            rows.append({"lo": round(lo, 2), "hi": round(hi, 2), "games": int(m.sum()),
                         "predicted": float(q[m].mean()), "observed": float(y[m].mean())})
    return rows


# How sure a pick is, by the blended probability of the side picked. Each
# band's hit rate is MEASURED on the confirm period and printed beside the
# label, so "STRONG" means "picks like this won N% of 2016-2025 games" and
# never "bet this".
TIERS = (("STRONG", 0.75), ("SOLID", 0.65), ("LEAN", 0.55), ("TOSS-UP", 0.0))
BREAK_EVEN_110 = 110 / 210      # -110 both sides
VALUE_EDGES = (0.03, 0.05, 0.08)
MIN_GATE_BETS = 100


def tier_of(pick_prob):
    if pick_prob is None or pick_prob != pick_prob:
        return None
    for name, floor in TIERS:
        if pick_prob >= floor:
            return name
    return TIERS[-1][0]


def tier_report(frame, p, periods=(DEV, CONFIRM)):
    """Hit rate of the pick in each confidence band, per period."""
    y = (frame["result"] > 0).astype(float)
    pick_p = np.maximum(p, 1 - p)
    correct = ((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))
    out = {}
    for lo, hi in periods:
        sel = decided(frame) & frame["season"].between(lo, hi) & p.notna()
        rows = {}
        for name, _floor in TIERS:
            m = sel & (pick_p.map(tier_of) == name)
            rows[name] = {"picks": int(m.sum()),
                          "hit_rate": float(correct[m].mean()) if m.any() else None,
                          "mean_prob": float(pick_p[m].mean()) if m.any() else None}
        out[f"{lo}-{hi}"] = rows
    return out


def value_gate(frame, model_p, edges=VALUE_EDGES):
    """Does backing the model where it disagrees with the closing price make
    money? Flat 1-unit stakes at the closing moneyline on the side the RAW
    model (no market in it - an edge measured against a number that already
    contains the line is the line measured against itself) rates at least
    `edge` above the no-vig market. A label is switched on only at an edge
    that was profitable in BOTH periods on at least MIN_GATE_BETS bets each."""
    y = (frame["result"] > 0).astype(float)
    rows, enabled = {}, None
    for edge in edges:
        per = {}
        for lo, hi in (DEV, CONFIRM):
            sel = (decided(frame) & frame["season"].between(lo, hi) & model_p.notna()
                   & frame["home_ml"].notna() & frame["away_ml"].notna())
            per[f"{lo}-{hi}"] = flat_roi(frame[sel], model_p[sel], y[sel], min_edge=edge)
        rows[str(edge)] = per
        if enabled is None and all(r["bets"] >= MIN_GATE_BETS and r["roi"] == r["roi"] and r["roi"] > 0
                                   for r in per.values()):
            enabled = edge
    return {"edges": rows, "enabled_at": enabled}


def line_gate(report, min_bets=200):
    """The smallest model-vs-line gap whose side won often enough to beat
    -110 in both periods, or None."""
    for edge in sorted(next(iter(report.values())).keys(), key=float):
        per = [report[k][edge] for k in report]
        if all(r["bets"] >= min_bets and r["hit_rate"] == r["hit_rate"] and r["hit_rate"] > BREAK_EVEN_110 for r in per):
            return float(edge)
    return None


def by_season(frame, p, first=DEV[0]):
    """Accuracy of the pick and of the market's favourite, season by season."""
    y = (frame["result"] > 0).astype(float)
    mkt = market_prob(frame)
    rows = []
    for season in sorted(frame["season"].unique()):
        if season < first:
            continue
        sel = decided(frame) & (frame["season"] == season) & p.notna() & mkt.notna()
        if sel.sum() < 16:
            continue
        rows.append({"season": int(season), "games": int(sel.sum()),
                     "accuracy": float(((p[sel] > 0.5) == (y[sel] == 1)).mean()),
                     "market_accuracy": float(((mkt[sel] > 0.5) == (y[sel] == 1)).mean()),
                     "log_loss": log_loss(p[sel], y[sel]),
                     "market_log_loss": log_loss(mkt[sel], y[sel])})
    return rows


def _fmt(report):
    lines = []
    for label, periods in report.items():
        for period, r in periods.items():
            lines.append(
                f"  {label:24s} {period}  ll {r['log_loss']:.4f} (mkt {r['market_log_loss']:.4f})"
                f"  acc {r['accuracy']:.3f} (mkt {r['market_accuracy']:.3f})"
                f"  auc {r['auc']:.3f}  roi {r['roi']['roi']:+.3f} on {r['roi']['bets']}  n={r['games']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import nfl_model
    frame = nfl_model.feature_frame()
    result = nfl_model.backtest(frame)
    print(_fmt({k: v for k, v in result["winner"].items()}))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps(result, indent=1, default=float))
