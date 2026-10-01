"""Is the man who will really be heavier at the bell underrated? True Fight
Weight, estimated from measurements PUBLISHED before each fight, against
the model and the market.

Everyone at lightweight weighs in at 155 and nobody fights at 155. The
measured fight-night weights (fight_night_weights.csv, California's
second-day checks as Wikipedia reports them) say the middle half regain
7-13%, and a fighter's own earlier regain is the only thing that predicts
the next one (true_fight_weight.py: forward-chained, the earlier regain
alone beats the shrinkage model on the few fighters with one, and k is
not identified from them; without one the estimate is the division norm
and nothing beats it).

Point in time, every arm: a measurement enters an estimate only from its
publication date (published_at, event + 7 days where no dated citation
exists), so a fight's own weight check - taken on fight day and released
after the bout - never informs that fight, and nothing from the same event
does. mu, k and the frame ridge are refitted from the labels published
before each fight. Thresholds are frozen on the FIRST HALF (by date) of
each arm's eligible priced fights before anything is measured.

Pre-registered arms, a family of three (design 4.4), all through
residual_harness: priced fights only, each marked fighter compared with
unmarked fighters in the same win-probability decile and age-gap band,
bootstrapped over events and over fighters, family-wise 1 - 0.05/3, at
least 150 rows and 40 fighters.

  regain_edge_top_third     fights where at least one fighter has an earlier
                            published measurement; the side whose
                            estimated regain advantage (r_hat red - r_hat
                            blue) is in the top third. The marked side can
                            be the UNMEASURED one (the measured opponent
                            regains below the norm); the share is reported.
  tfw_gap_top_third         the same fights, marked on the full estimated
                            fight-night gap as % of the limit: the limit plus
                            any recorded missed-weight pounds, times
                            (1 + r_hat), plus the frame ridge ONLY if
                            validation earned it. Until per-fight official
                            weigh-in weights exist this DUPLICATES arm 1
                            except where someone missed weight, and it is a
                            T1 variable (design 2.1) that the design says
                            to test only against odds taken after the
                            weigh-in - odds.csv carries no timestamp, so
                            that rule is NOT met. Kept so the family stays
                            the pre-registered three (conservative); the
                            overlap with arm 1 is reported.
  frame_only_gap_top_third  all mask-eligible priced fights where NEITHER
                            fighter has a measurement: the gap from the
                            division norm plus the frame ridge (height,
                            reach, age, class drop). Expected null; it checks
                            the variable is not re-pricing frame size, which
                            size_edge already found inside the noise

  placebo (not in the family): the same fights and the same fighters, but
  each measurement's residual is handed to a DIFFERENT measured fighter of
  the same label event (keys permuted within each event's usable rows), so
  every fight keeps its coverage and its threshold and loses only the link
  between a fighter and their own number. The harness must say inside the
  noise. (Shuffling the fight-level delta within archive events, the
  design's wording, was tried first: most archive events hold one such
  fight, so that shuffle was the identity on two-thirds of the marked
  rows.) The overlap with arm 1's marked rows is reported.

  secondary, not a ship criterion (design 4.4): logistic regression of the
  red win on logit(market p) + delta_pct over the arm-1 priced fights,
  standard errors clustered by event.

Mask: a division limit (no catchweight), not heavyweight (265 is a ceiling,
not a target), and at least one label published before the fight. The
bout gate in true_fight_weight.py decides whether the variable may go to
this test; when it is not passed (or not testable with so few events) the
arms still run, for the coverage counts, and nothing may ship. With only
the Wikipedia labels most arms are expected to be too small to report;
that is the honest result, and the coverage counts say exactly how small.

    APP_DATA_DIR=... PREDICTIONS_LOG=... python engine/experiments/true_fight_weight_edge.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.residual_harness import header, mark, measure, predictions, sides
from experiments.size_edge import misses
from name_resolution import norm_name
import true_fight_weight as tfw

OUT = Path(__file__).with_suffix(".json")
ARMS = ["regain_edge_top_third", "tfw_gap_top_third", "frame_only_gap_top_third"]
FAMILY = len(ARMS)
TOP_SHARE = 1 / 3
PLACEBO = "placebo_residuals_permuted"
PLACEBO_SEED = 0


def estimates(frame, labels, archive=None, missed=None, forward=None, frame_earned=False):
    """One row per fight in `frame`: both corners' estimates as of the date.

    eligible      a limit, not heavyweight, and mu exists (a label was
                  published before the fight)
    delta_regain  r_hat(red) - r_hat(blue), the hidden part (arm 1)
    delta_pct     full fight-night gap as % of the limit, from the limit
                  plus recorded missed-weight pounds (arm 2)
    delta_frame   mu + frame ridge for both, no history (arm 3)
    k             the shrinkage k the fight may use (k_for, by publication)
    """
    est = tfw.RegainEstimator(labels)
    ridge = tfw.FrameRidge(est, archive) if archive is not None else None
    forward = tfw.forward_rows(labels) if forward is None else forward
    missed = missed or {}
    k_cache = {}
    first_public = est.labels["published_at"].min()
    rows = []
    for fight in frame.itertuples():
        date = pd.Timestamp(fight.date)
        limit = tfw.limit_of(fight.division)
        sex = "w" if tfw.is_women(fight.division) else "m"
        row = {"fight": fight.Index, "date": date, "event": fight.event, "limit": limit,
               "eligible": False, "n_red": 0, "n_blue": 0, "delta_regain": np.nan,
               "delta_pct": np.nan, "delta_frame": np.nan, "k": np.nan}
        if not np.isfinite(limit) or limit == tfw.HEAVYWEIGHT or date <= first_public:
            rows.append(row)
            continue
        if date not in k_cache:
            k_cache[date] = tfw.k_for(forward, date)
        k = k_cache[date]
        red = est.estimate(norm_name(fight.red_raw), date, limit, sex, k=k)
        blue = est.estimate(norm_name(fight.blue_raw), date, limit, sex, k=k)
        if not (np.isfinite(red["r_hat"]) and np.isfinite(blue["r_hat"])):
            rows.append(row)
            continue
        row.update({"eligible": True, "k": k, "n_red": red["n"], "n_blue": blue["n"],
                    "delta_regain": red["r_hat"] - blue["r_hat"]})
        day = date.date()
        beta = {"red": 0.0, "blue": 0.0}
        if ridge is not None:
            beta = {"red": ridge.adjustment(fight.red_raw, date, limit),
                    "blue": ridge.adjustment(fight.blue_raw, date, limit)}
        weights = {}
        for side, name, e in (("red", fight.red_raw, red), ("blue", fight.blue_raw, blue)):
            over = missed.get((day, norm_name(name)))
            scale = limit + (float(over) if over is not None and np.isfinite(over) else 0.0)
            r_hat = e["r_hat"] + (beta[side] if frame_earned else 0.0)
            weights[side] = tfw.fight_night_lbs(scale, r_hat)
        row["delta_pct"] = 100.0 * (weights["red"] - weights["blue"]) / limit
        if ridge is not None and red["n"] == 0 and blue["n"] == 0:
            row["delta_frame"] = (red["mu"] + beta["red"]) - (blue["mu"] + beta["blue"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("fight")


def placebo_labels(labels, seed=PLACEBO_SEED):
    """The labels with each event's usable rows re-keyed at random: the same
    fighters still have a measurement published on the same day, but it is
    another fighter's number (another residual). Off-limit and suspect rows
    keep their keys, since they are never history anyway."""
    out = labels.copy()
    rng = np.random.default_rng(seed)
    usable = ~(out.get("off_limit", pd.Series(False, index=out.index)).astype(bool)
               | out.get("suspect", pd.Series(False, index=out.index)).astype(bool))
    for _event, idx in out[usable].groupby(["date", "event"]).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        order = rng.permutation(len(idx))
        for column in ("key", "fighter"):
            out.loc[idx, column] = out.loc[idx, column].to_numpy()[order]
    return out


def freeze_threshold(values, dates, share=TOP_SHARE):
    """The |value| that marks the top `share` of the FIRST HALF by date.

    Frozen on the training period so no threshold is tuned on outcomes.
    Returns (threshold, n used); NaN with fewer than 2 values.
    """
    values, dates = np.asarray(values, float), pd.to_datetime(pd.Series(dates))
    keep = np.isfinite(values)
    values, dates = values[keep], dates[keep].reset_index(drop=True)
    if len(values) < 2:
        return float("nan"), 0
    order = np.argsort(dates.values, kind="stable")
    half = order[: max(1, len(order) // 2)]
    return float(np.quantile(np.abs(values[half]), 1 - share)), int(len(half))


def arm_values(est, priced):
    """{arm: (values Series over priced eligible fights)}."""
    e = est[est["eligible"] & est.index.isin(priced)]
    either = e[(e["n_red"] > 0) | (e["n_blue"] > 0)]
    neither = e[(e["n_red"] == 0) & (e["n_blue"] == 0)]
    return {"regain_edge_top_third": either["delta_regain"],
            "tfw_gap_top_third": either["delta_pct"],
            "frame_only_gap_top_third": neither["delta_frame"]}


def thresholds(est, priced):
    return {arm: freeze_threshold(v.values, est.loc[v.index, "date"].values)
            for arm, v in arm_values(est, priced).items()}


def populations(est, thresholds, placebo=None):
    """{name: marked(fight) -> 'red' | 'blue' | None} for the arms and, with
    `placebo` (the estimates built from placebo_labels), the placebo marked
    at arm 1's frozen threshold over arm 1's population."""
    values = {arm: v for arm, v in arm_values(est, est.index).items()}
    cuts = dict(thresholds)
    if placebo is not None:
        real = values["regain_edge_top_third"]
        values[PLACEBO] = placebo.loc[real.index, "delta_regain"]
        cuts[PLACEBO] = thresholds["regain_edge_top_third"]

    def side_of(name):
        series, cut = values[name], cuts[name][0]

        def marked(fight):
            if fight.Index not in series.index or not np.isfinite(cut):
                return None
            d = series.at[fight.Index]
            if not np.isfinite(d) or abs(d) < cut or d == 0:
                return None
            return "red" if d > 0 else "blue"
        return marked
    return {name: side_of(name) for name in values}


def overlap(marks, rows, est):
    """Per population: marked rows shared with arm 1, and the share whose
    marked side has no measurement of its own."""
    arm1 = marks["regain_edge_top_third"]
    out = {}
    for name, m in marks.items():
        n = int(m.sum())
        shared = int((m & arm1).sum())
        sides_ = rows.loc[m, ["fight", "side"]]
        unmeasured = [int(est.at[f, f"n_{s}"]) == 0 for f, s in zip(sides_["fight"], sides_["side"])]
        out[name] = {"marked": n, "shared_with_arm1": shared,
                     "shared_with_arm1_share": float(shared / n) if n else float("nan"),
                     "marked_side_unmeasured": int(sum(unmeasured)),
                     "marked_side_unmeasured_share": float(np.mean(unmeasured)) if n else float("nan")}
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def clustered_logit(y, X, clusters, iterations=50):
    """Logistic regression by Newton's method with cluster-robust (sandwich)
    standard errors. Returns (beta, se)."""
    y, X, clusters = np.asarray(y, float), np.asarray(X, float), np.asarray(clusters)
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ beta))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
    Hinv = np.linalg.inv(H)
    scores = X * (y - p)[:, None]
    meat = np.zeros_like(H)
    for c in np.unique(clusters):
        s = scores[clusters == c].sum(axis=0)
        meat += np.outer(s, s)
    g = len(np.unique(clusters))
    n, k = X.shape
    adjust = g / (g - 1) * (n - 1) / (n - k) if g > 1 and n > k else np.nan
    V = Hinv @ meat @ Hinv * adjust
    return beta, np.sqrt(np.diag(V))


def secondary_logit(est, frame, priced):
    """Design 4.4's secondary check on the arm-1 priced fights: does
    delta_pct add to the market's logit? Not a ship criterion."""
    e = est[est["eligible"] & est.index.isin(priced) & ((est["n_red"] > 0) | (est["n_blue"] > 0))]
    e = e[np.isfinite(e["delta_pct"])]
    out = {"n": int(len(e)), "events": int(e["event"].nunique())}
    if len(e) < 20 or e["event"].nunique() < 3:
        out["note"] = "too few fights for a regression"
        return out
    y = frame.loc[e.index, "won"].to_numpy(float)
    X = np.column_stack([np.ones(len(e)), _logit(frame.loc[e.index, "p_market"]), e["delta_pct"].to_numpy()])
    beta, se = clustered_logit(y, X, e["event"].to_numpy())
    out.update({"delta_pct_per_point": float(beta[2]), "se": float(se[2]),
                "z": float(beta[2] / se[2]) if se[2] > 0 else float("nan"),
                "logit_market": float(beta[1]), "logit_market_se": float(se[1]),
                "note": "red win on logit(market) + delta_pct, errors clustered by event"})
    return out


def coverage(est, frame, labels):
    """How many fights the labels reach - the honest size of the test."""
    priced = frame.index[np.isfinite(frame["p_blend"])]
    since = est[est["date"] > labels["published_at"].min()]
    either = since[(since["n_red"] > 0) | (since["n_blue"] > 0)]
    both = since[(since["n_red"] > 0) & (since["n_blue"] > 0)]
    measured = set()
    for fight in either.itertuples():
        row = frame.loc[fight.Index]
        if fight.n_red > 0:
            measured.add(row["red"])
        if fight.n_blue > 0:
            measured.add(row["blue"])
    return {"labels": int(len(labels)), "label_events": int(labels["event"].nunique()),
            "usable_labels": int(len(tfw.RegainEstimator(labels).available(pd.Timestamp.max))),
            "first_publication": str(labels["published_at"].min().date()),
            "fights_since_first_publication": int(len(since)),
            "priced_since_first_publication": int(since.index.isin(priced).sum()),
            "eligible_priced": int((since["eligible"] & since.index.isin(priced)).sum()),
            "with_a_measured_fighter": int(len(either)),
            "with_a_measured_fighter_priced": int(either.index.isin(priced).sum()),
            "distinct_measured_fighters": int(len(measured)),
            "with_two_measured_fighters": int(len(both)),
            "with_two_measured_fighters_priced": int(both.index.isin(priced).sum())}


def main():
    archive = tfw.load_archive()
    world = tfw.load_world()
    print("validating the estimator forward-chained...")
    labels, validation, forward = tfw.full_validation(archive, world)
    tfw._print_report(validation)
    frame_earned = bool(validation["frame_features"].get("earned"))
    gate = validation["gate"]
    print("\nbuilding predictions...")
    frame = predictions()
    rows = sides(frame)
    missed = misses()
    est = estimates(frame, labels, archive, missed, forward, frame_earned)
    placebo = estimates(frame, placebo_labels(labels), None, missed, forward, False)
    priced = frame.index[np.isfinite(frame["p_blend"])]
    cover = coverage(est, frame, labels)
    print("\n  coverage: " + ", ".join(f"{k} {v}" for k, v in cover.items()))
    cuts = thresholds(est, priced)
    for arm, (cut, n) in cuts.items():
        print(f"  threshold {arm:<28} |delta| >= {cut:.3f} (frozen on the first {n} eligible priced fights)")
    same = arm_values(est, priced)
    differ = int((np.abs(same["regain_edge_top_third"] - same["tfw_gap_top_third"]) > 1e-9).sum())
    print(f"  arm 2 differs from arm 1 on {differ} of {len(same['tfw_gap_top_third'])} priced eligible "
          f"fights (a recorded missed weight); it is otherwise the same test")
    print(f"  gate: {'PASSED' if gate['passed'] else 'NOT PASSED' if gate.get('testable', True) else 'NOT TESTABLE'}"
          f" - {gate['reason']}" + ("" if gate["passed"] else "; arms run for the coverage counts, nothing may ship"))
    pops = populations(est, cuts, placebo)
    marks = {name: mark(frame, rows, side) for name, side in pops.items()}
    shares = overlap(marks, rows, est)
    header(FAMILY)
    results = []
    for name, marked in marks.items():
        result = measure(rows, marked, name, family=FAMILY)
        result["placebo"] = name == PLACEBO
        result["threshold"] = cuts.get(name, cuts["regain_edge_top_third"])[0]
        result.update(shares[name])
        if name == "tfw_gap_top_third":
            result["duplicates_arm1_except_missed_weight"] = True
            result["t1_odds_timing_rule_met"] = False
        results.append(result)
    for name, s in shares.items():
        print(f"    {name:<40} shares {s['shared_with_arm1']} of {s['marked']} marked rows with arm 1; "
              f"marked side unmeasured on {s['marked_side_unmeasured']}")
    logit = secondary_logit(est, frame, priced)
    print("\n  secondary (not a ship criterion): " + (
        f"delta_pct {logit['delta_pct_per_point']:+.4f} per point (se {logit['se']:.4f}, z {logit['z']:+.2f}) "
        f"over {logit['n']} fights in {logit['events']} events, {logit['note']}"
        if "delta_pct_per_point" in logit else f"{logit['note']} ({logit['n']} fights)"))
    beyond = [r for r in results if r.get("verdict") == "REAL beyond the market"
              and not r["placebo"]]
    placebo_result = next(r for r in results if r["placebo"])
    print("\n  " + ("BEYOND THE MARKET: " + ", ".join(r["label"] for r in beyond)
                    if beyond else "NOTHING clears the family-wise bar against the market"
                    + (" - most arms are too few to test" if any(r.get("too_few")
                                                               for r in results) else "."))
          + (f"\n  placebo: {placebo_result.get('verdict', 'too few to test')}")
          + ("" if gate["passed"] else "\n  the bout gate was not passed, so nothing could ship regardless"))
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "family": FAMILY, "arms": ARMS, "placebo": PLACEBO,
        "gate": gate,
        "k_for_next_fight": validation["k_final"],
        "k_identified": validation["k_identified"],
        "frame_features_earned": frame_earned,
        "validation": validation, "coverage": cover,
        "thresholds": {arm: {"abs_delta_at_least": cut, "frozen_on": n}
                       for arm, (cut, n) in cuts.items()},
        "arm2_differs_from_arm1_on": differ,
        "secondary_logit": logit,
        "results": results}, indent=1, default=float))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
