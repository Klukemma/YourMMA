"""Idea #3: is a fitted stack a better blend with the market than 0.75?

The app's headline probability is 0.25 * model + 0.75 * market, in
probability space, the weight chosen by walk-forward ACCURACY
(market_blend.py). A fixed probability-space average is the crudest
combiner there is: it cannot learn that the market is sharper on heavy
favourites, that the model is worth less when a fighter has no UFC record,
or that the market's own prices carry a favourite-longshot bias. This asks
whether any of those, fitted honestly, beats the fixed weight.

PRE-REGISTERED PLAN - written before the first run and reported unchanged.

Data. The baseline walk-forward predictions from experiments/model_compare.py
(production recipe: LR + RF + XGB averaged and Platt-calibrated, each year
fitted on earlier years only), with the de-vigged closing line where
odds.csv prices the fight. Only priced fights are in any arm; 2025 is
unpriced and is simply absent.

Baseline arm. p = clip(0.25 * p_model + 0.75 * p_market, 0.01, 0.99),
exactly market_blend.blend with MARKET_WEIGHT = 0.75. Nothing is fitted.

Candidate arms - a family of three, so alpha = 0.05 / 3:

  (1) stack         logit(p) = a * logit(market) + b * logit(model) + c
  (2) stack+thin    (1) + d * logit(model) * thin,
                    thin = 1 / (1 + min(red_prior_bouts, blue_prior_bouts)),
                    where prior_bouts is each corner's UFC bouts strictly
                    before the fight in either corner (model_compare.
                    prior_bouts). thin is 1 for a debutant on either side,
                    1/2 after one bout, 1/11 after ten. Both fighters have
                    it; it is symmetric under a corner swap, so it enters
                    only through the model term and has no main effect.
  (3) stack+FL      (1) + a2 * logit(market) * |logit(market)|: a
                    favourite-longshot correction of the market term. This
                    form is chosen over isotonic/Platt-on-market-alone
                    because it is one coefficient and stays monotone.

Fitting. Every arm is an unregularised logistic regression (sklearn
LogisticRegression, C = 1e6, lbfgs) on the stated design. For each
predicted year Y the coefficients are fitted ONLY on priced fights of years
< Y (expanding window) using the WALK-FORWARD model probabilities, never
in-sample ones, and need at least MIN_STACK_TRAIN = 200 such fights; the
first year with that many is the first scored year. Probabilities are
clipped to [0.001, 0.999] before the logit. The comparison set is the
priced fights of every scored year, identical for every arm and for the
baseline.

Primary metric and pass bar. Log loss on the comparison set. An arm passes
if the paired bootstrap interval of (arm - baseline) log loss - the wider
of event-resampled and fighter-resampled, at the family-wise level
1 - 0.05/3 - lies entirely below zero. Nothing else is a pass.

Secondary, reported but not the bar: accuracy, Brier, calibration by
decile, and the same table on 2020+ only (the confirm period of
historical_backtest.py). Per-year coefficients are printed for the record.

Shipping. Only an arm that passes is implemented in market_blend.py, with
coefficients fitted on the full priced history and the date range stated.
If none passes, market_blend.py is not touched.

POST-REGISTRATION NOTES - added after the first run; nothing above this line
changed (its sha256 is in stacked_blend_plan.sha256 and main() refuses to run
if it moves), and the pass bar is exactly the one above.

  * thin. The plan says "UFC bouts strictly before the fight"; the first run's
    prior_bouts counted DECIDED bouts only (draws and no-contests dropped
    before counting). model_compare now counts every post-2001 archive row,
    as the plan's wording says, and the experiment was re-run. Both runs'
    numbers are in the write-up; the verdict did not move.
  * The baseline is not point-in-time, and the comparison is asymmetric.
    The 0.75 weight was chosen on the whole priced history (market_blend.py)
    and is applied in every scored year, while the candidates refit on years
    < Y. The asymmetry favours the BASELINE (it has seen the future), so it
    cannot manufacture a pass; it is reported here because the harness is
    meant to be the shared yardstick. For the record a second baseline,
    NOT pre-registered and NOT the bar, is reported alongside: "pit-weight",
    p = clip((1 - w_Y) * model + w_Y * market, 0.01, 0.99) with w_Y the grid
    value in {0, 0.05, ..., 1} that minimises LOG LOSS (the pre-registered
    metric) on priced fights of years < Y, same expanding window and
    MIN_STACK_TRAIN as the candidates. The accuracy-chosen weight per year
    - market_blend.py's stated procedure - is printed too, because under
    the production recipe it does not come back 0.75 "for every year".
  * Staleness. Harness models for year Y are fitted through 31 December of
    Y-1 (model_compare docstring); production retrains before every card.
    The shipping coefficients of a passing arm would therefore be fitted on
    staler model probabilities than they are applied to; before any ship,
    refit on a finer walk-forward or measure the fresh-versus-stale shift.
  * Family. The reference baseline adds no candidate; alpha stays 0.05 / 3.

    python engine/experiments/stacked_blend.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import market_blend  # noqa: E402
from experiments import model_compare as mc  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
PLAN_FILE = Path(__file__).with_name("stacked_blend_plan.sha256")
MIN_STACK_TRAIN = 200
CONFIRM_FROM = 2020
STACK_C = 1e6
CLIP = 1e-3
FAMILY = 3
ALPHA = 0.05 / FAMILY
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PLAN_START = "PRE-REGISTERED PLAN"
PLAN_END = "POST-REGISTRATION NOTES"


def plan_text(doc=None):
    """The pre-registered section of the docstring, and nothing after it."""
    doc = __doc__ if doc is None else doc
    return doc[doc.index(PLAN_START):doc.index(PLAN_END)]


def plan_sha256(doc=None):
    return hashlib.sha256(plan_text(doc).encode()).hexdigest()


def check_plan(path=PLAN_FILE, doc=None):
    """Register the plan's hash the first time; refuse to run if it has
    moved since. Returns {"sha256", "registered_at"}."""
    current = plan_sha256(doc)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored["sha256"] != current:
            raise SystemExit(
                f"the pre-registered plan has changed since it was registered "
                f"({stored['registered_at']}): {stored['sha256'][:12]} -> {current[:12]}. "
                f"Restore the plan, or register a new experiment under a new name.")
        return stored
    stored = {"sha256": current,
              "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path.write_text(json.dumps(stored, indent=1) + "\n")
    return stored


def logit(p):
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def thinness(frame):
    return 1.0 / (1.0 + np.minimum(frame["red_prior_bouts"].to_numpy(float),
                                   frame["blue_prior_bouts"].to_numpy(float)))


def design_stack(frame):
    return np.column_stack([logit(frame["p_market"]), logit(frame["p_model"])])


def design_thin(frame):
    lm = logit(frame["p_model"])
    return np.column_stack([logit(frame["p_market"]), lm, lm * thinness(frame)])


def design_fl(frame):
    lk = logit(frame["p_market"])
    return np.column_stack([lk, logit(frame["p_model"]), lk * np.abs(lk)])


ARMS = {
    "stack": (design_stack, ["a_market", "b_model"]),
    "stack+thin": (design_thin, ["a_market", "b_model", "d_model_x_thin"]),
    "stack+FL": (design_fl, ["a_market", "b_model", "a2_market_x_abs"]),
}


def baseline(frame):
    return np.array([market_blend.blend(m, k)
                     for m, k in zip(frame["p_model"], frame["p_market"])])


def weighted_blend(w, p_model, p_market):
    """market_blend.blend's arithmetic for any weight: probability space,
    clipped to [0.01, 0.99]."""
    return np.clip((1.0 - w) * np.asarray(p_model, float) + w * np.asarray(p_market, float),
                   0.01, 0.99)


def best_weight(frame, metric="log_loss", grid=WEIGHT_GRID):
    """The grid weight that minimises log loss (or maximises accuracy) on
    the rows given - the caller passes EARLIER years only."""
    y = frame["y"].to_numpy(float)
    scores = []
    for w in grid:
        p = weighted_blend(w, frame["p_model"], frame["p_market"])
        scores.append(mc.log_loss_rows(y, p).mean() if metric == "log_loss"
                      else -mc.correct_rows(y, p).mean())
    return float(grid[int(np.argmin(scores))])


def walk_forward_weight(priced, metric="log_loss", verbose=True):
    """The reference baseline (post-registration, not the bar): each year's
    weight chosen on earlier priced years only, by `metric`."""
    p = np.full(len(priced), np.nan)
    weights = {}
    years = priced["year"].to_numpy()
    for year in sorted(np.unique(years)):
        train = years < year
        if train.sum() < MIN_STACK_TRAIN:
            continue
        w = best_weight(priced[train], metric)
        test = years == year
        p[test] = weighted_blend(w, priced.loc[test, "p_model"], priced.loc[test, "p_market"])
        weights[int(year)] = {"weight": w, "fitted_on": int(train.sum())}
    if verbose:
        print(f"\n  pit-weight by {metric}: market weight chosen on earlier priced fights")
        print("    " + "  ".join(f"{y}:{v['weight']:.2f}" for y, v in weights.items()))
    return p, weights


def fit_arm(design, frame):
    model = LogisticRegression(C=STACK_C, solver="lbfgs", max_iter=5000)
    model.fit(design(frame), frame["y"].to_numpy(float))
    return model


def walk_forward_arm(name, priced, verbose=True):
    """Each year's stack fitted on earlier priced walk-forward rows only."""
    design, names = ARMS[name]
    p = np.full(len(priced), np.nan)
    coefficients = {}
    years = priced["year"].to_numpy()
    for year in sorted(np.unique(years)):
        train = years < year
        if train.sum() < MIN_STACK_TRAIN:
            continue
        model = fit_arm(design, priced[train])
        test = years == year
        p[test] = model.predict_proba(design(priced[test]))[:, 1]
        coefficients[int(year)] = {
            **{n: float(v) for n, v in zip(names, model.coef_[0])},
            "c": float(model.intercept_[0]), "fitted_on": int(train.sum())}
    if verbose:
        print(f"\n  {name}: coefficients by predicted year (fitted on earlier priced fights)")
        for year, c in coefficients.items():
            print("    " + f"{year}: " + "  ".join(f"{k}={v:+.3f}" for k, v in c.items()
                                                    if k != "fitted_on")
                  + f"  (n={c['fitted_on']})")
    return p, coefficients


def passes_bar(delta):
    """THE pre-registered bar: the wider interval of (arm - baseline) log
    loss lies entirely below zero. `delta` is compare()["delta"]["log_loss"];
    "high" is already the max of the event and fighter upper bounds."""
    return bool(delta["high"] < 0)


def table(frame, arms, label, alpha=ALPHA, base_col="p_baseline",
          base_name="baseline 0.75", decides=True):
    """Every arm against the column `base_col`. Only the pre-registered
    baseline (decides=True) can pass an arm; a reference baseline reports
    the same numbers with no verdict."""
    print(f"\n{'=' * 79}\n{label}: {len(frame):,} priced fights, "
          f"{frame['event'].nunique():,} events, years "
          f"{frame['year'].min()}-{frame['year'].max()}\n{'=' * 79}")
    y = frame["y"].to_numpy()
    base = frame[base_col].to_numpy()
    results = {"n": int(len(frame)), "years": [int(v) for v in sorted(frame["year"].unique())],
               "baseline_arm": base_name, "decides": bool(decides),
               "baseline": mc.metrics(y, base), "arms": {}}
    print(f"  {'arm':<14}{'log loss':>10}{'delta':>9}  {'interval (wider of event/fighter)':<36}"
          f"{'accuracy':>9}{'brier':>8}  " + ("verdict" if decides else "(no verdict)"))
    b = results["baseline"]
    print(f"  {base_name:<14}{b['log_loss']:>10.4f}{'':>9}  {'':<36}"
          f"{b['accuracy']:>9.1%}{b['brier']:>8.4f}")
    for name in arms:
        r = mc.compare(y, base, frame[f"p_{name}"].to_numpy(), frame["event"].to_numpy(),
                       red=frame["red_norm"].to_numpy(), blue=frame["blue_norm"].to_numpy(),
                       alpha=alpha, label_cand=name)
        d = r["delta"]["log_loss"]
        passes = passes_bar(d) if decides else None
        r["passes"] = passes
        r["calibration"] = mc.calibration_by_decile(y, frame[f"p_{name}"].to_numpy())
        results["arms"][name] = r
        m = r[name]
        interval = f"[{d['low']:+.4f}, {d['high']:+.4f}]"
        print(f"  {name:<14}{m['log_loss']:>10.4f}{d['mean']:>+9.4f}  {interval:<36}"
              f"{m['accuracy']:>9.1%}{m['brier']:>8.4f}  "
              + ("" if passes is None else "PASS" if passes else "no"))
        acc, bri = r["delta"]["accuracy"], r["delta"]["brier"]
        print(f"  {'':<14}{'':>19}  event [{d['event_low']:+.4f}, {d['event_high']:+.4f}]  "
              f"fighter [{d['fighter_low']:+.4f}, {d['fighter_high']:+.4f}]")
        print(f"  {'':<14}{'':>19}  accuracy delta {acc['mean']:+.4f} "
              f"[{acc['low']:+.4f}, {acc['high']:+.4f}]; brier delta {bri['mean']:+.4f} "
              f"[{bri['low']:+.4f}, {bri['high']:+.4f}]")
    results["baseline_calibration"] = mc.calibration_by_decile(y, base)
    return results


def print_calibration(frame, arms):
    print("\n  calibration by predicted-probability decile (red corner), all scored years")
    cols = ["baseline", "pit"] + list(arms)
    print(f"  {'decile':<10}{'n':>6}  " + "".join(f"{c[:14]:>22}" for c in cols))
    print(f"  {'':<10}{'':>6}  " + "".join(f"{'pred / obs':>22}" for _ in cols))
    y = frame["y"].to_numpy()
    per = {c: {r["decile"]: r for r in mc.calibration_by_decile(y, frame[f"p_{c}"].to_numpy())}
           for c in cols}
    for d in range(10):
        if d not in per["baseline"]:
            continue
        row = per["baseline"][d]
        print(f"  {row['lo']:.1f}-{row['hi']:.1f}{'':<3}{row['n']:>6}  " + "".join(
            f"{per[c][d]['predicted']:>10.3f} / {per[c][d]['observed']:.3f}"
            if d in per[c] else f"{'-':>22}" for c in cols))


def main():
    registered = check_plan()
    print(f"plan sha256 {registered['sha256'][:12]} registered {registered['registered_at']}")
    print("baseline walk-forward predictions (cached if the archive is unchanged)...")
    frame = mc.predictions(verbose=True)
    priced = frame[np.isfinite(frame["p_market"])].reset_index(drop=True).copy()
    print(f"\n{len(priced):,} priced of {len(frame):,} walk-forward predictions; "
          f"unpriced years: {sorted(set(frame['year']) - set(priced['year']))}")
    priced["p_baseline"] = baseline(priced)
    thin = thinness(priced)
    print(f"thin: mean {thin.mean():.3f}, share of fights with a debutant {np.mean(thin == 1):.1%}")

    coefficients = {}
    for name in ARMS:
        priced[f"p_{name}"], coefficients[name] = walk_forward_arm(name, priced)
    # Post-registration reference baseline (not the bar), and the
    # accuracy-chosen weight market_blend.py describes, for the record.
    priced["p_pit"], pit_weights = walk_forward_weight(priced, "log_loss")
    _, acc_weights = walk_forward_weight(priced, "accuracy")
    scored = priced[np.isfinite(priced[[f"p_{n}" for n in ARMS] + ["p_pit"]])
                    .all(axis=1)].reset_index(drop=True)
    print(f"\ncomparison set: {len(scored):,} priced fights in scored years "
          f"{scored['year'].min()}-{scored['year'].max()} (first year with "
          f">= {MIN_STACK_TRAIN} earlier priced fights); family of {FAMILY}, alpha {ALPHA:.4f}")

    confirm = scored[scored["year"] >= CONFIRM_FROM].reset_index(drop=True)
    results = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "plan": __doc__,
        "plan_sha256": registered["sha256"], "plan_registered_at": registered["registered_at"],
        "odds_sha256": frame.attrs.get("odds_sha256"),
        "family": FAMILY, "alpha": ALPHA, "min_stack_train": MIN_STACK_TRAIN,
        "priced_rows": int(len(priced)), "comparison_rows": int(len(scored)),
        "coefficients_by_year": coefficients,
        "pit_weight_by_year": {"log_loss": pit_weights, "accuracy": acc_weights},
        "all_years": table(scored, ARMS, "PRIMARY - all scored years"),
        "confirm": table(confirm, ARMS, f"SECONDARY - {CONFIRM_FROM} onward"),
        "reference_pit_weight": {
            "note": "post-registration reference baseline, weight chosen on years < Y "
                    "by log loss; reported, never the bar",
            "all_years": table(scored, ARMS, "REFERENCE (not the bar) - all scored years, "
                               "against pit-weight baseline", base_col="p_pit",
                               base_name="pit-weight", decides=False),
            "confirm": table(confirm, ARMS, f"REFERENCE (not the bar) - {CONFIRM_FROM} onward, "
                             "against pit-weight baseline", base_col="p_pit",
                             base_name="pit-weight", decides=False),
            "pit_vs_0.75": mc.compare(scored["y"].to_numpy(), scored["p_baseline"].to_numpy(),
                                      scored["p_pit"].to_numpy(), scored["event"].to_numpy(),
                                      red=scored["red_norm"].to_numpy(),
                                      blue=scored["blue_norm"].to_numpy(),
                                      alpha=ALPHA, label_cand="pit-weight"),
        },
    }
    print("\n  pit-weight baseline against the 0.75 baseline (for the record):")
    print(mc.format_compare(results["reference_pit_weight"]["pit_vs_0.75"],
                            label_cand="pit-weight"))
    print_calibration(scored, ARMS)

    passing = [n for n in ARMS if results["all_years"]["arms"][n]["passes"]]
    results["passing"] = passing
    if passing:
        best = min(passing, key=lambda n: results["all_years"]["arms"][n][n]["log_loss"])
        results["verdict"] = f"PASS: {', '.join(passing)}; lowest log loss {best}"
        # Coefficients on the full priced history, for shipping.
        design, names = ARMS[best]
        model = fit_arm(design, priced)
        results["ship"] = {
            "arm": best,
            "coefficients": {**{n: float(v) for n, v in zip(names, model.coef_[0])},
                             "c": float(model.intercept_[0])},
            "fitted_on": int(len(priced)),
            "date_range": [str(priced["date"].min().date()), str(priced["date"].max().date())],
            "caveat": "fitted on harness (up to a year stale) model probabilities; refit on a "
                      "finer walk-forward or measure the fresh-vs-stale shift before shipping",
        }
    else:
        results["verdict"] = ("NO ARM PASSES: no stacked blend's log loss interval lies "
                              "below the 0.75 blend at the family-wise level; "
                              "market_blend.py unchanged")
    print(f"\nVERDICT: {results['verdict']}")
    OUT.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
