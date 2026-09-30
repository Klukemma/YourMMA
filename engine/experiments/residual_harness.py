"""The shared test for "does X move results beyond the model and the market?"

Every soft factor - late replacements, injuries, camp power, mentality -
faces the same test, so the answers are comparable and none gets a
friendlier one. Rebuilt after an adversarial review found five ways the
first version flattered its populations:

  1. PRICED FIGHTS ONLY, BOTH ARMS. The blended probability fell back to the
     model's own number wherever there were no odds (all of 2025), so
     16-24% of every "vs blend" arm was really "vs model". Now a fight
     without a price is out of both arms, and the model arm is measured on
     the same fights as the blend arm.
  2. AGE AND CALIBRATION CONTROLLED. Every career-built signal is 2-3 years
     older or younger than its opponents, and age alone beats the market -
     so a raw residual partly measures age. The predictor is also
     miscalibrated by probability band (favourite/longshot). Each marked
     fighter's residual is now compared with UNMARKED fighters in the same
     win-probability decile and the same age-gap band, and only the
     difference counts.
  3. FAMILY-WISE INTERVALS. Eight populations at a 95% bar give about a
     one-in-three chance of a false "real". The interval is widened to
     1 - 0.05/family.
  4. FIGHTERS, NOT JUST EVENTS. A sticky signal puts the same few fighters
     in the sample dozens of times. Intervals are bootstrapped over events
     and over fighters, and the wider one is reported; a population needs
     MIN_FIGHTERS distinct fighters as well as MIN_LABELS rows.
  5. Winners spelled like their corners (name_resolution.canonical_winners).

A factor ships only if its adjusted, family-wise interval against the BLEND
excludes zero.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.calibrator_mismatch import fit_models, proba
from experiments.short_notice_weight import FIRST_PREDICTED_YEAR, MIN_TRAIN

MIN_LABELS = 150
MIN_FIGHTERS = 40
DRAWS = 5000
AGE_BANDS = [-99, -6, -3, -1, 1, 3, 6, 99]


def predictions():
    """One row per archive fight predicted out of sample: p_model, p_market,
    p_blend (NaN without a price), red/blue names, ages, date, event, won."""
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/residual_harness.json")
    import predict_card as engine
    import market_blend as blend
    from name_resolution import norm_name

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    meta = engine.ufc_valid.reset_index(drop=True)
    dates = pd.to_datetime(meta["date"])
    years = dates.dt.year

    rows = []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).to_numpy()
        test = (years == year).to_numpy()
        if train.sum() < MIN_TRAIN or test.sum() < 50:
            continue
        models = fit_models(X[train], y[train])
        p = proba(models, X[test], 3)
        for position, row in enumerate(np.flatnonzero(test)):
            rows.append({"i": int(row), "p_model": float(p[position])})
        print(f"    {year}: {test.sum()} fights", flush=True)

    frame = pd.DataFrame(rows).set_index("i")
    frame["date"] = dates.loc[frame.index].values
    frame["event"] = meta.loc[frame.index, "event_name"].values
    frame["red_raw"] = meta.loc[frame.index, "r_name"].values
    frame["blue_raw"] = meta.loc[frame.index, "b_name"].values
    frame["red"] = [norm_name(n) for n in frame["red_raw"]]
    frame["blue"] = [norm_name(n) for n in frame["blue_raw"]]
    frame["won"] = y[frame.index]
    for side, col in (("red", "r_dob"), ("blue", "b_dob")):
        if col in meta.columns:
            dob = pd.to_datetime(meta.loc[frame.index, col], errors="coerce")
            frame[f"{side}_age"] = ((pd.to_datetime(frame["date"]) - dob.values)
                                    .dt.days / 365.25).values
        else:
            frame[f"{side}_age"] = np.nan

    odds = pd.read_csv(ENGINE / "data" / "odds.csv")
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    priced = {}
    for a, b, oa, ob, d in zip(odds.get("fighter_a", []),
                               odds.get("fighter_b", []),
                               odds.get("odds_a", []), odds.get("odds_b", []),
                               odds.get("date", [])):
        if pd.isna(d) or pd.isna(oa) or pd.isna(ob):
            continue
        priced[(norm_name(a), norm_name(b), d.date())] = (oa, ob)

    def market_for(row):
        day = pd.Timestamp(row.date).date()
        for key, flip in (((row.red, row.blue, day), False),
                          ((row.blue, row.red, day), True)):
            if key in priced:
                a, b = priced[key]
                return blend.devig(b, a) if flip else blend.devig(a, b)
        return np.nan

    frame["p_market"] = [market_for(r) for r in frame.itertuples()]
    # NO FALLBACK. An unpriced fight has no blended probability, and
    # pretending the model's number is one put the model's residual into
    # the arm that decides what ships.
    frame["p_blend"] = [blend.blend(m, k) if np.isfinite(k) else np.nan
                        for m, k in zip(frame.p_model, frame.p_market)]
    return frame


def sides(frame):
    """Two rows per PRICED fight, one from each fighter's side."""
    priced = frame[np.isfinite(frame["p_blend"])]
    red = pd.DataFrame({
        "fight": priced.index, "event": priced["event"].values,
        "date": priced["date"].values, "fighter": priced["red"].values,
        "fighter_raw": priced["red_raw"].values,
        "opponent_raw": priced["blue_raw"].values, "side": "red",
        "won": priced["won"].values, "p_model": priced["p_model"].values,
        "p_blend": priced["p_blend"].values,
        "age_diff": (priced["red_age"] - priced["blue_age"]).values})
    blue = pd.DataFrame({
        "fight": priced.index, "event": priced["event"].values,
        "date": priced["date"].values, "fighter": priced["blue"].values,
        "fighter_raw": priced["blue_raw"].values,
        "opponent_raw": priced["red_raw"].values, "side": "blue",
        "won": 1.0 - priced["won"].values,
        "p_model": 1.0 - priced["p_model"].values,
        "p_blend": 1.0 - priced["p_blend"].values,
        "age_diff": (priced["blue_age"] - priced["red_age"]).values})
    rows = pd.concat([red, blue], ignore_index=True)
    rows["age_band"] = pd.cut(rows["age_diff"].fillna(0), AGE_BANDS,
                              labels=False)
    for column in ("p_model", "p_blend"):
        rows[f"{column}_band"] = np.minimum((rows[column] * 10).astype(int), 9)
    return rows


def mark(frame, rows, marked):
    """Boolean over `rows`: marked(fight row) returns red/blue/both/None."""
    chosen = set()
    for fight in frame[np.isfinite(frame["p_blend"])].itertuples():
        side = marked(fight)
        if side in ("red", "both"):
            chosen.add((fight.Index, "red"))
        if side in ("blue", "both"):
            chosen.add((fight.Index, "blue"))
    return np.array([(f, s) in chosen for f, s in zip(rows["fight"],
                                                      rows["side"])])


def _interval(values, groups, alpha, seed=0):
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return float("nan"), float("nan")
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(unique), len(unique))
        means[i] = values[np.concatenate([index[unique[j]] for j in pick])].mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def measure(rows, marked, label, family=1, control_age=True):
    """Adjusted residual of the marked side against model and blend.

    control_age=False only for a population that IS an age difference -
    controlling for age there would subtract the thing being measured.
    """
    out = {"label": label, "n": int(marked.sum()),
           "fighters": int(rows.loc[marked, "fighter"].nunique()),
           "events": int(rows.loc[marked, "event"].nunique()),
           "family": family}
    if out["n"] < MIN_LABELS or out["fighters"] < MIN_FIGHTERS:
        out["too_few"] = True
        print(f"  {label:<40}{out['n']:>6} rows, {out['fighters']:>4} "
              f"fighters - too few to report")
        return out
    alpha = 0.05 / family
    for column in ("p_model", "p_blend"):
        resid = rows["won"] - rows[column]
        age = rows["age_band"] if control_age else pd.Series(0, index=rows.index)
        baseline = resid[~marked].groupby(
            [rows[f"{column}_band"][~marked], age[~marked]]).mean()
        keys = list(zip(rows.loc[marked, f"{column}_band"], age[marked]))
        expected = np.array([baseline.get(k, np.nan) for k in keys])
        # A marked row whose cell holds no unmarked fighter has nothing to be
        # compared with - a population that IS an age band, say. It is left
        # out, and the share left in is reported, rather than quietly
        # compared with the league average.
        matched = ~np.isnan(expected)
        out[f"{column.split('_')[1]}_matched"] = float(matched.mean())
        keep_events = rows.loc[marked, "event"].to_numpy()[matched]
        keep_fighters = rows.loc[marked, "fighter"].to_numpy()[matched]
        expected = expected[matched]
        adjusted = resid[marked].to_numpy()[matched] - expected
        if len(adjusted) < MIN_LABELS:
            out["too_few"] = True
            print(f"  {label:<40} only {len(adjusted)} marked rows have a "
                  f"matched comparison - not reported")
            return out
        by_event = _interval(adjusted, keep_events, alpha)
        by_fighter = _interval(adjusted, keep_fighters, alpha)
        low = min(by_event[0], by_fighter[0])
        high = max(by_event[1], by_fighter[1])
        arm = column.split("_")[1]
        out[f"{arm}_raw"] = float(resid[marked].mean())
        out[f"{arm}_adjusted"] = float(adjusted.mean())
        out[f"{arm}_low"], out[f"{arm}_high"] = low, high
    verdict = ("REAL beyond the market" if out["blend_low"] > 0
               or out["blend_high"] < 0 else
               "real vs the model only" if out["model_low"] > 0
               or out["model_high"] < 0 else "inside the noise")
    out["verdict"] = verdict
    print(f"  {label:<40}{out['n']:>6}{out['fighters']:>5}  "
          f"model {out['model_adjusted']:+.4f} [{out['model_low']:+.3f},"
          f"{out['model_high']:+.3f}]  blend {out['blend_adjusted']:+.4f} "
          f"[{out['blend_low']:+.3f},{out['blend_high']:+.3f}]  {verdict}")
    return out


def header(family):
    print(f"\n  priced fights only; residuals adjusted for win-probability "
          f"band and age gap;\n  intervals family-wise over {family} "
          f"populations, the wider of event- and fighter-resampled")
    print(f"  {'population (marked side)':<40}{'rows':>6}{'ftrs':>5}  "
          f"{'adjusted residual [interval]'}")
