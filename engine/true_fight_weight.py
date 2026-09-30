"""True Fight Weight: how heavy a fighter actually is when the bell rings.

The division limit says everyone at lightweight fights at 155. The measured
fight-night weights say otherwise: median regain about 10%, the middle half
7-13%, individuals from -1% to +19% (fight_night_weights.csv). At 155 that
spreads fighters from about 166 to 175 lb, and further at the tails.

This estimates each fighter's regain for a fight from what was knowable
BEFORE it, and turns that into a size difference between opponents:

  features (all pre-fight, all from the archive's earlier rows)
    limit          the division's weight limit
    women          women's division
    height_z       height against the division's typical height - a tall
                   frame at a given limit carries more mass back
    reach_z        the same for reach
    age            older fighters are thought to cut and regain less
    came_down      the heaviest limit they fought at BEFORE this bout, minus
                   this one: moving down means a bigger cut, so a bigger regain
    prior_regain   their own earlier measured regain, where one exists
  target           regain as % of the official weigh-in weight

  validation       leave one EVENT out: every event's regains are predicted
                   by a model that never saw that event, and the error is
                   compared with a naive estimate (the division's median).

A fight-night weight measured at a fight is never used for that fight: it
is taken on fight day and published after the bout. Only EARLIER measured
regains may inform a later estimate.

    python engine/true_fight_weight.py --validate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from name_resolution import canonical_winners, norm_name

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
MEASURED = ENGINE / "data" / "fight_night_weights.csv"

LIMITS = {"strawweight": 115, "flyweight": 125, "bantamweight": 135,
          "featherweight": 145, "lightweight": 155, "welterweight": 170,
          "middleweight": 185, "light heavyweight": 205, "heavyweight": 265}
FEATURES = ["limit", "women", "height_z", "reach_z", "age", "came_down",
            "has_prior", "prior_regain"]


def limit_of(division):
    d = str(division).lower()
    for name in sorted(LIMITS, key=len, reverse=True):   # "light heavyweight"
        if name in d:                                    # before "heavyweight"
            return LIMITS[name]
    return np.nan


class Archive:
    """Per-fighter bouts from the UFC archive, for as-of-date features."""

    def __init__(self, frame):
        a = canonical_winners(frame).copy()
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        a["limit"] = a["division"].map(limit_of)
        a["women"] = a["division"].astype(str).str.contains("women", case=False)
        self.bouts = {}
        rows = []
        for side in ("r", "b"):
            part = pd.DataFrame({
                "fighter": a[f"{side}_name"].map(norm_name), "date": a["date"],
                "limit": a["limit"], "women": a["women"],
                "height": a[f"{side}_height"], "reach": a[f"{side}_reach"],
                "dob": pd.to_datetime(a[f"{side}_dob"], errors="coerce")})
            rows.append(part)
        long = pd.concat(rows).dropna(subset=["date"]).sort_values("date")
        # Frame norms per division from the whole archive: a height is a
        # physical constant, not a result, so using every fighter's height to
        # define "typical" leaks nothing about outcomes.
        self.norms = long.groupby("limit")[["height", "reach"]].agg(["median", "std"])
        for fighter, g in long.groupby("fighter"):
            self.bouts[fighter] = g

    def features(self, fighter, date, limit, women, measured=None):
        """Pre-fight features for one fighter at one fight."""
        date = pd.Timestamp(date)
        g = self.bouts.get(norm_name(fighter))
        before = g[g["date"] < date] if g is not None else None
        height = reach = age = np.nan
        came_down = 0.0
        if g is not None and len(g):
            height = g["height"].dropna().iloc[-1] if g["height"].notna().any() else np.nan
            reach = g["reach"].dropna().iloc[-1] if g["reach"].notna().any() else np.nan
            dob = g["dob"].dropna()
            if len(dob):
                age = (date - dob.iloc[0]).days / 365.25
            if before is not None and len(before) and before["limit"].notna().any():
                came_down = max(0.0, float(before["limit"].max()) - float(limit))
        def z(value, column):
            if limit not in self.norms.index or not np.isfinite(value):
                return 0.0
            med, sd = self.norms.loc[limit, (column, "median")], \
                self.norms.loc[limit, (column, "std")]
            return float((value - med) / sd) if sd and np.isfinite(sd) else 0.0
        prior = np.nan
        if measured is not None:
            mine = measured[(measured["key"] == norm_name(fighter))
                            & (measured["date"] < date)]
            if len(mine):
                prior = float(mine["gain_pct"].iloc[-1])
        return {"limit": float(limit), "women": float(bool(women)),
                "height_z": z(height, "height"), "reach_z": z(reach, "reach"),
                "age": float(age) if np.isfinite(age) else 30.0,
                "came_down": came_down,
                "has_prior": float(np.isfinite(prior)),
                "prior_regain": float(prior) if np.isfinite(prior) else 0.0}


def training_rows(archive, measured):
    rows = []
    for m in measured.itertuples():
        limit = LIMITS.get(_nearest_limit(m.official_lbs))
        women = False
        g = archive.bouts.get(m.key)
        if g is not None and len(g):
            near = g.iloc[(g["date"] - m.date).abs().argsort()[:1]]
            if near["limit"].notna().all():
                limit = float(near["limit"].iloc[0])
            women = bool(near["women"].iloc[0])
        feats = archive.features(m.fighter, m.date, limit, women, measured)
        rows.append({**feats, "gain_pct": m.gain_pct, "event": m.event,
                     "fighter": m.fighter, "in_archive": g is not None})
    return pd.DataFrame(rows)


def _nearest_limit(official):
    return min(LIMITS, key=lambda k: abs(LIMITS[k] - official))


def fit(rows):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=3.0).fit(rows[FEATURES], rows["gain_pct"])


def validate(rows):
    """Leave one event out: MAE of the estimator vs the division median."""
    errors, naive = [], []
    for event in rows["event"].unique():
        train, test = rows[rows["event"] != event], rows[rows["event"] == event]
        if len(train) < 20:
            continue
        model = fit(train)
        pred = model.predict(test[FEATURES])
        errors += list(np.abs(pred - test["gain_pct"]))
        med = train.groupby("limit")["gain_pct"].median()
        base = test["limit"].map(med).fillna(train["gain_pct"].median())
        naive += list(np.abs(base - test["gain_pct"]))
    return float(np.mean(errors)), float(np.mean(naive)), len(errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--validate", action="store_true")
    parser.parse_args(argv)
    measured = pd.read_csv(MEASURED)
    measured["date"] = pd.to_datetime(measured["date"], errors="coerce")
    measured["key"] = measured["fighter"].map(norm_name)
    archive = Archive(pd.read_csv(ARCHIVE, usecols=[
        "date", "division", "r_name", "b_name", "winner", "r_height",
        "b_height", "r_reach", "b_reach", "r_dob", "b_dob"], low_memory=False))
    rows = training_rows(archive, measured)
    print(f"  {len(rows)} measured regains, {rows.in_archive.sum()} of the "
          f"fighters in the UFC archive, {rows.event.nunique()} events")
    mae, naive, n = validate(rows)
    print(f"  leave-one-event-out error on {n} regains: estimator "
          f"{mae:.2f} pts, division median {naive:.2f} pts")
    model = fit(rows)
    for name, coef in zip(FEATURES, model.coef_):
        print(f"    {name:<14}{coef:+.3f}")
    print("  " + ("the estimator beats the naive guess" if mae < naive else
                  "the estimator does NOT beat the division median - "
                  "no True Fight Weight is estimated from this yet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
