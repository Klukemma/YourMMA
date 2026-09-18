"""Turn two fighters' stats into the one-row frame the feature specs expect.

Training rows and a live prediction have to produce identical features, and
until now they did not: the training matrix was built from dataframe columns
while a prediction was assembled by hand into a dict. The two drifted, and a
run died with a KeyError ninety seconds in.

Both go through feature_spec.build_all now. This module is the adapter - it
maps the per-fighter stats dict onto the r_*/b_* column names the specs read.

Anything this cannot supply becomes NaN, which the specs turn into a neutral
difference and a _known flag of 0. That is the honest outcome: the old code
filled the same gaps with 0 and told the model a debutant lands 0% of strikes.
"""

import numpy as np
import pandas as pd

# spec column suffix -> key in the stats dict, where the two differ
ALIASES = {
    "td_avg_acc": "td_acc",
    "has_been_kod": "ko_losses",
}

# Suffixes a live prediction genuinely cannot reconstruct, with the reason.
# The stats dict is rebuilt from a fighter's most recent logged bout, so the
# rolling windows and anything derived across a sequence of fights are not
# available. Listing them here keeps the gap visible: previously they were
# silently filled with 0 and read as real values.
UNAVAILABLE = {
    "won_L3": "rolling window over the last three bouts, not carried in the stats dict",
    "splm_L3": "rolling window, as above",
    "str_acc_L3": "rolling window, as above",
    "td_avg_L3": "rolling window, as above",
    "data_reliability": "depends on a running fight count across the dataset",
    "damage_log": "cumulative career damage, accumulated across all bouts",
    "striking_trajectory": "needs the rolling window it is measured against",
    "accuracy_trajectory": "needs the rolling window it is measured against",
    # career_stats accumulates a fighter's whole prior record in date order
    # across both corners. A live prediction holds one rebuilt snapshot of the
    # fighter, not their bout history, so these cannot be reconstructed from
    # it. They become NaN, which the specs turn into a neutral difference and
    # a _known flag of 0 - the honest reading of "we did not compute this".
    "cd_bouts": "needs the fighter's full prior bout list, not a snapshot",
    "cd_minutes": "needs the fighter's full prior bout list",
    "cd_kd_per15": "needs the fighter's full prior bout list",
    "cd_ctrl_share": "needs the fighter's full prior bout list",
    "cd_head_share": "needs the fighter's full prior bout list",
}


def _value(stats, suffix, extra):
    """One attribute for one corner, or NaN when it is not available."""
    if extra and suffix in extra:
        return extra[suffix]
    if suffix in UNAVAILABLE:
        return np.nan
    key = ALIASES.get(suffix, suffix)
    value = stats.get(key)
    return np.nan if value is None else value


def build_prediction_frame(red, blue, suffixes, red_extra=None, blue_extra=None):
    """A one-row frame carrying r_<suffix> and b_<suffix> for each suffix.

    `red_extra` and `blue_extra` carry values the caller has already computed -
    age, experience, win rate and the rest are worked out in the prediction
    function rather than stored on the fighter.
    """
    row = {}
    for suffix in suffixes:
        row[f"r_{suffix}"] = _value(red, suffix, red_extra)
        row[f"b_{suffix}"] = _value(blue, suffix, blue_extra)
    return pd.DataFrame([row])


def required_suffixes(specs):
    """Every attribute the specs read, without its corner prefix."""
    out = set()
    for spec in specs:
        for attr in ("red", "blue", "red_attack", "blue_defence"):
            col = getattr(spec, attr, None)
            if isinstance(col, str) and col[:2] in ("r_", "b_"):
                out.add(col[2:])
    return sorted(out)
