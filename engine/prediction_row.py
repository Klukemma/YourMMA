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
    # THESE ENTRIES ARE STALE AND COST REAL ACCURACY. They were written when a
    # live prediction held one rebuilt snapshot of a fighter and could not
    # replay their bout history. career_stats.final_stats() was added later
    # and does exactly that - it carries each fighter's accumulated state
    # after their last bout - and predict_card merges it into the stats dict
    # before any of this is read. The values are present; _value throws them
    # away because their names are listed here.
    #
    # The cost is a training/live mismatch on THIRTEEN features the winner
    # model was trained with and never sees at prediction time: cd_bouts,
    # cd_ctrl_share, cd_head_share, cd_kd_per15 and cd_minutes, each as a
    # difference, a level and a _known flag.
    #
    # They are left in place here deliberately. Removing them changes what the
    # deployed winner model reads, and the winner model's probability is what
    # the ROI, the calibration and the failure model are all measured against;
    # that is a change to make with a walk-forward measurement in hand, not as
    # a side effect of a method-model commit. cd_opp_sub_per15 is off the list
    # because it feeds the method model alone, so unlisting it cannot move any
    # of those numbers.
    "cd_bouts": "STALE - final_stats supplies this; see the note above",
    "cd_minutes": "STALE - final_stats supplies this; see the note above",
    "cd_kd_per15": "STALE - final_stats supplies this; see the note above",
    "cd_ctrl_share": "STALE - final_stats supplies this; see the note above",
    "cd_head_share": "STALE - final_stats supplies this; see the note above",
    "cd_opp_ctrl_share": "STALE - final_stats supplies this; see the note above",
    "cd_wins": "STALE - final_stats supplies this; see the note above",
    "cd_losses": "STALE - final_stats supplies this; see the note above",
    "cd_win_rate": "STALE - final_stats supplies this; see the note above",
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


def build_prediction_frame(red, blue, suffixes, red_extra=None, blue_extra=None,
                           fight_extra=None):
    """A one-row frame carrying r_<suffix> and b_<suffix> for each suffix.

    `red_extra` and `blue_extra` carry values the caller has already computed -
    age, experience, win rate and the rest are worked out in the prediction
    function rather than stored on the fighter.

    `fight_extra` carries columns that belong to the FIGHT rather than to
    either corner, and so have no r_/b_ prefix to derive them from. The matchup
    advantages are the case that needed it: a striking advantage is one signed
    number about a pairing, not a property of one fighter.
    """
    row = {}
    for suffix in suffixes:
        row[f"r_{suffix}"] = _value(red, suffix, red_extra)
        row[f"b_{suffix}"] = _value(blue, suffix, blue_extra)
    if fight_extra:
        row.update(fight_extra)
    return pd.DataFrame([row])


def fight_level_columns(specs):
    """Input columns a spec declares it needs that carry no corner prefix.

    A Paired spec's inputs are r_/b_ columns and build_prediction_frame derives
    them from the two stats dicts. A Derived spec that reads a whole-fight
    column - a matchup advantage is one signed number about a PAIRING, not a
    property of either fighter - has no such route, so it declares the column
    in `needs` and the caller supplies it through `fight_extra`.

    Reading this off the spec list rather than a hand-written constant is what
    stops a newly declared fight-level feature from reaching the prediction
    path as a KeyError, which is how mmr_diff and mu_sum broke CI.
    """
    out = set()
    for spec in specs:
        out.update(getattr(spec, "needs", ()) or ())
    return sorted(out)


def required_suffixes(specs):
    """Every attribute the specs read, without its corner prefix."""
    out = set()
    for spec in specs:
        for attr in ("red", "blue", "red_attack", "blue_defence"):
            col = getattr(spec, attr, None)
            if isinstance(col, str) and col[:2] in ("r_", "b_"):
                out.add(col[2:])
    return sorted(out)
