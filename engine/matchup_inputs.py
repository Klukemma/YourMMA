"""Turn career_stats output into the Physique and Form that matchup.py takes.

matchup.py has been dead code since it was written: it states a precise input
contract - per-minute rates, accuracies as fractions, NaN for no exposure, and
SHRUNK, because rate_ratio() is meaningless on an unshrunk one-bout rate - and
nothing in the repository produced one. This module is the missing half, and
it is deliberately the only place the two conventions are joined, so a unit
drift has one place to happen and one place to be caught.

WHAT career_stats DOES NOT DO, AND WHY IT IS DONE HERE. career_stats states
that it holds no smoothing constant anywhere: it reports ratios of observed
counts and publishes the evidence behind each one (cd_minutes and the four
attempt denominators) so a caller can decide how much to believe it. That is
the right choice for a module whose job is to report what happened. matchup's
K constants are the decision about how much to believe it, they were measured
in matchup, and they live there. This module applies them, and so is the only
thing in the pipeline that both knows a fighter's exposure and is allowed to
pull their rate toward the league.

EVERY RATE IS RECONSTRUCTED AS A NUMERATOR OVER cd_minutes, never as the
published rate re-divided. A share times its own exposure returns the count
that produced it exactly, which keeps the shrinkage denominator and the rate's
denominator the same number. Re-dividing would use cd_minutes against a
numerator that career_stats had divided by a slightly different exposure - the
pair denominator over bouts where that statistic was recorded - and the two
disagree wherever a bout is missing a count.
"""

import numpy as np
import pandas as pd

import matchup as mu

SECONDS_PER_MINUTE = 60.0
PER_15_MINUTES = 15.0

# (Form field, career column, league constant, prior weight, unit conversion).
# The conversion turns the published column into the numerator of a per-minute
# rate over cd_minutes; None means the column is already a total.
_RATE_FIELDS = (
    ("sig_att_per_min", "cd_sig_atmpted",
     mu.LEAGUE_SIG_ATT_PER_MIN, mu.K_STRIKE_VOLUME_MIN, None),
    ("sig_att_faced_per_min", "cd_opp_sig_atmpted",
     mu.LEAGUE_SIG_ATT_PER_MIN, mu.K_STRIKE_ABSORBED_MIN, None),
    ("td_att_per_min", "cd_td_atmpted",
     mu.LEAGUE_TD_ATT_PER_MIN, mu.K_TD_RATE_MIN, None),
    ("td_att_faced_per_min", "cd_opp_td_atmpted",
     mu.LEAGUE_TD_ATT_PER_MIN, mu.K_TD_CONCEDED_MIN, None),
    ("ctrl_sec_per_min", "cd_ctrl_share",
     mu.LEAGUE_CTRL_SEC_PER_MIN, mu.K_CTRL_MIN, SECONDS_PER_MINUTE),
    ("ctrl_sec_conceded_per_min", "cd_opp_ctrl_share",
     mu.LEAGUE_CTRL_SEC_PER_MIN, mu.K_CTRL_CONCEDED_MIN, SECONDS_PER_MINUTE),
    ("sub_att_per_min", "cd_sub_per15",
     mu.LEAGUE_SUB_ATT_PER_MIN, mu.K_SUB_ATT_MIN, 1.0 / PER_15_MINUTES),
    ("sub_att_conceded_per_min", "cd_opp_sub_per15",
     mu.LEAGUE_SUB_ATT_PER_MIN, mu.K_SUB_CONCEDED_MIN, 1.0 / PER_15_MINUTES),
)

# (Form field, career column, evidence column, league constant, prior weight,
#  invert). Accuracies shrink over ATTEMPTS, not minutes. invert=True turns a
# defence into the accuracy conceded, which is the one convention this module
# and matchup.Form share.
_ACCURACY_FIELDS = (
    ("sig_accuracy", "cd_str_acc", "cd_sig_atmpted",
     mu.LEAGUE_SIG_ACC, mu.K_SIG_ACC_ATT, False),
    ("sig_accuracy_conceded", "cd_str_def", "cd_opp_sig_atmpted",
     mu.LEAGUE_SIG_ACC, mu.K_SIG_DEF_ATT, True),
    ("td_accuracy", "cd_td_acc", "cd_td_atmpted",
     mu.LEAGUE_TD_ACC, mu.K_TD_ACC_ATT, False),
    ("td_accuracy_conceded", "cd_td_def", "cd_opp_td_atmpted",
     mu.LEAGUE_TD_ACC, mu.K_TD_DEF_ATT, True),
)

FORM_FIELDS = tuple(f[0] for f in _RATE_FIELDS) + tuple(
    f[0] for f in _ACCURACY_FIELDS)


def _shrink(numerator, denominator, league_rate, prior_weight):
    """shrunk_rate over whole columns, with its NaN rule preserved.

    A fighter with no exposure has no rate, NOT a league-average one, so the
    denominator being zero gives NaN rather than the league constant. The
    difference matters: NaN reaches the model as a _known flag of 0, while the
    league constant asserts a perfectly average fighter nobody observed.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    total = denominator + prior_weight
    out = np.full(denominator.shape, np.nan, dtype=float)
    usable = np.isfinite(numerator) & (denominator > 0) & (total > 0)
    out[usable] = ((numerator[usable] + league_rate * prior_weight)
                   / total[usable])
    return out


def form_frame(careers, corner):
    """The twelve shrunk Form fields for one corner, as a DataFrame."""
    def col(name):
        return pd.to_numeric(careers[f"{corner}_{name}"],
                             errors="coerce").to_numpy(dtype=float)

    minutes = col("cd_minutes")
    out = {}
    for field, source, league, weight, scale in _RATE_FIELDS:
        values = col(source)
        # A published rate times its own exposure is the count that produced
        # it; a published total is already that count.
        numerator = values if scale is None else values * scale * minutes
        out[field] = _shrink(numerator, minutes, league, weight)
    for field, source, evidence, league, weight, invert in _ACCURACY_FIELDS:
        attempts = col(evidence)
        fraction = col(source)
        if invert:
            fraction = 1.0 - fraction
        out[field] = _shrink(fraction * attempts, attempts, league, weight)
    return pd.DataFrame(out, index=careers.index)


def physique_frame(df, corner):
    """height/reach/weight in CENTIMETRES and KILOGRAMS, implausible dropped.

    The plausibility gates live in matchup, which measured them; this only
    routes the columns. Seventeen fights carry a height between 233 and 381 cm
    and those become NaN rather than being clipped to something believable.
    """
    def col(name):
        if f"{corner}_{name}" not in df.columns:
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[f"{corner}_{name}"],
                             errors="coerce").to_numpy(dtype=float)

    return pd.DataFrame({
        "height_cm": [mu.plausible_height_cm(v) for v in col("height")],
        "reach_cm": [mu.plausible_reach_cm(v) for v in col("reach")],
        "weight_kg": col("weight"),
    }, index=df.index)


def forms(careers, corner):
    """One matchup.Form per row. Use form_frame when a column will do."""
    frame = form_frame(careers, corner)
    return [mu.Form(**record) for record in frame.to_dict("records")]


def physiques(df, corner):
    """One matchup.Physique per row."""
    frame = physique_frame(df, corner)
    return [mu.Physique(**record) for record in frame.to_dict("records")]
