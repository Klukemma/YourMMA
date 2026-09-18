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


# --- the features the pipeline consumes ------------------------------------
# Every one of these is red-minus-blue already, so they are single signed
# columns rather than Paired specs. What makes them worth having is that they
# are not differences: striking_advantage crosses red's attempt rate and
# accuracy against BLUE's defence through log5, so it says how the fight goes
# rather than which fighter has the better profile. A wrestler's takedown rate
# means one thing against a sprawler and another against a debutant, and no
# subtraction of two career averages can express that.

ADVANTAGE_COLUMNS = {
    "mx_strike_adv": "expected significant strikes landed per minute, red "
                     "over blue, each crossed against the other's defence",
    "mx_td_adv": "expected takedowns landed per minute, both directions",
    "mx_ctrl_adv": "expected control seconds per minute, both directions",
    "mx_sub_adv": "expected submission attempts per minute, both directions",
    "mx_grappling_adv": "the three grappling advantages standardised and "
                        "weighted 0.66/0.19/0.15",
    "mx_size_adv": "height and reach, standardised and weighted equally "
                   "because the split is not identifiable",
    "mx_mass_adv": "listed weight difference, which fires on catchweights and "
                   "short-notice replacements and is otherwise 0",
}

KNOWN_COLUMNS = {
    "mx_striking_known": "1.0 when both corners have a striking profile",
    "mx_grappling_known": "1.0 when both corners have a grappling profile",
    "mx_size_known": "1.0 when both corners have a plausible height and reach",
}


def matchup_features(df, careers):
    """Every matchup advantage for every bout, as one DataFrame.

    NaN wherever an input is missing, never 0.0. An even matchup and an unknown
    one are different facts and the _known columns are what separates them; a
    zero-filled advantage would tell the model the fighters were level.
    """
    red_form, blue_form = forms(careers, "r"), forms(careers, "b")
    red_body, blue_body = physiques(df, "r"), physiques(df, "b")

    out = {name: np.full(len(df), np.nan) for name in ADVANTAGE_COLUMNS}
    known = {name: np.zeros(len(df)) for name in KNOWN_COLUMNS}
    for i in range(len(df)):
        rf, bf = red_form[i], blue_form[i]
        rb, bb = red_body[i], blue_body[i]
        out["mx_strike_adv"][i] = mu.striking_advantage(rf, bf)
        out["mx_td_adv"][i] = mu.takedown_advantage(rf, bf)
        out["mx_ctrl_adv"][i] = mu.control_advantage(rf, bf)
        out["mx_sub_adv"][i] = mu.submission_advantage(rf, bf)
        out["mx_grappling_adv"][i] = mu.grappling_advantage(rf, bf)
        out["mx_size_adv"][i] = mu.size_advantage(rb, bb)
        out["mx_mass_adv"][i] = mu.mass_advantage(rb, bb)
        known["mx_striking_known"][i] = mu.advantage_known(
            out["mx_strike_adv"][i])
        known["mx_grappling_known"][i] = mu.advantage_known(
            out["mx_grappling_adv"][i])
        known["mx_size_known"][i] = mu.size_advantage_known(rb, bb)
    out.update(known)
    return pd.DataFrame(out, index=df.index)


def _form_from_stats(stats):
    """One matchup.Form from a fighter snapshot carrying cd_ columns.

    The prediction-time counterpart of form_frame. A snapshot holds ONE
    fighter's final career state - career_stats.final_stats - so this goes
    through the same shrinkage with the same constants, and a missing column
    gives NaN rather than a league-average stand-in.
    """
    def get(name):
        value = stats.get(name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.nan
        return value

    minutes = get("cd_minutes")
    fields = {}
    for field, source, league, weight, scale in _RATE_FIELDS:
        value = get(source)
        numerator = value if scale is None else value * scale * minutes
        fields[field] = float(_shrink([numerator], [minutes], league, weight)[0])
    for field, source, evidence, league, weight, invert in _ACCURACY_FIELDS:
        attempts = get(evidence)
        fraction = get(source)
        if invert:
            fraction = 1.0 - fraction
        fields[field] = float(
            _shrink([fraction * attempts], [attempts], league, weight)[0])
    return mu.Form(**fields)


def _physique_from_stats(stats):
    """One matchup.Physique from a snapshot, in CENTIMETRES.

    The snapshot stores height and reach in INCHES, under keys named for the
    unit they are not in. That mismatch already cost this project a fake
    110-unit reach gap on 9.4% of fights, so the conversion is explicit and
    named here rather than assumed anywhere.
    """
    def cm(key):
        value = stats.get(key)
        try:
            return float(value) * CM_PER_INCH
        except (TypeError, ValueError):
            return np.nan

    return mu.Physique(height_cm=mu.plausible_height_cm(cm("height")),
                       reach_cm=mu.plausible_reach_cm(cm("reach")),
                       weight_kg=_as_float(stats.get("weight")))


CM_PER_INCH = 2.54


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def matchup_extra(red_stats, blue_stats):
    """The fight-level matchup columns for one pairing, as a plain dict.

    Same functions, same constants and same NaN rule as matchup_features uses
    over the training frame, so a fight predicted live is described the way the
    model was taught to read it.
    """
    rf, bf = _form_from_stats(red_stats), _form_from_stats(blue_stats)
    rb, bb = _physique_from_stats(red_stats), _physique_from_stats(blue_stats)
    out = {
        "mx_strike_adv": mu.striking_advantage(rf, bf),
        "mx_td_adv": mu.takedown_advantage(rf, bf),
        "mx_ctrl_adv": mu.control_advantage(rf, bf),
        "mx_sub_adv": mu.submission_advantage(rf, bf),
        "mx_grappling_adv": mu.grappling_advantage(rf, bf),
        "mx_size_adv": mu.size_advantage(rb, bb),
        "mx_mass_adv": mu.mass_advantage(rb, bb),
    }
    out["mx_striking_known"] = mu.advantage_known(out["mx_strike_adv"])
    out["mx_grappling_known"] = mu.advantage_known(out["mx_grappling_adv"])
    out["mx_size_known"] = mu.size_advantage_known(rb, bb)
    return out
