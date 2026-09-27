"""Every model feature, declared in one place.

Read with feature_spec.py, which holds the primitives and explains why an
operand is never filled.

Grouped by what the feature is trying to say about a fight. `why` on each one
is not decoration: three features here computed a constant for their entire
life and nobody noticed, because nothing recorded what they were for.

Naming: a Paired spec called "acc" emits acc_diff, and acc_level / acc_known
when those are switched on. Level answers "how good is this fight at X", which
a difference cannot - two elite strikers and two novices share a difference of
zero. Known separates "no data" from "evenly matched", which otherwise look
identical once the matrix fills NaN with 0.
"""

import numpy as np
import pandas as pd

from feature_spec import Derived, Interaction, Paired, paired_diff, to_number


# --- striking --------------------------------------------------------------

STRIKING = [
    Paired("off_striking", "r_splm", "b_splm", level=True, known=True,
           why="significant strikes landed per minute - volume"),
    Paired("acc", "r_str_acc", "b_str_acc", level=True, known=True,
           why="striking accuracy; a percentage, so filling it with 0 asserted "
               "the fighter lands nothing"),
    Paired("def", "r_str_def", "b_str_def", level=True, known=True,
           why="strikes avoided, the defensive half of an exchange"),
    Paired("sapm", "r_sapm", "b_sapm", level=True, known=True,
           why="strikes absorbed per minute - how hittable a fighter is"),
    Paired("footwork", "r_footwork_proxy", "b_footwork_proxy", level=True,
           why="landed over absorbed; the single strongest feature in the "
               "model at AUC 0.668"),
]

# --- grappling -------------------------------------------------------------

GRAPPLING = [
    Paired("td_off", "r_td_avg", "b_td_avg", level=True, known=True,
           why="takedowns landed per fifteen minutes"),
    Paired("td_def", "r_td_def", "b_td_def", level=True, known=True,
           why="takedowns stuffed, AUC 0.611 on its own"),
    Paired("td_acc", "r_td_avg_acc", "b_td_avg_acc", known=True,
           why="takedown accuracy - attempts that land"),
    Paired("sub", "r_sub_avg", "b_sub_avg", known=True,
           why="submission attempts per fifteen minutes"),
    Paired("sub_def", "r_sub_def_score", "b_sub_def_score", known=True,
           level=True,
           why="submission defence; weak alone but the counterpart to sub"),
    Paired("cage_control_cap", "r_ctrl_rate_ewm", "b_ctrl_rate_ewm", level=True,
           why="share of fight time in controlling position"),
    Paired("clinch_activity", "r_clinch_activity_ewm", "b_clinch_activity_ewm",
           why="how much of the work happens in the clinch"),
    Paired("grind_tendency", "r_grind_rate", "b_grind_rate",
           why="tendency to grind rather than strike at range"),
]

# --- physical --------------------------------------------------------------

PHYSICAL = [
    Paired("height", "r_height", "b_height", level=True, known=True,
           why="height in centimetres, read from the column that holds "
               "centimetres rather than one mislabelled inches"),
    Paired("reach", "r_reach", "b_reach", level=True, known=True,
           why="reach in centimetres, AUC 0.618 even while filled with 70 "
               "inches on 9.4% of fights"),
    Paired("ape_index", "r_ape_index", "b_ape_index", known=True,
           why="reach relative to height - long-limbed for their size"),
    Paired("age", "r_age", "b_age", level=True, known=True,
           why="age, AUC 0.395 - inverted, so the younger fighter wins"),
]

# --- record and experience -------------------------------------------------

RECORD = [
    Paired("winrate", "r_winrate", "b_winrate", level=True,
           why="career win rate, AUC 0.658"),
    Paired("exp", "r_exp", "b_exp", level=True,
           why="professional bouts, including those outside the UFC"),
    Paired("data_sparsity", "r_data_reliability", "b_data_reliability",
           why="how much history each stat is averaged over"),
    # Point-in-time columns from career_stats. These carry information the
    # leaking profile never could: it reported one career-long number per
    # fighter, so it could not say how much of that career had happened yet.
    Paired("cd_bouts", "r_cd_bouts", "b_cd_bouts", level=True, known=True,
           why="prior UFC bouts at the time of this fight, not career total"),
    Paired("cd_minutes", "r_cd_minutes", "b_cd_minutes", level=True,
           why="prior cage time in minutes; twenty rounds of experience is not "
               "the same as twenty first-round knockouts"),
    Paired("cd_kd_per15", "r_cd_kd_per15", "b_cd_kd_per15", level=True, known=True,
           why="knockdowns per fifteen minutes - power, which the profile "
               "columns never reported at all"),
    Paired("cd_ctrl_share", "r_cd_ctrl_share", "b_cd_ctrl_share", level=True,
           known=True,
           why="share of prior fight time spent in control"),
    Paired("cd_head_share", "r_cd_head_share", "b_cd_head_share",
           why="share of landed strikes aimed at the head - a head-hunter and "
               "a leg-kicker with equal volume are different fights"),
    Paired("ko_rate", "r_ko_rate", "b_ko_rate", level=True,
           why="share of wins by knockout - a style marker more than a "
               "predictor of who wins. The LEVEL is for the method model: two "
               "knockout artists and two point-fighters have the same "
               "difference and are not the same fight"),
    Paired("sub_rate", "r_sub_rate", "b_sub_rate", level=True,
           why="share of wins by submission, same caveat and same level as "
               "ko_rate"),
]

# --- durability ------------------------------------------------------------

DURABILITY = [
    Paired("ko_vulnerability", "r_has_been_kod", "b_has_been_kod",
           level=True,
           why="has been knocked out before - the level says whether either "
               "chin has ever gone, which is what decides a finish"),
    Paired("been_finished", "r_been_finished", "b_been_finished", level=True,
           why="times finished, however it happened"),
    Paired("absorption_eff", "r_absorption_eff", "b_absorption_eff", level=True,
           why="damage taken relative to output"),
    Paired("career_damage", "r_damage_log", "b_damage_log", level=True,
           why="cumulative career punishment, log-scaled"),
    Paired("cardio", "r_cardio", "b_cardio", level=True,
           why="output held into the later rounds"),
]

# --- form and trajectory ---------------------------------------------------
# The three trajectory features were one dead column: exactly 0 on 100% of
# rows, and all three identical to each other. "Is this fighter improving?" is
# a sound idea, so it is computed here instead of asserted - recent form
# against career baseline, which is what the phrase means.

def _trajectory(recent_col, career_col):
    def build(df):
        return to_number(df[recent_col]) - to_number(df[career_col])
    return build


FORM = [
    Paired("recent_form", "r_won_L3", "b_won_L3", level=True,
           why="wins in the last three bouts"),
    Paired("splm_L3", "r_splm_L3", "b_splm_L3", known=True,
           why="recent striking volume rather than the career average"),
    Paired("str_acc_L3", "r_str_acc_L3", "b_str_acc_L3", known=True,
           why="recent striking accuracy"),
    Paired("td_avg_L3", "r_td_avg_L3", "b_td_avg_L3", known=True,
           why="recent takedown volume"),
    Derived("r_striking_trajectory", _trajectory("r_splm_L3", "r_splm"),
            why="recent volume minus career volume: rising or fading"),
    Derived("b_striking_trajectory", _trajectory("b_splm_L3", "b_splm"),
            why="the same for the blue corner"),
    Derived("r_accuracy_trajectory", _trajectory("r_str_acc_L3", "r_str_acc"),
            why="recent accuracy minus career accuracy"),
    Derived("b_accuracy_trajectory", _trajectory("b_str_acc_L3", "b_str_acc"),
            why="the same for the blue corner"),
    Paired("ring_rust", "r_ring_rust", "b_ring_rust",
           why="time since the last bout"),
    Paired("mom_quality", "r_mom_quality", "b_mom_quality",
           why="momentum weighted by who the wins came against"),
    Paired("prime_wc", "r_prime_wc", "b_prime_wc",
           why="how close to the prime age for this weight class"),
    Paired("wc_move", "r_wc_move", "b_wc_move",
           why="recent move up or down in weight"),
]

# The trajectory differences, built from the four Derived columns above.
TRAJECTORY_DIFFS = [
    Derived("striking_trajectory_diff",
            lambda df: paired_diff(df["r_striking_trajectory"],
                                   df["b_striking_trajectory"]),
            why="who is trending up in volume"),
    Derived("accuracy_trajectory_diff",
            lambda df: paired_diff(df["r_accuracy_trajectory"],
                                   df["b_accuracy_trajectory"]),
            why="who is trending up in accuracy"),
]

# --- skill rating ----------------------------------------------------------

RATING = [
    Paired("mu", "r_mu", "b_mu", level=True,
           why="TrueSkill skill estimate; the level term says whether this is "
               "a good fight or a bad one"),
    Paired("skill_conservative", "r_skill_conservative", "b_skill_conservative",
           why="mu penalised by uncertainty, kept because pruning weak "
               "features has always made this model worse"),
    Paired("consistency", "r_consistency", "b_consistency",
           why="how settled each rating is"),
    Paired("opp_quality", "r_opp_quality", "b_opp_quality", level=True, known=True,
           why="strength of schedule on the mu scale; the known flag replaces "
               "a constant 1500 that fired on a quarter of all fights"),
]

# --- stance and style ------------------------------------------------------
# stance_interaction correlated 1.0000 with southpaw_diff, so it was a copy
# rather than an interaction. The genuine question is whether the stances
# differ, which is one feature, not four.

STANCE = [
    Paired("southpaw", "r_southpaw", "b_southpaw",
           why="which fighters are southpaw"),
    Derived("stance_mismatch",
            lambda df: (to_number(df["r_southpaw"])
                        != to_number(df["b_southpaw"])).astype(float),
            why="open-stance matchups create different angles; the only "
                "stance term that is not a duplicate of this one"),
]

# --- matchup interactions --------------------------------------------------
# The gap a difference-only design cannot close. Takedown defence matters
# enormously against a wrestler and not at all against a kickboxer, and no
# subtraction can say that.

INTERACTIONS = [
    Interaction("r_td_vs_b_tdd", "r_td_avg", "b_td_def",
                why="can red get it down against this opponent"),
    Interaction("b_td_vs_r_tdd", "b_td_avg", "r_td_def",
                why="the mirror, since the corners are not symmetric"),
    Interaction("r_strike_vs_b_def", "r_splm", "b_str_def",
                why="can red land against this opponent's defence"),
    Interaction("b_strike_vs_r_def", "b_splm", "r_str_def",
                why="the mirror: can blue land against red's defence"),
    Interaction("r_sub_vs_b_subdef", "r_sub_avg", "b_sub_def_score",
                why="submission threat against submission defence"),
    Interaction("b_sub_vs_r_subdef", "b_sub_avg", "r_sub_def_score",
                why="the mirror: blue's submission threat against red"),
    Interaction("r_power_vs_b_chin", "r_splm", "b_has_been_kod",
                why="volume against a fighter who has been stopped before"),
    Interaction("b_power_vs_r_chin", "b_splm", "r_has_been_kod",
                why="the mirror: blue's volume against red's chin"),
]


# --- matchup advantages ----------------------------------------------------
# Prefixed mx_, not mu_: mu_diff and mu_level already exist and are the
# TrueSkill rating mean. Two unrelated families sharing a prefix is how a
# reader ends up believing a size advantage is a rating term.
#
# Built by matchup_inputs.matchup_features from point-in-time career stats and
# assigned upstream, so these specs only name and justify them. They are
# already signed red-minus-blue, which is why they are Derived rather than
# Paired.
#
# They are NOT more differences. striking_advantage crosses red's attempt rate
# and accuracy against BLUE's defence through log5, so it answers how this
# fight goes rather than which fighter has the better profile - the same gap
# INTERACTIONS below exists to close, but computed from a measured league
# baseline instead of a raw product of two columns.
#
# Measured standalone on held-out 2019-2026 fights: strike 0.582, ctrl 0.563,
# grappling composite 0.563, sub 0.548, td 0.543, size 0.527, mass 0.516. The
# striking advantage holds 0.56 to 0.62 in EVERY year from 2015 to 2026,
# 2026 included, which is the shape of a real effect rather than a leak.

def _column(name):
    return lambda df: to_number(df[name])


MATCHUP = [
    Derived("mx_strike_adv", _column("mx_strike_adv"),
            needs=("mx_strike_adv",),
            why="expected strikes landed per minute, each fighter's offence "
                "crossed against the other's defence; AUC 0.582 held out"),
    Derived("mx_td_adv", _column("mx_td_adv"),
            needs=("mx_td_adv",),
            why="expected takedowns per minute, both directions"),
    Derived("mx_ctrl_adv", _column("mx_ctrl_adv"),
            needs=("mx_ctrl_adv",),
            why="expected control seconds per minute, both directions"),
    Derived("mx_sub_adv", _column("mx_sub_adv"),
            needs=("mx_sub_adv",),
            why="expected submission attempts per minute, both directions"),
    Derived("mx_grappling_adv", _column("mx_grappling_adv"),
            needs=("mx_grappling_adv",),
            why="the three grappling advantages standardised and weighted on "
                "pre-2019 outcomes only"),
    Derived("mx_size_adv", _column("mx_size_adv"),
            needs=("mx_size_adv",),
            why="height and reach weighted equally, because three windows put "
                "the split at +0.21, -1.36 and 1.0 and it cannot be resolved"),
    Derived("mx_mass_adv", _column("mx_mass_adv"),
            needs=("mx_mass_adv",),
            why="listed weight gap; fires on catchweights and short-notice "
                "replacements and is 0 on two thirds of fights"),
    Derived("mx_striking_known", _column("mx_striking_known"),
            needs=("mx_striking_known",),
            why="separates an even striking matchup from an unmeasured one"),
    Derived("mx_grappling_known", _column("mx_grappling_known"),
            needs=("mx_grappling_known",),
            why="the same for grappling, which is missing more often"),
    Derived("mx_size_known", _column("mx_size_known"),
            needs=("mx_size_known",),
            why="blue-corner reach is absent on 7.7% of fights"),
]

# --- how a fight ends ------------------------------------------------------
#
# These are for the METHOD, FINISH and ROUND models, and predict_card keeps
# them out of the winner model the way it keeps cage control out.
#
# They exist because the method model had no features of its own. It was
# trained on the matrix built and audited for "who wins", where almost every
# column is a difference between the corners. For who wins that is the right
# shape. For how it ends it is close to the wrong one:
#
#     ko_rate_diff   0.0   two men who knock everyone out
#     ko_rate_diff   0.0   two men who have never knocked anyone out
#
# Same number, opposite fights - and on 2026-09-26 the model called Hiestand
# vs Nakamura a decision at 55%, a fight in which the two between them attempt
# 2.4 submissions per fifteen minutes and neither has ever been finished. It
# ended by submission in the second.
#
# Every one of these carries more signal as a level than as a difference,
# measured over 7,219 fights from 2011 (AUC against "was this fight
# finished"): KO wins 0.575 against 0.553, knockdowns 0.568 against 0.552,
# being knocked out 0.572 against 0.561. Summed across both corners they sort
# the finish rate from 40.7% in the bottom fifth to 61.6% in the top.
#
# These are rates per fifteen minutes, which is what ko_rate is not: ko_rate
# is the share of WINS that were knockouts, undefined for a fighter with no
# wins and silent about how fast anything happens.
#
# cd_kd_per15 belongs here by rights and is declared under RECORD already,
# with its level, so it is not repeated.
METHOD_RATES = [
    Paired("cd_ko_for_per15", "r_cd_ko_for_per15", "b_cd_ko_for_per15",
           level=True, known=True,
           why="knockouts landed per fifteen minutes - how often this fighter "
               "ends it, as a rate rather than a share of wins"),
    Paired("cd_ko_against_per15", "r_cd_ko_against_per15",
           "b_cd_ko_against_per15", level=True,
           why="knockouts suffered per fifteen minutes - a chin, measured"),
    Paired("cd_sub_for_per15", "r_cd_sub_for_per15", "b_cd_sub_for_per15",
           level=True, known=True,
           why="submissions landed per fifteen minutes"),
    Paired("cd_sub_against_per15", "r_cd_sub_against_per15",
           "b_cd_sub_against_per15", level=True,
           why="submissions suffered per fifteen minutes"),
    Paired("cd_sub_per15", "r_cd_sub_per15", "b_cd_sub_per15", level=True,
           why="submission ATTEMPTS per fifteen minutes - the grappler who "
               "keeps hunting is a different fight from the one who does not, "
               "whether or not the attempts land"),
    Paired("cd_opp_sub_per15", "r_cd_opp_sub_per15", "b_cd_opp_sub_per15",
           level=True,
           why="submission attempts faced per fifteen minutes"),
    Paired("cd_opp_kd_per15", "r_cd_opp_kd_per15", "b_cd_opp_kd_per15",
           level=True,
           why="knockdowns conceded per fifteen minutes"),
]


def all_specs():
    """Every spec, in the order the features are emitted."""
    return (STRIKING + GRAPPLING + PHYSICAL + RECORD + DURABILITY
            + FORM + TRAJECTORY_DIFFS + RATING + STANCE + INTERACTIONS
            + MATCHUP + METHOD_RATES)


# The five columns above whose LEVEL was added for the method model. Their
# differences stay in the winner model where they have always been; only the
# levels are new, and only the method, finish and round models see them.
FINISH_LEVELS = ("ko_rate_level", "sub_rate_level", "ko_vulnerability_level",
                 "been_finished_level", "sub_def_level")


def finish_level_names():
    """The finish-family levels, checked against what the specs really emit."""
    emitted = {n for spec in all_specs() for n in spec.emits}
    missing = [n for n in FINISH_LEVELS if n not in emitted]
    if missing:
        raise KeyError(f"declared as method-only but never emitted: {missing}")
    return list(FINISH_LEVELS)


def method_rate_names():
    """What METHOD_RATES emits. predict_card keeps these out of the winner
    model, and experiments/method_model.py measures exactly this set, so the
    thing that was measured and the thing that ships cannot drift apart."""
    return [n for spec in METHOD_RATES for n in spec.emits]
