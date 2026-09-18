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
    Paired("ko_rate", "r_ko_rate", "b_ko_rate",
           why="share of wins by knockout - a style marker more than a "
               "predictor of who wins"),
    Paired("sub_rate", "r_sub_rate", "b_sub_rate",
           why="share of wins by submission, same caveat as ko_rate"),
]

# --- durability ------------------------------------------------------------

DURABILITY = [
    Paired("ko_vulnerability", "r_has_been_kod", "b_has_been_kod",
           why="has been knocked out before"),
    Paired("been_finished", "r_been_finished", "b_been_finished",
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


def all_specs():
    """Every spec, in the order the features are emitted."""
    return (STRIKING + GRAPPLING + PHYSICAL + RECORD + DURABILITY
            + FORM + TRAJECTORY_DIFFS + RATING + STANCE + INTERACTIONS)
