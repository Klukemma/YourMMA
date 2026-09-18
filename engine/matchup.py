"""Matchup variables: how two specific fighters' attributes interact.

WHY THIS MODULE EXISTS

engine/leakage.py found that r_splm, r_str_acc, r_sapm, r_str_def, r_td_avg,
r_td_def, r_td_avg_acc and r_sub_avg (and their b_ twins) are career averages
as of the data pull, joined onto every bout the fighter ever had. A 2014 fight
is described by a striking accuracy computed over 2014-2026, including the
fight being predicted. That inflated the walk-forward backtest to 68.6% and
+16.2% ROI over 5,943 priced bets; the only honest year was 2026 (61.5%,
-3.2%), because the data ends in August 2026 and there is no future left.

NOT ONE OF THOSE EIGHT COLUMNS IS REFERENCED HERE. Every quantity this module
consumes is derivable from the honest per-fight count columns - sig_str_landed,
sig_str_atmpted, td_landed, td_atmpted, ctrl, sub_att, kd - which record what
happened in that bout and nothing else. Because each row carries BOTH corners'
counts, the defensive side needs no join: strikes absorbed are the opponent's
landed count, takedowns stuffed are the opponent's missed attempts.

The second reason this module exists is that a difference of two main effects
is not a matchup. "Red lands 4.1 a minute, blue lands 3.2" says nothing about
whether blue is hard to hit. A high-volume striker against a porous defender
and the same striker against an elusive one are different fights, and only an
interaction can say so.

WHAT IT DOES NOT DO

No I/O, no DataFrame-wide magic, no pipeline. Every public function is pure and
takes two fighters' point-in-time stats as explicit frozen dataclasses, so all
of it is testable without importing predict_card.py (which takes ten minutes
because importing it runs the whole pipeline).

THE THREE MODELLING IDEAS, each with a published precedent

(1) MULTIPLICATIVE RATE INTERACTION for volumes. A rate in a specific matchup
    is the league baseline times the attacker's ratio-to-league times the
    defender's conceded-ratio-to-league. This is the log-linear attack/defence
    form of Dixon & Coles (1997), "Modelling Association Football Scores and
    Inefficiencies in the Football Betting Market", JRSS-C 46(2).

(2) LOG5 ODDS COMBINATION for accuracies. Bill James, Baseball Abstract 1981.
    Same structure as (1) but in odds space, which is the correct support for a
    bounded quantity.

(3) PIECEWISE-EXPONENTIAL COMPETING RISKS for pace and duration. Four
    directional hazards per minute (KO and submission, each way) sum to a total
    hazard; survival across the scheduled rounds gives P(decision) and expected
    duration in closed form, and the same four hazards drive the Monte Carlo
    sampler.

SIGN CONVENTION, uniform and repeated in every docstring: A POSITIVE ADVANTAGE
FAVOURS THE RED CORNER, the corner whose columns carry the r_ prefix. Swapping
the corners negates every advantage exactly; there is a test for it.

MISSING INPUTS: every function returns NaN if any required input is NaN, and
advantage_known() returns 1.0/0.0 so a caller following feature_spec's
paired_known pattern can tell "genuinely even" (0.0, known 1.0) from "unknown"
(NaN, known 0.0). Nothing here is ever filled with a number that means
something else; a missing accuracy filled with 0 asserts the fighter lands 0%
of his strikes, which is the defect feature_spec was rewritten to remove.

================================================================================
MEASURED ON engine/data/UFC_with_mmr_rebuilt_dedup.csv
8,587 bouts, 2000-11-17 to 2026-08-08, 17,174 fighter-appearances, 183,687.2
fighter-minutes. Every number below was computed from that file before any
formula in this module was fixed; none is quoted from elsewhere.
================================================================================

match_time_sec IS NOT TOTAL FIGHT TIME, AND READING IT AS SUCH HALVES EVERY
FIGHT. On 8,486 rows it is seconds INTO the finishing round (median 296,
capped at exactly 300 on 4,088 rows). On 101 rows dated 2025-09-13 to
2025-12-06 it is CUMULATIVE elapsed time, up to 1500. The rule "cumulative iff
> 300" is EXACT, not a heuristic: a cumulative time of 300 or less can only
come from a round-1 finish, and in round 1 the two conventions give the same
number. Of the 145 rows in that scrape window, 101 exceed 300 and the other 44
are round-1 finishes where the conventions coincide. Median fight length is
14.18 minutes repaired against 4.93 unrepaired.

FINISH HAZARD IS NOT CONSTANT ACROSS ROUNDS, and assuming it is costs ten
points of decision rate. 2011+, 3-round bouts (n=6,539): per-round finish
hazard 0.2578, 0.2129, 0.1319. A constant-hazard model calibrated on round 1
predicts 40.9% decisions against an actual 50.7%; for 5-round bouts it predicts
29.3% against an actual 40.4%.

WITH THE EMPIRICAL SHAPE VECTORS THE MODEL IS EXACT ON WHAT IT WAS FIT TO AND
CLOSE ON WHAT IT WAS NOT. Decision share 0.5071 modelled vs 0.5071 actual
(3 rounds) and 0.4044 vs 0.4044 (5 rounds) - exact, since the shapes are fit to
those hazards. Mean duration is an INDEPENDENT check it was not fit to: 10.35
vs 10.51 minutes (3 rounds, -1.6%) and 15.30 vs 15.46 (5 rounds, -1.1%). The
under-prediction is the within-round hazard not being flat - finishes cluster
late in a round, and the model spreads them evenly.

KO HAZARD DECAYS FASTER THAN SUBMISSION HAZARD, so the two shapes are kept
separate. 3-round, relative to round 1: KO 1.000 / 0.746 / 0.436, submission
1.000 / 0.908 / 0.547. The difference is 3.1 standard errors in round 2 and 2.5
in round 3.

ROUND-1 KO HAZARD IS IDENTICAL IN 3- AND 5-ROUND BOUTS: 0.038803 against
0.038808 per minute, agreeing to five decimal places. The entire difference in
decision rate between the two schedules is the submission hazard (0.020834
against 0.010282) plus the longer schedule. Five-rounders are not harder to
knock out; they are half as easy to submit.

THE PACE MODEL SEPARATES FIGHTS OUT OF SAMPLE, USING PRIOR-FIGHT DATA ONLY.
5,085 fights where both fighters had 15+ prior minutes, sorted into quintiles
by modelled decision probability:

    quintile   n     observed decisions   modelled   observed minutes
    1        1017        31.8%             30.9%          9.80
    2        1017        44.0%             45.3%         10.82
    3        1017        52.9%             52.6%         11.52
    4        1017        59.9%             58.8%         12.19
    5        1017        66.7%             67.7%         13.00

AUC for predicting a finish is 0.6438, and the modelled rate tracks the
observed one to within 1.3 points in every quintile, so the model is separating
AND calibrated rather than merely ordering.

THE MATCHUP ADVANTAGES PREDICT THE WINNER HONESTLY. Computed strictly from
prior bouts on those 5,085 fights, fit on 2000-2018 (n=2,521) and tested on
2019-2026 (n=2,564), the four advantages alone give out-of-sample AUC 0.6078.
Per-standard-deviation log-odds in the full-sample joint fit: striking +0.332,
submission threat +0.130, control +0.124, takedown +0.092.

TAKEDOWN DIFFERENTIAL ADDS NOTHING OVER CONTROL-TIME DIFFERENTIAL. Alone it is
worth +0.170 log-odds per SD; in the grappling trio with control time it falls
to -0.009. The two correlate at r = 0.739. Takedowns are how you get control;
control is the thing that matters. This is reported rather than hidden, and the
takedown weight in the composite is a declared-arbitrary floor.

TAKEDOWN ACCURACY IS ESSENTIALLY NOISE: split-half r = +0.071 over a mean 18.6
attempts per half. Takedown DEFENCE is three times better, r = +0.237. That is
exactly the column (r_td_avg_acc) that looked informative in the leaking
feature set, because a career total that includes the future is not noisy.

APE INDEX ADDS NOTHING ONCE REACH IS CONTROLLED. Alone, ape_diff is worth
+0.0092 log-odds per cm; entered alongside reach_diff its coefficient goes to
-0.0042. It is exposed as its own signed component because it is a distinct
question and cheap to answer, but SIZE_W_APE is 0.0.

THE HEIGHT/REACH SPLIT IS NOT IDENTIFIABLE. Bootstrapping 400 resamples of a
joint logit, height's share of the combined per-SD size effect has mean 0.207
with a 90% interval of [-0.223, +0.598], and the raw height coefficient flips
across eras (-0.00034 for 2000-2015, +0.00737 for 2016-2026). SIZE_W_HEIGHT =
0.20 is a CHOICE INSIDE A BAND THE DATA CANNOT RESOLVE, not a measurement.

WEIGHT IS A DIVISION LABEL, NOT FIGHT-NIGHT MASS. r_weight is constant across
97.5% of careers with five or more bouts - the same signature as the leaking
profile columns - and weight_diff is exactly 0 on 66.0% of fights. It is
therefore NOT folded into size_advantage; mass_advantage is a separate function
that fires on catchweights, short-notice replacements and division jumps.

SEVENTEEN FIGHTS CARRY AN IMPOSSIBLE HEIGHT, 233.68 to 381.00 cm, all dated
2025-09-13 to 2025-11-22 and all exact inch multiples (92, 100, 102, 112, 122,
140, 150 inches) - the same scrape window that produced the cumulative-time
block. The tallest legitimate fighter in the data is Stefan Struve at 210.82
cm, so the plausibility gate sits at 215 and returns NaN rather than clipping.

AVAILABILITY. This module decides no exposure floor; career_stats.py returns
NaN below whatever floor it sets and NaN propagates through here. The curve, as
a share of the 8,587 fights where BOTH corners clear a prior-minutes threshold:
100% at 0, 68.4% at 5, 59.9% at 15, 40.4% at 30, 29.6% at 45, 15.4% at 75.
Blue-corner reach is missing on 7.74% of fights and red-corner reach on 2.49%,
so size_advantage is available on 90.6%.

THE RED CORNER WINS 61.9% OF FIGHTS IN THIS DATASET. Corner assignment is not
random - red is the higher-billed fighter. Every per-SD coefficient quoted
above comes from a fit with an intercept that absorbs this, so the coefficients
are clean, but reading a positive advantage as a win probability without an
intercept is wrong by about twelve points.

KNOWN LIMITS

- The multiplicative form assumes offence and defence combine independently.
  A wrestler nobody bothers to strike at on the ground shows an excellent
  conceded-accuracy figure without being hard to hit, and this module then
  credits his next opponent with a phantom discount. The per-position columns
  (dist_/clinch_/ground_) that could diagnose it are not used here.
- The round-shape vectors are measured on SURVIVORS, so they mix genuine
  fatigue with selection: whoever reaches round 3 is whoever could not be
  finished earlier. Applying a survivor-derived shape to a pair whose hazard
  was estimated separately double-counts that selection for a durable pair.
  The quintile table above says the effect is not severe in the middle; it says
  nothing about the tails.
- The 5-round submission shape rests on 8 to 31 events per round. Its apparent
  rise in rounds 2 and 3 is well within noise.
- SIZE_W_* and GRAP_W_* were fit on outcomes from this same dataset. The
  inputs are strictly prior-fight, so this is not the career-total leak, but it
  is an in-sample weighting choice and a backtest treating these as given will
  be mildly optimistic. The 2019-2026 hold-out (AUC 0.6078) bounds how mild.
- The eight rate baselines are pooled over 2000-2026 while the hazard constants
  are 2011+. The sport changed: the 3-round decision share went 28.8% (2000-05)
  -> 39.9% -> 49.9% -> 52.3% -> 49.9% (2021-26). Pooled and 2011+ league
  accuracies agree to 0.003 (striking) and 0.009 (takedowns), which is why the
  pooled figures are tolerable; there is a test pinning that agreement so a
  future re-pull that breaks it is noticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping, Optional, Tuple

import numpy as np


# --- time and schedule -----------------------------------------------------

# A UFC round under the Unified Rules, in force for every event in this file
# (earliest bout 2000-11-17, UFC 28). Confirmed on the data: the within-round
# reading of match_time_sec tops out at exactly 300.0 on 4,088 rows.
ROUND_SECONDS = 300.0
SECONDS_PER_MINUTE = 60.0
ROUND_MINUTES = ROUND_SECONDS / SECONDS_PER_MINUTE

# Only these two schedules have calibrated round-shape vectors. The file also
# holds 5 two-round bouts, which is too few to calibrate anything.
SUPPORTED_ROUNDS = (3, 5)
TWO_ROUND_BOUTS_IN_DATA = 5


# --- physique gates --------------------------------------------------------

# Shortest fighter in the data is 152.40 cm; the floor sits below that with
# room, because a real 150 cm fighter would be a data point, not a defect.
HEIGHT_MIN_CM = 140.0
# Tallest legitimate fighter is Stefan Struve at 210.82 cm. The 17 corrupt
# fights start at 233.68 cm, so 215 separates them with no ambiguity.
HEIGHT_MAX_CM = 215.0

# Observed reach range is 147.32 to 213.36 cm with NO corruption present. This
# gate is a tripwire for a future scrape defect, not a repair of a current one.
REACH_MIN_CM = 140.0
REACH_MAX_CM = 220.0

# Observed standard deviations of the signed corner differences, measured after
# the plausibility gates: height over 8,560 fights, reach over 7,777, ape over
# 7,758, weight over 8,584.
HEIGHT_DIFF_SD_CM = 6.38
REACH_DIFF_SD_CM = 8.26
APE_DIFF_SD_CM = 6.58
WEIGHT_DIFF_SD_KG = 4.99

# NOT IDENTIFIABLE, AND THE SPLIT IS THEREFORE NOT FITTED AT ALL. The previous
# 0.20/0.80 came from a joint logit over the WHOLE 2000-2026 sample, which is
# the calibration leak simulate.py removes by fitting only before 2019. Refit
# honestly, the answer does not merely move - it leaves the range:
#
#   window                        height's share of the combined per-SD effect
#   full sample 2000-2026                  +0.207   90% [-0.223, +0.598]
#   calibration only, pre-2019             -1.358   90% [-3.342, +0.276]
#   held-out 2019-2026, by ranking          ~1.0
#
# Three windows, three answers spanning the entire line. Height and reach
# correlate at r = 0.62 and the combined effect is worth AUC 0.52 to 0.54 out
# of sample, so the split is a rounding error on a near-null quantity dressed
# up as a measurement.
#
# 0.50 IS THE MAXIMUM-IGNORANCE CHOICE, NOT A FIT. It was NOT selected by the
# held-out AUC above - that column is reported to show the estimate is
# unstable, and choosing by it would turn the held-out period into a tuning
# set, which is exactly the mistake that produced a +5.6% edge that vanished.
# An equal split is what "we cannot resolve this" looks like written down.
SIZE_W_HEIGHT = 0.50
SIZE_W_REACH = 0.50

# MEASURED, NOT ASSUMED. Ape index alone is worth +0.0092 log-odds per cm;
# entered alongside reach difference its coefficient is -0.0042. Reach already
# contains whatever ape index knows. The term is kept in the formula with a
# zero coefficient so its absence is visible in the code rather than implied.
SIZE_W_APE = 0.0


# --- league baselines ------------------------------------------------------
# All eight pooled over 17,174 fighter-bouts / 183,687.2 fighter-minutes,
# 2000-2026. See the era caveat in the module docstring.

# 648,264 significant strikes landed over 183,687.2 fighter-minutes. Identically
# equal to the league absorbed rate, because every strike one fighter lands
# another absorbs.
LEAGUE_SIG_LANDED_PER_MIN = 3.5292

# 648,264 landed / 1,446,118 attempted. League striking defence is exactly
# 1 - 0.4483 = 0.5517.
LEAGUE_SIG_ACC = 0.4483

# Stated as a derivation rather than a separate measurement so the three cannot
# drift apart under a re-pull. The directly pooled figure is 7.8727, which
# agrees with this to 0.0003.
LEAGUE_SIG_ATT_PER_MIN = LEAGUE_SIG_LANDED_PER_MIN / LEAGUE_SIG_ACC

# 3.9563 takedown attempts per 15 minutes, pooled.
LEAGUE_TD_ATT_PER_MIN = 0.26375
# League takedown defence is exactly 1 - 0.3805 = 0.6195.
LEAGUE_TD_ACC = 0.3805

# 12.3970 control seconds per fighter-minute: 20.66% of the clock is somebody's
# control time.
LEAGUE_CTRL_SEC_PER_MIN = 12.3970

# 0.5194 submission attempts per 15 minutes, pooled.
LEAGUE_SUB_ATT_PER_MIN = 0.034627

# 0.3068 knockdowns per 15 minutes. Carried for callers that want a power
# proxy; NOT USED IN ANY FORMULA HERE, because knockdown rate is among the
# least reliable statistics measured (split-half r = +0.316).
LEAGUE_KD_PER_MIN = 0.020453

# 2,787 KO/TKO finishes over 183,687.2 fighter-minutes, one winner and one
# loser each, so the two directions are equal by construction.
LEAGUE_KO_FOR_PER_MIN = 0.015173
LEAGUE_KO_AGAINST_PER_MIN = LEAGUE_KO_FOR_PER_MIN

# 1,626 submission finishes over the same exposure.
LEAGUE_SUB_FOR_PER_MIN = 0.008852
LEAGUE_SUB_AGAINST_PER_MIN = LEAGUE_SUB_FOR_PER_MIN


# --- numerical guards ------------------------------------------------------

# MEASURED, NOT INVENTED. Over 12,730 fighter-bouts where the fighter had 15 or
# more prior minutes, the shrunk ratio-to-league across all eight rate stats
# runs from 0.147 (takedown attempt rate) to 4.205 (submission attempt rate).
# The clip sits just outside that range, so it never binds on a well-observed
# fighter and only bounds the product of two extremes for a thin one. It
# matters because the interaction MULTIPLIES two of these: a rate of exactly 0
# would otherwise assert the fight cannot end that way at all.
RATE_RATIO_FLOOR = 0.12
RATE_RATIO_CAP = 5.0

# Keeps log5's odds finite. Chosen for float64 arithmetic, not from the data;
# after career_stats' shrinkage no fighter with meaningful exposure can reach
# exactly 0 or 1, so this guards a degenerate input rather than a real one.
PROBABILITY_EPS = 1e-6

# Below this the exponential survival expressions are replaced by their
# zero-hazard limits, to avoid a 0/0 in expected_minutes.
HAZARD_EPS = 1e-12


# --- calibrated finish hazards (2011+) -------------------------------------

# The pair's round-1 hazard per minute, converted from the discrete round
# hazard by the competing-risks split of -ln(1 - h_total) / 5. From 1,097
# round-1 KOs in 6,539 three-round bouts and 117 in 680 five-round bouts.
# THAT THE TWO SCHEDULES AGREE TO FIVE DECIMAL PLACES IS A FINDING, NOT A FIT.
LAM_KO_R1_PER_MIN = {3: 0.038803, 5: 0.038808}

# The same for submissions: 589 round-1 submissions in 6,539 three-round bouts
# and 31 in 680 five-round bouts. Five-round main events are half as easy to
# submit, and that is the entire reason their decision rate differs from
# three-rounders' beyond the longer schedule.
LAM_SUB_R1_PER_MIN = {3: 0.020834, 5: 0.010282}

# Round-1-relative continuous hazard by round, among fighters who reached each
# round. 3-round: 1097/625/301 KOs from 6539/4853/3820 reaching. 5-round:
# 117/82/47/27/24 from 680/532/421/350/307. Knockout power fades with fatigue
# and with the selection of durable survivors.
KO_ROUND_SHAPE = {
    3: (1.0, 0.7464, 0.4355),
    5: (1.0, 0.8909, 0.6301, 0.4242, 0.4255),
}

# The same for submissions (3-round: 589/408/203; 5-round: 31/29/24/16/8). The
# submission hazard decays far more slowly than the KO hazard - 3.1 standard
# errors apart in round 2 and 2.5 in round 3 of three-round bouts - which is
# why the two shapes are kept cause-specific rather than shared.
# THE FIVE-ROUND SUBMISSION SHAPE RESTS ON 8 TO 31 EVENTS PER ROUND. Its rise
# in rounds 2 and 3 is not statistically distinguishable from flat.
SUB_ROUND_SHAPE = {
    3: (1.0, 0.9075, 0.5471),
    5: (1.0, 1.1892, 1.2144, 0.9487, 0.5353),
}


# --- observed spreads of the four advantages -------------------------------
# Standard deviations over the 2,551 fights BEFORE 2019 where both fighters had
# 15 or more prior minutes, computed strictly from bouts before each fight.
# Units are real: strikes per minute, takedowns per minute, control seconds per
# minute, submission attempts per minute.
#
# These are spreads of a feature, not fits to an outcome, so they carry no
# calibration leak either way. They are nonetheless measured on the same window
# as the weights, so that every constant in this section has one provenance and
# a reader need not check which. test_matchup.py recomputes all four from the
# dataset through matchup_inputs and fails if any drifts, which is what stops
# them from silently ageing as the dataset grows.
CALIBRATION_CUTOFF = "2019-01-01"   # the same cutoff simulate.py calibrates on
STRIKE_ADV_SD = 0.6828   # strikes per minute
TD_ADV_SD = 0.0842       # 1.26 takedowns over a 15-minute fight
CTRL_ADV_SD = 8.8567     # 2.21 minutes of control over a 15-minute fight
SUB_ADV_SD = 0.0338      # 0.51 submission attempts over 15 minutes

# REFIT ON THE CALIBRATION WINDOW ALONE. The previous 0.60/0.25 came from a
# logistic fit over all 5,085 qualifying fights, outcomes from 2000 to 2026
# included - the same calibration leak simulate.py removes with a 2019 cutoff.
# Refitting on the 2,162 qualifying fights before 2019 gives control +0.2663
# and submission +0.0779 log-odds per SD, a 77:23 split of the non-takedown
# weight rather than 71:29.
#
# THE SIZE OF THAT CORRECTION IS WORTH RECORDING, because it is the opposite
# of what the win-rate leak did. Scored on the held-out 2019-2026 fights:
#
#   0.60/0.25/0.15  (leaked fit)      AUC 0.5675
#   0.66/0.19/0.15  (honest refit)    AUC 0.5667
#   0.77/0.23/0.00  (no takedown)     AUC 0.5676
#   1.00/0.00/0.00  (control only)    AUC 0.5642
#
# Every weighting lands within 0.0034 of every other. The leak was real and is
# removed on principle - no constant here should trace to a fit that saw the
# future - but unlike r_wins it was never carrying the result.
#
# THE TAKEDOWN WEIGHT IS ARBITRARY AND THIS COMMENT IS THE ONLY HONEST
# JUSTIFICATION FOR IT. The honest fit gives takedowns -0.0771, which is not a
# real negative effect but the signature of collinearity with control time at
# r = 0.739. Alone, takedown advantage is worth +0.170 per SD. 0.15 is a FLOOR
# CHOSEN BY JUDGEMENT so that the one grappling signal directly observable
# before a fight is not zeroed out on the strength of an unstable coefficient.
GRAP_W_CTRL = 0.66
GRAP_W_SUB = 0.19
GRAP_W_TD = 0.15


# --- shrinkage constants ---------------------------------------------------
#
# These belong to the input contract, so they live with the contract. Today
# career_stats.py emits UNSHRUNK prior-only ratios and states that it holds no
# smoothing constant anywhere; a one-bout fighter therefore arrives with a
# striking accuracy estimated from ninety attempts. rate_ratio() below is only
# meaningful on a shrunk rate, so shrunk_rate() and its weights are published
# here, where they were measured, rather than being re-derived by each caller.
#
# Every K is a split-half reliability converted by k = m(1 - r)/r, over 1,104
# fighters with six or more bouts, alternating bouts, mean 65.8 minutes or the
# stated attempt count per half.
#
# CAVEAT THAT APPLIES TO ALL OF THEM: split-half reliability conflates noise
# with genuine career change. A fighter who really did improve his takedown
# defence between fight 3 and fight 9 reads as unreliable, so every K is an
# OVER-estimate of the shrinkage needed. The bias is toward advantages that are
# too small rather than too confident, which is the safe direction, but it will
# systematically under-rate fighters who genuinely changed.
K_STRIKE_VOLUME_MIN = 53.9       # r = +0.550, attempts per minute
K_STRIKE_ABSORBED_MIN = 70.0     # r = +0.484, attempts faced per minute
K_TD_RATE_MIN = 31.2             # r = +0.678, the most stable volume stat
K_TD_CONCEDED_MIN = 158.4        # r = +0.293; being shot on is not your choice
K_CTRL_MIN = 46.9                # r = +0.584, the most reliable stat measured
K_CTRL_CONCEDED_MIN = 106.0      # r = +0.383
K_SUB_ATT_MIN = 106.5            # r = +0.382
K_SUB_CONCEDED_MIN = 231.6       # r = +0.221, measured separately
K_KD_MIN = 142.4                 # r = +0.316

# In ATTEMPTS, not minutes. A fighter throws about 110 significant strikes a
# bout, so striking accuracy needs roughly six fights to carry as much weight
# as the league prior.
K_SIG_ACC_ATT = 612.6            # r = +0.465 over a mean 532.9 attempts/half
K_SIG_DEF_ATT = 746.6            # r = +0.409 over 517.4 opponent attempts

# TAKEDOWN ACCURACY IS ESSENTIALLY NOISE: r = +0.071 over a mean 18.6 attempts
# per half. This constant shrinks a typical career's ~37 attempts about 87% of
# the way back to the league mean. It is exactly the quantity the leaked
# r_td_avg_acc column appeared to measure precisely, because a career total
# that includes the future is not noisy at all.
K_TD_ACC_ATT = 245.0
# Takedown defence is roughly three times more reliable: r = +0.237 over a mean
# 17.6 opponent attempts per half.
K_TD_DEF_ATT = 56.7

# JUDGEMENT, NOT MEASUREMENT. Finishes are too rare for a split-half estimate
# of their own, so this was set to match the split-half prior weights of the
# other low-frequency rate statistics (knockdowns 142.4, submission attempts
# 106.5). 110 to 150 is equally defensible. At 120 the pace model reached AUC
# 0.6438 for predicting a finish on 5,085 prior-only fights and the quintile
# calibration in the module docstring.
K_FINISH_MIN = 120.0


# --- input contract --------------------------------------------------------

def _require(mapping: Mapping, cls) -> dict:
    """Pull exactly this dataclass's fields out of a mapping, or raise.

    A missing field raises rather than defaulting, because a silently defaulted
    stat is indistinguishable downstream from a measured league-average one,
    and that is precisely the confusion this module exists to remove.
    """
    missing = [f.name for f in fields(cls) if f.name not in mapping]
    if missing:
        raise KeyError(
            f"{cls.__name__}.from_mapping is missing: {', '.join(missing)}")
    return {f.name: _as_float(mapping[f.name]) for f in fields(cls)}


def _as_float(value) -> float:
    """float(value), with None and unparseable entries as NaN."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


@dataclass(frozen=True)
class Physique:
    """One fighter's frame. CENTIMETRES and KILOGRAMS, as the dataset stores.

    A fighter is about 178 cm tall with a 183 cm reach. feature_spec.py records
    that the previous hand-written version filled these with 70, meaning
    inches, against columns holding centimetres - a fake 110-unit gap on 9.4%
    of fights. Naming the unit in this docstring is the cheapest guard there is.

    weight_kg is the PROFILE LISTED weight, which is a division label rather
    than fight-night mass: it is constant across 97.5% of careers with five or
    more bouts. See mass_advantage.
    """

    height_cm: float = math.nan
    reach_cm: float = math.nan
    weight_kg: float = math.nan

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "Physique":
        return cls(**_require(mapping, cls))


@dataclass(frozen=True)
class Form:
    """One fighter's point-in-time rate and accuracy stats.

    THIS IS THE CONTRACT career_stats.py MUST MEET, and it is the largest risk
    in the module. Every league baseline and every shrinkage constant above
    assumes:

      * rates are PER MINUTE of REPAIRED elapsed time (fight_duration_seconds),
        not per 15 minutes and not per bout;
      * accuracies are FRACTIONS in [0, 1], pooled landed/attempted over PRIOR
        bouts only - career_stats.py already uses this convention and warns
        that it is a deliberate 100x departure from the leaked r_str_acc, which
        reads 46.5 where this reads 0.465;
      * a fighter with no exposure gets NaN, NEVER 0.0.

    If a caller supplies per-15-minute rates instead, every rate_ratio is
    silently wrong by a factor of 15 and NOTHING WILL FAIL - the advantages
    will merely be mis-scaled. There is a test that feeds a synthetic
    league-average fighter and asserts every expected_* function returns its
    league constant to 1e-9, which fails loudly the moment a unit drifts.

    The _conceded fields are what the fighter's OPPONENTS achieved against him,
    which every row of the dataset supplies directly: one bout carries both
    corners' counts. sig_accuracy_conceded is 1 - striking_defence; the
    dataclass stores the conceded form so the module has one convention rather
    than two.

    ZERO-ATTEMPT ACCURACIES ARE THE CALLER'S RESPONSIBILITY. A fighter who has
    never attempted a takedown has an UNDEFINED takedown accuracy, and
    career_stats.py must report NaN for it - 663 red corners in this file have
    no career takedown attempt. This module does not repair it; log5 propagates
    the NaN, which is the correct answer.
    """

    sig_att_per_min: float = math.nan          # significant strikes thrown/min
    sig_att_faced_per_min: float = math.nan    # opponents' attempts/min
    sig_accuracy: float = math.nan             # fraction [0,1], not per cent
    sig_accuracy_conceded: float = math.nan    # = 1 - striking defence
    td_att_per_min: float = math.nan           # takedown attempts/min
    td_att_faced_per_min: float = math.nan     # opponents' td attempts/min
    td_accuracy: float = math.nan              # fraction [0,1]
    td_accuracy_conceded: float = math.nan     # = 1 - takedown defence
    ctrl_sec_per_min: float = math.nan         # control SECONDS per minute
    ctrl_sec_conceded_per_min: float = math.nan
    sub_att_per_min: float = math.nan          # submission attempts/min
    sub_att_conceded_per_min: float = math.nan

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "Form":
        return cls(**_require(mapping, cls))


@dataclass(frozen=True)
class Durability:
    """One fighter's finishing and finishability rates, per minute fought.

    Same contract as Form: per minute of repaired elapsed time, prior bouts
    only, NaN for no exposure. These are rare events - the league finishes 1.52
    KOs and 0.89 submissions per hundred fighter-minutes - so career_stats.py
    is expected to shrink them toward the league with K_FINISH_MIN, or the
    hazards of a three-fight fighter will swing by a factor of five.
    """

    ko_for_per_min: float = math.nan
    ko_against_per_min: float = math.nan
    sub_for_per_min: float = math.nan
    sub_against_per_min: float = math.nan

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "Durability":
        return cls(**_require(mapping, cls))


@dataclass(frozen=True)
class StrikingExchange:
    """Both directions of the striking matchup. Per minute, fractions for acc."""

    red_landed_per_min: float
    blue_landed_per_min: float
    red_attempts_per_min: float
    blue_attempts_per_min: float
    red_accuracy: float
    blue_accuracy: float


@dataclass(frozen=True)
class GrapplingExchange:
    """All six directional grappling quantities, per minute of fight."""

    red_takedowns_per_min: float
    blue_takedowns_per_min: float
    red_control_sec_per_min: float
    blue_control_sec_per_min: float
    red_sub_attempts_per_min: float
    blue_sub_attempts_per_min: float


@dataclass(frozen=True)
class FinishHazards:
    """The four round-1 directional hazards per minute for one specific pair.

    Four numbers rather than one total, because the simulator needs to know who
    won and by what. These are ROUND-1 hazards for the schedule they were built
    for; the round shape is applied downstream by round_outcome and
    sample_fight_outcome, which must be handed the SAME scheduled_rounds that
    directional_finish_hazards was given.
    """

    ko_red_over_blue: float
    ko_blue_over_red: float
    sub_red_over_blue: float
    sub_blue_over_red: float


@dataclass(frozen=True)
class RoundOutcome:
    """The closed-form marginals of one fight. The five probabilities sum to 1."""

    p_decision: float
    p_red_ko: float
    p_blue_ko: float
    p_red_sub: float
    p_blue_sub: float
    expected_seconds: float
    finish_probability_by_round: Tuple[float, ...]


@dataclass(frozen=True)
class SampledFight:
    """One draw. winner is None for a decision; method is 'KO', 'SUB' or 'DEC'."""

    winner: Optional[str]
    method: str
    finish_round: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SimulationSummary:
    """Empirical marginals over `draws` simulated fights, with Monte Carlo SEs.

    At one million draws the Monte Carlo standard error on a probability near
    0.5 is 0.0005, far below this model's own calibration error of about 0.003
    (the worst quintile gap in the module docstring is 1.3 points). MORE THAN A
    MILLION DRAWS BUYS NOTHING; the simulator exists for joint distributions
    and for propagating uncertainty, not for these marginals, which
    round_outcome gives exactly.
    """

    draws: int
    red_win_share: float
    blue_win_share: float
    decision_share: float
    ko_share: float
    sub_share: float
    red_ko_share: float
    blue_ko_share: float
    red_sub_share: float
    blue_sub_share: float
    finish_share_by_round: Tuple[float, ...]
    mean_seconds: float
    median_seconds: float
    red_win_standard_error: float
    decision_standard_error: float


# --- duration repair -------------------------------------------------------

def fight_duration_seconds(match_time_sec: float, finish_round: float,
                           round_seconds: float = ROUND_SECONDS) -> float:
    """True elapsed seconds, repairing the two conventions in match_time_sec.

    That column means two different things in the same place. On 8,486 of 8,587
    rows it is the clock WITHIN the finishing round; on 101 rows dated
    2025-09-13 to 2025-12-06 it is cumulative elapsed time, up to 1500.

    THE DISAMBIGUATION IS EXACT, NOT A HEURISTIC. A cumulative time of 300 or
    less can only come from a round-1 finish, and in round 1 the two
    conventions give the same number, so nothing is ambiguous. Of the 145 rows
    in that scrape window, 101 exceed 300 and the other 44 are round-1 finishes
    where both readings coincide.

    Pure and public because every rate in the engine divides by it, so a
    mistake here is invisible everywhere and wrong everywhere. Reading
    match_time_sec at face value puts the median fight at 4.93 minutes instead
    of 14.18.

    FRAGILE TO THE NEXT DEFECT, DELIBERATELY. The rule works only because a UFC
    round is exactly 300 seconds. A future scrape reporting time REMAINING, or
    a different round length, would be silently misread. engine/career_stats.py
    carries the companion assertion (a duration must land in
    (0, total_rounds * 300]) that would catch it.

    NaN in either argument gives NaN.
    """
    if not (_finite(match_time_sec) and _finite(finish_round)):
        return math.nan
    if match_time_sec > round_seconds:
        return float(match_time_sec)
    return (float(finish_round) - 1.0) * round_seconds + float(match_time_sec)


# --- physique --------------------------------------------------------------

def plausible_height_cm(height_cm: float) -> float:
    """The height in cm if it can be real, else NaN. Never clipped.

    Gates the 17 fights carrying 233.68 to 381.00 cm, all dated 2025-09-13 to
    2025-11-22 and all exact inch multiples (92 to 150 inches) - the same
    scrape window that produced the cumulative match_time_sec block.

    CLIPPING 381 TO 215 WOULD ASSERT THE FIGHTER IS THE TALLEST IN UFC HISTORY,
    which is the same class of error as filling a missing accuracy with 0. NaN
    is the honest answer and the caller decides what to do with it.
    """
    if not _finite(height_cm):
        return math.nan
    return float(height_cm) if HEIGHT_MIN_CM <= height_cm <= HEIGHT_MAX_CM \
        else math.nan


def plausible_reach_cm(reach_cm: float) -> float:
    """The reach in cm if it can be real, else NaN.

    No corruption is present in the reach columns - observed range 147.32 to
    213.36 cm. This gate exists so that a future scrape defect in reach fails
    the way the height one now does, rather than propagating quietly into every
    size advantage on the card.
    """
    if not _finite(reach_cm):
        return math.nan
    return float(reach_cm) if REACH_MIN_CM <= reach_cm <= REACH_MAX_CM \
        else math.nan


def ape_index_cm(height_cm: float, reach_cm: float) -> float:
    """Reach minus height for ONE fighter, in cm, after both gates.

    Positive means longer arms than the frame implies. League distribution over
    16,276 clean fighter-appearances: mean +4.60, sd 4.87, range -22.86 to
    +22.86 (the extremes are exact 9-inch multiples, i.e. rounding of a figure
    recorded in inches).

    Not an advantage: no sign convention applies, this is one fighter.
    """
    return plausible_reach_cm(reach_cm) - plausible_height_cm(height_cm)


def height_advantage_cm(red: Physique, blue: Physique) -> float:
    """Signed height difference in cm. POSITIVE FAVOURS RED.

    Exposed as its own component so the composite's weighting can be audited
    against it rather than taken on trust. Observed sd 6.38 cm over the 8,560
    fights where both heights are plausible; worth +0.0042 log-odds per cm in a
    joint fit with reach, against reach's +0.0120.
    """
    return plausible_height_cm(red.height_cm) - plausible_height_cm(blue.height_cm)


def reach_advantage_cm(red: Physique, blue: Physique) -> float:
    """Signed reach difference in cm. POSITIVE FAVOURS RED.

    The dominant size signal: +0.0120 log-odds per cm in a joint fit with
    height, against height's +0.0042. Observed sd 8.26 cm over 7,777 fights;
    available on 90.6% of fights, because blue-corner reach is missing on 7.74%
    and red-corner reach on 2.49%.
    """
    return plausible_reach_cm(red.reach_cm) - plausible_reach_cm(blue.reach_cm)


def ape_advantage_cm(red: Physique, blue: Physique) -> float:
    """Signed difference of the two ape indices, in cm. POSITIVE FAVOURS RED.

    Reach relative to height is a different question from raw reach, so it is
    answered - and the answer is that IT CARRIES NO INFORMATION ONCE REACH IS
    IN THE MODEL. Alone it is worth +0.0092 log-odds per cm; entered alongside
    reach difference its coefficient goes to -0.0042. SIZE_W_APE is therefore
    0.0 and this function exists as a reported component only.
    """
    return (ape_index_cm(red.height_cm, red.reach_cm)
            - ape_index_cm(blue.height_cm, blue.reach_cm))


def size_advantage(red: Physique, blue: Physique) -> float:
    """One signed, unitless frame advantage. POSITIVE FAVOURS RED.

    A weighted sum of the height and reach differences, each divided by its own
    observed between-corner spread so the two enter on a common scale. Roughly:
    +1.0 is a one-standard-deviation frame edge.

    THE WEIGHTING IS PARTLY ARBITRARY AND SAYING SO IS THE POINT. Height and
    reach differences correlate at r = 0.62 and the data cannot split their
    contributions: bootstrapping 400 resamples of a joint logit puts height's
    share of the combined per-SD effect at mean 0.207 with a 90% interval of
    [-0.223, +0.598], and the raw height coefficient flips sign across eras.
    0.20/0.80 is the rounded bootstrap mean; anything from 0.0 to 0.55 fits the
    data equally well.

    The ape term is written out with a zero coefficient so its exclusion is
    visible in the code rather than merely documented.

    MASS IS DELIBERATELY EXCLUDED - see mass_advantage.

    NaN if either fighter's height or reach is missing or implausible. That is
    7.74% of fights on blue's reach alone, and a zero there would assert an
    even matchup, which is a different claim entirely.
    """
    return (SIZE_W_HEIGHT * (height_advantage_cm(red, blue) / HEIGHT_DIFF_SD_CM)
            + SIZE_W_REACH * (reach_advantage_cm(red, blue) / REACH_DIFF_SD_CM)
            + SIZE_W_APE * (ape_advantage_cm(red, blue) / APE_DIFF_SD_CM))


def mass_advantage(red: Physique, blue: Physique) -> float:
    """Signed weight difference in kg over its spread. POSITIVE FAVOURS RED.

    KEPT OUT OF size_advantage ON PURPOSE. r_weight is the profile listed
    weight, constant across 97.5% of careers with five or more bouts - the same
    signature as the leaking profile columns - and the difference is exactly 0
    on 66.0% of fights. Its distribution is a spike at zero with tails to
    plus and minus 36.29 kg, so WEIGHT_DIFF_SD_KG = 4.99 is the spread of a
    mismatch indicator, not of a continuous size variable.

    What this variable detects is a catchweight, a short-notice replacement or
    a division jump. It does not measure size within a division, and folding it
    into size_advantage would let a division label masquerade as a frame edge.
    """
    return (_as_float(red.weight_kg) - _as_float(blue.weight_kg)) / WEIGHT_DIFF_SD_KG


def size_advantage_known(red: Physique, blue: Physique) -> float:
    """1.0 when every input size_advantage needs is present and plausible.

    Mirrors feature_spec.paired_known so a genuinely even matchup (0.0, known
    1.0) is distinguishable from two fighters with no recorded reach (NaN,
    known 0.0). Without it the model has no way to discount the second.
    """
    return advantage_known(size_advantage(red, blue))


# --- interaction primitives ------------------------------------------------

def rate_ratio(fighter_rate: float, league_rate: float) -> float:
    """A fighter's rate as a multiple of the league mean, clipped.

    THE CLIP IS NOT COSMETIC. Every interaction below multiplies two of these,
    so an unclipped rate of exactly 0 - a fighter with no career submission
    attempt, say - would make the product exactly 0 and assert that the fight
    CANNOT go that way at all. RATE_RATIO_FLOOR bounds it.

    Measured, not invented: over 12,730 fighter-bouts with 15 or more prior
    minutes the observed shrunk ratio runs 0.147 to 4.205 across all eight rate
    stats, so [0.12, 5.0] never binds on a well-observed fighter and only
    bounds the product of two extremes for a thin one.

    NaN in gives NaN out; a missing rate is not a league-average rate.
    """
    if not _finite(fighter_rate) or not _finite(league_rate):
        return math.nan
    return float(np.clip(fighter_rate / league_rate, RATE_RATIO_FLOOR,
                         RATE_RATIO_CAP))


def log5(attacker_rate: float, defender_conceded_rate: float,
         league_rate: float) -> float:
    """Combine two probabilities against a league baseline, in odds space.

    Bill James's log5 (Baseball Abstract, 1981): the accuracy the attacker
    achieves against THIS defender is the attacker's odds times the defender's
    conceded odds divided by the league odds. Two league-average fighters
    return the league rate exactly; a 55%-accurate striker against a defender
    who concedes 40% lands somewhere sensibly between.

    Raises ValueError if league_rate is not strictly inside (0, 1). THAT IS A
    BROKEN CONSTANT, NOT BAD DATA, and it should fail loudly: skill_features.py
    documents what happens when a constant from the wrong scale survives a
    rewrite unchallenged (a threshold no rating could reach, so the adjustment
    it guarded never once fired).

    Operands are clipped to PROBABILITY_EPS so the odds stay finite. NaN in
    either operand gives NaN.
    """
    if not _finite(league_rate) or not (0.0 < league_rate < 1.0):
        raise ValueError(
            f"log5 league_rate must be strictly inside (0, 1); got "
            f"{league_rate!r}. This is a broken constant, not bad data.")
    if not _finite(attacker_rate) or not _finite(defender_conceded_rate):
        return math.nan
    a = min(max(float(attacker_rate), PROBABILITY_EPS), 1.0 - PROBABILITY_EPS)
    b = min(max(float(defender_conceded_rate), PROBABILITY_EPS),
            1.0 - PROBABILITY_EPS)
    odds = (a / (1.0 - a)) * (b / (1.0 - b)) / (league_rate / (1.0 - league_rate))
    return odds / (1.0 + odds)


# --- striking --------------------------------------------------------------

def expected_sig_attempts_per_min(attacker: Form, defender: Form) -> float:
    """How fast the attacker throws at THIS defender. POSITIVE-VALUED rate.

    League attempt rate scaled by the attacker's output ratio and the
    defender's conceded-attempt ratio. Separating pace from accuracy is what
    lets a grinder who smothers output be told apart from a defender who is
    merely hard to hit: the first drags this number down, the second does not
    touch it and shows up in expected_sig_accuracy instead.
    """
    return (LEAGUE_SIG_ATT_PER_MIN
            * rate_ratio(attacker.sig_att_per_min, LEAGUE_SIG_ATT_PER_MIN)
            * rate_ratio(defender.sig_att_faced_per_min, LEAGUE_SIG_ATT_PER_MIN))


def expected_sig_accuracy(attacker: Form, defender: Form) -> float:
    """Significant-strike accuracy the attacker achieves against THIS defender.

    log5 of the attacker's accuracy and the accuracy the defender concedes,
    against the league mean of 0.4483. The conceded form is 1 - striking
    defence; Form stores the conceded form so there is one convention here.
    """
    return log5(attacker.sig_accuracy, defender.sig_accuracy_conceded,
                LEAGUE_SIG_ACC)


def expected_sig_landed_per_min(attacker: Form, defender: Form) -> float:
    """Significant strikes the attacker lands per minute against THIS defender.

    Attempts times accuracy - the single directional striking quantity
    everything else in the module is built from. Two league-average fighters
    return LEAGUE_SIG_LANDED_PER_MIN (3.5292) exactly, which is the unit test
    that catches a rate arriving per-15-minutes instead of per-minute.
    """
    return (expected_sig_attempts_per_min(attacker, defender)
            * expected_sig_accuracy(attacker, defender))


def striking_exchange(red: Form, blue: Form) -> StrikingExchange:
    """Both directions of the striking matchup at once.

    Exposed as a record because the components answer the question the
    composite cannot: a high-volume striker against a porous defender and the
    same striker against an elusive one differ in the ACCURACY fields, not the
    volume fields, and the caller can see which of the two it is looking at.
    """
    return StrikingExchange(
        red_landed_per_min=expected_sig_landed_per_min(red, blue),
        blue_landed_per_min=expected_sig_landed_per_min(blue, red),
        red_attempts_per_min=expected_sig_attempts_per_min(red, blue),
        blue_attempts_per_min=expected_sig_attempts_per_min(blue, red),
        red_accuracy=expected_sig_accuracy(red, blue),
        blue_accuracy=expected_sig_accuracy(blue, red),
    )


def striking_advantage(red: Form, blue: Form) -> float:
    """Net significant strikes landed per minute. POSITIVE FAVOURS RED.

    Red's output against blue's defence minus blue's output against red's
    defence. The units are real strikes per minute, so the number is readable
    rather than an index: observed sd 0.7722, 5th-95th percentile -1.22 to
    +1.27 over 5,085 prior-only fights.

    The strongest single component, at +0.332 log-odds per standard deviation
    in the four-variable joint fit.
    """
    return (expected_sig_landed_per_min(red, blue)
            - expected_sig_landed_per_min(blue, red))


# --- grappling -------------------------------------------------------------

def expected_td_attempts_per_min(attacker: Form, defender: Form) -> float:
    """Takedown attempts the attacker makes per minute against THIS defender."""
    return (LEAGUE_TD_ATT_PER_MIN
            * rate_ratio(attacker.td_att_per_min, LEAGUE_TD_ATT_PER_MIN)
            * rate_ratio(defender.td_att_faced_per_min, LEAGUE_TD_ATT_PER_MIN))


def expected_takedowns_per_min(attacker: Form, defender: Form) -> float:
    """Takedowns the attacker COMPLETES per minute against THIS defender.

    Interacted attempt rate times log5 of the attacker's takedown accuracy with
    the accuracy the defender concedes.

    MOST OF THIS QUANTITY'S SIGNAL COMES FROM THE DEFENDER. Takedown accuracy
    is close to noise - split-half r = +0.071 over a mean 18.6 attempts per
    half - while takedown defence is about three times more reliable at
    r = +0.237. Shrunk by K_TD_ACC_ATT, a typical career's attempts move the
    accuracy about 13% off the league mean and no further. That is exactly the
    quantity the leaked r_td_avg_acc column appeared to measure precisely,
    because a career total that includes the future is not noisy at all.
    """
    return (expected_td_attempts_per_min(attacker, defender)
            * log5(attacker.td_accuracy, defender.td_accuracy_conceded,
                   LEAGUE_TD_ACC))


def expected_control_per_min(attacker: Form, defender: Form) -> float:
    """Control SECONDS the attacker holds per minute of fight. Rate, not share.

    A pure rate interaction with no accuracy term, because control time has no
    attempt denominator in the data - there is no "control attempted" column.

    The most reliable statistic measured anywhere in this dataset: split-half
    r = +0.584 for control held, +0.383 for control conceded.
    """
    return (LEAGUE_CTRL_SEC_PER_MIN
            * rate_ratio(attacker.ctrl_sec_per_min, LEAGUE_CTRL_SEC_PER_MIN)
            * rate_ratio(defender.ctrl_sec_conceded_per_min,
                         LEAGUE_CTRL_SEC_PER_MIN))


def expected_sub_attempts_per_min(attacker: Form, defender: Form) -> float:
    """Submission attempts the attacker makes per minute against THIS defender.

    Submission threat crossed with the rate the defender historically concedes
    attempts - the "submission threat against submission defence" direction.
    The conceded side is the least reliable rate measured here (split-half
    r = +0.221), so K_SUB_CONCEDED_MIN shrinks it hard.
    """
    return (LEAGUE_SUB_ATT_PER_MIN
            * rate_ratio(attacker.sub_att_per_min, LEAGUE_SUB_ATT_PER_MIN)
            * rate_ratio(defender.sub_att_conceded_per_min,
                         LEAGUE_SUB_ATT_PER_MIN))


def grappling_exchange(red: Form, blue: Form) -> GrapplingExchange:
    """All six directional grappling quantities as one record.

    Exposed separately because the joint fit CANNOT untangle takedowns from
    control time (r = 0.739), so a downstream model should be handed the
    components and allowed to weight them itself rather than only the composite
    with its partly-arbitrary weights.
    """
    return GrapplingExchange(
        red_takedowns_per_min=expected_takedowns_per_min(red, blue),
        blue_takedowns_per_min=expected_takedowns_per_min(blue, red),
        red_control_sec_per_min=expected_control_per_min(red, blue),
        blue_control_sec_per_min=expected_control_per_min(blue, red),
        red_sub_attempts_per_min=expected_sub_attempts_per_min(red, blue),
        blue_sub_attempts_per_min=expected_sub_attempts_per_min(blue, red),
    )


def takedown_advantage(red: Form, blue: Form) -> float:
    """Net takedowns completed per minute. POSITIVE FAVOURS RED.

    Observed sd 0.0805 per minute, which is 1.21 takedowns over a 15-minute
    fight. Worth +0.170 log-odds per SD on its own and -0.009 alongside control
    time, which is collinearity (r = 0.739) rather than a real negative effect.
    """
    return (expected_takedowns_per_min(red, blue)
            - expected_takedowns_per_min(blue, red))


def control_advantage(red: Form, blue: Form) -> float:
    """Net control seconds per minute of fight. POSITIVE FAVOURS RED.

    Observed sd 8.39 seconds per minute, which is 2.10 minutes of control over
    a 15-minute fight. THE GRAPPLING VARIABLE THAT ACTUALLY CARRIES THE SIGNAL:
    +0.233 log-odds per SD in the grappling trio, where takedowns go to zero.
    """
    return (expected_control_per_min(red, blue)
            - expected_control_per_min(blue, red))


def submission_advantage(red: Form, blue: Form) -> float:
    """Net submission attempts per minute. POSITIVE FAVOURS RED.

    Observed sd 0.0285 per minute, which is 0.43 attempts over 15 minutes.
    Worth +0.130 log-odds per SD in the full four-variable fit, where it is the
    second-strongest component behind striking.
    """
    return (expected_sub_attempts_per_min(red, blue)
            - expected_sub_attempts_per_min(blue, red))


def grappling_advantage(red: Form, blue: Form) -> float:
    """One signed, unitless grappling number. POSITIVE FAVOURS RED.

    A weighted sum of the three grappling advantages, each divided by its own
    observed spread.

    ONLY THE CONTROL/SUBMISSION SPLIT IS SUPPORTED BY THE FIT. Over 5,085
    prior-only fights the grappling trio gave control +0.233 and submission
    +0.095 log-odds per SD - a 71:29 split that 0.60:0.25 reproduces. The same
    fit gave takedowns -0.009.

    GRAP_W_TD = 0.15 IS ARBITRARY. It is a floor chosen by judgement so that
    the one grappling signal directly observable before a fight is not zeroed
    out on the strength of a coefficient that is unstable because takedowns and
    control correlate at r = 0.739. There is no measurement behind the 0.15 and
    this docstring will not pretend otherwise.
    """
    return (GRAP_W_CTRL * (control_advantage(red, blue) / CTRL_ADV_SD)
            + GRAP_W_SUB * (submission_advantage(red, blue) / SUB_ADV_SD)
            + GRAP_W_TD * (takedown_advantage(red, blue) / TD_ADV_SD))


# --- pace, duration and the finish ----------------------------------------

def _check_rounds(scheduled_rounds) -> int:
    """The schedule as an int, or ValueError naming what is supported.

    total_rounds arrives from the CSV as float64, so 3.0 must be accepted; 3.5
    must not. Only 3 and 5 have calibrated round-shape vectors. The file also
    holds 5 two-round bouts, which is far too few to calibrate anything, and
    SILENTLY REUSING THE 3-ROUND SHAPE FOR THEM WOULD BE AN INVENTED CONSTANT.
    """
    value = _as_float(scheduled_rounds)
    if not _finite(value) or value != int(value):
        raise ValueError(
            f"scheduled_rounds must be a whole number, got "
            f"{scheduled_rounds!r}")
    rounds = int(value)
    if rounds not in SUPPORTED_ROUNDS:
        raise ValueError(
            f"scheduled_rounds must be one of {SUPPORTED_ROUNDS}; got {rounds}. "
            f"Only these have calibrated round-shape vectors. The dataset also "
            f"holds {TWO_ROUND_BOUTS_IN_DATA} two-round bouts, too few to "
            f"calibrate, and reusing the 3-round shape would invent a constant.")
    return rounds


def directional_finish_hazards(red: Durability, blue: Durability,
                               scheduled_rounds: int) -> FinishHazards:
    """The four round-1 per-minute hazards for this specific pair.

    Each is the schedule's round-1 DIRECTIONAL baseline - half the pair
    baseline, since one fighter supplies each direction - scaled by the
    finisher's ratio-to-league and the victim's vulnerability ratio-to-league.
    That is the same multiplicative attack/defence interaction the striking
    rates use, applied to rare events.

    Four directional numbers rather than one total, because the simulator needs
    to know who won and by what, and because "this pair finishes often" and
    "red finishes blue often" are different statements.

    NaN in any input propagates, and simulate_fights refuses to draw from a NaN
    hazard rather than returning a confident coin flip.
    """
    rounds = _check_rounds(scheduled_rounds)
    ko_base = 0.5 * LAM_KO_R1_PER_MIN[rounds]
    sub_base = 0.5 * LAM_SUB_R1_PER_MIN[rounds]
    return FinishHazards(
        ko_red_over_blue=ko_base
        * rate_ratio(red.ko_for_per_min, LEAGUE_KO_FOR_PER_MIN)
        * rate_ratio(blue.ko_against_per_min, LEAGUE_KO_AGAINST_PER_MIN),
        ko_blue_over_red=ko_base
        * rate_ratio(blue.ko_for_per_min, LEAGUE_KO_FOR_PER_MIN)
        * rate_ratio(red.ko_against_per_min, LEAGUE_KO_AGAINST_PER_MIN),
        sub_red_over_blue=sub_base
        * rate_ratio(red.sub_for_per_min, LEAGUE_SUB_FOR_PER_MIN)
        * rate_ratio(blue.sub_against_per_min, LEAGUE_SUB_AGAINST_PER_MIN),
        sub_blue_over_red=sub_base
        * rate_ratio(blue.sub_for_per_min, LEAGUE_SUB_FOR_PER_MIN)
        * rate_ratio(red.sub_against_per_min, LEAGUE_SUB_AGAINST_PER_MIN),
    )


def _round_hazards(hazards: FinishHazards, rounds: int, index: int):
    """The four directional hazards in round `index`, plus their total.

    The KO pair carries the KO shape and the submission pair the submission
    shape, because the two decay differently: by round 3 of a three-rounder the
    KO hazard is at 44% of its round-1 value and the submission hazard at 55%.
    """
    ko_shape = KO_ROUND_SHAPE[rounds][index]
    sub_shape = SUB_ROUND_SHAPE[rounds][index]
    parts = (
        hazards.ko_red_over_blue * ko_shape,
        hazards.ko_blue_over_red * ko_shape,
        hazards.sub_red_over_blue * sub_shape,
        hazards.sub_blue_over_red * sub_shape,
    )
    return parts, sum(parts)


def round_outcome(hazards: FinishHazards, scheduled_rounds: int) -> RoundOutcome:
    """Closed-form piecewise-exponential survival across the scheduled rounds.

    Within a round the hazard is constant, so survival over a round is
    exp(-lambda * 5). Across rounds the hazard steps down by the empirical
    shape vectors. Everything below is the exact integral of that, so A MILLION
    SIMULATIONS ARE NOT NEEDED FOR THESE MARGINALS - the simulator exists for
    joint distributions and for propagating uncertainty, not for these.

    The four directional finish probabilities and p_decision sum to exactly 1
    by construction of the competing-risks decomposition: within a round each
    cause takes its hazard's share of that round's finishes.

    expected_seconds integrates the survival curve. Survival mass that reaches
    the final horn contributes the full round, which is why a decision needs no
    special case. Against 2011+ data this returns 620.8 seconds for a
    league-average 3-round pair (actual mean 630.8) and 917.9 for a 5-round
    pair (actual 927.6) - a 1.1-1.6% UNDER-prediction, because the real
    within-round hazard is not flat: finishes cluster late in a round and this
    model spreads them evenly.

    scheduled_rounds must be the SAME value directional_finish_hazards was
    given; the round-1 baselines differ between the two schedules.
    """
    rounds = _check_rounds(scheduled_rounds)
    survival = 1.0
    cause = [0.0, 0.0, 0.0, 0.0]
    by_round = []
    minutes = 0.0
    for index in range(rounds):
        parts, total = _round_hazards(hazards, rounds, index)
        if not _finite(total):
            return RoundOutcome(math.nan, math.nan, math.nan, math.nan,
                                math.nan, math.nan, (math.nan,) * rounds)
        if total < HAZARD_EPS:
            # Zero-hazard limit taken explicitly: the round cannot end the
            # fight, and it contributes its full five minutes to the duration.
            # Without this the closed forms divide by zero.
            by_round.append(0.0)
            minutes += survival * ROUND_MINUTES
            continue
        finished_here = survival * (1.0 - math.exp(-total * ROUND_MINUTES))
        by_round.append(finished_here)
        minutes += finished_here / total
        for i, part in enumerate(parts):
            cause[i] += finished_here * part / total
        survival *= math.exp(-total * ROUND_MINUTES)
    return RoundOutcome(
        p_decision=survival,
        p_red_ko=cause[0],
        p_blue_ko=cause[1],
        p_red_sub=cause[2],
        p_blue_sub=cause[3],
        expected_seconds=minutes * SECONDS_PER_MINUTE,
        finish_probability_by_round=tuple(by_round),
    )


def probability_of_decision(red: Durability, blue: Durability,
                            scheduled_rounds: int) -> float:
    """Probability the fight reaches the judges. Symmetric in the two corners.

    A thin wrapper over directional_finish_hazards and round_outcome, kept
    because this is the quantity a card prediction quotes and it should not
    require the caller to assemble two objects to get it.

    League-average pair: 0.5071 over three rounds, 0.4044 over five - which are
    the observed 2011+ shares to four decimal places, because the shape vectors
    are fit to exactly those hazards.
    """
    rounds = _check_rounds(scheduled_rounds)
    return round_outcome(directional_finish_hazards(red, blue, rounds),
                         rounds).p_decision


def expected_fight_seconds(red: Durability, blue: Durability,
                           scheduled_rounds: int) -> float:
    """Expected fight length in seconds. Symmetric in the two corners.

    League-average pair: 620.8 seconds over three rounds against an actual mean
    of 630.8, and 917.9 over five against 927.6. The 1.1-1.6% under-prediction
    is the within-round hazard not being flat, and it is an INDEPENDENT check -
    the shape vectors were fit to the decision share, not to the duration.
    """
    rounds = _check_rounds(scheduled_rounds)
    return round_outcome(directional_finish_hazards(red, blue, rounds),
                         rounds).expected_seconds


# --- simulation ------------------------------------------------------------

_CAUSES = (("red", "KO"), ("blue", "KO"), ("red", "SUB"), ("blue", "SUB"))


def sample_fight_outcome(rng: np.random.Generator, hazards: FinishHazards,
                         scheduled_rounds: int) -> SampledFight:
    """Draw one fight from the competing-risks model.

    Round-by-round inverse-CDF sampling of the exponential: in each round draw
    u ~ U(0,1) and set dt = -ln(u) / lambda_total. If dt reaches the horn the
    fight survives the round and the next round's lower hazard applies;
    otherwise it ends at that instant and the cause is drawn in proportion to
    the four directional hazards AS THEY STAND IN THAT ROUND, which is what
    makes a late submission more likely than a late knockout.

    Takes an explicit np.random.Generator and never touches global numpy state,
    so a seeded test pins exact draws.

    Returns winner None and method 'DEC' for a decision. A NaN hazard gives a
    NaN elapsed time and method 'DEC' is NOT asserted - see simulate_fights,
    which refuses the whole run instead.
    """
    rounds = _check_rounds(scheduled_rounds)
    for index in range(rounds):
        parts, total = _round_hazards(hazards, rounds, index)
        if not _finite(total):
            return SampledFight(None, "DEC", rounds, math.nan)
        if total < HAZARD_EPS:
            continue
        dt = -math.log(rng.random()) / total
        if dt >= ROUND_MINUTES:
            continue
        elapsed = (index * ROUND_MINUTES + dt) * SECONDS_PER_MINUTE
        pick = rng.random() * total
        running = 0.0
        for part, (winner, method) in zip(parts, _CAUSES):
            running += part
            if pick < running:
                return SampledFight(winner, method, index + 1, elapsed)
        # Float rounding only; the last cause is the correct fallback.
        return SampledFight(_CAUSES[-1][0], _CAUSES[-1][1], index + 1, elapsed)
    return SampledFight(None, "DEC", rounds,
                        rounds * ROUND_MINUTES * SECONDS_PER_MINUTE)


def simulate_fights(hazards: FinishHazards, scheduled_rounds: int, draws: int,
                    seed: int = 0) -> SimulationSummary:
    """Vectorised repetition of sample_fight_outcome over `draws` fights.

    Implements the SAME model as round_outcome, twice, which is the point and
    the risk: the two can disagree silently. The cross-check test uses 200,000
    draws, which resolves a discrepancy of about 0.002 in p_decision and
    nothing smaller, so it also asserts the simulated ROUND DISTRIBUTION
    against finish_probability_by_round - a sampler applying the wrong round's
    shape would sit under the aggregate threshold but not under that one.

    At one million draws the Monte Carlo standard error on a probability near
    0.5 is 0.0005, far below the model's own calibration error of about 0.003.
    MORE DRAWS THAN THAT BUY NOTHING.

    A NaN hazard returns a summary of NaN shares with draws = 0, rather than a
    confident 50/50 built from a fighter with no recorded history.
    """
    rounds = _check_rounds(scheduled_rounds)
    if draws <= 0:
        raise ValueError(f"draws must be positive, got {draws}")
    per_round = [_round_hazards(hazards, rounds, i) for i in range(rounds)]
    if not all(_finite(total) for _, total in per_round):
        return _unknown_summary(rounds)

    rng = np.random.default_rng(seed)
    # 0 red KO, 1 blue KO, 2 red SUB, 3 blue SUB, 4 decision.
    cause = np.full(draws, 4, dtype=np.int8)
    finish_round = np.zeros(draws, dtype=np.int16)
    elapsed = np.full(draws, rounds * ROUND_SECONDS, dtype=float)
    alive = np.arange(draws)

    for index, (parts, total) in enumerate(per_round):
        if alive.size == 0 or total < HAZARD_EPS:
            continue
        dt = -np.log(rng.random(alive.size)) / total
        ends = dt < ROUND_MINUTES
        if not ends.any():
            continue
        ending = alive[ends]
        elapsed[ending] = (index * ROUND_MINUTES + dt[ends]) * SECONDS_PER_MINUTE
        finish_round[ending] = index + 1
        # The cause is drawn at the finishing instant, from this round's
        # hazards, so the KO/submission mix shifts round by round.
        edges = np.cumsum(np.asarray(parts) / total)
        cause[ending] = np.searchsorted(edges, rng.random(ending.size))
        alive = alive[~ends]

    finish_round[alive] = rounds
    share = lambda mask: float(np.mean(mask))
    red_win = share((cause == 0) | (cause == 2))
    decision = share(cause == 4)
    return SimulationSummary(
        draws=draws,
        red_win_share=red_win,
        blue_win_share=share((cause == 1) | (cause == 3)),
        decision_share=decision,
        ko_share=share((cause == 0) | (cause == 1)),
        sub_share=share((cause == 2) | (cause == 3)),
        red_ko_share=share(cause == 0),
        blue_ko_share=share(cause == 1),
        red_sub_share=share(cause == 2),
        blue_sub_share=share(cause == 3),
        finish_share_by_round=tuple(
            share((finish_round == i + 1) & (cause != 4)) for i in range(rounds)),
        mean_seconds=float(np.mean(elapsed)),
        median_seconds=float(np.median(elapsed)),
        red_win_standard_error=monte_carlo_standard_error(red_win, draws),
        decision_standard_error=monte_carlo_standard_error(decision, draws),
    )


def monte_carlo_standard_error(p: float, draws: int) -> float:
    """sqrt(p(1-p)/draws): how much of a simulated share is sampling noise."""
    if not _finite(p) or draws <= 0:
        return math.nan
    return math.sqrt(p * (1.0 - p) / draws)


def _unknown_summary(rounds: int) -> SimulationSummary:
    """Every share NaN and draws 0, for a pair whose hazards are unknown."""
    n = math.nan
    return SimulationSummary(0, n, n, n, n, n, n, n, n, n,
                             (n,) * rounds, n, n, n, n)


# --- shared helpers --------------------------------------------------------

def shrunk_rate(observed_numerator: float, observed_denominator: float,
                league_rate: float, prior_weight: float) -> float:
    """Empirical-Bayes shrinkage of one rate toward the league mean.

    (numerator + league * k) / (denominator + k): a fighter with k units of
    exposure sits halfway between his own rate and the league's.

    PUBLISHED HERE BECAUSE THE K CONSTANTS WERE MEASURED HERE. career_stats.py
    states that it holds no smoothing constant anywhere and reports thin
    evidence through cd_bouts and the attempt denominators instead, which is
    the right choice for a module whose job is to report what happened.
    rate_ratio(), though, is only meaningful on a shrunk rate: an unshrunk
    one-bout takedown accuracy of 1.0 would sail through the interaction as a
    genuine 100% takedown artist. Every calibration figure in this module's
    docstring was produced with these weights applied.

    NaN denominator or NaN numerator gives NaN; a fighter with no exposure has
    no rate, not a league-average one.
    """
    if not (_finite(observed_numerator) and _finite(observed_denominator)
            and _finite(league_rate) and _finite(prior_weight)):
        return math.nan
    total = observed_denominator + prior_weight
    if total <= 0:
        return math.nan
    return (observed_numerator + league_rate * prior_weight) / total


def advantage_known(*values: float) -> float:
    """1.0 when every value is present and finite, else 0.0.

    THE SINGLE PLACE THIS MODULE IMPLEMENTS THE feature_spec CONVENTION, so no
    caller writes its own and gets the polarity backwards. An even matchup
    returns 0.0 with known 1.0; an unknown one returns NaN with known 0.0, and
    the model needs to be able to tell them apart.
    """
    return 1.0 if all(_finite(v) for v in values) else 0.0


def _finite(value) -> bool:
    """True for a real, finite number. None, NaN and inf are all not that."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
