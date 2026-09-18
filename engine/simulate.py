"""Monte Carlo fight simulator: a distribution over outcomes, not a point guess.

WHY THIS EXISTS

Every other prediction path in this repo ends at a single number - P(red wins).
That number cannot price a method market, a round market, or an over/under on
fight time, and it cannot say whether a 62% favourite is 62% because he wins a
kickboxing match on the cards or 62% because he lands one left hook in every
third simulation. Those are different bets at different prices.

This module steps a fight forward in five-second ticks through three positions
and reports what happened across many thousands of runs: who won, by what
method, in which round, and after how long.

THE LEAK THIS MODULE MUST NOT REINHERIT

The eight r_/b_ profile columns (r_splm, r_str_acc, r_str_def, r_td_avg,
r_td_def, r_sub_avg and friends) are career averages as of the data pull,
joined onto every bout a fighter ever had. 80-90% of fighters with five or more
bouts carry ONE value for their whole career, so a 2014 fight is described by a
2014-2026 striking accuracy that includes the fight being predicted. That leak
inflated the walk-forward backtest to 68.6% and +16.2% over 5,943 bets; the
only honest year was 2026 at 61.5% / -3.2%.

This module therefore takes its inputs as an explicit FighterRates bundle and
never reads the dataset's profile columns. Feed it engine/career_stats.py's
point-in-time cd_* columns (see rates_from_career_stats below) and the
simulation is as honest as those are. Feed it the leaked columns and you will
get a beautifully calibrated simulation of the future.

WHAT CAREER_STATS ACTUALLY SUPPLIES, AND WHAT IT DOES NOT

This section used to say career_stats read no outcome column, so that five of
the thirteen rates below could not come from it and were imputed from the
league for EVERY fighter. It reads winner_id and method now - point-in-time,
prior bouts only, verified against the brute-force recomputation like every
other column - so all five are measured per fighter:

    sub_success_per_att         cd_sub_for_per15 / cd_sub_per15
    sub_loss_per_sub_faced      cd_sub_against_per15 / cd_opp_sub_per15
    kd_abs_per_str_absorbed     cd_opp_kd_per15 / (cd_sapm * 15)

THE TWO KNOCKOUT-PATH RATES ARE NOT SEPARATELY IDENTIFIABLE, and the first
attempt at them was wrong in a way worth recording. This model finishes a
fighter down two routes - a knockdown that gets followed up, and accumulated
head damage - and the dataset records only that a bout ended in KO/TKO, never
which route it took. Dividing a fighter's KO losses by their knockdowns
absorbed therefore does not produce a share: 533 of 2,052 eligible fights had
a fighter KO'd MORE often than knocked down, Charles Oliveira at 1.61 and Jose
Aldo at 1.05, because a doctor stoppage or ground-and-pound needs no knockdown
at all. It also charged the same losses to the accumulation path a second time.

What IS identified is a fighter's TOTAL rate of being knocked out, so both
hazards are scaled by it together:

    durability   = shrunk cd_ko_against_per15 / the league's own rate
    ko_loss_per_kd_absorbed     = KD_TO_FINISH_LEAGUE * durability
    tko_loss_per_head_absorbed  = TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE * durability

A fighter stopped twice as often as the league is twice as easy to stop by
either route, and the league's split between the two routes is left alone
because nothing here can improve on it. This personalises the chin - the whole
point - without inventing a decomposition the data cannot support.

This is the difference between simulating two fighters and simulating two
league-average fighters wearing their names. A chin is not a league constant:
being knocked out is the single most fighter-specific thing in the sport.

The NaN rule is unchanged and still matters, because a ratio needs its
denominator. A fighter who has never been knocked down has no measurable
"knockouts per knockdown absorbed" - that is 0/0, not zero - so it stays NaN,
resolve_rates fills it from the caller's divisional default, and it is named
in FightDistribution.imputed_a / imputed_b. A caller can still always see
which parts of a fighter's durability were assumed. What has changed is how
often the answer is "all of them".

MEASURED ON engine/data/UFC_with_mmr_rebuilt_dedup.csv (8,587 bouts,
2000-11-17 to 2026-08-08). Every constant below carries its own provenance
comment; these are the headline fits:

  league striking     648,264 significant strikes landed / 1,446,118 attempted
                      = 44.83% accuracy, 3.5292 landed per fighter-minute
  positional fit      no-intercept least squares of a fighter's significant
                      strikes on (neutral, own-control, opponent-control)
                      minutes over 16,484 fighter-bouts with elapsed >= 60s:
                      4.9683 / 2.2689 / 0.3770 landed per minute, R^2 0.522
  knockdowns          3,757 / 648,264 landed = 0.005795 per landed strike
  KD conversion       1,949 KO/TKO wins by knockdown-scorers / 3,757
                      knockdowns = 0.5188 per knockdown absorbed
  accumulation TKO    838 of 2,787 KO/TKO losses (30.1%) had ZERO knockdowns
                      recorded against the loser, so a knockdown-only model
                      cannot reach the observed KO rate at all
  grappling           18,434 / 48,448 takedowns = 38.05% accuracy;
                      1,626 submission finishes / 6,361 attempts = 25.56%
  judging             logistic fit on 4,000 real decisions with a clean
                      winner, 84.35% agreement, weights 1 : 1.919 : 2.880 :
                      10.124 for strike : takedown : control-minute : knockdown

  targets a correct run must reproduce (measured, not assumed):
    3-round (n=7,698)  decision 48.78%  KO/TKO 31.74%  submission 19.49%
                       finish rounds 53.01 / 32.01 / 14.99%, mean 617.6s
    5-round (n=772)    decision 38.99%  KO/TKO 44.82%  submission 16.19%
                       finish rounds 37.15 / 26.96 / 18.05 / 10.40 / 7.43%,
                       mean 911.7s
    decisions (n=4,059) split 20.30%, majority 2.49%, draw 1.45%

END-TO-END VALIDATION, RUN RATHER THAN ASSUMED. The simulator was run over
real matchups - point-in-time career_stats rates for both corners, league
defaults for the five outcome-derived rates career_stats cannot supply - on a
random sample of bouts where both fighters had prior history, 1,500 sims each.
It is reported here in full because the misses matter more than the hits:

  3-round, n=1,200      model            actual
    decision            45.5%            52.3%
    KO/TKO              32.6%            28.6%
    submission          21.9%            19.2%
    mean duration       616s             635s
    finish rounds       45.2/31.8/23.0   53.1/33.5/13.4

  5-round, n=569        model            actual
    decision            23.8%            41.1%
    KO/TKO              53.1%            41.8%
    submission          23.1%            17.1%
    mean duration       771s             946s
    finish rounds       36.2/24.7/17.4/12.5/9.2   33.4/28.4/20.0/10.2/8.1

  winner accuracy       55.5% (3-round), 53.1% (5-round)

So: the three-round method split is within about seven points, the five-round
one is not, and BOTH over-produce late finishes. Three causes, in order of
size, none of them fixable inside this module:

1. NO DURABILITY HETEROGENEITY. All five outcome-derived rates are imputed
   from one league default, so every simulated fighter has an identical chin.
   The method split is dominated by them: scaling league durability to 0.7x
   moves the five-round decision rate from 31.5% to 44.5% and the three-round
   rate from 49.9% to 61.6%. Five-round bouts are title and main-event fights
   between unusually durable fighters, which is most of why that column is the
   worse one. The fix is for career_stats.py to emit these five rates
   point-in-time, or for the caller to pass divisional defaults; until then
   FightDistribution.imputed_a / imputed_b name them on every single call.

2. THE ACCUMULATION HAZARD IS CONSTANT PER HEAD STRIKE AND THE REAL ONE IS
   NOT. Measured on this dataset, the TKO rate per head strike absorbed falls
   7.4-fold as damage accumulates:

     head strikes absorbed   0-10     10-25    25-45    45-70   70-120   120+
     TKO per head strike     0.00464  0.00324  0.00203  0.00119 0.00074  0.00062

   Most of that decline is SELECTION - durable fighters survive to absorb many
   strikes - which a model with per-fighter durability would reproduce for
   free, and which a model without it (see 1) cannot reproduce at all. It is
   deliberately NOT patched with a fitted within-fight decay, because that
   would model a between-fighter selection effect as a within-fight mechanism
   and would fit the finish-round curve for the wrong reason.

3. NO FATIGUE, as listed below. This pushes the same way in round 3 and 5.

The model is also corner-agnostic, so its mean red-corner win probability is
49.96% against an actual 58.8%. That gap is matchmaking, not an effect, and
reproducing it would be a bug.

MEASURED THROUGHPUT on this machine, timed with the constants as shipped (the
design asked for a million simulations to be feasible, so it was measured
rather than asserted):

    100,000 sims x 3 rounds, 5s tick     2.5s    (40,400 sims/s)
    1,000,000 sims x 3 rounds, 5s tick  30.7s    (32,600 sims/s)
    1,000,000 sims x 5 rounds, 5s tick  43.9s    (22,800 sims/s)
    1,000,000 sims x 3 rounds, 1s tick 125.0s    ( 8,000 sims/s)

A million simulations is therefore comfortable at the 5s default and slow at
1s. The default 100,000 takes about 2.5 seconds, which is what makes a full
card practical. Only the tick loop is in Python; every draw is vectorised over
simulations, and finished simulations are compacted out so a fight that ends
in round 1 costs nothing thereafter.

The 1s and 5s ticks are statistically equivalent by construction: counts are
drawn Poisson(lambda*dt) and binary events use p = 1 - exp(-lambda*dt), never
lambda*dt as a probability. At the 5s default and league distance striking
lambda*dt = 0.41, so the linear form would lose 18% of events and the answer
would depend on the tick.

WHAT THIS MODEL DOES NOT CAPTURE

It is a rate model. It knows how often a fighter does things, not when or why.

  - No styles and no stylistic matchup. A pressure fighter and a counter
    striker with the same SLpM and striking defence are the same fighter here.
    Southpaw/orthodox, level changes, feints and range management do not exist.
  - No gameplanning or in-fight adaptation. Rates are fixed for 15 or 25
    minutes. Nobody notices the left hook keeps landing and nobody adjusts.
  - No fatigue. Real output falls and finish hazards rise in later rounds.
    This error is one-directional: the model under-prices late finishes and
    over-prices decisions.
  - No cage-control subtlety. Position is one of three states. Pressing an
    opponent against the fence, ring generalship, hand-fighting and the
    difference between half guard and mount are all invisible.
  - No referee tendencies. Stand-up timing is one league-average exponential
    hazard, and the stoppage threshold is one league-average conversion rate.
    Referees vary enormously on both and the model has no per-official term.
  - No judging bias. Simulated judges are unbiased and independent; real ones
    are neither, favour aggression and octagon control in ways the four fitted
    weights only approximate, and score round by round on what they saw.
  - No injuries, weight cuts, layoff, short-notice bookings, altitude, or
    anything else that happens outside the cage.
  - No corner effect. The simulator is corner-agnostic, although the red
    corner wins 61.5% of the decisions it was fitted on - that is matchmaking,
    not an effect, and every fit here pools the two corners so as not to bake
    it in.
  - The KO/TKO class is the dataset's, which merges clean knockouts,
    ground-and-pound stoppages, 86 doctor's stoppages and corner retirements.
    A caller pricing a market that distinguishes them will be wrong.

Finally: the Monte Carlo standard error on a win probability at 1,000,000 sims
is 0.05 percentage points. That is far tighter than the model's own accuracy.
win_prob_se_a reports sampling error only and says nothing about model error.
"""

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np

# --- time base -------------------------------------------------------------

# A round under the UFC Unified Rules, in force for every event in the file
# (earliest bout 2000-11-17, UFC 28). Confirmed on the data: 3,691 of 3,755
# three-round decisions record match_time_sec exactly 300.0.
ROUND_SECONDS = 300.0

# Chosen on measured throughput, not taste - see the timings in the module
# docstring. A million simulations is 7.3s at a 5s tick and 32.6s at 1s, and
# the hazard formulation makes the two statistically equivalent.
DEFAULT_TICK_SECONDS = 5.0

SECONDS_PER_MINUTE = 60.0
PER_15_MINUTES = 15.0

# Only 3 and 5 are calibrated (7,802 and 780 bouts). 2 occurs 5 times in the
# data and is permitted, but nothing here was fitted on it.
SUPPORTED_ROUNDS = (1, 2, 3, 5)
CALIBRATED_ROUNDS = (3, 5)

# Below this the standard error on a win probability exceeds 1.6 percentage
# points (sqrt(0.25/1000) = 0.0158) and the output invites false precision.
MIN_RECOMMENDED_SIMS = 1_000

# --- positions -------------------------------------------------------------

NEUTRAL = 0
CONTROL_A = 1
CONTROL_B = 2

# --- league anchors, all measured on the 8,587-bout dataset ----------------

# 648,264 significant strikes landed over 2 x 91,838.9 fighter-minutes.
LEAGUE_SLPM = 3.5292


# ---------------------------------------------------------------------------
# CALIBRATION WINDOW
# ---------------------------------------------------------------------------
# Every constant below that needs an OUTCOME label - method, winner, or the
# verdict type - is fitted on fights strictly before this date, never on the
# whole file.
#
# The first version fitted them on all 8,587 bouts, 2000 to 2026. That is the
# same defect this module was written to escape, one level up: a walk-forward
# backtest scoring 2020-2026 would have been using constants that had already
# seen those fights' results. It is not cosmetic - refitting on the training
# window alone moves TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE by 24%:
#
#     constant                           whole file    train only
#     KD_TO_FINISH_LEAGUE                  0.518765      0.507456
#     TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE     0.002086      0.002582
#     SUB_SUCCESS_LEAGUE                   0.255620      0.230788
#
# Anything evaluated from 2019 onward is therefore clean. Evaluating BEFORE
# 2019 with these constants is not: they saw those fights. Rate constants that
# need no outcome label (accuracy, volume, position shares) are descriptive
# rather than predictive and are noted individually where it matters.
CALIBRATION_CUTOFF = "2019-01-01"
CALIBRATION_BOUTS = 4692        # of 8,587, 2000-11-17 to 2018-12-29

# 648,264 landed / 1,446,118 attempted, pooled over both corners.
STR_ACC_LEAGUE = 0.4483

# One minus the pooled accuracy: the average fighter avoids 55.17% of the
# significant strikes thrown at him.
STR_DEF_LEAGUE = 1.0 - STR_ACC_LEAGUE

# No-intercept least-squares fit of each fighter's significant strikes landed
# on (neutral minutes, own-control minutes, opponent-control minutes) over the
# 16,484 fighter-bout rows with elapsed >= 60s: 4.9683, 2.2689 and 0.3770
# landed per minute, R^2 = 0.522. Divided by LEAGUE_SLPM to become multipliers.
POSITION_STRIKE_LANDED_PER_MIN = {"neutral": 4.9683, "top": 2.2689, "bottom": 0.3770}
POSITION_STRIKE_MULT = {
    key: value / LEAGUE_SLPM
    for key, value in POSITION_STRIKE_LANDED_PER_MIN.items()
}  # neutral 1.4078, top 0.6429, bottom 0.1068

# Mean of (elapsed - r_ctrl - b_ctrl)/elapsed and (r_ctrl + b_ctrl)/elapsed
# over the same 16,484 rows. Control is split evenly between the two fighters
# for this budgeting arithmetic, so each is top half the control time.
NEUTRAL_TIME_SHARE = 0.5835
CONTROL_TIME_SHARE = 1.0 - NEUTRAL_TIME_SHARE
OWN_CONTROL_TIME_SHARE = CONTROL_TIME_SHARE / 2.0

# The time-weighted sum of the three multipliers. They are divided by it so a
# league-average fighter spending a league-average time split lands exactly his
# stated SLpM. Without it the simulator loses 2.3% of all striking output - the
# regression's three positional means do not reproduce the pooled total,
# because the fit is loose (R^2 0.522) and the positions are unbalanced.
POSITION_MULT_NORMALISER = (
    NEUTRAL_TIME_SHARE * POSITION_STRIKE_MULT["neutral"]
    + OWN_CONTROL_TIME_SHARE * POSITION_STRIKE_MULT["top"]
    + OWN_CONTROL_TIME_SHARE * POSITION_STRIKE_MULT["bottom"]
)  # 0.9776

# 3,757 knockdowns / 648,264 significant strikes landed. Cross-checked as
# position-independent: the positional regression gives 0.0178 knockdowns per
# neutral minute against 4.9683 landed (0.00358 per strike) and 0.00849 per
# top minute against 2.2689 landed (0.00374 per strike), so one position-free
# constant is used. The mirror quantity, knockdowns absorbed per significant
# strike absorbed, is 0.005786 on the rows where both are present - the same
# number, as it must be, which is what makes the ratio composition symmetric.
KD_PER_LANDED_LEAGUE = 0.005795

# 401,810 head / (401,810 head + 131,957 body + 103,942 leg) = 0.6301.
# NOTE the denominator: head+body+leg is 637,709, not the 648,264 total
# significant strikes, because 8,442 of 8,587 bouts carry a target breakdown
# and the rest do not. Dividing by the full significant total would understate
# the head share as 0.6198 and silently under-produce accumulation TKOs.
HEAD_SHARE_OF_SIG = 0.6301

# KO/TKO wins by sides that scored at least one knockdown, divided by the
# knockdowns those sides scored: 1,055 / 2,079 inside the calibration window.
# The coarser P(KO win | at least one KD) would over-finish anyone who scores
# multiple knockdowns in one fight. Needs `method` and `winner`, so it is
# fitted pre-2019; the whole-file value is 0.5188.
KD_TO_FINISH_LEAGUE = 0.5075

# 2,789 KO/TKO losses over 183,687.2 fighter-minutes. Identical to the KO-for
# rate by construction, since every knockout has one of each, and it agrees
# with matchup.LEAGUE_KO_FOR_PER_MIN (0.015173) to four decimal places from a
# completely separate derivation.
KO_AGAINST_PER_MIN_LEAGUE = 0.015183

# How far a single fighter's durability may depart from the league before it is
# treated as a small-sample artefact rather than a chin. Shrinkage with
# K_FINISH_MIN already pulls a thin record most of the way back; this bounds
# what survives it. 0.25 to 3.0 spans "stopped a quarter as often as the
# league" to "three times as often", which covers every real career and nothing
# beyond one.
DURABILITY_FLOOR = 0.25
DURABILITY_CAP = 3.0

# KO/TKO losses in which the loser absorbed zero recorded knockdowns, divided
# by all head strikes landed: 501 / 194,009 inside the calibration window.
# This path is not optional - roughly 30% of KO/TKO losses record no
# knockdown, so a knockdown-only model tops out near 22% KO against an
# observed 31.7%. Needs `method` and `winner`, so it is fitted pre-2019; the
# whole-file value is 0.002086, 24% lower, which is the largest shift any
# constant here shows and the clearest evidence the window matters.
TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE = 0.002582

# 18,434 takedowns landed / 48,448 attempted, pooled over both corners.
TD_ACC_LEAGUE = 0.3805
TD_DEF_LEAGUE = 1.0 - TD_ACC_LEAGUE

# 48,448 attempts over 2 x 91,838.9 fighter-minutes, x15.
TD_ATT_PER15_LEAGUE = 3.9563

# Takedown attempts per minute by position, from the same no-intercept fit:
# 0.1664 neutral, 0.8159 own-control, 0.0025 opponent-control (R^2 0.400). The
# own-control figure is re-takedowns and position advancement inside a control
# spell the simulator is already in, so only the neutral share is shot.
TD_ATT_LANDED_PER_MIN = {"neutral": 0.1664, "top": 0.8159, "bottom": 0.0025}
TD_NEUTRAL_SHARE = (
    TD_ATT_LANDED_PER_MIN["neutral"] * NEUTRAL_TIME_SHARE
    / (
        TD_ATT_LANDED_PER_MIN["neutral"] * NEUTRAL_TIME_SHARE
        + TD_ATT_LANDED_PER_MIN["top"] * OWN_CONTROL_TIME_SHARE
        + TD_ATT_LANDED_PER_MIN["bottom"] * OWN_CONTROL_TIME_SHARE
    )
)  # 0.3629

# Converts a fighter's takedown attempts per 15 minutes of TOTAL time (the
# published UFCStats definition, which is what career_stats emits) into
# attempts per minute of NEUTRAL time, the only position the simulator shoots
# from. League check: 3.9563/15 * 0.6220 = 0.1640 per neutral minute against
# the regression's 0.1664, 1.4% low.
TD_ATT_NEUTRAL_FACTOR = TD_NEUTRAL_SHARE / NEUTRAL_TIME_SHARE  # 0.6220

# Total control seconds divided by total takedowns landed, over the 7,812
# fighter-bouts with at least one takedown: 793,096 / 7,812 = 101.5s. Used as
# the mean of an exponential stand-up hazard.
#
# The mean of per-fighter ratios is 115.8s, and a design written before this
# was measured used that figure. The pooled ratio is the correct one for an
# exponential mean - the per-fighter mean is dominated by fighters with one
# takedown and one long spell - so 101.5 is used and the 13% gap is recorded
# here rather than buried.
CONTROL_SPELL_SEC = 101.5

# CALIBRATED against the measured control share, not derived analytically.
#
# The analytic derivation gives 0.1485: sustaining the measured 41.65% control
# share with a 101.5s mean spell needs (0.4165/0.5835) * 60/101.5 / 2 = 0.2110
# control entries per fighter per neutral minute, and takedowns supply only
# 0.1640 * 0.3805 = 0.0624 of them. But that steady-state arithmetic assumes
# control spells run to completion, and in the simulator - as in a real fight -
# every spell is cut short by the end of the round. Using 0.1485 produces a
# simulated control share of 32.8% against the observed 41.65%, a 21% shortfall
# entirely explained by that truncation, which the observed share already
# contains.
#
# So the number is solved instead: bisecting a league-vs-league 3-round
# simulation onto the observed 41.65% control share gives 0.2459, which
# reproduces it at 41.60%. This is pinned by
# test_the_simulated_control_share_matches_the_observed_one.
#
# It remains the least trustworthy constant in the module, for a reason that
# calibration does not fix: it accounts for roughly three quarters of all
# control entries (0.2459 of 0.3083) and is driven by no fighter statistic at
# all, so a dominant wrestler and a pure striker spend the same league-average
# time in clinch control. See the honesty note in simulate_fight's docstring.
CLINCH_CONTROL_ENTRY_PER_NEUTRAL_MIN = 0.2459

# 6,361 submission attempts over 2 x 91,838.9 fighter-minutes, x15.
SUB_ATT_PER15_LEAGUE = 0.5194

# 1,626 submission finishes / 6,361 attempts, pooled.
SUB_SUCCESS_LEAGUE = 0.2308

# Submission attempts per minute by position from the same fit: -0.0038
# neutral, 0.0820 own-control, 0.0651 opponent-control. The neutral
# coefficient is negative and indistinguishable from zero, which is why the
# simulator only attempts submissions from a control position. The two control
# figures are divided by the league 0.03463 attempts per minute of total time
# (0.5194/15). The bottom factor is not a rounding artefact: guillotines,
# armbars and triangles from guard are real, and a model that lets only the
# top fighter attack loses all of them.
SUB_TOP_FACTOR = 0.0820 / (SUB_ATT_PER15_LEAGUE / PER_15_MINUTES)  # 2.368
SUB_BOTTOM_FACTOR = 0.0651 / (SUB_ATT_PER15_LEAGUE / PER_15_MINUTES)  # 1.880

# --- judging ---------------------------------------------------------------

# Logistic regression (no intercept) of the decision winner on the four
# whole-fight margins, over the 4,000 decisions with a clean winner and
# complete statistics, expressed relative to one significant strike.
# Cross-checked by exhaustive grid search over integer weights, which found
# (1, 2, 2, 9) at 84.50% against the regression's 84.35%; the fitted values are
# used because they are fitted rather than snapped to a grid.
SCORE_W_STRIKE = 1.0
SCORE_W_TAKEDOWN = 1.9192
SCORE_W_CONTROL_MIN = 2.8804
SCORE_W_KNOCKDOWN = 10.1239

# Solved from E[Phi(-|margin|/s)] = 0.1565, the residual disagreement between
# the deterministic scorer above and the real verdicts over those same 4,000
# decisions. It is an UPPER BOUND on judging noise, not an estimate of it,
# because it conflates genuine judging idiosyncrasy with the linear scorer's
# own misspecification - real judges score round by round on what they saw, not
# on whole-fight totals. It is used only as the bisection bracket in
# calibrate_judge_noise.
JUDGE_NOISE_SD_FIGHT_UPPER_BOUND = 22.282

# Observed decision composition over all 4,059 decisions: 824 split (20.30%),
# 101 majority (2.49%), 59 draws (1.45%). The split share is the only directly
# observable anchor and is what calibrate_judge_noise bisects on.
OBSERVED_SPLIT_SHARE = 0.2030
OBSERVED_MAJORITY_SHARE = 0.0249
OBSERVED_DRAW_SHARE = 0.0145

# A round whose noisy margin exceeds this is scored 10-8 rather than 10-9.
#
# THIS IS A DEPARTURE FROM 10-9-ONLY SCORING AND IT IS NOT OPTIONAL. With only
# 10-9 rounds and an odd number of rounds, a judge's card can never be level,
# so a majority decision and a draw are ARITHMETICALLY IMPOSSIBLE - a first
# implementation without this term produced exactly 0.00% of each against an
# observed 2.49% and 1.45%. Real majority decisions and draws exist precisely
# because judges award 10-8 rounds, so reproducing them requires modelling one.
#
# Value solved jointly with DEFAULT_JUDGE_NOISE_SD by grid search over the
# 3,703 real three-round decisions, targeting the observed split and draw
# shares; see that constant for the fit and its residuals.
DOMINANT_ROUND_MARGIN = 14.0

# Per-round judge noise, in the same strike units as round_score_margin.
#
# Solved jointly with DOMINANT_ROUND_MARGIN over the 3,703 real three-round
# decisions with a clean winner and complete statistics, each fight's
# whole-fight margin split evenly across its three rounds. Fitted on split and
# draw share; majority and unanimous are then free and come out close:
#
#     split      20.25%   observed 20.30%
#     draw        1.46%   observed  1.45%
#     majority    2.91%   observed  2.49%    (not targeted)
#     unanimous  75.38%   observed 77.72%    (not targeted)
#
# Splitting a whole-fight margin evenly across rounds is an approximation
# forced by the data: the dataset carries no per-round statistics at all, so
# round-level noise cannot be fitted directly. It removes real round-to-round
# variation, so this is a floor on the true figure.
#
# The deeper caveat, which calibration does not remove: this number is the
# residual between a linear scorer and real verdicts, and that is not the same
# thing as judging noise. It also contains the scorer's own misspecification -
# real judges score round by round on what they saw, not on whole-fight totals
# of strikes and control seconds. A model tuned to hit the split share may be
# hitting it for the wrong reason.
DEFAULT_JUDGE_NOISE_SD = 7.4

VERDICT_UNANIMOUS = 0
VERDICT_SPLIT = 1
VERDICT_MAJORITY = 2
VERDICT_DRAW = 3
VERDICT_TYPE_NAMES = {
    VERDICT_UNANIMOUS: "unanimous",
    VERDICT_SPLIT: "split",
    VERDICT_MAJORITY: "majority",
    VERDICT_DRAW: "draw",
}

# --- reach -----------------------------------------------------------------

# +0.0147 significant strikes landed per minute per cm of reach advantage over
# 14,902 fighter-bouts with both reaches present (correlation 0.047), and
# -0.00038 accuracy per cm. The signs are the interesting part and they are
# consistent: longer fighters throw more and land a lower share, so reach is
# applied as a VOLUME term and never as an accuracy term.
#
# DEFAULT OFF. A fighter's SLpM and striking defence already contain his own
# reach as measured against real opposition, so applying it again double-counts
# it. Enable it only for a fighter whose rates are a divisional default.
REACH_SLPM_PER_CM = 0.0147
REACH_ACC_PER_CM = -0.00038

# A composed multiplier above this many times its league value is not refused -
# it is recorded in FightDistribution.extreme_multipliers so the caller can
# discount the result. Shrinking noisy per-fighter rates is career_stats' job,
# not this module's, so no clip threshold of its own is invented.
EXTREME_MULTIPLIER = 5.0

# Fields that are shares and must lie in [0, 1]; the rest are non-negative
# rates. Declared once so validate_rates and resolve_rates cannot disagree.
_SHARE_FIELDS = (
    "str_acc", "str_def", "td_acc", "td_def",
    "sub_success_per_att", "kd_per_str_landed", "kd_abs_per_str_absorbed",
    "ko_loss_per_kd_absorbed", "tko_loss_per_head_absorbed",
    "sub_loss_per_sub_faced",
)
_RATE_FIELDS = ("slpm", "sapm", "td_att_per15", "sub_att_per15")

# A defence of exactly 1.0 makes a fighter literally untouchable for 25
# minutes. It is always a one-bout small-sample artefact.
_UNIT_DEFENCE_FIELDS = ("str_def", "td_def")

# Fields resolve_rates must see a finite value for. name, reach_cm and bouts
# are metadata, not rates.
_REQUIRED_RATE_FIELDS = _RATE_FIELDS + _SHARE_FIELDS


@dataclass(frozen=True)
class FighterRates:
    """The point-in-time rate bundle one fighter enters the simulation with.

    Every field is a rate or a share, never a count, so it is dimensionally
    safe to mix a fighter with three bouts against one with thirty.

    `bouts` is carried only so the caller can judge how much to trust the
    rates. The simulator does not shrink them toward any mean - that is
    career_stats' job, and career_stats deliberately declines to do it,
    reporting evidence counts instead of hiding thin samples inside a prior.

    `sapm` is carried for completeness and for the caller's own diagnostics but
    is NOT a driver: strikes absorbed are produced by the opponent's landing
    rate against this fighter's str_def, so using sapm as well would count the
    same thing twice.

    The five outcome-derived fields (sub_success_per_att, kd_abs_per_str_-
    absorbed, ko_loss_per_kd_absorbed, tko_loss_per_head_absorbed,
    sub_loss_per_sub_faced) cannot come from career_stats.py, which reads no
    outcome column by design. Leave them NaN and let resolve_rates fill them
    from a divisional default.
    """

    name: str
    slpm: float
    str_acc: float
    sapm: float
    str_def: float
    td_att_per15: float
    td_acc: float
    td_def: float
    sub_att_per15: float
    sub_success_per_att: float
    kd_per_str_landed: float
    kd_abs_per_str_absorbed: float
    ko_loss_per_kd_absorbed: float
    tko_loss_per_head_absorbed: float
    sub_loss_per_sub_faced: float
    reach_cm: float = float("nan")
    bouts: int = 0


def league_rates(name="league"):
    """The pooled league-average fighter, every field measured on the data.

    Offered as an explicit last-resort argument to `defaults`; it is never
    applied silently, because a divisional average is a much better fallback
    than a league one and the caller usually has it. A flyweight and a
    heavyweight do not share a knockdown rate.
    """
    return FighterRates(
        name=name,
        slpm=LEAGUE_SLPM,
        str_acc=STR_ACC_LEAGUE,
        # 648,264 strikes absorbed over the same fighter-minutes: by
        # construction identical to the pooled SLpM, since one fighter's landed
        # strike is the other's absorbed one.
        sapm=LEAGUE_SLPM,
        str_def=STR_DEF_LEAGUE,
        td_att_per15=TD_ATT_PER15_LEAGUE,
        td_acc=TD_ACC_LEAGUE,
        td_def=TD_DEF_LEAGUE,
        sub_att_per15=SUB_ATT_PER15_LEAGUE,
        sub_success_per_att=SUB_SUCCESS_LEAGUE,
        kd_per_str_landed=KD_PER_LANDED_LEAGUE,
        kd_abs_per_str_absorbed=KD_PER_LANDED_LEAGUE,
        ko_loss_per_kd_absorbed=KD_TO_FINISH_LEAGUE,
        tko_loss_per_head_absorbed=TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE,
        sub_loss_per_sub_faced=SUB_SUCCESS_LEAGUE,
        reach_cm=float("nan"),
        bouts=0,
    )


def rates_from_career_stats(row, corner, *, name=None):
    """Adapt one career_stats.py output row into a FighterRates bundle.

    This is the reconciliation between what career_stats emits and what the
    simulator needs, and it is not a straight rename. Two fields are converted:

        td_att_per15       career_stats emits cd_td_per15, takedowns LANDED per
                           15 minutes. Attempts are landed / accuracy.
        kd_per_str_landed  career_stats emits cd_kd_per15, knockdowns per 15
                           minutes. Per landed strike is that divided by
                           cd_slpm * 15.

    and five are left NaN because career_stats reads no outcome column and so
    cannot produce them at all - see the module docstring. They must be
    supplied through resolve_rates' `defaults`, and they will be reported as
    imputed.

    `row` is any mapping keyed by the emitted column names, so a DataFrame row,
    a dict from a single-fight prediction, or a test fixture all work.
    """
    def get(field):
        value = row.get(f"{corner}_{field}")
        return float("nan") if value is None else float(value)

    slpm = get("cd_slpm")
    td_per15 = get("cd_td_per15")
    td_acc = get("cd_td_acc")
    kd_per15 = get("cd_kd_per15")

    # A takedown accuracy of 0 means the fighter has attempted takedowns and
    # landed none. His attempt rate is then unknown from these two columns
    # alone (0/0), not zero, so it stays NaN rather than becoming a fighter who
    # never shoots.
    td_att_per15 = td_per15 / td_acc if td_acc and td_acc > 0 else float("nan")

    # Likewise a fighter who has landed no strikes has no measurable knockdown
    # rate per strike.
    sig_per15 = slpm * PER_15_MINUTES
    kd_per_landed = kd_per15 / sig_per15 if sig_per15 > 0 else float("nan")

    if name is None:
        name = row.get(f"{corner}_name") or f"{corner}-corner"

    bouts = row.get(f"{corner}_cd_bouts")
    reach = row.get(f"{corner}_reach")

    # The five outcome-derived rates. Each is a ratio whose denominator is the
    # exposure that could have produced it, so a fighter who has never faced
    # the event has no rate rather than a rate of zero: never knocked down is
    # not the same as unknockoutable, and 0/0 is NaN, not 0.
    def ratio(top, bottom):
        numerator, denominator = get(top), get(bottom)
        if not (denominator > 0) or numerator != numerator:
            return float("nan")
        return numerator / denominator

    sapm = get("cd_sapm")
    absorbed_per15 = sapm * PER_15_MINUTES
    opp_kd_per15 = get("cd_opp_kd_per15")
    kd_abs_per_absorbed = (opp_kd_per15 / absorbed_per15
                           if absorbed_per15 > 0 else float("nan"))

    # One durability multiplier for both knockout routes; see the docstring.
    # Clipped at the top because a probability cannot exceed 1 however fragile
    # the fighter, and at the bottom away from 0 because nobody is unstoppable.
    league_ko_against_per15 = KO_AGAINST_PER_MIN_LEAGUE * PER_15_MINUTES
    ko_against = get("cd_ko_against_per15")
    if ko_against == ko_against and league_ko_against_per15 > 0:
        durability = ko_against / league_ko_against_per15
        durability = min(max(durability, DURABILITY_FLOOR), DURABILITY_CAP)
        ko_per_kd = min(KD_TO_FINISH_LEAGUE * durability, 1.0)
        tko_per_head = min(TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE * durability, 1.0)
    else:
        ko_per_kd = float("nan")
        tko_per_head = float("nan")

    return FighterRates(
        name=str(name),
        slpm=slpm,
        str_acc=get("cd_str_acc"),
        sapm=sapm,
        str_def=get("cd_str_def"),
        td_att_per15=td_att_per15,
        td_acc=td_acc,
        td_def=get("cd_td_def"),
        sub_att_per15=get("cd_sub_per15"),
        sub_success_per_att=ratio("cd_sub_for_per15", "cd_sub_per15"),
        kd_per_str_landed=kd_per_landed,
        kd_abs_per_str_absorbed=kd_abs_per_absorbed,
        ko_loss_per_kd_absorbed=ko_per_kd,
        tko_loss_per_head_absorbed=tko_per_head,
        sub_loss_per_sub_faced=ratio("cd_sub_against_per15",
                                     "cd_opp_sub_per15"),
        reach_cm=float("nan") if reach is None else float(reach),
        bouts=0 if bouts is None or bouts != bouts else int(bouts),
    )


def resolve_rates(rates, defaults=None):
    """Fill NaN rates from `defaults`; return (filled, frozenset of imputed).

    The single choke point for the missing-data policy. Nothing downstream may
    see a NaN, and nothing anywhere may see a zero that meant "unknown".

    A missing rate filled with 0 is not a neutral assumption, it is the most
    extreme assumption available: a zero SLpM is a fighter who never throws, a
    zero striking defence is a fighter hit by every strike aimed at him, and a
    zero knockdown rate is a fighter who cannot hurt anyone. That inversion is
    exactly the defect engine/feature_spec.py was rewritten to remove, where a
    debutant's missing 45% striking accuracy became 0% and handed his opponent
    a fake maximal advantage on 9.4% of fights.

    Raises ValueError naming the field when a rate is NaN and no default
    supplies it, because a caller who has no divisional average should find out
    loudly rather than receive a confident simulation of a fighter nobody
    measured.
    """
    imputed = []
    updates = {}
    for field in _REQUIRED_RATE_FIELDS:
        value = getattr(rates, field)
        if value == value:  # not NaN
            continue
        fallback = getattr(defaults, field, float("nan")) if defaults is not None else float("nan")
        if fallback != fallback:
            raise ValueError(
                f"{rates.name}: {field} is missing and no default supplies it. "
                "Pass `defaults` as the divisional average (or league_rates() "
                "as a last resort); it is never filled with 0."
            )
        updates[field] = float(fallback)
        imputed.append(field)

    filled = replace(rates, **updates) if updates else rates
    return filled, frozenset(imputed)


def validate_rates(rates):
    """Raise ValueError on inputs that make a fighter impossible, not unusual.

    Deliberately permissive about the merely extreme - a 6.5 SLpM or a 0.85
    takedown defence is a real fighter - and strict about the three things that
    break the simulation rather than skew it: a negative rate, a share outside
    [0, 1], and a defence of exactly 1.0. The last is always a one-bout
    artefact (one fight, one takedown attempt, one stuff) and accepting it
    simulates a fighter who cannot be touched for 25 minutes.
    """
    for field in _RATE_FIELDS:
        value = getattr(rates, field)
        if value != value:
            raise ValueError(f"{rates.name}: {field} is NaN; call resolve_rates first")
        if value < 0:
            raise ValueError(f"{rates.name}: {field} is negative ({value})")

    for field in _SHARE_FIELDS:
        value = getattr(rates, field)
        if value != value:
            raise ValueError(f"{rates.name}: {field} is NaN; call resolve_rates first")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{rates.name}: {field} is {value}, outside [0, 1]; it is a share"
            )

    for field in _UNIT_DEFENCE_FIELDS:
        if getattr(rates, field) == 1.0:
            raise ValueError(
                f"{rates.name}: {field} is exactly 1.0, which makes the fighter "
                "invulnerable for the whole fight. This is a small-sample "
                "artefact; shrink it or supply a divisional default."
            )


# --- pure composition rules ------------------------------------------------
#
# All fighter-vs-fighter composition uses the ratio form
#     rate_attacker * (1 - def_defender) / (1 - def_LEAGUE)
# so a fighter's own statistic - which was already measured against average
# opposition, and therefore already contains an average opponent's defence - is
# not charged against this opponent a second time. When both sides are
# league-average the formula returns the league value exactly, which is the
# property every test below pins.


def strike_land_rate(slpm, opp_str_def, position_mult):
    """Significant strikes landed per minute, for this attacker, defender and
    position.

    `position_mult` is a raw entry of POSITION_STRIKE_MULT; the normaliser is
    applied here so that the three positions, weighted by the league time
    split, average to exactly the attacker's stated SLpM. The neutral value for
    a league pair is 5.082/min, which is the regression's 4.968 grossed up by
    the 2.3% the three positional means fail to account for.

    Pure and vectorised with no RNG, because this is where a double-count of
    the average opponent would hide.
    """
    slpm = np.asarray(slpm, dtype=float)
    opp_str_def = np.asarray(opp_str_def, dtype=float)
    ratio = (1.0 - opp_str_def) / (1.0 - STR_DEF_LEAGUE)
    return slpm * (position_mult / POSITION_MULT_NORMALISER) * ratio


def knockdown_probability(att_kd_per_landed, def_kd_abs_per_absorbed):
    """Probability that one landed significant strike is a knockdown.

    Composes the attacker's power with the defender's chin around the league
    rate. Both league-average returns the league 0.005795 exactly.
    """
    att = np.asarray(att_kd_per_landed, dtype=float)
    dfn = np.asarray(def_kd_abs_per_absorbed, dtype=float)
    return np.clip(att * dfn / KD_PER_LANDED_LEAGUE, 0.0, 1.0)


def takedown_success_probability(att_td_acc, def_td_def):
    """Probability a takedown attempt completes.

    Both league-average returns the league accuracy 0.3805 exactly.
    """
    att = np.asarray(att_td_acc, dtype=float)
    dfn = np.asarray(def_td_def, dtype=float)
    return np.clip(att * (1.0 - dfn) / (1.0 - TD_DEF_LEAGUE), 0.0, 1.0)


def submission_success_probability(att_sub_success, def_sub_loss):
    """Probability a submission attempt finishes the fight.

    Both league-average returns the league 0.2556 exactly.
    """
    att = np.asarray(att_sub_success, dtype=float)
    dfn = np.asarray(def_sub_loss, dtype=float)
    return np.clip(att * dfn / SUB_SUCCESS_LEAGUE, 0.0, 1.0)


def round_score_margin(sig_a, sig_b, td_a, td_b, ctrl_sec_a, ctrl_sec_b, kd_a, kd_b):
    """Scoring margin for one round in favour of A, in units of one strike.

    Pure and vectorised so the weights can be re-fitted and pinned by a test
    against the 4,000 real decisions without running a simulation.
    """
    return (
        SCORE_W_STRIKE * (np.asarray(sig_a, dtype=float) - np.asarray(sig_b, dtype=float))
        + SCORE_W_TAKEDOWN * (np.asarray(td_a, dtype=float) - np.asarray(td_b, dtype=float))
        + SCORE_W_CONTROL_MIN
        * (np.asarray(ctrl_sec_a, dtype=float) - np.asarray(ctrl_sec_b, dtype=float))
        / SECONDS_PER_MINUTE
        + SCORE_W_KNOCKDOWN * (np.asarray(kd_a, dtype=float) - np.asarray(kd_b, dtype=float))
    )


def judge_verdict(round_margins, rng, judge_noise_sd, n_judges=3,
                  dominant_margin=None):
    """Score n_judges independent cards; return (verdict, verdict_type).

    `round_margins` is (n_sims, rounds). Each judge perturbs each round's
    margin independently and awards the round on the sign: 10-9 normally, 10-8
    when the perturbed margin exceeds `dominant_margin`, and 10-10 only on an
    exact tie - which in practice means a round where neither fighter did
    anything at all.

    The 10-8 round is what makes a level card possible. With 10-9 rounds only
    and an odd number of rounds a card can never be level, so majority
    decisions and draws would both be arithmetically impossible; see
    DOMINANT_ROUND_MARGIN.

    verdict is +1 for A, -1 for B, 0 for a draw. A draw is REPORTED, never
    broken toward either fighter: a majority of the three judges' signs that
    comes to zero is a genuine draw and 1.45% of real decisions are one.

    verdict_type distinguishes a split (2-1 with the third judge on the other
    side) from a majority (2-1 with the third judge level), matching how the
    real 20.30% / 2.49% shares are labelled.
    """
    if dominant_margin is None:
        dominant_margin = DOMINANT_ROUND_MARGIN
    margins = np.atleast_2d(np.asarray(round_margins, dtype=float))
    n_sims, n_rounds = margins.shape

    noise = rng.normal(0.0, judge_noise_sd, size=(n_judges, n_sims, n_rounds))
    perceived = margins[None, :, :] + noise
    # sign() gives +1/-1/0 directly, and 0 is the 10-10 round. The width is 1
    # for a 10-9 round and 2 for a 10-8 one.
    round_awards = np.sign(perceived) * np.where(
        np.abs(perceived) >= dominant_margin, 2, 1
    )
    cards = round_awards.sum(axis=2)          # (n_judges, n_sims)
    judge_signs = np.sign(cards).astype(np.int8)

    verdict = np.sign(judge_signs.sum(axis=0)).astype(np.int8)

    agree = (judge_signs == verdict[None, :]).sum(axis=0)
    level = (judge_signs == 0).sum(axis=0)

    verdict_type = np.full(n_sims, VERDICT_SPLIT, dtype=np.int8)
    verdict_type[agree == n_judges] = VERDICT_UNANIMOUS
    # 2-1 where the dissenting card is level, not opposed.
    verdict_type[(agree < n_judges) & (level > 0)] = VERDICT_MAJORITY
    verdict_type[verdict == 0] = VERDICT_DRAW

    return verdict, verdict_type


@dataclass(frozen=True)
class FightDistribution:
    """Everything the caller needs, including the fields that keep it honest.

    The honesty fields are not decoration. `imputed_a`/`imputed_b` name every
    rate that was assumed rather than measured - for a fighter straight out of
    career_stats that is always at least the five outcome-derived durability
    rates. `extreme_multipliers` names any composed multiplier above 5x its
    league value, which is the signature of a small-sample rate. `seed` is None
    whenever the result cannot be reproduced.

    finish_round_prob has length rounds+1 and INDEX 0 MEANS "went to decision".
    A decision in a 3-round fight must never be counted as a round-3 finish;
    they are different betting markets and conflating them is a silent error.

    duration_hist is a probability per 60-second bin over [0, rounds*300], so
    index i covers [60i, 60(i+1)) seconds.
    """

    win_prob_a: float
    win_prob_b: float
    draw_prob: float
    ko_prob_a: float
    ko_prob_b: float
    sub_prob_a: float
    sub_prob_b: float
    dec_prob_a: float
    dec_prob_b: float
    finish_round_prob: np.ndarray
    mean_duration_sec: float
    median_duration_sec: float
    duration_hist: np.ndarray
    verdict_type_prob: dict
    win_prob_se_a: float
    n_sims: int
    rounds: int
    tick_seconds: float
    seed: object
    imputed_a: frozenset
    imputed_b: frozenset
    extreme_multipliers: tuple
    mean_round_tallies: dict


def _as_generator(rng):
    """Return (Generator, seed_to_report).

    The seed is reported only when an int was passed, because that is the only
    case where this module can promise reproducibility. A caller-supplied
    Generator may or may not have been seeded and this module cannot tell, so
    it does not claim it was.
    """
    if rng is None:
        return np.random.default_rng(), None
    if isinstance(rng, np.random.Generator):
        return rng, None
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng)), int(rng)
    raise TypeError(
        f"rng must be a numpy Generator, an int seed, or None; got {type(rng).__name__}"
    )


def _extreme_multipliers(a, b):
    """Composed multipliers more than EXTREME_MULTIPLIER times league.

    Reported rather than refused. The module does not invent a clip of its own
    because shrinking noisy per-fighter rates belongs in career_stats, where
    the evidence counts live.
    """
    checks = (
        ("a_strike_vs_b_defence", (1.0 - b.str_def) / (1.0 - STR_DEF_LEAGUE)),
        ("b_strike_vs_a_defence", (1.0 - a.str_def) / (1.0 - STR_DEF_LEAGUE)),
        ("a_knockdown", float(knockdown_probability(a.kd_per_str_landed, b.kd_abs_per_str_absorbed)) / KD_PER_LANDED_LEAGUE),
        ("b_knockdown", float(knockdown_probability(b.kd_per_str_landed, a.kd_abs_per_str_absorbed)) / KD_PER_LANDED_LEAGUE),
        ("a_takedown", float(takedown_success_probability(a.td_acc, b.td_def)) / TD_ACC_LEAGUE),
        ("b_takedown", float(takedown_success_probability(b.td_acc, a.td_def)) / TD_ACC_LEAGUE),
        ("a_submission", float(submission_success_probability(a.sub_success_per_att, b.sub_loss_per_sub_faced)) / SUB_SUCCESS_LEAGUE),
        ("b_submission", float(submission_success_probability(b.sub_success_per_att, a.sub_loss_per_sub_faced)) / SUB_SUCCESS_LEAGUE),
    )
    return tuple((name, value) for name, value in checks if value > EXTREME_MULTIPLIER)


def _reach_volume_factor(attacker, defender):
    """Volume multiplier from a reach advantage, or (1.0, imputed) if unknown.

    A missing reach disables the term for that pair rather than being read as
    a zero advantage, and says so through the imputed set - a fighter whose
    reach nobody recorded is not a fighter with average reach.
    """
    if attacker.reach_cm != attacker.reach_cm or defender.reach_cm != defender.reach_cm:
        return 1.0, True
    delta = attacker.reach_cm - defender.reach_cm
    # Guarded because a large negative advantage against a low LEAGUE_SLPM
    # could otherwise drive the factor below zero, which is not a rate.
    return max(0.0, 1.0 + REACH_SLPM_PER_CM * delta / LEAGUE_SLPM), False


def _assign_finish(winner, method, fires_a, fires_b, a_first, method_code):
    """Resolve one finish category into `winner`/`method`, in place.

    Simultaneity is broken by `a_first`, a per-tick coin flip drawn from the
    same Generator, never by a fixed A-then-B order. A fixed order would hand
    fighter A a systematic advantage in every tie - a corner bias of exactly
    the kind the rest of this repo is fighting.
    """
    free = winner == 0
    a = fires_a & free
    b = fires_b & free
    both = a & b
    a_wins = (a & ~b) | (both & a_first)
    b_wins = (b & ~a) | (both & ~a_first)
    winner[a_wins] = 1
    winner[b_wins] = -1
    method[a_wins | b_wins] = method_code


# Method codes used inside the simulation and in the returned tallies.
_METHOD_DECISION = 0
_METHOD_KO = 1
_METHOD_SUB = 2


def simulate_fight(a, b, *, rounds=3, n_sims=100_000, rng=None,
                   tick_seconds=DEFAULT_TICK_SECONDS, defaults=None,
                   judge_noise_sd=None, n_judges=3, reach_effect=False):
    """Run the Monte Carlo and return the outcome distribution.

    The fight is stepped in `tick_seconds` increments through three positions -
    NEUTRAL, CONTROL_A, CONTROL_B - with position reset to NEUTRAL at the start
    of every round, which is the actual rule. Per-round tallies reset with it;
    accumulated head damage does NOT, because carrying damage across rounds is
    the entire mechanism of the accumulation TKO.

    Vectorised over simulations: the only Python loop is over ticks, and dead
    simulations are compacted out so a fight that ends in round 1 costs nothing
    for the remaining rounds.

    EVENT PRECEDENCE within one tick, applied in this fixed order after the
    per-tick fighter priority has been drawn:

        1. knockdown finish
        2. accumulation TKO
        3. submission
        4. takedown / position change
        5. stand-up

    The order is declared because without one the outcome depends on evaluation
    order and two correct-looking implementations disagree. Note the honest
    limit: the single-event hazards are tick-invariant by construction, but the
    joint handling of two events landing in the same tick is not, so running at
    1s and 5s gives slightly different METHOD splits even though the win
    probability agrees.

    THE WEAKEST PART OF THIS MODEL, stated plainly: roughly two thirds of all
    control entries come from CLINCH_CONTROL_ENTRY_PER_NEUTRAL_MIN, a flat
    league rate driven by no fighter statistic at all (0.1485 of 0.2110 entries
    per neutral minute). A dominant wrestler and a pure striker therefore spend
    the same league-average time in clinch control, which dilutes exactly the
    grappling advantage this model exists to represent. Only the takedown
    component, 0.0624 of 0.2110, responds to the fighters.
    """
    if rounds not in SUPPORTED_ROUNDS:
        raise ValueError(f"rounds must be one of {SUPPORTED_ROUNDS}, got {rounds}")
    if rounds not in CALIBRATED_ROUNDS:
        warnings.warn(
            f"{rounds}-round fights are uncalibrated: the finish-rate targets "
            f"were measured on {CALIBRATED_ROUNDS}-round bouts only",
            stacklevel=2,
        )
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")
    if n_sims < MIN_RECOMMENDED_SIMS:
        warnings.warn(
            f"n_sims={n_sims} gives a standard error above "
            f"{math.sqrt(0.25 / n_sims):.3f} on a win probability; the output "
            "will invite more precision than it has",
            stacklevel=2,
        )
    if tick_seconds <= 0:
        raise ValueError(f"tick_seconds must be positive, got {tick_seconds}")

    ticks_per_round = ROUND_SECONDS / tick_seconds
    if abs(ticks_per_round - round(ticks_per_round)) > 1e-9:
        raise ValueError(
            f"tick_seconds={tick_seconds} does not divide {ROUND_SECONDS} exactly. "
            "A short final tick would apply a whole tick's hazard to a partial "
            "interval and quietly inflate late-round finishes."
        )
    ticks_per_round = int(round(ticks_per_round))

    a, imputed_a = resolve_rates(a, defaults)
    b, imputed_b = resolve_rates(b, defaults)
    validate_rates(a)
    validate_rates(b)

    generator, seed = _as_generator(rng)
    if judge_noise_sd is None:
        judge_noise_sd = DEFAULT_JUDGE_NOISE_SD

    dt_min = tick_seconds / SECONDS_PER_MINUTE

    # --- per-position striking rates, composed once -----------------------
    reach_a = reach_b = 1.0
    if reach_effect:
        reach_a, missing_a = _reach_volume_factor(a, b)
        reach_b, missing_b = _reach_volume_factor(b, a)
        if missing_a:
            imputed_a = imputed_a | {"reach_cm"}
        if missing_b:
            imputed_b = imputed_b | {"reach_cm"}

    # Indexed by position: A is top in CONTROL_A and bottom in CONTROL_B.
    mult_a = np.array([POSITION_STRIKE_MULT["neutral"],
                       POSITION_STRIKE_MULT["top"],
                       POSITION_STRIKE_MULT["bottom"]])
    mult_b = np.array([POSITION_STRIKE_MULT["neutral"],
                       POSITION_STRIKE_MULT["bottom"],
                       POSITION_STRIKE_MULT["top"]])

    lam_land_a = strike_land_rate(a.slpm, b.str_def, mult_a) * reach_a
    lam_land_b = strike_land_rate(b.slpm, a.str_def, mult_b) * reach_b

    p_kd_a = float(knockdown_probability(a.kd_per_str_landed, b.kd_abs_per_str_absorbed))
    p_kd_b = float(knockdown_probability(b.kd_per_str_landed, a.kd_abs_per_str_absorbed))
    p_ko_given_kd_a = float(np.clip(b.ko_loss_per_kd_absorbed, 0.0, 1.0))
    p_ko_given_kd_b = float(np.clip(a.ko_loss_per_kd_absorbed, 0.0, 1.0))
    p_tko_head_a = float(np.clip(b.tko_loss_per_head_absorbed, 0.0, 1.0))
    p_tko_head_b = float(np.clip(a.tko_loss_per_head_absorbed, 0.0, 1.0))

    # Takedowns are shot from NEUTRAL only; the published per-15 rate is over
    # total time, so it is converted to per neutral minute.
    lam_td_a = a.td_att_per15 / PER_15_MINUTES * TD_ATT_NEUTRAL_FACTOR
    lam_td_b = b.td_att_per15 / PER_15_MINUTES * TD_ATT_NEUTRAL_FACTOR
    p_td_attempt_a = 1.0 - math.exp(-lam_td_a * dt_min)
    p_td_attempt_b = 1.0 - math.exp(-lam_td_b * dt_min)
    p_td_ok_a = float(takedown_success_probability(a.td_acc, b.td_def))
    p_td_ok_b = float(takedown_success_probability(b.td_acc, a.td_def))

    p_clinch = 1.0 - math.exp(-CLINCH_CONTROL_ENTRY_PER_NEUTRAL_MIN * dt_min)
    p_standup = 1.0 - math.exp(-tick_seconds / CONTROL_SPELL_SEC)

    # Submissions are attempted from a control position by BOTH fighters.
    sub_base_a = a.sub_att_per15 / PER_15_MINUTES
    sub_base_b = b.sub_att_per15 / PER_15_MINUTES
    lam_sub_a = np.array([0.0, sub_base_a * SUB_TOP_FACTOR, sub_base_a * SUB_BOTTOM_FACTOR])
    lam_sub_b = np.array([0.0, sub_base_b * SUB_BOTTOM_FACTOR, sub_base_b * SUB_TOP_FACTOR])
    p_sub_attempt_a = 1.0 - np.exp(-lam_sub_a * dt_min)
    p_sub_attempt_b = 1.0 - np.exp(-lam_sub_b * dt_min)
    p_sub_ok_a = float(submission_success_probability(a.sub_success_per_att, b.sub_loss_per_sub_faced))
    p_sub_ok_b = float(submission_success_probability(b.sub_success_per_att, a.sub_loss_per_sub_faced))

    # --- state -------------------------------------------------------------
    n = int(n_sims)
    winner = np.zeros(n, dtype=np.int8)
    method = np.full(n, _METHOD_DECISION, dtype=np.int8)
    finish_round = np.zeros(n, dtype=np.int8)      # 0 = still alive / decision
    duration = np.full(n, rounds * ROUND_SECONDS, dtype=np.float32)
    position = np.zeros(n, dtype=np.int8)

    # Per-round tallies. Narrow dtypes keep a million 5-round simulations at
    # about 60MB rather than 320MB; no per-round count can overflow them.
    sig = np.zeros((2, n, rounds), dtype=np.int16)
    ctrl = np.zeros((2, n, rounds), dtype=np.int16)
    tds = np.zeros((2, n, rounds), dtype=np.int8)
    kds = np.zeros((2, n, rounds), dtype=np.int8)

    alive = np.arange(n)

    for r in range(rounds):
        # The real rule: every round starts standing, wherever the last one
        # ended. Accumulated damage is carried; position is not.
        position[alive] = NEUTRAL

        for tick in range(ticks_per_round):
            if alive.size == 0:
                break

            pos = position[alive]
            m = alive.size

            # Control time is credited on the position held at the START of
            # the tick, before any position change this tick.
            in_a = pos == CONTROL_A
            in_b = pos == CONTROL_B
            ctrl[0, alive[in_a], r] += np.int16(tick_seconds)
            ctrl[1, alive[in_b], r] += np.int16(tick_seconds)

            # --- striking ------------------------------------------------
            landed_a = generator.poisson(lam_land_a[pos] * dt_min)
            landed_b = generator.poisson(lam_land_b[pos] * dt_min)
            sig[0, alive, r] += landed_a.astype(np.int16)
            sig[1, alive, r] += landed_b.astype(np.int16)

            kd_a = generator.binomial(landed_a, p_kd_a)
            kd_b = generator.binomial(landed_b, p_kd_b)
            kds[0, alive, r] += kd_a.astype(np.int8)
            kds[1, alive, r] += kd_b.astype(np.int8)

            head_a = generator.binomial(landed_a, HEAD_SHARE_OF_SIG)
            head_b = generator.binomial(landed_b, HEAD_SHARE_OF_SIG)

            # Whose events resolve first when two land in the same tick.
            a_first = generator.random(m) < 0.5

            w = np.zeros(m, dtype=np.int8)
            meth = np.full(m, _METHOD_DECISION, dtype=np.int8)

            # 1. knockdown finish - the conversion is applied once per
            #    knockdown, so a two-knockdown tick gets two chances.
            ko_a = kd_a > 0
            if ko_a.any():
                ko_a = ko_a & (generator.random(m) < 1.0 - (1.0 - p_ko_given_kd_a) ** kd_a)
            ko_b = kd_b > 0
            if ko_b.any():
                ko_b = ko_b & (generator.random(m) < 1.0 - (1.0 - p_ko_given_kd_b) ** kd_b)
            _assign_finish(w, meth, ko_a, ko_b, a_first, _METHOD_KO)

            # 2. accumulation TKO: an independent per-head-strike hazard, the
            #    path that produces the 30.1% of KO/TKO wins with no recorded
            #    knockdown.
            tko_a = generator.random(m) < 1.0 - (1.0 - p_tko_head_a) ** head_a
            tko_b = generator.random(m) < 1.0 - (1.0 - p_tko_head_b) ** head_b
            _assign_finish(w, meth, tko_a, tko_b, a_first, _METHOD_KO)

            # 3. submission, attempted from control by top AND bottom.
            sub_a = (generator.random(m) < p_sub_attempt_a[pos]) & (generator.random(m) < p_sub_ok_a)
            sub_b = (generator.random(m) < p_sub_attempt_b[pos]) & (generator.random(m) < p_sub_ok_b)
            _assign_finish(w, meth, sub_a, sub_b, a_first, _METHOD_SUB)

            finished = w != 0
            if finished.any():
                idx = alive[finished]
                winner[idx] = w[finished]
                method[idx] = meth[finished]
                finish_round[idx] = r + 1
                duration[idx] = r * ROUND_SECONDS + (tick + 1) * tick_seconds

            # --- position changes, for the simulations still standing -----
            live = ~finished
            if not live.any():
                alive = alive[live]
                continue

            neutral = live & (pos == NEUTRAL)
            controlled = live & (pos != NEUTRAL)

            # 4. takedown / position change, from NEUTRAL only.
            td_a = neutral & (generator.random(m) < p_td_attempt_a) & (generator.random(m) < p_td_ok_a)
            td_b = neutral & (generator.random(m) < p_td_attempt_b) & (generator.random(m) < p_td_ok_b)
            both_td = td_a & td_b
            a_takes = (td_a & ~td_b) | (both_td & a_first)
            b_takes = (td_b & ~td_a) | (both_td & ~a_first)

            position[alive[a_takes]] = CONTROL_A
            position[alive[b_takes]] = CONTROL_B
            tds[0, alive[a_takes], r] += 1
            tds[1, alive[b_takes], r] += 1

            # Clinch control that no takedown produced: the residual the
            # module docstring flags as its least trustworthy constant.
            still_neutral = neutral & ~a_takes & ~b_takes
            cl_a = still_neutral & (generator.random(m) < p_clinch)
            cl_b = still_neutral & (generator.random(m) < p_clinch)
            both_cl = cl_a & cl_b
            position[alive[(cl_a & ~cl_b) | (both_cl & a_first)]] = CONTROL_A
            position[alive[(cl_b & ~cl_a) | (both_cl & ~a_first)]] = CONTROL_B

            # 5. stand-up, an exponential hazard on the control spell.
            stand = controlled & (generator.random(m) < p_standup)
            position[alive[stand]] = NEUTRAL

            alive = alive[live]

    # --- decisions ---------------------------------------------------------
    # Applied ONLY to simulations still alive after the last tick of the last
    # round has been fully resolved, so a finish on the final tick is a finish.
    went_to_decision = winner == 0
    if went_to_decision.any():
        margins = round_score_margin(
            sig[0][went_to_decision], sig[1][went_to_decision],
            tds[0][went_to_decision], tds[1][went_to_decision],
            ctrl[0][went_to_decision], ctrl[1][went_to_decision],
            kds[0][went_to_decision], kds[1][went_to_decision],
        )
        verdict, verdict_type = judge_verdict(margins, generator, judge_noise_sd, n_judges)
        winner[went_to_decision] = verdict
        # method is already _METHOD_DECISION and finish_round already 0.
    else:
        verdict_type = np.empty(0, dtype=np.int8)

    # --- assemble ----------------------------------------------------------
    is_dec = method == _METHOD_DECISION
    is_ko = method == _METHOD_KO
    is_sub = method == _METHOD_SUB
    a_won = winner == 1
    b_won = winner == -1

    win_prob_a = float(a_won.mean())
    win_prob_b = float(b_won.mean())

    finish_round_prob = np.zeros(rounds + 1, dtype=float)
    finish_round_prob[0] = float(is_dec.mean())
    for r in range(1, rounds + 1):
        finish_round_prob[r] = float((finish_round == r).mean())

    verdict_type_prob = {
        name: float((verdict_type == code).sum()) / n
        for code, name in VERDICT_TYPE_NAMES.items()
    }

    edges = np.arange(0.0, rounds * ROUND_SECONDS + SECONDS_PER_MINUTE, SECONDS_PER_MINUTE)
    hist, _ = np.histogram(duration, bins=edges)

    mean_round_tallies = {
        "sig_a": sig[0].mean(axis=0), "sig_b": sig[1].mean(axis=0),
        "td_a": tds[0].mean(axis=0), "td_b": tds[1].mean(axis=0),
        "ctrl_sec_a": ctrl[0].mean(axis=0), "ctrl_sec_b": ctrl[1].mean(axis=0),
        "kd_a": kds[0].mean(axis=0), "kd_b": kds[1].mean(axis=0),
    }

    return FightDistribution(
        win_prob_a=win_prob_a,
        win_prob_b=win_prob_b,
        draw_prob=float((winner == 0).mean()),
        ko_prob_a=float((is_ko & a_won).mean()),
        ko_prob_b=float((is_ko & b_won).mean()),
        sub_prob_a=float((is_sub & a_won).mean()),
        sub_prob_b=float((is_sub & b_won).mean()),
        dec_prob_a=float((is_dec & a_won).mean()),
        dec_prob_b=float((is_dec & b_won).mean()),
        finish_round_prob=finish_round_prob,
        mean_duration_sec=float(duration.mean()),
        median_duration_sec=float(np.median(duration)),
        duration_hist=hist / n,
        verdict_type_prob=verdict_type_prob,
        win_prob_se_a=float(math.sqrt(max(win_prob_a * (1.0 - win_prob_a), 0.0) / n)),
        n_sims=n,
        rounds=rounds,
        tick_seconds=float(tick_seconds),
        seed=seed,
        imputed_a=frozenset(imputed_a),
        imputed_b=frozenset(imputed_b),
        extreme_multipliers=_extreme_multipliers(a, b),
        mean_round_tallies=mean_round_tallies,
    )


def calibrate_judge_noise(margins_by_round, target_split_share=OBSERVED_SPLIT_SHARE,
                          rng=0, bracket=(0.0, JUDGE_NOISE_SD_FIGHT_UPPER_BOUND),
                          n_judges=3, dominant_margin=None, tol=1e-3,
                          max_iter=60):
    """Bisect judge_noise_sd so the simulated split share hits the observed one.

    The split-decision share is the only directly observable anchor: the
    dataset labels 824 of 4,059 decisions split (20.30%), and nothing else
    about judge disagreement is recorded. The upper bracket is the
    whole-fight residual-disagreement figure, which is an UPPER bound because
    it conflates real judging idiosyncrasy with the linear scorer's own
    misspecification.

    Split share is monotonically increasing in the noise (at zero noise every
    judge scores the same card and every decision is unanimous), so bisection
    is safe.
    """
    generator, _ = _as_generator(rng)
    margins = np.atleast_2d(np.asarray(margins_by_round, dtype=float))

    def split_share(sd):
        if sd <= 0:
            return 0.0
        # A fresh child stream per evaluation keeps the objective a function of
        # sd alone; reusing one Generator would make each call depend on how
        # many came before it and the bisection could fail to converge.
        _, verdict_type = judge_verdict(margins, generator.spawn(1)[0], sd,
                                        n_judges, dominant_margin)
        return float((verdict_type == VERDICT_SPLIT).mean())

    lo, hi = bracket
    if split_share(hi) < target_split_share:
        raise ValueError(
            f"split share at the upper bracket {hi} is below the target "
            f"{target_split_share}; widen `bracket`"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if split_share(mid) < target_split_share:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def simulate_card(pairs, *, n_sims=100_000, rng=None, **kwargs):
    """One FightDistribution per bout on a card.

    `pairs` is a sequence of (fighter_a, fighter_b, rounds).

    Each bout draws from an independent child stream of the parent Generator,
    so adding, removing or reordering a bout does not change any other bout's
    numbers. With a shared stream, scratching one fight would silently move
    every prediction below it on the card.
    """
    generator, _ = _as_generator(rng)
    children = generator.spawn(len(pairs))
    return [
        simulate_fight(a, b, rounds=rounds, n_sims=n_sims, rng=child, **kwargs)
        for (a, b, rounds), child in zip(pairs, children)
    ]
