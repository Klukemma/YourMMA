"""The training feature pipeline, as one function, so live rows come out of it too.

Every accuracy figure this project quotes is measured on rows this pipeline
builds. The live card used to rebuild the same features by hand from each
fighter's last archived row - a second implementation, and it drifted: an
audit found 70 differences (stale-by-one-bout career numbers, rolling
windows and damage forced to zero, every interaction term zero on a one-row
frame, heights converted from inches twice), and replaying the last four
events live moved the winner probability a mean 15.6 points against the
training-style row for the same fight (experiments/live_parity.py).

This is the code that used to run at module level in predict_card.py
(sections 1 to 4), moved here verbatim. predict_card calls build() once on
the archive and binds everything it defines, so nothing downstream changed.
A live fight is served by build() over the archive cut the day before plus a
"pending row" for the fight (pending_rows.py), with the interaction centres
frozen at the training values, and its features are read off that row - the
same code, so the same numbers.

    built = build(raw)                          # training
    built = build(cut, pending=rows, centres=built["INTERACTION_CENTRES"])
"""

import contextlib
import io
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from name_resolution import norm_name as _norm_name


def build(raw, *, pending=None, centres=None, verbose=True):
    """Every training feature for every row of `raw` (the archive as read).

    Returns the names the pipeline defines - ufc, X, feature_cols, the
    history dicts, the helpers - as a dict.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # as predict_card always did
        if verbose:
            return _build(raw, pending, centres)
        with contextlib.redirect_stdout(io.StringIO()):
            return _build(raw, pending, centres)


def _build(ufc, pending, centres):
    # ---- moved verbatim from predict_card.py (sections 1-4) ----------------
    # The winner must be spelled exactly as a corner, or `winner == r_name`
    # mislabels the fight. See name_resolution.canonical_winners.
    from name_resolution import canonical_winners as _canonical_winners
    ufc = _canonical_winners(ufc)
    print(f"    Loaded {len(ufc):,} fights, {len(ufc.columns)} columns")

    # Parse dates
    ufc['date'] = pd.to_datetime(ufc['date'], errors='coerce')

    # Post-2001 filter
    ufc = ufc[ufc['date'] >= '2001-01-01'].copy()
    ufc = ufc.sort_values('date').reset_index(drop=True)

    # PENDING ROWS: fights not yet fought (pending_rows.py), appended after
    # the sort so every archive row keeps its training position. Each is
    # later than every archive bout, so every step that reads earlier rows
    # (the career accumulators, the calc_* loops, the rolling windows) gives
    # them exactly what a next bout would get and cannot reach back into an
    # archive row. No fighter may appear twice among them: a second pending
    # bout would read the first as history.
    N_ARCHIVE = len(ufc)
    if pending is not None and len(pending):
        _pend = pending.copy()
        _pend['date'] = pd.to_datetime(_pend['date'], errors='coerce')
        if _pend['date'].isna().any() or (_pend['date'] <= ufc['date'].max()).any():
            raise ValueError("pending rows must be dated after every archive bout")
        _ids = pd.concat([_pend['r_id'], _pend['b_id']])
        if _ids.duplicated().any():
            raise ValueError("a fighter appears twice among the pending rows")
        ufc = pd.concat([ufc, _pend], ignore_index=True)

    # ============================================================================
    # POINT-IN-TIME CAREER STATISTICS
    # ============================================================================
    # r_splm, r_str_acc, r_sapm, r_str_def, r_td_avg, r_td_def, r_td_avg_acc and
    # r_sub_avg come from the upstream fighter profile, which publishes CAREER
    # AVERAGES AS OF THE DATA PULL and joins them onto every bout that fighter
    # ever had. 80-90% of fighters with five or more bouts carry one value for
    # their whole career, so a 2014 fight was described by a striking accuracy
    # computed over 2014-2026 - including the fight being predicted.
    #
    # That inflated a walk-forward backtest to 68.6% and +16.2% over 5,943 priced
    # bets. The only honest year was 2026, at 61.5% and -3.2%, because the data
    # ends in August of it and there was almost no career left to leak.
    #
    # career_stats rebuilds every one of them from PRIOR FIGHTS ONLY, accumulating
    # across both corners in date order. Expect the measured numbers to fall.
    print("\n[1.5] BUILDING POINT-IN-TIME CAREER STATISTICS...")
    from career_stats import career_stats as _career_stats

    _careers = _career_stats(ufc)
    # The other end of the same accumulation: what is known AFTER each fighter's
    # last bout, which is what predicting their NEXT one needs. Without it every
    # cd_ column reaches the prediction path as "unknown" while the model was
    # trained on fights where it was known.
    from career_stats import final_stats as _final_stats
    _final = _final_stats(ufc)
    for _col in _careers.columns:
        ufc[_col] = _careers[_col]
    _known = ufc['r_cd_bouts'].notna() & ufc['b_cd_bouts'].notna()
    print(f"    {len(_careers.columns)} point-in-time columns")
    print(f"    both corners have prior history on {_known.mean():.1%} of fights")

    # The profile columns are percentages (46.5); the honest ones are fractions
    # (0.465). Rescaling here keeps every downstream feature on the scale its
    # thresholds and comments were written for.
    for _corner in ('r', 'b'):
        ufc[f'{_corner}_splm'] = ufc[f'{_corner}_cd_slpm']
        ufc[f'{_corner}_sapm'] = ufc[f'{_corner}_cd_sapm']
        ufc[f'{_corner}_str_acc'] = ufc[f'{_corner}_cd_str_acc'] * 100.0
        ufc[f'{_corner}_str_def'] = ufc[f'{_corner}_cd_str_def'] * 100.0
        ufc[f'{_corner}_td_avg'] = ufc[f'{_corner}_cd_td_per15']
        ufc[f'{_corner}_td_def'] = ufc[f'{_corner}_cd_td_def'] * 100.0
        ufc[f'{_corner}_td_avg_acc'] = ufc[f'{_corner}_cd_td_acc'] * 100.0
        ufc[f'{_corner}_sub_avg'] = ufc[f'{_corner}_cd_sub_per15']
        # r_wins/r_losses are the fighter's LIFETIME record as of the data pull,
        # one value repeated over every bout of their career - 98.1% and 97.8%
        # constant across careers with five or more bouts, the same signature as
        # the eight profile columns above. Alone, the win rate built from them
        # scores AUC 0.80 in 2024 and 0.58 in 2026, because by 2026 there is no
        # future career left to leak. The point-in-time record scores 0.62 in both.
        # r_draws has no honest equivalent here: a bout with no recorded winner is
        # a draw, a no-contest or an overturned result and the dataset does not
        # separate them, so it is zeroed rather than guessed. It was worth nothing
        # anyway - 99.9% constant, mean 0.22.
        ufc[f'{_corner}_wins'] = ufc[f'{_corner}_cd_wins']
        ufc[f'{_corner}_losses'] = ufc[f'{_corner}_cd_losses']
        ufc[f'{_corner}_draws'] = 0.0

    # The matchup advantages. These cross one fighter's offence against the other's
    # defence through a measured league baseline, which is the one thing a frame of
    # differences cannot say: a takedown rate means one thing against a sprawler
    # and another against a debutant.
    from matchup_inputs import matchup_features as _matchup_features

    _matchups = _matchup_features(ufc, _careers)
    for _col in _matchups.columns:
        ufc[_col] = _matchups[_col]
    print(f"    {len(_matchups.columns)} matchup advantage columns")
    print(f"    striking advantage available on "
          f"{ufc['mx_striking_known'].mean():.1%} of fights, grappling on "
          f"{ufc['mx_grappling_known'].mean():.1%}")
    print(f"    After post-2001 filter: {len(ufc):,} fights")
    print(f"    Date range: {ufc['date'].min().date()} to {ufc['date'].max().date()}")

    # Convert numeric columns (they may be strings)
    numeric_cols = [
        'r_mmr_pre', 'b_mmr_pre', 'r_mu_pre', 'b_mu_pre', 'r_sigma_pre', 'b_sigma_pre',
        'r_wins', 'r_losses', 'r_draws', 'b_wins', 'b_losses', 'b_draws',
        'r_splm', 'b_splm', 'r_str_acc', 'b_str_acc', 'r_sapm', 'b_sapm',
        'r_str_def', 'b_str_def', 'r_td_avg', 'b_td_avg', 'r_td_def', 'b_td_def',
        'r_sub_avg', 'b_sub_avg', 'r_kd', 'b_kd', 'r_td_avg_acc', 'b_td_avg_acc',
        'finish_round', 'total_rounds',
        # Cage control columns
        'r_ctrl', 'b_ctrl', 'match_time_sec',
        'r_clinch_landed', 'r_clinch_atmpted', 'b_clinch_landed', 'b_clinch_atmpted',
        'r_landed_clinch_per', 'b_landed_clinch_per',
        'r_ground_landed', 'b_ground_landed',
        'r_landed_ground_per', 'b_landed_ground_per',
    ]
    for col in numeric_cols:
        if col in ufc.columns:
            ufc[col] = pd.to_numeric(ufc[col], errors='coerce')

    # MMR check
    mmr_unique = ufc['r_mmr_pre'].nunique()
    mu_unique = ufc['r_mu_pre'].nunique()
    print(f"    MMR unique values: {mmr_unique} (should be >100 for valid MMR)")
    print(f"    Mu (skill) unique values: {mu_unique} (TrueSkill backbone)")

    # ============================================================================
    # SECTION 2: TARGET VARIABLES
    # ============================================================================
    print("\n[2] CREATING TARGETS...")

    # Win target (1 = red corner wins)
    ufc['target_win'] = (ufc['winner'] == ufc['r_name']).astype(float)
    # Mark draws/NC as NaN
    invalid_winner = ufc['winner'].isna() | (ufc['winner'] == 'Draw') | (ufc['winner'] == 'NC') | (ufc['winner'] == '')
    ufc.loc[invalid_winner, 'target_win'] = np.nan
    print(f"    Valid win targets: {ufc['target_win'].notna().sum():,}")

    # Method target
    def categorize_method(m):
        if pd.isna(m): return np.nan
        m = str(m).upper()
        if 'KO' in m or 'TKO' in m: return 'KO/TKO'
        elif 'SUB' in m: return 'Submission'
        elif 'DEC' in m or 'UNANIMOUS' in m or 'SPLIT' in m or 'MAJORITY' in m: return 'Decision'
        return 'Other'

    ufc['target_method'] = ufc['method'].apply(categorize_method)

    print(f"    Method distribution: {ufc['target_method'].value_counts().to_dict()}")

    # Round target
    ufc['target_round'] = pd.to_numeric(ufc['finish_round'], errors='coerce')
    print(f"    Round distribution: {ufc['target_round'].value_counts().sort_index().to_dict()}")

    # ============================================================================
    # SECTION 3: FEATURE ENGINEERING (MATCHING ORIGINAL + IMPROVEMENTS)
    # ============================================================================
    print("\n[3] BUILDING FEATURES...")

    # Helper functions
    def safe_num(val, default=0):
        try:
            v = float(val)
            return v if pd.notna(v) and not np.isinf(v) else default
        except:
            return default

    def is_southpaw(stance):
        if pd.isna(stance): return 0
        return 1 if 'southpaw' in str(stance).lower() else 0

    # --- AGE from DOB ---
    ufc['r_dob'] = pd.to_datetime(ufc['r_dob'], errors='coerce')
    ufc['b_dob'] = pd.to_datetime(ufc['b_dob'], errors='coerce')
    ufc['r_age'] = ((ufc['date'] - ufc['r_dob']).dt.days / 365.25).replace([np.inf, -np.inf], np.nan)
    ufc['b_age'] = ((ufc['date'] - ufc['b_dob']).dt.days / 365.25).replace([np.inf, -np.inf], np.nan)

    # --- AGE PRIME INDICATOR (fighters typically peak 28-32) ---
    def age_prime_score(age):
        """Returns score indicating how close to prime age (28-32). Peak=1, decline after 34."""
        if pd.isna(age):
            return 0.5  # neutral
        if 28 <= age <= 32:
            return 1.0
        elif age < 28:
            return 0.7 + 0.3 * max(0, (age - 22)) / 6  # ramp up from 22
        else:  # age > 32
            return max(0.3, 1.0 - 0.1 * (age - 32))  # decline after 32

    ufc['r_prime'] = ufc['r_age'].apply(age_prime_score)
    ufc['b_prime'] = ufc['b_age'].apply(age_prime_score)
    ufc['prime_diff'] = ufc['r_prime'] - ufc['b_prime']

    # --- EXPERIENCE ---
    # Prior UFC bouts, which is a count and so is genuinely 0 on a debut - that is
    # a fact about the fighter, not a filled-in unknown.
    ufc['r_exp'] = ufc['r_cd_wins'] + ufc['r_cd_losses']
    ufc['b_exp'] = ufc['b_cd_wins'] + ufc['b_cd_losses']

    # --- LAYOFF DAYS AND WIN/LOSS STREAKS ---
    print("    Calculating layoff days and streaks...")

    def calc_layoff_and_streaks(df):
        """Calculate layoff days and win/loss streaks for each fighter."""
        fighter_last_fight = {}
        fighter_streak = {}  # positive = win streak, negative = loss streak

        layoff_r = []
        layoff_b = []
        streak_r = []
        streak_b = []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            fight_date = row['date']
            winner = row.get('winner', None)

            # Determine winner
            winner_is_r = (winner == r_name) if pd.notna(winner) else None

            # Layoff for red corner
            if r_name in fighter_last_fight:
                days = (fight_date - fighter_last_fight[r_name]).days
                layoff_r.append(min(days, 1000))  # cap at ~3 years
            else:
                layoff_r.append(500)  # default for debut (neutral-ish)

            # Layoff for blue corner
            if b_name in fighter_last_fight:
                days = (fight_date - fighter_last_fight[b_name]).days
                layoff_b.append(min(days, 1000))
            else:
                layoff_b.append(500)

            # Streaks (before this fight)
            streak_r.append(fighter_streak.get(r_name, 0))
            streak_b.append(fighter_streak.get(b_name, 0))

            # Update for next iteration (after this fight)
            fighter_last_fight[r_name] = fight_date
            fighter_last_fight[b_name] = fight_date

            # Update streaks only if we know the winner
            if winner_is_r is not None:
                if winner_is_r:
                    fighter_streak[r_name] = max(0, fighter_streak.get(r_name, 0)) + 1
                    fighter_streak[b_name] = min(0, fighter_streak.get(b_name, 0)) - 1
                else:
                    fighter_streak[r_name] = min(0, fighter_streak.get(r_name, 0)) - 1
                    fighter_streak[b_name] = max(0, fighter_streak.get(b_name, 0)) + 1

        return layoff_r, layoff_b, streak_r, streak_b

    layoff_r, layoff_b, streak_r, streak_b = calc_layoff_and_streaks(ufc)
    ufc['r_layoff'] = layoff_r
    ufc['b_layoff'] = layoff_b
    ufc['r_streak'] = streak_r
    ufc['b_streak'] = streak_b

    # Layoff diff (negative = red corner has longer layoff = slight disadvantage)
    ufc['layoff_diff'] = (ufc['b_layoff'] - ufc['r_layoff']) / 100.0  # scale down
    ufc['streak_diff'] = ufc['r_streak'] - ufc['b_streak']

    # ============================================================================
    # SECTION 3.5: BAYESIAN STATE-SPACE SKILL MODEL (TrueSkill Integration)
    # ============================================================================
    print("\n[3.5] BUILDING BAYESIAN SKILL FEATURES...")

    # Rating-derived variables live in skill_features.py so they can be tested
    # without importing this file, which runs the whole pipeline. That is how a set
    # of Elo-era constants survived the switch to TrueSkill unnoticed.
    from feature_inventory import (all_specs, finish_level_names,
                                   method_rate_names)
    from feature_spec import build_all, emitted_names
    from skill_features import (
        MMR_SCALE,
        TRUESKILL_DEFAULT_MMR,
        base_probability,
        loss_penalty_scale,
        standardised_skill_gap,
        trueskill_post_fight,
        weighted_opponent_quality,
    )



    # TrueSkill parameters (standard values)
    TRUESKILL_BETA = 4.17  # Performance variance (~sigma_perf)
    TRUESKILL_DEFAULT_MU = 25.0
    TRUESKILL_DEFAULT_SIGMA = 8.33

    # --- Raw skill estimates (mu) ---
    ufc['r_mu'] = ufc['r_mu_pre'].fillna(TRUESKILL_DEFAULT_MU)
    ufc['b_mu'] = ufc['b_mu_pre'].fillna(TRUESKILL_DEFAULT_MU)
    ufc['mu_diff'] = ufc['r_mu'] - ufc['b_mu']

    # --- Skill uncertainty (sigma) ---
    ufc['r_sigma'] = ufc['r_sigma_pre'].fillna(TRUESKILL_DEFAULT_SIGMA)
    ufc['b_sigma'] = ufc['b_sigma_pre'].fillna(TRUESKILL_DEFAULT_SIGMA)
    ufc['sigma_diff'] = ufc['r_sigma'] - ufc['b_sigma']  # Lower sigma = more reliable rating

    # --- Bayesian Win Probability (TrueSkill formula) ---
    # P(r > b) = Î¦((mu_r - mu_b) / sqrt(sigma_r^2 + sigma_b^2 + 2*beta^2))
    def bayesian_win_prob(mu_r, sigma_r, mu_b, sigma_b, beta=TRUESKILL_BETA):
        """
        Calculate win probability using TrueSkill Bayesian formula.
        This properly accounts for uncertainty in both fighters' ratings.
        """
        mu_diff = mu_r - mu_b
        combined_sigma = np.sqrt(sigma_r**2 + sigma_b**2 + 2 * beta**2)
        # Avoid division by zero
        combined_sigma = np.where(combined_sigma < 0.01, 0.01, combined_sigma)
        return norm.cdf(mu_diff / combined_sigma)

    ufc['bayesian_prob'] = bayesian_win_prob(
        ufc['r_mu'].values, ufc['r_sigma'].values,
        ufc['b_mu'].values, ufc['b_sigma'].values
    )
    ufc['bayesian_prob'] = ufc['bayesian_prob'].clip(0.01, 0.99)

    # --- Skill Consistency Score (inverse of sigma, normalized) ---
    # Lower sigma = more consistent/reliable rating = higher consistency score
    def skill_consistency(sigma):
        """Convert sigma to a 0-1 consistency score. Low sigma = high consistency."""
        # Typical sigma ranges from ~2 (very consistent) to ~8+ (uncertain/new)
        # Map to 0-1 where 1 = very consistent
        return np.clip(1.0 - (sigma - 2.0) / 8.0, 0.0, 1.0)

    ufc['r_consistency'] = skill_consistency(ufc['r_sigma'])
    ufc['b_consistency'] = skill_consistency(ufc['b_sigma'])
    ufc['consistency_diff'] = ufc['r_consistency'] - ufc['b_consistency']

    # --- Conservative Skill Gap (accounting for uncertainty) ---
    # Use the "conservative" estimate: mu - k*sigma (where k=1 gives ~84% confidence bound)
    # This penalizes fighters with uncertain ratings
    CONSERVATIVE_K = 1.0

    ufc['r_skill_conservative'] = ufc['r_mu'] - CONSERVATIVE_K * ufc['r_sigma']
    ufc['b_skill_conservative'] = ufc['b_mu'] - CONSERVATIVE_K * ufc['b_sigma']
    ufc['skill_conservative_diff'] = ufc['r_skill_conservative'] - ufc['b_skill_conservative']

    # --- Combined uncertainty (lower = more confident matchup prediction) ---
    ufc['combined_uncertainty'] = np.sqrt(ufc['r_sigma']**2 + ufc['b_sigma']**2)

    print(f"    Bayesian prob range: [{ufc['bayesian_prob'].min():.3f}, {ufc['bayesian_prob'].max():.3f}]")
    print(f"    Mu diff range: [{ufc['mu_diff'].min():.2f}, {ufc['mu_diff'].max():.2f}]")
    print(f"    Combined uncertainty range: [{ufc['combined_uncertainty'].min():.2f}, {ufc['combined_uncertainty'].max():.2f}]")

    # --- MMR FEATURES ---
    # These carried two constants from an era when the rating was Elo-like and
    # centred on 1500. The rating is TrueSkill now: mmr = mu - 3*sigma, which spans
    # -25..+31 with a difference std of 7.3. Both constants were left behind.
    #
    #   MMR_SCALE = 120.0 squashed a variable of std 7.3 through a logistic scaled
    #   for hundreds of points, so base_prob only ever spanned 0.448-0.565 with a
    #   std of 0.0153 - a documented headline feature that was nearly a constant.
    #
    #   fillna(1500) never fires here (no rating is null in the dataset) but the
    #   same default is live in the prediction path, where one missing rating
    #   produces an mmr_diff of +/-1500 against a normal +/-25 and saturates
    #   base_prob to 0 or 1.
    #
    # The scale now matches the variable's own spread, so base_prob covers a usable
    # range. Choosing a scale from a feature's spread uses no outcome information,
    # so it is not leakage.
    # MMR_SCALE and TRUESKILL_DEFAULT_MMR come from skill_features.

    ufc['mmr_diff'] = (ufc['r_mmr_pre'].fillna(TRUESKILL_DEFAULT_MMR)
                       - ufc['b_mmr_pre'].fillna(TRUESKILL_DEFAULT_MMR))
    ufc['base_prob'] = base_probability(ufc['mmr_diff'])
    ufc['base_prob'] = ufc['base_prob'].clip(1e-6, 1 - 1e-6)

    # --- Additional skill variables ---
    # The model is told how far apart two fighters are but never how good the
    # fight is. A title bout between two elites and a prelim between two novices
    # can share a skill gap while behaving nothing alike.
    ufc['mu_sum'] = ufc['r_mu'] + ufc['b_mu']

    # The skill gap in units of its own uncertainty. bayesian_prob is Phi() of
    # exactly this, but a linear model cannot invert Phi, so the raw z is worth
    # exposing alongside it.
    ufc['mu_diff_z'] = standardised_skill_gap(
        ufc['r_mu'], ufc['r_sigma'], ufc['b_mu'], ufc['b_sigma'], TRUESKILL_BETA)

    # --- DIFF FEATURES (matching original) ---
    ufc['exp_diff'] = ufc['r_exp'] - ufc['b_exp']
    ufc['age_diff'] = (ufc['r_age'] - ufc['b_age']).fillna(0)

    # Striking
    ufc['off_striking_diff'] = ufc['r_splm'].fillna(0) - ufc['b_splm'].fillna(0)
    ufc['acc_diff'] = ufc['r_str_acc'].fillna(0) - ufc['b_str_acc'].fillna(0)
    ufc['def_diff'] = ufc['r_str_def'].fillna(0) - ufc['b_str_def'].fillna(0)

    # Power diff (composite feature - FIXED: removed per-fight KD which can leak outcome)
    # Using only career striking stats
    ufc['power_diff'] = (
        (ufc['r_splm'].fillna(0) - ufc['b_splm'].fillna(0)) * 1.0 +
        (ufc['r_str_def'].fillna(0) - ufc['b_str_def'].fillna(0)) * 0.5 +
        ((ufc['r_str_acc'].fillna(0) - ufc['b_str_acc'].fillna(0)) / 100.0) * 2.0 +
        (ufc['b_sapm'].fillna(0) - ufc['r_sapm'].fillna(0)) * 0.8  # opponent absorbs more = good
    )

    # Grappling
    ufc['td_off_diff'] = ufc['r_td_avg'].fillna(0) - ufc['b_td_avg'].fillna(0)
    ufc['td_def_diff'] = ufc['r_td_def'].fillna(0) - ufc['b_td_def'].fillna(0)
    ufc['sub_diff'] = ufc['r_sub_avg'].fillna(0) - ufc['b_sub_avg'].fillna(0)

    # Stance
    ufc['r_southpaw'] = ufc['r_stance'].apply(is_southpaw)
    ufc['b_southpaw'] = ufc['b_stance'].apply(is_southpaw)
    ufc['southpaw_diff'] = ufc['r_southpaw'] - ufc['b_southpaw']

    # --- STANCE MATCHUP INTERACTION ---
    # Orthodox vs Southpaw is a specific dynamic - southpaws historically have an edge.
    # stance_mismatch = 1 when fighters have different stances (creates awkward angles).
    # southpaw_advantage captures the known southpaw edge in orthodox vs southpaw matchups.
    ufc['stance_mismatch'] = (ufc['r_southpaw'] != ufc['b_southpaw']).astype(int)
    # Southpaw advantage: +1 if red is southpaw vs orthodox, -1 if blue is southpaw vs orthodox, 0 if same stance
    ufc['southpaw_advantage'] = ufc['r_southpaw'].astype(int) - ufc['b_southpaw'].astype(int)
    # Interaction: advantage is amplified when there IS a mismatch
    ufc['stance_interaction'] = ufc['southpaw_advantage'] * ufc['stance_mismatch']
    _val = ufc['stance_mismatch'].mean()*100
    print(f'      Stance mismatch rate: {_val:.1f}%')

    # Cluster (set to 0 if not available)
    if 'r_cluster5' in ufc.columns and 'b_cluster5' in ufc.columns:
        ufc['same_cluster'] = (ufc['r_cluster5'].fillna(-1) == ufc['b_cluster5'].fillna(-1)).astype(int)
    else:
        ufc['same_cluster'] = 0

    # 5-round fight indicator
    ufc['is_5rnd'] = (ufc['total_rounds'] == 5).astype(int)

    # Title fight
    ufc['is_title'] = ufc['title_fight'].fillna(0).astype(int)

    # --- ADDITIONAL FEATURES (improvements) ---
    # Win rate
    # NaN, not 0.5, for a fighter with no decided prior bout. 0.5 asserts a
    # coin-flip fighter; the paired feature's _known flag says "unknown" instead.
    ufc['r_winrate'] = ufc['r_cd_win_rate']
    ufc['b_winrate'] = ufc['b_cd_win_rate']
    ufc['winrate_diff'] = ufc['r_winrate'] - ufc['b_winrate']

    # TD accuracy diff
    ufc['td_acc_diff'] = ufc['r_td_avg_acc'].fillna(0) - ufc['b_td_avg_acc'].fillna(0)

    # Absorbed strikes diff
    ufc['sapm_diff'] = ufc['r_sapm'].fillna(0) - ufc['b_sapm'].fillna(0)

    # ============================================================================
    # SECTION 3.7: NEW FEATURES - PHYSICAL, FINISH RATES, DURABILITY
    # ============================================================================
    print("    Building physical, finish rate, and durability features...")

    # --- PHYSICAL ATTRIBUTES ---
    def parse_height(h):
        """Parse height to inches."""
        if pd.isna(h): return np.nan
        h = str(h).strip()
        if "'" in h:
            try:
                parts = h.replace('"', '').split("'")
                return int(parts[0]) * 12 + int(parts[1]) if len(parts) == 2 else np.nan
            except:
                return np.nan
        try:
            return float(h)
        except:
            return np.nan

    def parse_reach(r):
        """Parse reach to inches."""
        if pd.isna(r): return np.nan
        r = str(r).strip().replace('"', '').replace("'", '')
        try:
            return float(r)
        except:
            return np.nan

    ufc['r_height_inches'] = ufc['r_height'].apply(parse_height)
    ufc['b_height_inches'] = ufc['b_height'].apply(parse_height)
    ufc['r_reach_inches'] = ufc['r_reach'].apply(parse_reach)
    ufc['b_reach_inches'] = ufc['b_reach'].apply(parse_reach)

    ufc['height_diff'] = ufc['r_height_inches'].fillna(70) - ufc['b_height_inches'].fillna(70)
    ufc['reach_diff'] = ufc['r_reach_inches'].fillna(70) - ufc['b_reach_inches'].fillna(70)

    ufc['r_ape_index'] = ufc['r_reach_inches'].fillna(70) / ufc['r_height_inches'].replace(0, 70).fillna(70)
    ufc['b_ape_index'] = ufc['b_reach_inches'].fillna(70) / ufc['b_height_inches'].replace(0, 70).fillna(70)
    ufc['ape_index_diff'] = ufc['r_ape_index'] - ufc['b_ape_index']

    print(f"      Height diff range: [{ufc['height_diff'].min():.1f}, {ufc['height_diff'].max():.1f}] inches")
    print(f"      Reach diff range: [{ufc['reach_diff'].min():.1f}, {ufc['reach_diff'].max():.1f}] inches")

    # --- FINISH RATE FEATURES ---
    print("    Calculating historical finish rates...")

    def calc_finish_rates(df):
        fighter_stats_hist = {}
        ko_rate_r, ko_rate_b = [], []
        sub_rate_r, sub_rate_b = [], []
        dec_rate_r, dec_rate_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            winner = row.get('winner', None)
            method = str(row.get('method', '')).upper()

            r_stats = fighter_stats_hist.get(r_name, {'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0, 'total_wins': 0})
            b_stats = fighter_stats_hist.get(b_name, {'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0, 'total_wins': 0})

            r_total = max(r_stats['total_wins'], 1)
            b_total = max(b_stats['total_wins'], 1)

            ko_rate_r.append(r_stats['ko_wins'] / r_total)
            ko_rate_b.append(b_stats['ko_wins'] / b_total)
            sub_rate_r.append(r_stats['sub_wins'] / r_total)
            sub_rate_b.append(b_stats['sub_wins'] / b_total)
            dec_rate_r.append(r_stats['dec_wins'] / r_total)
            dec_rate_b.append(b_stats['dec_wins'] / b_total)

            if pd.notna(winner):
                if winner == r_name:
                    if r_name not in fighter_stats_hist:
                        fighter_stats_hist[r_name] = {'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0, 'total_wins': 0}
                    fighter_stats_hist[r_name]['total_wins'] += 1
                    if 'KO' in method or 'TKO' in method:
                        fighter_stats_hist[r_name]['ko_wins'] += 1
                    elif 'SUB' in method:
                        fighter_stats_hist[r_name]['sub_wins'] += 1
                    elif 'DEC' in method or 'UNANIMOUS' in method or 'SPLIT' in method:
                        fighter_stats_hist[r_name]['dec_wins'] += 1
                elif winner == b_name:
                    if b_name not in fighter_stats_hist:
                        fighter_stats_hist[b_name] = {'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0, 'total_wins': 0}
                    fighter_stats_hist[b_name]['total_wins'] += 1
                    if 'KO' in method or 'TKO' in method:
                        fighter_stats_hist[b_name]['ko_wins'] += 1
                    elif 'SUB' in method:
                        fighter_stats_hist[b_name]['sub_wins'] += 1
                    elif 'DEC' in method or 'UNANIMOUS' in method or 'SPLIT' in method:
                        fighter_stats_hist[b_name]['dec_wins'] += 1

            if r_name not in fighter_stats_hist:
                fighter_stats_hist[r_name] = {'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0, 'total_wins': 0}
            if b_name not in fighter_stats_hist:
                fighter_stats_hist[b_name] = {'ko_wins': 0, 'sub_wins': 0, 'dec_wins': 0, 'total_wins': 0}

        return ko_rate_r, ko_rate_b, sub_rate_r, sub_rate_b, dec_rate_r, dec_rate_b, fighter_stats_hist

    ko_rate_r, ko_rate_b, sub_rate_r, sub_rate_b, dec_rate_r, dec_rate_b, FINISH_STATS = calc_finish_rates(ufc)

    ufc['r_ko_rate'] = ko_rate_r
    ufc['b_ko_rate'] = ko_rate_b
    ufc['r_sub_rate'] = sub_rate_r
    ufc['b_sub_rate'] = sub_rate_b
    ufc['r_dec_rate'] = dec_rate_r
    ufc['b_dec_rate'] = dec_rate_b

    ufc['ko_rate_diff'] = ufc['r_ko_rate'] - ufc['b_ko_rate']
    ufc['sub_rate_diff'] = ufc['r_sub_rate'] - ufc['b_sub_rate']
    ufc['dec_rate_diff'] = ufc['r_dec_rate'] - ufc['b_dec_rate']
    ufc['finish_rate_diff'] = (ufc['r_ko_rate'] + ufc['r_sub_rate']) - (ufc['b_ko_rate'] + ufc['b_sub_rate'])

    print(f"      KO rate diff range: [{ufc['ko_rate_diff'].min():.2f}, {ufc['ko_rate_diff'].max():.2f}]")

    # --- DURABILITY FEATURES ---
    print("    Calculating durability features...")

    def calc_durability(df):
        fighter_durability = {}
        ko_losses_r, ko_losses_b = [], []
        been_finished_r, been_finished_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            winner = row.get('winner', None)
            method = str(row.get('method', '')).upper()

            r_dur = fighter_durability.get(r_name, {'ko_losses': 0, 'sub_losses': 0})
            b_dur = fighter_durability.get(b_name, {'ko_losses': 0, 'sub_losses': 0})

            ko_losses_r.append(r_dur['ko_losses'])
            ko_losses_b.append(b_dur['ko_losses'])
            been_finished_r.append(r_dur['ko_losses'] + r_dur['sub_losses'])
            been_finished_b.append(b_dur['ko_losses'] + b_dur['sub_losses'])

            if pd.notna(winner):
                loser = r_name if winner == b_name else (b_name if winner == r_name else None)
                if loser:
                    if loser not in fighter_durability:
                        fighter_durability[loser] = {'ko_losses': 0, 'sub_losses': 0}
                    if 'KO' in method or 'TKO' in method:
                        fighter_durability[loser]['ko_losses'] += 1
                    elif 'SUB' in method:
                        fighter_durability[loser]['sub_losses'] += 1

            if r_name not in fighter_durability:
                fighter_durability[r_name] = {'ko_losses': 0, 'sub_losses': 0}
            if b_name not in fighter_durability:
                fighter_durability[b_name] = {'ko_losses': 0, 'sub_losses': 0}

        return ko_losses_r, ko_losses_b, been_finished_r, been_finished_b, fighter_durability

    ko_losses_r, ko_losses_b, been_finished_r, been_finished_b, DURABILITY_STATS = calc_durability(ufc)

    ufc['r_ko_losses'] = ko_losses_r
    ufc['b_ko_losses'] = ko_losses_b
    ufc['r_been_finished'] = been_finished_r
    ufc['b_been_finished'] = been_finished_b

    ufc['r_has_been_kod'] = (ufc['r_ko_losses'] > 0).astype(int)
    ufc['b_has_been_kod'] = (ufc['b_ko_losses'] > 0).astype(int)
    ufc['ko_vulnerability_diff'] = ufc['r_has_been_kod'] - ufc['b_has_been_kod']
    ufc['been_finished_diff'] = ufc['r_been_finished'] - ufc['b_been_finished']

    ufc['r_absorption_eff'] = ufc['r_str_def'].fillna(50) / (ufc['r_sapm'].fillna(3) + 0.1)
    ufc['b_absorption_eff'] = ufc['b_str_def'].fillna(50) / (ufc['b_sapm'].fillna(3) + 0.1)
    ufc['absorption_eff_diff'] = ufc['r_absorption_eff'] - ufc['b_absorption_eff']

    print(f"      Absorption eff diff range: [{ufc['absorption_eff_diff'].min():.2f}, {ufc['absorption_eff_diff'].max():.2f}]")

    # --- FOOTWORK / EVASION EFFICIENCY ---
    # Measures how efficiently a fighter lands strikes relative to what they absorb.
    # High value = good footwork/movement (lands a lot, absorbs little).
    # Low value = flat-footed / hittable (e.g. Derrick Lewis: high SAPM, moderate SLpM).
    # The rates are point-in-time now, so a debutant genuinely has none. Filling
    # with 3 invented an average fighter; the ratio is NaN instead and the _known
    # flag on the paired feature tells the model which it is.
    ufc['r_footwork_proxy'] = ufc['r_splm'] / ufc['r_sapm'].clip(lower=0.5)
    ufc['b_footwork_proxy'] = ufc['b_splm'] / ufc['b_sapm'].clip(lower=0.5)
    ufc['footwork_diff'] = ufc['r_footwork_proxy'] - ufc['b_footwork_proxy']
    _lo, _hi = ufc['footwork_diff'].min(), ufc['footwork_diff'].max()
    print(f'      Footwork diff range: [{_lo:.2f}, {_hi:.2f}]')

    # --- DATA SPARSITY (FIGHT COUNT) ---
    # Fighters with very few UFC fights have unreliable stats.
    # log1p smooths the scale: 0 fights=0, 3 fights=1.4, 10 fights=2.4, 30 fights=3.4
    ufc['r_fight_count'] = ufc.groupby('r_name').cumcount()
    ufc['b_fight_count'] = ufc.groupby('b_name').cumcount()
    ufc['r_data_reliability'] = np.log1p(ufc['r_fight_count'].clip(upper=30))
    ufc['b_data_reliability'] = np.log1p(ufc['b_fight_count'].clip(upper=30))
    ufc['data_sparsity_diff'] = ufc['r_data_reliability'] - ufc['b_data_reliability']
    _lo, _hi = ufc['data_sparsity_diff'].min(), ufc['data_sparsity_diff'].max()
    print(f'      Data sparsity diff range: [{_lo:.2f}, {_hi:.2f}]')

    # --- DAMAGE ACCUMULATION (CAREER WEAR) ---
    # Cumulative significant strikes absorbed over UFC career.
    # High career absorption = chin deterioration risk, even if not yet KOd.
    # Uses SAPM * estimated fight minutes as proxy for total damage taken.
    ufc['r_career_fights'] = ufc.groupby('r_name').cumcount()
    ufc['b_career_fights'] = ufc.groupby('b_name').cumcount()
    # Estimated career damage: SAPM * avg fight minutes (assume ~12 min avg) * fight count
    ufc['r_career_damage'] = ufc['r_sapm'].fillna(3) * 12.0 * ufc['r_career_fights'].clip(lower=1)
    ufc['b_career_damage'] = ufc['b_sapm'].fillna(3) * 12.0 * ufc['b_career_fights'].clip(lower=1)
    # Log-scale to prevent huge fighters from dominating (30-fight vet vs 3-fight newcomer)
    ufc['r_damage_log'] = np.log1p(ufc['r_career_damage'])
    ufc['b_damage_log'] = np.log1p(ufc['b_career_damage'])
    ufc['career_damage_diff'] = ufc['r_damage_log'] - ufc['b_damage_log']
    _lo, _hi = ufc['career_damage_diff'].min(), ufc['career_damage_diff'].max()
    print(f'      Career damage diff range: [{_lo:.2f}, {_hi:.2f}]')

    # --- OPPONENT QUALITY (STRENGTH OF SCHEDULE) ---

    print("    Calculating opponent quality (strength of schedule)...")

    def calc_opponent_quality(df):
        """Average quality of opponents faced, with recency weighting.

        Two things were wrong here and both came from the Elo era.

        It averaged past opponents' *mmr* (mu - 3*sigma). That rating is dominated
        by uncertainty rather than skill - the penalty term's spread (4.88) is
        larger than the skill difference it adjusts (4.47) - so strength of
        schedule was being measured with the weakest available ruler. It averages
        mu now, which is the skill estimate itself.

        And a fighter with no recorded opponents was given the constant 1500, on a
        scale where real values run about -3 to 31. That fires on 24.5% of fights,
        turning opp_quality_diff into a debut flag multiplied by roughly 1486: its
        std is 628.8 overall against 6.30 among fights where both fighters have
        history. Whatever genuine signal strength of schedule carries was drowned.

        An unknown schedule is now NaN, which the feature matrix turns into 0 - the
        neutral value for a difference - alongside an explicit flag per corner so
        the model can tell "no history" from "equally matched schedules" instead of
        being handed a number that means neither.

        Last 4 fights are weighted 2x, which is unchanged.
        """
        fighter_history = {}  # fighter -> list of (date, opponent_mu)
        opponent_quality_r, opponent_quality_b = [], []
        has_history_r, has_history_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            r_mu = row.get('r_mu', TRUESKILL_DEFAULT_MU)
            b_mu = row.get('b_mu', TRUESKILL_DEFAULT_MU)
            fight_date = row.get('date')

            r_hist = fighter_history.get(r_name, [])
            b_hist = fighter_history.get(b_name, [])
            opponent_quality_r.append(
                weighted_opponent_quality([mu for _, mu in r_hist]))
            opponent_quality_b.append(
                weighted_opponent_quality([mu for _, mu in b_hist]))
            has_history_r.append(1.0 if r_hist else 0.0)
            has_history_b.append(1.0 if b_hist else 0.0)

            # Update after recording, so a fight never sees its own opponent.
            fighter_history.setdefault(r_name, []).append((fight_date, b_mu))
            fighter_history.setdefault(b_name, []).append((fight_date, r_mu))

        return (opponent_quality_r, opponent_quality_b,
                has_history_r, has_history_b, fighter_history)

    (opp_quality_r, opp_quality_b, opp_hist_r, opp_hist_b,
     OPP_QUALITY_HISTORY) = calc_opponent_quality(ufc)

    ufc['r_opp_quality'] = opp_quality_r
    ufc['b_opp_quality'] = opp_quality_b
    ufc['r_has_opp_history'] = opp_hist_r
    ufc['b_has_opp_history'] = opp_hist_b
    ufc['opp_quality_diff'] = ufc['r_opp_quality'] - ufc['b_opp_quality']
    # Both schedules known: the only rows where opp_quality_diff means anything.
    ufc['opp_history_known'] = ufc['r_has_opp_history'] * ufc['b_has_opp_history']

    _known = ufc['opp_quality_diff'].notna()
    print(f"      Opponent quality known for {_known.mean():.1%} of fights")
    print(f"      Opponent quality diff range (known rows): "
          f"[{ufc.loc[_known, 'opp_quality_diff'].min():.2f}, "
          f"{ufc.loc[_known, 'opp_quality_diff'].max():.2f}]  "
          f"std {ufc.loc[_known, 'opp_quality_diff'].std():.2f}")

    # --- WEIGHT CLASS FEATURES ---
    print("    Adding weight class features...")

    def parse_weight_class(division):
        """Parse division string to extract weight class info."""
        if pd.isna(division):
            return 'unknown', False, False

        div_lower = str(division).lower()

        # Check if women's division
        is_womens = 'women' in div_lower or 'female' in div_lower

        # Check if heavyweight (men's HW typically has more KOs)
        is_heavyweight = 'heavyweight' in div_lower and 'light heavyweight' not in div_lower

        # Normalize weight class name
        if 'strawweight' in div_lower:
            wc = 'strawweight'
        elif 'flyweight' in div_lower:
            wc = 'flyweight'
        elif 'bantamweight' in div_lower:
            wc = 'bantamweight'
        elif 'featherweight' in div_lower:
            wc = 'featherweight'
        elif 'lightweight' in div_lower:
            wc = 'lightweight'
        elif 'welterweight' in div_lower:
            wc = 'welterweight'
        elif 'middleweight' in div_lower:
            wc = 'middleweight'
        elif 'light heavyweight' in div_lower:
            wc = 'light_heavyweight'
        elif 'heavyweight' in div_lower:
            wc = 'heavyweight'
        else:
            wc = 'other'

        return wc, is_womens, is_heavyweight

    # Parse divisions
    weight_classes = []
    is_womens_list = []
    is_heavyweight_list = []

    for div in ufc['division']:
        wc, is_w, is_hw = parse_weight_class(div)
        weight_classes.append(wc)
        is_womens_list.append(int(is_w))
        is_heavyweight_list.append(int(is_hw))

    ufc['weight_class'] = weight_classes
    ufc['is_womens'] = is_womens_list
    ufc['is_heavyweight'] = is_heavyweight_list

    print(f"      Heavyweight fights: {sum(is_heavyweight_list)}")
    print(f"      Women's fights: {sum(is_womens_list)}")

    # --- TIME-AWARE FEATURES (L3/L5 rolling + momentum) ---
    print("    Building time-aware features (L3/L5 rolling averages)...")

    # Create fighter-centric long format for rolling calculations
    def build_fighter_history(df):
        """Build per-fighter history with rolling stats."""
        # Red corner rows
        red = df[['date', 'r_name', 'target_win', 'r_splm', 'r_str_acc', 'r_td_avg']].copy()
        red.columns = ['date', 'fighter', 'won', 'splm', 'str_acc', 'td_avg']
        red['won'] = red['won'].fillna(0.5)  # unknown = 0.5

        # Blue corner rows (invert win)
        blue = df[['date', 'b_name', 'target_win', 'b_splm', 'b_str_acc', 'b_td_avg']].copy()
        blue.columns = ['date', 'fighter', 'won', 'splm', 'str_acc', 'td_avg']
        blue['won'] = 1 - blue['won'].fillna(0.5)

        # Combine
        history = pd.concat([red, blue], ignore_index=True)
        history = history.sort_values(['fighter', 'date']).reset_index(drop=True)

        # Rolling stats per fighter (shifted to avoid leakage)
        for stat in ['won', 'splm', 'str_acc', 'td_avg']:
            history[f'{stat}_shifted'] = history.groupby('fighter')[stat].shift(1)
            history[f'{stat}_L3'] = history.groupby('fighter')[f'{stat}_shifted'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            history[f'{stat}_L5'] = history.groupby('fighter')[f'{stat}_shifted'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

        # Momentum (L3 - L5)
        history['momentum'] = history['won_L3'].fillna(0.5) - history['won_L5'].fillna(0.5)

        return history

    # Build history
    fighter_history = build_fighter_history(ufc)

    # Create lookup for latest stats per fighter per date
    def get_fighter_rolling_stats(fighter_name, fight_date, history_df):
        """Get rolling stats for a fighter before a specific date."""
        mask = (history_df['fighter'] == fighter_name) & (history_df['date'] < fight_date)
        matches = history_df[mask].sort_values('date', ascending=False)
        if len(matches) > 0:
            return matches.iloc[0]
        return None

    # For efficiency, merge rolling stats back to main dataframe
    # This is a simplified approach - for each row, we look up the fighter's pre-fight rolling stats
    print("    Merging rolling stats (this may take a moment)...")

    # Create a unique fight key for merging
    history_lookup = fighter_history.groupby(['fighter', 'date']).first().reset_index()

    # Merge for red corner
    ufc = ufc.merge(
        history_lookup[['fighter', 'date', 'won_L3', 'won_L5', 'momentum', 'splm_L3', 'str_acc_L3', 'td_avg_L3']],
        left_on=['r_name', 'date'], right_on=['fighter', 'date'], how='left', suffixes=('', '_r')
    )
    ufc = ufc.rename(columns={
        'won_L3': 'r_won_L3', 'won_L5': 'r_won_L5', 'momentum': 'r_momentum',
        'splm_L3': 'r_splm_L3', 'str_acc_L3': 'r_str_acc_L3', 'td_avg_L3': 'r_td_avg_L3'
    })
    ufc = ufc.drop(columns=['fighter'], errors='ignore')

    # Merge for blue corner
    ufc = ufc.merge(
        history_lookup[['fighter', 'date', 'won_L3', 'won_L5', 'momentum', 'splm_L3', 'str_acc_L3', 'td_avg_L3']],
        left_on=['b_name', 'date'], right_on=['fighter', 'date'], how='left', suffixes=('', '_b')
    )
    ufc = ufc.rename(columns={
        'won_L3': 'b_won_L3', 'won_L5': 'b_won_L5', 'momentum': 'b_momentum',
        'splm_L3': 'b_splm_L3', 'str_acc_L3': 'b_str_acc_L3', 'td_avg_L3': 'b_td_avg_L3'
    })
    ufc = ufc.drop(columns=['fighter'], errors='ignore')

    # Time-aware diff features
    ufc['recent_form_diff'] = ufc['r_won_L3'].fillna(0.5) - ufc['b_won_L3'].fillna(0.5)
    ufc['momentum_diff'] = ufc['r_momentum'].fillna(0) - ufc['b_momentum'].fillna(0)
    ufc['splm_L3_diff'] = ufc['r_splm_L3'].fillna(0) - ufc['b_splm_L3'].fillna(0)
    ufc['str_acc_L3_diff'] = ufc['r_str_acc_L3'].fillna(0) - ufc['b_str_acc_L3'].fillna(0)
    ufc['td_avg_L3_diff'] = ufc['r_td_avg_L3'].fillna(0) - ufc['b_td_avg_L3'].fillna(0)
    # --- FIGHTER TRAJECTORY / DECLINE DETECTION ---
    # Compare recent stats (L3) to career averages to detect improvement or decline.
    # Positive = improving (recent > career), Negative = declining (recent < career).
    # This catches fighters like Derrick Lewis whose recent output trails their career norms.
    ufc['r_striking_trajectory'] = ufc['r_splm_L3'].fillna(ufc['r_splm'].fillna(0)) - ufc['r_splm'].fillna(0)
    ufc['b_striking_trajectory'] = ufc['b_splm_L3'].fillna(ufc['b_splm'].fillna(0)) - ufc['b_splm'].fillna(0)
    ufc['striking_trajectory_diff'] = ufc['r_striking_trajectory'] - ufc['b_striking_trajectory']

    ufc['r_accuracy_trajectory'] = ufc['r_str_acc_L3'].fillna(ufc['r_str_acc'].fillna(0)) - ufc['r_str_acc'].fillna(0)
    ufc['b_accuracy_trajectory'] = ufc['b_str_acc_L3'].fillna(ufc['b_str_acc'].fillna(0)) - ufc['b_str_acc'].fillna(0)
    ufc['accuracy_trajectory_diff'] = ufc['r_accuracy_trajectory'] - ufc['b_accuracy_trajectory']

    # Combined trajectory: normalized composite of striking output + accuracy trends
    ufc['r_combined_trajectory'] = (ufc['r_striking_trajectory'] / 3.0) + (ufc['r_accuracy_trajectory'] / 20.0)
    ufc['b_combined_trajectory'] = (ufc['b_striking_trajectory'] / 3.0) + (ufc['b_accuracy_trajectory'] / 20.0)
    ufc['trajectory_diff'] = ufc['r_combined_trajectory'] - ufc['b_combined_trajectory']
    _lo, _hi = ufc['trajectory_diff'].min(), ufc['trajectory_diff'].max()
    print(f'      Trajectory diff range: [{_lo:.2f}, {_hi:.2f}]')

    # ============================================================================
    # SECTION 3.9: CAGE CONTROL FEATURES
    # ============================================================================
    print("    Building cage control features (grind score, rolling history)...")

    # --- Compute per-fight grind score ---
    # Estimate fight duration: use match_time_sec if available, else estimate from finish_round
    if 'match_time_sec' in ufc.columns:
        ufc['_fight_duration'] = pd.to_numeric(ufc['match_time_sec'], errors='coerce')
    else:
        ufc['_fight_duration'] = np.nan

    # Fallback: estimate from finish_round (5 min per round = 300 sec)
    ufc['_fight_duration'] = ufc['_fight_duration'].fillna(
        ufc['finish_round'].fillna(ufc['total_rounds'].fillna(3)) * 300
    )
    ufc['_fight_duration'] = ufc['_fight_duration'].clip(lower=60)  # minimum 1 minute

    # Compute grind score for red corner
    r_ctrl_rate = ufc['r_ctrl'].fillna(0) / ufc['_fight_duration']
    r_clinch_activity = ufc['r_clinch_atmpted'].fillna(0).clip(upper=30) / 30.0
    r_clinch_pct = ufc['r_landed_clinch_per'].fillna(0).clip(upper=40) / 40.0

    ufc['r_grind_score'] = (
        0.50 * r_ctrl_rate.clip(upper=1.0) +
        0.25 * r_clinch_activity +
        0.25 * r_clinch_pct
    ).fillna(0)

    # Compute grind score for blue corner
    b_ctrl_rate = ufc['b_ctrl'].fillna(0) / ufc['_fight_duration']
    b_clinch_activity = ufc['b_clinch_atmpted'].fillna(0).clip(upper=30) / 30.0
    b_clinch_pct = ufc['b_landed_clinch_per'].fillna(0).clip(upper=40) / 40.0

    ufc['b_grind_score'] = (
        0.50 * b_ctrl_rate.clip(upper=1.0) +
        0.25 * b_clinch_activity +
        0.25 * b_clinch_pct
    ).fillna(0)

    # --- Build cage control rolling history per fighter ---
    def build_cage_control_history(df):
        """Build per-fighter cage control history with rolling stats."""
        # Red corner rows
        red = df[['date', 'r_name', 'r_grind_score', 'r_ctrl', '_fight_duration', 'r_clinch_atmpted']].copy()
        red.columns = ['date', 'fighter', 'grind_score', 'ctrl_time', 'fight_duration', 'clinch_atmpted']

        # Blue corner rows
        blue = df[['date', 'b_name', 'b_grind_score', 'b_ctrl', '_fight_duration', 'b_clinch_atmpted']].copy()
        blue.columns = ['date', 'fighter', 'grind_score', 'ctrl_time', 'fight_duration', 'clinch_atmpted']

        # Combine and sort
        history = pd.concat([red, blue], ignore_index=True)
        history = history.sort_values(['fighter', 'date']).reset_index(drop=True)

        # Compute ctrl_rate per fight
        history['ctrl_rate'] = (history['ctrl_time'].fillna(0) / history['fight_duration'].clip(lower=60)).clip(upper=1.0)

        # Shift to prevent leakage (current fight stats not visible for prediction)
        for stat in ['grind_score', 'ctrl_rate', 'clinch_atmpted']:
            history[f'{stat}_shifted'] = history.groupby('fighter')[stat].shift(1)

        # Rolling stats: EWM (halflife=5) for long-term tendency
        history['ctrl_rate_ewm'] = history.groupby('fighter')['ctrl_rate_shifted'].transform(
            lambda x: x.ewm(halflife=5, min_periods=1).mean()
        )
        history['grind_score_ewm'] = history.groupby('fighter')['grind_score_shifted'].transform(
            lambda x: x.ewm(halflife=5, min_periods=1).mean()
        )
        history['clinch_activity_ewm'] = history.groupby('fighter')['clinch_atmpted_shifted'].transform(
            lambda x: x.ewm(halflife=5, min_periods=1).mean()
        )

        # L3 rolling for recent behavior
        history['grind_score_L3'] = history.groupby('fighter')['grind_score_shifted'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        # Grind rate (proportion of past fights with grind_score >= 0.35)
        history['is_grind_fight'] = (history['grind_score_shifted'] >= 0.35).astype(float)
        history['grind_rate'] = history.groupby('fighter')['is_grind_fight'].transform(
            lambda x: x.expanding(min_periods=1).mean()
        )

        # Fight count for confidence weighting
        history['cage_ctrl_fights'] = history.groupby('fighter')['grind_score_shifted'].transform(
            lambda x: x.expanding().count()
        )

        return history

    cage_ctrl_history = build_cage_control_history(ufc)

    # Merge back to main dataframe
    print("    Merging cage control stats...")
    cc_lookup = cage_ctrl_history.groupby(['fighter', 'date']).first().reset_index()
    cc_cols = ['fighter', 'date', 'ctrl_rate_ewm', 'grind_score_ewm', 'clinch_activity_ewm',
               'grind_score_L3', 'grind_rate', 'cage_ctrl_fights']

    # Merge for red corner
    ufc = ufc.merge(
        cc_lookup[cc_cols],
        left_on=['r_name', 'date'], right_on=['fighter', 'date'], how='left', suffixes=('', '_ccr')
    )
    ufc = ufc.rename(columns={
        'ctrl_rate_ewm': 'r_ctrl_rate_ewm', 'grind_score_ewm': 'r_grind_score_ewm',
        'clinch_activity_ewm': 'r_clinch_activity_ewm', 'grind_score_L3': 'r_grind_score_L3',
        'grind_rate': 'r_grind_rate', 'cage_ctrl_fights': 'r_cage_ctrl_fights'
    })
    ufc = ufc.drop(columns=['fighter'], errors='ignore')

    # Merge for blue corner
    ufc = ufc.merge(
        cc_lookup[cc_cols],
        left_on=['b_name', 'date'], right_on=['fighter', 'date'], how='left', suffixes=('', '_ccb')
    )
    ufc = ufc.rename(columns={
        'ctrl_rate_ewm': 'b_ctrl_rate_ewm', 'grind_score_ewm': 'b_grind_score_ewm',
        'clinch_activity_ewm': 'b_clinch_activity_ewm', 'grind_score_L3': 'b_grind_score_L3',
        'grind_rate': 'b_grind_rate', 'cage_ctrl_fights': 'b_cage_ctrl_fights'
    })
    ufc = ufc.drop(columns=['fighter'], errors='ignore')

    # --- Cage control diff features (for training) ---
    ufc['cage_control_cap_diff'] = ufc['r_ctrl_rate_ewm'].fillna(0) - ufc['b_ctrl_rate_ewm'].fillna(0)
    ufc['clinch_activity_diff'] = ufc['r_clinch_activity_ewm'].fillna(0) - ufc['b_clinch_activity_ewm'].fillna(0)
    ufc['grind_tendency_diff'] = ufc['r_grind_rate'].fillna(0) - ufc['b_grind_rate'].fillna(0)

    print(f"    Cage control features built. Grind fights (>=0.35): R={int((ufc['r_grind_score']>=0.35).sum())}, B={int((ufc['b_grind_score']>=0.35).sum())}")

    # ============================================================================
    # SECTION 3.95: ADVANCED FEATURES (Cardio, Archetypes, Pace, Weight Movement, etc.)
    # ============================================================================
    print("\n[3.95] BUILDING ADVANCED FEATURES...")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 1: CARDIO / GAS TANK MODELING
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Approximation: Compare per-minute output in short vs long fights.
    # Fighters whose output drops significantly in longer fights have poor cardio.
    print("    Building cardio/gas tank features...")

    def calc_cardio_features(df):
        """
        Compute cardio proxy: compare per-minute striking output in
        short fights (finished R1-R2) vs long fights (went to R3+/decision).
        A big drop = poor cardio. Consistent output = good gas tank.
        """
        fighter_fight_outputs = {}  # fighter -> list of (rounds_fought, output_per_min)
        cardio_r, cardio_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']

            # Get fight duration info
            finish_rd = row.get('finish_round', 3)
            if pd.isna(finish_rd):
                finish_rd = row.get('total_rounds', 3)
            finish_rd = max(1, int(finish_rd))

            fight_time_sec = row.get('match_time_sec', finish_rd * 300)
            if pd.isna(fight_time_sec) or fight_time_sec <= 0:
                fight_time_sec = finish_rd * 300
            fight_time_min = max(fight_time_sec / 60.0, 0.5)

            # Per-minute output for this fight
            r_sig = pd.to_numeric(row.get('r_sig_str_landed', 0), errors='coerce')
            b_sig = pd.to_numeric(row.get('b_sig_str_landed', 0), errors='coerce')
            r_sig = r_sig if pd.notna(r_sig) else 0
            b_sig = b_sig if pd.notna(b_sig) else 0

            r_output_pm = r_sig / fight_time_min
            b_output_pm = b_sig / fight_time_min

            # Compute cardio score for each fighter BEFORE this fight
            r_history = fighter_fight_outputs.get(r_name, [])
            b_history = fighter_fight_outputs.get(b_name, [])

            def compute_cardio_score(history):
                if len(history) < 2:
                    return 0.5  # neutral (not enough data)
                short_fights = [opm for rds, opm in history if rds <= 2]
                long_fights = [opm for rds, opm in history if rds >= 3]
                if len(short_fights) == 0 or len(long_fights) == 0:
                    # All short or all long - use consistency as proxy
                    outputs = [opm for _, opm in history]
                    if len(outputs) < 2:
                        return 0.5
                    mean_out = np.mean(outputs)
                    if mean_out == 0:
                        return 0.5
                    cv = np.std(outputs) / (mean_out + 0.01)  # coefficient of variation
                    return np.clip(1.0 - cv, 0.2, 0.9)  # low CV = consistent = good cardio

                avg_short = np.mean(short_fights)
                avg_long = np.mean(long_fights)
                if avg_short == 0:
                    return 0.5
                # Ratio of long-fight output to short-fight output
                # 1.0 = maintains pace perfectly, <1 = fades
                ratio = avg_long / (avg_short + 0.01)
                return np.clip(ratio, 0.2, 1.2)

            cardio_r.append(compute_cardio_score(r_history))
            cardio_b.append(compute_cardio_score(b_history))

            # Update history for next iteration
            if r_name not in fighter_fight_outputs:
                fighter_fight_outputs[r_name] = []
            fighter_fight_outputs[r_name].append((finish_rd, r_output_pm))

            if b_name not in fighter_fight_outputs:
                fighter_fight_outputs[b_name] = []
            fighter_fight_outputs[b_name].append((finish_rd, b_output_pm))

        return cardio_r, cardio_b, fighter_fight_outputs

    cardio_r, cardio_b, CARDIO_HISTORY = calc_cardio_features(ufc)
    ufc['r_cardio'] = cardio_r
    ufc['b_cardio'] = cardio_b
    ufc['cardio_diff'] = ufc['r_cardio'] - ufc['b_cardio']
    print(f"      Cardio diff range: [{ufc['cardio_diff'].min():.3f}, {ufc['cardio_diff'].max():.3f}]")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 3: STYLISTIC ARCHETYPE CLASSIFICATION
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Classify fighters into archetypes and model archetype-vs-archetype matchups.
    print("    Building stylistic archetype features...")

    def classify_archetype(splm, td_avg, sub_avg, str_def, str_acc,
                           dist_pct=0.7, clinch_pct=0.15, ground_pct=0.15):
        """
        Classify fighter into archetype based on career stats.
        Returns: (archetype_id, archetype_name)
        Archetypes:
          0 = Pressure Boxer (high output, forward pressure)
          1 = Counter-Striker (high accuracy, high defense, lower output)
          2 = Wrestler (high TD avg, moderate striking)
          3 = Submission Artist (high sub avg, moderate TD)
          4 = Point Fighter (balanced, high defense, goes to decision)
        """
        splm = float(splm) if pd.notna(splm) else 3.0
        td_avg = float(td_avg) if pd.notna(td_avg) else 1.5
        sub_avg = float(sub_avg) if pd.notna(sub_avg) else 0.5
        str_def = float(str_def) if pd.notna(str_def) else 55
        str_acc = float(str_acc) if pd.notna(str_acc) else 45

        # Scoring system for each archetype
        scores = [0.0] * 5

        # Pressure Boxer: high output, lower accuracy tolerance
        scores[0] = (splm / 5.0) * 0.6 + (1 - str_def/100) * 0.2 + (1 - td_avg/5) * 0.2

        # Counter-Striker: high accuracy + defense, lower output
        scores[1] = (str_acc / 60.0) * 0.4 + (str_def / 70.0) * 0.4 + (1 - splm/6) * 0.2

        # Wrestler: high TD avg
        scores[2] = (td_avg / 4.0) * 0.6 + (1 - splm/6) * 0.2 + (str_def/70) * 0.2

        # Submission Artist: high sub + moderate TD
        scores[3] = (sub_avg / 2.0) * 0.5 + (td_avg / 4.0) * 0.3 + (1 - splm/6) * 0.2

        # Point Fighter: balanced stats, high defense
        scores[4] = (str_def / 70.0) * 0.35 + (str_acc / 60.0) * 0.35 + (1 - td_avg/4) * 0.15 + (1 - sub_avg/2) * 0.15

        archetype_id = int(np.argmax(scores))
        archetype_names = ['pressure_boxer', 'counter_striker', 'wrestler', 'sub_artist', 'point_fighter']
        return archetype_id, archetype_names[archetype_id]

    # Classify each fighter per fight (using their career stats AT THAT POINT)
    r_archetypes = []
    b_archetypes = []

    for idx, row in ufc.iterrows():
        r_arch_id, _ = classify_archetype(
            row.get('r_splm', 3), row.get('r_td_avg', 1.5), row.get('r_sub_avg', 0.5),
            row.get('r_str_def', 55), row.get('r_str_acc', 45)
        )
        b_arch_id, _ = classify_archetype(
            row.get('b_splm', 3), row.get('b_td_avg', 1.5), row.get('b_sub_avg', 0.5),
            row.get('b_str_def', 55), row.get('b_str_acc', 45)
        )
        r_archetypes.append(r_arch_id)
        b_archetypes.append(b_arch_id)

    ufc['r_archetype'] = r_archetypes
    ufc['b_archetype'] = b_archetypes

    # Create archetype matchup features
    # Encode as interaction: some matchups favor one style over another
    # Key insight: pressure beats counter, wrestler beats pressure, counter beats wrestler
    ARCHETYPE_MATCHUP_MATRIX = np.array([
        # vs: press  counter  wrestl  sub    point
        [0.50, 0.55, 0.42, 0.48, 0.52],  # pressure boxer
        [0.45, 0.50, 0.53, 0.50, 0.48],  # counter striker
        [0.58, 0.47, 0.50, 0.45, 0.55],  # wrestler
        [0.52, 0.50, 0.55, 0.50, 0.48],  # sub artist
        [0.48, 0.52, 0.45, 0.52, 0.50],  # point fighter
    ])

    ufc['archetype_matchup'] = [
        ARCHETYPE_MATCHUP_MATRIX[r, b] - 0.5  # center at 0
        for r, b in zip(ufc['r_archetype'], ufc['b_archetype'])
    ]

    # Also encode if it's a stylistic clash (different archetypes = more unpredictable)
    ufc['archetype_clash'] = (ufc['r_archetype'] != ufc['b_archetype']).astype(int)

    print(f"      Archetype distribution (R): {pd.Series(r_archetypes).value_counts().to_dict()}")
    print(f"      Archetype matchup advantage range: [{ufc['archetype_matchup'].min():.3f}, {ufc['archetype_matchup'].max():.3f}]")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 4: PACE/OUTPUT PREDICTION
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Combined pace of both fighters predicts fight tempo.
    # High-pace fights finish more often. Low-pace fights go to decision.
    print("    Building pace/output features...")

    ufc['combined_pace'] = ufc['r_splm'].fillna(3) + ufc['b_splm'].fillna(3)
    ufc['pace_diff'] = ufc['r_splm'].fillna(3) - ufc['b_splm'].fillna(3)

    # Pace category: helps method prediction
    # High pace (>8 combined SLpM) -> more finishes
    # Low pace (<5 combined SLpM) -> more decisions
    ufc['high_pace'] = (ufc['combined_pace'] > 7.5).astype(int)

    print(f"      Combined pace range: [{ufc['combined_pace'].min():.1f}, {ufc['combined_pace'].max():.1f}]")
    print(f"      High-pace fights: {ufc['high_pace'].sum()} ({ufc['high_pace'].mean()*100:.1f}%)")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 6: WEIGHT CLASS MOVEMENT
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Track if a fighter is moving up or down in weight class.
    print("    Building weight class movement features...")

    # Map divisions to approximate weight in pounds
    DIVISION_WEIGHT_MAP = {
        'strawweight': 115, 'flyweight': 125, 'bantamweight': 135,
        'featherweight': 145, 'lightweight': 155, 'welterweight': 170,
        'middleweight': 185, 'light heavyweight': 205, 'heavyweight': 265,
        "women's strawweight": 115, "women's flyweight": 125,
        "women's bantamweight": 135, "women's featherweight": 145,
    }

    def get_division_weight(division):
        """Get approximate weight from division name."""
        if pd.isna(division):
            return None
        div_lower = str(division).lower()
        for key, weight in DIVISION_WEIGHT_MAP.items():
            if key in div_lower:
                return weight
        return None

    def calc_weight_movement(df):
        """Calculate weight class movement for each fighter."""
        fighter_last_weight = {}
        wc_move_r, wc_move_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            current_weight = get_division_weight(row.get('division', ''))

            # Red corner movement
            r_last_weight = fighter_last_weight.get(r_name)
            if r_last_weight and current_weight:
                if current_weight > r_last_weight:
                    wc_move_r.append(1)   # Moving UP
                elif current_weight < r_last_weight:
                    wc_move_r.append(-1)  # Moving DOWN (cutting more)
                else:
                    wc_move_r.append(0)   # Same weight class
            else:
                wc_move_r.append(0)

            # Blue corner movement
            b_last_weight = fighter_last_weight.get(b_name)
            if b_last_weight and current_weight:
                if current_weight > b_last_weight:
                    wc_move_b.append(1)
                elif current_weight < b_last_weight:
                    wc_move_b.append(-1)
                else:
                    wc_move_b.append(0)
            else:
                wc_move_b.append(0)

            # Update last known weight
            if current_weight:
                fighter_last_weight[r_name] = current_weight
                fighter_last_weight[b_name] = current_weight

        return wc_move_r, wc_move_b, fighter_last_weight

    wc_move_r, wc_move_b, WC_MOVEMENT_HISTORY = calc_weight_movement(ufc)
    ufc['r_wc_move'] = wc_move_r
    ufc['b_wc_move'] = wc_move_b
    ufc['wc_move_diff'] = ufc['r_wc_move'] - ufc['b_wc_move']

    print(f"      Weight class movers: R_up={sum(1 for x in wc_move_r if x>0)}, R_down={sum(1 for x in wc_move_r if x<0)}")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 8: WEIGHT-CLASS-SPECIFIC AGE CURVES
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Different weight classes peak at different ages.
    print("    Building weight-class-specific age curves...")

    # Peak age ranges by weight class (evidence-based estimates)
    WC_PEAK_AGES = {
        'strawweight': (27, 32),    # Lighter = peaks slightly earlier
        'flyweight': (27, 32),
        'bantamweight': (28, 33),
        'featherweight': (28, 33),
        'lightweight': (28, 33),
        'welterweight': (29, 34),
        'middleweight': (29, 34),
        'light_heavyweight': (30, 35),
        'heavyweight': (30, 36),    # HW peaks later, declines slower
        'other': (28, 33),
        'unknown': (28, 33),
    }

    def age_prime_score_wc(age, weight_class='unknown'):
        """Weight-class-specific prime score."""
        if pd.isna(age):
            return 0.5
        peak_start, peak_end = WC_PEAK_AGES.get(weight_class, (28, 33))
        if peak_start <= age <= peak_end:
            return 1.0
        elif age < peak_start:
            return 0.7 + 0.3 * max(0, (age - (peak_start - 6))) / 6
        else:  # past peak
            decline_rate = 0.08 if weight_class in ['heavyweight', 'light_heavyweight'] else 0.12
            return max(0.25, 1.0 - decline_rate * (age - peak_end))

    # Compute weight-class-aware prime scores
    ufc['r_prime_wc'] = [
        age_prime_score_wc(age, wc)
        for age, wc in zip(ufc['r_age'], ufc['weight_class'])
    ]
    ufc['b_prime_wc'] = [
        age_prime_score_wc(age, wc)
        for age, wc in zip(ufc['b_age'], ufc['weight_class'])
    ]
    ufc['prime_wc_diff'] = ufc['r_prime_wc'] - ufc['b_prime_wc']

    print(f"      WC-aware prime diff range: [{ufc['prime_wc_diff'].min():.3f}, {ufc['prime_wc_diff'].max():.3f}]")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 9: RING RUST NON-LINEARITY
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Transform linear layoff to non-linear (bucketed with quadratic penalty).
    print("    Building non-linear ring rust features...")

    def ring_rust_transform(layoff_days):
        """
        Non-linear ring rust transformation.
        0-180 days: Normal (score = 0)
        180-365 days: Slight rust (score = -0.05 to -0.15)
        365-730 days: Significant (score = -0.15 to -0.40)
        730+ days: Major unknown (score = -0.40 to -0.60)
        """
        if pd.isna(layoff_days):
            return 0.0
        days = float(layoff_days)
        if days <= 180:
            return 0.0  # Normal training camp spacing
        elif days <= 365:
            # Linear ramp from 0 to -0.15
            return -0.15 * (days - 180) / 185
        elif days <= 730:
            # Quadratic ramp from -0.15 to -0.40
            progress = (days - 365) / 365
            return -0.15 - 0.25 * (progress ** 1.5)
        else:
            # Severe but capped
            return min(-0.40, -0.40 - 0.10 * min((days - 730) / 365, 2.0))

    ufc['r_ring_rust'] = ufc['r_layoff'].apply(ring_rust_transform)
    ufc['b_ring_rust'] = ufc['b_layoff'].apply(ring_rust_transform)
    ufc['ring_rust_diff'] = ufc['r_ring_rust'] - ufc['b_ring_rust']

    print(f"      Ring rust diff range: [{ufc['ring_rust_diff'].min():.3f}, {ufc['ring_rust_diff'].max():.3f}]")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 11: SUBMISSION DEFENSE SPECIFICS
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Track sub attempts survived vs submission losses.
    print("    Building submission defense features...")

    def calc_sub_defense(df):
        """
        Track submission defense: sub attempts faced vs times submitted.
        High sub_attempts_faced with low sub_losses = great ground defense.
        """
        fighter_sub_defense = {}  # fighter -> {sub_attempts_faced, times_submitted, fights}
        sub_def_r, sub_def_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            winner = row.get('winner', None)
            method = str(row.get('method', '')).upper()

            # Get red corner's sub defense BEFORE this fight
            r_def = fighter_sub_defense.get(r_name, {'faced': 0, 'submitted': 0, 'fights': 0})
            b_def = fighter_sub_defense.get(b_name, {'faced': 0, 'submitted': 0, 'fights': 0})

            # Sub defense score: proportion of sub attempts survived
            if r_def['faced'] > 0:
                r_score = 1.0 - (r_def['submitted'] / r_def['faced'])
            elif r_def['fights'] > 2:
                r_score = 0.8  # Never faced subs = decent but uncertain
            else:
                r_score = 0.5  # Unknown

            if b_def['faced'] > 0:
                b_score = 1.0 - (b_def['submitted'] / b_def['faced'])
            elif b_def['fights'] > 2:
                b_score = 0.8
            else:
                b_score = 0.5

            sub_def_r.append(r_score)
            sub_def_b.append(b_score)

            # Update histories AFTER recording pre-fight values
            # Red fighter faced blue's sub attempts
            b_sub_att = pd.to_numeric(row.get('b_sub_att', 0), errors='coerce')
            b_sub_att = int(b_sub_att) if pd.notna(b_sub_att) else 0

            r_sub_att = pd.to_numeric(row.get('r_sub_att', 0), errors='coerce')
            r_sub_att = int(r_sub_att) if pd.notna(r_sub_att) else 0

            if r_name not in fighter_sub_defense:
                fighter_sub_defense[r_name] = {'faced': 0, 'submitted': 0, 'fights': 0}
            if b_name not in fighter_sub_defense:
                fighter_sub_defense[b_name] = {'faced': 0, 'submitted': 0, 'fights': 0}

            # Red faced blue's sub attempts
            fighter_sub_defense[r_name]['faced'] += b_sub_att
            fighter_sub_defense[r_name]['fights'] += 1

            # Blue faced red's sub attempts
            fighter_sub_defense[b_name]['faced'] += r_sub_att
            fighter_sub_defense[b_name]['fights'] += 1

            # Track who got submitted
            if pd.notna(winner) and 'SUB' in method:
                loser = r_name if winner != r_name else b_name
                fighter_sub_defense[loser]['submitted'] += 1

        return sub_def_r, sub_def_b, fighter_sub_defense

    sub_def_r, sub_def_b, SUB_DEFENSE_STATS = calc_sub_defense(ufc)
    ufc['r_sub_def_score'] = sub_def_r
    ufc['b_sub_def_score'] = sub_def_b
    ufc['sub_def_diff'] = ufc['r_sub_def_score'] - ufc['b_sub_def_score']

    print(f"      Sub defense diff range: [{ufc['sub_def_diff'].min():.3f}, {ufc['sub_def_diff'].max():.3f}]")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # FEATURE 12: MOMENTUM QUALITY
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Weight streaks by impressiveness of wins.
    # KO/TKO R1 = most impressive, Split Decision = least impressive.
    print("    Building momentum quality features...")

    def calc_momentum_quality(df):
        """
        Weighted momentum: recent wins weighted by how impressive they were.
        Weights: R1 KO=1.5, R2 KO=1.3, R3+ KO=1.1, Sub=1.2, Dec=0.8, Split Dec=0.6
        Losses weighted inversely (being KO'd early = worse momentum hit).
        """
        fighter_momentum = {}  # fighter -> list of recent quality scores (last 5)
        mom_quality_r, mom_quality_b = [], []

        for idx, row in df.iterrows():
            r_name = row['r_name']
            b_name = row['b_name']
            winner = row.get('winner', None)
            method = str(row.get('method', '')).upper()
            finish_rd = row.get('finish_round', 3)
            if pd.isna(finish_rd):
                finish_rd = 3

            # Record pre-fight momentum quality
            r_hist = fighter_momentum.get(r_name, [])
            b_hist = fighter_momentum.get(b_name, [])

            # Compute EWM-weighted quality (last 5 fights)
            def compute_quality(hist):
                if not hist:
                    return 0.0
                # More recent fights weighted more
                weights = [0.85 ** (len(hist) - 1 - i) for i in range(len(hist))]
                w_sum = sum(weights)
                return sum(q * w for q, w in zip(hist, weights)) / w_sum

            mom_quality_r.append(compute_quality(r_hist[-5:]))
            mom_quality_b.append(compute_quality(b_hist[-5:]))

            # Determine win quality score
            def get_win_quality(method, finish_rd):
                if 'KO' in method or 'TKO' in method:
                    if finish_rd <= 1: return 1.5
                    elif finish_rd <= 2: return 1.3
                    else: return 1.1
                elif 'SUB' in method:
                    if finish_rd <= 2: return 1.3
                    else: return 1.1
                elif 'UNANIMOUS' in method:
                    return 0.85
                elif 'SPLIT' in method:
                    return 0.6
                elif 'MAJORITY' in method:
                    return 0.7
                elif 'DEC' in method:
                    return 0.8
                return 0.7

            # Update histories
            if pd.notna(winner):
                win_quality = get_win_quality(method, finish_rd)
                if winner == r_name:
                    if r_name not in fighter_momentum:
                        fighter_momentum[r_name] = []
                    fighter_momentum[r_name].append(win_quality)
                    if b_name not in fighter_momentum:
                        fighter_momentum[b_name] = []
                    # Scale loss penalty by opponent quality (losing to elite = minor hit)
                    loss_scale = loss_penalty_scale(
                        row.get('r_mu', TRUESKILL_DEFAULT_MU))
                    fighter_momentum[b_name].append(-win_quality * loss_scale)
                elif winner == b_name:
                    if b_name not in fighter_momentum:
                        fighter_momentum[b_name] = []
                    fighter_momentum[b_name].append(win_quality)
                    if r_name not in fighter_momentum:
                        fighter_momentum[r_name] = []
                    # Scale loss penalty by opponent quality
                    loss_scale = loss_penalty_scale(
                        row.get('b_mu', TRUESKILL_DEFAULT_MU))
                    fighter_momentum[r_name].append(-win_quality * loss_scale)
            else:
                # Draw/NC/Unknown
                if r_name not in fighter_momentum:
                    fighter_momentum[r_name] = []
                if b_name not in fighter_momentum:
                    fighter_momentum[b_name] = []

        return mom_quality_r, mom_quality_b, fighter_momentum

    mom_quality_r, mom_quality_b, MOMENTUM_QUALITY_HISTORY = calc_momentum_quality(ufc)
    ufc['r_mom_quality'] = mom_quality_r
    ufc['b_mom_quality'] = mom_quality_b
    ufc['mom_quality_diff'] = ufc['r_mom_quality'] - ufc['b_mom_quality']

    print(f"      Momentum quality diff range: [{ufc['mom_quality_diff'].min():.3f}, {ufc['mom_quality_diff'].max():.3f}]")

    print("    [3.95] Advanced features complete!")




    # ============================================================================
    # SECTION 3.99: BUILD THE DECLARED FEATURES
    # ============================================================================
    # feature_inventory declares every paired feature, level, known flag and
    # matchup interaction; feature_spec builds them. Assigning them here overwrites
    # the hand-written versions of the same names, which is the point: fifteen of
    # them filled each operand with 0 before subtracting, so a debutant's missing
    # 45% striking accuracy read as "lands nothing" and handed the opponent a fake
    # maximal advantage.
    print("\n[3.99] BUILDING DECLARED FEATURES...")

    SPECS = all_specs()
    # Interaction centres are a training-frame statistic: frozen from
    # training for a live build (centres=), recorded here either way.
    INTERACTION_CENTRES = {}
    _declared = build_all(SPECS, ufc, centres=centres,
                          fitted=INTERACTION_CENTRES)
    for _col in _declared.columns:
        ufc[_col] = _declared[_col]
    print(f"    {len(SPECS)} specs -> {len(_declared.columns)} columns")
    _new = [c for c in _declared.columns if c not in ufc.columns]
    print(f"    replaced hand-written definitions where names matched")

    # ============================================================================
    # SECTION 4: DEFINE FEATURE COLUMNS
    # ============================================================================
    print("\n[4] DEFINING FEATURE SET...")

    # Every name the declaration emits. Cage-control features stay out of the
    # winner model, as they were before, and are listed separately for that.
    CAGE_CONTROL_FEATURES = [
        'cage_control_cap_diff', 'cage_control_cap_level',
        'clinch_activity_diff', 'grind_tendency_diff',
    ]

    # How a fight ends is a different question from who wins it, and these answer
    # the first. They stay out of the winner model for the same reason cage
    # control does - not because they leak, but because the winner model's numbers
    # are what everything downstream is calibrated against, and this change was
    # measured on the method model alone. Adding columns to the winner model would
    # move the AUC, the ROI and the flag quality all at once, none of it measured.
    # A division finish prior was built, measured and REJECTED. Heavyweight
    # finishes 63.9% of the time and women's strawweight 33.6%, so it looked like
    # the largest single effect available - and over eight seeds on the confirm
    # period it beats the old feature set (+0.0060 log loss) and LOSES to the one
    # already shipping (1.0281 against 1.0266; macro-F1 0.4017 against 0.4036).
    # Clearing the baseline is not the test. Beating what is already there is.
    #
    # division_prior.py, its tests and the measurement all stay, so the decision
    # can be re-examined rather than re-derived; it is simply not computed here,
    # because a rejected feature should not cost every production run. See
    # experiments/method_noise.py.
    METHOD_RATE_FEATURES = method_rate_names() + finish_level_names()
    METHOD_ONLY_FEATURES = CAGE_CONTROL_FEATURES + METHOD_RATE_FEATURES

    SPEC_FEATURES = [n for n in emitted_names(SPECS)
                     if n not in METHOD_ONLY_FEATURES]

    # Features the declaration does not cover: context flags, transforms of the
    # rating, and the archetype terms. Three of the old names are gone:
    #
    #   same_cluster         AUC 0.500 - constant
    #   stance_interaction   correlated 1.0000 with southpaw_diff, a copy
    #   trajectory_diff      exactly 0 on 100% of rows
    #
    # Removing a constant cannot change a tree model, and removing an exact
    # duplicate cannot either, so unlike importance-based pruning - which made this
    # model worse every time it was tried - these three are free.
    bespoke_features = [
        'bayesian_prob',          # TrueSkill win probability
        'combined_uncertainty',   # Total uncertainty in the matchup
        'mu_sum',                 # How good the fight is, not just how lopsided
        'mu_diff_z',              # Skill gap in units of its own uncertainty
        'base_prob',              # Logistic of the rating difference
        'mmr_diff',               # Conservative rating difference
        'power_diff',             # Weighted striking composite
        'finish_rate_diff',       # KO plus submission rate
        'archetype_matchup',      # Stylistic matchup advantage
        'archetype_clash',        # Whether the archetypes differ
        'combined_pace',          # Fight tempo, AUC 0.416 - a level that works
        'high_pace',              # Binary: high-pace fight likely
        'opp_history_known',      # Both schedules known
        'is_womens', 'is_heavyweight', 'is_5rnd', 'is_title',
    ]

    feature_cols = SPEC_FEATURES + bespoke_features + METHOD_ONLY_FEATURES

    # The winner model gets neither the cage-control columns nor the finish rates.
    # The method, finish and round models get everything.
    feature_cols_winner = SPEC_FEATURES + bespoke_features
    print(f"    Total features: {len(feature_cols)} "
          f"({len(SPEC_FEATURES)} declared + {len(bespoke_features)} bespoke "
          f"+ {len(CAGE_CONTROL_FEATURES)} cage control "
          f"+ {len(METHOD_RATE_FEATURES)} finish rates)")
    print(f"    Winner model sees {len(feature_cols_winner)}; "
          f"method, finish and round see all {len(feature_cols)}")

    # Build feature matrix
    X = ufc[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # ---- end of the moved code ---------------------------------------------
    return dict(locals())
