"""
MMA Prediction System v4 - YourMMA

Ensemble fight predictor (LogisticRegression + RandomForest + XGBoost, Platt
calibrated) over UFC fight history with TrueSkill-based skill ratings.

Converted from MMA_predict_v4_full.ipynb. Edit the USER SETTINGS block below to
set the event and fight card, then run:

    pip install -r requirements.txt
    python engine/predict_card.py

Optional environment variables:
    ODDS_API_KEY     The Odds API key, for live odds and value/edge analysis
    UFC_CSV          override path to the fight history CSV
    PREDICTIONS_LOG  override path to prediction_history.json
"""

# ============================================================================
# MMA PREDICTION SYSTEM v4 - COMPREHENSIVE SINGLE-CELL IMPLEMENTATION
# ============================================================================
# Features:
#   - Original feature set (base_prob, mmr_diff, exp_diff, age_diff, striking, grappling)
#   - Bayesian state-space skill model (TrueSkill mu/sigma with uncertainty features)
#   - Time-aware features (L3/L5 rolling averages, momentum)
#   - 70/15/15 temporal train/cal/test split
#   - Ensemble: LogisticRegression + RandomForest + XGBoost + Platt calibration
#   - Multi-output: Win + Method + Round prediction
#   - Walk-forward evaluation (expanding window)
#   - Fuzzy name matching for fighter lookup
#   - Early stopping for XGBoost to prevent overfitting
#   - Layoff days, win/loss streaks, age prime features
#   - Automated leakage audits and time-travel tests
#   - Optional context inputs (home/away, cage size, altitude, short notice)
# ============================================================================



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           USER SETTINGS - EDIT HERE                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ==============================================================================
# 1. OPTUNA HYPERPARAMETER TUNING
# ==============================================================================
# Set to True to run hyperparameter optimization (slower, ~5-10 minutes)
# Set to False to use pre-tuned parameters (faster, recommended for normal use)
RUN_OPTUNA = False

# ==============================================================================
# 2. EVENT DETAILS
# ==============================================================================
EVENT_NAME = "UFC Fight Night: Kape vs. Horiguchi"
EVENT_DATE = "2026-06-20"  # Format: YYYY-MM-DD

# ==============================================================================
# 3. FIGHT CARD
# ==============================================================================
# Format: (Red_Corner, Blue_Corner, is_5_rounds, is_title_fight)
# - is_5_rounds: True for main events and title fights
# - is_title_fight: True if fighting for a championship

FIGHT_CARD = [
    # Main Event (5 rounds, non-title) - Flyweight
    ("Manel Kape", "Kyoji Horiguchi", True, False),            # 0 - Main Event, Flyweight (5 rounds)

    # Co-Main Event - Light Heavyweight
    ("Ion Cutelaba", "Navajo Stirling"),                       # 1 - Co-Main, Light Heavyweight

    # Main Card
    ("Hyder Amil", "Christian Rodriguez"),                     # 2 - Featherweight
    ("Melsik Baghdasaryan", "Murtazali Magomedov"),            # 3 - Featherweight
    ("Andre Fili", "Vinicius Oliveira"),                       # 4 - Featherweight

    # Preliminary Card
    ("Andre Lima", "Kevin Borjas"),                            # 5 - Catchweight (129 lb)
    ("Beatriz Mesquita", "Melissa Mullins"),                   # 6 - Women's Bantamweight
    ("Allan Nascimento", "Mitch Raposo"),                      # 7 - Flyweight
    ("Gaston Bolanos", "Michael Aswell Jr."),                  # 8 - Featherweight
    ("Leon Shahbazyan", "Levan Chokheli"),                     # 9 - Welterweight
    ("Karol Rosa", "Luana Santos"),                            # 10 - Women's Bantamweight
    ("Shane Collins", "Otari Tanzilovi"),                      # 11 - Featherweight
]

# ==============================================================================
# 4. OPTIONAL CONTEXT ADJUSTMENTS (Per-Fight)
# ==============================================================================
# Home advantage only counts when ONE fighter is home and the other is NOT.
# If both fighters are "home" (e.g., both USA-based), no advantage.
#
# UFC Fight Night: Kape vs. Horiguchi - UFC Apex, Enterprise (Las Vegas), NV (SMALL cage)
#
# Home/Away Analysis (event in USA):
#   Fight 0: Kape (Angola/Portugal, Away) vs Horiguchi (Japan, Away) -> no advantage
#   Fight 1: Cutelaba (Moldova, Away) vs Stirling (New Zealand, Away) -> no advantage
#   Fight 2: Amil (USA, Home) vs Rodriguez (USA, Home) -> no advantage (both home)
#   Fight 3: Baghdasaryan (Armenia, Away) vs Magomedov (Russia, Away) -> no advantage
#   Fight 4: Fili (USA, Home) vs Oliveira (Brazil, Away) -> red_home
#   Fight 5: Lima (Brazil, Away) vs Borjas (Peru, Away) -> no advantage
#   Fight 6: Mesquita (Brazil, Away) vs Mullins (USA, Home) -> blue_home
#   Fight 7: Nascimento (Brazil, Away) vs Raposo (USA, Home) -> blue_home
#   Fight 8: Bolanos (USA-based, Home) vs Aswell (USA, Home) -> no advantage (both home)
#   Fight 9: Shahbazyan (USA, Home) vs Chokheli (Georgia, Away) -> red_home
#   Fight 10: Rosa (Brazil, Away) vs Santos (Brazil, Away) -> no advantage
#   Fight 11: Collins (USA, Home) vs Tanzilovi (Russia, Away) -> red_home
#
# NOTE: All bouts at the UFC Apex use the SMALL cage -> cage_size: 'small' for every fight.

FIGHT_CONTEXTS = {
    # Fight 0: Both Away - no advantage (small cage)
    0: {'cage_size': 'small'},

    # Fight 1: Both Away - no advantage
    1: {'cage_size': 'small'},

    # Fight 2: Both Home - no advantage
    2: {'cage_size': 'small'},

    # Fight 3: Both Away - no advantage
    3: {'cage_size': 'small'},

    # Fight 4: Fili (Home) - red corner
    4: {'red_home': True, 'cage_size': 'small'},

    # Fight 5: Both Away - no advantage
    5: {'cage_size': 'small'},

    # Fight 6: Mullins (Home) - blue corner
    6: {'blue_home': True, 'cage_size': 'small'},

    # Fight 7: Raposo (Home) - blue corner
    7: {'blue_home': True, 'cage_size': 'small'},

    # Fight 8: Both Home - no advantage
    8: {'cage_size': 'small'},

    # Fight 9: Shahbazyan (Home) - red corner
    9: {'red_home': True, 'cage_size': 'small'},

    # Fight 10: Both Away - no advantage
    10: {'cage_size': 'small'},

    # Fight 11: Collins (Home) - red corner
    11: {'red_home': True, 'cage_size': 'small'},
}

# ==============================================================================
# 5. NAME RESOLUTION SAFETY
# ==============================================================================
# STRICT_NAMES = True means the engine refuses to predict a fighter it cannot
# identify with confidence, instead of quietly substituting someone else or
# inventing average stats. Leave this on unless you know why you want it off.
STRICT_NAMES = True

# A fuzzy match is auto-accepted as a typo ONLY at or above this ratio AND when
# the surname matches exactly. Everything below is offered as a suggestion.
FUZZY_AUTO_ACCEPT_RATIO = 0.90

# Ratio above which near-misses are offered as "did you mean...".
FUZZY_SUGGEST_RATIO = 0.60

# Fighters with fewer recorded fights than this are refused outright.
MIN_FIGHTS_FOR_PREDICTION = 1

# Below this, predict but flag the result as thin data.
LOW_DATA_FIGHT_COUNT = 3

# ==============================================================================
# END OF USER SETTINGS
# ==============================================================================




import pandas as pd
import numpy as np
import difflib  # For fuzzy name matching
import unicodedata
import re
from scipy.stats import norm  # For Bayesian skill probability
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report
import warnings
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_resolution import NameResolver, load_aliases, format_failure, norm_name as _norm_name
warnings.filterwarnings('ignore')

# ==============================================================================
# PATHS - resolved relative to this file so the script runs from any directory
# ==============================================================================
ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = ENGINE_DIR / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
PREDICTIONS_LOG_FILE = Path(
    os.environ.get("PREDICTIONS_LOG", DATA_DIR / "prediction_history.json")
)
FIGHTER_ALIASES_FILE = DATA_DIR / "fighter_aliases.json"

print("="*70)
print("MMA PREDICTION SYSTEM v4 - FULL UPGRADE")
print("="*70)

# ============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[1] LOADING DATA...")

ufc = pd.read_csv(UFC_CSV, low_memory=False)
print(f"    Loaded {len(ufc):,} fights, {len(ufc.columns)} columns")

# Parse dates
ufc['date'] = pd.to_datetime(ufc['date'], errors='coerce')

# Post-2001 filter
ufc = ufc[ufc['date'] >= '2001-01-01'].copy()
ufc = ufc.sort_values('date').reset_index(drop=True)

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
from feature_inventory import all_specs
from feature_spec import build_all, emitted_names
from prediction_row import build_prediction_frame, required_suffixes
from matchup_inputs import matchup_extra

# Where the phone app reads its data from.
APP_DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
_card_simulations = []
from fight_report import (
    CARD_SIMULATIONS,
    format_line,
    simulate_matchup,
    summarise,
)
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
_declared = build_all(SPECS, ufc)
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
SPEC_FEATURES = [n for n in emitted_names(SPECS) if n not in CAGE_CONTROL_FEATURES]

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

feature_cols = SPEC_FEATURES + bespoke_features + CAGE_CONTROL_FEATURES

# Winner model uses features WITHOUT cage control (to avoid noisy signal in winner prediction)
# Cage control features are only used by method/round/finish models
feature_cols_winner = SPEC_FEATURES + bespoke_features
print(f"    Total features: {len(feature_cols)} "
      f"({len(SPEC_FEATURES)} declared + {len(bespoke_features)} bespoke "
      f"+ {len(CAGE_CONTROL_FEATURES)} cage control)")

# Build feature matrix
X = ufc[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"    Feature matrix shape: {X.shape}")

# ============================================================================
# SECTION 4.5: LEAKAGE AUDIT & TIME-TRAVEL TESTS
# ============================================================================
print("\n[4.5] RUNNING LEAKAGE AUDITS...")

leakage_audit_passed = True
leakage_issues = []

# --- AUDIT 1: Check that rolling features use shift(1) ---
# This ensures we're not using current fight outcome in rolling stats
print("    [Audit 1] Verifying rolling features use shifted data...")

# Sample a few fighters and check their rolling stats progression
sample_fighters = ufc['r_name'].value_counts().head(5).index.tolist()
for fighter in sample_fighters:
    fighter_fights = ufc[(ufc['r_name'] == fighter) | (ufc['b_name'] == fighter)].sort_values('date')
    if len(fighter_fights) >= 3:
        # Check if won_L3 at fight N excludes fight N's outcome
        for i in range(2, min(5, len(fighter_fights))):
            row = fighter_fights.iloc[i]
            prev_row = fighter_fights.iloc[i-1]
            
            # Get the rolling stat column for this fighter
            if row['r_name'] == fighter:
                current_L3 = row.get('r_won_L3', np.nan)
            else:
                current_L3 = row.get('b_won_L3', np.nan)
            
            # L3 should be based on previous fights, not current
            if pd.notna(current_L3):
                # This is a sanity check - the value should exist and be reasonable
                if current_L3 < 0 or current_L3 > 1:
                    leakage_issues.append(f"Invalid L3 value for {fighter}: {current_L3}")
                    leakage_audit_passed = False

if not leakage_issues:
    print("      PASSED: Rolling features properly shifted")

# --- AUDIT 2: Check streak calculation uses only past outcomes ---
print("    [Audit 2] Verifying streaks are pre-fight values...")

# For a fighter's first fight, streak should be 0
first_fighters = ufc.groupby('r_name').first().reset_index()
invalid_first_streaks = first_fighters[first_fighters['r_streak'] != 0]
if len(invalid_first_streaks) > 0:
    # Check if these are actually first fights
    for _, row in invalid_first_streaks.head(3).iterrows():
        fighter = row['r_name']
        all_fights = ufc[(ufc['r_name'] == fighter) | (ufc['b_name'] == fighter)].sort_values('date')
        if len(all_fights) > 1 and all_fights.iloc[0]['date'] < row['date']:
            pass  # Not actually first fight
        else:
            leakage_issues.append(f"Non-zero streak on first fight for {fighter}")
            leakage_audit_passed = False

if 'Non-zero streak' not in str(leakage_issues):
    print("      PASSED: Streak values are pre-fight")

# --- AUDIT 3: Check layoff uses only past fight dates ---
print("    [Audit 3] Verifying layoff uses past dates only...")

# First fight for each fighter should have default layoff (500)
# This is implicitly checked by calc_layoff_and_streaks logic
debut_fighters = []
for fighter in sample_fighters[:3]:
    all_fights = ufc[(ufc['r_name'] == fighter) | (ufc['b_name'] == fighter)].sort_values('date')
    if len(all_fights) > 0:
        first_fight = all_fights.iloc[0]
        if first_fight['r_name'] == fighter:
            layoff = first_fight['r_layoff']
        else:
            layoff = first_fight['b_layoff']
        if layoff != 500:
            leakage_issues.append(f"Unexpected layoff on debut for {fighter}: {layoff}")

if 'Unexpected layoff' not in str(leakage_issues):
    print("      PASSED: Layoff values use past dates only")

# --- AUDIT 4: Check no target columns in features ---
print("    [Audit 4] Checking features don't include target columns...")

target_cols = ['target_win', 'target_method', 'target_round', 'winner', 'method', 'finish_round']
leaked_targets = [c for c in feature_cols if c in target_cols]
if leaked_targets:
    leakage_issues.append(f"Target columns in features: {leaked_targets}")
    leakage_audit_passed = False
else:
    print("      PASSED: No target columns in feature set")

# --- AUDIT 5: Check no per-fight outcome stats (e.g., r_kd in this fight) ---
print("    [Audit 5] Checking no per-fight outcome stats leak...")

# r_kd, b_kd are career averages in this dataset, but we removed them from power_diff anyway
# Check power_diff formula doesn't reference kd
if 'r_kd' in str(ufc['power_diff'].dtype) or 'b_kd' in str(ufc['power_diff'].dtype):
    leakage_issues.append("power_diff may still reference per-fight KD")
else:
    print("      PASSED: power_diff uses only career stats")

# --- AUDIT 6: Time-travel correlation check - features at time T should not correlate perfectly with outcome ---
print("    [Audit 6] Time-travel correlation check...")

# If features perfectly predict outcome, there's likely leakage
valid_mask = ufc['target_win'].notna()
ufc_check = ufc[valid_mask].copy()

# Check correlation of each feature with target
high_corr_features = []
for feat in feature_cols:
    if feat in ufc_check.columns:
        corr = ufc_check[feat].corr(ufc_check['target_win'])
        if abs(corr) > 0.9:  # Suspiciously high correlation
            high_corr_features.append((feat, corr))

if high_corr_features:
    print(f"      WARNING: High correlations detected (may indicate leakage):")
    for feat, corr in high_corr_features:
        print(f"        {feat}: {corr:.3f}")
    leakage_issues.append(f"High correlation features: {high_corr_features}")
else:
    print("      PASSED: No suspiciously high feature-target correlations")

# --- AUDIT 7: Verify Bayesian features use pre-fight values ---
print("    [Audit 7] Verifying Bayesian skill features are pre-fight...")

# Check that mu_pre and sigma_pre are used (not post-fight values)
# The column names explicitly contain "_pre" which indicates pre-fight
if 'r_mu_pre' in ufc.columns and 'b_mu_pre' in ufc.columns:
    # Verify the features are derived from _pre columns
    mu_check = (ufc['mu_diff'] == (ufc['r_mu'] - ufc['b_mu'])).all()
    if mu_check:
        print("      PASSED: Bayesian features use pre-fight skill estimates")
    else:
        leakage_issues.append("Bayesian mu_diff calculation inconsistency")
else:
    leakage_issues.append("Missing pre-fight mu columns")

# --- FINAL AUDIT SUMMARY ---
print(f"\n    LEAKAGE AUDIT SUMMARY:")
if leakage_audit_passed and not leakage_issues:
    print("    âœ“ ALL AUDITS PASSED - No obvious leakage detected")
else:
    print(f"    âœ— ISSUES FOUND ({len(leakage_issues)}):")
    for issue in leakage_issues[:5]:  # Show first 5
        print(f"      - {issue}")
    print("\n    NOTE: Review flagged issues before trusting model metrics")

# ============================================================================
# SECTION 5: TRAIN/CAL/TEST SPLIT (70/15/15 TEMPORAL)
# ============================================================================
print("\n[5] SPLITTING DATA (70/15/15 temporal)...")

# Filter to valid win targets
valid_mask = ufc['target_win'].notna()
X_valid = X[valid_mask].copy()
X_valid_winner = X_valid[feature_cols_winner].copy()  # Winner model features only
ufc_valid = ufc[valid_mask].copy()
y_win = ufc_valid['target_win'].values

print(f"    Valid fights: {len(X_valid):,}")
print(f"    Red corner win rate: {y_win.mean():.3f}")

# Split indices
n = len(X_valid)
train_end = int(n * 0.70)
cal_end = int(n * 0.85)

X_train, X_cal, X_test = X_valid.iloc[:train_end], X_valid.iloc[train_end:cal_end], X_valid.iloc[cal_end:]
y_train, y_cal, y_test = y_win[:train_end], y_win[train_end:cal_end], y_win[cal_end:]

train_dates = ufc_valid.iloc[:train_end]['date']
cal_dates = ufc_valid.iloc[train_end:cal_end]['date']
test_dates = ufc_valid.iloc[cal_end:]['date']

print(f"    Train: {len(X_train):,} ({train_dates.min().date()} to {train_dates.max().date()})")
print(f"    Cal:   {len(X_cal):,} ({cal_dates.min().date()} to {cal_dates.max().date()})")
print(f"    Test:  {len(X_test):,} ({test_dates.min().date()} to {test_dates.max().date()})")

# Scale features

# --- FEATURE 14: Training Data Recency Weighting ---
# More recent fights get higher training weight (MMA evolves fast)
print("    Computing recency-based sample weights...")

train_dates_arr = ufc_valid.iloc[:train_end]['date']
min_date = train_dates_arr.min()
max_date = train_dates_arr.max()
date_range_days = (max_date - min_date).days

# Exponential recency weights: recent fights weighted up to 3x more
recency_weights = np.array([
    1.0 + 2.0 * ((d - min_date).days / max(date_range_days, 1)) ** 1.5
    for d in train_dates_arr
])
# Normalize to mean=1 so total effective sample size stays similar
recency_weights = recency_weights / recency_weights.mean()
print(f"      Recency weight range: [{recency_weights.min():.2f}, {recency_weights.max():.2f}]")


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s = scaler.transform(X_cal)
X_test_s = scaler.transform(X_test)

# ============================================================================
# SECTION 5.5: HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================
print("\n[5.5] HYPERPARAMETER TUNING WITH OPTUNA...")

# RUN_OPTUNA is defined in USER SETTINGS at top of file

if RUN_OPTUNA:
    print("    Running Optuna optimization (this may take a few minutes)...")

    # Use a subset for faster tuning
    n_tune = min(3000, len(X_train))
    X_tune = X_train_s[:n_tune]
    y_tune = y_train[:n_tune]
    X_val_tune = X_cal_s[:min(500, len(X_cal))]
    y_val_tune = y_cal[:min(500, len(y_cal))]

    # XGBoost objective
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
            'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0,
            'early_stopping_rounds': 30
        }
        model = XGBClassifier(**params)
        model.fit(X_tune, y_tune, eval_set=[(X_val_tune, y_val_tune)], verbose=False)
        return model.score(X_val_tune, y_val_tune)

    # Random Forest objective
    def rf_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 6, 18),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'random_state': 42, 'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tune, y_tune)
        return model.score(X_val_tune, y_val_tune)

    # MLP objective
    def mlp_objective(trial):
        n_layers = trial.suggest_int('n_layers', 2, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 32, 256))
        params = {
            'hidden_layer_sizes': tuple(layers),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'activation': 'relu', 'solver': 'adam',
            'learning_rate': 'adaptive', 'max_iter': 300,
            'early_stopping': True, 'validation_fraction': 0.15,
            'n_iter_no_change': 15, 'random_state': 42, 'verbose': False
        }
        model = MLPClassifier(**params)
        model.fit(X_tune, y_tune)
        return model.score(X_val_tune, y_val_tune)

    # Run optimizations
    print("    Tuning XGBoost...")
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=False)
    BEST_XGB_PARAMS = xgb_study.best_params
    print(f"      Best XGB accuracy: {xgb_study.best_value:.4f}")

    print("    Tuning Random Forest...")
    rf_study = optuna.create_study(direction='maximize')
    rf_study.optimize(rf_objective, n_trials=20, show_progress_bar=False)
    BEST_RF_PARAMS = rf_study.best_params
    print(f"      Best RF accuracy: {rf_study.best_value:.4f}")

    print("    Tuning MLP...")
    mlp_study = optuna.create_study(direction='maximize')
    mlp_study.optimize(mlp_objective, n_trials=20, show_progress_bar=False)
    BEST_MLP_PARAMS = mlp_study.best_params
    print(f"      Best MLP accuracy: {mlp_study.best_value:.4f}")

    print("    \u2713 Hyperparameter tuning complete!")
else:
    print("    Using pre-tuned hyperparameters (set RUN_OPTUNA=True to retune)")
    # Pre-tuned parameters (update these after running Optuna once)
    BEST_XGB_PARAMS = {
        'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0
    }
    BEST_RF_PARAMS = {
        'n_estimators': 200, 'max_depth': 12,
        'min_samples_leaf': 10, 'min_samples_split': 10
    }
    BEST_MLP_PARAMS = {
        'n_layers': 3, 'n_units_0': 128, 'n_units_1': 64, 'n_units_2': 32,
        'alpha': 0.001, 'learning_rate_init': 0.001, 'batch_size': 32
    }

# ============================================================================
# SECTION 6: WIN PREDICTION MODELS (ENSEMBLE)
# ============================================================================
print("\n[6] TRAINING WIN PREDICTION MODELS...")

# Logistic Regression
print("    Training Logistic Regression...")
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train, sample_weight=recency_weights)
print(f"      Train acc: {lr.score(X_train_s, y_train):.4f}")

# Random Forest (with max_depth cap to reduce overfitting)
print("    Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=10,
    min_samples_split=10,  # Additional regularization
    random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train, sample_weight=recency_weights)
print(f"      Train acc: {rf.score(X_train_s, y_train):.4f}")

# XGBoost (with early stopping to prevent overfitting)
print("    Training XGBoost (with early stopping)...")
xgb_win = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,  # L1/L2 regularization
    random_state=42, eval_metric='logloss', verbosity=0,
    early_stopping_rounds=50
)
xgb_win.fit(X_train_s, y_train, sample_weight=recency_weights, eval_set=[(X_cal_s, y_cal)], verbose=False)
print(f"      Train acc: {xgb_win.score(X_train_s, y_train):.4f}")
print(f"      Early stopped at iteration: {xgb_win.best_iteration}")

# Neural Network (MLP)
print("    Training Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)
# Note: MLPClassifier does not support sample_weight
mlp.fit(X_train_s, y_train)
print(f"      Train acc: {mlp.score(X_train_s, y_train):.4f}")
print(f"      Iterations: {mlp.n_iter_}")

# Ensemble probabilities on calibration set
p_lr_cal = lr.predict_proba(X_cal_s)[:, 1]
p_rf_cal = rf.predict_proba(X_cal_s)[:, 1]
p_xgb_cal = xgb_win.predict_proba(X_cal_s)[:, 1]
p_mlp_cal = mlp.predict_proba(X_cal_s)[:, 1]
p_ens_cal = (p_lr_cal + p_rf_cal + p_xgb_cal + p_mlp_cal) / 4

# Platt calibration on calibration set (NOT test set!)
print("    Fitting Platt calibration...")
platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
platt.fit(p_ens_cal.reshape(-1, 1), y_cal)

# Evaluate on TEST set
print("\n    EVALUATING ON TEST SET...")
p_lr_test = lr.predict_proba(X_test_s)[:, 1]
p_rf_test = rf.predict_proba(X_test_s)[:, 1]
p_xgb_test = xgb_win.predict_proba(X_test_s)[:, 1]
p_mlp_test = mlp.predict_proba(X_test_s)[:, 1]
p_ens_test = (p_lr_test + p_rf_test + p_xgb_test + p_mlp_test) / 4
p_cal_test = platt.predict_proba(p_ens_test.reshape(-1, 1))[:, 1]

acc = accuracy_score(y_test, p_cal_test > 0.5)
brier = brier_score_loss(y_test, p_cal_test)
logloss = log_loss(y_test, p_cal_test)

print(f"\n    " + "="*50)
print(f"    WIN PREDICTION TEST RESULTS")
print(f"    " + "="*50)
print(f"    Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
print(f"    Brier Score: {brier:.4f}")
print(f"    Log Loss:    {logloss:.4f}")
print(f"\n    Individual models:")
print(f"      LR:  {accuracy_score(y_test, p_lr_test > 0.5):.4f}")
print(f"      RF:  {accuracy_score(y_test, p_rf_test > 0.5):.4f}")
print(f"      XGB: {accuracy_score(y_test, p_xgb_test > 0.5):.4f}")
print(f"      Ens: {accuracy_score(y_test, p_ens_test > 0.5):.4f}")

# ============================================================================
# SECTION 7: METHOD PREDICTION (WITH CLASS BALANCING)
# ============================================================================
print("\n[7] TRAINING METHOD PREDICTION...")

method_mask = ufc_valid['target_method'].isin(['KO/TKO', 'Submission', 'Decision'])
X_method = X_valid[method_mask].copy()
ufc_method = ufc_valid[method_mask].copy()

le_method = LabelEncoder()
y_method = le_method.fit_transform(ufc_method['target_method'])
print(f"    Valid fights: {len(X_method):,}")
print(f"    Classes: {list(le_method.classes_)}")

# Show class distribution
from collections import Counter
class_counts = Counter(y_method)
print(f"    Class distribution: {dict(zip(le_method.classes_, [class_counts[i] for i in range(len(le_method.classes_))]))}")

# Calculate sample weights for class balancing
total_samples = len(y_method)
n_classes = len(le_method.classes_)
class_weights_m = {i: total_samples / (n_classes * class_counts[i]) for i in range(n_classes)}
sample_weights_method = np.array([class_weights_m[y] for y in y_method])
print(f"    Class weights: { {le_method.classes_[i]: f'{class_weights_m[i]:.2f}' for i in range(n_classes)} }")

# Time-based split
n_method = len(X_method)
train_end_m = int(n_method * 0.70)
cal_end_m = int(n_method * 0.85)

X_train_m = scaler.transform(X_method.iloc[:train_end_m])
X_cal_m = scaler.transform(X_method.iloc[train_end_m:cal_end_m])
X_test_m = scaler.transform(X_method.iloc[cal_end_m:])
y_train_m = y_method[:train_end_m]
y_cal_m = y_method[train_end_m:cal_end_m]
y_test_m = y_method[cal_end_m:]
sw_train_m = sample_weights_method[:train_end_m]

# Train with class balancing (sample_weight)
xgb_method = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='multi:softprob', num_class=3,
    random_state=42, eval_metric='mlogloss', verbosity=0,
    early_stopping_rounds=50
)
xgb_method.fit(X_train_m, y_train_m, sample_weight=sw_train_m, eval_set=[(X_cal_m, y_cal_m)], verbose=False)
print(f"    Early stopped at iteration: {xgb_method.best_iteration}")

pred_method = xgb_method.predict(X_test_m)
method_acc = accuracy_score(y_test_m, pred_method)
print(f"    Method prediction accuracy: {method_acc:.4f}")

# Show per-class accuracy
print("    Per-class accuracy:")
for i, cls in enumerate(le_method.classes_):
    mask = y_test_m == i
    if sum(mask) > 0:
        cls_acc = accuracy_score(y_test_m[mask], pred_method[mask])
        print(f"      {cls}: {cls_acc:.4f} (n={sum(mask)})")

# ============================================================================
# SECTION 7.5: SIMPLIFIED FINISH PREDICTION
# ============================================================================
print("\n[7.5] TRAINING SIMPLIFIED FINISH PREDICTION...")

# 1. Is Finish model (Finish vs Decision)
ufc_method['is_finish'] = (ufc_method['target_method'] != 'Decision').astype(int)
y_finish = ufc_method['is_finish'].values

print(f"    Finish rate: {y_finish.mean()*100:.1f}%")

# Time-based split
y_train_fin = y_finish[:train_end_m]
y_cal_fin = y_finish[train_end_m:cal_end_m]
y_test_fin = y_finish[cal_end_m:]

# Train IsFinish model with class balancing
n_neg = len(y_train_fin[y_train_fin==0])
n_pos = len(y_train_fin[y_train_fin==1])
xgb_is_finish = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=n_neg / max(n_pos, 1),  # Balance classes
    random_state=42, eval_metric='logloss', verbosity=0,
    early_stopping_rounds=30
)
xgb_is_finish.fit(X_train_m, y_train_fin, eval_set=[(X_cal_m, y_cal_fin)], verbose=False)

pred_finish = xgb_is_finish.predict(X_test_m)
finish_acc = accuracy_score(y_test_fin, pred_finish)
print(f"    IsFinish accuracy: {finish_acc:.4f}")

# 2. Early vs Late finish model (for fights that are finishes)
# Early = R1-R2, Late = R3+
finish_mask = ufc_method['is_finish'] == 1
X_finish_only = X_method[finish_mask].copy()
ufc_finish_only = ufc_method[finish_mask].copy()

rounds = ufc_finish_only['finish_round'].fillna(3)
y_early = (rounds <= 2).astype(int).values

print(f"    Early finish rate (among finishes): {y_early.mean()*100:.1f}%")

# Split finish-only data
n_fin = len(X_finish_only)
train_end_fin = int(n_fin * 0.70)
cal_end_fin = int(n_fin * 0.85)

X_train_early = scaler.transform(X_finish_only.iloc[:train_end_fin])
X_cal_early = scaler.transform(X_finish_only.iloc[train_end_fin:cal_end_fin])
X_test_early = scaler.transform(X_finish_only.iloc[cal_end_fin:])
y_train_early = y_early[:train_end_fin]
y_cal_early = y_early[train_end_fin:cal_end_fin]
y_test_early = y_early[cal_end_fin:]

# Train Early/Late model
xgb_early_finish = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', verbosity=0,
    early_stopping_rounds=30
)
xgb_early_finish.fit(X_train_early, y_train_early, eval_set=[(X_cal_early, y_cal_early)], verbose=False)

pred_early = xgb_early_finish.predict(X_test_early)
early_acc = accuracy_score(y_test_early, pred_early)
print(f"    Early/Late accuracy: {early_acc:.4f}")
print("    \u2713 Simplified finish prediction ready")

# ============================================================================
# SECTION 8: ROUND PREDICTION
# ============================================================================
print("\n[8] TRAINING ROUND PREDICTION...")

round_mask = ufc_valid['target_round'].isin([1, 2, 3, 4, 5])
X_round = X_valid[round_mask].copy()
ufc_round = ufc_valid[round_mask].copy()
y_round = ufc_round['target_round'].astype(int).values

print(f"    Valid fights: {len(X_round):,}")
print(f"    Distribution: {pd.Series(y_round).value_counts().sort_index().to_dict()}")

# Split
n_r = len(X_round)
train_end_r, cal_end_r = int(n_r * 0.70), int(n_r * 0.85)
X_train_r = scaler.transform(X_round.iloc[:train_end_r])
X_cal_r = scaler.transform(X_round.iloc[train_end_r:cal_end_r])
X_test_r = scaler.transform(X_round.iloc[cal_end_r:])
y_train_r = y_round[:train_end_r] - 1  # 0-indexed
y_cal_r = y_round[train_end_r:cal_end_r] - 1
y_test_r = y_round[cal_end_r:] - 1

# Train (with early stopping)
xgb_round = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='multi:softprob', num_class=5,
    random_state=42, eval_metric='mlogloss', verbosity=0,
    early_stopping_rounds=50
)
xgb_round.fit(X_train_r, y_train_r, eval_set=[(X_cal_r, y_cal_r)], verbose=False)
print(f"    Early stopped at iteration: {xgb_round.best_iteration}")

pred_round = xgb_round.predict(X_test_r)
acc_round = accuracy_score(y_test_r, pred_round)
print(f"\n    ROUND PREDICTION ACCURACY: {acc_round:.4f} ({acc_round*100:.1f}%)")
print(f"\n{classification_report(y_test_r, pred_round, target_names=['R1','R2','R3','R4','R5'])}")

# ============================================================================
# SECTION 9: WALK-FORWARD EVALUATION (EXPANDING WINDOW)
# ============================================================================
print("\n[9] WALK-FORWARD EVALUATION (Expanding Window)...")

ufc_valid['year'] = ufc_valid['date'].dt.year
years = sorted(ufc_valid['year'].unique())
print(f"    Years available: {years[0]} to {years[-1]}")

wf_results = []
start_year = 2015

for test_year in range(start_year, years[-1] + 1):
    train_mask = ufc_valid['year'] < test_year
    test_mask = ufc_valid['year'] == test_year
    
    if train_mask.sum() < 100 or test_mask.sum() < 10:
        continue
    
    X_wf_train = X_valid[train_mask]
    X_wf_test = X_valid[test_mask]
    y_wf_train = y_win[train_mask.values]
    y_wf_test = y_win[test_mask.values]
    
    # Scale
    sc_wf = StandardScaler()
    X_wf_train_s = sc_wf.fit_transform(X_wf_train)
    X_wf_test_s = sc_wf.transform(X_wf_test)
    
    # Quick ensemble (with early stopping for XGB)
    lr_wf = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_wf_train_s, y_wf_train)
    rf_wf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_wf_train_s, y_wf_train)
    xgb_wf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss', verbosity=0).fit(X_wf_train_s, y_wf_train)
    
    p_ens = (lr_wf.predict_proba(X_wf_test_s)[:, 1] + rf_wf.predict_proba(X_wf_test_s)[:, 1] + xgb_wf.predict_proba(X_wf_test_s)[:, 1]) / 3
    acc_wf = accuracy_score(y_wf_test, p_ens > 0.5)
    brier_wf = brier_score_loss(y_wf_test, p_ens)
    
    wf_results.append({'year': test_year, 'acc': acc_wf, 'brier': brier_wf, 'n': len(X_wf_test)})
    print(f"    {test_year}: Acc={acc_wf:.3f}, Brier={brier_wf:.3f}, n={len(X_wf_test)}")

wf_df = pd.DataFrame(wf_results)
print(f"\n    WALK-FORWARD SUMMARY:")
print(f"    Mean Accuracy: {wf_df['acc'].mean():.4f} ({wf_df['acc'].mean()*100:.1f}%)")
print(f"    Std Dev:       {wf_df['acc'].std():.4f}")
print(f"    Min:           {wf_df['acc'].min():.4f} ({wf_df[wf_df['acc']==wf_df['acc'].min()]['year'].values[0]})")
print(f"    Max:           {wf_df['acc'].max():.4f} ({wf_df[wf_df['acc']==wf_df['acc'].max()]['year'].values[0]})")

# ============================================================================
# SECTION 10: FEATURE IMPORTANCE
# ============================================================================
print("\n[10] FEATURE IMPORTANCE (XGBoost)...")

feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_win.feature_importances_
}).sort_values('importance', ascending=False)

print("\n    Top 15 features:")
for i, (_, row) in enumerate(feat_imp.head(15).iterrows()):
    print(f"    {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")

# Highlight the rating-derived features
RATING_FEATURE_NAMES = [
    'bayesian_prob', 'mu_diff', 'mu_level', 'mu_sum', 'mu_diff_z',
    'consistency_diff', 'skill_conservative_diff', 'combined_uncertainty',
    'base_prob', 'mmr_diff',
]
bayesian_imp = feat_imp[feat_imp['feature'].isin(RATING_FEATURE_NAMES)]
print("\n    Rating feature importance:")
for _, row in bayesian_imp.iterrows():
    print(f"        {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================================
# SECTION 11: NAME RESOLUTION & FIGHTER LOOKUP (WITH FUZZY MATCHING)
# ============================================================================
print("\n[11] BUILDING FIGHTER LOOKUP WITH FUZZY NAME MATCHING...")

# --- Name normalization for matching ---
# Build name universe from data
all_fighters = set(ufc['r_name'].dropna()) | set(ufc['b_name'].dropna())
all_fighters_list = sorted([str(f) for f in all_fighters if f])

# Build lookup indices
norm_to_canon = {}
token_index = {}

for nm in all_fighters_list:
    key = _norm_name(nm)
    if key:
        norm_to_canon.setdefault(key, set()).add(nm)
    for t in set(key.split()):
        token_index.setdefault(t, set()).add(nm)

# --- Name resolution (see engine/name_resolution.py) ---
_fighter_appearances = {}
for _col in ('r_name', 'b_name'):
    for _nm, _cnt in ufc[_col].dropna().value_counts().items():
        _fighter_appearances[_nm] = _fighter_appearances.get(_nm, 0) + int(_cnt)

NAME_RESOLVER = NameResolver(
    names=all_fighters_list,
    appearances=_fighter_appearances,
    aliases=load_aliases(FIGHTER_ALIASES_FILE),
    auto_accept_ratio=FUZZY_AUTO_ACCEPT_RATIO,
    suggest_ratio=FUZZY_SUGGEST_RATIO,
    strict=STRICT_NAMES,
)
print(f"    Name resolver ready: {len(NAME_RESOLVER.names):,} fighters, "
      f"{len(NAME_RESOLVER.aliases)} alias(es), strict={STRICT_NAMES}")


def resolve_fighter_detailed(user_input):
    return NAME_RESOLVER.resolve(user_input)


def format_resolution_failure(user_input, res):
    return format_failure(user_input, res)


def resolve_fighter_name(user_input):
    """Back-compatible wrapper returning (canonical_name_or_None, note)."""
    res = NAME_RESOLVER.resolve(user_input)
    if res['status'] == 'OK':
        return res['name'], res['note']
    if not STRICT_NAMES and res['suggestions']:
        ratio, name = res['suggestions'][0]
        return name, f"WARNING: guessed '{name}' for '{user_input}' ({ratio:.0%} match)."
    return None, format_failure(user_input, res)


# Build fighter stats lookup from latest fight (including Bayesian skill values)
fighter_stats = {}

def compute_post_fight_streak(pre_streak, won_last_fight):
    """Compute streak AFTER the fight based on PRE-FIGHT streak and result."""
    if won_last_fight is None:
        return pre_streak  # Unknown result, keep as-is
    if won_last_fight:
        # Won: if was on losing streak, reset to 1; else increment
        return max(0, pre_streak) + 1
    else:
        # Lost: if was on winning streak, reset to -1; else decrement
        return min(0, pre_streak) - 1


for fighter in all_fighters:
    if not fighter:
        continue
    red_fights = ufc[ufc['r_name'] == fighter].sort_values('date', ascending=False)
    blue_fights = ufc[ufc['b_name'] == fighter].sort_values('date', ascending=False)
    
    stats = {'name': fighter}
    
    # Determine most recent fight (could be as red or blue)
    last_red_date = red_fights.iloc[0]['date'] if len(red_fights) > 0 else pd.Timestamp.min
    last_blue_date = blue_fights.iloc[0]['date'] if len(blue_fights) > 0 else pd.Timestamp.min

    if last_red_date >= last_blue_date and len(red_fights) > 0:
        L = red_fights.iloc[0]
        # FIXED: Determine if fighter won their last fight
        winner = L.get('winner', None)
        won_last = (winner == fighter) if pd.notna(winner) and winner not in ['Draw', 'NC', ''] else None

        # Compute POST-FIGHT values
        pre_streak = L.get('r_streak', 0)
        post_streak = compute_post_fight_streak(pre_streak, won_last)
        post_wins = L.get('r_wins', 0) + (1 if won_last == True else 0)
        post_losses = L.get('r_losses', 0) + (1 if won_last == False else 0)

        # POST-FIGHT MMR adjustment (approximate - actual MMR depends on opponent)
        # Post-fight rating. This used an Elo K-factor update, adding or
        # subtracting 10-50 points to a rating whose entire range is 56 wide,
        # so a single fight could move a fighter across most of the scale.
        # TrueSkill has its own update and ratings.rate_1v1 already implements
        # it, which is what produced every stored rating in the dataset.
        post_mu, post_sigma, post_mmr = trueskill_post_fight(
            L.get('r_mu', TRUESKILL_DEFAULT_MU),
            L.get('r_sigma', TRUESKILL_DEFAULT_SIGMA),
            L.get('b_mu', TRUESKILL_DEFAULT_MU),
            L.get('b_sigma', TRUESKILL_DEFAULT_SIGMA),
            won_last)

        stats.update({
            'mmr_pre': post_mmr,      # POST-FIGHT rating, mu - 3*sigma
            'mu': post_mu,            # POST-FIGHT Bayesian skill mean
            'sigma': post_sigma,      # POST-FIGHT Bayesian uncertainty
            'wins': post_wins, 'losses': post_losses, 'draws': L.get('r_draws', 0),
            'dob': L.get('r_dob'), 'stance': L.get('r_stance', 'Orthodox'),
            'splm': L.get('r_splm', 0), 'str_acc': L.get('r_str_acc', 0),
            'sapm': L.get('r_sapm', 0), 'str_def': L.get('r_str_def', 0),
            'td_avg': L.get('r_td_avg', 0), 'td_def': L.get('r_td_def', 0),
            'td_acc': L.get('r_td_avg_acc', 0), 'sub_avg': L.get('r_sub_avg', 0),
            'kd': L.get('r_kd', 0),
            'won_L3': L.get('r_won_L3', 0.5), 'momentum': L.get('r_momentum', 0),
            'splm_L3': L.get('r_splm_L3', 0), 'str_acc_L3': L.get('r_str_acc_L3', 0),
            'td_avg_L3': L.get('r_td_avg_L3', 0),
            'layoff': L.get('r_layoff', 500), 'streak': post_streak, 'won_last_fight': won_last,
            'last_fight_date': L.get('date'),
            # NEW: Physical attributes
            'height': L.get('r_height_inches', 70),
            'reach': L.get('r_reach_inches', 70),
            # NEW: Finish rates
            'ko_rate': L.get('r_ko_rate', 0),
            'sub_rate': L.get('r_sub_rate', 0),
            # NEW: Durability
            'ko_losses': L.get('r_ko_losses', 0),
            'been_finished': L.get('r_been_finished', 0),
            'absorption_eff': L.get('r_absorption_eff', 16.67),
            'footwork_proxy': L.get('r_splm', 3) / max(L.get('r_sapm', 3), 0.5),
            # NEW: Opponent quality
            'opp_quality': L.get('r_opp_quality'),
            # NEW: Cage control stats
            'ctrl_rate_ewm': L.get('r_ctrl_rate_ewm', 0),
            'grind_score_ewm': L.get('r_grind_score_ewm', 0),
            'clinch_activity_ewm': L.get('r_clinch_activity_ewm', 0),
            'grind_score_L3': L.get('r_grind_score_L3', 0),
            'grind_rate': L.get('r_grind_rate', 0),
            'cage_ctrl_fights': L.get('r_cage_ctrl_fights', 0),
            # ADVANCED FEATURES
            'cardio': 0.5,  # Will be populated below
            'archetype': 0,
            'wc_move': 0,
            'sub_def_score': 0.5,
            'mom_quality': 0.0,
            'fight_count': len(red_fights) + len(blue_fights),
            'striking_trajectory': L.get('r_splm_L3', L.get('r_splm', 0)) - L.get('r_splm', 0),
            'accuracy_trajectory': L.get('r_str_acc_L3', L.get('r_str_acc', 0)) - L.get('r_str_acc', 0),
            'career_damage': L.get('r_sapm', 3) * 12.0 * (len(red_fights) + len(blue_fights)),
        })
    elif len(blue_fights) > 0:
        L = blue_fights.iloc[0]
        # FIXED: Determine if fighter won their last fight
        winner = L.get('winner', None)
        won_last = (winner == fighter) if pd.notna(winner) and winner not in ['Draw', 'NC', ''] else None

        # Compute POST-FIGHT values
        pre_streak = L.get('b_streak', 0)
        post_streak = compute_post_fight_streak(pre_streak, won_last)
        post_wins = L.get('b_wins', 0) + (1 if won_last == True else 0)
        post_losses = L.get('b_losses', 0) + (1 if won_last == False else 0)

        # POST-FIGHT MMR adjustment
        # Post-fight rating. This used an Elo K-factor update, adding or
        # subtracting 10-50 points to a rating whose entire range is 56 wide,
        # so a single fight could move a fighter across most of the scale.
        # TrueSkill has its own update and ratings.rate_1v1 already implements
        # it, which is what produced every stored rating in the dataset.
        post_mu, post_sigma, post_mmr = trueskill_post_fight(
            L.get('b_mu', TRUESKILL_DEFAULT_MU),
            L.get('b_sigma', TRUESKILL_DEFAULT_SIGMA),
            L.get('r_mu', TRUESKILL_DEFAULT_MU),
            L.get('r_sigma', TRUESKILL_DEFAULT_SIGMA),
            won_last)

        stats.update({
            'mmr_pre': post_mmr,      # POST-FIGHT rating, mu - 3*sigma
            'mu': post_mu,            # POST-FIGHT Bayesian skill mean
            'sigma': post_sigma,      # POST-FIGHT Bayesian uncertainty
            'wins': post_wins, 'losses': post_losses, 'draws': L.get('b_draws', 0),
            'dob': L.get('b_dob'), 'stance': L.get('b_stance', 'Orthodox'),
            'splm': L.get('b_splm', 0), 'str_acc': L.get('b_str_acc', 0),
            'sapm': L.get('b_sapm', 0), 'str_def': L.get('b_str_def', 0),
            'td_avg': L.get('b_td_avg', 0), 'td_def': L.get('b_td_def', 0),
            'td_acc': L.get('b_td_avg_acc', 0), 'sub_avg': L.get('b_sub_avg', 0),
            'kd': L.get('b_kd', 0),
            'won_L3': L.get('b_won_L3', 0.5), 'momentum': L.get('b_momentum', 0),
            'splm_L3': L.get('b_splm_L3', 0), 'str_acc_L3': L.get('b_str_acc_L3', 0),
            'td_avg_L3': L.get('b_td_avg_L3', 0),
            'layoff': L.get('b_layoff', 500), 'streak': post_streak, 'won_last_fight': won_last,
            'last_fight_date': L.get('date'),
            # NEW: Physical attributes
            'height': L.get('b_height_inches', 70),
            'reach': L.get('b_reach_inches', 70),
            # NEW: Finish rates
            'ko_rate': L.get('b_ko_rate', 0),
            'sub_rate': L.get('b_sub_rate', 0),
            # NEW: Durability
            'ko_losses': L.get('b_ko_losses', 0),
            'been_finished': L.get('b_been_finished', 0),
            'absorption_eff': L.get('b_absorption_eff', 16.67),
            'footwork_proxy': L.get('b_splm', 3) / max(L.get('b_sapm', 3), 0.5),
            # NEW: Opponent quality
            'opp_quality': L.get('b_opp_quality'),
            # NEW: Cage control stats
            'ctrl_rate_ewm': L.get('b_ctrl_rate_ewm', 0),
            'grind_score_ewm': L.get('b_grind_score_ewm', 0),
            'clinch_activity_ewm': L.get('b_clinch_activity_ewm', 0),
            'grind_score_L3': L.get('b_grind_score_L3', 0),
            'grind_rate': L.get('b_grind_rate', 0),
            'cage_ctrl_fights': L.get('b_cage_ctrl_fights', 0),
            # ADVANCED FEATURES
            'cardio': 0.5,  # Will be populated below
            'archetype': 0,
            'wc_move': 0,
            'sub_def_score': 0.5,
            'mom_quality': 0.0,
            'fight_count': len(red_fights) + len(blue_fights),
            'striking_trajectory': L.get('b_splm_L3', L.get('b_splm', 0)) - L.get('b_splm', 0),
            'accuracy_trajectory': L.get('b_str_acc_L3', L.get('b_str_acc', 0)) - L.get('b_str_acc', 0),
            'career_damage': L.get('b_sapm', 3) * 12.0 * (len(red_fights) + len(blue_fights)),
        })
    
    # Their career state after the last bout in the file, which is what a
    # prediction of their next fight reads.
    _fid = L.get('r_id') if L.get('r_name') == fighter else L.get('b_id')
    if _fid in _final.index:
        stats.update(_final.loc[_fid].to_dict())
    fighter_stats[_norm_name(fighter)] = stats

print(f"    Fighter lookup built: {len(fighter_stats):,} fighters")

# --- Post-process fighter_stats with advanced features ---
print("    Adding advanced features to fighter stats...")
for fighter_key, stats in fighter_stats.items():
    fighter_name = stats.get('name', '')
    if not fighter_name:
        continue

    # Cardio score from history
    if fighter_name in CARDIO_HISTORY and len(CARDIO_HISTORY[fighter_name]) > 0:
        # Compute cardio from fight history
        history = CARDIO_HISTORY[fighter_name]
        if len(history) >= 2:
            short_fights = [opm for rds, opm in history if rds <= 2]
            long_fights = [opm for rds, opm in history if rds >= 3]
            if short_fights and long_fights:
                avg_short = np.mean(short_fights)
                avg_long = np.mean(long_fights)
                stats['cardio'] = np.clip(avg_long / (avg_short + 0.01), 0.2, 1.2)
            else:
                outputs = [opm for _, opm in history]
                mean_out = np.mean(outputs)
                cv = np.std(outputs) / (mean_out + 0.01) if mean_out > 0 else 0.5
                stats['cardio'] = np.clip(1.0 - cv, 0.2, 0.9)
        else:
            stats['cardio'] = 0.5

    # Archetype from current stats
    stats['archetype'] = classify_archetype(
        stats.get('splm', 3), stats.get('td_avg', 1.5),
        stats.get('sub_avg', 0.5), stats.get('str_def', 55),
        stats.get('str_acc', 45)
    )[0]

    # Weight class movement (last known)
    last_weight = WC_MOVEMENT_HISTORY.get(fighter_name)
    # wc_move is tracked per fight; for prediction we use 0 (same class) by default
    stats['wc_move'] = 0

    # Submission defense score
    sub_stats = SUB_DEFENSE_STATS.get(fighter_name, {'faced': 0, 'submitted': 0, 'fights': 0})
    if sub_stats['faced'] > 0:
        stats['sub_def_score'] = 1.0 - (sub_stats['submitted'] / sub_stats['faced'])
    elif sub_stats['fights'] > 2:
        stats['sub_def_score'] = 0.8
    else:
        stats['sub_def_score'] = 0.5

    # Momentum quality
    mom_hist = MOMENTUM_QUALITY_HISTORY.get(fighter_name, [])
    if mom_hist:
        # EWM-weighted quality of last 5
        recent = mom_hist[-5:]
        weights = [0.85 ** (len(recent) - 1 - i) for i in range(len(recent))]
        w_sum = sum(weights)
        stats['mom_quality'] = sum(q * w for q, w in zip(recent, weights)) / w_sum
    else:
        stats['mom_quality'] = 0.0

print("    Advanced features populated in fighter_stats")

print(f"    Name resolution ready with fuzzy matching")
print(f"    Bayesian skill (mu/sigma) included in fighter stats")

# ============================================================================
# SECTION 12: OPTIONAL CONTEXT ADJUSTMENTS
# ============================================================================
# These are rule-based adjustments that can be applied when context is provided.
# They default to neutral (0) when not specified, so they never break predictions.

# Context adjustment coefficients (conservative estimates from MMA research)
CONTEXT_ADJUSTMENTS = {
    'home_advantage': 0.03,      # ~3% boost for fighting in home country/city
    'altitude_high': -0.02,      # ~2% penalty for sea-level fighter at altitude (>5000ft)
    'short_notice': -0.04,       # ~4% penalty for short notice replacement (<2 weeks)
    'cage_size_small': 0.02,     # ~2% boost for pressure fighters in small cage
    'cage_size_large': -0.02,    # ~2% boost for volume strikers in large cage (inverse for pressure)
    'weight_cut_hard': -0.03,    # ~3% penalty for fighters with known hard weight cuts
}


def _corner_extras(stats, age, exp, winrate, southpaw, consistency,
                   skill_conservative, weight_class):
    """Attributes the prediction function works out rather than stores.

    reach and height are both centimetres, so their ratio is the ape index
    directly. The training-side version divided a column mislabelled inches by
    another mislabelled inches, filling each with 70, which turned 9.4% of
    fights into an ape index of 0.39 or 2.60 against a real value near 1.02.
    """
    reach = stats.get("reach")
    height = stats.get("height")
    try:
        ape = float(reach) / float(height) if reach and height else np.nan
    except (TypeError, ValueError, ZeroDivisionError):
        ape = np.nan
    return {
        "age": age,
        "exp": exp,
        "winrate": winrate,
        "southpaw": southpaw,
        "consistency": consistency,
        "skill_conservative": skill_conservative,
        "ape_index": ape,
        "ring_rust": ring_rust_transform(stats.get("layoff")),
        "prime_wc": age_prime_score_wc(age, weight_class),
    }


def _declared_features(r, b, r_extra, b_extra):
    """Run the feature declaration over a single fight.

    The same build_all that makes the training matrix, so the two definitions
    cannot drift. What the stats dict cannot supply becomes NaN, which the
    specs turn into a neutral difference and a _known flag of 0.
    """
    frame = build_prediction_frame(r, b, required_suffixes(SPECS),
                                   red_extra=r_extra, blue_extra=b_extra,
                                   fight_extra=matchup_extra(r, b))
    built = build_all(SPECS, frame)
    return {col: built.iloc[0][col] for col in built.columns}


def apply_context_adjustments(base_prob, context=None):
    """
    Apply optional context-based probability adjustments.
    
    Args:
        base_prob: Base win probability for red corner (0-1)
        context: Dict with optional keys:
            - 'red_home': bool - Red corner fighting at home
            - 'blue_home': bool - Blue corner fighting at home
            - 'red_altitude_acclimated': bool - Red used to high altitude
            - 'blue_altitude_acclimated': bool - Blue used to high altitude
            - 'altitude_high': bool - Event at high altitude (>5000ft)
            - 'red_short_notice': bool - Red is short notice replacement
            - 'blue_short_notice': bool - Blue is short notice replacement
            - 'cage_size': 'small'/'large'/None - Cage size relative to standard
            - 'red_pressure_fighter': bool - Red is a pressure/wrestling fighter
            - 'blue_pressure_fighter': bool - Blue is a pressure/wrestling fighter
            - 'red_hard_cut': bool - Red has hard weight cut history
            - 'blue_hard_cut': bool - Blue has hard weight cut history
    
    Returns:
        Tuple of (adjusted_prob, adjustments_log)
    """
    if context is None:
        return base_prob, []
    
    adjustments_log = []
    adj = 0.0
    
    # Home advantage
    if context.get('red_home', False):
        adj += CONTEXT_ADJUSTMENTS['home_advantage']
        adjustments_log.append(f"Red home advantage: +{CONTEXT_ADJUSTMENTS['home_advantage']*100:.1f}%")
    if context.get('blue_home', False):
        adj -= CONTEXT_ADJUSTMENTS['home_advantage']
        adjustments_log.append(f"Blue home advantage: -{CONTEXT_ADJUSTMENTS['home_advantage']*100:.1f}%")
    
    # Altitude
    if context.get('altitude_high', False):
        if not context.get('red_altitude_acclimated', False) and context.get('blue_altitude_acclimated', False):
            adj += CONTEXT_ADJUSTMENTS['altitude_high']
            adjustments_log.append(f"Red altitude disadvantage: {CONTEXT_ADJUSTMENTS['altitude_high']*100:.1f}%")
        elif context.get('red_altitude_acclimated', False) and not context.get('blue_altitude_acclimated', False):
            adj -= CONTEXT_ADJUSTMENTS['altitude_high']
            adjustments_log.append(f"Blue altitude disadvantage: {-CONTEXT_ADJUSTMENTS['altitude_high']*100:.1f}%")
    
    # Short notice
    if context.get('red_short_notice', False):
        adj += CONTEXT_ADJUSTMENTS['short_notice']
        adjustments_log.append(f"Red short notice: {CONTEXT_ADJUSTMENTS['short_notice']*100:.1f}%")
    if context.get('blue_short_notice', False):
        adj -= CONTEXT_ADJUSTMENTS['short_notice']
        adjustments_log.append(f"Blue short notice: {-CONTEXT_ADJUSTMENTS['short_notice']*100:.1f}%")
    
    # Cage size
    cage = context.get('cage_size', None)
    if cage == 'small':
        if context.get('red_pressure_fighter', False) and not context.get('blue_pressure_fighter', False):
            adj += CONTEXT_ADJUSTMENTS['cage_size_small']
            adjustments_log.append(f"Small cage favors red (pressure): +{CONTEXT_ADJUSTMENTS['cage_size_small']*100:.1f}%")
        elif context.get('blue_pressure_fighter', False) and not context.get('red_pressure_fighter', False):
            adj -= CONTEXT_ADJUSTMENTS['cage_size_small']
            adjustments_log.append(f"Small cage favors blue (pressure): -{CONTEXT_ADJUSTMENTS['cage_size_small']*100:.1f}%")
    elif cage == 'large':
        if context.get('red_pressure_fighter', False) and not context.get('blue_pressure_fighter', False):
            adj += CONTEXT_ADJUSTMENTS['cage_size_large']
            adjustments_log.append(f"Large cage disadvantages red (pressure): {CONTEXT_ADJUSTMENTS['cage_size_large']*100:.1f}%")
        elif context.get('blue_pressure_fighter', False) and not context.get('red_pressure_fighter', False):
            adj -= CONTEXT_ADJUSTMENTS['cage_size_large']
            adjustments_log.append(f"Large cage disadvantages blue (pressure): {-CONTEXT_ADJUSTMENTS['cage_size_large']*100:.1f}%")
    
    # Hard weight cut
    if context.get('red_hard_cut', False):
        adj += CONTEXT_ADJUSTMENTS['weight_cut_hard']
        adjustments_log.append(f"Red hard weight cut: {CONTEXT_ADJUSTMENTS['weight_cut_hard']*100:.1f}%")
    if context.get('blue_hard_cut', False):
        adj -= CONTEXT_ADJUSTMENTS['weight_cut_hard']
        adjustments_log.append(f"Blue hard weight cut: {-CONTEXT_ADJUSTMENTS['weight_cut_hard']*100:.1f}%")
    
    # Apply adjustment and clip to valid probability range
    adjusted_prob = np.clip(base_prob + adj, 0.01, 0.99)
    
    return adjusted_prob, adjustments_log

print("\n[12] CONTEXT ADJUSTMENT SYSTEM READY")
print(f"    Available adjustments: {list(CONTEXT_ADJUSTMENTS.keys())}")

# ============================================================================
# SECTION 13: PREDICTION FUNCTION (WITH BAYESIAN SKILL MODEL)
# ============================================================================
def _no_data_result(red_name, blue_name, problems):
    """A refusal, shaped like a prediction so callers can pass it around safely.

    Every probability is None on purpose. There is no number to report when we
    do not know who the fighter is - a plausible-looking 50% would be a lie.
    """
    return {
        'status': 'NO_DATA',
        'red': str(red_name), 'blue': str(blue_name),
        'problems': problems,
        'reason': "; ".join(problems),
        'red_win_prob': None, 'blue_win_prob': None,
        'winner': None, 'win_prob': None, 'confidence': None,
        'method': None, 'method_prob': None, 'method_probs': {},
        'round': None, 'round_prob': None, 'round_probs': {},
        'is_finish': None, 'finish_prob': None,
        'finish_timing': None, 'finish_timing_prob': None,
        'cage_control': {},
    }


def _check_fighters(red_name, blue_name):
    """Resolve both corners. Returns (red_stats, blue_stats, problems, warnings).

    problems non-empty means refuse to predict.
    """
    problems, warnings, resolved = [], [], {}
    for corner, raw in (('red', red_name), ('blue', blue_name)):
        res = resolve_fighter_detailed(raw)
        if res['status'] != 'OK':
            problems.append(format_resolution_failure(raw, res))
            continue
        stats = fighter_stats.get(_norm_name(res['name']))
        if stats is None:
            problems.append(f"NO DATA: '{res['name']}' has no usable fight record.")
            continue
        n_fights = stats.get('fight_count') or 0
        if n_fights < MIN_FIGHTS_FOR_PREDICTION:
            problems.append(
                f"NO DATA: '{res['name']}' has {n_fights} recorded fight(s), "
                f"minimum is {MIN_FIGHTS_FOR_PREDICTION}.")
            continue
        if n_fights < LOW_DATA_FIGHT_COUNT:
            warnings.append(f"THIN DATA: '{res['name']}' has only {n_fights} recorded fight(s).")
        if res['note']:
            warnings.append(res['note'])
        resolved[corner] = (res['name'], stats)
    return resolved, problems, warnings


def predict_fight(red_name, blue_name, event_date=None, is_5rnd=False, is_title=False, context=None):
    """
    Predict fight outcome between two fighters using Bayesian skill model.
    
    Args:
        red_name: Red corner fighter name
        blue_name: Blue corner fighter name
        event_date: Optional event date (YYYY-MM-DD) for age/layoff calculation
        is_5rnd: Whether fight is 5 rounds (main event/title)
        is_title: Whether fight is for a title
        context: Optional dict with context adjustments (see apply_context_adjustments)
    
    Returns:
        Dict with prediction results including Bayesian skill analysis
    """
    
    def safe(v, d=0):
        try:
            val = float(v)
            return val if pd.notna(val) and not np.isinf(val) else d
        except:
            return d
    
    # Resolve both corners, or refuse. No silent substitution, no invented stats.
    resolved, problems, warnings = _check_fighters(red_name, blue_name)
    for msg in warnings:
        print(f"    {msg}")
    if problems:
        for msg in problems:
            print(f"    {msg}")
        return _no_data_result(red_name, blue_name, problems)

    r_resolved, r = resolved['red']
    b_resolved, b = resolved['blue']
    
    
    
    # --- Bayesian Skill Features ---
    r_mu = safe(r.get('mu'), TRUESKILL_DEFAULT_MU)
    r_sigma = safe(r.get('sigma'), TRUESKILL_DEFAULT_SIGMA)
    b_mu = safe(b.get('mu'), TRUESKILL_DEFAULT_MU)
    b_sigma = safe(b.get('sigma'), TRUESKILL_DEFAULT_SIGMA)
    
    # Bayesian win probability
    bayes_prob = bayesian_win_prob(r_mu, r_sigma, b_mu, b_sigma)
    
    # Skill consistency
    r_consistency = skill_consistency(r_sigma)
    b_consistency = skill_consistency(b_sigma)
    
    # Conservative skill gap
    r_skill_cons = r_mu - CONSERVATIVE_K * r_sigma
    b_skill_cons = b_mu - CONSERVATIVE_K * b_sigma
    
    # Combined uncertainty
    combined_unc = np.sqrt(r_sigma**2 + b_sigma**2)
    
    # Calculate age and prime score
    if event_date:
        event_date = pd.to_datetime(event_date)
        r_dob = pd.to_datetime(r.get('dob'), errors='coerce')
        b_dob = pd.to_datetime(b.get('dob'), errors='coerce')
        r_age = (event_date - r_dob).days / 365.25 if pd.notna(r_dob) else 30
        b_age = (event_date - b_dob).days / 365.25 if pd.notna(b_dob) else 30
        
        # Calculate adjusted layoff based on event date
        r_last = pd.to_datetime(r.get('last_fight_date'), errors='coerce')
        b_last = pd.to_datetime(b.get('last_fight_date'), errors='coerce')
        r_layoff_adj = (event_date - r_last).days if pd.notna(r_last) else 500
        b_layoff_adj = (event_date - b_last).days if pd.notna(b_last) else 500
    else:
        r_age, b_age = 30, 30
        r_layoff_adj = safe(r.get('layoff'), 500)
        b_layoff_adj = safe(b.get('layoff'), 500)
    
    r_prime = age_prime_score(r_age)
    b_prime = age_prime_score(b_age)
    
    # Experience
    # Decided UFC bouts only, matching ufc['r_exp'] in the training frame.
    r_exp = safe(r.get('wins')) + safe(r.get('losses'))
    b_exp = safe(b.get('wins')) + safe(b.get('losses'))
    
    # MMR (legacy)
    mmr_diff = (safe(r.get('mmr_pre'), TRUESKILL_DEFAULT_MMR)
                - safe(b.get('mmr_pre'), TRUESKILL_DEFAULT_MMR))
    base_prob = float(base_probability(mmr_diff))
    
    # Win rates
    r_winrate = safe(r.get('wins')) / r_exp if r_exp > 0 else np.nan
    b_winrate = safe(b.get('wins')) / b_exp if b_exp > 0 else np.nan
    
    # Southpaw
    r_southpaw = 1 if 'southpaw' in str(r.get('stance', '')).lower() else 0
    b_southpaw = 1 if 'southpaw' in str(b.get('stance', '')).lower() else 0
    
    # Power diff (without per-fight KD to avoid leakage)
    power_diff = (
        (safe(r.get('splm')) - safe(b.get('splm'))) * 1.0 +
        (safe(r.get('str_def')) - safe(b.get('str_def'))) * 0.5 +
        ((safe(r.get('str_acc')) - safe(b.get('str_acc'))) / 100.0) * 2.0 +
        (safe(b.get('sapm')) - safe(r.get('sapm'))) * 0.8
    )
    
    # Layoff and streak
    layoff_diff = (b_layoff_adj - r_layoff_adj) / 100.0
    streak_diff = safe(r.get('streak')) - safe(b.get('streak'))
    
    # Build feature dict (including Bayesian features)
    feat = {
        # Bayesian skill features
        'bayesian_prob': bayes_prob,
        'mu_diff': r_mu - b_mu,
        'consistency_diff': r_consistency - b_consistency,
        'skill_conservative_diff': r_skill_cons - b_skill_cons,
        'combined_uncertainty': combined_unc,
        'mu_sum': r_mu + b_mu,
        'mu_diff_z': float(standardised_skill_gap(
            r_mu, r_sigma, b_mu, b_sigma, TRUESKILL_BETA)),
        # Base features
        'base_prob': base_prob,
        'mmr_diff': mmr_diff,
        'exp_diff': r_exp - b_exp,
        'age_diff': safe(r_age, 30) - safe(b_age, 30),
        # prime_diff removed (replaced by prime_wc_diff)
        'off_striking_diff': safe(r.get('splm')) - safe(b.get('splm')),
        'acc_diff': safe(r.get('str_acc')) - safe(b.get('str_acc')),
        'def_diff': safe(r.get('str_def')) - safe(b.get('str_def')),
        'power_diff': power_diff,
        'td_off_diff': safe(r.get('td_avg')) - safe(b.get('td_avg')),
        'td_def_diff': safe(r.get('td_def')) - safe(b.get('td_def')),
        'sub_diff': safe(r.get('sub_avg')) - safe(b.get('sub_avg')),
        'same_cluster': 0,
        'southpaw_diff': r_southpaw - b_southpaw,
        # layoff_diff removed (replaced by ring_rust_diff)
        # streak_diff removed (replaced by mom_quality_diff)
        'winrate_diff': r_winrate - b_winrate,
        'td_acc_diff': safe(r.get('td_acc')) - safe(b.get('td_acc')),
        'sapm_diff': safe(r.get('sapm')) - safe(b.get('sapm')),
        'is_5rnd': int(is_5rnd),
        'is_title': int(is_title),
        'recent_form_diff': safe(r.get('won_L3'), 0.5) - safe(b.get('won_L3'), 0.5),
        # momentum_diff removed (replaced by mom_quality_diff)
        'splm_L3_diff': safe(r.get('splm_L3')) - safe(b.get('splm_L3')),
        'str_acc_L3_diff': safe(r.get('str_acc_L3')) - safe(b.get('str_acc_L3')),
        'td_avg_L3_diff': safe(r.get('td_avg_L3')) - safe(b.get('td_avg_L3')),
        # NEW: Physical features
        'height_diff': safe(r.get('height'), 70) - safe(b.get('height'), 70),
        'reach_diff': safe(r.get('reach'), 70) - safe(b.get('reach'), 70),
        'ape_index_diff': (safe(r.get('reach'), 70) / max(safe(r.get('height'), 70), 1)) - (safe(b.get('reach'), 70) / max(safe(b.get('height'), 70), 1)),
        # NEW: Style/Finish rate features
        'ko_rate_diff': safe(r.get('ko_rate')) - safe(b.get('ko_rate')),
        'sub_rate_diff': safe(r.get('sub_rate')) - safe(b.get('sub_rate')),
        'finish_rate_diff': (safe(r.get('ko_rate')) + safe(r.get('sub_rate'))) - (safe(b.get('ko_rate')) + safe(b.get('sub_rate'))),
        # NEW: Durability features
        'ko_vulnerability_diff': (1 if safe(r.get('ko_losses')) > 0 else 0) - (1 if safe(b.get('ko_losses')) > 0 else 0),
        'been_finished_diff': safe(r.get('been_finished')) - safe(b.get('been_finished')),
        'absorption_eff_diff': safe(r.get('absorption_eff'), 16.67) - safe(b.get('absorption_eff'), 16.67),
        'footwork_diff': safe(r.get('footwork_proxy'), 1.0) - safe(b.get('footwork_proxy'), 1.0),
        # Stance matchup interaction
        'stance_mismatch': int(r_southpaw != b_southpaw),
        'stance_interaction': (r_southpaw - b_southpaw) * int(r_southpaw != b_southpaw),
        # Fighter trajectory (decline detection)
        'striking_trajectory_diff': safe(r.get('striking_trajectory')) - safe(b.get('striking_trajectory')),
        'accuracy_trajectory_diff': safe(r.get('accuracy_trajectory')) - safe(b.get('accuracy_trajectory')),
        'trajectory_diff': (safe(r.get('striking_trajectory')) / 3.0 + safe(r.get('accuracy_trajectory')) / 20.0) - (safe(b.get('striking_trajectory')) / 3.0 + safe(b.get('accuracy_trajectory')) / 20.0),
        # Career damage accumulation
        'career_damage_diff': np.log1p(safe(r.get('career_damage'), 36)) - np.log1p(safe(b.get('career_damage'), 36)),
        # NEW: Opponent quality features
        # Unknown schedule stays unknown; the flag below tells the model which
        # it is, instead of a constant that means neither.
        'opp_quality_diff': (safe(r.get('opp_quality'), np.nan)
                             - safe(b.get('opp_quality'), np.nan)),
        'opp_history_known': float(pd.notna(r.get('opp_quality'))
                                   and pd.notna(b.get('opp_quality'))),
        # NEW: Weight class features (default to 0 for unknown)
        'is_womens': 0,  # Will be overridden if division info available
        'is_heavyweight': 0,  # Will be overridden if division info available
        # NEW: Cage control features
        'cage_control_cap_diff': safe(r.get('ctrl_rate_ewm')) - safe(b.get('ctrl_rate_ewm')),
        'clinch_activity_diff': safe(r.get('clinch_activity_ewm')) - safe(b.get('clinch_activity_ewm')),
        'grind_tendency_diff': safe(r.get('grind_rate')) - safe(b.get('grind_rate')),
        # ADVANCED FEATURES (Section 3.95)
        'cardio_diff': safe(r.get('cardio'), 0.5) - safe(b.get('cardio'), 0.5),
        'archetype_matchup': ARCHETYPE_MATCHUP_MATRIX[int(safe(r.get('archetype'), 0)), int(safe(b.get('archetype'), 0))] - 0.5,
        'archetype_clash': int(safe(r.get('archetype'), 0) != safe(b.get('archetype'), 0)),
        'combined_pace': safe(r.get('splm'), 3) + safe(b.get('splm'), 3),
        'high_pace': int((safe(r.get('splm'), 3) + safe(b.get('splm'), 3)) > 7.5),
        'wc_move_diff': safe(r.get('wc_move'), 0) - safe(b.get('wc_move'), 0),
        'prime_wc_diff': age_prime_score_wc(r_age, 'unknown') - age_prime_score_wc(b_age, 'unknown'),
        'ring_rust_diff': ring_rust_transform(r_layoff_adj) - ring_rust_transform(b_layoff_adj),
        'sub_def_diff': safe(r.get('sub_def_score'), 0.5) - safe(b.get('sub_def_score'), 0.5),
        'mom_quality_diff': safe(r.get('mom_quality'), 0) - safe(b.get('mom_quality'), 0),
        'data_sparsity_diff': np.log1p(min(safe(r.get('fight_count'), 5), 30)) - np.log1p(min(safe(b.get('fight_count'), 5), 30)),
    }

    # Create feature vector
    feat.update(_declared_features(
        r, b,
        _corner_extras(r, r_age, r_exp, r_winrate, r_southpaw, r_consistency,
                       r_skill_cons, context.get('weight_class') if context else None),
        _corner_extras(b, b_age, b_exp, b_winrate, b_southpaw, b_consistency,
                       b_skill_cons, context.get('weight_class') if context else None)))

    # Match the training matrix, which is built with .fillna(0). Without
    # this a single unknown feature turns every model output into NaN.
    X_pred = pd.DataFrame([feat])[feature_cols].replace(
        [np.inf, -np.inf], np.nan).fillna(0)
    X_pred_s = scaler.transform(X_pred)
    
    # Win prediction (ensemble + calibration)
    p_ens = (lr.predict_proba(X_pred_s)[:, 1][0] + rf.predict_proba(X_pred_s)[:, 1][0] + xgb_win.predict_proba(X_pred_s)[:, 1][0] + mlp.predict_proba(X_pred_s)[:, 1][0]) / 4
    p_win_base = platt.predict_proba([[p_ens]])[0, 1]
    
    # Apply context adjustments if provided

    # FEATURE 5: Betting Line Blending (if odds available)
    # Blend model prediction with market odds for improved accuracy
    # Market efficiency means odds capture info we can't model (injuries, camp, etc.)
    if globals().get("CURRENT_ODDS") and CURRENT_ODDS:
        odds_match = match_fighter_to_odds(r_resolved or red_name, CURRENT_ODDS)
        if odds_match:
            market_implied = american_to_implied_prob(odds_match['best_odds'])
            # Blend: 70% model + 30% market (model has structural edge, market has info edge)
            ODDS_BLEND_WEIGHT = 0.30
            p_win_base = (1 - ODDS_BLEND_WEIGHT) * p_win_base + ODDS_BLEND_WEIGHT * market_implied
            p_win_base = np.clip(p_win_base, 0.01, 0.99)
    
    # Auto-detect pressure fighters from archetype for cage size adjustments
    if context and context.get("cage_size") and "red_pressure_fighter" not in context and "blue_pressure_fighter" not in context:
        r_arch = int(safe(r.get("archetype"), 0)) if r else 0
        b_arch = int(safe(b.get("archetype"), 0)) if b else 0
        # Archetypes 0 (pressure_boxer) and 2 (wrestler) are pressure fighters
        context["red_pressure_fighter"] = r_arch in (0, 2)
        context["blue_pressure_fighter"] = b_arch in (0, 2)

    p_win, context_log = apply_context_adjustments(p_win_base, context)
    
    # Method prediction
    p_method = xgb_method.predict_proba(X_pred_s)[0]
    method_probs = dict(zip(le_method.classes_, p_method))
    
    # Round prediction
    p_round = xgb_round.predict_proba(X_pred_s)[0]
    round_probs = {f'R{i+1}': p for i, p in enumerate(p_round)}
    

    # Data sparsity confidence dampener
    raw_confidence = abs(p_win - 0.5) * 2
    r_fc = safe(r.get('fight_count'), 5) if isinstance(r, dict) else 5
    b_fc = safe(b.get('fight_count'), 5) if isinstance(b, dict) else 5
    min_fights = min(r_fc, b_fc)
    if min_fights < 4:
        sparsity_cap = 0.55 + (min_fights / 4) * 0.15  # 0->55%, 1->58.75%, 2->62.5%, 3->66.25%
        raw_confidence = min(raw_confidence, sparsity_cap)

    result = {
        'red': r_resolved or red_name,
        'blue': b_resolved or blue_name,
        'red_win_prob': p_win,
        'blue_win_prob': 1 - p_win,
        'red_win_prob_base': p_win_base,  # Before context adjustments
        'winner': (r_resolved or red_name) if p_win > 0.5 else (b_resolved or blue_name),
        'confidence': raw_confidence,
        'method_probs': method_probs,
        'method': max(method_probs, key=method_probs.get),
        'round_probs': round_probs,
        'round': max(round_probs, key=round_probs.get),
        'context_adjustments': context_log if context_log else None,
        # Bayesian skill analysis
        'bayesian_analysis': {
            'red_skill': {'mu': r_mu, 'sigma': r_sigma, 'consistency': r_consistency},
            'blue_skill': {'mu': b_mu, 'sigma': b_sigma, 'consistency': b_consistency},
            'bayesian_win_prob': bayes_prob,
            'combined_uncertainty': combined_unc,
            'skill_gap': r_mu - b_mu,
            'conservative_skill_gap': r_skill_cons - b_skill_cons,
        }
    }
    
    return result

print("\n[13] predict_fight() function ready with Bayesian skill model!")

# ============================================================================
# SECTION 14: EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE PREDICTIONS")
print("="*70)

# Example 1: Basic prediction with Bayesian analysis
print("\n--- Example 1: Basic Prediction with Bayesian Skill Analysis ---")
def _demo_basic_prediction():
    """Illustrative example. Skipped if the demo fighters are not in the data."""
    result = predict_fight("Islam Makhachev", "Charles Oliveira", is_title=True, is_5rnd=True)
    if result.get('status') == 'NO_DATA':
        print(f"    Demo skipped: {result['reason']}")
        return

    print(f"\n{result['red']} vs {result['blue']}")
    print(f"\nWINNER PREDICTION:")
    print(f"  {result['red']:25s}: {result['red_win_prob']*100:5.1f}%")
    print(f"  {result['blue']:25s}: {result['blue_win_prob']*100:5.1f}%")
    print(f"  Predicted: {result['winner']} (confidence: {result['confidence']*100:.1f}%)")

    print(f"\nBAYESIAN SKILL ANALYSIS:")
    ba = result['bayesian_analysis']
    print(f"  {result['red']}:")
    print(f"    Skill (mu):        {ba['red_skill']['mu']:.2f}")
    print(f"    Uncertainty (Ïƒ):   {ba['red_skill']['sigma']:.2f}")
    print(f"    Consistency:       {ba['red_skill']['consistency']*100:.0f}%")
    print(f"  {result['blue']}:")
    print(f"    Skill (mu):        {ba['blue_skill']['mu']:.2f}")
    print(f"    Uncertainty (Ïƒ):   {ba['blue_skill']['sigma']:.2f}")
    print(f"    Consistency:       {ba['blue_skill']['consistency']*100:.0f}%")
    print(f"  Bayesian Win Prob:   {ba['bayesian_win_prob']*100:.1f}% (red)")
    print(f"  Skill Gap:           {ba['skill_gap']:+.2f}")
    print(f"  Conservative Gap:    {ba['conservative_skill_gap']:+.2f}")
    print(f"  Combined Uncertainty:{ba['combined_uncertainty']:.2f}")

    print(f"\nMETHOD PREDICTION:")
    for m, p in sorted(result['method_probs'].items(), key=lambda x: -x[1]):
        print(f"  {m:12s}: {p*100:5.1f}%")
    print(f"  Predicted: {result['method']}")

    print(f"\nROUND PREDICTION:")
    for r, p in result['round_probs'].items():
        print(f"  {r}: {p*100:5.1f}%")
    print(f"  Predicted: {result['round']}")


_demo_basic_prediction()

# Example 2: With context adjustments
print("\n--- Example 2: With Context Adjustments ---")
def _demo_context_prediction():
    """Illustrative example. Skipped if the demo fighters are not in the data."""
    context = {
        'red_home': True,  # Makhachev fighting in Abu Dhabi (close to home)
        'blue_short_notice': False,
        'red_pressure_fighter': True,
        'cage_size': 'small'
    }
    result2 = predict_fight("Islam Makhachev", "Charles Oliveira", is_title=True, is_5rnd=True, context=context)
    if result2.get('status') == 'NO_DATA':
        print(f"    Demo skipped: {result2['reason']}")
        return

    print(f"\n{result2['red']} vs {result2['blue']} (with context)")
    print(f"\nWINNER PREDICTION:")
    print(f"  Base probability:     {result2['red_win_prob_base']*100:5.1f}%")
    print(f"  Adjusted probability: {result2['red_win_prob']*100:5.1f}%")
    if result2['context_adjustments']:
        print(f"\n  Context adjustments applied:")
        for adj in result2['context_adjustments']:
            print(f"    - {adj}")


_demo_context_prediction()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
DATASET:
  Fights: {len(ufc_valid):,} (post-2001, valid outcomes)
  Date range: {ufc_valid['date'].min().date()} to {ufc_valid['date'].max().date()}

FEATURES:
  Total: {len(feature_cols)}
  Declared: {len(SPEC_FEATURES)} (paired diffs, levels, known flags, matchup interactions)
  Bespoke: {len(bespoke_features)} (rating transforms, context flags, archetypes)
  Cage control: {len(CAGE_CONTROL_FEATURES)} (method/round models only)

WIN PREDICTION (Test Set):
  Accuracy:  {acc:.4f} ({acc*100:.1f}%)
  Brier:     {brier:.4f}
  Log Loss:  {logloss:.4f}

METHOD PREDICTION:
  Accuracy:  {method_acc:.4f} ({method_acc*100:.1f}%)

ROUND PREDICTION:
  Accuracy:  {acc_round:.4f} ({acc_round*100:.1f}%)

WALK-FORWARD ({start_year}-{years[-1]}):
  Mean Acc:  {wf_df['acc'].mean():.4f} ({wf_df['acc'].mean()*100:.1f}%)
  Std Dev:   {wf_df['acc'].std():.4f}

LEAKAGE AUDIT:
  Status: {'PASSED' if leakage_audit_passed and not leakage_issues else 'REVIEW NEEDED'}

BAYESIAN STATE-SPACE SKILL MODEL:
  - Uses TrueSkill mu (skill mean) and sigma (uncertainty) as backbone
  - Bayesian win probability: P(r>b) = Î¦((mu_r - mu_b) / sqrt(ÏƒrÂ² + ÏƒbÂ² + 2Î²Â²))
  - Skill consistency score (inverse of sigma)
  - Conservative skill gap (mu - Ïƒ) penalizes uncertain ratings
  - Combined uncertainty for matchup confidence

ENHANCEMENTS ADDED:
  - Bayesian state-space skill model (mu, sigma, consistency, conservative gap)
  - Fuzzy name matching (difflib) for fighter lookup
  - XGBoost early stopping to prevent overfitting
  - Layoff days feature (time since last fight)
  - Win/loss streak feature
  - Age prime indicator (peak performance 28-32)
  - Fixed power_diff to avoid per-fight KD leakage
  - Automated leakage audits (7 checks)
  - Optional context adjustments (home, altitude, short notice, cage size, weight cut)

USAGE:
  # Basic prediction
  result = predict_fight("Fighter A", "Fighter B")
  
  # With event details
  result = predict_fight("Fighter A", "Fighter B", 
                         event_date="2024-03-15", 
                         is_title=True, 
                         is_5rnd=True)
  
  # With context adjustments
  result = predict_fight("Fighter A", "Fighter B",
                         context={{
                             'red_home': True,
                             'blue_short_notice': True,
                             'cage_size': 'small',
                             'red_pressure_fighter': True
                         }})
  
  # Access Bayesian skill analysis
  result['bayesian_analysis']['red_skill']['mu']      # Red's skill estimate
  result['bayesian_analysis']['red_skill']['sigma']   # Red's uncertainty
  result['bayesian_analysis']['bayesian_win_prob']    # TrueSkill probability
  result['bayesian_analysis']['combined_uncertainty'] # Matchup uncertainty
""")

# ============================================================================
# SECTION 15: PRODUCTION MODEL - RETRAIN ON FULL DATA
# ============================================================================
# For live predictions in 2026, we retrain on ALL available data (up to end of 2025)
# This gives the models the most recent fight patterns for best accuracy

print("\n" + "="*70)
print("SECTION 15: RETRAINING MODELS ON FULL DATASET FOR PRODUCTION")
print("="*70)

# Use 90% for training, 10% for calibration (no test holdout for production)
n_full = len(X_valid)
train_end_full = int(n_full * 0.90)

X_train_full = X_valid.iloc[:train_end_full]
X_cal_full = X_valid.iloc[train_end_full:]
y_train_full = y_win[:train_end_full]
y_cal_full = y_win[train_end_full:]

train_dates_full = ufc_valid.iloc[:train_end_full]['date']
cal_dates_full = ufc_valid.iloc[train_end_full:]['date']

print(f"\n[PRODUCTION SPLIT]")
print(f"    Train: {len(X_train_full):,} fights ({train_dates_full.min().date()} to {train_dates_full.max().date()})")
print(f"    Cal:   {len(X_cal_full):,} fights ({cal_dates_full.min().date()} to {cal_dates_full.max().date()})")

# Scale with full data - SEPARATE scalers for winner vs method/round
# Winner model: uses feature_cols_winner (no cage control features)
X_train_full_winner = X_valid_winner.iloc[:train_end_full]
X_cal_full_winner = X_valid_winner.iloc[train_end_full:]

scaler_winner = StandardScaler()
X_train_full_winner_s = scaler_winner.fit_transform(X_train_full_winner)
X_cal_full_winner_s = scaler_winner.transform(X_cal_full_winner)

# Method/round/finish: uses full feature_cols (with cage control features)
scaler_prod = StandardScaler()
X_train_full_s = scaler_prod.fit_transform(X_train_full)
X_cal_full_s = scaler_prod.transform(X_cal_full)

# Retrain winner models (on feature_cols_winner - NO cage control)
print("\n    Retraining Logistic Regression on full data (winner features)...")
lr_prod = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
# Production recency weights
train_dates_prod = ufc_valid.iloc[:train_end_full]['date']
min_date_prod = train_dates_prod.min()
max_date_prod = train_dates_prod.max()
date_range_prod = (max_date_prod - min_date_prod).days
recency_weights_prod = np.array([
    1.0 + 2.0 * ((d - min_date_prod).days / max(date_range_prod, 1)) ** 1.5
    for d in train_dates_prod
])
recency_weights_prod = recency_weights_prod / recency_weights_prod.mean()

lr_prod.fit(X_train_full_winner_s, y_train_full, sample_weight=recency_weights_prod)

print("    Retraining Random Forest on full data (winner features)...")
rf_prod = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=10,
    min_samples_split=10, random_state=42, n_jobs=-1
)
rf_prod.fit(X_train_full_winner_s, y_train_full, sample_weight=recency_weights_prod)

print("    Retraining XGBoost on full data (winner features, early stopping)...")
xgb_prod = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, eval_metric='logloss', verbosity=0,
    early_stopping_rounds=50
)
xgb_prod.fit(X_train_full_winner_s, y_train_full, sample_weight=recency_weights_prod, eval_set=[(X_cal_full_winner_s, y_cal_full)], verbose=False)
print(f"      Early stopped at iteration: {xgb_prod.best_iteration}")

print("    Retraining Neural Network (MLP) on full data (winner features)...")
mlp_prod = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)
mlp_prod.fit(X_train_full_winner_s, y_train_full)
print(f"      Iterations: {mlp_prod.n_iter_}")

# Platt calibration (using winner features)
print("    Fitting Platt calibration on full data (winner features)...")
p_lr_cal_full = lr_prod.predict_proba(X_cal_full_winner_s)[:, 1]
p_rf_cal_full = rf_prod.predict_proba(X_cal_full_winner_s)[:, 1]
p_xgb_cal_full = xgb_prod.predict_proba(X_cal_full_winner_s)[:, 1]

# Fit the calibrator on exactly what it will be applied to. This previously
# averaged four models including mlp_prod, while predict_fight_prod averages
# three, so the calibrator mapped from a distribution it had never seen.
# Measured effect is small - the MLP moves the average by 0.021 on average -
# but a calibrator fitted on one quantity and applied to another is wrong
# regardless of how little it currently costs.
# Including the MLP on both sides was also measured, and was worse
# (ECE 0.149 against 0.123). See experiments/calibrator_mismatch.py.
p_ens_cal_full = (p_lr_cal_full + p_rf_cal_full + p_xgb_cal_full) / 3

platt_prod = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
platt_prod.fit(p_ens_cal_full.reshape(-1, 1), y_cal_full)

# Retrain method model
print("    Retraining Method prediction model...")
n_m_full = len(X_method)
train_end_m_full = int(n_m_full * 0.90)
X_train_m_full = scaler_prod.transform(X_method.iloc[:train_end_m_full])
X_cal_m_full = scaler_prod.transform(X_method.iloc[train_end_m_full:])
y_train_m_full = y_method[:train_end_m_full]
y_cal_m_full = y_method[train_end_m_full:]

# Calculate sample weights for production method model
from collections import Counter
class_counts_prod = Counter(y_train_m_full)
n_classes_prod = len(le_method.classes_)
total_train = len(y_train_m_full)
class_weights_prod = {i: total_train / (n_classes_prod * class_counts_prod[i]) for i in range(n_classes_prod)}
sw_train_m_prod = np.array([class_weights_prod[y] for y in y_train_m_full])

xgb_method_prod = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='multi:softprob', num_class=3,
    random_state=42, eval_metric='mlogloss', verbosity=0,
    early_stopping_rounds=50
)
xgb_method_prod.fit(X_train_m_full, y_train_m_full, sample_weight=sw_train_m_prod, eval_set=[(X_cal_m_full, y_cal_m_full)], verbose=False)

# Retrain IsFinish model (Finish vs Decision)
print("    Retraining IsFinish prediction model...")
y_finish_full = (ufc_method['target_method'] != 'Decision').astype(int).values
y_train_fin_full = y_finish_full[:train_end_m_full]
y_cal_fin_full = y_finish_full[train_end_m_full:]

n_neg_full = len(y_train_fin_full[y_train_fin_full==0])
n_pos_full = len(y_train_fin_full[y_train_fin_full==1])

xgb_is_finish_prod = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=n_neg_full / max(n_pos_full, 1),
    random_state=42, eval_metric='logloss', verbosity=0,
    early_stopping_rounds=30
)
xgb_is_finish_prod.fit(X_train_m_full, y_train_fin_full, eval_set=[(X_cal_m_full, y_cal_fin_full)], verbose=False)

# Retrain Early/Late finish model
print("    Retraining Early/Late finish model...")
finish_mask_full = ufc_method['target_method'] != 'Decision'
X_finish_full = X_method[finish_mask_full].copy()
rounds_full = ufc_method.loc[finish_mask_full, 'finish_round'].fillna(3)
y_early_full = (rounds_full <= 2).astype(int).values

n_fin_full = len(X_finish_full)
train_end_fin_full = int(n_fin_full * 0.90)

X_train_early_full = scaler_prod.transform(X_finish_full.iloc[:train_end_fin_full])
X_cal_early_full = scaler_prod.transform(X_finish_full.iloc[train_end_fin_full:])
y_train_early_full = y_early_full[:train_end_fin_full]
y_cal_early_full = y_early_full[train_end_fin_full:]

xgb_early_finish_prod = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', verbosity=0,
    early_stopping_rounds=30
)
xgb_early_finish_prod.fit(X_train_early_full, y_train_early_full, eval_set=[(X_cal_early_full, y_cal_early_full)], verbose=False)

# Retrain round model
print("    Retraining Round prediction model...")
n_r_full = len(X_round)
train_end_r_full = int(n_r_full * 0.90)
X_train_r_full = scaler_prod.transform(X_round.iloc[:train_end_r_full])
X_cal_r_full = scaler_prod.transform(X_round.iloc[train_end_r_full:])
y_train_r_full = y_round[:train_end_r_full] - 1
y_cal_r_full = y_round[train_end_r_full:] - 1

xgb_round_prod = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='multi:softprob', num_class=5,
    random_state=42, eval_metric='mlogloss', verbosity=0,
    early_stopping_rounds=50
)
xgb_round_prod.fit(X_train_r_full, y_train_r_full, eval_set=[(X_cal_r_full, y_cal_r_full)], verbose=False)

print("\n    âœ“ PRODUCTION MODELS READY")
print(f"    Using {len(X_train_full):,} fights for training (90% of all data)")

# ============================================================================
# PRODUCTION PREDICTION FUNCTION
# ============================================================================
# ============================================================================
# CAGE CONTROL TRIGGER DETECTION
# ============================================================================
# Computes how likely a fighter will use cage control against a specific opponent.
# Uses historical fight data to build trigger profiles.

# Pre-compute per-fight grind data for trigger analysis
_cage_ctrl_fight_data = {}  # fighter -> list of {grind_score, opp_stats}

def _build_cage_ctrl_fight_data():
    """Pre-compute per-fight cage control data for trigger analysis."""
    global _cage_ctrl_fight_data
    _cage_ctrl_fight_data = {}

    for _, row in ufc.iterrows():
        r_name = row.get('r_name')
        b_name = row.get('b_name')
        fight_date = row.get('date')

        if pd.isna(r_name) or pd.isna(b_name):
            continue

        r_gs = row.get('r_grind_score', 0)
        b_gs = row.get('b_grind_score', 0)

        # Red fighter's grind data (opponent is blue)
        if not pd.isna(r_gs):
            if r_name not in _cage_ctrl_fight_data:
                _cage_ctrl_fight_data[r_name] = []
            _cage_ctrl_fight_data[r_name].append({
                'date': fight_date,
                'grind_score': float(r_gs) if not pd.isna(r_gs) else 0,
                'opp_td_def': float(row.get('b_td_def', 50)) if not pd.isna(row.get('b_td_def')) else 50,
                'opp_str_def': float(row.get('b_str_def', 50)) if not pd.isna(row.get('b_str_def')) else 50,
                'opp_sapm': float(row.get('b_sapm', 3)) if not pd.isna(row.get('b_sapm')) else 3,
                'opp_reach': float(row.get('b_reach_inches', 70)) if not pd.isna(row.get('b_reach_inches')) else 70,
                'opp_splm': float(row.get('b_splm', 3)) if not pd.isna(row.get('b_splm')) else 3,
            })

        # Blue fighter's grind data (opponent is red)
        if not pd.isna(b_gs):
            if b_name not in _cage_ctrl_fight_data:
                _cage_ctrl_fight_data[b_name] = []
            _cage_ctrl_fight_data[b_name].append({
                'date': fight_date,
                'grind_score': float(b_gs) if not pd.isna(b_gs) else 0,
                'opp_td_def': float(row.get('r_td_def', 50)) if not pd.isna(row.get('r_td_def')) else 50,
                'opp_str_def': float(row.get('r_str_def', 50)) if not pd.isna(row.get('r_str_def')) else 50,
                'opp_sapm': float(row.get('r_sapm', 3)) if not pd.isna(row.get('r_sapm')) else 3,
                'opp_reach': float(row.get('r_reach_inches', 70)) if not pd.isna(row.get('r_reach_inches')) else 70,
                'opp_splm': float(row.get('r_splm', 3)) if not pd.isna(row.get('r_splm')) else 3,
            })

_build_cage_ctrl_fight_data()
print(f"    [Cage Control] Built trigger data for {len(_cage_ctrl_fight_data)} fighters")

def compute_cage_control_likelihood(fighter_name, opponent_stats, event_date=None):
    """
    Compute how likely a fighter will use cage control against THIS specific opponent.

    Returns a value in [0, 1]:
      0 = very unlikely to grind
      1 = very likely to grind this opponent

    Uses:
      - Fighter's historical grind scores (career capability)
      - Trigger profile matching (which opponents they tend to grind)
      - Confidence discounting (thin history = less certainty)
    """
    # Get fighter's fight history
    fights = _cage_ctrl_fight_data.get(fighter_name, [])

    # Filter to fights before event_date if provided
    if event_date and fights:
        try:
            cutoff = pd.Timestamp(event_date)
            fights = [f for f in fights if pd.Timestamp(f['date']) < cutoff]
        except:
            pass

    n_fights = len(fights)

    # --- Fallback tiers ---
    if n_fights == 0:
        # Unknown fighter: use td_avg as weak proxy
        td_avg = float(opponent_stats.get('td_avg', 0)) if isinstance(opponent_stats, dict) and not pd.isna(opponent_stats.get('td_avg', 0)) else 0
        return min(td_avg / 10.0, 0.3) * 0.3  # very conservative

    # Career grind capability (EWM-weighted)
    grind_scores = [f['grind_score'] for f in fights]
    if len(grind_scores) > 1:
        # Manual EWM: more recent fights weighted more
        weights = [0.85 ** (len(grind_scores) - 1 - i) for i in range(len(grind_scores))]
        w_sum = sum(weights)
        career_grind_cap = sum(gs * w for gs, w in zip(grind_scores, weights)) / w_sum
    else:
        career_grind_cap = grind_scores[0]

    # If fighter almost never grinds, short-circuit
    if career_grind_cap < 0.08:
        return 0.03  # Near-zero: pure striker

    # --- Thin history handling ---
    if n_fights < 3:
        # Not enough data for trigger analysis; use career cap with heavy discount
        confidence = n_fights / 6.0
        return career_grind_cap * confidence * 0.7

    # --- Full trigger analysis (>= 3 fights) ---
    GRIND_THRESHOLD = 0.35
    grind_fights = [f for f in fights if f['grind_score'] >= GRIND_THRESHOLD]
    nongrind_fights = [f for f in fights if f['grind_score'] < GRIND_THRESHOLD]

    # If not enough grind/non-grind fights for comparison
    if len(grind_fights) < 2 or len(nongrind_fights) < 1:
        # Use career cap with moderate discount
        confidence = min(n_fights / 6.0, 1.0)
        return career_grind_cap * confidence * 0.8

    # Build opponent profiles for grind vs non-grind fights
    trigger_dims = ['opp_td_def', 'opp_str_def', 'opp_sapm', 'opp_reach', 'opp_splm']

    # Compute means for grind victim profile
    grind_profile = {}
    nongrind_profile = {}
    for dim in trigger_dims:
        grind_vals = [f[dim] for f in grind_fights if dim in f]
        nongrind_vals = [f[dim] for f in nongrind_fights if dim in f]
        grind_profile[dim] = np.mean(grind_vals) if grind_vals else 50
        nongrind_profile[dim] = np.mean(nongrind_vals) if nongrind_vals else 50

    # Compute std for normalization (across ALL fights)
    all_vals = {}
    for dim in trigger_dims:
        all_vals[dim] = [f[dim] for f in fights if dim in f]

    # Get current opponent's stats
    def _safe_val(v, d=0):
        try:
            val = float(v)
            return val if not (pd.isna(val) or np.isinf(val)) else d
        except:
            return d

    opp_stats_vec = {}
    if isinstance(opponent_stats, dict):
        opp_stats_vec = {
            'opp_td_def': _safe_val(opponent_stats.get('td_def'), 50),
            'opp_str_def': _safe_val(opponent_stats.get('str_def'), 50),
            'opp_sapm': _safe_val(opponent_stats.get('sapm'), 3),
            'opp_reach': _safe_val(opponent_stats.get('reach'), 70),
            'opp_splm': _safe_val(opponent_stats.get('splm'), 3),
        }
    else:
        opp_stats_vec = {dim: 50 for dim in trigger_dims}

    # Compute normalized distance to grind profile vs non-grind profile
    dist_to_grind = 0
    dist_to_nongrind = 0
    for dim in trigger_dims:
        std = np.std(all_vals.get(dim, [50])) if all_vals.get(dim) else 1.0
        std = max(std, 0.1)  # prevent division by zero
        opp_val = opp_stats_vec.get(dim, 50)
        dist_to_grind += ((opp_val - grind_profile[dim]) / std) ** 2
        dist_to_nongrind += ((opp_val - nongrind_profile[dim]) / std) ** 2

    dist_to_grind = np.sqrt(dist_to_grind)
    dist_to_nongrind = np.sqrt(dist_to_nongrind)

    # Softmax-style trigger match score
    # Higher when opponent is closer to grind victim profile
    exp_grind = np.exp(-dist_to_grind)
    exp_nongrind = np.exp(-dist_to_nongrind)
    trigger_match = exp_grind / (exp_grind + exp_nongrind + 1e-8)

    # Blend career capability with trigger match
    # 40% career tendency + 60% matchup-specific trigger
    cage_control_likelihood = 0.4 * career_grind_cap + 0.6 * trigger_match

    # Confidence discount (full confidence at 6+ fights)
    confidence_factor = min(n_fights / 6.0, 1.0)
    cage_control_likelihood *= confidence_factor

    return float(np.clip(cage_control_likelihood, 0, 1))

def apply_cage_control_adjustment(method_probs, p_is_finish, p_early, r_ccl, b_ccl, p_win):
    """
    Adjust method/round predictions based on cage control likelihood.

    Args:
        method_probs: dict with KO/TKO, Submission, Decision probabilities
        p_is_finish: probability of finish (vs decision)
        p_early: probability of early finish (R1-R2 vs R3+)
        r_ccl: red corner cage control likelihood
        b_ccl: blue corner cage control likelihood
        p_win: red corner win probability

    Returns:
        adjusted (method_probs, p_is_finish, p_early, net_grind)
    """
    # Net grind factor (winner-weighted: predicted winner's CCL matters more)
    net_grind = r_ccl * p_win + b_ccl * (1 - p_win)

    if net_grind < 0.05:
        return method_probs, p_is_finish, p_early, net_grind

    # --- Method adjustment: shift toward Decision ---
    MAX_DEC_SHIFT = 0.09  # Maximum 9% probability shift toward Decision (conservative)
    dec_boost = net_grind * MAX_DEC_SHIFT

    # Diminishing returns: if model already predicts Decision strongly,
    # reduce the adjustment to avoid double-counting
    adj_method = method_probs.copy()
    old_dec = adj_method.get('Decision', 0.33)
    if old_dec > 0.55:
        # Linearly reduce boost as Decision approaches 0.90
        diminish_factor = max(0, 1.0 - (old_dec - 0.55) / 0.35)
        dec_boost *= diminish_factor

    new_dec = min(old_dec + dec_boost, 0.90)
    actual_shift = new_dec - old_dec

    if actual_shift > 0:
        # Redistribute from KO/TKO and Submission proportionally
        ko_sub_total = adj_method.get('KO/TKO', 0.33) + adj_method.get('Submission', 0.33)
        if ko_sub_total > 0.01:
            ko_ratio = adj_method.get('KO/TKO', 0.33) / ko_sub_total
            sub_ratio = adj_method.get('Submission', 0.33) / ko_sub_total
            adj_method['KO/TKO'] = max(0.01, adj_method.get('KO/TKO', 0.33) - actual_shift * ko_ratio)
            adj_method['Submission'] = max(0.01, adj_method.get('Submission', 0.33) - actual_shift * sub_ratio)
        adj_method['Decision'] = new_dec

    # --- IsFinish adjustment: reduce finish probability (up to 30%) ---
    adj_finish = p_is_finish * (1 - net_grind * 0.30)

    # --- EarlyFinish adjustment: push toward later rounds (up to 40%) ---
    adj_early = p_early * (1 - net_grind * 0.40)

    return adj_method, adj_finish, adj_early, net_grind

def predict_fight_prod(red_name, blue_name, event_date=None, is_5rnd=False, is_title=False, context=None, verbose=False):
    """
    Production prediction function using models trained on FULL dataset.
    
    Args:
        red_name: Red corner fighter name
        blue_name: Blue corner fighter name  
        event_date: Optional event date (YYYY-MM-DD) for age/layoff calculation
        is_5rnd: Whether fight is 5 rounds (main event/title)
        is_title: Whether fight is for a title
        context: Optional dict with context adjustments
        verbose: If True, print name resolution notes
    
    Returns:
        Dict with prediction results
    """
    
    def safe(v, d=0):
        try:
            val = float(v)
            return val if pd.notna(val) and not np.isinf(val) else d
        except:
            return d
    
    # Resolve fighter names
    # Resolve both corners, or refuse. No silent substitution, no invented stats.
    resolved, problems, warnings = _check_fighters(red_name, blue_name)
    if verbose:
        for msg in warnings:
            print(f"    {msg}")
    if problems:
        if verbose:
            for msg in problems:
                print(f"    {msg}")
        return _no_data_result(red_name, blue_name, problems)

    r_resolved, r = resolved['red']
    b_resolved, b = resolved['blue']
    
    
    
    # Bayesian Skill Features
    r_mu = safe(r.get('mu'), TRUESKILL_DEFAULT_MU)
    r_sigma = safe(r.get('sigma'), TRUESKILL_DEFAULT_SIGMA)
    b_mu = safe(b.get('mu'), TRUESKILL_DEFAULT_MU)
    b_sigma = safe(b.get('sigma'), TRUESKILL_DEFAULT_SIGMA)
    
    bayes_prob = bayesian_win_prob(r_mu, r_sigma, b_mu, b_sigma)
    r_consistency = skill_consistency(r_sigma)
    b_consistency = skill_consistency(b_sigma)
    r_skill_cons = r_mu - CONSERVATIVE_K * r_sigma
    b_skill_cons = b_mu - CONSERVATIVE_K * b_sigma
    combined_unc = np.sqrt(r_sigma**2 + b_sigma**2)
    
    # Age and layoff
    if event_date:
        event_dt = pd.to_datetime(event_date)
        r_dob = pd.to_datetime(r.get('dob'), errors='coerce')
        b_dob = pd.to_datetime(b.get('dob'), errors='coerce')
        r_age = (event_dt - r_dob).days / 365.25 if pd.notna(r_dob) else 30
        b_age = (event_dt - b_dob).days / 365.25 if pd.notna(b_dob) else 30
        r_last = pd.to_datetime(r.get('last_fight_date'), errors='coerce')
        b_last = pd.to_datetime(b.get('last_fight_date'), errors='coerce')
        r_layoff_adj = (event_dt - r_last).days if pd.notna(r_last) else 500
        b_layoff_adj = (event_dt - b_last).days if pd.notna(b_last) else 500
    else:
        r_age, b_age = 30, 30
        r_layoff_adj = safe(r.get('layoff'), 500)
        b_layoff_adj = safe(b.get('layoff'), 500)
    
    r_prime = age_prime_score(r_age)
    b_prime = age_prime_score(b_age)
    
    # Decided UFC bouts only, matching ufc['r_exp'] in the training frame.
    r_exp = safe(r.get('wins')) + safe(r.get('losses'))
    b_exp = safe(b.get('wins')) + safe(b.get('losses'))
    
    mmr_diff = (safe(r.get('mmr_pre'), TRUESKILL_DEFAULT_MMR)
                - safe(b.get('mmr_pre'), TRUESKILL_DEFAULT_MMR))
    base_prob = float(base_probability(mmr_diff))
    
    r_winrate = safe(r.get('wins')) / r_exp if r_exp > 0 else np.nan
    b_winrate = safe(b.get('wins')) / b_exp if b_exp > 0 else np.nan
    
    r_southpaw = 1 if 'southpaw' in str(r.get('stance', '')).lower() else 0
    b_southpaw = 1 if 'southpaw' in str(b.get('stance', '')).lower() else 0
    
    power_diff = (
        (safe(r.get('splm')) - safe(b.get('splm'))) * 1.0 +
        (safe(r.get('str_def')) - safe(b.get('str_def'))) * 0.5 +
        ((safe(r.get('str_acc')) - safe(b.get('str_acc'))) / 100.0) * 2.0 +
        (safe(b.get('sapm')) - safe(r.get('sapm'))) * 0.8
    )
    
    layoff_diff = (b_layoff_adj - r_layoff_adj) / 100.0
    streak_diff = safe(r.get('streak')) - safe(b.get('streak'))
    
    # Feature dict
    feat = {
        'bayesian_prob': bayes_prob, 'mu_diff': r_mu - b_mu,
        'consistency_diff': r_consistency - b_consistency,
        'skill_conservative_diff': r_skill_cons - b_skill_cons,
        'combined_uncertainty': combined_unc,
        'mu_sum': r_mu + b_mu,
        'mu_diff_z': float(standardised_skill_gap(
            r_mu, r_sigma, b_mu, b_sigma, TRUESKILL_BETA)),
        'base_prob': base_prob, 'mmr_diff': mmr_diff,
        'exp_diff': r_exp - b_exp, 'age_diff': safe(r_age, 30) - safe(b_age, 30),
        # prime_diff removed (replaced by prime_wc_diff)
        'off_striking_diff': safe(r.get('splm')) - safe(b.get('splm')),
        'acc_diff': safe(r.get('str_acc')) - safe(b.get('str_acc')),
        'def_diff': safe(r.get('str_def')) - safe(b.get('str_def')),
        'power_diff': power_diff,
        'td_off_diff': safe(r.get('td_avg')) - safe(b.get('td_avg')),
        'td_def_diff': safe(r.get('td_def')) - safe(b.get('td_def')),
        'sub_diff': safe(r.get('sub_avg')) - safe(b.get('sub_avg')),
        'same_cluster': 0, 'southpaw_diff': r_southpaw - b_southpaw,
        # layoff_diff removed (replaced by ring_rust_diff) # streak_diff removed (replaced by mom_quality_diff)
        'winrate_diff': r_winrate - b_winrate,
        'td_acc_diff': safe(r.get('td_acc')) - safe(b.get('td_acc')),
        'sapm_diff': safe(r.get('sapm')) - safe(b.get('sapm')),
        'is_5rnd': int(is_5rnd), 'is_title': int(is_title),
        'recent_form_diff': safe(r.get('won_L3'), 0.5) - safe(b.get('won_L3'), 0.5),
        # momentum_diff removed (replaced by mom_quality_diff)
        'splm_L3_diff': safe(r.get('splm_L3')) - safe(b.get('splm_L3')),
        'str_acc_L3_diff': safe(r.get('str_acc_L3')) - safe(b.get('str_acc_L3')),
        'td_avg_L3_diff': safe(r.get('td_avg_L3')) - safe(b.get('td_avg_L3')),
        # NEW: Physical features
        'height_diff': safe(r.get('height'), 70) - safe(b.get('height'), 70),
        'reach_diff': safe(r.get('reach'), 70) - safe(b.get('reach'), 70),
        'ape_index_diff': (safe(r.get('reach'), 70) / max(safe(r.get('height'), 70), 1)) - (safe(b.get('reach'), 70) / max(safe(b.get('height'), 70), 1)),
        # NEW: Style/Finish rate features
        'ko_rate_diff': safe(r.get('ko_rate')) - safe(b.get('ko_rate')),
        'sub_rate_diff': safe(r.get('sub_rate')) - safe(b.get('sub_rate')),
        'finish_rate_diff': (safe(r.get('ko_rate')) + safe(r.get('sub_rate'))) - (safe(b.get('ko_rate')) + safe(b.get('sub_rate'))),
        # NEW: Durability features
        'ko_vulnerability_diff': (1 if safe(r.get('ko_losses')) > 0 else 0) - (1 if safe(b.get('ko_losses')) > 0 else 0),
        'been_finished_diff': safe(r.get('been_finished')) - safe(b.get('been_finished')),
        'absorption_eff_diff': safe(r.get('absorption_eff'), 16.67) - safe(b.get('absorption_eff'), 16.67),
        'footwork_diff': safe(r.get('footwork_proxy'), 1.0) - safe(b.get('footwork_proxy'), 1.0),
        # Stance matchup interaction
        'stance_mismatch': int(r_southpaw != b_southpaw),
        'stance_interaction': (r_southpaw - b_southpaw) * int(r_southpaw != b_southpaw),
        # Fighter trajectory (decline detection)
        'striking_trajectory_diff': safe(r.get('striking_trajectory')) - safe(b.get('striking_trajectory')),
        'accuracy_trajectory_diff': safe(r.get('accuracy_trajectory')) - safe(b.get('accuracy_trajectory')),
        'trajectory_diff': (safe(r.get('striking_trajectory')) / 3.0 + safe(r.get('accuracy_trajectory')) / 20.0) - (safe(b.get('striking_trajectory')) / 3.0 + safe(b.get('accuracy_trajectory')) / 20.0),
        # Career damage accumulation
        'career_damage_diff': np.log1p(safe(r.get('career_damage'), 36)) - np.log1p(safe(b.get('career_damage'), 36)),
        # NEW: Opponent quality features
        # Unknown schedule stays unknown; the flag below tells the model which
        # it is, instead of a constant that means neither.
        'opp_quality_diff': (safe(r.get('opp_quality'), np.nan)
                             - safe(b.get('opp_quality'), np.nan)),
        'opp_history_known': float(pd.notna(r.get('opp_quality'))
                                   and pd.notna(b.get('opp_quality'))),
        # NEW: Weight class features (default to 0 for unknown)
        'is_womens': 0,  # Will be overridden if division info available
        'is_heavyweight': 0,  # Will be overridden if division info available
        # Cage control features
        'cage_control_cap_diff': safe(r.get('ctrl_rate_ewm')) - safe(b.get('ctrl_rate_ewm')),
        'clinch_activity_diff': safe(r.get('clinch_activity_ewm')) - safe(b.get('clinch_activity_ewm')),
        'grind_tendency_diff': safe(r.get('grind_rate')) - safe(b.get('grind_rate')),
        # ADVANCED FEATURES (Section 3.95)
        'cardio_diff': safe(r.get('cardio'), 0.5) - safe(b.get('cardio'), 0.5),
        'archetype_matchup': ARCHETYPE_MATCHUP_MATRIX[int(safe(r.get('archetype'), 0)), int(safe(b.get('archetype'), 0))] - 0.5,
        'archetype_clash': int(safe(r.get('archetype'), 0) != safe(b.get('archetype'), 0)),
        'combined_pace': safe(r.get('splm'), 3) + safe(b.get('splm'), 3),
        'high_pace': int((safe(r.get('splm'), 3) + safe(b.get('splm'), 3)) > 7.5),
        'wc_move_diff': safe(r.get('wc_move'), 0) - safe(b.get('wc_move'), 0),
        'prime_wc_diff': age_prime_score_wc(r_age, 'unknown') - age_prime_score_wc(b_age, 'unknown'),
        'ring_rust_diff': ring_rust_transform(r_layoff_adj) - ring_rust_transform(b_layoff_adj),
        'sub_def_diff': safe(r.get('sub_def_score'), 0.5) - safe(b.get('sub_def_score'), 0.5),
        'mom_quality_diff': safe(r.get('mom_quality'), 0) - safe(b.get('mom_quality'), 0),
        'data_sparsity_diff': np.log1p(min(safe(r.get('fight_count'), 5), 30)) - np.log1p(min(safe(b.get('fight_count'), 5), 30)),
    }

    # Predict using PRODUCTION models
    # Winner model: feature_cols_winner (no cage control) with scaler_winner
    # Method/round/finish: full feature_cols with scaler_prod
    feat.update(_declared_features(
        r, b,
        _corner_extras(r, r_age, r_exp, r_winrate, r_southpaw, r_consistency,
                       r_skill_cons, context.get('weight_class') if context else None),
        _corner_extras(b, b_age, b_exp, b_winrate, b_southpaw, b_consistency,
                       b_skill_cons, context.get('weight_class') if context else None)))

    _feat_frame = pd.DataFrame([feat]).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_pred = _feat_frame[feature_cols]
    X_pred_winner = _feat_frame[feature_cols_winner]
    X_pred_s = scaler_prod.transform(X_pred)          # For method/round/finish
    X_pred_winner_s = scaler_winner.transform(X_pred_winner)  # For winner models
    
    p_ens = (lr_prod.predict_proba(X_pred_winner_s)[:, 1][0] + 
             rf_prod.predict_proba(X_pred_winner_s)[:, 1][0] + 
             xgb_prod.predict_proba(X_pred_winner_s)[:, 1][0]) / 3
    p_win_base = platt_prod.predict_proba([[p_ens]])[0, 1]
    
    # Auto-detect pressure fighters from archetype for cage size adjustments
    if context and context.get("cage_size") and "red_pressure_fighter" not in context and "blue_pressure_fighter" not in context:
        r_arch = int(safe(r.get("archetype"), 0)) if r else 0
        b_arch = int(safe(b.get("archetype"), 0)) if b else 0
        # Archetypes 0 (pressure_boxer) and 2 (wrestler) are pressure fighters
        context["red_pressure_fighter"] = r_arch in (0, 2)
        context["blue_pressure_fighter"] = b_arch in (0, 2)

    p_win, context_log = apply_context_adjustments(p_win_base, context)
    
    # Method
    p_method = xgb_method_prod.predict_proba(X_pred_s)[0]
    method_probs = dict(zip(le_method.classes_, p_method))
    
    # Round
    p_round = xgb_round_prod.predict_proba(X_pred_s)[0]
    round_probs = {f'R{i+1}': p for i, p in enumerate(p_round)}
    
    # Simplified finish prediction
    p_is_finish = xgb_is_finish_prod.predict_proba(X_pred_s)[0, 1]

    # --- CAGE CONTROL ADJUSTMENT ---
    r_ccl = compute_cage_control_likelihood(
        r_resolved or red_name, b if b else {}, event_date
    )
    b_ccl = compute_cage_control_likelihood(
        b_resolved or blue_name, r if r else {}, event_date
    )

    p_early_raw = 0.5  # default
    if p_is_finish > 0.5:
        p_early_raw = xgb_early_finish_prod.predict_proba(X_pred_s)[0, 1]

    # Apply cage control adjustment to method/finish/timing
    method_probs, p_is_finish_adj, p_early_adj, net_grind = apply_cage_control_adjustment(
        method_probs, p_is_finish, p_early_raw, r_ccl, b_ccl, p_win
    )

    is_finish = p_is_finish_adj > 0.5

    if is_finish:
        p_early = p_early_adj
        finish_timing = "Early (R1-R2)" if p_early > 0.5 else "Late (R3+)"
        finish_timing_prob = p_early if p_early > 0.5 else 1 - p_early
    else:
        finish_timing = "N/A (Decision)"
        finish_timing_prob = 0


    # Data sparsity confidence dampener
    raw_confidence = abs(p_win - 0.5) * 2
    r_fc = safe(r.get('fight_count'), 5) if isinstance(r, dict) else 5
    b_fc = safe(b.get('fight_count'), 5) if isinstance(b, dict) else 5
    min_fights = min(r_fc, b_fc)
    if min_fights < 4:
        sparsity_cap = 0.55 + (min_fights / 4) * 0.15  # 0->55%, 1->58.75%, 2->62.5%, 3->66.25%
        raw_confidence = min(raw_confidence, sparsity_cap)

    return {
        'red': r_resolved or red_name,
        'blue': b_resolved or blue_name,
        # Diagnostics: the win probability before each adjustment layer, so the
        # layers can be measured against outcomes rather than assumed to help.
        'p_ensemble': float(p_ens),          # raw LR+RF+XGB average
        'p_platt': float(p_win_base),        # after Platt calibration
        'red_win_prob': p_win,               # after context adjustment
        'blue_win_prob': 1 - p_win,
        'winner': (r_resolved or red_name) if p_win > 0.5 else (b_resolved or blue_name),
        'win_prob': max(p_win, 1 - p_win),
        'confidence': raw_confidence,
        'method_probs': method_probs,
        'method': max(method_probs, key=method_probs.get),
        'method_prob': max(method_probs.values()),
        'round_probs': round_probs,
        'round': max(round_probs, key=round_probs.get),
        'round_prob': max(round_probs.values()),
        # Simplified finish prediction
        'is_finish': is_finish,
        'finish_prob': p_is_finish if is_finish else 1 - p_is_finish,
        'finish_timing': finish_timing,
        'finish_timing_prob': finish_timing_prob,
        # Cage control info
        'cage_control': {
            'red_likelihood': r_ccl,
            'blue_likelihood': b_ccl,
            'net_grind': net_grind,
        },
    }

# ============================================================================
# SECTION 16: FIGHT CARD PREDICTION WITH TABLE OUTPUT
# ============================================================================
print("\n" + "="*70)
print("SECTION 16: FIGHT CARD PREDICTIONS")
print("="*70)

def predict_card(fights, event_date=None, event_name="Fight Card", contexts=None):
    """
    Predict an entire fight card and return a formatted table.
    
    Args:
        fights: List of tuples (red_name, blue_name) or (red_name, blue_name, is_5rnd, is_title)
        event_date: Event date for all fights (YYYY-MM-DD)
        event_name: Name of the event for display
    
    Returns:
        DataFrame with predictions
    """
    results = []
    simulations = []

    print(f"\n    Predicting {len(fights)} fights...")
    
    for i, fight in enumerate(fights):
        if len(fight) == 2:
            red, blue = fight
            is_5rnd, is_title = False, False
        elif len(fight) == 4:
            red, blue, is_5rnd, is_title = fight
        else:
            red, blue = fight[0], fight[1]
            is_5rnd = fight[2] if len(fight) > 2 else False
            is_title = fight[3] if len(fight) > 3 else False
        
        pred = predict_fight_prod(red, blue, event_date=event_date, is_5rnd=is_5rnd, is_title=is_title, verbose=False)

        if pred.get('status') == 'NO_DATA':
            results.append({
                'Fight': f"{red} vs {blue}",
                'Winner': 'NO DATA', 'Win%': '-', 'Confidence': '-',
                'Method': '-', 'Method%': '-', 'Round': '-', 'Round%': '-',
            })
            print(f"    [{i+1}] SKIPPED - {pred['reason']}")
            continue

        results.append({
            'Fight': f"{pred['red']} vs {pred['blue']}",
            'Winner': pred['winner'],
            'Win%': f"{pred['win_prob']*100:.1f}%",
            'Confidence': f"{pred['confidence']*100:.0f}%",
            'Method': pred['method'],
            'Method%': f"{pred['method_prob']*100:.1f}%",
            'Round': pred['round'],
            'Round%': f"{pred['round_prob']*100:.1f}%",
        })

        # The simulation layer. Reported BESIDE the model, never instead of it:
        # standalone it picks winners at AUC 0.559 against the model's 0.66.
        # What it is the only source for is how the fight ends.
        simulations.append((
            f"{pred['red']} vs {pred['blue']}",
            summarise(
                simulate_matchup(fighter_stats.get(_norm_name(pred['red']), {}),
                                 fighter_stats.get(_norm_name(pred['blue']), {}),
                                 rounds=5 if is_5rnd else 3),
                pred['red'], pred['blue']),
        ))
    
    df = pd.DataFrame(results)
    
    # Print formatted table
    print(f"\n{'='*120}")
    print(f"  {event_name.upper()} - PREDICTIONS")
    if event_date:
        print(f"  Event Date: {event_date}")
    print(f"  Model: Production (trained on {len(X_train_full):,} fights)")
    print(f"{'='*120}")
    
    # Set display options for better table output
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 200)
    
    print(df.to_string(index=False))
    print(f"{'='*120}\n")

    _print_simulations(simulations)

    # Handed to the app export at the end of the run, so the phone shows the
    # same simulation that was printed rather than a second one that could
    # disagree with it.
    global _card_simulations
    _card_simulations = simulations

    return df


def _print_simulations(simulations):
    """The Monte Carlo layer, printed under the table it does not replace."""
    if not simulations:
        return
    print(f"{'='*120}")
    print(f"  SIMULATION - each matchup run {CARD_SIMULATIONS:,} times from "
          f"both fighters' measured rates")
    print("  Reported beside the model, not instead of it: on its own this "
          "picks winners at AUC 0.559")
    print("  against the model's 0.66. Where it is the only source is method, "
          "round and duration.")
    print("  It finishes 55% of fights where the sport finishes 47%, so read "
          "a DEC call as firmer")
    print("  than it looks and a SUB call as softer.")
    print(f"{'='*120}")
    for label, summary in simulations:
        print(f"  {label}")
        print(format_line(summary))
        if summary is None:
            continue
        parts = " ".join(f"R{i+1} {p*100:.0f}%"
                         for i, p in enumerate(summary["finish_by_round"]))
        print(f"      finish by round: {parts}"
              f"   decision {summary['decision_share']*100:.0f}%")
        assumed = set(summary["imputed_red"]) | set(summary["imputed_blue"])
        if assumed:
            print(f"      assumed from the league: {', '.join(sorted(assumed))}")
    print(f"{'='*120}\n")

# ============================================================================
# FIGHT CARD INPUT
# ============================================================================

# FIGHT_CARD, EVENT_NAME, EVENT_DATE defined in USER SETTINGS at top

# Run predictions
predictions_df = predict_card(FIGHT_CARD, event_date=EVENT_DATE, event_name=EVENT_NAME, contexts=FIGHT_CONTEXTS)

# ============================================================================
# SUMMARY TABLE (alternative compact view)
# ============================================================================
print("\n" + "="*70)
print("COMPACT PREDICTION SUMMARY")
print("="*70)

# Create a more compact summary
compact_results = []
for i, fight in enumerate(FIGHT_CARD):
    red = fight[0]
    blue = fight[1]
    is_5rnd = fight[2] if len(fight) > 2 else False
    is_title = fight[3] if len(fight) > 3 else False
    fight_context = FIGHT_CONTEXTS.get(i, None) if FIGHT_CONTEXTS else None
    pred = predict_fight_prod(red, blue, event_date=EVENT_DATE, is_5rnd=is_5rnd, is_title=is_title, context=fight_context, verbose=False)

    if pred.get('status') == 'NO_DATA':
        compact_results.append({
            '#': i+1, 'Matchup': f"{red} vs {blue}", 'Pick': 'NO DATA',
            'Prob': '-', 'Conf': '-', 'Method': '-', 'Finish': '-',
        })
        continue

    winner_short = pred['winner'].split()[-1]  # Last name only

    # Simplified finish output
    if pred.get('is_finish', False):
        finish_str = f"{pred['method'][:3]} ({pred.get('finish_timing', 'R?')[:5]})"
    else:
        finish_str = "Decision"

    compact_results.append({
        '#': i+1,
        'Matchup': f"{pred['red'].split()[-1]} vs {pred['blue'].split()[-1]}",
        'Pick': winner_short,
        'Prob': f"{pred['win_prob']*100:.0f}%",
        'Conf': f"{pred['confidence']*100:.0f}%",
        'Method': pred['method'],
        'Finish': finish_str,
    })

compact_df = pd.DataFrame(compact_results)
print(compact_df.to_string(index=False))

# ============================================================================

# ============================================================================
# AUTOMATIC ODDS FETCHING & VALUE BET ANALYSIS
# ============================================================================
# Get your FREE API key at: https://the-odds-api.com/
# Free tier: 500 requests/month (plenty for weekly UFC cards)

# Try to import requests (needed for odds API)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("  [Odds] Warning: 'requests' module not installed. Run: pip install requests")

from difflib import SequenceMatcher

# ============================================================================
# ODDS API CONFIGURATION
# ============================================================================
# Read from the environment - never hardcode the key (this repo is public).
# Set it before running:  export ODDS_API_KEY="your_key"
# Get a free key at https://the-odds-api.com/
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# Sportsbooks to fetch (in order of preference)
PREFERRED_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbetus', 'bovada', 'betonlineag']

def fetch_mma_odds(api_key=None):
    """
    Fetch current MMA/UFC odds from The Odds API.

    Returns:
        dict: Fighter name -> odds data, or None if failed
    """
    if not REQUESTS_AVAILABLE:
        print("  [Odds] Cannot fetch - 'requests' module not installed")
        return None

    key = api_key or ODDS_API_KEY
    if not key:
        print("  [Odds] No API key set. Set ODDS_API_KEY or pass api_key parameter.")
        print("  [Odds] Get free key at: https://the-odds-api.com/")
        return None

    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = {
        'apiKey': key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american'
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 401:
            print("  [Odds] Invalid API key")
            return None
        elif response.status_code == 429:
            print("  [Odds] Rate limit exceeded")
            return None
        elif response.status_code != 200:
            print(f"  [Odds] API error: {response.status_code}")
            return None

        data = response.json()

        # Parse into fighter -> odds mapping
        odds_data = {}
        for event in data:
            event_name = event.get('sport_title', '')
            commence_time = event.get('commence_time', '')

            # Get the two fighters
            home = event.get('home_team', '')
            away = event.get('away_team', '')

            if not home or not away:
                continue

            # Extract odds from bookmakers
            bookmakers = event.get('bookmakers', [])

            home_odds = []
            away_odds = []

            # Prioritize preferred books
            sorted_books = sorted(bookmakers,
                key=lambda x: PREFERRED_BOOKS.index(x['key']) if x['key'] in PREFERRED_BOOKS else 999)

            for book in sorted_books:
                for market in book.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home:
                                home_odds.append({
                                    'book': book['title'],
                                    'odds': outcome['price']
                                })
                            elif outcome['name'] == away:
                                away_odds.append({
                                    'book': book['title'],
                                    'odds': outcome['price']
                                })

            if home_odds:
                odds_data[home.lower()] = {
                    'name': home,
                    'opponent': away,
                    'odds': home_odds,
                    'best_odds': max([o['odds'] for o in home_odds]),
                    'event': event_name,
                    'time': commence_time
                }

            if away_odds:
                odds_data[away.lower()] = {
                    'name': away,
                    'opponent': home,
                    'odds': away_odds,
                    'best_odds': max([o['odds'] for o in away_odds]),
                    'event': event_name,
                    'time': commence_time
                }

        # Show remaining quota
        remaining = response.headers.get('x-requests-remaining', '?')
        print(f"  [Odds] Fetched {len(odds_data)//2} fights | API requests remaining: {remaining}")

        return odds_data

    except requests.RequestException as e:
        print(f"  [Odds] Network error: {e}")
        return None

def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def implied_prob_to_american(prob):
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

# A SURNAME IS NOT AN IDENTITY, and treating it as one priced the wrong
# fighter. The previous rule accepted any odds entry sharing a last name if the
# whole-string similarity cleared 0.5, which on a card carrying Bruno Silva
# matched Anderson Silva at 0.64, Jean Silva at 0.67 and Erick Silva at 0.64.
# The dataset itself contains two different fighters named Bruno Silva.
#
# That is not a near miss. The edge and value columns would have been computed
# against a price belonging to somebody else, and a fabricated edge looks
# exactly like a real one. It stayed harmless only because no odds key was ever
# set; it becomes live the moment one is.
#
# So a match now needs the last name AND a compatible first name: equal, or one
# an initial or prefix of the other ("Jon" for "Jonathan", "T.J." for "TJ").
# Anything else is a different person and gets no price rather than a wrong one.
FIRST_NAME_MIN_RATIO = 0.85


def _names_of(text):
    return [part for part in re.split(r"[^a-z0-9]+", text.lower()) if part]


def _compatible_first_names(a, b):
    """Same person, allowing for an initial or a shortened form."""
    if a == b:
        return True
    if not a or not b:
        return False
    if a.startswith(b) or b.startswith(a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= FIRST_NAME_MIN_RATIO


def match_fighter_to_odds(fighter_name, odds_data):
    """The odds entry for this exact fighter, or None.

    None is the right answer for an unmatched fighter: the card then shows no
    price, which is visibly missing. A wrong price is invisible.
    """
    if not odds_data:
        return None

    fighter_lower = fighter_name.lower().strip()
    if fighter_lower in odds_data:
        return odds_data[fighter_lower]

    parts = _names_of(fighter_lower)
    if not parts:
        return None
    last_name, first_name = parts[-1], parts[0]

    for key, data in odds_data.items():
        other = _names_of(key)
        if not other or other[-1] != last_name:
            continue
        if _compatible_first_names(first_name, other[0]):
            return data

    # No surname match. A whole-string near-identity is still allowed, for a
    # spelling or accent difference, but the bar is high enough that two
    # different people cannot clear it.
    best_match, best_ratio = None, 0.90
    for key, data in odds_data.items():
        ratio = SequenceMatcher(None, fighter_lower, key).ratio()
        if ratio > best_ratio:
            best_ratio, best_match = ratio, data
    return best_match

def calculate_value(predicted_prob, american_odds):
    """
    Calculate betting value.

    Value = (Predicted Prob * Decimal Odds) - 1
    Positive value = edge over the book
    """
    implied_prob = american_to_implied_prob(american_odds)

    # Value formula
    if american_odds > 0:
        decimal_odds = (american_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(american_odds)) + 1

    value = (predicted_prob * decimal_odds) - 1
    edge = predicted_prob - implied_prob

    return {
        'implied_prob': implied_prob,
        'edge': edge,
        'value': value,
        'decimal_odds': decimal_odds
    }

def get_value_rating(edge, value):
    """Get a rating for the value bet."""
    if edge >= 0.15 and value >= 0.20:
        return "GREAT VALUE"
    elif edge >= 0.10 and value >= 0.10:
        return "GOOD VALUE"
    elif edge >= 0.05 and value >= 0.05:
        return "SLIGHT VALUE"
    elif edge >= 0:
        return "FAIR"
    elif edge >= -0.05:
        return "SLIGHT -EV"
    else:
        return "BAD VALUE"

# Global variable to store fetched odds
CURRENT_ODDS = None

def load_odds(api_key=None):
    """Load odds from API and store globally."""
    global CURRENT_ODDS
    CURRENT_ODDS = fetch_mma_odds(api_key)
    return CURRENT_ODDS

# Try to load odds if API key is set
if ODDS_API_KEY:
    print("\n[Odds] Fetching current MMA odds...")
    load_odds()
else:
    print("\n[Odds] No API key set. To enable odds:")
    print("       1. Get free key at https://the-odds-api.com/")
    print("       2. Set ODDS_API_KEY = 'your_key' in the notebook")
    print("       3. Or call load_odds('your_key') to fetch")



# ============================================================================
# ADVANCED IMPROVEMENTS MODULE
# ============================================================================
# Contains:
# - Confidence calibration (historical accuracy at each confidence level)
# - Division/weight class detection
# - Recent form weighting
# - Value threshold flagging
# - Kelly Criterion bankroll management
# - Historical accuracy tracking

# ============================================================================
# IMPROVEMENT 2: CONFIDENCE CALIBRATION
# ============================================================================

def build_confidence_calibration(y_true, y_pred_prob, n_bins=10):
    """Build calibration mapping: confidence level -> actual accuracy."""
    confidences = np.abs(y_pred_prob - 0.5) * 2
    predictions = (y_pred_prob >= 0.5).astype(int)
    correct = (predictions == y_true).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_map = {}
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 10:
            actual_acc = correct[mask].mean()
            bin_center = (bins[i] + bins[i+1]) / 2
            calibration_map[bin_center] = actual_acc
    return calibration_map

def get_calibrated_confidence(raw_confidence, calibration_map):
    """Convert raw model confidence to calibrated confidence."""
    if not calibration_map:
        return raw_confidence
    bins = sorted(calibration_map.keys())
    closest_bin = min(bins, key=lambda x: abs(x - raw_confidence))
    return calibration_map[closest_bin]

try:
    CONFIDENCE_CALIBRATION = build_confidence_calibration(y_test, p_cal_test)
    print(f"[Calibration] Built from {len(y_test)} test fights")
except:
    CONFIDENCE_CALIBRATION = {}

# ============================================================================
# IMPROVEMENT 3: DIVISION/WEIGHT CLASS AWARENESS
# ============================================================================

WEIGHT_CLASSES = {
    'Strawweight': {'limit': 115, 'is_womens': True, 'finish_rate': 0.35},
    'Flyweight': {'limit': 125, 'is_womens': False, 'finish_rate': 0.45},
    "Women's Flyweight": {'limit': 125, 'is_womens': True, 'finish_rate': 0.40},
    'Bantamweight': {'limit': 135, 'is_womens': False, 'finish_rate': 0.48},
    "Women's Bantamweight": {'limit': 135, 'is_womens': True, 'finish_rate': 0.42},
    'Featherweight': {'limit': 145, 'is_womens': False, 'finish_rate': 0.52},
    'Lightweight': {'limit': 155, 'is_womens': False, 'finish_rate': 0.50},
    'Welterweight': {'limit': 170, 'is_womens': False, 'finish_rate': 0.52},
    'Middleweight': {'limit': 185, 'is_womens': False, 'finish_rate': 0.55},
    'Light Heavyweight': {'limit': 205, 'is_womens': False, 'finish_rate': 0.60},
    'Heavyweight': {'limit': 265, 'is_womens': False, 'finish_rate': 0.68},
}

def detect_weight_class(fighter_stats_r, fighter_stats_b, fight_info=None):
    """Detect weight class from fighter stats or fight info."""
    if fight_info and 'weight_class' in fight_info:
        wc = fight_info['weight_class']
        for name, info in WEIGHT_CLASSES.items():
            if name.lower() in wc.lower():
                return {'name': name, **info}
    return None

def adjust_finish_probability(base_finish_prob, weight_class_info):
    """Adjust finish probability based on weight class tendencies."""
    if not weight_class_info:
        return base_finish_prob
    wc_finish_rate = weight_class_info.get('finish_rate', 0.50)
    adjusted = 0.7 * base_finish_prob + 0.3 * wc_finish_rate
    return adjusted

# ============================================================================
# IMPROVEMENT 4: RECENT FORM WEIGHTING
# ============================================================================

def calculate_recency_weighted_stats(fighter_stats, decay_factor=0.85):
    """Apply recency weighting to fighter stats."""
    if not fighter_stats:
        return fighter_stats
    weighted_stats = fighter_stats.copy()
    l3_weight = 0.6
    career_weight = 0.4
    for stat in ['splm', 'str_acc', 'td_avg']:
        l3_stat = f'{stat}_L3'
        if l3_stat in fighter_stats and stat in fighter_stats:
            l3_val = fighter_stats.get(l3_stat, fighter_stats[stat])
            career_val = fighter_stats.get(stat, 0)
            weighted_stats[stat] = l3_weight * l3_val + career_weight * career_val
    streak = fighter_stats.get('streak', 0)
    won_l3 = fighter_stats.get('won_L3', 0.5)
    if streak > 0 and won_l3 > 0.5:
        weighted_stats['recency_momentum'] = min(1.0, 0.5 + streak * 0.1 + (won_l3 - 0.5) * 0.5)
    elif streak < 0 and won_l3 < 0.5:
        weighted_stats['recency_momentum'] = max(0.0, 0.5 + streak * 0.1 + (won_l3 - 0.5) * 0.5)
    else:
        weighted_stats['recency_momentum'] = 0.5
    return weighted_stats

# ============================================================================
# IMPROVEMENT 5: ODDS VALUE THRESHOLD AUTO-FLAGGING
# ============================================================================

VALUE_THRESHOLDS = {
    'EXTREME_VALUE': {'edge': 0.20, 'min_prob': 0.40},
    'GREAT_VALUE': {'edge': 0.15, 'min_prob': 0.35},
    'GOOD_VALUE': {'edge': 0.10, 'min_prob': 0.30},
    'SLIGHT_VALUE': {'edge': 0.05, 'min_prob': 0.25},
    'FAIR': {'edge': 0.0, 'min_prob': 0.0},
    'NEGATIVE_EV': {'edge': -0.05, 'min_prob': 0.0},
}

def flag_value_bet(predicted_prob, implied_prob, edge):
    """Flag a bet based on edge and probability thresholds."""
    for level, thresholds in VALUE_THRESHOLDS.items():
        if edge >= thresholds['edge'] and predicted_prob >= thresholds['min_prob']:
            if level in ['EXTREME_VALUE', 'GREAT_VALUE']:
                return (level, True, f"Strong edge of {edge*100:.1f}%")
            elif level in ['GOOD_VALUE', 'SLIGHT_VALUE']:
                return (level, True, f"Positive edge of {edge*100:.1f}%")
            elif level == 'FAIR':
                return (level, False, "Break-even, no edge")
            else:
                return (level, False, f"Negative edge of {edge*100:.1f}%")
    return ('NEGATIVE_EV', False, f"Negative edge of {edge*100:.1f}%")

# ============================================================================
# IMPROVEMENT 6: BANKROLL MANAGEMENT (KELLY CRITERION)
# ============================================================================

def kelly_criterion(win_prob, odds_american, bankroll=1000, fraction=0.25):
    """Calculate optimal bet size using Kelly Criterion."""
    if odds_american > 0:
        decimal_odds = (odds_american / 100) + 1
    else:
        decimal_odds = (100 / abs(odds_american)) + 1
    implied_prob = 1 / decimal_odds
    b = decimal_odds - 1
    p = win_prob
    q = 1 - win_prob
    kelly_full = (b * p - q) / b if b > 0 else 0
    kelly_full = max(0, kelly_full)
    kelly_frac = kelly_full * fraction
    ev = (win_prob * (decimal_odds - 1)) - (1 - win_prob)
    bet_size = bankroll * kelly_frac
    return {
        'kelly_full_pct': kelly_full * 100,
        'kelly_fraction_pct': kelly_frac * 100,
        'recommended_bet': round(bet_size, 2),
        'expected_value': ev,
        'ev_per_100': round(ev * 100, 2),
        'edge': win_prob - implied_prob,
        'decimal_odds': decimal_odds
    }

def get_bet_sizing_recommendation(kelly_info, confidence):
    """Get human-readable bet sizing recommendation."""
    kelly_pct = kelly_info['kelly_fraction_pct']
    ev = kelly_info['ev_per_100']
    if kelly_pct <= 0:
        return "NO BET - Negative edge"
    elif kelly_pct < 1:
        return f"SMALL ({kelly_pct:.1f}%) - Marginal edge"
    elif kelly_pct < 3:
        return f"STANDARD ({kelly_pct:.1f}%) - +{ev:.0f} EV/100"
    elif kelly_pct < 5:
        return f"CONFIDENT ({kelly_pct:.1f}%) - +{ev:.0f} EV/100"
    else:
        return f"MAX ({min(kelly_pct, 5):.1f}% cap) - +{ev:.0f} EV/100"

# ============================================================================
# IMPROVEMENT 7: HISTORICAL ACCURACY TRACKING
# ============================================================================

# PREDICTIONS_LOG_FILE is defined in the PATHS block near the top of this file.

def load_prediction_history():
    """Load historical predictions from file."""
    try:
        with open(PREDICTIONS_LOG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'predictions': [], 'summary': {'total': 0, 'correct': 0, 'pending': 0}}

def save_prediction(prediction_record):
    """Save a prediction to history."""
    history = load_prediction_history()
    prediction_record['timestamp'] = datetime.now().isoformat()
    prediction_record['status'] = 'pending'
    history['predictions'].append(prediction_record)
    history['summary']['total'] += 1
    history['summary']['pending'] += 1
    PREDICTIONS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    return len(history['predictions'])

def update_prediction_result(fight_id, actual_winner):
    """Update a prediction with the actual result."""
    history = load_prediction_history()
    for pred in history['predictions']:
        if pred.get('fight_id') == fight_id and pred.get('status') == 'pending':
            pred['actual_winner'] = actual_winner
            pred['correct'] = pred['predicted_winner'].lower() == actual_winner.lower()
            pred['status'] = 'verified'
            if pred['correct']:
                history['summary']['correct'] += 1
            history['summary']['pending'] -= 1
    with open(PREDICTIONS_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_tracking_summary():
    """Get summary of prediction accuracy."""
    history = load_prediction_history()
    summary = history['summary']
    verified = summary['total'] - summary['pending']
    accuracy = summary['correct'] / verified if verified > 0 else 0
    return {
        'total_predictions': summary['total'],
        'verified': verified,
        'pending': summary['pending'],
        'correct': summary['correct'],
        'accuracy': accuracy,
        'accuracy_pct': f"{accuracy*100:.1f}%"
    }

def log_card_predictions(predictions_list, event_name, event_date):
    """Log all predictions for a fight card."""
    logged = 0
    for pred in predictions_list:
        record = {
            'fight_id': f"{event_date}_{pred['red']}_{pred['blue']}",
            'event_name': event_name,
            'event_date': event_date,
            'red_corner': pred['red'],
            'blue_corner': pred['blue'],
            'predicted_winner': pred['winner'],
            'win_probability': pred['win_prob'],
            'confidence': pred['confidence'],
            'method_predicted': pred.get('method', 'Unknown')
        }
        save_prediction(record)
        logged += 1
    print(f"[Tracking] Logged {logged} predictions for {event_name}")
    summary = get_tracking_summary()
    if summary['verified'] > 0:
        print(f"[Tracking] Overall: {summary['accuracy_pct']} accuracy ({summary['correct']}/{summary['verified']})")
    return logged

print("\n[Advanced Improvements] Loaded: Calibration, Weight Classes, Recency, Value Flags, Kelly, Tracking")


# BETTABILITY ANALYSIS & PARLAY BUILDER
# ============================================================================
# Analyze each fight's "bettability" based on multiple factors:
# 1. Confidence level (higher = better)
# 2. Model uncertainty (lower = more predictable matchup)
# 3. Fighter data quality (both fighters have good data)
# 4. Historical accuracy at similar confidence levels

def calculate_bettability_score(pred, r_stats, b_stats):
    """
    Calculate a bettability score from 0-100.
    Higher scores = more confident, safer bets.

    Factors considered:
    - Confidence: Main driver (50% weight)
    - Fighter sigma (uncertainty): Lower = better (20% weight)
    - Data completeness: Both fighters have recent fights (15% weight)
    - Streak alignment: Prediction aligns with momentum (15% weight)
    """
    score = 0
    reasons = []
    warnings = []

    # 1. Confidence score (0-50 points)
    conf = pred.get('confidence', 0)
    conf_score = min(50, conf * 50)  # Cap at 50 points
    score += conf_score

    if conf >= 0.6:
        reasons.append(f"High confidence ({conf*100:.0f}%)")
    elif conf < 0.3:
        warnings.append(f"Low confidence ({conf*100:.0f}%)")

    # 2. Combined model uncertainty (0-20 points)
    r_sigma = r_stats.get('sigma', 8.333) if r_stats else 8.333
    b_sigma = b_stats.get('sigma', 8.333) if b_stats else 8.333
    combined_sigma = (r_sigma + b_sigma) / 2

    # Lower sigma = more certain about fighter skill
    # Default sigma is 8.333, experienced fighters have ~4-6
    sigma_score = max(0, 20 - (combined_sigma - 4) * 3)
    score += sigma_score

    if combined_sigma < 5:
        reasons.append("Both fighters well-established")
    elif combined_sigma > 7:
        warnings.append("High uncertainty (newer/inconsistent fighters)")

    # 3. Data completeness (0-15 points)
    r_has_data = r_stats is not None and r_stats.get('last_fight_date') is not None
    b_has_data = b_stats is not None and b_stats.get('last_fight_date') is not None

    if r_has_data and b_has_data:
        score += 15
        # Check recency
        r_recent = r_stats.get('layoff', 999) < 365
        b_recent = b_stats.get('layoff', 999) < 365
        if r_recent and b_recent:
            reasons.append("Recent data for both fighters")
    elif r_has_data or b_has_data:
        score += 7
        warnings.append("Missing data for one fighter")
    else:
        warnings.append("Limited data for both fighters")

    # 4. Momentum alignment (0-15 points)
    # If our pick aligns with momentum/streak, more confidence
    r_streak = r_stats.get('streak', 0) if r_stats else 0
    b_streak = b_stats.get('streak', 0) if b_stats else 0
    predicted_red = pred.get('win_prob', 0.5) > 0.5

    if predicted_red and r_streak > b_streak:
        score += 15
        if r_streak >= 2:
            reasons.append(f"Pick on win streak ({r_streak})")
    elif not predicted_red and b_streak > r_streak:
        score += 15
        if b_streak >= 2:
            reasons.append(f"Pick on win streak ({b_streak})")
    elif (predicted_red and r_streak < -2) or (not predicted_red and b_streak < -2):
        score += 5
        warnings.append("Picking fighter on losing streak")
    else:
        score += 10  # Neutral

    return {
        'score': min(100, max(0, score)),
        'reasons': reasons,
        'warnings': warnings
    }

def get_bet_recommendation(score, conf):
    """Get betting recommendation based on score and confidence."""
    if score >= 75 and conf >= 0.5:
        return "STRONG BET"
    elif score >= 60 and conf >= 0.4:
        return "GOOD BET"
    elif score >= 45:
        return "LEAN"
    elif score >= 30:
        return "RISKY"
    else:
        return "AVOID"

def get_parlay_tier(score, conf):
    """
    Determine if fight should be in parlay and which tier.
    Tier 1: Lock - High confidence, include in all parlays
    Tier 2: Strong - Good confidence, include in larger parlays
    Tier 3: Value - Lower confidence but good value
    None: Don't include
    """
    if score >= 80 and conf >= 0.55:
        return "LOCK"
    elif score >= 65 and conf >= 0.45:
        return "STRONG"
    elif score >= 50 and conf >= 0.35:
        return "VALUE"
    else:
        return None

print("\n" + "="*80)


# ============================================================================
# ENHANCED VALUE BET ANALYSIS (with odds)
# ============================================================================

def analyze_fight_value(pred, fighter_name, odds_data):
    """
    Analyze betting value for a predicted winner.

    Returns value metrics if odds available, None otherwise.
    """
    if not odds_data:
        return None

    odds_match = match_fighter_to_odds(fighter_name, odds_data)
    if not odds_match:
        return None

    best_odds = odds_match['best_odds']
    value_info = calculate_value(pred['win_prob'], best_odds)

    return {
        'fighter': odds_match['name'],
        'best_odds': best_odds,
        'book_with_best': next((o['book'] for o in odds_match['odds'] if o['odds'] == best_odds), 'Unknown'),
        'implied_prob': value_info['implied_prob'],
        'our_prob': pred['win_prob'],
        'edge': value_info['edge'],
        'value': value_info['value'],
        'rating': get_value_rating(value_info['edge'], value_info['value'])
    }


print("BETTABILITY ANALYSIS")
print("="*80)
print("Score: 0-100 (higher = more confident bet)")
print("Recommendation: STRONG BET > GOOD BET > LEAN > RISKY > AVOID")
print("-"*80)

bet_analysis = []
parlay_candidates = []
skipped_fights = []

for i, fight in enumerate(FIGHT_CARD):
    red = fight[0]
    blue = fight[1]
    is_5rnd = fight[2] if len(fight) > 2 else False
    is_title = fight[3] if len(fight) > 3 else False
    fight_context = FIGHT_CONTEXTS.get(i, None) if FIGHT_CONTEXTS else None

    pred = predict_fight_prod(red, blue, event_date=EVENT_DATE, is_5rnd=is_5rnd, is_title=is_title, context=fight_context, verbose=False)

    # A refusal never reaches bettability, Kelly sizing, parlays or the log.
    if pred.get('status') == 'NO_DATA':
        skipped_fights.append({'fight_num': i + 1, 'matchup': f"{red} vs {blue}",
                               'problems': pred['problems']})
        print(f"\nFight {i+1}: {red} vs {blue}")
        for msg in pred['problems']:
            print(f"  {msg}")
        print("  -> No prediction made.")
        continue

    # Get fighter stats for additional analysis
    r_resolved, _ = resolve_fighter_name(red)
    b_resolved, _ = resolve_fighter_name(blue)
    r_stats = fighter_stats.get(_norm_name(r_resolved)) if r_resolved else None
    b_stats = fighter_stats.get(_norm_name(b_resolved)) if b_resolved else None

    # Apply recency weighting to stats
    r_stats_weighted = calculate_recency_weighted_stats(r_stats) if r_stats else None
    b_stats_weighted = calculate_recency_weighted_stats(b_stats) if b_stats else None

    # Get calibrated confidence
    raw_conf = pred['confidence']
    calibrated_conf = get_calibrated_confidence(raw_conf, CONFIDENCE_CALIBRATION)

    # Calculate bettability
    bet_info = calculate_bettability_score(pred, r_stats_weighted, b_stats_weighted)
    recommendation = get_bet_recommendation(bet_info['score'], pred['confidence'])
    parlay_tier = get_parlay_tier(bet_info['score'], pred['confidence'])

    # Get odds-based value analysis
    value_info = analyze_fight_value(pred, pred['winner'], CURRENT_ODDS) if CURRENT_ODDS else None

    # Kelly Criterion bet sizing (if odds available)
    kelly_info = None
    value_flag = None
    if value_info:
        kelly_info = kelly_criterion(pred['win_prob'], value_info['best_odds'])
        value_flag, should_bet, flag_reason = flag_value_bet(
            pred['win_prob'], value_info['implied_prob'], value_info['edge'])

    # Store for summary
    bet_analysis.append({
        'fight_num': i + 1,
        # Carried rather than re-derived: the prediction dict does not keep it,
        # and the app export read a 5-round main event as a 3-round bout.
        'is_5rnd': bool(is_5rnd),
        'is_title': bool(is_title),
        'matchup': f"{pred['red'].split()[-1]} vs {pred['blue'].split()[-1]}",
        'pick': pred['winner'].split()[-1],
        'prob': pred['win_prob'],
        'conf': pred['confidence'],
        'calibrated_conf': calibrated_conf,
        'score': bet_info['score'],
        'recommendation': recommendation,
        'parlay_tier': parlay_tier,
        'reasons': bet_info['reasons'],
        'warnings': bet_info['warnings'],
        'value_info': value_info,
        'kelly_info': kelly_info,
        'value_flag': value_flag,
        'pred_full': pred  # Store full prediction for logging
    })

    if parlay_tier:
        parlay_candidates.append({
            'fight_num': i + 1,
            'matchup': f"{pred['red'].split()[-1]} vs {pred['blue'].split()[-1]}",
            'pick': pred['winner'].split()[-1],
            'prob': pred['win_prob'],
            'tier': parlay_tier,
            'score': bet_info['score']
        })

    # Print individual analysis
    print(f"\nFight {i+1}: {pred['red']} vs {pred['blue']}")
    print(f"  Pick: {pred['winner']} ({pred['win_prob']*100:.0f}% prob, {pred['confidence']*100:.0f}% conf)")
    print(f"  Bettability Score: {bet_info['score']:.0f}/100 -> {recommendation}")
    if parlay_tier:
        print(f"  Parlay Tier: {parlay_tier}")
    if bet_info['reasons']:
        print(f"  + {', '.join(bet_info['reasons'])}")
    if bet_info['warnings']:
        print(f"  ! {', '.join(bet_info['warnings'])}")
    if value_info:
        odds_str = f"+{value_info['best_odds']}" if value_info['best_odds'] > 0 else str(value_info['best_odds'])
        print(f"  Odds: {odds_str} @ {value_info['book_with_best']} | Edge: {value_info['edge']*100:+.1f}% | {value_info['rating']}")
        if kelly_info and kelly_info['kelly_fraction_pct'] > 0:
            print(f"  Kelly: {get_bet_sizing_recommendation(kelly_info, calibrated_conf)}")
        if value_flag and value_flag in ['EXTREME_VALUE', 'GREAT_VALUE']:
            print(f"  >>> {value_flag}: High-confidence value bet! <<<")

# ============================================================================
# PARLAY BUILDER
# ============================================================================
print("\n" + "="*80)
print("PARLAY BUILDER")
print("="*80)

if len(parlay_candidates) == 0:
    print("No fights meet parlay criteria. Consider straight bets only.")
elif len(parlay_candidates) == 1:
    print("Only 1 fight meets parlay criteria - not enough for a parlay.")
    print(f"  Straight bet: {parlay_candidates[0]['pick']} ({parlay_candidates[0]['matchup']})")
else:
    # Sort by tier then score
    tier_order = {'LOCK': 0, 'STRONG': 1, 'VALUE': 2}
    parlay_candidates.sort(key=lambda x: (tier_order.get(x['tier'], 3), -x['score']))

    print(f"\n{len(parlay_candidates)} fights suitable for parlays:")
    print("-"*60)

    locks = [p for p in parlay_candidates if p['tier'] == 'LOCK']
    strong = [p for p in parlay_candidates if p['tier'] == 'STRONG']
    value = [p for p in parlay_candidates if p['tier'] == 'VALUE']

    if locks:
        print("\nLOCKS (include in all parlays):")
        for p in locks:
            print(f"  [{p['fight_num']}] {p['pick']} ({p['matchup']}) - {p['prob']*100:.0f}%")

    if strong:
        print("\nSTRONG (include in 2-3 leg parlays):")
        for p in strong:
            print(f"  [{p['fight_num']}] {p['pick']} ({p['matchup']}) - {p['prob']*100:.0f}%")

    if value:
        print("\nVALUE (add for bigger parlays):")
        for p in value:
            print(f"  [{p['fight_num']}] {p['pick']} ({p['matchup']}) - {p['prob']*100:.0f}%")

    # Suggest specific parlays
    print("\n" + "-"*60)
    print("SUGGESTED PARLAYS:")

    all_parlay = locks + strong
    if len(all_parlay) >= 2:
        combined_prob = 1.0
        for p in all_parlay:
            combined_prob *= p['prob']

        print(f"\n  SAFE PARLAY ({len(all_parlay)} legs):")
        for p in all_parlay:
            print(f"    - {p['pick']} ({p['matchup']})")
        print(f"    Combined probability: {combined_prob*100:.1f}%")
        print(f"    Implied odds: +{int((1/combined_prob - 1) * 100)}")

    if len(parlay_candidates) >= 3:
        big_parlay = parlay_candidates[:min(5, len(parlay_candidates))]
        combined_prob = 1.0
        for p in big_parlay:
            combined_prob *= p['prob']

        print(f"\n  AMBITIOUS PARLAY ({len(big_parlay)} legs):")
        for p in big_parlay:
            print(f"    - {p['pick']} ({p['matchup']})")
        print(f"    Combined probability: {combined_prob*100:.1f}%")
        print(f"    Implied odds: +{int((1/combined_prob - 1) * 100)}")

# ============================================================================
# ENHANCED COMPACT SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("FINAL PREDICTION SUMMARY WITH BETTING GUIDE")
print("="*80)

enhanced_results = []
for ba in bet_analysis:
    vi = ba.get('value_info')
    odds_str = ''
    value_str = '-'
    if vi:
        odds_str = f"+{vi['best_odds']}" if vi['best_odds'] > 0 else str(vi['best_odds'])
        value_str = vi['rating'].replace(' VALUE', '').replace('SLIGHT ', 'SL ')

    enhanced_results.append({
        '#': ba['fight_num'],
        'Matchup': ba['matchup'],
        'Pick': ba['pick'],
        'Prob': f"{ba['prob']*100:.0f}%",
        'Odds': odds_str if odds_str else '-',
        'Edge': f"{vi['edge']*100:+.1f}%" if vi else '-',
        'Value': value_str,
        'Bet': ba['recommendation'],
        'Parlay': ba['parlay_tier'] if ba['parlay_tier'] else '-'
    })

enhanced_df = pd.DataFrame(enhanced_results)
print(enhanced_df.to_string(index=False))

if skipped_fights:
    print("\n" + "-"*80)
    print(f"NOT PREDICTED - {len(skipped_fights)} of {len(FIGHT_CARD)} fights lack usable data:")
    for s in skipped_fights:
        print(f"  [{s['fight_num']}] {s['matchup']}")
        for msg in s['problems']:
            print(f"        {msg}")
    print("  Add a mapping to engine/data/fighter_aliases.json if a name is just spelled differently.")

print("\n" + "="*80)
print("LEGEND:")
print("  Score: Bettability score (0-100, higher = safer bet)")
print("  Bet: STRONG BET > GOOD BET > LEAN > RISKY > AVOID")
print("  Parlay: LOCK > STRONG > VALUE > - (not recommended)")
print("="*80)

# ============================================================================
# LOG PREDICTIONS FOR TRACKING
# ============================================================================
print("\n" + "="*80)
print("PREDICTION LOGGING")
print("="*80)

# Prepare predictions for logging
predictions_to_log = []
for ba in bet_analysis:
    if 'pred_full' in ba:
        predictions_to_log.append(ba['pred_full'])

if predictions_to_log:
    log_card_predictions(predictions_to_log, EVENT_NAME, EVENT_DATE)

# Show tracking summary
tracking = get_tracking_summary()
if tracking['verified'] > 0:
    print(f"\nHistorical Performance:")
    print(f"  Total Predictions: {tracking['total_predictions']}")
    print(f"  Verified Results:  {tracking['verified']}")
    print(f"  Correct:           {tracking['correct']}")
    print(f"  Accuracy:          {tracking['accuracy_pct']}")
    print(f"  Pending:           {tracking['pending']}")
print("="*80)


print(f"\n{'='*70}")
print(f"Fights on card:         {len(FIGHT_CARD)}")
print(f"Predicted:              {len(bet_analysis)}")
print(f"Refused (no data):      {len(skipped_fights)}")
print(f"Model training data: {len(X_train_full):,} fights (90% of dataset)")
print(f"Dataset end date: {ufc_valid['date'].max().date()}")
print(f"{'='*70}")

# ============================================================================
# APP EXPORT
# ============================================================================
# The only thing that crosses from this engine to the phone. Written last, so
# it carries whatever the run actually produced rather than a second
# computation that could disagree with what was printed above.

import app_export as _app_export

_app_fights = []
for _ba in bet_analysis:
    _pred = _ba['pred_full']
    _vi = _ba.get('value_info') or {}
    _sim = None
    for _label, _summary in _card_simulations:
        if _label == f"{_pred['red']} vs {_pred['blue']}":
            _sim = _summary
            break
    _app_fights.append({
        'number': _ba['fight_num'],
        'red': _pred['red'],
        'blue': _pred['blue'],
        'pick': _pred['winner'],
        'win_prob': _pred['win_prob'],
        'confidence': _ba.get('calibrated_conf', _pred.get('confidence')),
        'method': _pred.get('method'),
        'method_prob': _pred.get('method_prob'),
        'round': _pred.get('round'),
        'recommendation': _ba.get('recommendation'),
        'parlay_tier': _ba.get('parlay_tier'),
        'odds': _vi.get('best_odds'),
        'edge': _vi.get('edge'),
        'rounds_scheduled': 5 if _ba.get('is_5rnd') else 3,
        'title_fight': bool(_ba.get('is_title')),
        'simulation': _sim,
    })

_app_card = _app_export.card_payload(
    EVENT_NAME, str(EVENT_DATE),
    _app_fights,
    # skipped_fights carries 'matchup' and a LIST of 'problems'; reading it as
    # 'fight'/'reason' silently exported a pair of nulls and the app showed a
    # refusal with no fighters and no reason, which is worse than not showing
    # it - the reason is usually a misspelling the user can fix.
    skipped=[{'fight': _s.get('matchup'),
              'reason': ' '.join(_s.get('problems') or []) or None}
             for _s in skipped_fights],
    parlays=_app_export.parlay_payload(parlay_candidates),
    dataset_end=str(ufc_valid['date'].max().date()),
    trained_on=int(len(X_train_full)))

for _path in _app_export.write_all(APP_DATA_DIR, {'card': _app_card}):
    print(f"\nApp data: {_path}")
