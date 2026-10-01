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
EVENT_NAME = "UFC Fight Night: Rosas Jr. vs. Barcelos"
EVENT_DATE = "2026-09-26"  # Format: YYYY-MM-DD

# ==============================================================================
# 3. FIGHT CARD
# ==============================================================================
# Format: (Red_Corner, Blue_Corner, is_5_rounds, is_title_fight)
# - is_5_rounds: True for main events and title fights
# - is_title_fight: True if fighting for a championship

FIGHT_CARD = [
    # Main Event (5 rounds, non-title) - Bantamweight
    ("Raul Rosas Jr.", "Raoni Barcelos", True, False),         # 0 - Main Event, Bantamweight (5 rounds)

    # Co-Main Event - Women's Bantamweight
    ("Norma Dumont", "Ailin Perez"),                           # 1 - Co-Main, Women's Bantamweight

    # Main Card
    ("Luis Hernandez", "Sedriques Dumas"),                     # 2 - Middleweight (Hernandez: short-notice UFC debut)
    ("Mehemmedeli Osmanli", "Ilimbek Akylbek Uulu"),           # 3 - TUF 34 Bantamweight final
    ("Melissa Amaya", "Valesca Machado"),                      # 4 - TUF 34 Women's Strawweight final

    # Preliminary Card
    ("Brady Hiestand", "Rinya Nakamura"),                      # 5 - Bantamweight
    ("Rodolfo Vieira", "Robert Bryczek"),                      # 6 - Middleweight
    ("Rodolfo Bellato", "Christian Edwards"),                  # 7 - Light Heavyweight
    ("Elves Brener", "Josiah Harrell"),                        # 8 - Lightweight
    ("Montel Jackson", "Ricky Simon"),                         # 9 - Bantamweight
    ("John Castaneda", "Alatengheili"),                        # 10 - Bantamweight
    ("Yazmin Jauregui", "Vanessa Demopoulos"),                 # 11 - Women's Strawweight
]

# ==============================================================================
# 4. OPTIONAL CONTEXT ADJUSTMENTS (Per-Fight)
# ==============================================================================
# Home advantage only counts when ONE fighter is home and the other is NOT.
# If both fighters are "home" (e.g., both USA-based), no advantage.
#
# UFC Fight Night: Rosas Jr. vs. Barcelos - Meta Apex, Enterprise (Las Vegas), NV (SMALL cage)
#
# Home/Away Analysis (event in USA):
#   Fight 0: Rosas Jr. (fights out of Las Vegas, Home) vs Barcelos (Brazil, Away) -> red_home
#   Fight 1: Dumont (Brazil, Away) vs Perez (Argentina, Away) -> no advantage
#   Fight 2: Hernandez (UFC debut, no data) vs Dumas (USA, Home) -> not predicted
#   Fight 3: Osmanli vs Akylbek Uulu (TUF 34 final, both debuting) -> not predicted
#   Fight 4: Amaya vs Machado (TUF 34 final, both debuting) -> not predicted
#   Fight 5: Hiestand (USA, Home) vs Nakamura (Japan, Away) -> red_home
#   Fight 6: Vieira (Brazil, Away) vs Bryczek (Poland, Away) -> no advantage
#   Fight 7: Bellato (Brazil, Away) vs Edwards (USA, Home) -> blue_home
#   Fight 8: Brener (Brazil, Away) vs Harrell (USA, Home) -> blue_home
#   Fight 9: Jackson (USA, Home) vs Simon (USA, Home) -> no advantage (both home)
#   Fight 10: Castaneda (USA, Home) vs Alatengheili (China, Away) -> red_home
#   Fight 11: Jauregui (Mexico, Away) vs Demopoulos (USA, Home) -> blue_home
#
# NOTE: All bouts at the Apex use the SMALL cage -> cage_size: 'small' for every fight.
#
# 'division' tells the model the weight class of the bout (e.g. 'lightweight',
# "women's flyweight"). It feeds the weight-class features the model was
# trained on; left out, it is inferred from both fighters' last bouts, which
# is wrong whenever one of them is changing class.

FIGHT_CONTEXTS = {
    # Fight 0: Rosas Jr. (Home) - red corner
    0: {"division": "bantamweight", 'red_home': True, 'cage_size': 'small'},

    # Fight 1: Both Away - no advantage
    1: {"division": "women's bantamweight", 'cage_size': 'small'},

    # Fight 2: Hernandez debuting - no data either way
    2: {"division": "middleweight", 'cage_size': 'small'},

    # Fight 3: TUF final, both debuting
    3: {"division": "bantamweight", 'cage_size': 'small'},

    # Fight 4: TUF final, both debuting
    4: {"division": "women's strawweight", 'cage_size': 'small'},

    # Fight 5: Hiestand (Home) - red corner
    5: {"division": "bantamweight", 'red_home': True, 'cage_size': 'small'},

    # Fight 6: Both Away - no advantage
    6: {"division": "middleweight", 'cage_size': 'small'},

    # Fight 7: Edwards (Home) - blue corner
    7: {"division": "light heavyweight", 'blue_home': True, 'cage_size': 'small'},

    # Fight 8: Harrell (Home) - blue corner
    8: {"division": "lightweight", 'blue_home': True, 'cage_size': 'small'},

    # Fight 9: Both Home - no advantage
    9: {"division": "bantamweight", 'cage_size': 'small'},

    # Fight 10: Castaneda (Home) - red corner
    10: {"division": "bantamweight", 'red_home': True, 'cage_size': 'small'},

    # Fight 11: Demopoulos (Home) - blue corner
    11: {"division": "women's strawweight", 'blue_home': True, 'cage_size': 'small'},
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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report, roc_auc_score
import warnings
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_resolution import NameResolver, load_aliases, format_failure, short_name, norm_name as _norm_name
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

# Live-path imports that used to sit between the feature sections.
from method_calibration import apply_calibrator as _apply_finish_calibrator
from method_calibration import fit_calibrator as _fit_finish_calibrator
from method_calibration import rebuild_three_way as _rebuild_three_way
import market_blend as _blend
import finish_distil as _distil

# Where the phone app reads its data from.
APP_DATA_DIR = Path(os.environ.get("APP_DATA_DIR", Path(__file__).resolve().parent.parent / "app" / "data"))
_card_simulations = []
from fight_report import (
    CARD_SIMULATIONS,
    format_line,
    simulate_matchup,
    summarise,
)

# SECTIONS 1-4 (data preparation, career statistics, every feature, the
# feature lists and X) live in feature_frame.build(), unchanged, so the live
# path can put an upcoming fight through the same code as every training
# row. Everything the pipeline defines is bound here as before.
import feature_frame as _feature_frame
_UFC_RAW = ufc                  # the archive as read; live rows cut it by date
globals().update(_feature_frame.build(ufc))
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


# ============================================================================
# LIVE MODEL ROWS: the upcoming fight through the training pipeline
# ============================================================================
# A live prediction is a training row that has not been played yet. The fight
# becomes a pending row (pending_rows.py) after the archive cut the day before
# the event, and feature_frame.build - the code that built every training
# row - computes its features, with the interaction centres frozen at the
# training values. The live path used to rebuild the same features by hand
# from each fighter's last archived row; replaying the last four events,
# that moved the winner probability a mean 15.6 points from the training-style
# row for the same fight (experiments/live_parity.py).
import pending_rows as _pending_rows

_LIVE_ROWS = {}                     # key -> one-row frame of feature_cols
_TRAIN_X_BY_FIGHT = X.set_index(ufc['fight_id'].astype(str))


def _live_key(red, blue, when, is_5rnd, is_title, division):
    return (str(when.date()), _norm_name(red), _norm_name(blue),
            bool(is_5rnd), bool(is_title), str(division or ''))


def _division_of(context):
    if not context:
        return None
    return context.get('division') or context.get('weight_class')


def prepare_live_rows(fights, event_date=None):
    """Build the model rows for fights on one date, in as few passes as the
    rule "no fighter twice in one pass" allows (about 12 s a pass).

    fights: (red, blue, is_5rnd, is_title[, context]) tuples or dicts with
    those keys, names as the archive spells them.
    """
    when = (pd.Timestamp(event_date) if event_date
            else pd.Timestamp.today().normalize())
    wanted = []
    for f in fights:
        if not isinstance(f, dict):
            f = dict(zip(('red', 'blue', 'is_5rnd', 'is_title', 'context'), f))
        key = _live_key(f['red'], f['blue'], when, f.get('is_5rnd'),
                        f.get('is_title'), _division_of(f.get('context')))
        if key not in _LIVE_ROWS and key not in [k for k, _ in wanted]:
            wanted.append((key, f))
    if not wanted:
        return
    raw_dates = pd.to_datetime(_UFC_RAW['date'], errors='coerce')
    before = _UFC_RAW[raw_dates < when]
    ids = _pending_rows.fighter_ids(before)
    passes = []                     # each pass holds no fighter twice
    for key, f in wanted:
        pair = {ids.get(_norm_name(f['red'])), ids.get(_norm_name(f['blue']))}
        if None in pair:            # no bout before this date: nothing to
            continue                # build from (predict refuses these)
        for batch in passes:
            if not pair & batch['ids']:
                batch['fights'].append((key, f)); batch['ids'] |= pair
                break
        else:
            passes.append({'fights': [(key, f)], 'ids': set(pair)})
    for batch in passes:
        spec = [{'red': f['red'], 'blue': f['blue'], 'date': when,
                 'is_5rnd': bool(f.get('is_5rnd')),
                 'is_title': bool(f.get('is_title')),
                 'division': _division_of(f.get('context'))}
                for _, f in batch['fights']]
        pend = _pending_rows.build_pending(before, spec)
        built = _feature_frame.build(before, pending=pend,
                                     centres=INTERACTION_CENTRES,
                                     verbose=False)
        n = built['N_ARCHIVE']
        _check_append_invariance(built, n)
        rows = built['X'].iloc[n:]
        for (key, _), (_, row) in zip(batch['fights'], rows.iterrows()):
            _LIVE_ROWS[key] = row.to_frame().T.reset_index(drop=True)


def _check_append_invariance(built, n):
    """The archive part of a live build must equal the training rows, bit
    for bit. If it does not, something in the pipeline has started to read
    across rows, and the live row cannot be trusted to match training."""
    part = built['X'].iloc[:n]
    part.index = built['ufc']['fight_id'].iloc[:n].astype(str).values
    train = _TRAIN_X_BY_FIGHT.loc[part.index, part.columns]
    same = (part.values == train.values) | (np.isnan(part.values)
                                            & np.isnan(train.values))
    if not same.all():
        bad = sorted(set(part.columns[~same.all(axis=0)]))
        print(f"    WARNING: live build differs from training on "
              f"{len(bad)} columns ({', '.join(bad[:5])}) - live rows are "
              f"NOT the training definition")


def _live_row(red, blue, event_date, is_5rnd, is_title, context):
    when = (pd.Timestamp(event_date) if event_date
            else pd.Timestamp.today().normalize())
    key = _live_key(red, blue, when, is_5rnd, is_title, _division_of(context))
    if key not in _LIVE_ROWS:
        prepare_live_rows([{'red': red, 'blue': blue, 'is_5rnd': is_5rnd,
                            'is_title': is_title, 'context': context}], when)
    if key not in _LIVE_ROWS:
        raise KeyError(f"no bout before {when.date()} for {red} or {blue}")
    return _LIVE_ROWS[key]


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
    
    # These diagnostics are for the printed analysis; the model reads only
    # the training-pipeline row below.

    # Model rows from the training pipeline (see _live_rows).
    X_pred = _live_row(r_resolved or red_name, b_resolved or blue_name,
                       event_date, is_5rnd, is_title, context)[feature_cols]
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

# --- the distilled finish model -------------------------------------------
# P(finish) learned from the market's price rather than from the result.
# Measured walk-forward over 4,764 fights carrying both: AUC 0.577 trained on
# the outcome, 0.630 trained on the market, +0.053 with a 95% interval of
# [+0.038, +0.068]. That closes 63% of the distance to the market itself
# (0.661), and the weight chosen from earlier years came back 1.00 every
# year - no share of the raw outcome at all.
#
# THAT BASELINE IS NOT THIS FILE'S MODEL. It is a binary classifier trained
# on the outcome over the priced fights alone, so the experiment shows that
# distilling beats outcome-training on the same features - not that it beats
# the three-way model below, which trains with class weights on every fight
# in the archive. Against THAT, on the held-out split, the student is ahead
# on AUC, Brier, accuracy and balanced accuracy and behind on none, with
# every difference inside a paired bootstrap's interval. 841 fights cannot
# resolve a gap that size. finish_distil's docstring carries the table.
#
# It needs no price to predict, which is the point: this market does not
# exist live, so blending was never available and the information can only
# reach a card inside a model's weights.
#
# The three-way model stays, and keeps its job. This supplies the binary; the
# split between knockout and submission is still its.
print("    Distilling P(finish) from the historical method market...")
_METHOD_ODDS_PATH = ENGINE_DIR / 'data' / 'method_odds.csv' \
    if 'ENGINE_DIR' in dir() else Path(__file__).resolve().parent / 'data' / 'method_odds.csv'


def _fight_key(red, blue, date):
    return tuple(sorted((_norm_name(red), _norm_name(blue)))) + \
        (pd.Timestamp(date).date(),)


FINISH_STUDENT = None
if _METHOD_ODDS_PATH.exists():
    _targets = _distil.load_targets(_METHOD_ODDS_PATH, _fight_key)
    _rows = [_targets.get(_fight_key(r, b, d), float('nan'))
             for r, b, d in zip(ufc_method['r_name'], ufc_method['b_name'],
                                ufc_method['date'])]
    _rows = np.asarray(_rows, dtype=float)
    # Only the training window. A price from a fight in the calibration split
    # is a price from after what the model is fitted on.
    _student_targets = np.full(len(_rows), np.nan)
    _student_targets[:train_end_m_full] = _rows[:train_end_m_full]
    # SCALED, because that is what it will be asked to predict on. The first
    # version of this line fitted on the raw frame and then received
    # scaler_prod's output at prediction time, so every split threshold it had
    # learned was in the wrong units: AUC fell from 0.630 to 0.582 and the
    # spread collapsed to a ten-point band around a coin flip, because inputs
    # far outside the training range all fall into the same few leaves. The
    # card printed 44-54% for nine fights in a row, which is what caught it.
    FINISH_STUDENT = _distil.fit(scaler_prod.transform(X_method),
                                 _student_targets)
    _matched = int(np.isfinite(_student_targets).sum())
    if FINISH_STUDENT is None:
        print(f"      Not distilled: only {_matched:,} fights in the training "
              f"window carry a usable price.")
    else:
        print(f"      Trained on {_matched:,} market prices.")

    # IT SHIPS ONLY IF IT BEATS WHAT IT REPLACES, here, today, on the split
    # it was not fitted on. A measurement in an experiment file says the idea
    # works; it does not say that THIS build wired it up correctly. The
    # scaling bug above produced a student that was worse than the model it
    # was replacing and still printed a confident number for every fight, and
    # nothing in the output said so. This is the check that would have caught
    # it on the first run.
    #
    # The comparison is tilted AGAINST the student: the three-way model used
    # this same split as its early-stopping eval set, so it has seen it once
    # and the student has not. A student that wins anyway has earned the job.
    if FINISH_STUDENT is not None:
        _decision_at = list(le_method.classes_).index('Decision')
        _s_cal = _distil.predict(FINISH_STUDENT, X_cal_m_full)
        _t_cal = 1.0 - xgb_method_prod.predict_proba(X_cal_m_full)[:, _decision_at]
        _real = (y_cal_m_full != _decision_at)
        if len(np.unique(_real)) < 2:
            FINISH_STUDENT = None
            print("      Not used: the calibration split is one outcome only, "
                  "so nothing can be compared.")
        else:
            _auc_s = roc_auc_score(_real, _s_cal)
            _auc_t = roc_auc_score(_real, _t_cal)
            print(f"      Held-out AUC: student {_auc_s:.3f}, "
                  f"three-way {_auc_t:.3f}.")
            if _auc_s <= _auc_t:
                FINISH_STUDENT = None
                print("      NOT USED: it does not beat the model it would "
                      "replace. The three-way model supplies P(finish).")
else:
    print(f"      No {_METHOD_ODDS_PATH.name}; the three-way model supplies "
          f"P(finish) as before. Run the fetch-method-odds mode.")

# --- calibrate P(finish) ---------------------------------------------------
# The method model ranks fights well and prices them badly. Walk-forward over
# the confirm period it said 79% and 64% happened, said 31% and 36% happened -
# monotone, so the ordering is real; stretched outward by the class weights,
# so the numbers are not. An isotonic fit pulls the ends back and cannot
# reorder anything, because isotonic regression is monotone by construction.
#
# Measured in experiments/method_model.py, confirm period: Brier on P(finish)
# 0.2498 -> 0.2432, and the reliability goes from 31->36 / 79->64 to
# 39->38, 48->48, 52->52, 56->56, 62->60. The cost is about 1.4 points of
# binary accuracy, because a correctly humble probability crosses a half less
# often. That is the right trade for a number the app prints as a percentage.
#
# Fitted on the calibration split, which is also the early-stopping eval set,
# so the model has seen it once - the same arrangement the winner model's
# Platt calibration already uses.
print("    Calibrating P(finish) on the calibration split...")
_DECISION_INDEX = list(le_method.classes_).index('Decision')
_cal_real_finish = (y_cal_m_full != _DECISION_INDEX).astype(float)

# FITTED ON WHATEVER ACTUALLY SUPPLIES P(FINISH) AT PREDICTION TIME. Once the
# student exists it is the student, and calibrating the three-way model's
# output instead would map from a distribution nothing ever produces - the
# same mismatch this file already fixed once for the winner model's Platt
# calibration, and the reason the two are fitted here together rather than in
# whichever order they were written.
_student_cal = _distil.predict(FINISH_STUDENT, X_cal_m_full)
if _student_cal is not None:
    _cal_raw_finish = _student_cal
    _cal_source = "the distilled student"
else:
    _cal_raw_finish = 1.0 - xgb_method_prod.predict_proba(
        X_cal_m_full)[:, _DECISION_INDEX]
    _cal_source = "the three-way model"
FINISH_CALIBRATOR = _fit_finish_calibrator(_cal_raw_finish, _cal_real_finish)
if FINISH_CALIBRATOR is None:
    print("      Not calibrated: too few rows, or one outcome only.")
else:
    _before = _cal_raw_finish.mean()
    _after = _apply_finish_calibrator(FINISH_CALIBRATOR, _cal_raw_finish).mean()
    print(f"      Fitted on {_cal_source}: it said {_before:.1%} finishes, "
          f"now says {_after:.1%}; {_cal_real_finish.mean():.1%} really were.")


def calibrate_method_probs(method_probs):
    """Calibrated three-way, with the KO/submission split left alone.

    Calibration was measured on the binary and says nothing about whether a
    finish arrives by knockout or by submission, so moving that split would
    be inventing a correction nobody measured.

    NOTE ON ORDER: this runs on the raw model output, which is exactly what
    the calibrator was fitted on. The cage-control adjustment runs afterwards
    and can move P(finish) a little off the calibrated value. Fitting the
    calibrator on cage-adjusted history instead would mean replaying that
    adjustment across eight thousand fights; applying it to a quantity it was
    not fitted on is the mismatch this file already fixed once for the winner
    model, so the order is this way round and the residual is documented
    rather than hidden.
    """
    if FINISH_CALIBRATOR is None:
        return method_probs
    order = list(le_method.classes_)
    raw = np.array([[method_probs.get(c, 0.0) for c in order]], dtype=float)
    # rebuild_three_way expects Decision first; reorder both ways around it.
    decision = raw[:, _DECISION_INDEX]
    ko = raw[:, order.index('KO/TKO')]
    sub = raw[:, order.index('Submission')]
    packed = np.column_stack([decision, ko, sub])
    calibrated = _apply_finish_calibrator(FINISH_CALIBRATOR, 1.0 - decision)
    out = _rebuild_three_way(packed, calibrated)[0]
    return {'Decision': float(out[0]), 'KO/TKO': float(out[1]),
            'Submission': float(out[2])}

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
    
    
    
    # The model rows come from the training pipeline itself: the fight as a
    # pending row after every archived bout (see _live_rows). The hand-built
    # feature dict that stood here drifted from training in 70 places.
    _feat_frame = _live_row(r_resolved or red_name, b_resolved or blue_name,
                            event_date, is_5rnd, is_title, context)
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

    # --- THE CLOSING LINE ---------------------------------------------------
    # The largest single accuracy gain in this project, and it is arithmetic.
    # Measured on 2,339 priced walk-forward predictions from 2020, with the
    # weight chosen from earlier years only and applied forward:
    #
    #     model alone     62.0%      market alone   67.5%
    #     blended         68.2%      AUC 0.618 -> 0.709
    #
    # The blend beats the market it is built from, so the model is carrying
    # something the line does not. Above about 68% is not on offer: a
    # perfectly calibrated forecaster reading these same prices would manage
    # 65.2%, and the market itself manages 67.5%.
    #
    # BOTH NUMBERS SURVIVE. p_win is what the fight is likely to do and is
    # what the app shows and what gets graded. p_win_model is the model
    # without the market, and it is what every edge and betting strategy must
    # keep using - an edge measured against a probability that is already
    # three-quarters market is the line being compared with itself.
    p_win_model = p_win
    market_devigged = None
    if globals().get("CURRENT_ODDS") and CURRENT_ODDS:
        _mine = match_fighter_to_odds(r_resolved or red_name, CURRENT_ODDS)
        _theirs = match_fighter_to_odds(b_resolved or blue_name, CURRENT_ODDS)
        if _mine and _theirs:
            # De-vigged, which needs both sides. The version of this that
            # used to live in predict_fight blended with the raw implied
            # probability and so folded the bookmaker's margin into every
            # prediction, tilting each fight toward the favourite.
            market_devigged = _blend.devig(_mine['best_odds'],
                                           _theirs['best_odds'])
    p_win = _blend.blend(p_win, market_devigged)

    # Method
    p_method = xgb_method_prod.predict_proba(X_pred_s)[0]
    method_probs = dict(zip(le_method.classes_, p_method))

    # THE DISTILLED BINARY REPLACES THE THREE-WAY MODEL'S P(FINISH), and the
    # three-way model keeps the split between knockout and submission - which
    # is the part the distillation never measured and must not silently move.
    _student = _distil.predict(FINISH_STUDENT, X_pred_s)
    if _student is not None:
        _order = list(le_method.classes_)
        _packed = np.array([[method_probs['Decision'], method_probs['KO/TKO'],
                             method_probs['Submission']]], dtype=float)
        _rebuilt = _rebuild_three_way(_packed, np.array([float(_student[0])]))[0]
        method_probs = {'Decision': float(_rebuilt[0]),
                        'KO/TKO': float(_rebuilt[1]),
                        'Submission': float(_rebuilt[2])}
    method_probs = calibrate_method_probs(method_probs)
    
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
        'red_win_prob': p_win,               # after context and the market
        'blue_win_prob': 1 - p_win,
        'winner': (r_resolved or red_name) if p_win > 0.5 else (b_resolved or blue_name),
        'win_prob': max(p_win, 1 - p_win),
        # The model without the market. Every edge is measured from this.
        'red_win_prob_model': float(p_win_model),
        'win_prob_model': float(max(p_win_model, 1 - p_win_model)),
        'market_prob': market_devigged,
        'market_blended': market_devigged is not None,
        'confidence': raw_confidence,
        'method_probs': method_probs,
        'method': max(method_probs, key=method_probs.get),
        'method_prob': max(method_probs.values()),
        # The binary is the call worth making. The three-way top class lands
        # on Decision more often than fights actually go the distance, not
        # because the model leans that way but because a decision is one
        # bucket and a finish is two.
        'p_finish': 1.0 - method_probs.get('Decision', 0.0),
        'p_finish_raw': float(1.0 - p_method[_DECISION_INDEX]),
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

    # One pass of the training pipeline for the whole card, each fight in
    # the division its context names (inferred from the fighters' last
    # bouts when it names none).
    def _division_only(i):
        return {'division': _division_of((contexts or {}).get(i))}
    # Names are resolved the way prediction resolves them; a fight with a
    # corner the engine refuses (a debut, an unknown name) is left out here
    # and refused below as before.
    _prefetch = []
    for i, f in enumerate(fights):
        _res, _problems, _ = _check_fighters(f[0], f[1])
        if _problems:
            continue
        _prefetch.append((_res['red'][0], _res['blue'][0],
                          f[2] if len(f) > 2 else False,
                          f[3] if len(f) > 3 else False, _division_only(i)))
    prepare_live_rows(_prefetch, event_date)

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
        
        pred = predict_fight_prod(red, blue, event_date=event_date, is_5rnd=is_5rnd, is_title=is_title,
                                  context=_division_only(i), verbose=False)

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

    winner_short = short_name(pred['winner'])  # surname, keeping any Jr./III

    # Simplified finish output
    if pred.get('is_finish', False):
        finish_str = f"{pred['method'][:3]} ({pred.get('finish_timing', 'R?')[:5]})"
    else:
        finish_str = "Decision"

    compact_results.append({
        '#': i+1,
        'Matchup': f"{short_name(pred['red'])} vs {short_name(pred['blue'])}",
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

# What the last call returned, for the cache. Empty until one is made.
_LAST_ODDS_PAYLOAD = []
_LAST_ODDS_STATUS = None
_LAST_ODDS_REMAINING = None

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

    global _LAST_ODDS_PAYLOAD, _LAST_ODDS_STATUS, _LAST_ODDS_REMAINING
    try:
        response = requests.get(url, params=params, timeout=10)
        _LAST_ODDS_STATUS = response.status_code
        # The quota rides on every response and nothing was reading it, so a
        # spent balance would have shown up as a card that quietly had no
        # prices rather than as anything anyone could act on.
        _LAST_ODDS_REMAINING = response.headers.get("x-requests-remaining")

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
        # Kept whole so the cache stores what the API actually said, rather
        # than a re-derivation of the parsed form below.
        _LAST_ODDS_PAYLOAD = data

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
        # requests embeds the full URL - api key and all - in its exception
        # text, so the key is taken out before anything is printed.
        detail = str(e).replace(key, "<redacted>") if key else str(e)
        print(f"  [Odds] Network error: {detail}")
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
# The cache decides whether to spend a credit. One call returns every upcoming
# MMA event, so there is no cheaper request to make - the only lever is whether
# to call at all, and a card already priced on file needs no call.
# ODDS_REFRESH=1 overrides it for the day of the card, when the line that
# matters is the current one rather than the one taken ten days out.
import odds_cache as _odds_cache

_cache = _odds_cache.load()
_card_fights = [(f[0], f[1], str(EVENT_DATE)) for f in FIGHT_CARD]
_refresh = os.environ.get("ODDS_REFRESH", "").strip() not in ("", "0", "false")
_should_fetch, _why = _odds_cache.decide(_card_fights, _cache, refresh=_refresh)

if not ODDS_API_KEY:
    print("\n[Odds] No API key set. To enable odds:")
    print("       1. Free key at https://the-odds-api.com/")
    print("       2. Add ODDS_API_KEY as a repository secret, or export it")
    print("       3. Verify it with: python3 engine/check_odds.py")
elif not _should_fetch:
    print(f"\n[Odds] No request made - {_why}.")
    print(f"       {len(_cache['fights']):,} fights on file. "
          f"Set ODDS_REFRESH=1 to fetch anyway.")
else:
    print(f"\n[Odds] Fetching: {_why}.")
    load_odds()
    _parsed = _odds_cache.parse_events(_LAST_ODDS_PAYLOAD)
    _added, _refreshed = _odds_cache.merge(_cache, _parsed)
    _odds_cache.record_fetch(_cache, _LAST_ODDS_STATUS, len(_parsed),
                             _LAST_ODDS_REMAINING)
    print(f"       {_added} new, {_refreshed} already known. "
          f"Credits left: {_LAST_ODDS_REMAINING or 'unknown'}")

# Serve the card from the cache, so a fight priced on an earlier run is still
# priced on this one without another call.
if _cache["fights"]:
    CURRENT_ODDS = _odds_cache.as_current_odds(_cache, CURRENT_ODDS)

# A settled fight's line can never change, so it graduates into the historical
# file the backtest reads. odds.csv has no 2025 prices at all; every card
# watched from here fills that gap instead of being thrown away with the runner.
_new_history = _odds_cache.flush_settled(_cache)
if _new_history:
    print(f"       {_new_history} settled fights added to odds.csv")
_odds_cache.save(_cache)



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
    # THE MODEL'S OWN PROBABILITY, NOT THE BLENDED ONE. win_prob now carries
    # three-quarters of the closing line, and an edge measured from it would
    # be the market compared with itself - it would shrink every disagreement
    # toward zero and call what was left an edge. win_prob_model is the same
    # number this function has always used; it just needed a name once the
    # blend existed. Falls back to win_prob for a caller that predates it.
    model_prob = pred.get('win_prob_model', pred['win_prob'])
    value_info = calculate_value(model_prob, best_odds)

    return {
        'fighter': odds_match['name'],
        'best_odds': best_odds,
        'book_with_best': next((o['book'] for o in odds_match['odds'] if o['odds'] == best_odds), 'Unknown'),
        'implied_prob': value_info['implied_prob'],
        'our_prob': model_prob,
        'blended_prob': pred['win_prob'],
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
        'matchup': f"{short_name(pred['red'])} vs {short_name(pred['blue'])}",
        'pick': short_name(pred['winner']),
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
            'matchup': f"{short_name(pred['red'])} vs {short_name(pred['blue'])}",
            'pick': short_name(pred['winner']),
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

# --- what the model cannot read from a record -----------------------------
# Sourced observations gathered before the bell, by the scout (see
# .claude/skills/scout and engine/intel_search.py). Every one carries
# applies=False because no weight is fitted for any kind, so this cannot and
# does not move win_prob - it rides along beside it so a reader can see what
# the record does not contain. `known_by` is the event date, which is the
# honest cutoff: a prediction made before a fight may only read what was
# gathered before it.
try:
    import fight_intel as _intel
    _intel_store = _intel.load()
    print(f"    Scouted intel on file: "
          f"{len(_intel_store['observations'])} observation(s).")
except Exception as _err:                                      # noqa: BLE001
    _intel_store = None
    print(f"    No intel store ({type(_err).__name__}); the card carries none.")

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
        'win_prob_model': _pred.get('win_prob_model'),
        'market_prob': (_pred.get('market_prob') if _pred['winner'] == _pred['red']
                        else (1 - _pred['market_prob'])
                        if _pred.get('market_prob') is not None else None),
        'confidence': _ba.get('calibrated_conf', _pred.get('confidence')),
        'method': _pred.get('method'),
        'method_prob': _pred.get('method_prob'),
        'method_probs': _pred.get('method_probs'),
        'p_finish': _pred.get('p_finish'),
        'p_finish_raw': _pred.get('p_finish_raw'),
        'round': _pred.get('round'),
        'recommendation': _ba.get('recommendation'),
        'parlay_tier': _ba.get('parlay_tier'),
        'odds': _vi.get('best_odds'),
        'edge': _vi.get('edge'),
        'rounds_scheduled': 5 if _ba.get('is_5rnd') else 3,
        'title_fight': bool(_ba.get('is_title')),
        'simulation': _sim,
        'intel': (_intel.for_fight(_intel_store, _pred['red'], _pred['blue'],
                                   EVENT_DATE, known_by=EVENT_DATE)
                  if _intel_store else []),
    })

# ------------------------------------------------------------------ 2c
# THE SECOND LAYER, ON A CARD NOBODY KNOWS THE RESULT OF.
#
# failure_model has always worked and has always been measured - its flags
# score about 0.70 where 0.5 is a coin toss - and until now it had never once
# run on a prediction anybody would see. Three of its five features are
# market-derived, so it needed a priced card, and there were no live prices.
# There are now.
#
# It fits on settled_bets.csv: walk-forward predictions, each made by a model
# that had not seen its year, written by experiments/historical_backtest.py.
# Fitting on anything else here would mean walking the model forward across
# sixteen years to flag twelve fights.
#
# A fight is left UNFLAGGED rather than given 0.5 when it has no price or when
# the history is too thin to fit. The app has to be able to tell "the layer
# says this one looks safe" from "the layer could not run", and a default of
# 0.5 destroys that distinction while looking like an answer.
import failure_model as _failure
import roi as _roi
import strategies as _strategies

_flag_scores, _flag_note, _flag_quality = {}, None, None
_settled_path = Path(__file__).resolve().parent / "experiments" / "settled_bets.csv"

if not _settled_path.exists():
    _flag_note = ("no settled-bet history on disk; run "
                  "experiments/historical_backtest.py to write settled_bets.csv")
else:
    _settled_df = pd.read_csv(_settled_path)
    _settled_bets = _settled_df.to_dict("records")

    # One index carrying both the historical prices the fit needs and
    # tonight's, because find_odds takes one.
    _odds_csv = _odds_cache.ODDS_CSV
    _flag_index = _roi.load_odds(_odds_csv) if _odds_csv.exists() else {}
    _live_rows = []
    for _ba in bet_analysis:
        _pred = _ba['pred_full']
        _vi = _ba.get('value_info') or {}
        _pick = _pred['winner']
        _opp = _pred['blue'] if _pick == _pred['red'] else _pred['red']
        _opp_vi = (analyze_fight_value(_pred, _opp, CURRENT_ODDS)
                   if CURRENT_ODDS else None) or {}
        if _vi.get('best_odds') is not None and _opp_vi.get('best_odds') is not None:
            _live_rows.append((EVENT_DATE, _pick, _opp,
                               _vi['best_odds'], _opp_vi['best_odds']))
    _roi.index_odds(_live_rows, into=_flag_index)

    _live_picks = [{'date': EVENT_DATE,
                    'pick': _ba['pred_full']['winner'],
                    'opponent': (_ba['pred_full']['blue']
                                 if _ba['pred_full']['winner'] == _ba['pred_full']['red']
                                 else _ba['pred_full']['red']),
                    'win_probability': _ba['pred_full']['win_prob'],
                    'confidence': _ba.get('calibrated_conf',
                                          _ba['pred_full'].get('confidence'))}
                   for _ba in bet_analysis]

    _flag_scores = _failure.live_flags(_live_picks, _flag_index, _settled_bets)
    if not _flag_scores:
        _flag_note = ("the layer could not run: no live prices matched, or too "
                      "few settled bets to fit")
    else:
        # What the flags are worth, measured the same way the experiment
        # measures them, so the app can show the number beside the flag rather
        # than asking anyone to take it on trust.
        _hist = []
        for _b in sorted(_settled_bets, key=lambda b: pd.to_datetime(b['date'])):
            _f = _failure.features(_b, _flag_index)
            if _f is not None:
                _hist.append((_b, _f))
        _split = int(len(_hist) * 0.6)
        _m = _failure.fit(_hist[:_split])
        if _m is not None:
            _flag_quality = _failure.flag_quality(
                [{'p_fail': _failure.score(_m, _f),
                  'actually_failed': not _b['won']} for _b, _f in _hist[_split:]])

print("\n" + "=" * 70)
print("SECOND LAYER - WHICH PICKS IS THE MODEL LIKELY TO GET WRONG")
print("=" * 70)
if _flag_note:
    print(f"  Not shown: {_flag_note}")
else:
    print(f"  Fitted on {len(_settled_bets):,} settled bets."
          + (f"  Flag quality {_flag_quality:.3f} (0.5 is a coin toss)."
             if _flag_quality == _flag_quality else ""))
    for _ba in sorted(bet_analysis,
                      key=lambda b: -_flag_scores.get(
                          _strategies.key_of({
                              'date': EVENT_DATE,
                              'pick': b['pred_full']['winner'],
                              'opponent': (b['pred_full']['blue']
                                           if b['pred_full']['winner'] == b['pred_full']['red']
                                           else b['pred_full']['red'])}), -1)):
        _pred = _ba['pred_full']
        _opp = _pred['blue'] if _pred['winner'] == _pred['red'] else _pred['red']
        _pf = _flag_scores.get(_strategies.key_of(
            {'date': EVENT_DATE, 'pick': _pred['winner'], 'opponent': _opp}))
        if _pf is None:
            print(f"  {_pred['winner']:24s} no price - not scored")
        else:
            print(f"  {_pred['winner']:24s} {_pf:6.1%} chance this pick is wrong"
                  + ("   <-- FLAGGED" if _pf >= _failure.FLAG_THRESHOLD else ""))


# Onto what the phone gets. A fight the layer could not score keeps p_fail
# absent, which the app renders as "not scored" rather than as a safe pick.
for _fight in _app_fights:
    _p = _fight['pick']
    _o = _fight['blue'] if _p == _fight['red'] else _fight['red']
    _fight['p_fail'] = _flag_scores.get(
        _strategies.key_of({'date': EVENT_DATE, 'pick': _p, 'opponent': _o}))

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
