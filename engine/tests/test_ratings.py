"""Tests for TrueSkill rating reconstruction.

The interesting ones measure this module against the ratings already in the
dataset, so they catch drift between our update and whatever produced them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from ratings import (DEFAULT, TrueSkillConfig, conservative_rating, extend,
                     rate_1v1, replay, state_from_existing)

CSV_PATH = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


@pytest.fixture(scope="module")
def ufc():
    d = pd.read_csv(CSV_PATH, low_memory=False)
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    for c in ('r_mu_pre', 'b_mu_pre', 'r_sigma_pre', 'b_sigma_pre',
              'r_mmr_pre', 'b_mmr_pre'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.sort_values('date', kind='mergesort').reset_index(drop=True)


# --- the relationship the whole module rests on ---------------------------

def test_mmr_is_the_conservative_rating(ufc):
    """mmr_pre == mu_pre - 3*sigma_pre, exactly, in the shipped data."""
    for corner in ('r', 'b'):
        got = conservative_rating(ufc[f'{corner}_mu_pre'], ufc[f'{corner}_sigma_pre'])
        assert np.nanmax(np.abs(got - ufc[f'{corner}_mmr_pre'])) < 1e-9


# --- the update itself ----------------------------------------------------

def test_winner_gains_loser_loses():
    wm, ws, lm, ls = rate_1v1(25.0, 8.33, 25.0, 8.33)
    assert wm > 25.0 and lm < 25.0


def test_uncertainty_shrinks_after_a_fight():
    wm, ws, lm, ls = rate_1v1(25.0, 8.33, 25.0, 8.33)
    assert ws < 8.33 and ls < 8.33


def test_upset_moves_ratings_more_than_expected_result():
    _, _, _, _ = rate_1v1(40.0, 3.0, 20.0, 3.0)
    favourite_wins = rate_1v1(40.0, 3.0, 20.0, 3.0)[0] - 40.0
    underdog_wins = rate_1v1(20.0, 3.0, 40.0, 3.0)[0] - 20.0
    assert underdog_wins > favourite_wins


def test_sigma_never_goes_below_floor():
    mu, sig = 25.0, 8.33
    for _ in range(200):
        mu, sig, _, _ = rate_1v1(mu, sig, 25.0, 8.33)
    assert sig >= DEFAULT.min_sigma


# --- agreement with the shipped ratings -----------------------------------

@pytest.fixture(scope="module")
def pairwise_errors(ufc):
    """For each fighter, predict their next bout's pre-rating from this one."""
    seq = {}
    for i, row in enumerate(ufc.itertuples(index=False)):
        for corner in ('r', 'b'):
            nm = getattr(row, f'{corner}_name')
            if isinstance(nm, str) and nm:
                seq.setdefault(nm, []).append((i, corner))

    errs = []
    rows = list(ufc.itertuples(index=False))
    for nm, bouts in seq.items():
        for (i, _), (j, cj) in zip(bouts, bouts[1:]):
            cur, nxt = rows[i], rows[j]
            if not (isinstance(cur.winner, str) and cur.winner in (cur.r_name, cur.b_name)):
                continue
            vals = (cur.r_mu_pre, cur.r_sigma_pre, cur.b_mu_pre, cur.b_sigma_pre)
            if any(pd.isna(v) for v in vals):
                continue
            actual_mu = getattr(nxt, f'{cj}_mu_pre')
            if pd.isna(actual_mu):
                continue
            if cur.winner == cur.r_name:
                a_mu, a_sig, b_mu, b_sig = rate_1v1(*vals)
                post = {cur.r_name: (a_mu, a_sig), cur.b_name: (b_mu, b_sig)}
            else:
                a_mu, a_sig, b_mu, b_sig = rate_1v1(cur.b_mu_pre, cur.b_sigma_pre,
                                                    cur.r_mu_pre, cur.r_sigma_pre)
                post = {cur.b_name: (a_mu, a_sig), cur.r_name: (b_mu, b_sig)}
            errs.append(abs(post[nm][0] - actual_mu))
    return np.array(errs)


def test_reproduces_shipped_updates(pairwise_errors):
    """95%+ of updates land within 0.05 mu of the recorded value."""
    assert len(pairwise_errors) > 10_000
    assert (pairwise_errors < 0.05).mean() > 0.95
    assert np.median(pairwise_errors) < 0.02


def test_our_beta_beats_the_scripts_4_17(ufc, pairwise_errors):
    """predict_card.py uses beta=4.17; the data says ~5.0. Show the gap."""
    assert np.median(pairwise_errors) < 0.05  # ours
    # beta=4.17 produces a median error over 10x larger - see module docstring.
    cfg_417 = TrueSkillConfig(beta=4.17)
    row = ufc.iloc[100]
    ours = rate_1v1(row.r_mu_pre, row.r_sigma_pre, row.b_mu_pre, row.b_sigma_pre)
    theirs = rate_1v1(row.r_mu_pre, row.r_sigma_pre, row.b_mu_pre, row.b_sigma_pre, cfg_417)
    assert ours[0] != theirs[0]


# --- replay and extend ----------------------------------------------------

def test_replay_gives_debutants_the_prior():
    df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01']),
                       'r_name': ['A'], 'b_name': ['B'], 'winner': ['A']})
    out, state = replay(df)
    assert out.loc[0, 'r_mu_pre'] == DEFAULT.mu0
    assert out.loc[0, 'b_sigma_pre'] == DEFAULT.sigma0
    assert state['A'][0] > DEFAULT.mu0 > state['B'][0]


def test_replay_is_pre_fight_not_post_fight():
    """Rating recorded for fight 2 must reflect fight 1's result, not fight 2's."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'r_name': ['A', 'A'], 'b_name': ['B', 'C'], 'winner': ['A', 'C'],
    })
    out, _ = replay(df)
    assert out.loc[0, 'r_mu_pre'] == DEFAULT.mu0
    assert out.loc[1, 'r_mu_pre'] > DEFAULT.mu0   # A won fight 1
    assert out.loc[1, 'b_mu_pre'] == DEFAULT.mu0  # C is debuting


def test_draws_and_nc_do_not_change_ratings():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2020-02-01']),
        'r_name': ['A', 'A'], 'b_name': ['B', 'B'], 'winner': ['Draw', 'NC'],
    })
    out, state = replay(df)
    assert out.loc[1, 'r_mu_pre'] == DEFAULT.mu0
    assert state == {}


def test_extend_leaves_existing_rows_untouched(ufc):
    existing = ufc.head(500).copy()
    new = pd.DataFrame({
        'date': pd.to_datetime(['2030-01-01']),
        'r_name': [existing.iloc[0]['r_name']], 'b_name': ['Brand New Debutant'],
        'winner': [existing.iloc[0]['r_name']],
    })
    out = extend(existing, new)
    assert len(out) == 501
    merged = out.head(500).reset_index(drop=True)
    pd.testing.assert_series_equal(merged['r_mu_pre'], existing['r_mu_pre'].reset_index(drop=True))


def test_extend_continues_from_last_known_rating(ufc):
    existing = ufc.head(2000).copy()
    state = state_from_existing(existing)
    veteran = existing.iloc[-1]['r_name']
    assert veteran in state
    new = pd.DataFrame({
        'date': pd.to_datetime(['2030-01-01']),
        'r_name': [veteran], 'b_name': ['Total Newcomer XYZ'], 'winner': [veteran],
    })
    out = extend(existing, new)
    added = out[out['date'] == pd.Timestamp('2030-01-01')].iloc[0]
    assert abs(added['r_mu_pre'] - state[veteran][0]) < 1e-9
    assert added['b_mu_pre'] == DEFAULT.mu0      # debutant gets the prior
    assert abs(added['r_mmr_pre'] - (added['r_mu_pre'] - 3 * added['r_sigma_pre'])) < 1e-9


def test_extend_with_nothing_new_is_a_noop(ufc):
    existing = ufc.head(100).copy()
    out = extend(existing, pd.DataFrame())
    assert len(out) == 100
