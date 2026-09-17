"""Tests for the upstream -> local schema transform."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import schema_map as sm
from transform import (_clean_division, _is_title_fight, _rounds_from_format,
                       _to_seconds, carry_forward_records, transform)


# --- the map itself -------------------------------------------------------

def test_every_local_column_is_accounted_for():
    """No local column may be silently left unfilled by a sync."""
    import csv
    with open(ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv",
              newline="", encoding="utf-8") as f:
        local = next(csv.reader(f))
    covered = (set(sm.COLUMN_MAP.values()) | set(sm.DERIVED) | set(sm.RATING_COLUMNS)
               | set(sm.CARRIED_FORWARD) | set(sm.CONVERTED) | set(sm.RESOLVED)
               | set(sm.profile_columns('r')) | set(sm.profile_columns('b')))
    assert [c for c in local if c not in covered] == []


def test_map_is_injective():
    """Two upstream columns must never target the same local column."""
    values = list(sm.COLUMN_MAP.values())
    assert len(values) == len(set(values))


def test_career_and_per_fight_accuracy_stay_separate():
    """r_str_acc is a career stat; r_total_str_acc is this bout. Never merge."""
    assert 'r_str_acc' in sm.profile_columns('r')
    assert 'r_total_str_acc' in sm.DERIVED
    assert 'r_str_acc' not in sm.DERIVED


# --- conversions ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("4:31", 271), ("0:00", 0), ("5:00", 300), ("15:00", 900), ("271", 271),
])
def test_finish_time_to_seconds(text, expected):
    assert _to_seconds(text) == expected


def test_bad_finish_time_is_nan_not_zero():
    for bad in [None, "", "abc", np.nan]:
        assert pd.isna(_to_seconds(bad))


@pytest.mark.parametrize("text,expected", [
    ("3 Rnd (5-5-5)", 3), ("5 Rnd (5-5-5-5-5)", 5), ("1 Rnd (20)", 1),
    ("No Time Limit", 1),
])
def test_rounds_from_time_format(text, expected):
    assert _rounds_from_format(text) == expected


@pytest.mark.parametrize("wc,expected", [
    ("UFC Lightweight Title Bout", 1), ("UFC Lightweight Bout", 0),
    ("UFC Interim Welterweight Title Bout", 1),
])
def test_title_fight_detection(wc, expected):
    assert _is_title_fight(wc) == expected


def test_division_strips_title_and_promotion_noise():
    assert _clean_division("UFC Lightweight Title Bout") == "lightweight"
    assert _clean_division("UFC Women's Bantamweight Bout") == "women's bantamweight"


# --- the transform --------------------------------------------------------

@pytest.fixture
def master():
    return pd.DataFrame({
        'fight_id': ['f1', 'f2'],
        'event_id': ['e1', 'e1'],
        'event_name': ['UFC 300', 'UFC 300'],
        'event_date': ['2024-04-13', '2024-04-13'],
        'event_location': ['Las Vegas', 'Las Vegas'],
        'weight_class': ['UFC Lightweight Title Bout', 'UFC Featherweight Bout'],
        'method': ['KO/TKO', 'Decision - Unanimous'],
        'referee': ['Herb Dean', 'Marc Goddard'],
        'finish_time': ['4:31', '5:00'],
        'time_format': ['5 Rnd (5-5-5-5-5)', '3 Rnd (5-5-5)'],
        'rounds_fought': [2, 3],
        'r_fighter_id': ['a', 'c'], 'b_fighter_id': ['b', 'd'],
        'r_fighter_name': ['Red One', 'Red Two'],
        'b_fighter_name': ['Blue One', 'Blue Two'],
        'winner_id': ['a', None],
        'r_total_sig_landed': [50, 30], 'r_total_sig_atmp': [100, 60],
        'b_total_sig_landed': [20, 40], 'b_total_sig_atmp': [80, 50],
        'r_total_sig_str_landed_head': [30, 10],
        'r_total_sig_str_atmp_head': [60, 20],
        'r_total_ctrl_seconds': [120, 0],
        'r_reach_inches': [72.0, 70.0], 'r_slpm': [4.2, 3.1],
    })


@pytest.fixture
def fighters():
    return pd.DataFrame({
        'fighter_id': ['a', 'b', 'c', 'd'],
        'fighter_name': ['Red One', 'Blue One', 'Red Two', 'Blue Two'],
        'height': [70, 71, 69, 72], 'dob': ['1990/01/01'] * 4,
        'stance': ['Orthodox', 'Southpaw', 'Orthodox', 'Orthodox'],
        'str_acc': [50, 45, 48, 52], 'sapm': [3.0, 3.5, 2.8, 3.2],
        'str_def': [60, 55, 58, 61], 'td_avg': [1.5, 0.5, 2.0, 1.0],
        'td_acc': [40, 35, 45, 38], 'td_def': [70, 65, 72, 68],
        'sub_avg': [0.5, 0.2, 0.8, 0.3],
    })


def test_renames_and_winner_resolution(master, fighters):
    out = transform(master, fighters)
    assert out.loc[0, 'r_name'] == 'Red One'
    assert out.loc[0, 'winner'] == 'Red One'
    assert out.loc[0, 'r_sig_str_landed'] == 50
    assert out.loc[0, 'r_head_landed'] == 30
    assert out.loc[0, 'r_ctrl'] == 120
    assert out.loc[0, 'r_reach'] == 72.0
    assert out.loc[0, 'r_splm'] == 4.2


def test_draw_leaves_winner_blank_not_a_corner(master, fighters):
    """winner_id is null on fight 2 - it must not default to the red corner."""
    out = transform(master, fighters)
    assert pd.isna(out.loc[1, 'winner'])


def test_conversions_applied(master, fighters):
    out = transform(master, fighters)
    assert out.loc[0, 'match_time_sec'] == 271
    assert out.loc[0, 'total_rounds'] == 5
    assert out.loc[0, 'finish_round'] == 2
    assert out.loc[0, 'title_fight'] == 1
    assert out.loc[1, 'title_fight'] == 0
    assert out.loc[0, 'division'] == 'lightweight'


def test_percentages_are_computed_not_read(master, fighters):
    out = transform(master, fighters)
    assert out.loc[0, 'r_sig_str_acc'] == 50.0      # 50/100
    assert out.loc[0, 'r_head_acc'] == 50.0         # 30/60
    assert out.loc[0, 'r_landed_head_per'] == 60.0  # 30/50


def test_zero_attempts_gives_nan_not_zero_accuracy():
    df = pd.DataFrame({
        'winner_id': ['a'], 'r_fighter_id': ['a'], 'b_fighter_id': ['b'],
        'r_fighter_name': ['A'], 'b_fighter_name': ['B'],
        'r_total_td_success': [0], 'r_total_td_atmp': [0],
    })
    out = transform(df)
    assert pd.isna(out.loc[0, 'r_td_acc']), "0 of 0 takedowns is unknown, not 0%"


def test_profile_columns_joined_from_fighter_table(master, fighters):
    out = transform(master, fighters)
    assert out.loc[0, 'r_stance'] == 'Orthodox'
    assert out.loc[0, 'b_stance'] == 'Southpaw'
    assert out.loc[0, 'r_str_acc'] == 50
    assert out.loc[0, 'r_td_avg_acc'] == 40      # fighter.td_acc -> local td_avg_acc


def test_missing_winner_id_column_raises(master):
    with pytest.raises(ValueError, match="winner_id"):
        transform(master.drop(columns=['winner_id']))


# --- carried-forward records ---------------------------------------------

def test_records_carry_forward_and_increment():
    existing = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01']),
        'r_name': ['Red One'], 'b_name': ['Blue One'], 'winner': ['Red One'],
        'r_wins': [10.0], 'r_losses': [2.0], 'r_draws': [0.0],
        'b_wins': [5.0], 'b_losses': [3.0], 'b_draws': [0.0],
    })
    new = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01', '2024-09-01']),
        'r_name': ['Red One', 'Red One'], 'b_name': ['Blue One', 'Blue One'],
        'winner': ['Red One', 'Blue One'],
    })
    out = carry_forward_records(existing, new).reset_index(drop=True)
    assert out.loc[0, 'r_wins'] == 10.0          # pre-fight, unchanged
    assert out.loc[1, 'r_wins'] == 11.0          # won the June bout
    assert out.loc[1, 'b_losses'] == 4.0


def test_unknown_fighter_record_is_blank_not_zero():
    existing = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01']),
        'r_name': ['Known'], 'b_name': ['Other'], 'winner': ['Known'],
        'r_wins': [10.0], 'r_losses': [2.0], 'r_draws': [0.0],
        'b_wins': [5.0], 'b_losses': [3.0], 'b_draws': [0.0],
    })
    new = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01']),
        'r_name': ['Debutant'], 'b_name': ['Known'], 'winner': ['Known'],
    })
    out = carry_forward_records(existing, new).reset_index(drop=True)
    assert pd.isna(out.loc[0, 'r_wins']), "a debutant's record is unknown, not 0-0"
    assert out.loc[0, 'b_wins'] == 10.0


# --- total strikes, aggregated from the per-round table -------------------

def test_total_strikes_summed_across_rounds():
    from transform import aggregate_total_strikes
    rounds = pd.DataFrame({
        'fight_id': ['f1', 'f1', 'f1', 'f2'],
        'round_no': [1, 2, 3, 1],
        'r_total_str_landed': [37, 20, 11, 5],
        'r_total_str_atmp': [40, 31, 15, 8],
        'b_total_str_landed': [20, 10, 4, 3],
        'b_total_str_atmp': [31, 15, 9, 6],
    })
    out = aggregate_total_strikes(rounds)
    assert out.loc['f1', 'r_total_str_landed'] == 68
    assert out.loc['f1', 'r_total_str_atmpted'] == 86
    assert out.loc['f2', 'b_total_str_landed'] == 3


def test_missing_round_table_is_tolerated(master, fighters):
    """No round.csv means those columns stay absent, not wrong."""
    out = transform(master, fighters, rounds=None)
    assert 'r_head_landed' in out.columns


def test_total_strike_columns_are_filled_when_rounds_given(master, fighters):
    rounds = pd.DataFrame({
        'fight_id': ['f1', 'f1', 'f2'],
        'round_no': [1, 2, 1],
        'r_total_str_landed': [37, 20, 5], 'r_total_str_atmp': [40, 31, 8],
        'b_total_str_landed': [20, 10, 3], 'b_total_str_atmp': [31, 15, 6],
    })
    out = transform(master, fighters, rounds)
    assert out.loc[0, 'r_total_str_landed'] == 57
    assert out.loc[0, 'r_total_str_atmpted'] == 71
    # and the accuracy derived from them
    assert out.loc[0, 'r_total_str_acc'] == pytest.approx(57 / 71 * 100, abs=0.01)


def test_round_table_without_the_columns_is_ignored(master, fighters):
    rounds = pd.DataFrame({'fight_id': ['f1'], 'round_no': [1]})
    out = transform(master, fighters, rounds)
    assert 'r_total_str_landed' not in out.columns or out['r_total_str_landed'].isna().all()
