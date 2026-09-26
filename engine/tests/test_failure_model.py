"""Tests for the flagging model.

The whole value of a flag is that it was decided before the fight. A leak
here - fitting on the bet being scored, or on bets that settled after it -
would produce a flattering number and a strategy that loses real money.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from failure_model import COLUMNS, features, flag_quality, walk_forward_flags
from strategies import key_of


def bet(pick, opponent, won, p=0.7, date="2026-01-24"):
    return {"date": pd.Timestamp(date), "event": "UFC 1", "pick": pick,
            "opponent": opponent, "win_probability": p,
            "confidence": abs(2 * p - 1), "won": won}


def odds_for(pairs):
    index = {}
    for (a, b), (date, oa, ob) in pairs.items():
        index.setdefault((a, b), []).append((pd.Timestamp(date), oa, ob))
        index.setdefault((b, a), []).append((pd.Timestamp(date), ob, oa))
    return index


ODDS = odds_for({("alpha", "beta"): ("2026-01-24", -200.0, 170.0)})


# --- features --------------------------------------------------------------

def test_features_need_a_price_on_both_fighters():
    assert features(bet("Alpha", "Nobody", True), ODDS) is None


def test_the_market_probability_has_the_margin_removed():
    """-200 and +170 imply 66.7% and 37.0%, which sum to more than one."""
    f = features(bet("Alpha", "Beta", True), ODDS)
    assert 0.6 < f["market_fair"] < 0.68


def test_disagreement_is_the_model_minus_the_market():
    f = features(bet("Alpha", "Beta", True, p=0.9), ODDS)
    assert f["disagreement"] == pytest.approx(0.9 - f["market_fair"])


def test_backing_the_underdog_is_flagged():
    favourite = features(bet("Alpha", "Beta", True), ODDS)
    underdog = features(bet("Beta", "Alpha", True), ODDS)
    assert favourite["backing_underdog"] == 0.0
    assert underdog["backing_underdog"] == 1.0


def test_every_declared_column_is_produced():
    f = features(bet("Alpha", "Beta", True), ODDS)
    assert set(f) == set(COLUMNS)


# --- the walk-forward guarantee -------------------------------------------

def many_bets(n=120):
    """Alternating results so both outcomes exist throughout.

    Each fight's odds carry that fight's own date: odds are matched within ten
    days, so a single shared date would put most of these out of range and
    they would be skipped for want of a price.
    """
    pairs, bets = {}, []
    for i in range(n):
        a, b = f"fighter{i}a", f"fighter{i}b"
        date = (pd.Timestamp("2026-01-05") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        pairs[(a, b)] = (date, -150.0, 130.0)
        bets.append(bet(f"Fighter{i}a", f"Fighter{i}b", won=(i % 3 != 0),
                        p=0.55 + (i % 5) / 50, date=date))
    return bets, odds_for(pairs)


def test_the_first_bets_are_never_flagged():
    """With too little settled history there is nothing honest to fit, so the
    answer is 'unknown' rather than a guess."""
    bets, odds = many_bets()
    _, rows = walk_forward_flags(bets, odds, min_train=40)
    assert len(rows) == len(bets) - 40


def test_a_flag_is_only_ever_decided_from_earlier_bets():
    """Removing everything after a bet must not change its flag."""
    bets, odds = many_bets()
    ordered = sorted(bets, key=lambda b: pd.to_datetime(b["date"]))
    _, full = walk_forward_flags(ordered, odds, min_train=40)
    cut = len(ordered) - 10
    _, truncated = walk_forward_flags(ordered[:cut], odds, min_train=40)
    shared = {r["key"]: r["p_fail"] for r in truncated}
    for row in full:
        if row["key"] in shared:
            assert row["p_fail"] == pytest.approx(shared[row["key"]])


def test_flagged_keys_match_the_fights_they_came_from():
    bets, odds = many_bets()
    flagged, rows = walk_forward_flags(bets, odds, min_train=40, threshold=0.0)
    assert flagged == {r["key"] for r in rows}


def test_a_higher_threshold_flags_fewer_fights():
    bets, odds = many_bets()
    loose, _ = walk_forward_flags(bets, odds, min_train=40, threshold=0.2)
    strict, _ = walk_forward_flags(bets, odds, min_train=40, threshold=0.9)
    assert len(strict) <= len(loose)


def test_unpriced_bets_are_skipped_rather_than_guessed():
    bets, odds = many_bets()
    bets.append(bet("Ghost", "Phantom", True))
    _, rows = walk_forward_flags(bets, odds, min_train=40)
    assert all("ghost" not in r["key"][1] for r in rows)


# --- quality ---------------------------------------------------------------

def test_perfect_flags_score_one():
    rows = [{"p_fail": 0.9, "actually_failed": True} for _ in range(15)]
    rows += [{"p_fail": 0.1, "actually_failed": False} for _ in range(15)]
    assert flag_quality(rows) == pytest.approx(1.0)


def test_useless_flags_score_a_coin_toss():
    rng = np.random.default_rng(0)
    rows = [{"p_fail": 0.5, "actually_failed": bool(rng.integers(2))}
            for _ in range(200)]
    assert flag_quality(rows) == pytest.approx(0.5, abs=0.05)


def test_too_few_scored_picks_makes_no_claim():
    assert np.isnan(flag_quality([{"p_fail": 0.5, "actually_failed": True}]))


def test_all_one_outcome_makes_no_claim():
    rows = [{"p_fail": 0.5, "actually_failed": True} for _ in range(30)]
    assert np.isnan(flag_quality(rows))
