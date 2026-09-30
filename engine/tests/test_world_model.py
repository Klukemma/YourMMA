"""The results-only model: ratings flow in date order and never see ahead."""

import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import world_model as wm


def bouts(rows):
    frame = pd.DataFrame(rows, columns=["date", "event", "winner", "loser",
                                        "result", "method", "source"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_a_bouts_features_never_include_its_own_result():
    frame = bouts([
        ("2020-01-01", "LFA 1", "Ann", "Bea", "win", "KO (punch)", "x"),
        ("2020-06-01", "LFA 2", "Ann", "Cat", "win", "Decision", "x"),
    ])
    rows, _ = wm.build(frame)
    first = rows.iloc[0]
    # Before their first bout both are unrated and unrecorded.
    assert first["elo_diff"] == 0 and first["exp_diff"] == 0
    second = rows.iloc[1]
    ann_is_a = second["a"] == "ann"
    # By the second bout Ann has one win and a higher rating - from bout 1.
    assert (second["elo_diff"] > 0) == ann_is_a


def test_the_winner_is_not_always_fighter_a():
    frame = bouts([(f"2020-01-{d:02d}", "E", f"W{d}", f"L{d}", "win", "KO",
                    "x") for d in range(1, 29)])
    rows, _ = wm.build(frame)
    assert 0 < rows["y"].mean() < 1


def test_draws_and_no_contests_are_not_training_rows_but_update_activity():
    frame = bouts([
        ("2020-01-01", "E", "Ann", "Bea", "draw", "Draw (split)", "x"),
        ("2020-02-01", "E", "Ann", "Bea", "nc", "No Contest", "x"),
    ])
    rows, states = wm.build(frame)
    assert rows.empty
    assert states["ann"].bouts == 1          # the draw counts, the NC doesn't
    assert states["ann"].last == pd.Timestamp("2020-02-01")


def test_beating_a_strong_fighter_is_worth_more():
    strong = [(f"2019-0{m}-01", "E", "Strong", f"X{m}", "win", "KO", "x")
              for m in range(1, 9)]
    frame = bouts(strong + [
        ("2020-01-01", "E", "Upset", "Strong", "win", "Decision", "x"),
        ("2020-01-01", "F", "Plain", "Nobody", "win", "Decision", "x")])
    _, states = wm.build(frame)
    assert states["upset"].elo > states["plain"].elo


def test_contender_series_bouts_are_flagged():
    frame = bouts([("2023-09-26", "Dana White's Contender Series season 7",
                    "Danny Silva", "Angel Pacheco", "win", "Decision", "x")])
    rows, _ = wm.build(frame)
    assert rows.iloc[0]["dwcs"] and not rows.iloc[0]["ufc"]
