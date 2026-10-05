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


def record_row(target, fid, first, bouts_):
    return {"target": target, "id": fid, "first_contender": first,
            "record": [{"date": d, "result": r, "opponent": o,
                        "opponent_id": oid, "event": e, "method": "KO",
                        "round": "1", "time": "1:00"}
                       for d, r, o, oid, e in bouts_]}


def test_sherdog_bouts_are_named_as_the_world_frame_names_them():
    records = [
        record_row("Jose Aldo Jr", 1, "2021-08-01", [
            ("2020-01-01", "win", "Ann Other", 2, "LFA 1"),
            ("2019-01-01", "loss", "Bob Stranger", 3, "Regional 2")]),
        # Ann was looked up too - under the name her Contender page uses.
        record_row("Anne Other", 2, "2022-08-01", [
            ("2020-01-01", "loss", "Jose Aldo", 1, "LFA 1")]),
    ]
    got = wm.sherdog_bouts(records)
    # The bout both records hold is kept once, with both names mapped.
    lfa = got[got["event"] == "LFA 1"]
    assert len(lfa) == 1
    assert (lfa.iloc[0]["winner"], lfa.iloc[0]["loser"]) == ("Jose Aldo Jr",
                                                            "Anne Other")
    loss = got[got["event"] == "Regional 2"].iloc[0]
    assert (loss["winner"], loss["loser"]) == ("Bob Stranger", "Jose Aldo Jr")


def test_a_bout_already_on_an_event_page_is_not_added_twice():
    world = bouts([("2020-01-02", "LFA 1", "Ann", "Bea", "win", "KO", "x")])
    extra = bouts([("2020-01-01", "LFA 1", "Bea", "Ann", "win", "KO", "s"),
                   ("2020-03-01", "LFA 2", "Ann", "Cat", "win", "KO", "s")])
    assert list(wm.not_in(extra, world)["event"]) == ["LFA 2"]


def test_a_record_is_unknown_before_its_owner_reached_the_contender_series():
    world = bouts([("2019-08-01", "Contender Series 1", "Ann", "Bea", "win",
                    "KO", "x"),
                   ("2021-08-01", "Contender Series 2", "Cat", "Dee", "win",
                    "KO", "x")])
    extra = bouts([("2018-01-01", "LFA 1", "Cat", "Eve", "win", "KO", "s"),
                   ("2019-01-01", "LFA 2", "Ann", "Fay", "win", "KO", "s"),
                   ("2019-09-01", "LFA 3", "Ann", "Gil", "win", "KO", "s")])
    extra["disclosed"] = pd.to_datetime(["2021-08-01", "2019-08-01",
                                         "2019-08-01"])
    at_2019 = wm.gated(world, extra, pd.Timestamp("2019-08-01"))
    # Cat's 2018 bout is not yet known (Cat is looked up in 2021), and
    # Ann's later bout is after the date: only Ann's earlier bout joins.
    assert set(at_2019["event"]) == {"Contender Series 1", "LFA 2"}
    at_2021 = wm.gated(world, extra, pd.Timestamp("2021-08-01"))
    assert {"LFA 1", "LFA 3"} <= set(at_2021["event"])


def test_age_gap_needs_both_birth_dates_and_is_signed_toward_fighter_a():
    births = {"ann": pd.Timestamp("1990-01-01"), "bea": pd.Timestamp("2000-01-01")}
    assert abs(wm._age_diff(births, "ann", "bea") - 1.0) < 0.01   # 10y older
    assert wm._age_diff(births, "ann", "cat") == 0.0
    assert wm._age_diff(None, "ann", "bea") == 0.0


def test_birth_dates_become_known_with_the_record():
    got = wm.births_of([
        {"target": "Ann Lee", "id": 1, "first_contender": "2021-08-03",
         "bio": {"birth_date": "1995-02-01"}},
        {"target": "Bea Kim", "id": None, "first_contender": "2021-08-03"},
        {"target": "Cat Poe", "id": 3, "first_contender": "2021-08-03",
         "bio": {"birth_date": None}}])
    assert got == {"ann lee": (pd.Timestamp("1995-02-01"),
                               pd.Timestamp("2021-08-03"))}
