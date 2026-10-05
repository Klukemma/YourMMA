"""Size marks read only bouts before the fight."""

import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.size_edge import History, populations


def archive():
    return pd.DataFrame([
        ("2019-01-01", "welterweight", "Ann", "Bea", "Ann", 180, 175, 185, 180),
        ("2020-01-01", "lightweight", "Ann", "Cat", "Ann", 180, 170, 185, 175),
        ("2020-06-01", "lightweight", "Cat", "Dee", "Cat", 170, 170, 175, 175),
        ("2021-01-01", "lightweight", "Ann", "Cat", "Cat", 180, 170, 185, 175),
        ("2022-01-01", "welterweight", "Cat", "Eve", "Cat", 170, 170, 175, 175),
    ], columns=["date", "division", "r_name", "b_name", "winner", "r_height",
                "b_height", "r_reach", "b_reach"])


def row(date, division, red, blue):
    return pd.Series({"date": pd.Timestamp(date), "division": division,
                      "red_raw": red, "blue_raw": blue})


def test_heavier_history_and_moves_use_only_earlier_bouts():
    pops = populations(History(archive()))
    fight = row("2020-01-01", "lightweight", "Ann", "Cat")
    # Cat has no UFC bout before this one: "never heavier" is unknowable.
    assert pops["fought_heavier_before"](fight) is None
    assert pops["first_fight_down"](fight) == "red"
    later = row("2021-01-01", "lightweight", "Ann", "Cat")
    assert pops["fought_heavier_before"](later) == "red"
    assert pops["first_fight_down"](later) is None     # Ann's last was 155
    up = row("2022-01-01", "welterweight", "Cat", "Eve")
    assert pops["first_fight_up"](up) == "red"
    # Cat's later welterweight bout is not seen from 2021.
    assert History(archive()).size("Cat", "2021-01-01")[0] == 155


def test_frame_edge_needs_both_height_and_reach():
    pops = populations(History(archive()))
    assert pops["taller_and_longer"](row("2020-01-01", "lightweight",
                                         "Ann", "Cat")) == "red"
    assert pops["taller_and_longer"](row("2020-01-01", "catch weight",
                                         "Ann", "Cat")) is None


def test_missed_weight_marks_the_fighter_who_missed(tmp_path):
    path = tmp_path / "intel.csv"
    pd.DataFrame([
        ("2021-01-01", "Ann", "missed_weight", 1.5),
        ("2021-01-01", "Cat", "stepped_in", None),
        ("2022-01-01", "Eve", "missed_weight", None),
    ], columns=["event_date", "fighter", "kind", "over_by_lbs"]).to_csv(path, index=False)
    from experiments.size_edge import misses
    pops = populations(History(archive()), misses(path))
    assert pops["missed_weight"](row("2021-01-01", "lightweight", "Ann", "Cat")) == "red"
    assert pops["missed_by_2lb"](row("2021-01-01", "lightweight", "Ann", "Cat")) is None
    # A miss with no stated margin counts as a miss, not as 2+ lb.
    assert pops["missed_weight"](row("2022-01-01", "welterweight", "Cat", "Eve")) == "blue"
    assert pops["missed_by_2lb"](row("2022-01-01", "welterweight", "Cat", "Eve")) is None
