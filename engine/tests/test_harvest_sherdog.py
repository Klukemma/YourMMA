"""Who is looked up on Sherdog, and that a stopped harvest keeps its work."""

import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import harvest_sherdog as hs


def world(tmp_path):
    frame = pd.DataFrame([
        ("2021-08-03", "Dana White's Contender Series 41", "Ann", "Bea",
         "season:Contender Series season 5"),
        ("2021-08-10", "Week 2 – August 10", "Cat", "Dee",
         "Dana White's Contender Series season 5"),
        # Known only from Ann's own article: never chooses who is looked up.
        ("2019-01-01", "Contender Series 9", "Eve", "Ann", "record:Ann"),
        ("2021-09-01", "LFA 100", "Fay", "Gil", "2021 in LFA"),
    ], columns=["date", "event", "winner", "loser", "source"])
    path = tmp_path / "bouts.csv.gz"
    frame.to_csv(path, index=False)
    return path


def test_both_fighters_of_every_contender_bout_and_nobody_else(tmp_path):
    got = hs.contender_appearances(world(tmp_path))
    assert set(got) == {"Ann", "Bea", "Cat", "Dee"}
    assert got["Bea"] == [("2021-08-03", "Ann")]


def test_a_stopped_run_keeps_its_work_and_the_next_one_carries_on(
        tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "BOUTS", world(tmp_path))
    out = tmp_path / "records.jsonl.gz"
    monkeypatch.setattr(hs, "OUT", out)
    asked = []

    def find(name, agent, bouts):
        asked.append(name)
        if name == "Bea":
            return None
        return {"id": len(asked), "slug": name, "name": name,
                "verified": True, "bio": {}, "record": []}
    monkeypatch.setattr(hs.sherdog, "find", find)
    hs.main(["--minutes", "0"])            # budget already spent
    assert asked == [] and hs.load() == {}
    hs.main([])
    first = hs.load()
    assert set(first) == {"Ann", "Bea", "Cat", "Dee"}
    assert first["Ann"]["first_contender"] == "2021-08-03"
    assert first["Bea"]["id"] is None
    asked.clear()
    hs.main([])                            # nothing left to do
    assert asked == []
    hs.main(["--retry-missing"])
    assert asked == ["Bea"]
