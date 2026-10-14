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


def test_ufc_debuts_are_every_fighter_with_their_first_bout_newest_first(tmp_path):
    path = tmp_path / "archive.csv"
    pd.DataFrame([
        ("2019-05-04", "Ann", "Bea"),
        ("2021-08-03", "Ann", "Cat"),      # Ann's second bout: not her debut
        ("2023-01-14", "Dee", "Cat"),
    ], columns=["date", "r_name", "b_name"]).to_csv(path, index=False)
    got = hs.ufc_debuts(path)
    assert list(got) == ["Dee", "Cat", "Ann", "Bea"]      # newest debut first
    assert got["Ann"] == [("2019-05-04", "Bea")]
    assert got["Cat"] == [("2021-08-03", "Ann")]


def test_a_ufc_card_confirms_a_record_without_a_surname_match():
    import sherdog
    rec = [{"date": "2024-02-17", "result": "win", "opponent": "Jo Blow",
            "opponent_id": 1, "event": "UFC 298 - Volkanovski vs. Topuria",
            "method": "KO", "round": "1", "time": "1:00"}]
    assert sherdog.confirms(rec, "2024-02-17", "Joseph Blough")
    assert not sherdog.confirms(rec, "2024-05-17", "Joseph Blough")


def test_a_card_lookup_covers_everyone_on_the_card_with_bouts_to_check(tmp_path):
    source = tmp_path / "predict_card.py"
    source.write_text('EVENT_DATE = "2026-10-17"\n'
                      'FIGHT_CARD = [("Ann Vet", "Bea New", True, False), ("Cat Old", "Ann Vet")]\n')
    archive = tmp_path / "a.csv"
    pd.DataFrame({"date": ["2026-01-01", "2025-01-01", "2026-10-17"],
                  "r_name": ["Ann Vet", "Cat Old", "Ann Vet"],
                  "b_name": ["Cat Old", "Ann Vet", "Late Bout"]}).to_csv(archive, index=False)
    event, names = hs.card_fighters(source)
    assert event == "2026-10-17" and names == ["Ann Vet", "Bea New", "Cat Old"]
    targets = hs.card_targets(source, archive)
    assert targets["Ann Vet"] == [("2026-01-01", "Cat Old"), ("2025-01-01", "Cat Old")]
    assert targets["Bea New"] == []        # a debutant: name-only, unverified


def test_a_card_fighter_is_also_searched_under_sherdogs_spelling(monkeypatch):
    tried = []

    def fake_find(name, agent, bouts=(), tries=3):
        tried.append(name)
        return {"id": 1, "record": []} if name == "Cristian Perez Gonzalez" else None

    monkeypatch.setattr(hs.sherdog, "find", fake_find)
    assert hs.find_any("Cristian Perez", "agent", [])["id"] == 1
    assert tried == ["Cristian Perez Gonzalez"]
    tried.clear()
    assert hs.find_any("Nobody Known", "agent", []) is None and tried == ["Nobody Known"]


def test_sherdogs_spelling_is_tried_before_the_cards(monkeypatch):
    """The card's spelling can be a different real fighter's name."""
    tried = []
    monkeypatch.setattr(hs.sherdog, "find",
                        lambda name, agent, bouts=(), tries=3: tried.append(name))
    hs.find_any("Cristian Perez", "agent", [])
    assert tried == ["Cristian Perez Gonzalez", "Cristian Perez"]


def test_a_debutant_is_verified_against_their_contender_series_bout(tmp_path, monkeypatch):
    source = tmp_path / "predict_card.py"
    source.write_text('EVENT_DATE = "2026-10-17"\nFIGHT_CARD = [("New Guy", "Old Hand")]\n')
    archive = tmp_path / "a.csv"
    pd.DataFrame({"date": ["2026-01-01"], "r_name": ["Old Hand"],
                  "b_name": ["Some One"]}).to_csv(archive, index=False)
    world = tmp_path / "w.csv.gz"
    pd.DataFrame({"date": ["2025-09-30", "2026-11-01"], "winner": ["New Guy", "New Guy"],
                  "loser": ["Magno Dias", "Later Foe"]}).to_csv(world, index=False)
    monkeypatch.setattr(hs, "BOUTS", world)
    assert hs.card_targets(source, archive)["New Guy"] == [("2025-09-30", "Magno Dias")]


def _run_card(tmp_path, monkeypatch, on_file, found):
    out = tmp_path / "ufc.jsonl.gz"
    hs.save({"Ann Vet": on_file}, out)
    monkeypatch.setattr(hs, "OUT_UFC", out)
    monkeypatch.setattr(hs, "card_targets", lambda: {"Ann Vet": []})
    monkeypatch.setattr(hs, "find_any", lambda name, agent, bouts: found)
    hs.main(["--card", "--minutes", "1"])
    return hs.load(out)["Ann Vet"]


def test_a_different_sherdog_fighter_never_replaces_the_one_on_file(tmp_path, monkeypatch):
    kept = _run_card(tmp_path, monkeypatch,
                     {"target": "Ann Vet", "id": 111, "verified": True, "record": []},
                     {"id": 222, "slug": "x", "name": "Ann Vet", "verified": True, "record": []})
    assert kept["id"] == 111


def test_a_name_only_record_that_no_longer_resolves_is_dropped(tmp_path, monkeypatch):
    gone = _run_card(tmp_path, monkeypatch,
                     {"target": "Ann Vet", "id": 111, "verified": False, "record": []}, None)
    assert gone["id"] is None


def test_a_verified_record_survives_a_failed_lookup(tmp_path, monkeypatch):
    kept = _run_card(tmp_path, monkeypatch,
                     {"target": "Ann Vet", "id": 111, "verified": True, "record": []}, None)
    assert kept["id"] == 111
