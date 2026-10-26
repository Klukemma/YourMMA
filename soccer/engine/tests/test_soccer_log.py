"""The soccer prediction log: logged only before kickoff, graded once."""

import pandas as pd

import soccer_log as sl


def _match(**kw):
    m = {"match_id": "E0-20261031-arsenal-chelsea", "league": "E0", "season": 2026,
         "kickoff": "2026-10-31T15:00:00Z", "home": "Arsenal", "away": "Chelsea",
         "pick": "H", "pick_prob": 0.52, "p_h": 0.52, "p_d": 0.26, "p_a": 0.22, "p_o25": 0.55}
    m.update(kw)
    return m


def test_a_pick_is_written_before_kickoff_and_frozen_after():
    log = {"schema": sl.SCHEMA, "picks": {}}
    assert sl.log_pick(log, _match(), pd.Timestamp("2026-10-30 18:00"))
    first = log["picks"]["E0-20261031-arsenal-chelsea"]["first_logged"]
    assert sl.log_pick(log, _match(pick_prob=0.55), pd.Timestamp("2026-10-31 09:00"))
    entry = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert entry["pick_prob"] == 0.55 and entry["first_logged"] == first
    # At kickoff and after, nothing changes.
    assert not sl.log_pick(log, _match(pick="A"), pd.Timestamp("2026-10-31 15:00"))
    assert log["picks"]["E0-20261031-arsenal-chelsea"]["pick"] == "H"


def test_grading_marks_the_result_and_the_over_under_once():
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-30 18:00"))
    sl.log_pick(log, _match(match_id="x", home="Leeds", away="Stoke", pick="A", p_o25=None), pd.Timestamp("2026-10-30 18:00"))
    results = pd.DataFrame([{"match_id": "E0-20261031-arsenal-chelsea", "hg": 2, "ag": 2, "result": "D"},
                            {"match_id": "y", "hg": 1, "ag": 0, "result": "H"}])
    assert sl.grade(log, results) == 1
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["final"] == {"home": 2, "away": 2, "result": "D"} and p["correct"] is False
    assert p["o25_correct"] is True                  # four goals, over 2.5 was called at 55%
    assert sl.grade(log, results) == 0               # never twice
    assert sl.record(log) == {"settled": 1, "correct": 0, "ou_settled": 1, "ou_correct": 1, "pending": 1,
                              "provisional": 0, "corrected": 0}


def test_the_log_round_trips_without_nan(tmp_path):
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(p_o25=float("nan")), pd.Timestamp("2026-10-30 18:00"))
    sl.save(log, tmp_path / "h.json")
    back = sl.load(tmp_path / "h.json")
    assert back["picks"]["E0-20261031-arsenal-chelsea"]["p_o25"] is None


def test_a_moved_match_is_graded_once_and_a_lost_one_is_voided():
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-30 18:00"))
    # Postponed to 2 Dec: the replay is played under a new id.
    replay = pd.DataFrame([{"match_id": "E0-20261202-arsenal-chelsea", "league": "E0", "season": 2026,
                            "date": "2026-12-02", "home": "Arsenal", "away": "Chelsea", "hg": 1, "ag": 0, "result": "H"}])
    assert sl.grade(log, replay) == 1
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["correct"] is True and p["final"]["played_on"] == "2026-12-02"

    # Picked again under its new id before the replay: the older pick is void.
    log2 = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log2, _match(), pd.Timestamp("2026-10-30 18:00"))
    sl.log_pick(log2, _match(match_id="E0-20261202-arsenal-chelsea", kickoff="2026-12-02T19:45:00Z"),
                pd.Timestamp("2026-12-01 18:00"))
    assert log2["picks"]["E0-20261031-arsenal-chelsea"]["void"] == "rescheduled"
    assert sl.grade(log2, replay) == 1 and sl.record(log2)["settled"] == 1 and sl.record(log2)["pending"] == 0

    # Never played: void three weeks after kickoff, not pending for ever.
    log3 = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log3, _match(), pd.Timestamp("2026-10-30 18:00"))
    none = replay.iloc[:0]
    sl.grade(log3, none, now=pd.Timestamp("2026-11-10"))
    assert sl.record(log3)["pending"] == 1
    sl.grade(log3, none, now=pd.Timestamp("2026-11-30"))
    assert log3["picks"]["E0-20261031-arsenal-chelsea"]["void"] == "no result" and sl.record(log3)["pending"] == 0
