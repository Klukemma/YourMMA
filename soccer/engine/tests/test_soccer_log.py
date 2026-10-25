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
    sl.log_pick(log, _match(match_id="x", pick="A", p_o25=None), pd.Timestamp("2026-10-30 18:00"))
    results = pd.DataFrame([{"match_id": "E0-20261031-arsenal-chelsea", "hg": 2, "ag": 2, "result": "D"},
                            {"match_id": "y", "hg": 1, "ag": 0, "result": "H"}])
    assert sl.grade(log, results) == 1
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["final"] == {"home": 2, "away": 2, "result": "D"} and p["correct"] is False
    assert p["o25_correct"] is True                  # four goals, over 2.5 was called at 55%
    assert sl.grade(log, results) == 0               # never twice
    assert sl.record(log) == {"settled": 1, "correct": 0, "ou_settled": 1, "ou_correct": 1, "pending": 1}


def test_the_log_round_trips_without_nan(tmp_path):
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(p_o25=float("nan")), pd.Timestamp("2026-10-30 18:00"))
    sl.save(log, tmp_path / "h.json")
    back = sl.load(tmp_path / "h.json")
    assert back["picks"]["E0-20261031-arsenal-chelsea"]["p_o25"] is None
