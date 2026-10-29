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
    assert sl.record(log) == {"settled": 1, "correct": 0, "ou_settled": 1, "ou_correct": 1,
                              "btts_settled": 0, "btts_correct": 0, "pending": 1, "provisional": 0, "corrected": 0}


def test_both_teams_to_score_is_logged_and_graded_beside_the_over_under():
    log = {"schema": sl.SCHEMA, "picks": {}}
    scores = {"top": [{"score": "1-1", "p": 0.12}, {"score": "1-0", "p": 0.11}, {"score": "2-1", "p": 0.09}],
              "other": 0.01, "matrix": {"p_h": 0.5, "p_d": 0.25, "p_a": 0.25}, "o25": 0.55, "reason": None,
              "miss": 0.004, "miss_on": "draw"}
    now = pd.Timestamp("2026-10-30 18:00")
    sl.log_pick(log, _match(p_btts=0.56, p_o15=0.78, p_o35=0.31, scores=scores, score_source="market",
                            band="SOLID", band_basis="pick"), now)
    sl.log_pick(log, _match(match_id="b", home="Leeds", away="Stoke", p_btts=0.44), now)
    sl.log_pick(log, _match(match_id="c", home="Fulham", away="Burnley", p_btts=0.5), now)
    sl.log_pick(log, _match(match_id="d", home="Watford", away="Everton"), now)        # logged before the numbers
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert (p["p_btts"], p["p_o15"], p["p_o35"], p["band"], p["band_basis"], p["score_source"]) == \
        (0.56, 0.78, 0.31, "SOLID", "pick", "market")
    assert p["scores"]["top"][0] == {"score": "1-1", "p": 0.12}
    results = pd.DataFrame([{"match_id": k, "hg": hg, "ag": ag, "result": "H" if hg > ag else "A" if hg < ag else "D"}
                            for k, hg, ag in (("E0-20261031-arsenal-chelsea", 2, 1), ("b", 3, 0), ("c", 0, 0),
                                              ("d", 1, 1))])
    assert sl.grade(log, results) == 4
    # yes at 56% and both scored: right; no at 44% and one did not: right; a
    # 50% chance calls yes (as over 2.5 does) and 0-0 makes it wrong; no
    # chance logged, no grade.
    assert [log["picks"][k].get("btts_correct") for k in ("E0-20261031-arsenal-chelsea", "b", "c", "d")] == \
        [True, True, False, None]
    rec = sl.record(log)
    assert (rec["btts_settled"], rec["btts_correct"], rec["ou_settled"]) == (3, 2, 4)


def test_a_withdrawn_grade_takes_both_teams_to_score_with_it():
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(p_btts=0.6), pd.Timestamp("2026-10-30 18:00"))
    prov = pd.DataFrame([{"match_id": "E0-20261031-arsenal-chelsea", "league": "E0", "season": 2026,
                          "home": "Arsenal", "away": "Chelsea", "hg": 1, "ag": 1, "result": "D", "source": "espn"}])
    sl.grade(log, None, pd.Timestamp("2026-10-31 18:00"), provisional=prov)
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["btts_correct"] is True and p["final"]["provisional"]
    sl.grade(log, None, pd.Timestamp("2026-11-01 08:00"), provisional=prov.iloc[:0],
             disputed={("E0", 2026, "Arsenal", "Chelsea")})
    assert "btts_correct" not in p and "final" not in p and sl.record(log)["btts_settled"] == 0


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


def test_a_pick_frozen_at_midnight_stays_frozen_when_its_time_is_confirmed_later():
    log = {"schema": sl.SCHEMA, "picks": {}}
    unconfirmed = _match(freeze_at="2026-10-31T00:00:00Z", time_confirmed=False)
    assert sl.log_pick(log, unconfirmed, pd.Timestamp("2026-10-30 18:00"))
    confirmed = _match(pick="A", freeze_at="2026-10-31T15:00:00Z", time_confirmed=True)
    assert not sl.log_pick(log, confirmed, pd.Timestamp("2026-10-31 09:00"))
    assert log["picks"]["E0-20261031-arsenal-chelsea"]["pick"] == "H"


def test_a_pick_logged_after_the_real_kickoff_is_never_graded():
    """Brought forward to Friday 17:00, which no source had said by the
    Friday 18:00 run: whatever the result, that pick is void."""
    def friday(hg, ag, result):
        return pd.DataFrame([{"match_id": "E0-20261030-arsenal-chelsea", "league": "E0", "season": 2026,
                              "date": "2026-10-30", "kickoff": pd.Timestamp("2026-10-30 17:00"), "home": "Arsenal",
                              "away": "Chelsea", "hg": hg, "ag": ag, "result": result}])
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-30 18:00"))
    stats = {}
    assert sl.grade(log, friday(1, 0, "H"), now=pd.Timestamp("2026-11-02"), stats=stats) == 0
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["void"] == "logged after kickoff" and "final" not in p and stats["late"] == 1
    assert sl.record(log)["settled"] == 0 and sl.record(log)["pending"] == 0

    # Graded provisionally on the feeds' (wrong) time first, then football-data.co.uk's shows it was late.
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-30 18:00"))
    feeds = friday(1, 0, "H").assign(match_id="E0-20261031-arsenal-chelsea", date="2026-10-31",
                                     kickoff=pd.Timestamp("2026-10-31 15:00"), provisional=True, source="ESPN")
    assert sl.grade(log, friday(1, 0, "H").iloc[:0], provisional=feeds, now=pd.Timestamp("2026-10-31 21:47")) == 1
    sl.grade(log, friday(1, 0, "H"), now=pd.Timestamp("2026-11-02"))
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["void"] == "logged after kickoff" and "final" not in p
    assert p["corrections"][-1]["was"]["provisional"] is True and p["corrections"][-1]["now"] is None

    # Logged before it: graded as ever.
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-30 16:00"))
    assert sl.grade(log, friday(1, 0, "H")) == 1 and log["picks"]["E0-20261031-arsenal-chelsea"]["correct"] is True


def _feed_result(kickoff, **kw):
    row = {"match_id": "E0-20261031-arsenal-chelsea", "league": "E0", "season": 2026, "date": "2026-10-31",
           "kickoff": pd.Timestamp(kickoff), "home": "Arsenal", "away": "Chelsea", "hg": 0, "ag": 2, "result": "A",
           "provisional": True, "source": "ESPN"}
    row.update(kw)
    return pd.DataFrame([row])


def test_a_void_on_the_feeds_kickoff_is_restored_when_football_data_shows_it_was_in_time():
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-31 07:17"))
    stats = {}
    sl.grade(log, _feed_result("2026-10-31 15:00").iloc[:0], provisional=_feed_result("2026-10-31 07:00"),
             now=pd.Timestamp("2026-10-31 21:47"), stats=stats)
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert (p["void"], p["void_by"], stats["late"]) == ("logged after kickoff", "ESPN", 1)
    fd = _feed_result("2026-10-31 15:00").drop(columns=["provisional", "source"])
    stats = {}
    assert sl.grade(log, fd, now=pd.Timestamp("2026-11-02 09:17"), stats=stats) == 1
    assert not p.get("void") and p["final"]["result"] == "A" and stats["restored"] == 1
    assert p["corrections"][-1]["now"] == "restored" and p["corrections"][-1]["by"] == "football-data.co.uk"
    assert sl.record(log)["settled"] == 1
    # football-data.co.uk's own kickoff saying late is final: never restored.
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(), pd.Timestamp("2026-10-31 07:17"))
    sl.grade(log, fd.assign(kickoff=pd.Timestamp("2026-10-31 07:00")), now=pd.Timestamp("2026-11-02 09:17"))
    sl.grade(log, fd, now=pd.Timestamp("2026-11-03 09:17"))
    assert log["picks"]["E0-20261031-arsenal-chelsea"]["void_by"] == "football-data.co.uk"


def test_a_frozen_pick_follows_its_match_to_a_new_id_unchanged():
    now = pd.Timestamp("2026-10-30 18:00")
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, _match(freeze_at="2026-10-31T00:00:00Z", time_confirmed=False), now)
    first = dict(log["picks"]["E0-20261031-arsenal-chelsea"])
    # Saturday 07:17, after it froze and before its kickoff: posted for Sunday.
    sunday = _match(match_id="E0-20261101-arsenal-chelsea", kickoff="2026-11-01T14:00:00Z", pick="A",
                    p_h=0.2, freeze_at="2026-11-01T14:00:00Z", time_confirmed=True)
    assert not sl.log_pick(log, sunday, pd.Timestamp("2026-10-31 07:17"))
    assert list(log["picks"]) == ["E0-20261101-arsenal-chelsea"]
    p = log["picks"]["E0-20261101-arsenal-chelsea"]
    assert (p["pick"], p["p_h"], p["freeze_at"], p["last_logged"]) == ("H", 0.52, first["freeze_at"], first["last_logged"])
    assert (p["kickoff"], p["kickoff_was"], p["id_was"]) == ("2026-11-01T14:00:00Z", "2026-10-31T15:00:00Z",
                                                            "E0-20261031-arsenal-chelsea")
    # Past its own kickoff with no result, or listed as postponed: it is off,
    # and a pick on the new date replaces it.
    for when, off in (("2026-10-31 16:00", ()), ("2026-10-31 09:00", ("E0-20261031-arsenal-chelsea",))):
        log = {"schema": sl.SCHEMA, "picks": {}}
        sl.log_pick(log, _match(), now)
        assert sl.log_pick(log, dict(sunday, freeze_at=None), pd.Timestamp(when), off=off)
        assert log["picks"]["E0-20261031-arsenal-chelsea"]["void"] == "rescheduled"
        assert log["picks"]["E0-20261101-arsenal-chelsea"]["pick"] == "A"


def test_a_priced_pick_is_never_overwritten_by_an_unpriced_one():
    log = {"schema": sl.SCHEMA, "picks": {}}
    priced = _match(market={"p_h": 0.5, "p_d": 0.27, "p_a": 0.23}, price_source="football-data.co.uk",
                    freeze_at="2026-10-31T15:00:00Z", time_confirmed=True)
    sl.log_pick(log, priced, pd.Timestamp("2026-10-30 18:00"))
    before = dict(log["picks"]["E0-20261031-arsenal-chelsea"])
    bare = _match(pick="A", market=None, price_source=None, freeze_at="2026-10-31T00:00:00Z", time_confirmed=False)
    assert not sl.log_pick(log, bare, pd.Timestamp("2026-10-30 20:00"))
    p = log["picks"]["E0-20261031-arsenal-chelsea"]
    assert p["pick"] == "H" and p["market"] == before["market"] and p["last_logged"] == before["last_logged"]
    assert (p["freeze_at"], p["time_confirmed"]) == ("2026-10-31T00:00:00Z", False)       # its time follows the match
    # A new price replaces it as ever.
    assert sl.log_pick(log, dict(priced, pick_prob=0.55), pd.Timestamp("2026-10-30 22:00"))
    assert log["picks"]["E0-20261031-arsenal-chelsea"]["pick_prob"] == 0.55


def test_one_match_one_pick_even_from_an_old_log():
    log = {"schema": sl.SCHEMA, "picks": {}}
    now = pd.Timestamp("2026-10-29 09:00")
    sl.log_pick(log, _match(), now)
    log["picks"]["E0-20261101-arsenal-chelsea"] = dict(log["picks"]["E0-20261031-arsenal-chelsea"],
                                                         match_id="E0-20261101-arsenal-chelsea")
    assert sl.log_pick(log, _match(), pd.Timestamp("2026-10-29 10:00"))
    assert log["picks"]["E0-20261101-arsenal-chelsea"]["void"] == "duplicate"
    assert sl.record(log)["pending"] == 1
    # A void entry under the id a pick moves to is kept inside it.
    log["picks"]["E0-20261101-arsenal-chelsea"]["void"] = "rescheduled"
    assert sl.log_pick(log, _match(match_id="E0-20261101-arsenal-chelsea", kickoff="2026-11-01T14:00:00Z"),
                       pd.Timestamp("2026-10-29 11:00"))
    p = log["picks"]["E0-20261101-arsenal-chelsea"]
    assert p["void_was"]["void"] == "rescheduled" and p["id_was"] == "E0-20261031-arsenal-chelsea" and not p.get("void")


def test_a_frozen_pick_is_not_voided_while_another_sources_time_is_still_to_come():
    """The pick's id came from ESPN's date, which ESPN later corrects; no run
    saw the correction before ESPN's wrong time passed. fixturedownload's time
    is still to come, so the match is not known to be off: the pick moves to
    the new id as it is, nothing is logged after it froze, nothing is void."""
    log = {"schema": sl.SCHEMA, "picks": {}}
    first = _match(kickoff="2026-10-31T15:00:00Z",
                   kickoff_sources={"ESPN": "2026-10-31T15:00:00Z", "fixturedownload": "2026-11-01T16:15:00Z"})
    sl.log_pick(log, first, pd.Timestamp("2026-10-30 18:00"))
    moved = _match(match_id="E0-20261101-arsenal-chelsea", kickoff="2026-11-01T16:15:00Z")
    sl.log_pick(log, moved, pd.Timestamp("2026-10-31 16:00"))
    assert not any(p.get("void") for p in log["picks"].values())
    assert list(log["picks"]) == ["E0-20261101-arsenal-chelsea"]
    p = log["picks"]["E0-20261101-arsenal-chelsea"]
    assert p["id_was"] == "E0-20261031-arsenal-chelsea" and p["last_logged"] == "2026-10-30T18:00:00Z"
