"""The prediction log with every league in it: where a pairing repeats in a
season it is never one match, the record is told per league, and the file
stays small enough to commit on every run."""

import json

import pandas as pd

import soccer_feed as sfeed
import soccer_log as sl


def _pick(league="FIN", day="2026-11-08", home="HJK", away="AC Oulu", hour=15, **kw):
    m = {"match_id": f"{league}-{day.replace('-', '')}-{home.lower().replace(' ', '')}-{away.lower().replace(' ', '')}",
         "league": league, "season": 2026, "kickoff": f"{day}T{hour:02d}:00:00Z", "home": home, "away": away,
         "pick": "H", "pick_prob": 0.5, "p_h": 0.5, "p_d": 0.27, "p_a": 0.23, "p_o25": 0.55}
    m.update(kw)
    return m


def _result(league, day, home, away, hg, ag, mid=None, hour=15):
    return {"match_id": mid or f"{league}-{day.replace('-', '')}-x", "league": league, "season": 2026, "date": day,
            "kickoff": pd.Timestamp(f"{day} {hour:02d}:00"), "home": home, "away": away, "hg": hg, "ag": ag,
            "result": "H" if hg > ag else "A" if hg < ag else "D"}


def test_a_repeated_pairing_is_never_graded_on_its_earlier_meeting():
    """FIN plays a pairing three times a season. The pick for HJK v AC Oulu
    on 2026-11-08 was voided "logged after kickoff" on April's meeting the
    run it was logged (the map's real case): now April's result is another
    match, and the pick waits for its own."""
    assert "FIN" in sfeed.REPEATS
    log = {"schema": sl.SCHEMA, "picks": {}}
    p = _pick()
    assert sl.log_pick(log, p, pd.Timestamp("2026-11-06 18:00"))
    april = pd.DataFrame([_result("FIN", "2026-04-11", "HJK", "AC Oulu", 2, 0)])
    stats = {}
    assert sl.grade(log, april, now=pd.Timestamp("2026-11-06 18:30"), stats=stats) == 0
    entry = log["picks"][p["match_id"]]
    assert not entry.get("void") and "final" not in entry and stats["late"] == 0
    # Its own result, filed a day later than the pick's id (the sources dated
    # it apart): found by its pairing, within PAIR_DAYS - and graded.
    both = pd.DataFrame([_result("FIN", "2026-04-11", "HJK", "AC Oulu", 2, 0),
                         _result("FIN", "2026-11-09", "HJK", "AC Oulu", 0, 1)])
    assert sl.grade(log, both, now=pd.Timestamp("2026-11-10")) == 1
    assert entry["final"]["result"] == "A" and entry["final"]["played_on"] == "2026-11-09"
    # A league whose pairing is played once a season grades a moved match
    # however far it moved, as before.
    e0 = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(e0, _pick("E0", "2026-10-31", "Arsenal", "Chelsea"), pd.Timestamp("2026-10-30 18:00"))
    assert sl.grade(e0, pd.DataFrame([_result("E0", "2026-12-02", "Arsenal", "Chelsea", 1, 0)])) == 1


def test_a_meeting_further_than_pair_days_is_another_match_never_this_ones_result():
    log = {"schema": sl.SCHEMA, "picks": {}}
    p = _pick("SC1", "2026-11-08", "Dunfermline", "Partick")
    sl.log_pick(log, p, pd.Timestamp("2026-11-06 18:00"))
    later = pd.DataFrame([_result("SC1", "2026-11-12", "Dunfermline", "Partick", 3, 0)])
    sl.grade(log, later, now=pd.Timestamp("2026-11-13"))
    assert "final" not in log["picks"][p["match_id"]]
    # nor a provisional one
    prov = later.assign(provisional=True, source="ESPN")
    sl.grade(log, None, now=pd.Timestamp("2026-11-13"), provisional=prov)
    assert "final" not in log["picks"][p["match_id"]]
    near = later.assign(date="2026-11-10", kickoff=pd.Timestamp("2026-11-10 15:00"))
    assert sl.grade(log, near, now=pd.Timestamp("2026-11-13")) == 1


def test_two_meetings_of_a_pairing_both_to_play_keep_two_picks():
    """MLS's best-of-three: games 1 and 3 are the same home and away clubs,
    a week apart, both in the ten days the app lists. Logging game 3 must
    neither move game 1's pick onto it nor void it - and game 1 kicked off
    with its result not in yet is not "off"."""
    now = pd.Timestamp("2026-10-30 18:00")
    g1, g3 = _pick("USA", "2026-11-01", "Inter Miami", "Nashville", 23), \
        _pick("USA", "2026-11-08", "Inter Miami", "Nashville", 23, pick="A")
    log = {"schema": sl.SCHEMA, "picks": {}}
    assert sl.log_pick(log, g1, now) and sl.log_pick(log, g3, now)
    assert set(log["picks"]) == {g1["match_id"], g3["match_id"]} and not any(p.get("void") for p in log["picks"].values())
    later = pd.Timestamp("2026-11-02 09:00")                 # game 1 played, its result not in
    assert sl.log_pick(log, dict(g3, pick_prob=0.55), later)
    assert not log["picks"][g1["match_id"]].get("void") and log["picks"][g1["match_id"]]["pick"] == "H"
    # Within PAIR_DAYS it is the same match, dated apart: it moves, as ever.
    moved = _pick("USA", "2026-11-02", "Inter Miami", "Nashville", 1)
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, g1, now)
    sl.log_pick(log, moved, now)
    assert list(log["picks"]) == [moved["match_id"]] and log["picks"][moved["match_id"]]["id_was"] == g1["match_id"]


def test_a_pick_still_listed_to_play_is_never_moved_or_voided():
    now = pd.Timestamp("2026-10-30 18:00")
    a = _pick("E0", "2026-10-31", "Arsenal", "Chelsea")
    b = _pick("E0", "2026-11-01", "Arsenal", "Chelsea")
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, a, now)
    assert sl.log_pick(log, b, now, listed={a["match_id"], b["match_id"]})
    assert set(log["picks"]) == {a["match_id"], b["match_id"]} and not any(p.get("void") for p in log["picks"].values())
    # Not listed: one match, one pick (it moves), as before.
    log = {"schema": sl.SCHEMA, "picks": {}}
    sl.log_pick(log, a, now)
    sl.log_pick(log, b, now, listed={b["match_id"]})
    assert list(log["picks"]) == [b["match_id"]]


def test_the_feeds_status_of_another_meeting_never_withdraws_a_grade():
    log = {"schema": sl.SCHEMA, "picks": {}}
    p = _pick()
    sl.log_pick(log, p, pd.Timestamp("2026-11-06 18:00"))
    prov = pd.DataFrame([dict(_result("FIN", "2026-11-08", "HJK", "AC Oulu", 1, 1, mid=p["match_id"]),
                              provisional=True, source="ESPN")])
    sl.grade(log, None, now=pd.Timestamp("2026-11-08 20:00"), provisional=prov)
    assert log["picks"][p["match_id"]]["final"]["provisional"]
    april = ("FIN", 2026, "HJK", "AC Oulu", "2026-04-11")
    sl.grade(log, None, now=pd.Timestamp("2026-11-09"), provisional=prov.iloc[:0], disputed={april},
             feed_status={april: "postponed"})
    assert log["picks"][p["match_id"]]["final"]["provisional"]          # April's is another match
    mine = ("FIN", 2026, "HJK", "AC Oulu", "2026-11-08")
    sl.grade(log, None, now=pd.Timestamp("2026-11-09"), provisional=prov.iloc[:0], feed_status={mine: "postponed"})
    assert "final" not in log["picks"][p["match_id"]]


def test_the_record_is_told_per_league_group_and_basis():
    log = {"schema": sl.SCHEMA, "picks": {}}
    now = pd.Timestamp("2026-10-30 18:00")
    for lg, day, home, away, basis in (("E0", "2026-10-31", "Arsenal", "Chelsea", "recalibrated"),
                                       ("D1", "2026-10-31", "Bayern Munich", "Dortmund", "raw"),
                                       ("NOR", "2026-11-01", "Bodo/Glimt", "Molde", "model")):
        sl.log_pick(log, _pick(lg, day, home, away, basis=basis, group={"E0": "top", "D1": "top"}.get(lg, "extra")),
                    now)
    results = pd.DataFrame([_result("E0", "2026-10-31", "Arsenal", "Chelsea", 2, 0, mid="E0-20261031-arsenal-chelsea")])
    sl.grade(log, results)
    by = sl.record(log, by=lambda p: p["league"])
    assert set(by) == {"E0", "D1", "NOR"} and by["E0"]["settled"] == 1 and by["E0"]["correct"] == 1
    assert by["D1"] == dict(sl.record({"picks": {}}), pending=1)
    assert sl.record(log, by=lambda p: p["group"]) == {"extra": by["NOR"], "top": sl.record(
        {"picks": {k: v for k, v in log["picks"].items() if v["league"] != "NOR"}})}
    assert sl.record(log)["pending"] == 2 and sl.record(log)["settled"] == 1
    assert "basis" in sl.FIELDS and "price_kind" in sl.FIELDS and "group" in sl.FIELDS


def test_the_log_is_written_one_pick_a_line_and_settled_picks_move_to_monthly_files(tmp_path):
    path = tmp_path / "prediction_history.json"
    log = {"schema": sl.SCHEMA, "picks": {}}
    for week in range(10):                                   # Sep 5 .. Nov 7, four matches a week
        day = (pd.Timestamp("2026-09-05") + pd.Timedelta(weeks=week)).strftime("%Y-%m-%d")
        clubs = [f"Club {week} {k}" for k in range(8)]
        for h, a in zip(clubs[::2], clubs[1::2]):
            sl.log_pick(log, _pick("E0", day, h, a), pd.Timestamp(day) - pd.Timedelta(days=1))
    done = [p for p in log["picks"].values() if p["kickoff"] < "2026-10-01"]
    results = pd.DataFrame([_result("E0", p["kickoff"][:10], p["home"], p["away"], 1, 0, mid=p["match_id"])
                            for p in done])
    sl.grade(log, results)
    log["picks"][done[0]["match_id"]]["final"]["provisional"] = True   # can still change: stays in the main file
    sl.save(log, path)
    main = path.read_text()
    assert main.count("\n") == len(json.loads(main)["picks"]) + 3 and ": " not in main
    archived = sorted(f.name for f in sl.archive_dir(path).glob("*.json"))
    assert archived == ["2026-09.json"]
    month = json.loads((sl.archive_dir(path) / "2026-09.json").read_text())
    assert month["schema"] == sl.SCHEMA and len(month["picks"]) == len(done) - 1
    assert all(p["final"] and not p["final"].get("provisional") for p in month["picks"].values())
    assert done[0]["match_id"] in json.loads(main)["picks"]
    back = sl.load(path)
    assert back == json.loads(json.dumps(log)) and len(back["picks"]) == 40
    # The same log is always split the same way; nothing is rewritten.
    stamp = {f: f.stat().st_mtime_ns for f in [path, *sl.archive_dir(path).glob("*.json")]}
    sl.save(back, path)
    assert {f: f.stat().st_mtime_ns for f in stamp} == stamp
    # A log of before - one indented file - reads as it did.
    old = tmp_path / "old.json"
    old.write_text(json.dumps(log, indent=1, sort_keys=True))
    assert sl.load(old) == back
