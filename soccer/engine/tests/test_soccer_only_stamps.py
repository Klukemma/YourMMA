"""only_stamps: the runner's files whose only change is when the run happened
are put back before the commit step looks - in a throwaway git repository
laid out like this one."""

import json
import subprocess

import pytest

import only_stamps as os_

PAYLOADS = os_.PAYLOADS
PAGE = os_.PAGE


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True, text=True).stdout


def _write(root, path, data):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n" if not isinstance(data, str) else data)


def _files(at):
    """The runner's stamped files as one run at `at` writes them."""
    pick = {"match_id": "E0-20261107-arsenal-chelsea", "pick": "H", "pick_prob": 0.51, "kickoff": "2026-11-07T15:00:00Z",
            "prices_at": "2026-11-06T16:00:00Z", "last_logged": at}
    return {
        "soccer/app/data/matches.json": {"generated": at, "feeds_at": at, "matches": [dict(pick, logged_at=at)],
                                         "feeds": {"at": at, "data_problems": [], "data_synced_at": at}},
        "soccer/app/data/tables.json": {"generated": at, "leagues": [{"code": "E0", "rows": []}]},
        "soccer/app/data/performance.json": {"generated": at, "backtest": {"value": {"enabled_at": None}}},
        "soccer/data/feed.json": {"last_run": {"at": at, "answered": 2},
                                  "requests": {"espn:E0:20261107": at, "espn:E0:20261108": at},
                                  "rows": [{"match_id": "x", "seen_at": "2026-11-06T10:00:00Z", "status": "scheduled"}]},
        "soccer/data/prediction_history.json": {"schema": 1, "picks": {pick["match_id"]: pick}},
        PAGE: f"<html><!-- built {at} -->{at}</html>\n",
    }


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    for path, data in _files("2026-11-10T20:38:58Z").items():
        _write(tmp_path, path, data)
    _write(tmp_path, "soccer/data/matches/E0/2026.csv", "a,b\n1,2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "a run")
    return tmp_path


def _status(root):
    return sorted(line[3:] for line in _git(root, "status", "--porcelain", "--untracked-files=all").splitlines())


def test_a_run_where_only_the_stamps_changed_commits_nothing(repo):
    for path, data in _files("2026-11-10T22:01:16Z").items():
        _write(repo, path, data)
    assert len(_status(repo)) == 6
    back = os_.settle(repo, log=lambda *_: None)
    assert set(back) == set(PAYLOADS) | {PAGE, "soccer/data/feed.json", "soccer/data/prediction_history.json"}
    assert _status(repo) == []                               # the commit step says "Nothing changed."


def test_a_real_change_is_committed_and_the_page_is_rebuilt_only_when_its_data_changed(repo):
    later = _files("2026-11-10T22:01:16Z")
    for path, data in later.items():
        _write(repo, path, data)
    _write(repo, "soccer/data/matches/E0/2026.csv", "a,b\n1,2\n3,4\n")     # a result came in
    os_.settle(repo, log=lambda *_: None)
    # the payloads say the same thing: they and the page stay as committed;
    # the snapshot's change is committed, with the feed's and the log's new stamps
    assert _status(repo) == ["soccer/data/feed.json", "soccer/data/matches/E0/2026.csv",
                             "soccer/data/prediction_history.json"]
    # the pick moved: everything goes in, stamps and the rebuilt page with it
    later["soccer/app/data/matches.json"]["matches"][0]["pick_prob"] = 0.53
    for path, data in later.items():
        _write(repo, path, data)
    os_.settle(repo, log=lambda *_: None)
    assert set(PAYLOADS) | {PAGE} <= set(_status(repo))


def test_what_is_never_taken_for_a_stamp(repo):
    files = _files("2026-11-10T22:01:16Z")
    # a feed row seen anew changes with what it says; a new request is a day asked; a new file is new
    files["soccer/data/feed.json"]["rows"][0]["seen_at"] = "2026-11-10T22:01:16Z"
    for path, data in files.items():
        _write(repo, path, data)
    _write(repo, "soccer/data/prediction_history/2026-10.json", {"picks": {}})
    os_.settle(repo, log=lambda *_: None)
    assert "soccer/data/feed.json" in _status(repo)
    assert "soccer/data/prediction_history/2026-10.json" in _status(repo)
    assert os_.unstamped({"requests": {"a": 1}}) != os_.unstamped({"requests": {"a": 1, "b": 2}})
    assert os_.unstamped({"prices_at": 1, "kickoff": 2, "freeze_at": 3, "graded_at": 4}) == \
        {"prices_at": 1, "kickoff": 2, "freeze_at": 3, "graded_at": 4}


def test_a_file_the_commit_has_never_seen_or_that_is_not_json_is_real(repo):
    _write(repo, "soccer/app/data/matches.json", "not json")
    assert not os_.only_stamps(repo, "soccer/app/data/matches.json")
    assert not os_.only_stamps(repo, "soccer/app/data/new.json")
