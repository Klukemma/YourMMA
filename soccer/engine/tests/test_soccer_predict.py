"""The soccer run end to end on a small made-up league: what it logs, what it
grades, and the files the app reads.

The model is stood in for by a fixed rule, so these check the plumbing -
point in time, the log, the payloads - not the model, which the walk-forward
research in experiments/ measures."""

import json

import numpy as np
import pandas as pd
import pytest

import soccer_backtest as bt
import soccer_data as sd
import soccer_model as smod
import soccer_predict as sp

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford", "Burnley", "Watford", "Leeds", "Stoke"]


def _league(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2005, 2027):
        day = pd.Timestamp(f"{season}-08-10")
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=7 * (k // 5))
            if season == 2026 and date > pd.Timestamp("2026-10-20"):
                break
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            rows.append({"match_id": sd.match_id("E0", date.date(), h, a), "league": "E0", "season": season,
                         "date": date.strftime("%Y-%m-%d"), "time": "15:00",
                         "kickoff": date + pd.Timedelta(hours=14), "home": h, "away": a,
                         "hg": hg, "ag": ag, "result": "H" if hg > ag else "A" if hg < ag else "D",
                         "avg_h": 2.2, "avg_d": 3.3, "avg_a": 3.4, "max_h": 2.3, "max_d": 3.5, "max_a": 3.6,
                         "avg_o25": 1.9, "avg_u25": 1.9})
    return sd.frame(rows)


def _fixtures():
    rows = [{"match_id": sd.match_id("E0", pd.Timestamp("2026-10-31").date(), "Arsenal", "Chelsea"), "league": "E0",
             "season": 2026, "date": "2026-10-31", "time": "15:00", "kickoff": pd.Timestamp("2026-10-31 15:00"),
             "home": "Arsenal", "away": "Chelsea", "avg_h": 1.8, "avg_d": 3.6, "avg_a": 4.5,
             "max_h": 1.9, "max_d": 3.8, "max_a": 4.8, "avg_o25": 1.8, "avg_u25": 2.0},
            {"match_id": sd.match_id("E0", pd.Timestamp("2026-11-01").date(), "Leeds", "Stoke"), "league": "E0",
             "season": 2026, "date": "2026-11-01", "time": "16:30", "kickoff": pd.Timestamp("2026-11-01 16:30"),
             "home": "Leeds", "away": "Stoke"}]                      # no price posted
    fx = sd.frame(rows)
    fx["fetched_at"] = pd.Timestamp("2026-10-30 16:00")
    return fx


def _stub(frame, model=None):
    """A fixed rule standing in for the model: a little more home than the
    market, and the league's base rate where there is no price."""
    out = pd.DataFrame({"p_h": 0.47, "p_d": 0.27, "p_a": 0.26, "p_o25": 0.52}, index=frame.index)
    top = frame["league"].isin(sd.TOP) & (frame["season"] > 2005)
    out.loc[~top] = np.nan
    return out


@pytest.fixture
def run(tmp_path, monkeypatch):
    matches, fixtures = _league(), _fixtures()
    monkeypatch.setattr(sd, "load_matches", lambda *a, **k: matches.copy())
    monkeypatch.setattr(sd, "load_fixtures", lambda *a, **k: fixtures.copy())
    monkeypatch.setattr(smod, "predict", _stub)

    def go(now, results=None):
        if results is not None:
            monkeypatch.setattr(sd, "load_matches", lambda *a, **k: results.copy())
        sp.run(sync=False, now=pd.Timestamp(now), out_dir=tmp_path, log_path=tmp_path / "history.json",
               log=lambda *_: None)
        read = lambda n: json.loads((tmp_path / f"{n}.json").read_text())
        return read("matches"), read("tables"), read("performance"), json.loads((tmp_path / "history.json").read_text())
    return go


def test_the_week_is_picked_logged_and_written_for_the_app(run):
    m, tables, perf, log = run("2026-10-30 18:00")
    ups = {e["match_id"]: e for e in m["matches"]}
    assert len(ups) == 2 and set(log["picks"]) == set(ups)
    priced = next(e for e in ups.values() if e["home"]["name"] == "Arsenal")
    assert abs(priced["p_h"] + priced["p_d"] + priced["p_a"] - 1) < 1e-9
    assert priced["pick"] in ("H", "D", "A") and priced["market"] is not None
    assert priced["prices"]["avg"]["h"] == 1.8 and priced["prices_at"] == "2026-10-30T16:00:00Z"
    assert priced["home"]["code"] == "ARS" and priced["home_form"] and priced["home_position"]
    # No price: the model's own chances, said so - never a market of zeros.
    bare = next(e for e in ups.values() if e["home"]["name"].startswith("Leeds"))
    assert bare["market"] is None and bare["p_h"] == pytest.approx(0.47)
    assert m["league_order"] == list(sd.TOP) and "E0" in m["leagues"]
    assert tables["leagues"][0]["rows"][0]["club"]["code"]
    assert perf["backtest"]["periods"]["confirm"] == list(bt.CONFIRM)
    assert perf["track_record"]["pending"] == 2


def test_a_pick_is_frozen_at_kickoff_and_graded_when_the_result_lands(run):
    _, _, _, before = run("2026-10-30 18:00")
    first = before["picks"]
    # Saturday afternoon, after the first kickoff: that pick may not change.
    _, _, _, during = run("2026-10-31 15:30")
    arsenal = next(k for k in first if "arsenal" in k)
    assert during["picks"][arsenal]["last_logged"] == first[arsenal]["last_logged"]
    # The result lands in the next sync.
    played = _league()
    extra = _fixtures().iloc[[0]].drop(columns=["fetched_at"]).assign(hg=0, ag=2, result="A")
    m, _, perf, after = run("2026-11-02 09:00", results=pd.concat([played, sd.frame(extra.to_dict("records"))]))
    p = after["picks"][arsenal]
    assert p["final"] == {"home": 0, "away": 2, "result": "A"} and p["correct"] == (p["pick"] == "A")
    shown = next(e for e in m["matches"] if e["match_id"] == arsenal)
    assert shown["status"] == "final" and shown["score"] == {"home": 0, "away": 2}
    assert perf["track_record"]["settled"] == 1


def test_no_payload_carries_a_nan(run):
    m, tables, perf, _ = run("2026-10-30 18:00")
    for payload in (m, tables, perf):
        text = json.dumps(payload, allow_nan=False)      # raises on NaN
        assert "NaN" not in text and payload["schema"] == 1 and payload["generated"]


def test_the_standalone_page_carries_the_data(run, tmp_path):
    import soccer_standalone
    run("2026-10-30 18:00")
    out = soccer_standalone.build(tmp_path / "x.html", tmp_path, log=lambda *_: None)
    html = out.read_text()
    assert "window.__YOURSOCCER_DATA__" in html and '<script type="module" src="app.js">' not in html
    assert 'rel="manifest"' not in html
