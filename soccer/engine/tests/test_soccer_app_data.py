"""The committed app data, as the runner last wrote it: every league in the
app is there, each league's Record row is the record that decided it ships
(soccer_leagues.SHIPPED's measured numbers, to the digit), and a number a
league did not ship is null wherever it would show.

Snapshot checks (reported by the runner, never a gate): they read what the
last run wrote. A payload written before every league was switched on
(no "groups" in matches.json) is skipped, said so."""

import json

import pytest

import soccer_data as sd
import soccer_export as sx
import soccer_leagues as slg

pytestmark = pytest.mark.snapshot

RECORD = {"log_loss": "pick", "market_log_loss": "market", "model_log_loss": "model", "base_log_loss": "base",
          "accuracy": "acc_pick", "market_accuracy": "acc_market", "model_accuracy": "acc_model"}
# One unit of the digit SHIPPED stores (log loss to 5 decimals, accuracy to 4):
# the live frame holds other fixtures than the measured one, which moves the
# model's numbers by float noise - on the replay of 2026-11-06 POL's
# 2019-2025 model log loss rounded to 1.04944 against the stored 1.04943.
STORED = {k: (1.5e-5 if k.endswith("log_loss") else 1.5e-4) for k in RECORD}


def _payload(name):
    path = sx.APP_DATA / f"{name}.json"
    if not path.exists():
        pytest.skip(f"no {path.name} on file")
    return json.loads(path.read_text())


def _switched():
    m = _payload("matches")
    if "groups" not in m:
        pytest.skip("matches.json was written before every league was switched on")
    return m


def test_every_league_in_the_app_is_listed_with_every_caveat():
    m = _switched()
    assert m["league_order"] == [c for c in sd.ALL_LEAGUES if c in m["league_order"]]
    assert set(m["league_order"]) <= set(slg.SHIPPED) and set(m["league_order"]) <= set(m["leagues"])
    for e in m["matches"] + m["results"]:
        assert e["league"] in m["leagues"], e.get("match_id")
    for code, lg in m["leagues"].items():
        assert lg["caveats"] == slg.caveats(code), code
    assert set(m["not_in_app"]) == set(slg.SHIPPED) - set(m["league_order"])


def test_each_leagues_record_is_the_one_measured():
    _switched()
    perf = _payload("performance")
    rows = perf["backtest"]["by_league"]
    assert [r["code"] for r in rows] == _payload("matches")["league_order"]
    for r in rows:
        m = slg.SHIPPED[r["code"]]["measured"]
        for period in ("dev", "confirm"):
            got, want = r[period], m[period]
            assert (got["matches"], got["seasons"]) == (want["matches"], want["seasons"]), (r["code"], period)
            for k, v in RECORD.items():
                assert got[k] == pytest.approx(want[v], abs=STORED[k]), (r["code"], period, k)


def test_a_number_a_league_did_not_ship_is_null_where_it_would_show():
    m = _switched()
    names = {"btts": "p_btts", "o15": "p_o15", "o35": "p_o35"}
    for e in m["matches"]:
        if e.get("status") != "upcoming" or not e.get("pick") or not e.get("score_source"):
            continue
        hidden = set(slg.withheld(e["league"], e["score_source"]))
        for k, field in names.items():
            if k in hidden:
                assert e[field] is None, (e["match_id"], field)
        if "exact" in hidden:
            assert e["scores"]["top"] is None, e["match_id"]
        if "o25" in slg.withheld(e["league"], "model"):
            assert e["model_o25"] is None, e["match_id"]
