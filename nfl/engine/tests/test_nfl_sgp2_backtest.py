"""The v2 history and its choice (experiments/nfl_sgp2_backtest.py): the two
numbers come from 2014-2019 alone, by the docstring's objective and
tie-breaks; --choose writes them, generated, into nfl_sgp2.py; the history
on file is the shipped pair's, with everything reported beside it; and the
runner refreshes it without choosing."""

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

import nfl_data as nd
import nfl_sgp as sgp
import nfl_sgp2 as sgp2

from conftest import ENGINE

sys.path.insert(0, str(ENGINE / "experiments"))
import nfl_sgp2_backtest as bk2  # noqa: E402

WF = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "nfl.yml"


# ---------------------------------------------------------------- synthetic games

def _record(rng, season, i):
    """One game as records() gives it, with random chances and a random result."""
    p_home = rng.uniform(0.2, 0.85)
    tds = [{"gsis_id": f"{season}-{i}-{j}", "name": f"p{j}", "position": "RB",
            "team": rng.choice(["KC", "BUF"]), "prob": rng.uniform(0.1, 0.65),
            "rookie": rng.random() < 0.2} for j in range(rng.randint(2, 7))]
    legs = sgp.game_legs("KC", "BUF", p_home=p_home, spread_line=rng.choice([3.5, -3.5, 7.0]),
                         home_spread_price=1.91, away_spread_price=1.91, total_line=45.5,
                         over_price=1.91, under_price=1.91, td=tds)
    pick, p = sgp2.pick_of("KC", "BUF", p_home)
    return {"game_id": f"{season}_{i}", "season": season, "week": 1, "game_type": "REG", "home": "KC",
            "away": "BUF", "pick": pick, "p_pick": p, "legs": legs, "result": _result(rng, tds)}


def _result(rng, tds):
    return sgp.outcome("KC", "BUF", rng.randint(10, 35), rng.randint(10, 35),
                       {t["gsis_id"]: float(rng.random() < 0.45) for t in tds})


@pytest.fixture(scope="module")
def recs():
    rng = random.Random(3)
    return [_record(rng, season, i) for season in range(2014, 2026) for i in range(60)]


def test_the_choice_reads_2014_2019_only(recs):
    chosen = bk2.choose(recs)
    assert chosen is not None and chosen["pairs"] == len(sgp2.CONFIDENT_GRID) * len(sgp2.TD_FLOOR_GRID)
    pair = (chosen["confident"], chosen["td_floor"])
    dev_games = sum(sgp2.DEV[0] <= r["season"] <= sgp2.DEV[1] for r in recs)
    assert chosen["dev"]["games"] == dev_games
    # 2020 on dropped, or every result and chance of it scrambled: the same pair, the same figures.
    assert bk2.choose([r for r in recs if r["season"] <= sgp2.DEV[1]]) == chosen
    rng = random.Random(99)
    later = [r if r["season"] <= sgp2.DEV[1] else _record(rng, r["season"], int(r["game_id"].split("_")[1]))
             for r in recs]
    assert bk2.choose(later) == chosen and (bk2.choose(later)["confident"], bk2.choose(later)["td_floor"]) == pair
    # DEV's own results do reach it.
    redrawn = [dict(r, result=_result(rng, [{"gsis_id": leg["gsis_id"]} for leg in r["legs"]
                                            if leg["kind"] == "TD"]))
               if r["season"] <= sgp2.DEV[1] else r for r in recs]
    assert bk2.choose(redrawn)["dev"] != chosen["dev"]


def _cell(confident, td_floor, tickets, won, lost, games=100):
    return {"confident": confident, "td_floor": td_floor, "games": games,
            "small": {"tickets": tickets, "won": won, "lost": lost}}


def test_the_objective_is_the_best_small_hit_rate_above_the_coverage_floor_with_its_tie_breaks():
    assert sgp2.COVERAGE_MIN == 0.25
    # The best hit rate under the coverage floor does not count.
    best, cleared = bk2.objective([_cell(0.75, 0.5, 24, 20, 4), _cell(0.6, 0.35, 60, 30, 30),
                                   _cell(0.65, 0.4, 25, 15, 10)])
    assert (best["confident"], best["td_floor"], cleared) == (0.65, 0.4, 2)    # 25 of 100 is enough
    # A tie in hit rate goes to the larger coverage...
    best, _ = bk2.objective([_cell(0.7, 0.4, 30, 15, 15), _cell(0.6, 0.35, 40, 20, 20)])
    assert best["confident"] == 0.6
    # ...then the lower CONFIDENT, then the lower TD_FLOOR - an exact tie, not a rounded one.
    best, _ = bk2.objective([_cell(0.7, 0.35, 30, 10, 20), _cell(0.65, 0.45, 30, 10, 20),
                             _cell(0.65, 0.4, 30, 10, 20)])
    assert (best["confident"], best["td_floor"]) == (0.65, 0.4)
    assert bk2.objective([_cell(0.6, 0.35, 24, 20, 4)]) == (None, 0)


def _grid_cells(rate):
    """One DEV cell per pair of the grids, SMALL's hit rate rate(i, j) of
    100 graded tickets, built in 30 of 100 games (all clear the floor)."""
    return [_cell(c, t, 30, round(rate(i, j) * 100), 100 - round(rate(i, j) * 100), games=100)
            for i, c in enumerate(sgp2.CONFIDENT_GRID) for j, t in enumerate(sgp2.TD_FLOOR_GRID)]


def test_the_rules_always_rises_claim_is_counted_off_the_grid_never_typed():
    """FOUND AFTER THE RULE WAS FIXED: the docstring's claim is a claim about
    the data, and rises() checks it - every step of the grid, one bar raised
    a step with the other held, compared exactly."""
    steps = 2 * len(sgp2.CONFIDENT_GRID) * len(sgp2.TD_FLOOR_GRID) - len(sgp2.CONFIDENT_GRID) - len(sgp2.TD_FLOOR_GRID)
    up = bk2.rises(_grid_cells(lambda i, j: 0.30 + 0.02 * i + 0.01 * j))
    assert (up["steps"], up["falls"], up["level"], up["fell"]) == (steps, 0, 0, [])
    assert up["claim"] == sgp2.RISES_CLAIM and up["period"] == bk2.DEV_NAME and up["report_only"]
    assert up["note"] == sgp2.rises_note(up) and up["note"].startswith("Checked after the rule was fixed")
    # All pairs clear the floor, so it stops nothing: the highest bars are the best and are chosen.
    assert up["floor_binds"] is False and "did not stop the choice" in up["note"]
    # The top CONFIDENT a step worse at every TD floor, one level step: four falls, one level.
    worse = lambda i, j: 0.30 + 0.02 * i + 0.01 * j - (0.05 if i == 3 else 0) - (0.01 if (i, j) == (0, 1) else 0)   # noqa: E731
    down = bk2.rises(_grid_cells(worse))
    assert (down["falls"], down["level"]) == (len(sgp2.TD_FLOOR_GRID), 1)
    assert all(f["from"][0] == sgp2.CONFIDENT_GRID[2] and f["to"][0] == sgp2.CONFIDENT_GRID[3] for f in down["fell"])
    assert all(f["hit_rates"][1] < f["hit_rates"][0] for f in down["fell"])
    assert f"fell at {down['falls']} of the grid's {steps} steps" in down["note"]
    assert "and stayed level at 1" in down["note"] and f'"{sgp2.RISES_CLAIM}"' in down["note"]
    # A pair under the floor hit more often: the floor did stop the choice, and the note says which.
    cells = _grid_cells(lambda i, j: 0.30 + 0.02 * i + 0.01 * j)
    top = next(c for c in cells if (c["confident"], c["td_floor"]) == (sgp2.CONFIDENT_GRID[-1], sgp2.TD_FLOOR_GRID[-1]))
    top["small"]["tickets"] = 10
    bound = bk2.rises(cells)
    assert bound["floor_binds"] is True and bound["best_without_floor"]["coverage"] == pytest.approx(0.10)
    assert (bound["best_without_floor"]["confident"], bound["best_without_floor"]["td_floor"]) == (
        sgp2.CONFIDENT_GRID[-1], sgp2.TD_FLOOR_GRID[-1])
    assert "The floor did stop the choice" in bound["note"]


def test_the_claim_quoted_is_the_rules_own_and_the_choice_does_not_rest_on_it():
    doc = " ".join(sgp2.__doc__.split())
    assert sgp2.RISES_CLAIM in doc                      # the rule's words, quoted, never reworded
    note = sgp2.rises_note({"period": bk2.DEV_NAME, "steps": 24, "falls": 2, "level": 0})
    assert "The choice does not rest on that sentence" in note and f"{sgp2.COVERAGE_MIN:.0%}" in note
    assert sgp2.rises_note(None) is None and sgp2.rises_note({"steps": 0}) is None


def test_choose_writes_the_constants_with_a_generated_comment_and_keeps_the_rule(tmp_path, recs):
    rule = tmp_path / "nfl_sgp2.py"
    shutil.copy(ENGINE / "nfl_sgp2.py", rule)
    before = rule.read_text()
    doc = before.split('"""')[1]
    chosen = bk2.choose(recs)
    assert bk2.write_choice(chosen, rule, today="2031-01-02") == "2031-01-02"
    text = rule.read_text()
    assert text.split('"""')[1] == doc                             # the docstring, word for word
    assert f"CONFIDENT = {chosen['confident']!r}\nTD_FLOOR = {chosen['td_floor']!r}\n" in text
    assert 'CHOSEN_ON = "2031-01-02"\n' in text
    s = chosen["dev"]["small"]
    assert f"# CHOSEN 2031-01-02 by experiments/nfl_sgp2_backtest.py --choose, on {bk2.DEV_NAME} alone:" in text
    assert f"hit {s['hit_rate']:.1%} (won {s['won']:,}, lost {s['lost']:,})" in text
    # Choosing again replaces the block; nothing else of the file moves.
    bk2.write_choice(chosen, rule, today="2031-01-03")
    again = rule.read_text()
    assert again.count("# CHOSEN ") == 1 and again.count("CONFIDENT = ") == 1
    outside = lambda t: bk2.CHOSEN_BLOCK.sub("", t)                # noqa: E731
    assert outside(again) == outside(before)


# ---------------------------------------------------------------- the history on file

def _on_file():
    path = nd.DATA / f"{bk2.OUT_NAME}.json"
    if not path.exists():
        pytest.skip("no v2 history on file yet")
    return json.loads(path.read_text())


def test_the_shipped_constants_are_what_the_objective_gives_on_the_grid_on_file():
    h = _on_file()
    c = h["chosen"]
    assert (c["confident"], c["td_floor"], c["chosen_on"]) == (sgp2.CONFIDENT, sgp2.TD_FLOOR, sgp2.CHOSEN_ON)
    best, cleared = bk2.objective(bk2.dev_cells(h["grid"]["rows"]))
    assert (best["confident"], best["td_floor"], cleared) == (sgp2.CONFIDENT, sgp2.TD_FLOOR, c["cleared"])
    assert c["reproduced"] is True and c["on"] == list(sgp2.DEV) and c["objective"] == sgp2.OBJECTIVE


def test_the_history_on_file_has_every_kind_and_period_the_reports_beside_it_and_its_caveats():
    h = _on_file()
    assert h["version"] == sgp2.VERSION and h["known_bias"] == sgp.KNOWN_BIAS
    assert h["no_edge"] == sgp.NO_EDGE and h["no_edge_v2"] == sgp2.NO_EDGE
    assert h["payout_basis"] == sgp.EST_PAYOUT and h["est_erratum"] == sgp.EST_ERRATUM
    assert set(h["periods"]) == set(sgp.PERIODS) | {str(h["current_season"]), "all"}
    assert set(h["kinds"]) == set(sgp.KINDS)
    for kind, k in h["kinds"].items():
        for name, m in k["meters"].items():
            assert m["won"] + m["lost"] + m["void"] == m["graded"] <= m["tickets"]
            assert m["tickets"] + m["not_built"] == h["games"][name]
            assert sum(m["not_built_why"].values()) == m["not_built"]
            assert k["coverage"][name] == pytest.approx(m["tickets"] / h["games"][name], abs=1e-4)
            if not h["td_prices"]["on_file"]:
                assert m["roi"] is None and m["roi_reason"] == sgp.NO_TD_PRICE
            # Every game below the bar says so, with the chosen bar in the words.
            assert m["not_built_why"].get(sgp2.not_confident(sgp2.CONFIDENT), 0) <= (
                h["games"][name] - h["confident_games"][name])
    # Report-only beside it: the whole grid, v1 on the same games, the game's-best SMALL.
    assert h["grid"]["report_only"] and len(h["grid"]["rows"]) == 16
    assert {(r["confident"], r["td_floor"]) for r in h["grid"]["rows"]} == {
        (c, f) for c in sgp2.CONFIDENT_GRID for f in sgp2.TD_FLOOR_GRID}
    for r in h["grid"]["rows"]:
        assert set(r["periods"]) == set(h["periods"])
    v1 = h["v1_same_games"]
    assert v1["report_only"] and v1["version"] == sgp.VERSION and set(v1["kinds"]) == set(sgp.KINDS)
    assert v1["games"] == h["confident_games"]
    assert h["alt_small"]["report_only"] and set(h["alt_small"]["periods"]) == set(h["periods"])
    # Found after the rule was fixed, beside the choice: the claim counted
    # off the grid on file, and what "untouched" covers.
    c, again = h["chosen"], bk2.rises(bk2.dev_cells(h["grid"]["rows"]))
    for k in ("claim", "period", "steps", "falls", "level", "floor_binds", "note"):
        assert c["rises"][k] == again[k], k
    assert [(f["from"], f["to"]) for f in c["rises"]["fell"]] == [(f["from"], f["to"]) for f in again["fell"]]
    assert c["untouched"] == h["alt_small"]["untouched"] == sgp2.UNTOUCHED_NOTE
    assert f" chosen on {bk2.nt.DEV[0]}-{bk2.nt.CONFIRM[1]} walk-forward " in h["td_groups"]["note"]


def test_the_history_is_replayed_when_behind_or_for_another_pair(tmp_path, monkeypatch):
    import pandas as pd
    path = tmp_path / "h.json"
    games = pd.DataFrame({"game_id": ["a"], "season": [2026], "played": [True],
                          "kickoff": pd.to_datetime(["2026-11-01 18:00"])})
    box = snaps = pd.DataFrame({"game_id": ["a"]})
    assert bk2.stale(path, games, box, snaps) == "no history on file"
    path.write_text(json.dumps({"version": sgp.VERSION}))
    assert sgp.VERSION in bk2.stale(path, games, box, snaps)
    monkeypatch.setattr(sgp2, "CONFIDENT", 0.65)
    monkeypatch.setattr(sgp2, "TD_FLOOR", 0.4)
    path.write_text(json.dumps({"version": sgp2.VERSION, "through": "2026-11-01T18:00:00Z",
                                "chosen": {"confident": 0.65, "td_floor": 0.4}}))
    assert bk2.stale(path, games, box, snaps) is None
    monkeypatch.setattr(sgp2, "TD_FLOOR", 0.45)
    assert "TD_FLOOR" in bk2.stale(path, games, box, snaps)


def test_the_runner_refreshes_the_v2_history_like_v1s_and_never_chooses():
    nfl = WF.read_text()
    run = nfl.split("python nfl/engine/experiments/nfl_sgp_backtest.py --if-stale")[1]
    assert "python nfl/engine/experiments/nfl_sgp2_backtest.py --if-stale" in run.split("nfl_predict.py")[0]
    assert "--choose" not in nfl


@pytest.mark.snapshot
def test_the_shipped_constants_are_what_choose_gives_on_the_committed_data():
    """--choose replayed on the committed snapshot: the pair nfl_sgp2.py
    holds - and the same with 2020 on dropped, or its results scrambled."""
    data = bk2.bk.inputs()
    recs, _ = bk2.records(data["games"], data["blend"], data["td_rows"], data["prices"])
    pair = (sgp2.CONFIDENT, sgp2.TD_FLOOR)
    chosen = bk2.choose(recs)
    assert (chosen["confident"], chosen["td_floor"]) == pair
    assert bk2.choose([r for r in recs if r["season"] <= sgp2.DEV[1]]) == chosen
    rng = random.Random(1)
    scrambled = [r if r["season"] <= sgp2.DEV[1] else dict(r, result=sgp.outcome(
        r["home"], r["away"], rng.randint(0, 40), rng.randint(0, 40),
        {k: float(rng.random() < 0.5) for k in (r["result"]["players"] or {})})) for r in recs]
    assert bk2.choose(scrambled) == chosen
