"""Tests for the JSON the phone app reads.

The app cannot check its own inputs, so anything wrong here is wrong on a
phone screen with no way to tell. Most of these are about the difference
between a missing number and a confident zero.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import app_export as ax


def test_a_missing_number_is_null_and_never_zero():
    """0.0 on a phone reads as "we measured zero". null reads as "we did not
    measure it", which is the fact. This is the same inversion feature_spec
    exists to prevent, one layer further out."""
    assert ax._clean(float("nan")) is None
    assert ax._clean(float("inf")) is None
    assert ax._clean(None) is None
    assert ax._clean(0.0) == 0.0


def test_the_payload_never_contains_nan(tmp_path):
    """json.dumps writes bare NaN by default, which is not valid JSON and
    makes JSON.parse throw in the browser - a blank screen, not an error."""
    payload = ax.card_payload(
        "Card", "2026-01-01",
        [{"number": 1, "red": "A", "blue": "B", "win_prob": float("nan"),
          "confidence": None}])
    written = ax.write_all(tmp_path, {"card": payload})
    text = written[0].read_text()
    assert "NaN" not in text
    reloaded = json.loads(text)
    assert reloaded["fights"][0]["win_prob"] is None


def test_a_refused_fight_keeps_its_reason():
    """"No data" is an answer the app has to be able to show, and the reason
    is what turns it into something the user can fix - usually a misspelling."""
    payload = ax.card_payload(
        "Card", "2026-01-01", [],
        skipped=[{"fight": "A vs B", "reason": "No data for 'A'"}])
    assert payload["skipped"][0]["reason"] == "No data for 'A'"


def test_every_card_carries_the_caveats():
    """A probability shown with no context is how -1.8% gets mistaken for an
    edge. The app renders these; it does not get to omit them."""
    payload = ax.card_payload("Card", "2026-01-01", [])
    for key in ("model", "simulation", "parlay"):
        assert len(payload["caveats"][key]) > 60


def test_the_simulation_reports_which_rates_were_assumed():
    payload = ax.card_payload("Card", "2026-01-01", [{
        "red": "A", "blue": "B",
        "simulation": {"red_win": 0.6, "blue_win": 0.4, "draw": 0.0,
                       "imputed_red": ["td_acc"], "imputed_blue": ["td_acc",
                                                                  "td_def"]},
    }])
    assert payload["fights"][0]["simulation"]["assumed"] == ["td_acc", "td_def"]


def test_a_fight_without_a_simulation_says_so_rather_than_inventing_one():
    payload = ax.card_payload("Card", "2026-01-01",
                              [{"red": "A", "blue": "B", "simulation": None}])
    assert payload["fights"][0]["simulation"] is None


def test_a_parlay_multiplies_its_legs_and_prices_them():
    parlays = ax.parlay_payload([
        {"matchup": "A vs B", "pick": "A", "prob": 0.5, "tier": "LOCK"},
        {"matchup": "C vs D", "pick": "C", "prob": 0.5, "tier": "LOCK"},
    ])
    assert len(parlays) == 1
    assert parlays[0]["combined_prob"] == pytest.approx(0.25)
    # Break-even on a 25% shot is +300.
    assert parlays[0]["fair_odds"] == 300


def test_a_parlay_leg_with_no_probability_gives_no_combined_number():
    """Multiplying through a missing leg would quietly drop it and overstate
    the parlay."""
    parlays = ax.parlay_payload([
        {"matchup": "A vs B", "pick": "A", "prob": 0.5, "tier": "LOCK"},
        {"matchup": "C vs D", "pick": "C", "prob": None, "tier": "LOCK"},
    ])
    assert parlays[0]["combined_prob"] is None
    assert parlays[0]["fair_odds"] is None


def test_too_few_legs_produces_no_parlay_at_all():
    assert ax.parlay_payload([{"matchup": "A vs B", "pick": "A",
                               "prob": 0.7, "tier": "LOCK"}]) == []


def test_fighters_are_keyed_by_id_because_names_collide():
    """Two fighters in this dataset are both called Bruno Silva."""
    frame = pd.DataFrame({"cd_bouts": [3.0, 5.0], "cd_slpm": [4.0, float("nan")]},
                         index=["id-1", "id-2"])
    payload = ax.fighters_payload(frame)
    assert payload["count"] == 2
    assert payload["ids"] == ["id-1", "id-2"]
    bouts = payload["columns"].index("cd_bouts")
    slpm = payload["columns"].index("cd_slpm")
    assert payload["values"][0][bouts] == 3.0
    assert payload["values"][1][slpm] is None


def test_the_columnar_rows_line_up_with_the_columns():
    """A row that does not match the header silently shifts every stat by one,
    which reads as a plausible fighter rather than as an error."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=["x", "y"])
    payload = ax.fighters_payload(frame)
    assert len(payload["ids"]) == len(payload["values"])
    for row in payload["values"]:
        assert len(row) == len(payload["columns"])


def test_the_performance_payload_says_which_number_counts():
    """The tune-period +6.1% is exactly the figure that would get
    screenshotted, so the payload has to say what it is."""
    payload = ax.performance_payload(
        {"summary": {"total": 300, "graded": 200, "correct": 120,
                     "pending": 100, "accuracy": 0.6}})
    assert payload["track_record"]["accuracy"] == 0.6
    assert "confirm" in payload["notes"]["which_number_counts"].lower()
    assert "fiction" in payload["notes"]["leak"].lower()


def test_writing_twice_with_the_same_data_produces_the_same_bytes(tmp_path):
    """Apart from the timestamp. Unsorted keys would make every daily commit a
    full-file diff and hide the change that mattered."""
    frame = pd.DataFrame({"cd_bouts": [3.0]}, index=["id-1"])
    first = ax.fighters_payload(frame)
    second = ax.fighters_payload(frame)
    first["generated"] = second["generated"] = "fixed"
    a = ax.write_all(tmp_path / "a", {"f": first})[0].read_text()
    b = ax.write_all(tmp_path / "b", {"f": second})[0].read_text()
    assert a == b


def test_the_export_round_trips_through_real_career_stats(tmp_path):
    """The one end-to-end check: real columns, real NaNs, valid JSON out."""
    import career_stats as cs
    csv = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
    if not csv.exists():
        pytest.skip("dataset not present")
    df = pd.read_csv(csv, low_memory=False).sort_values("date")
    final = cs.final_stats(df.reset_index(drop=True)).head(50)
    payload = ax.fighters_payload(final)
    path = ax.write_all(tmp_path, {"fighters": payload})[0]
    reloaded = json.loads(path.read_text())
    assert reloaded["count"] == 50
    for row in reloaded["values"]:
        assert len(row) == len(reloaded["columns"])
        for value in row:
            assert value is None or not math.isnan(value)


def test_a_five_round_fight_is_exported_as_five_rounds():
    """The prediction dict does not carry is_5rnd, so an earlier version read
    every main event as a three-round bout while its own simulation ran five.
    """
    payload = ax.card_payload("Card", "2026-01-01", [{
        "red": "A", "blue": "B", "rounds_scheduled": 5, "title_fight": True,
        "simulation": {"red_win": 0.5, "blue_win": 0.5,
                       "finish_by_round": [0.1, 0.1, 0.1, 0.1, 0.1]},
    }])
    fight = payload["fights"][0]
    assert fight["rounds_scheduled"] == 5
    assert fight["title_fight"] is True
    assert len(fight["simulation"]["finish_by_round"]) == fight["rounds_scheduled"]


def test_the_strategy_table_is_read_from_disk_not_carried_as_constants(tmp_path,
                                                                       monkeypatch):
    """The app used to carry three hand-typed ROI figures. Re-running the
    backtest left the phone showing the old ones with nothing to say so."""
    import build_app_data as bad
    monkeypatch.setattr(bad, "EXPERIMENTS", tmp_path)
    rows, meta = bad.strategies()
    assert rows == [] and meta == {}, "missing file must yield nothing, not defaults"

    (tmp_path / "strategies.json").write_text(json.dumps({
        "confirm_from": 2020, "flag_quality": 0.7, "flagged": 565,
        "generated": "2026-09-19T00:00:00Z",
        "strategies": [{"label": "MODEL", "bets": 10, "hit_rate": 0.6,
                        "roi": -0.02, "roi_low": -0.1, "roi_high": 0.06}],
        "parlays": [],
    }))
    rows, meta = bad.strategies()
    assert rows[0]["name"] == "MODEL"
    assert rows[0]["roi"] == -0.02
    assert rows[0]["description"], "a strategy with no description is unreadable"
    assert meta["flag_quality"] == 0.7


def test_no_roi_figure_is_hardcoded_in_the_builder():
    """A regression guard with teeth: the numbers must come from a file."""
    import re
    source = (ENGINE / "build_app_data.py").read_text()
    # Any float that looks like a ROI or hit rate sitting in a dict literal.
    suspects = re.findall(r'"(?:roi|hit_rate|bets)":\s*[-0-9]', source)
    assert not suspects, f"hand-typed strategy numbers are back: {suspects}"


def test_a_strategy_that_placed_no_bets_serialises_as_null_not_a_crash():
    """summarise() returns NaN for an empty strategy and json.dumps writes a
    bare NaN, which is not valid JSON. It took down the whole backtest run in
    CI after the strategy writer was added."""
    from strategies import summarise
    empty = summarise([], "EMPTY")
    assert empty["roi"] != empty["roi"], "fixture assumes NaN"
    payload = {"strategies": [{k: ax._clean(v) for k, v in empty.items()}]}
    text = json.dumps(payload, allow_nan=False)   # must not raise
    assert json.loads(text)["strategies"][0]["roi"] is None
