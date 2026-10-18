"""Prices: parsing the reply, the cache's rules, and the key kept quiet."""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import nfl_odds as no

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _event(home="Kansas City Chiefs", away="Las Vegas Raiders", start="2026-10-25T20:25:00Z", books=None):
    books = books or [
        {"key": "dk", "title": "DraftKings", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": -400}, {"name": away, "price": 320}]},
            {"key": "spreads", "outcomes": [{"name": home, "price": -110, "point": -9.5},
                                            {"name": away, "price": -110, "point": 9.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 44.5},
                                           {"name": "Under", "price": -110, "point": 44.5}]}]},
        {"key": "fd", "title": "FanDuel", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": -380}, {"name": away, "price": 300}]},
            {"key": "spreads", "outcomes": [{"name": home, "price": -110, "point": -9.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 45.5}]}]},
    ]
    return {"home_team": home, "away_team": away, "commence_time": start, "bookmakers": books}


def test_parse_maps_names_flips_the_spread_and_takes_medians():
    parsed, unknown = no.parse_events([_event()])
    assert unknown == []
    (key, e), = parsed.items()
    assert key == "2026-10-25|LV|KC"
    # The API quotes the home handicap (-9.5); the schedule convention is the
    # home side's expected margin (+9.5).
    assert e["spread"] == 9.5
    assert e["total"] == 44.5                     # a line a book hung, not 45.0
    assert e["home_ml"] == pytest.approx(-389.8, abs=0.1)
    assert e["away_ml"] == pytest.approx(309.8, abs=0.1)
    assert 0.75 < e["home_prob"] < 0.79
    assert e["best_away_ml"] == 320 and e["best_away_book"] == "DraftKings"


def test_unknown_team_names_are_reported_not_guessed():
    parsed, unknown = no.parse_events([_event(home="Springfield Atoms")])
    assert parsed == {} and unknown == ["Springfield Atoms"]


def test_in_play_lines_are_never_stored_and_the_opening_line_is_kept():
    cache = no.empty()
    parsed, _ = no.parse_events([_event()])
    assert no.merge(cache, parsed, now=NOW) == (1, 0, 0)
    later, _ = no.parse_events([_event(books=[{"key": "dk", "title": "DraftKings", "markets": [
        {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": -10.5}]}]}])])
    assert no.merge(cache, later, now=NOW) == (0, 1, 0)
    entry = next(iter(cache["games"].values()))
    assert entry["opening"]["spread"] == 9.5 and entry["latest"]["spread"] == 10.5
    started = datetime(2026, 10, 25, 21, 0, tzinfo=timezone.utc)
    assert no.merge(cache, parsed, now=started) == (0, 0, 1)


def test_a_call_is_made_only_when_needed():
    cache = no.empty()
    upcoming = pd.DataFrame({"away": ["LV"], "home": ["KC"], "kickoff": [pd.Timestamp("2026-10-25 20:25")]})
    assert "no price" in no.needs_fetch(cache, upcoming, now=NOW)
    parsed, _ = no.parse_events([_event()])
    no.merge(cache, parsed, now=NOW)
    assert no.needs_fetch(cache, upcoming, now=NOW) is None
    assert no.needs_fetch(cache, upcoming, refresh=True, now=NOW) == "refresh asked for"
    stale = datetime(2026, 10, 24, 13, 0, tzinfo=timezone.utc)
    assert "days old" in no.needs_fetch(cache, upcoming, now=stale)


def test_no_key_means_no_call_and_says_so():
    cache = no.empty()
    upcoming = pd.DataFrame({"away": ["LV"], "home": ["KC"], "kickoff": [pd.Timestamp("2026-10-25 20:25")]})
    lines = []
    assert no.refresh(cache, upcoming, api_key="", refresh_asked=False, log=lines.append, now=NOW) == "no key"
    assert "ODDS_API_KEY is not set" in lines[0]


def test_the_key_is_redacted_from_failures(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("GET https://x/?apiKey=SECRET123 failed")
    monkeypatch.setattr(requests, "get", boom)
    lines = []
    status, events, remaining = no.fetch("SECRET123", log=lines.append)
    assert status is None and events == []
    assert "SECRET123" not in "".join(lines) and "<redacted>" in "".join(lines)


def test_the_schedule_line_is_the_fallback():
    game = {"away": "LV", "home": "KC", "kickoff": pd.Timestamp("2026-10-25 20:25"),
            "home_ml": -400.0, "away_ml": 320.0, "spread_line": 9.5, "total_line": 44.5}
    m = no.market_for(no.empty(), game)
    assert m["source"] == "nflverse" and m["spread"] == 9.5 and 0.75 < m["home_prob"] < 0.8


def test_the_request_prices_three_markets_in_one_region():
    # 3 credits a call: the cost the workflow's schedule is budgeted on.
    assert len(no.PARAMS["markets"].split(",")) * len(no.PARAMS["regions"].split(",")) == 3
    assert no.ENDPOINT.endswith("/americanfootball_nfl/odds")


def test_cache_round_trips_and_refuses_a_foreign_schema(tmp_path):
    cache = no.empty()
    no.save(cache, tmp_path / "c.json")
    assert no.load(tmp_path / "c.json") == cache
    (tmp_path / "bad.json").write_text(json.dumps({"schema": 99}))
    with pytest.raises(ValueError):
        no.load(tmp_path / "bad.json")


def test_a_consensus_price_straddling_even_money_is_a_real_price():
    # -110 and +105 average to -2.5 as raw numbers; in probability space they
    # meet near even money.
    price = no.consensus_price([-110, 105])
    assert abs(price) >= 100
    assert no.consensus_price([-400, -380]) < -380


def test_the_line_is_one_a_book_offered():
    assert no.offered_line([3.0, 3.5]) == 3.0
    assert no.offered_line([44.5, 45.5, 45.5]) == 45.5


def test_a_stale_price_loses_to_the_schedule_line():
    cache = no.empty()
    parsed, _ = no.parse_events([_event()])
    no.merge(cache, parsed, now=NOW)
    game = {"away": "LV", "home": "KC", "kickoff": pd.Timestamp("2026-10-25 20:25"),
            "home_ml": -800.0, "away_ml": 550.0, "spread_line": 14.0, "total_line": 40.5}
    assert no.market_for(cache, game, now=NOW)["source"] == "the-odds-api"
    later = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
    m = no.market_for(cache, game, now=later)
    assert m["source"] == "nflverse" and m["spread"] == 14.0
