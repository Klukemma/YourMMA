"""Anytime-TD prices for the same-game tickets (nfl_odds): the per-event
reply parsed into a consensus Yes price, the credit budget kept (only games
near kickoff, once per stale period, never in play, never under the
reserve), the key kept quiet, and a feed name reaching a player only when
it can be no one else of that game.

The replies under fixtures/odds are in the shape the-odds-api documents for
its per-event odds endpoint (market player_anytime_td) and its free events
list - an event's bookmakers[].markets[].outcomes[], each outcome
{name: Yes | No, description: the player, price}. They are written by hand,
not recorded: the sandbox has no key and must not spend credits. The
players are fictional, each name there for one case of td_prices_for. No
test touches the network."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import nfl_odds as no

FIX = Path(__file__).resolve().parent / "fixtures" / "odds"
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)        # LV @ KC kicks off 32 hours later
KICKOFF = pd.Timestamp("2026-10-25 20:25")
EVENT_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
UPCOMING = pd.DataFrame({"away": ["LV", "MIA", "DEN"], "home": ["KC", "BUF", "KC"],
                         "kickoff": pd.to_datetime(["2026-10-25 20:25", "2026-10-26 00:20",
                                                    "2026-11-01 17:00"])})


def _load(name):
    return json.loads((FIX / name).read_text())


def _cache_with_prices(now=NOW):
    cache = no.empty()
    key, entry = no.parse_td_event(_load("td_event.json"))
    assert no.merge_td(cache, key, entry, now=now)
    return cache


class _Reply:
    def __init__(self, body, status=200, remaining="420"):
        self.status_code, self._body = status, body
        self.headers = {} if remaining is None else {"x-requests-remaining": remaining}
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _fake_get(calls, remaining="420"):
    def get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if url == no.EVENTS_ENDPOINT:
            return _Reply(_load("events.json"))
        return _Reply(_load("td_event.json"), remaining=remaining)
    return get


# ---------------------------------------------------------------- the reply

def test_the_yes_price_is_the_median_across_books_in_probability_space():
    key, e = no.parse_td_event(_load("td_event.json"))
    assert key == "2026-10-25|LV|KC" and e["event_id"] == EVENT_ID and e["books"] == 3
    tight = e["players"]["Sam Tight"]
    assert tight["price"] == no.consensus_price([150, 140, 155])
    assert (tight["best_price"], tight["best_book"], tight["books"]) == (155, "BetMGM", 3)
    # Odds-on prices straddle nothing here, but the median is still taken in
    # probability space, as the moneyline's is.
    assert e["players"]["Tom Runner Jr."]["price"] == no.consensus_price([-120, -115, -125])


def test_only_yes_outcomes_count_and_a_book_quoting_a_player_twice_is_left_out():
    _, e = no.parse_td_event(_load("td_event.json"))
    # FanDuel's "No" prices are not Yes prices: Sam Tight's best stays BetMGM's +155.
    assert all(p["price"] > -200 for p in e["players"].values())
    # DraftKings quoted Max Wide at +300 and +320: neither counts.
    wide = e["players"]["Max Wide"]
    assert wide["books"] == 2 and wide["best_book"] == "FanDuel" and wide["best_price"] == 290


def test_an_unknown_team_is_not_guessed():
    event = dict(_load("td_event.json"), home_team="Springfield Atoms")
    assert no.parse_td_event(event) == (None, None)


def test_an_in_play_reply_is_never_kept():
    cache = no.empty()
    key, entry = no.parse_td_event(_load("td_event.json"))
    started = datetime(2026, 10, 25, 20, 30, tzinfo=timezone.utc)
    assert not no.merge_td(cache, key, entry, now=started)
    assert cache["td"] == {}
    assert no.merge_td(cache, key, entry, now=NOW) and cache["td"][key]["seen"] == "2026-10-24T12:00:00Z"


# ---------------------------------------------------------------- the budget

def test_one_call_is_one_market_in_one_region_on_the_per_event_endpoint():
    # 1 credit a game: the cost the workflow's schedule is budgeted on.
    assert no.TD_PARAMS == {"regions": "us", "markets": "player_anytime_td", "oddsFormat": "american"}
    assert len(no.TD_PARAMS["markets"].split(",")) * len(no.TD_PARAMS["regions"].split(",")) == 1
    assert no.EVENT_ODDS_ENDPOINT.format(event_id="X").endswith("/americanfootball_nfl/events/X/odds")


def test_only_games_near_kickoff_are_due_and_not_again_until_stale():
    cache = no.empty()
    # MIA @ BUF is 36 h 20 min away and DEN @ KC a week: not yet.
    assert no.td_due(cache, UPCOMING, now=NOW) == [("LV", "KC", KICKOFF)]
    cache = _cache_with_prices()
    assert no.td_due(cache, UPCOMING, now=NOW) == []
    later = datetime(2026, 10, 25, 7, 59, tzinfo=timezone.utc)         # 19 h 59 min on
    # MIA @ BUF is inside the window by now; LV @ KC was bought too recently.
    assert no.td_due(cache, UPCOMING, now=later) == [("MIA", "BUF", pd.Timestamp("2026-10-26 00:20"))]
    stale = datetime(2026, 10, 25, 8, 0, tzinfo=timezone.utc)
    assert ("LV", "KC", KICKOFF) in no.td_due(cache, UPCOMING, now=stale)


def test_a_refresh_asks_again_but_never_for_a_game_under_way():
    cache = _cache_with_prices()
    assert no.td_due(cache, UPCOMING, refresh=True, now=NOW) == [("LV", "KC", KICKOFF)]
    kicked = datetime(2026, 10, 25, 20, 25, tzinfo=timezone.utc)
    assert ("LV", "KC", KICKOFF) not in no.td_due(no.empty(), UPCOMING, refresh=True, now=kicked)


def test_the_event_id_comes_from_the_free_events_list_and_each_due_game_is_one_call(monkeypatch):
    import requests
    calls, lines = [], []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    cache = no.empty()
    assert no.refresh_td(cache, UPCOMING, api_key="SECRET123", refresh_asked=False,
                         log=lines.append, now=NOW) == "fetched"
    assert [url for url, _ in calls] == [no.EVENTS_ENDPOINT, no.EVENT_ODDS_ENDPOINT.format(event_id=EVENT_ID)]
    assert calls[1][1] == dict(no.TD_PARAMS, apiKey="SECRET123")
    assert set(cache["td"]) == {"2026-10-25|LV|KC"}
    fetch = cache["fetches"][-1]
    assert fetch["market"] == no.TD_MARKET and fetch["calls"] == 1 and fetch["credits_remaining"] == "420"
    assert "Credits left: 420" in lines[-1] and "SECRET123" not in "".join(lines)
    # Bought: the next run makes no call.
    calls.clear()
    assert no.refresh_td(cache, UPCOMING, api_key="SECRET123", refresh_asked=False,
                         log=lines.append, now=NOW) == "cached"
    assert calls == []


def test_an_event_id_the_odds_reply_kept_needs_no_events_call(monkeypatch):
    import requests
    calls = []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    cache = no.empty()
    odds_event = {"id": EVENT_ID, "home_team": "Kansas City Chiefs", "away_team": "Las Vegas Raiders",
                  "commence_time": "2026-10-25T20:25:00Z", "bookmakers": [
                      {"key": "dk", "title": "DraftKings", "markets": [{"key": "h2h", "outcomes": [
                          {"name": "Kansas City Chiefs", "price": -200},
                          {"name": "Las Vegas Raiders", "price": 170}]}]}]}
    parsed, _ = no.parse_events([odds_event])
    no.merge(cache, parsed, now=NOW)
    assert no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lambda *_: None,
                         now=NOW) == "fetched"
    assert [url for url, _ in calls] == [no.EVENT_ODDS_ENDPOINT.format(event_id=EVENT_ID)]


def test_no_key_means_no_call(monkeypatch):
    import requests

    def boom(*a, **k):
        raise AssertionError("no call without a key")
    monkeypatch.setattr(requests, "get", boom)
    lines = []
    assert no.refresh_td(no.empty(), UPCOMING, api_key="", refresh_asked=False, log=lines.append,
                         now=NOW) == "no key"
    assert "ODDS_API_KEY is not set" in lines[0]


def test_no_td_call_below_the_credits_kept_for_the_lines(monkeypatch):
    import requests
    calls, lines = [], []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    cache = no.empty()
    no.record_fetch(cache, 200, 28, str(no.TD_RESERVE_CREDITS - 1), now=NOW)
    no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lines.append, now=NOW)
    assert [url for url, _ in calls] == [no.EVENTS_ENDPOINT]          # the free list only
    assert cache["td"] == {} and any("no more TD calls" in line for line in lines)


def test_the_key_is_redacted_from_a_failed_td_call(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("GET https://x/events/E/odds?apiKey=SECRET123 failed")
    monkeypatch.setattr(requests, "get", boom)
    lines = []
    assert no.fetch_td("SECRET123", "E", log=lines.append) == (None, None, None)
    assert "SECRET123" not in "".join(lines) and "<redacted>" in "".join(lines)


def test_an_old_cache_without_td_prices_loads_and_keeps_its_schema(tmp_path):
    old = {"schema": no.SCHEMA, "games": {}, "fetches": []}
    (tmp_path / "c.json").write_text(json.dumps(old))
    cache = no.load(tmp_path / "c.json")
    assert no.td_due(cache, UPCOMING, now=NOW) == [("LV", "KC", KICKOFF)]
    key, entry = no.parse_td_event(_load("td_event.json"))
    no.merge_td(cache, key, entry, now=NOW)
    no.save(cache, tmp_path / "c.json")
    again = no.load(tmp_path / "c.json")
    assert again["schema"] == no.SCHEMA and set(again["td"]) == {key}


# ---------------------------------------------------------------- the match

def _prices(players, cache=None, away="LV", home="KC", kickoff=KICKOFF, now=NOW):
    return no.td_prices_for(cache or _cache_with_prices(), away, home, kickoff, players, now=now)


def test_a_price_reaches_a_player_by_his_exact_name_then_a_strict_normalised_one():
    got = _prices([("kc1", ["Tom Runner"]), ("kc2", ["Sam Tight"]), ("kc3", ["Samantha Wide", "Max Wide"])])
    assert got["kc2"]["match"] == "exact" and got["kc2"]["price"] == no.consensus_price([150, 140, 155])
    assert got["kc1"]["match"] == "normalised" and got["kc1"]["feed_name"] == "Tom Runner Jr."
    assert got["kc3"]["feed_name"] == "Max Wide"                      # the roster's name for him
    assert all(p["missing"] is None and p["as_of"] == "2026-10-24T12:00:00Z" for p in got.values())
    assert no.td_name_key("D.J. Catcher") == no.td_name_key("DJ Catcher") == "dj catcher"
    assert no.td_name_key("Kenneth Walker III") == "kenneth walker"
    assert no.td_name_key("Gabe Davis") != no.td_name_key("Gabriel Davis")


def test_an_unmatched_player_gets_no_price_and_the_reason():
    got = _prices([("kc9", ["Nicholas Name"]), ("kc2", ["Sam Tight"])])
    assert got["kc9"] == {"price": None, "missing": no.TD_PRICE_MISSING["unmatched"]}
    assert got["kc2"]["price"] is not None


def test_a_name_that_could_be_two_players_or_two_prices_prices_no_one():
    # "Chris Hands" is quoted once; this game has a Chris Hands and a Chris Hands Jr.
    got = _prices([("kc4", ["Chris Hands"]), ("lv3", ["Chris Hands Jr."]),
                   ("lv2", ["Dj Catcher"]), ("lv4", ["DJ Catcher", "D.J. Catcher"])])
    ambiguous = no.TD_PRICE_MISSING["ambiguous"]
    assert got["kc4"]["missing"] == ambiguous and got["lv3"]["missing"] == ambiguous
    # Neither of lv2's spellings is quoted exactly, and two feed entries share
    # its key; lv4's two names are both quoted.
    assert got["lv2"]["missing"] == ambiguous and got["lv4"]["missing"] == ambiguous
    assert all(p["price"] is None for p in got.values())


def test_a_price_is_only_ever_the_games_own():
    cache = _cache_with_prices()
    # Another game: nothing, though the same names are quoted for LV @ KC.
    far = _prices([("kc2", ["Sam Tight"])], cache, away="DEN", home="KC", kickoff=pd.Timestamp("2026-11-01 17:00"))
    assert far["kc2"] == {"price": None, "missing": no.TD_PRICE_MISSING["window"]}
    near = _prices([("x", ["Sam Tight"])], cache, away="MIA", home="BUF", kickoff=pd.Timestamp("2026-10-26 00:20"),
                   now=datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc))
    assert near["x"] == {"price": None, "missing": no.TD_PRICE_MISSING["none"]}
    # The same two teams two weeks on are another game too.
    later = _prices([("kc2", ["Sam Tight"])], cache, kickoff=pd.Timestamp("2026-11-08 18:00"))
    assert later["kc2"]["price"] is None


def test_a_book_with_no_td_prices_posted_says_so():
    cache = no.empty()
    key, entry = no.parse_td_event(dict(_load("td_event.json"), bookmakers=[]))
    no.merge_td(cache, key, entry, now=NOW)
    assert _prices([("kc2", ["Sam Tight"])], cache)["kc2"]["missing"] == no.TD_PRICE_MISSING["unposted"]


def test_a_price_seen_at_or_after_kickoff_is_never_read():
    cache = _cache_with_prices()
    cache["td"]["2026-10-25|LV|KC"]["seen"] = "2026-10-25T20:25:00Z"
    got = _prices([("kc2", ["Sam Tight"])], cache, now=datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc))
    assert got["kc2"]["price"] is None
