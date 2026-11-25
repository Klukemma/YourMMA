"""Anytime-TD prices for the same-game tickets and the TD lists (nfl_odds):
the per-event reply parsed into a consensus Yes price, the credit budget
kept (games near kickoff once per stale period, and each game early within
TD_EARLY_DAYS - asked again, spaced out and capped, while the books have not
posted; never in play, never under the reserve, an early call the first left
out), the key kept quiet, a feed name reaching a player only
when it can be no one else of that game, the price a TD list shows, and the
runner's monthly spend replayed.

The replies under fixtures/odds are in the shape the-odds-api documents for
its per-event odds endpoint (market player_anytime_td) and its free events
list - an event's bookmakers[].markets[].outcomes[], each outcome
{name: Yes | No, description: the player, price}. They are written by hand,
not recorded: the sandbox has no key and must not spend credits. The
players are fictional, each name there for one case of td_prices_for. No
test touches the network."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import nfl_odds as no

FIX = Path(__file__).resolve().parent / "fixtures" / "odds"
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)        # LV @ KC kicks off 32 hours later
KICKOFF = pd.Timestamp("2026-10-25 20:25")
EVENT_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
MIA_BUF_ID = "0f9e8d7c6b5a49382716f5e4d3c2b1a0"
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
    """The events list, LV @ KC's priced reply for its id, and for any other
    event its own reply with no bookmaker yet - books not posted."""
    def get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if url == no.EVENTS_ENDPOINT:
            return _Reply(_load("events.json"))
        if url == no.EVENT_ODDS_ENDPOINT.format(event_id=EVENT_ID):
            return _Reply(_load("td_event.json"), remaining=remaining)
        event = next(e for e in _load("events.json") if url == no.EVENT_ODDS_ENDPOINT.format(event_id=e["id"]))
        return _Reply(dict(event, bookmakers=[]), remaining=remaining)
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


def test_a_reply_that_comes_back_after_kickoff_is_in_play_and_not_kept(monkeypatch):
    """The run reads its clock once, at the start; a TD call made later can
    come back after kickoff with in-play prices. Those are judged - and
    stamped - by when the reply came back, never by the run's clock."""
    import requests
    kicked = datetime(2026, 10, 25, 20, 25, tzinfo=timezone.utc)
    run_clock = datetime(2026, 10, 25, 20, 23, 30, tzinfo=timezone.utc)     # 90 s before kickoff
    late = datetime(2026, 10, 25, 20, 26, tzinfo=timezone.utc)              # the reply, 60 s after it
    calls = []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    cache = no.empty()
    upcoming = UPCOMING.iloc[:1]            # LV @ KC alone: MIA @ BUF's early call is not the point here
    no.refresh_td(cache, upcoming, api_key="K", refresh_asked=False, log=lambda *_: None,
                  now=run_clock, clock=lambda: late)
    assert calls and cache["td"] == {}
    # The game counts as due by the run's clock, yet nothing in-play reaches a ticket.
    prices = no.td_prices_for(cache, "LV", "KC", KICKOFF, [("tight", ["Sam Tight"])], now=run_clock)
    assert prices["tight"]["price"] is None and prices["tight"]["missing"] == no.TD_PRICE_MISSING["none"]
    # A pre-game price already on file stays, as it was seen.
    cache = _cache_with_prices()
    no.refresh_td(cache, upcoming, api_key="K", refresh_asked=True, log=lambda *_: None,
                  now=run_clock, clock=lambda: late)
    assert cache["td"]["2026-10-25|LV|KC"]["seen"] == "2026-10-24T12:00:00Z"
    # A reply that came back before kickoff is stamped when it came back.
    just_before = kicked.replace(minute=24, second=10)
    cache = no.empty()
    no.refresh_td(cache, upcoming, api_key="K", refresh_asked=False, log=lambda *_: None,
                  now=run_clock, clock=lambda: just_before)
    assert cache["td"]["2026-10-25|LV|KC"]["seen"] == "2026-10-25T20:24:10Z"
    assert cache["fetches"][-1]["at"] == "2026-10-25T20:24:10Z"
    # And a clock that reads earlier than the run's never stamps before the run.
    cache = no.empty()
    no.refresh_td(cache, upcoming, api_key="K", refresh_asked=False, log=lambda *_: None,
                  now=run_clock, clock=lambda: NOW)
    assert cache["td"]["2026-10-25|LV|KC"]["seen"] == "2026-10-25T20:23:30Z"


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
                         log=lines.append, now=NOW, clock=lambda: NOW) == "fetched"
    # LV @ KC in its window, then MIA @ BUF's first early call (DEN @ KC is
    # more than TD_EARLY_DAYS away).
    assert [url for url, _ in calls] == [no.EVENTS_ENDPOINT, no.EVENT_ODDS_ENDPOINT.format(event_id=EVENT_ID),
                                         no.EVENT_ODDS_ENDPOINT.format(event_id=MIA_BUF_ID)]
    assert calls[1][1] == dict(no.TD_PARAMS, apiKey="SECRET123")
    assert set(cache["td"]) == {"2026-10-25|LV|KC", "2026-10-26|MIA|BUF"}
    fetch = cache["fetches"][-1]
    assert fetch["market"] == no.TD_MARKET and fetch["calls"] == 2 and fetch["early"] == 1
    assert fetch["credits_remaining"] == "420"
    assert "Credits left: 420" in lines[-1] and "SECRET123" not in "".join(lines)
    # Bought: the next run makes no call.
    calls.clear()
    assert no.refresh_td(cache, UPCOMING, api_key="SECRET123", refresh_asked=False,
                         log=lines.append, now=NOW, clock=lambda: NOW) == "cached"
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
    assert no.refresh_td(cache, UPCOMING.iloc[:1], api_key="K", refresh_asked=False, log=lambda *_: None,
                         now=NOW, clock=lambda: NOW) == "fetched"
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
    no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lines.append, now=NOW,
                  clock=lambda: NOW)
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


# ---------------------------------------------------------------- the early call

def test_each_game_is_due_early_within_td_early_days_and_never_once_started():
    # Every game of the week within TD_EARLY_DAYS with no entry on file:
    # LV @ KC and MIA @ BUF, not DEN @ KC (8 days away).
    assert no.td_early_due(no.empty(), UPCOMING, now=NOW) == [
        ("LV", "KC", KICKOFF), ("MIA", "BUF", pd.Timestamp("2026-10-26 00:20"))]
    cache = _cache_with_prices()
    assert no.td_early_due(cache, UPCOMING, now=NOW) == [("MIA", "BUF", pd.Timestamp("2026-10-26 00:20"))]
    # An entry the books had not posted for is on file: not asked again at
    # once (test_an_unposted_early_entry_is_asked_again_spaced_out_and_capped).
    key, entry = no.parse_td_event(dict(_load("events.json")[1], bookmakers=[]))
    no.merge_td(cache, key, entry, now=NOW)
    assert no.td_early_due(cache, UPCOMING, now=NOW) == []
    # DEN @ KC (2026-11-01 17:00) is due from exactly TD_EARLY_DAYS before kickoff.
    den = ("DEN", "KC", pd.Timestamp("2026-11-01 17:00"))
    edge = datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc) - timedelta(days=no.TD_EARLY_DAYS)
    assert den in no.td_early_due(cache, UPCOMING, now=edge)
    assert den not in no.td_early_due(cache, UPCOMING, now=edge - timedelta(minutes=1))
    kicked = datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc)
    assert den not in no.td_early_due(no.empty(), UPCOMING, now=kicked)


def test_the_window_round_is_bought_first_so_the_reserve_leaves_out_an_early_call(monkeypatch):
    import requests
    calls, lines = [], []
    # One credit above the reserve: the window's call spends it, then no early one.
    monkeypatch.setattr(requests, "get", _fake_get(calls, remaining=str(no.TD_RESERVE_CREDITS - 1)))
    cache = no.empty()
    no.record_fetch(cache, 200, 28, str(no.TD_RESERVE_CREDITS), now=NOW)
    no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lines.append, now=NOW,
                  clock=lambda: NOW)
    assert [url for url, _ in calls] == [no.EVENTS_ENDPOINT, no.EVENT_ODDS_ENDPOINT.format(event_id=EVENT_ID)]
    assert set(cache["td"]) == {"2026-10-25|LV|KC"} and cache["fetches"][-1]["early"] == 0
    assert any("no more TD calls" in line for line in lines)


def test_an_early_call_alone_is_a_round_and_says_so(monkeypatch):
    import requests
    calls, lines = [], []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    cache = _cache_with_prices()                    # LV @ KC bought: only MIA @ BUF's early call is due
    assert no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lines.append, now=NOW,
                         clock=lambda: NOW) == "fetched"
    assert [url for url, _ in calls][-1] == no.EVENT_ODDS_ENDPOINT.format(event_id=MIA_BUF_ID)
    assert cache["fetches"][-1]["calls"] == 1 and cache["fetches"][-1]["early"] == 1
    assert cache["fetches"][-1]["early_posted"] == 0 and cache["fetches"][-1]["early_retries"] == 0
    assert "1 games due (1 early), 1 calls (1 early, 0 of them posted, 0 of them retries" in lines[-1]
    assert cache["td"]["2026-10-26|MIA|BUF"]["early_calls"] == 1
    # Not posted yet: the game says so, and the same run's clock does not buy it again.
    got = _prices([("x", ["Someone"])], cache, away="MIA", home="BUF", kickoff=pd.Timestamp("2026-10-26 00:20"))
    assert got["x"]["missing"] == no.TD_PRICE_MISSING["unposted"]
    calls.clear()
    assert no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lines.append, now=NOW,
                         clock=lambda: NOW) == "cached"
    assert calls == []


DEN_KC = ("DEN", "KC", pd.Timestamp("2026-11-01 17:00"))
DEN_KICKOFF = datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc)


def test_an_unposted_early_entry_is_asked_again_spaced_out_and_capped(monkeypatch):
    """DEN @ KC's books never post early: its first early call comes back
    empty, and it is asked again once the entry is TD_EARLY_RETRY_HOURS old
    - not on the next slot - until TD_EARLY_TRIES early calls, each counted
    in the fetch log as a retry; then it waits for its window round."""
    import requests
    calls, lines = [], []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    upcoming = UPCOMING.iloc[2:]
    first = DEN_KICKOFF - timedelta(days=no.TD_EARLY_DAYS)
    cache = no.empty()
    at = first
    for n in range(1, no.TD_EARLY_TRIES + 1):
        assert no.td_early_due(cache, upcoming, now=at) == [DEN_KC], n
        assert no.refresh_td(cache, upcoming, api_key="K", refresh_asked=False, log=lines.append, now=at,
                             clock=lambda: at) == "fetched"
        f = cache["fetches"][-1]
        assert (f["early"], f["early_posted"], f["early_retries"]) == (1, 0, int(n > 1)), n
        assert cache["td"]["2026-11-01|DEN|KC"]["early_calls"] == n
        # Not again until the entry is TD_EARLY_RETRY_HOURS old.
        soon = at + timedelta(hours=no.TD_EARLY_RETRY_HOURS) - timedelta(minutes=1)
        assert no.td_early_due(cache, upcoming, now=soon) == [], n
        at += timedelta(hours=no.TD_EARLY_RETRY_HOURS)
    assert "0 of them posted, 1 of them retries" in lines[-1]
    assert at < DEN_KICKOFF - timedelta(hours=no.TD_WINDOW_HOURS)       # still outside the window: the cap stops it
    assert no.td_early_due(cache, upcoming, now=at) == []


def test_a_posted_early_entry_or_one_in_its_window_is_never_asked_again_early():
    first = DEN_KICKOFF - timedelta(days=no.TD_EARLY_DAYS)
    later = first + timedelta(hours=no.TD_EARLY_RETRY_HOURS)
    upcoming = UPCOMING.iloc[2:]
    posted = no.empty()
    _, entry = no.parse_td_event(dict(_load("td_event.json"), id=_load("events.json")[2]["id"],
                                      away_team="Denver Broncos", commence_time="2026-11-01T17:00:00Z"))
    assert no.merge_td(posted, "2026-11-01|DEN|KC", entry, now=first) and entry["players"]
    no.note_early_calls(posted, "2026-11-01|DEN|KC", 1)
    assert no.td_early_due(posted, upcoming, now=later) == []
    # Unposted, but its window has started: td_due buys it there, never early.
    unposted = no.empty()
    _, empty = no.parse_td_event(dict(_load("events.json")[2], bookmakers=[]))
    in_window = DEN_KICKOFF - timedelta(hours=no.TD_WINDOW_HOURS)
    assert no.merge_td(unposted, "2026-11-01|DEN|KC", empty, now=in_window - timedelta(hours=no.TD_EARLY_RETRY_HOURS))
    assert no.td_early_due(unposted, upcoming, now=in_window) == []
    assert no.td_due(unposted, upcoming, now=in_window) == [DEN_KC]
    # An entry kept before early calls were counted counts as one.
    assert no.early_tries(unposted["td"]["2026-11-01|DEN|KC"]) == 1 and no.early_tries(None) == 0


def test_the_fetch_log_counts_the_early_calls_that_found_prices_posted(monkeypatch):
    """Four days out, both games of the weekend are bought early: LV @ KC's
    books had posted, MIA @ BUF's had not - the log keeps both counts, so
    how early the books post is measured, not assumed."""
    import requests
    calls, lines = [], []
    monkeypatch.setattr(requests, "get", _fake_get(calls))
    early = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    cache = no.empty()
    assert no.td_due(cache, UPCOMING, now=early) == []
    assert no.refresh_td(cache, UPCOMING, api_key="K", refresh_asked=False, log=lines.append, now=early,
                         clock=lambda: early) == "fetched"
    f = cache["fetches"][-1]
    assert (f["calls"], f["early"], f["early_posted"], f["events"]) == (2, 2, 1, 2)
    assert "2 games due (2 early), 2 calls (2 early, 1 of them posted, 0 of them retries" in lines[-1]
    got = _prices([("kc2", ["Sam Tight"])], cache, now=early)
    assert got["kc2"]["price"] is not None and got["kc2"]["as_of"] == "2026-10-21T12:00:00Z"


def test_without_a_key_the_early_games_are_counted_not_bought(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    lines = []
    assert no.refresh_td(no.empty(), UPCOMING, api_key="", refresh_asked=False, log=lines.append,
                         now=NOW) == "no key"
    assert "2 games due (1 early)" in lines[0] and "ODDS_API_KEY is not set" in lines[0]


def test_a_game_further_off_than_the_early_call_says_when_it_is_bought():
    got = _prices([("kc2", ["Sam Tight"])], no.empty(), away="DEN", home="KC",
                  kickoff=pd.Timestamp("2026-11-01 17:00"))
    assert got["kc2"]["missing"] == no.TD_PRICE_MISSING["window"]
    assert f"{no.TD_EARLY_DAYS} days" in no.TD_PRICE_MISSING["window"]
    assert f"{no.TD_WINDOW_HOURS} hours" in no.TD_PRICE_MISSING["window"]
    # Inside TD_EARLY_DAYS with nothing bought: none on file, not "too early".
    near = _prices([("kc2", ["Sam Tight"])], no.empty(), away="DEN", home="KC",
                   kickoff=pd.Timestamp("2026-10-30 17:00"))
    assert near["kc2"]["missing"] == no.TD_PRICE_MISSING["none"]


# ---------------------------------------------------------------- the price a TD list shows

def test_a_td_lists_price_is_the_consensus_its_decimal_and_the_implied_chance_with_the_margin_in():
    import nfl_sgp as sgp
    got = _prices([("kc2", ["Sam Tight"]), ("kc1", ["Tom Runner"])], _cache_with_prices())
    view = no.td_price(got["kc2"])
    assert view["american"] == no.consensus_price([150, 140, 155])
    assert view["decimal"] == pytest.approx(sgp.decimal_from_american(view["american"]))
    assert view["implied"] == pytest.approx(1 / view["decimal"])
    assert view["best_american"] == 155 and view["best_book"] == "BetMGM" and view["books"] == 3
    assert view["as_of"] == "2026-10-24T12:00:00Z" and view["match"] == "exact"
    assert view["source"] == "the-odds-api" and view["missing"] is None and view["est"] is True
    assert no.td_price(got["kc1"])["match"] == "normalised"
    # The implied chances of a game's Yes prices sum above what can happen:
    # the books' margin is left in, as TD_ODDS_NOTE says.
    yes, no_ = 1 / sgp.decimal_from_american(-115), 1 / sgp.decimal_from_american(-105)   # FanDuel's pair
    assert yes + no_ > 1 and "margin left in" in no.TD_ODDS_NOTE and "not an edge" in no.TD_ODDS_NOTE


def test_a_td_lists_missing_price_has_every_number_none_and_the_reason():
    for why in ("window", "none", "unposted", "unmatched", "ambiguous"):
        view = no.td_price({"price": None, "missing": no.TD_PRICE_MISSING[why]})
        assert view["missing"] == no.TD_PRICE_MISSING[why] and view["est"] is True
        assert all(view[k] is None for k in ("american", "decimal", "implied", "best_american", "best_book",
                                             "books", "as_of", "match", "source"))
    assert no.td_price(None)["missing"] == no.TD_PRICE_MISSING["none"]
    # A number between -100 and +100 is not American odds: no price, not a wrong one.
    assert no.td_price({"price": 50.0, "best_price": 50.0, "books": 1})["decimal"] is None


# ---------------------------------------------------------------- the monthly spend

def test_the_spend_replay_counts_each_round_by_the_rules():
    """One Sunday game, the runner's week: Tuesday (lines: none on file;
    the game's first early TD call), Saturday (lines four days old; TD in the
    window, Tuesday's price stale), Sunday morning with ODDS_REFRESH=1
    (lines and TD again) and Sunday afternoon (nothing: both fresh)."""
    kickoff = pd.Timestamp("2026-11-29 18:00")
    tue, sat = pd.Timestamp("2026-11-24 10:17"), pd.Timestamp("2026-11-28 13:17")
    sun, late = pd.Timestamp("2026-11-29 06:17"), pd.Timestamp("2026-11-29 15:47")
    spend = no.replay_spend([kickoff], [tue, sat, sun, late], refresh_runs=[sun])
    line = len(no.PARAMS["markets"].split(",")) * len(no.PARAMS["regions"].split(","))
    assert spend == {"2026-11": {"lines": 3 * line, "td_window": 2, "td_early": 1, "total": 3 * line + 3}}
    # A game first seen inside its window is bought there, never early too.
    assert no.replay_spend([kickoff], [sat])["2026-11"]["td_early"] == 0
    # The replay's books never post early, so every retry is counted: with
    # Monday's slot the game is first bought six days out, and asked again
    # at Wednesday's and Friday's - TD_EARLY_RETRY_HOURS apart or more, not
    # Tuesday's or Thursday's - until TD_EARLY_TRIES.
    mon, wed = pd.Timestamp("2026-11-23 15:17"), pd.Timestamp("2026-11-25 06:17")
    thu, fri = pd.Timestamp("2026-11-26 12:17"), pd.Timestamp("2026-11-27 13:17")
    week = [mon, tue, wed, thu, fri, sat]
    assert no.replay_spend([kickoff], week)["2026-11"]["td_early"] == no.TD_EARLY_TRIES == 3
    assert no.replay_spend([kickoff], [mon, tue, sat])["2026-11"]["td_early"] == 1


def test_the_credits_other_engines_spend_are_read_off_the_shared_counter():
    cache = no.empty()
    for at, left, extra in (("2026-10-30T12:00:00", "300", {}),                   # October: 3-credit line call
                            ("2026-11-01T12:00:00", "497", {}),                   # a new month's allowance
                            ("2026-11-03T12:00:00", "493", {}),                   # 3 for the lines, 1 MMA
                            ("2026-11-04T12:00:00", "490", {"market": no.TD_MARKET, "calls": 3}),
                            ("2026-11-05T12:00:00", "485", {})):                  # 3 lines, 2 MMA
        cache["fetches"].append(dict({"at": at + "Z", "status": 200, "events": 1, "credits_remaining": left},
                                     **extra))
    assert no.others_spend(cache) == {"2026-11": 3.0}
