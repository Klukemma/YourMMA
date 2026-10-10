"""Tests for the odds cache.

The thing being protected is a free-tier quota: 500 credits a month, one per
call. Every test here is really about the same question - did we make a
request we did not have to?
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import odds_cache as oc


def event(home, away, when, prices, book="DK"):
    return {"home_team": home, "away_team": away,
            "commence_time": f"{when}T02:00:00Z",
            "bookmakers": [{"title": book, "markets": [{"key": "h2h",
                "outcomes": [{"name": home, "price": prices[0]},
                             {"name": away, "price": prices[1]}]}]}]}


# --- the key ---------------------------------------------------------------

def test_corner_order_does_not_create_two_entries():
    """The API's home/away is arbitrary for MMA and does not match the
    dataset's red/blue. Storing both orders would double every fight and make
    the "already priced" test useless."""
    assert (oc.fight_key("Alex Pereira", "Jon Jones", "2026-10-04")
            == oc.fight_key("Jon Jones", "Alex Pereira", "2026-10-04"))


def test_the_same_pair_on_a_different_date_is_a_different_fight():
    """Rematches are common and each has its own price."""
    assert (oc.fight_key("A B", "C D", "2026-10-04")
            != oc.fight_key("A B", "C D", "2027-02-01"))


# --- parsing ---------------------------------------------------------------

def test_the_best_price_on_each_side_is_kept_independently():
    parsed = oc.parse_events([{
        "home_team": "A", "away_team": "B",
        "commence_time": "2026-10-04T02:00:00Z",
        "bookmakers": [
            {"title": "DK", "markets": [{"key": "h2h", "outcomes": [
                {"name": "A", "price": -150}, {"name": "B", "price": 130}]}]},
            {"title": "FD", "markets": [{"key": "h2h", "outcomes": [
                {"name": "A", "price": -140}, {"name": "B", "price": 125}]}]},
        ]}])
    entry = next(iter(parsed.values()))
    assert entry["odds_a"] == -140 and entry["book_a"] == "FD"
    assert entry["odds_b"] == 130 and entry["book_b"] == "DK"


def test_an_event_with_no_usable_price_is_not_stored_at_all():
    """A fight recorded with no price would satisfy "already have it" and stop
    us ever asking again - the one failure that costs nothing and yields
    nothing forever."""
    parsed = oc.parse_events([{
        "home_team": "A", "away_team": "B",
        "commence_time": "2026-10-04T02:00:00Z",
        "bookmakers": [{"title": "DK", "markets": [{"key": "totals",
                                                    "outcomes": []}]}]}])
    assert parsed == {}


def test_a_malformed_event_is_skipped_rather_than_crashing_the_run():
    assert oc.parse_events([{}, {"home_team": "A"}, None and {}]) == {}
    assert oc.parse_events(None) == {}


# --- the decision, which is the whole point --------------------------------

CARD = [("Alex Pereira", "Magomed Ankalaev", "2026-10-04"),
        ("Jon Jones", "Tom Aspinall", "2026-10-04")]


def _primed(now="2026-10-01T00:00:00Z"):
    cache = oc.empty()
    parsed = oc.parse_events([
        event("Alex Pereira", "Magomed Ankalaev", "2026-10-04", (-150, 130)),
        event("Jon Jones", "Tom Aspinall", "2026-10-04", (120, -140))])
    oc.merge(cache, parsed, now=now)
    return cache


def test_a_fully_priced_card_makes_no_request():
    """The behaviour being paid for."""
    should, why = oc.decide(CARD, _primed(), now="2026-10-02T00:00:00Z")
    assert should is False
    assert "already priced" in why


def test_one_unpriced_fight_justifies_the_call():
    """There is no per-fight request - one call returns the whole card - so a
    single gap is worth the credit."""
    cache = _primed()
    card = CARD + [("New Guy", "Other Guy", "2026-10-04")]
    should, why = oc.decide(card, cache)
    assert should is True
    assert "1 of 3" in why


def test_refresh_overrides_a_fully_priced_card():
    """For the day of the card, when the line that matters is the current one."""
    should, why = oc.decide(CARD, _primed(), refresh=True)
    assert should is True
    assert "refresh" in why


def test_an_empty_card_never_spends_a_credit():
    should, why = oc.decide([], oc.empty())
    assert should is False


def test_a_stale_price_is_reported_but_still_does_not_spend():
    """Saving the credit is the instruction; the age is surfaced so the choice
    is visible rather than silent."""
    cache = _primed(now="2026-09-01T00:00:00Z")
    should, why = oc.decide(CARD, cache, now="2026-10-01T00:00:00Z")
    assert should is False
    assert "refresh" in why and "days ago" in why


# --- merging ---------------------------------------------------------------

def test_a_known_fight_keeps_the_price_the_prediction_was_made_against():
    """Overwriting would replace the line a pick was judged at with a later
    one, and the backtest would score bets at prices never available."""
    cache = _primed()
    key = oc.fight_key("Alex Pereira", "Magomed Ankalaev", "2026-10-04")
    moved = oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                   "2026-10-04", (-260, 210))])
    added, refreshed = oc.merge(cache, moved, now="2026-10-03T00:00:00Z")
    assert (added, refreshed) == (0, 1)
    assert cache["fights"][key]["odds_a"] == -150      # unchanged
    assert cache["fights"][key]["latest_a"] == -260    # recorded separately
    assert cache["fights"][key]["last_seen"] == "2026-10-03T00:00:00Z"


def test_merging_the_same_response_twice_adds_nothing():
    cache = _primed()
    before = len(cache["fights"])
    parsed = oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                    "2026-10-04", (-150, 130))])
    added, _ = oc.merge(cache, parsed)
    assert added == 0 and len(cache["fights"]) == before


# --- settled fights become history -----------------------------------------

def test_a_finished_fight_is_exported_for_the_backtest():
    """Its closing line can never change, and odds.csv has no 2025 prices at
    all - every card watched from here fills that gap permanently."""
    rows = oc.settled_rows(_primed(), before="2026-11-01")
    assert len(rows) == 2
    assert set(rows[0]) == {"date", "fighter_a", "fighter_b",
                            "odds_a", "odds_b"}


def test_a_fight_that_has_not_happened_is_not_exported_as_history():
    assert oc.settled_rows(_primed(), before="2026-10-01") == []


def test_a_card_commencing_today_is_not_settled_today():
    """A day's grace: an event starting tonight has not finished tonight."""
    assert oc.settled_rows(_primed(), before="2026-10-04") == []
    assert len(oc.settled_rows(_primed(), before="2026-10-06")) == 2


def test_the_exported_rows_match_the_schema_roi_reads():
    """A row shaped wrongly is silently dropped by load_odds."""
    import pandas as pd
    from roi import load_odds
    rows = oc.settled_rows(_primed(), before="2026-11-01")
    frame = pd.DataFrame(rows)
    path = ENGINE / "data" / "_test_odds_roundtrip.csv"
    try:
        frame.to_csv(path, index=False)
        index = load_odds(path)
        assert index, "roi could not read the exported rows"
    finally:
        path.unlink(missing_ok=True)


# --- persistence -----------------------------------------------------------

def test_the_cache_survives_a_round_trip(tmp_path):
    cache = _primed()
    path = oc.save(cache, tmp_path / "odds_cache.json")
    assert oc.load(path)["fights"] == cache["fights"]


def test_an_unreadable_cache_raises_rather_than_looking_empty(tmp_path):
    """Treating a broken cache as empty spends a credit and looks like normal
    operation - exactly the silence this module exists to remove."""
    path = tmp_path / "odds_cache.json"
    path.write_text(json.dumps({"schema": 99, "fights": {}}))
    with pytest.raises(ValueError, match="schema"):
        oc.load(path)


def test_a_missing_cache_is_simply_empty(tmp_path):
    assert oc.load(tmp_path / "nothing.json")["fights"] == {}


def test_every_call_is_logged_so_the_spend_is_visible():
    cache = oc.empty()
    oc.record_fetch(cache, 200, 12, "487", now="2026-10-01T00:00:00Z")
    assert cache["fetches"][-1]["credits_remaining"] == "487"


# --- serving the card from the cache ---------------------------------------

def test_the_cache_serves_the_shape_the_matcher_expects():
    """A fight priced last week must still be priced this week, without a call."""
    served = oc.as_current_odds(_primed())
    assert "alex pereira" in served
    entry = served["alex pereira"]
    assert entry["best_odds"] == -150
    assert entry["opponent"] == "Magomed Ankalaev"
    assert entry["name"] == "Alex Pereira"


def test_both_sides_of_a_fight_are_served():
    """The matcher looks up whichever fighter the model picked."""
    served = oc.as_current_odds(_primed())
    assert served["magomed ankalaev"]["best_odds"] == 130
    assert served["magomed ankalaev"]["opponent"] == "Alex Pereira"


def test_a_live_response_does_not_lose_the_cached_fights():
    live = {"someone else": {"name": "Someone Else", "best_odds": 100}}
    served = oc.as_current_odds(_primed(), live)
    assert "someone else" in served and "alex pereira" in served


# --- graduating into history -----------------------------------------------

def test_flushing_twice_does_not_duplicate_a_row(tmp_path):
    """Duplicated rows would inflate the backtest's sample and let one fight
    be counted as several bets."""
    csv = tmp_path / "odds.csv"
    cache = _primed()
    first = oc.flush_settled(cache, csv, before="2026-11-01")
    second = oc.flush_settled(cache, csv, before="2026-11-01")
    assert (first, second) == (2, 0)
    import pandas as pd
    assert len(pd.read_csv(csv)) == 2


def test_flushing_appends_rather_than_replacing_existing_history(tmp_path):
    """odds.csv already holds 6,565 historical prices; losing them to a flush
    would silently shrink every backtest."""
    import pandas as pd
    csv = tmp_path / "odds.csv"
    pd.DataFrame([{"date": "2010-03-21", "fighter_a": "Eric Schafer",
                   "fighter_b": "Jason Brilz", "odds_a": 140.0,
                   "odds_b": -160.0}]).to_csv(csv, index=False)
    added = oc.flush_settled(_primed(), csv, before="2026-11-01")
    frame = pd.read_csv(csv)
    assert added == 2 and len(frame) == 3
    assert "Eric Schafer" in set(frame["fighter_a"])


def test_nothing_settled_writes_nothing(tmp_path):
    csv = tmp_path / "odds.csv"
    assert oc.flush_settled(_primed(), csv, before="2026-10-01") == 0
    assert not csv.exists()


# --- the line a run is priced at -------------------------------------------

def _moved():
    """The primed Pereira fight, re-fetched later at a moved line."""
    cache = _primed()
    oc.merge(cache, oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                           "2026-10-04", (-260, 210))]),
             now="2026-10-03T00:00:00Z")
    return cache


def test_a_refreshed_line_is_the_one_the_card_is_priced_at():
    """A fight-eve refresh bought today's line and the card kept serving the
    first one: the credit was spent and the app showed a five-day-old price."""
    served = oc.as_current_odds(_moved())
    assert served["alex pereira"]["best_odds"] == -260
    assert served["magomed ankalaev"]["best_odds"] == 210
    assert served["alex pereira"]["cached_at"] == "2026-10-03T00:00:00Z"


def test_the_first_seen_price_is_kept_as_history():
    cache = _moved()
    key = oc.fight_key("Alex Pereira", "Magomed Ankalaev", "2026-10-04")
    assert cache["fights"][key]["odds_a"] == -150
    assert cache["fights"][key]["first_seen"] != cache["fights"][key]["latest_seen"]


def test_a_line_that_moves_back_is_recorded_as_moved_back():
    """Recording latest only when it differs from the FIRST price leaves a
    moved-away line as 'latest' after the market returns."""
    cache = _moved()
    oc.merge(cache, oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                           "2026-10-04", (-150, 130))]),
             now="2026-10-04T00:00:00Z")
    assert oc.as_current_odds(cache)["alex pereira"]["best_odds"] == -150


def test_a_settled_fight_is_scored_at_the_last_line_seen():
    """The graded pick is the newest one logged, made at the newest line."""
    rows = oc.settled_rows(_moved(), before="2026-11-01")
    row = next(r for r in rows if "Pereira" in (r["fighter_a"] + r["fighter_b"]))
    a_is_pereira = row["fighter_a"] == "Alex Pereira"
    assert (row["odds_a"], row["odds_b"]) == ((-260, 210) if a_is_pereira else (210, -260))


# --- finding the card's fights in the cache ---------------------------------

def test_a_us_evening_card_filed_under_the_next_utc_day_is_found():
    """The feed dates a fight by its UTC start; the card by the local date.
    Every predict run for an evening card bought a credit for prices it had."""
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([event("Brendan Allen", "Christian Leroy Duncan",
                                           "2026-10-11", (-133, 120))]))
    card = [("Brendan Allen", "Christian Leroy Duncan", "2026-10-10")]
    assert oc.missing(card, cache) == []
    should, why = oc.decide(card, cache)
    assert should is False, why


def test_a_rematch_two_days_away_is_a_different_fight():
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([event("Brendan Allen", "Christian Leroy Duncan",
                                           "2026-10-12", (-133, 120))]))
    assert oc.missing([("Brendan Allen", "Christian Leroy Duncan", "2026-10-10")], cache)


def test_a_generational_suffix_does_not_hide_a_priced_fight():
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([event("Andre Fili", "Kai Kamaka",
                                           "2026-10-11", (-150, 132))]))
    assert oc.missing([("Andre Fili", "Kai Kamaka III", "2026-10-10")], cache) == []


def test_staleness_is_measured_from_the_line_a_run_would_serve():
    cache = _moved()
    entry = cache["fights"][oc.fight_key("Alex Pereira", "Magomed Ankalaev", "2026-10-04")]
    assert oc.age_days(entry, now="2026-10-04T00:00:00Z") == pytest.approx(1.0)


def test_a_bout_re_dated_across_utc_midnight_refreshes_its_own_entry(tmp_path):
    """A second key for the same bout loads first after a save (sort_keys) and
    a later run served the stale line from it."""
    cache = _primed()
    moved = oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                   "2026-10-03", (-260, 210))])
    added, refreshed = oc.merge(cache, moved, now="2026-10-02T00:00:00Z")
    assert (added, refreshed) == (0, 1)
    path = oc.save(cache, tmp_path / "cache.json")
    served = oc.as_current_odds(oc.load(path))
    assert served["alex pereira"]["best_odds"] == -260


def test_a_fight_already_under_way_is_never_priced():
    """An in-play line is not a pre-fight price: served, blended and
    exported as the close, it would score the card at a knockdown's odds."""
    cache = _primed()
    live = oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                  "2026-10-04", (-2500, 1200)),
                            event("New Fighter", "Other Fighter",
                                  "2026-10-04", (-110, -110))])
    added, refreshed = oc.merge(cache, live, now="2026-10-04T03:00:00Z")
    assert (added, refreshed) == (0, 0)
    assert oc.as_current_odds(cache)["alex pereira"]["best_odds"] == -150


def _two_fights_for_one_fighter():
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([
        event("Ilia Topuria", "Justin Gaethje", "2026-12-31", (-250, 210)),
        event("Arman Tsarukyan", "Justin Gaethje", "2028-07-30", (-450, 350))]),
        now="2026-10-01T00:00:00Z")
    return cache


def test_a_card_bout_is_priced_from_its_own_entry():
    """Keyed by name alone, Gaethje was served his 2028 line against
    Tsarukyan on a card where he fights Topuria."""
    served = oc.for_card(_two_fights_for_one_fighter(),
                         [("Ilia Topuria", "Justin Gaethje", "2026-12-31")])
    assert served["justin gaethje"]["best_odds"] == 210
    assert served["justin gaethje"]["opponent"] == "Ilia Topuria"


def test_a_card_fighter_whose_bout_is_not_on_file_gets_no_price():
    served = oc.for_card(_two_fights_for_one_fighter(),
                         [("Justin Gaethje", "Someone New", "2027-03-01")],
                         existing={"justin gaethje": {"best_odds": 999}})
    assert "justin gaethje" not in served



def test_an_unreadable_start_time_does_not_crash_the_fetch():
    cache = _primed()
    parsed = oc.parse_events([event("Alex Pereira", "Magomed Ankalaev",
                                    "2026-10-04", (-200, 170))])
    for entry in parsed.values():
        entry["commence_time"] = "not a time"
    assert oc.merge(cache, parsed, now="2026-10-02T00:00:00Z") == (0, 1)



def test_a_missing_bout_reaches_none_of_its_fighters_other_fights():
    """Popping only the exact feed name left 'kai kamaka' from another bout
    for the matcher to find under 'Kai Kamaka III'."""
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([event("Kai Kamaka", "Someone Else",
                                           "2027-02-01", (-120, 100))]),
             now="2026-10-01T00:00:00Z")
    served = oc.for_card(cache, [("Andre Fili", "Kai Kamaka III", "2026-10-10")])
    assert "kai kamaka" not in served and "someone else" in served
