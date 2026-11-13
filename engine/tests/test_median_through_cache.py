"""The typical book's pair must survive the odds cache, end to end.

The card exports odds/odds_other as each side's BEST price across books and,
beside them, odds_median/odds_other_median/books_median (predict_card's
median_price over a matcher record's 'odds' list), which the same-day
underdog ticket (parlays/engine/build_sameday.py) prices its dog at when the
pair comes from 2+ books.

The medians never took effect live: every run serves the card through
odds_cache.for_card, which replaced each bout's live record with one cached
offer a side (the best price parse_events kept), so books_median was always
1. odds_cache now keeps every book's price beside the best one through
parse_events -> merge -> the cache file -> as_current_odds/for_card, and these
tests walk a three-book payload along that whole path, through the card's own
export functions, for a fetched run and for a cache-only run.

The export functions are lifted out of predict_card rather than imported:
importing that module runs the whole pipeline (as test_odds_matching does).
"""

import copy
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import odds_cache as oc

SOURCE = (ENGINE / "predict_card.py").read_text()


def _lift():
    """predict_card's matcher, value and median functions, as the card runs
    them: the exact source, executed in one namespace."""
    namespace = {"re": re, "SequenceMatcher": SequenceMatcher}
    for start, end in (
            ("def american_to_implied_prob(", "# A SURNAME IS NOT AN IDENTITY"),
            ("FIRST_NAME_MIN_RATIO = ", "def calculate_value("),
            ("def calculate_value(", "# Global variable to store fetched odds"),
            ("def median_price(", 'print("BETTABILITY ANALYSIS")')):
        exec(SOURCE[SOURCE.index(start):SOURCE.index(end)], namespace)
    return namespace


PC = _lift()
match = PC["match_fighter_to_odds"]


def _export(current_odds, pick, other):
    """The card's odds fields for one fight, built as predict_card builds them
    (the _app_fights entry, then the median pair written on after
    card_payload)."""
    pred = {"win_prob": 0.6, "win_prob_model": 0.58}
    vi = PC["analyze_fight_value"](pred, pick, current_odds) or {}
    other_vi = PC["analyze_fight_value"](pred, other, current_odds) or {}
    fight = {"odds": vi.get("best_odds"), "odds_other": other_vi.get("best_odds"),
             "book": vi.get("book_with_best"), "book_other": other_vi.get("book_with_best"),
             "odds_seen": vi.get("cached_at"), "odds_other_seen": other_vi.get("cached_at"),
             "edge": vi.get("edge")}
    fight.update(PC["with_median_prices"](
        PC["median_price"](match(pick, current_odds)),
        PC["median_price"](match(other, current_odds))))
    return fight


def _event(home, away, when, books):
    """One feed event; books is [(title, home price, away price)]."""
    return {"home_team": home, "away_team": away, "sport_title": "MMA",
            "commence_time": f"{when}T02:00:00Z",
            "bookmakers": [{"key": title.lower(), "title": title, "markets": [{
                "key": "h2h", "outcomes": [{"name": home, "price": h},
                                           {"name": away, "price": a}]}]}
                for title, h, a in books]}


# The best of each side comes from a different book (FanDuel's Pereira,
# BetMGM's Ankalaev): a pair no single book offers.
THREE_BOOKS = [("DraftKings", -150, 130), ("FanDuel", -140, 125), ("BetMGM", -160, 140)]
CARD = [("Alex Pereira", "Magomed Ankalaev", "2026-10-04")]

# Pereira's decimals 1.6667 / 1.7143 / 1.625 -> 1.6667 = -150; Ankalaev's
# 2.30 / 2.25 / 2.40 -> 2.30 = +130. Best of books is -140 / +140.
THREE_BOOK_MEDIANS = {"odds_median": -150, "odds_other_median": 130,
                      "decimal_median": pytest.approx(1 + 100 / 150),
                      "decimal_other_median": pytest.approx(2.3), "books_median": 3}


def _fetched(now="2026-10-01T00:00:00Z", books=THREE_BOOKS):
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([_event("Alex Pereira", "Magomed Ankalaev",
                                            "2026-10-04", books)]), now=now)
    return cache


# --- the whole path ---------------------------------------------------------

def test_a_three_book_payload_reaches_the_card_as_its_median_pair(tmp_path):
    """parse_events -> merge -> the cache file -> for_card -> the export."""
    cache = _fetched()
    path = oc.save(cache, tmp_path / "odds_cache.json")
    served = oc.for_card(oc.load(path), CARD)
    card = _export(served, "Alex Pereira", "Magomed Ankalaev")
    assert {k: card[k] for k in THREE_BOOK_MEDIANS} == THREE_BOOK_MEDIANS
    # and the best pair, its books and its time, exactly as before
    assert (card["odds"], card["odds_other"]) == (-140, 140)
    assert (card["book"], card["book_other"]) == ("FanDuel", "BetMGM")
    assert card["odds_seen"] == card["odds_other_seen"] == "2026-10-01T00:00:00Z"


def test_a_fetched_run_and_a_cache_only_run_export_the_same_card(tmp_path):
    """A fetched run merges the live payload and then serves the card from the
    cache (for_card over the live dict); a later run with nothing to buy loads
    the file and serves it from there alone. Same fields, same medians."""
    path = tmp_path / "odds_cache.json"
    payload = [_event("Alex Pereira", "Magomed Ankalaev", "2026-10-04", THREE_BOOKS)]

    # the fetched run, as predict_card runs it
    cache = oc.load(path)
    live = {"alex pereira": {"name": "Alex Pereira", "best_odds": -140,
                             "odds": [{"book": t, "odds": h} for t, h, _ in THREE_BOOKS]}}
    oc.merge(cache, oc.parse_events(payload), now="2026-10-01T00:00:00Z")
    fetched = _export(oc.for_card(cache, CARD, live), "Alex Pereira", "Magomed Ankalaev")
    oc.save(cache, path)

    # the next run: no request, the card priced from the file
    cache_only = _export(oc.for_card(oc.load(path), CARD), "Alex Pereira", "Magomed Ankalaev")
    assert cache_only == fetched
    assert cache_only["books_median"] == 3 and cache_only["odds_median"] == -150


def test_the_card_side_of_the_median_pair_follows_the_pick():
    """The pick's side first: picking Ankalaev swaps the pair, not the values."""
    card = _export(oc.for_card(_fetched(), CARD), "Magomed Ankalaev", "Alex Pereira")
    assert (card["odds_median"], card["odds_other_median"]) == (130, -150)
    assert (card["odds"], card["odds_other"]) == (140, -140)


def test_the_cards_own_spelling_is_served_the_same_books():
    """for_card serves a bout under the card's spelling too (here a suffix the
    feed drops); that record must carry the books, not a one-offer copy."""
    cache = oc.empty()
    oc.merge(cache, oc.parse_events([_event("Andre Fili", "Kai Kamaka", "2026-10-11", [
        ("DraftKings", -150, 130), ("FanDuel", -140, 125), ("BetMGM", -160, 140)])]),
        now="2026-10-01T00:00:00Z")
    served = oc.for_card(cache, [("Andre Fili", "Kai Kamaka III", "2026-10-10")])
    card = _export(served, "Andre Fili", "Kai Kamaka III")
    assert card["books_median"] == 3
    assert (card["odds_median"], card["odds_other_median"]) == (-150, 130)


# --- the line a run is priced at, and its books -----------------------------

MOVED = [("DraftKings", -200, 170), ("FanDuel", -190, 160),
         ("BetMGM", -210, 180), ("Caesars", -205, 176)]


def test_a_refreshed_line_is_served_with_its_own_books(tmp_path):
    """The latest fetch's books beside the latest best price, never the first
    fetch's books beside it: those priced another line."""
    cache = _fetched()
    oc.merge(cache, oc.parse_events([_event("Alex Pereira", "Magomed Ankalaev",
                                            "2026-10-04", MOVED)]),
             now="2026-10-03T00:00:00Z")
    path = oc.save(cache, tmp_path / "odds_cache.json")
    card = _export(oc.for_card(oc.load(path), CARD), "Alex Pereira", "Magomed Ankalaev")
    assert (card["odds"], card["odds_other"]) == (-190, 180)
    # Pereira 1.4762 / 1.4878 / 1.5 / 1.5263 -> (1.4878 + 1.5) / 2 = -202;
    # Ankalaev 2.6 / 2.7 / 2.76 / 2.8 -> 2.73 = +173
    assert (card["odds_median"], card["odds_other_median"], card["books_median"]) == (-202, 173, 4)
    # the first line and its books are kept as history
    entry = next(iter(cache["fights"].values()))
    assert entry["fighter_a"] == "Alex Pereira"
    assert (entry["odds_a"], entry["latest_a"]) == (-140, -190)
    assert len(entry["prices_a"]) == 3 and len(entry["latest_prices_a"]) == 4


def test_a_refresh_without_books_never_serves_an_older_fetchs_books():
    """A refresh that carries no per-book prices drops the old latest ones:
    the served side falls back to the best price alone (books 1), which
    build_sameday reads as no typical price."""
    cache = _fetched()
    oc.merge(cache, oc.parse_events([_event("Alex Pereira", "Magomed Ankalaev",
                                            "2026-10-04", MOVED)]),
             now="2026-10-02T00:00:00Z")
    bare = oc.parse_events([_event("Alex Pereira", "Magomed Ankalaev", "2026-10-04",
                                   [("DraftKings", -300, 250)])])
    for entry in bare.values():
        entry.pop("prices_a"), entry.pop("prices_b")
    oc.merge(cache, bare, now="2026-10-03T00:00:00Z")
    card = _export(oc.for_card(cache, CARD), "Alex Pereira", "Magomed Ankalaev")
    assert (card["odds"], card["odds_other"]) == (-300, 250)
    assert (card["odds_median"], card["odds_other_median"], card["books_median"]) == (-300, 250, 1)


def test_a_bout_cached_before_books_were_kept_gains_them_on_its_next_fetch():
    """Every entry on file today has only its best price: served alone, books
    1, until a fetch refreshes it - then its latest line has its books."""
    cache = oc.empty()
    key = oc.fight_key("Alex Pereira", "Magomed Ankalaev", "2026-10-04")
    cache["fights"][key] = {
        "fighter_a": "Alex Pereira", "fighter_b": "Magomed Ankalaev", "date": "2026-10-04",
        "odds_a": -150, "odds_b": 130, "book_a": "DraftKings", "book_b": "DraftKings",
        "first_seen": "2026-09-20T00:00:00Z", "last_seen": "2026-09-20T00:00:00Z",
        "commence_time": "2026-10-04T02:00:00Z"}
    before = _export(oc.for_card(cache, CARD), "Alex Pereira", "Magomed Ankalaev")
    assert (before["odds"], before["book"], before["books_median"]) == (-150, "DraftKings", 1)
    oc.merge(cache, oc.parse_events([_event("Alex Pereira", "Magomed Ankalaev",
                                            "2026-10-04", THREE_BOOKS)]),
             now="2026-10-01T00:00:00Z")
    after = _export(oc.for_card(cache, CARD), "Alex Pereira", "Magomed Ankalaev")
    assert {k: after[k] for k in THREE_BOOK_MEDIANS} == THREE_BOOK_MEDIANS
    assert cache["fights"][key]["odds_a"] == -150      # the first price stays history


# --- nothing else moves -----------------------------------------------------

def _without_books(cache):
    bare = copy.deepcopy(cache)
    for entry in bare["fights"].values():
        for field in ("prices_a", "prices_b", "latest_prices_a", "latest_prices_b"):
            entry.pop(field, None)
    return bare


def _assert_only_the_books_were_added(cache):
    """Every served record is the record the cache served before per-book
    prices were kept (the cache with them stripped), plus the other books
    after the first offer: same best price, same first book, same name,
    opponent and time - so the blend, the pick, the edge and every parlay,
    which read only best_odds and book_with_best, cannot move."""
    with_books = oc.as_current_odds(cache)
    without = oc.as_current_odds(_without_books(cache))
    assert set(with_books) == set(without)
    for name, record in with_books.items():
        assert {**record, "odds": record["odds"][:1]} == without[name]
        best = record["best_odds"]
        book_with_best = next((o["book"] for o in record["odds"] if o["odds"] == best), "Unknown")
        assert book_with_best == without[name]["odds"][0]["book"]
        assert max(o["odds"] for o in record["odds"]) == best


def test_only_the_books_are_added_to_a_served_record():
    cache = _fetched()
    oc.merge(cache, oc.parse_events([
        _event("Alex Pereira", "Magomed Ankalaev", "2026-10-04", MOVED),
        _event("Jon Jones", "Tom Aspinall", "2026-10-04",
               [("DraftKings", 120, -140), ("FanDuel", 125, -150)])]),
        now="2026-10-03T00:00:00Z")
    _assert_only_the_books_were_added(cache)


def test_the_cache_on_file_serves_what_it_served_before():
    """The committed cache, whatever it holds: only books are ever added."""
    if not oc.CACHE.exists():
        pytest.skip("no odds cache on disk")
    _assert_only_the_books_were_added(oc.load())


def test_a_tie_for_the_best_price_keeps_the_book_the_card_always_named():
    """Two books at the best price: _best names the first the feed listed,
    and the served list puts that one first, whatever the alphabet says."""
    cache = _fetched(books=[("Zed Book", -140, 130), ("Alpha Book", -140, 125),
                            ("BetMGM", -160, 120)])
    card = _export(oc.for_card(cache, CARD), "Alex Pereira", "Magomed Ankalaev")
    assert (card["odds"], card["book"]) == (-140, "Zed Book")
    assert card["books_median"] == 3


def test_only_a_named_book_with_a_finite_number_is_kept():
    assert oc._book_prices([("A", "-150"), ("B", True), ("C", float("inf")),
                            ("D", float("nan")), ("E", None), (None, -110), ("", -115),
                            ("F", -120), ("F", -118)]) == {"F": -118}


def test_unusable_book_prices_are_left_out_and_the_cache_still_saves(tmp_path):
    event = _event("Alex Pereira", "Magomed Ankalaev", "2026-10-04", THREE_BOOKS)
    outcomes = lambda h, a: [{"key": "h2h", "outcomes": [
        {"name": "Alex Pereira", "price": h}, {"name": "Magomed Ankalaev", "price": a}]}]
    event["bookmakers"] += [
        {"key": "nan", "title": "NaN Book", "markets": outcomes(float("nan"), None)},
        {"markets": outcomes(-145, 128)}]                      # no name at all
    parsed = oc.parse_events([event])
    entry = next(iter(parsed.values()))
    assert set(entry["prices_a"]) == set(entry["prices_b"]) == {"DraftKings", "FanDuel", "BetMGM"}
    cache = oc.empty()
    oc.merge(cache, parsed, now="2026-10-01T00:00:00Z")
    json.loads(oc.save(cache, tmp_path / "c.json").read_text())   # allow_nan=False held
    card = _export(oc.for_card(cache, CARD), "Alex Pereira", "Magomed Ankalaev")
    assert {k: card[k] for k in THREE_BOOK_MEDIANS} == THREE_BOOK_MEDIANS


def test_books_that_do_not_belong_to_the_best_price_are_not_served():
    """A hand-edited or inconsistent entry (its best book missing from the
    books, or a book above the best) serves the best price alone."""
    cache = _fetched()
    entry = next(iter(cache["fights"].values()))
    entry["prices_a"] = {"Somewhere": -150}
    entry["prices_b"] = {**entry["prices_b"], "Outlier": 900}
    served = oc.as_current_odds(cache)
    assert [len(served[n]["odds"]) for n in ("alex pereira", "magomed ankalaev")] == [1, 1]


# --- the engine wiring ------------------------------------------------------

def test_a_fetched_run_serves_the_card_from_the_cache_it_just_merged():
    """The live records (every book) are replaced by for_card's, so the cached
    books are what a fetched run's medians are taken over - merged first."""
    merged = SOURCE.index("_added, _refreshed = _odds_cache.merge(_cache, _parsed)")
    served = SOURCE.index("CURRENT_ODDS = _odds_cache.for_card(_cache, _card_fights, CURRENT_ODDS)")
    assert merged < served


def test_the_app_reads_no_renamed_field():
    """app/app.js reads the card's best price as f.odds; nothing was renamed."""
    app = (ENGINE.parent / "app" / "app.js").read_text()
    assert "f.odds" in app
    for field in ("'odds': _vi.get('best_odds'),", "'odds_other': _other_vi.get('best_odds'),",
                  "'book': _vi.get('book_with_best'),", "'odds_seen': _vi.get('cached_at'),"):
        assert field in SOURCE
