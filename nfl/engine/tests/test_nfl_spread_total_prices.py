"""The spread and total PRICES beside their lines: parsed per book, the
consensus at the consensus line, exported by market_for with their source
and, where None, why - from the-odds-api and from the nflverse schedule.

WHY: the same-day parlays' TICKETS v2 SPREAD / TOTAL legs (parlays/engine/
sameday.py) are priced live at "the consensus price at the consensus line",
and history at nflverse's closing home/away_spread_odds and over/under_odds.
A price must be at the line it is shown beside (a price at 3 is another bet
than one at 3.5), from the same source as that line, and missing with its
reason rather than absent or borrowed. The lines, the moneyline and the pick
must not change.

Synthetic payloads only: the API is never called."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import nfl_data as nd
import nfl_features as nf
import nfl_odds as no
import nfl_predict as npd

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
HOME, AWAY = "Kansas City Chiefs", "Las Vegas Raiders"
GAME = {"away": "LV", "home": "KC", "kickoff": pd.Timestamp("2026-10-25 20:25")}
PRICE_KEYS = {f"{m}_price_{x}" for m, sides in no.PRICE_SIDES.items() for x in sides + ("books", "source", "missing")}


def _book(title, h2h=None, spreads=None, totals=None):
    """A bookmaker as the API returns it: h2h (home, away) prices; spreads
    [(home point, home price), (away point, away price)] - either may be
    left out; totals [(over point, price), (under point, price)]."""
    markets = []
    if h2h:
        markets.append({"key": "h2h", "outcomes": [{"name": HOME, "price": h2h[0]}, {"name": AWAY, "price": h2h[1]}]})
    if spreads:
        markets.append({"key": "spreads", "outcomes": [
            {"name": name, "point": pt, "price": pr} for name, side in zip((HOME, AWAY), spreads) if side
            for pt, pr in [side]]})
    if totals:
        markets.append({"key": "totals", "outcomes": [
            {"name": name, "point": pt, "price": pr} for name, side in zip(("Over", "Under"), totals) if side
            for pt, pr in [side]]})
    return {"key": title.lower(), "title": title, "markets": markets}


def _event(books, start="2026-10-25T20:25:00Z"):
    return {"home_team": HOME, "away_team": AWAY, "commence_time": start, "bookmakers": books}


BOOKS = [
    # Both sides at the consensus lines (3.5, 44.5): these two count.
    _book("DraftKings", (-180, 155), [(-3.5, -110), (3.5, -110)], [(44.5, -110), (44.5, -110)]),
    _book("FanDuel", (-175, 150), [(-3.5, -105), (3.5, -115)], [(44.5, -102), (44.5, -118)]),
    # Both sides, but at another line: a price for a different bet.
    _book("BetMGM", (-185, 160), [(-3.0, -120), (3.0, 100)], [(45.5, -110), (45.5, -110)]),
    # The two sides at different points, and one side alone: no pair.
    _book("Caesars", (-180, 150), [(-3.5, -108), (3.0, -112)], [(44.5, -110), None]),
]


def _parsed(books=BOOKS):
    parsed, unknown = no.parse_events([_event(books)])
    assert unknown == []
    (entry,) = parsed.values()
    return entry


def _cached(entry, now=NOW):
    cache = no.empty()
    no.merge(cache, {no.game_key("LV", "KC", entry["commence_time"]): entry}, now=now)
    return cache


# ---------------------------------------------------------------- parse_events

def test_the_prices_are_each_sides_consensus_at_the_consensus_line_from_books_pricing_both_sides():
    e = _parsed()
    # The lines are as before: the lower-middle of every book's line.
    assert (e["spread"], e["total"]) == (3.5, 44.5)
    # The moneyline's own consensus (consensus_price), over the two books
    # that priced both sides at 3.5 / 44.5 - never BetMGM's 3 or 45.5,
    # never Caesars' mismatched or one-sided quotes.
    assert e["spread_price_home"] == no.consensus_price([-110, -105])
    assert e["spread_price_away"] == no.consensus_price([-110, -115])
    assert e["total_price_over"] == no.consensus_price([-110, -102])
    assert e["total_price_under"] == no.consensus_price([-110, -118])
    assert e["spread_price_books"] == 2 and e["total_price_books"] == 2
    assert -110 < e["spread_price_home"] < -105 and -115 < e["spread_price_away"] < -110
    # Nothing else moves: the moneyline is the same consensus as ever.
    assert e["home_ml"] == no.consensus_price([-180, -175, -185, -180]) and e["books"] == 4


def test_one_book_alone_at_the_line_prices_it_with_a_count_of_one():
    e = _parsed([_book("DraftKings", None, [(-7.0, -112), (7.0, -108)], [(41.0, 105), (41.0, -125)])])
    assert (e["spread"], e["spread_price_home"], e["spread_price_away"], e["spread_price_books"]) == (7.0, -112, -108, 1)
    assert (e["total"], e["total_price_over"], e["total_price_under"], e["total_price_books"]) == (41.0, 105, -125, 1)


def test_a_line_no_book_priced_both_sides_of_has_no_price_and_a_count_of_zero():
    # Every book quotes the home side alone, or the two sides at other lines.
    e = _parsed([_book("DraftKings", (-180, 155), [(-3.5, -110), None], [(44.5, -110), None]),
                 _book("FanDuel", (-175, 150), [(-3.5, -105), None], [(44.5, -105), (45.0, -115)])])
    assert e["spread"] == 3.5 and e["total"] == 44.5
    assert (e["spread_price_home"], e["spread_price_away"], e["spread_price_books"]) == (None, None, 0)
    assert (e["total_price_over"], e["total_price_under"], e["total_price_books"]) == (None, None, 0)


def test_no_line_means_no_price_and_no_count():
    e = _parsed([_book("DraftKings", (-180, 155))])
    assert e["spread"] is None and e["total"] is None
    assert (e["spread_price_home"], e["spread_price_books"], e["total_price_over"], e["total_price_books"]) == \
        (None, None, None, None)


def test_a_pair_needs_both_sides_priced_at_the_same_point():
    h = {"name": HOME, "point": -3.5, "price": -110}
    assert no._pair(h, {"name": AWAY, "point": 3.5, "price": -110}, -1) == (-110.0, -110.0, 3.5)
    assert no._pair(h, {"name": AWAY, "point": -3.5, "price": -110}, -1) is None      # the same sign: not a line
    assert no._pair(h, {"name": AWAY, "point": 3.5}, -1) is None                       # unpriced
    assert no._pair(h, None, -1) is None
    over = {"name": "Over", "point": 44.5, "price": -110}
    assert no._pair(over, {"name": "Under", "point": 44.5, "price": -110}, 1) == (-110.0, -110.0, 44.5)
    assert no._pair(over, {"name": "Under", "point": 45.0, "price": -110}, 1) is None


def test_the_cache_keeps_the_prices_and_stays_strict_json(tmp_path):
    cache = _cached(_parsed())
    no.save(cache, tmp_path / "c.json")                      # allow_nan=False
    latest = no.load(tmp_path / "c.json")["games"]["2026-10-25|LV|KC"]["latest"]
    assert latest["spread_price_books"] == 2 and latest["total_price_over"] == no.consensus_price([-110, -102])


# ---------------------------------------------------------------- market_for

def test_market_for_exports_the_api_prices_with_their_source():
    e = _parsed()
    m = no.market_for(_cached(e), GAME, now=NOW)
    assert m["source"] == "the-odds-api" and (m["spread"], m["total"]) == (3.5, 44.5)
    for k in ("spread_price_home", "spread_price_away", "total_price_over", "total_price_under",
              "spread_price_books", "total_price_books"):
        assert m[k] == e[k]
    assert m["spread_price_source"] == m["total_price_source"] == "the-odds-api"
    assert m["spread_price_missing"] is None and m["total_price_missing"] is None
    json.dumps(m, allow_nan=False)


def test_market_for_says_why_an_api_price_is_missing():
    one_sided = _parsed([_book("DraftKings", (-180, 155), [(-3.5, -110), None], [(44.5, -110), None])])
    m = no.market_for(_cached(one_sided), GAME, now=NOW)
    assert m["spread"] == 3.5 and m["spread_price_home"] is None and m["spread_price_away"] is None
    assert m["spread_price_missing"] == "no book priced both sides at the line" == no.PRICE_MISSING["unpaired"]
    assert m["spread_price_books"] == 0 and m["spread_price_source"] is None
    no_lines = _parsed([_book("DraftKings", (-180, 155))])
    m = no.market_for(_cached(no_lines), GAME, now=NOW)
    assert m["spread_price_missing"] == m["total_price_missing"] == "no line"
    assert m["spread_price_books"] is None and m["home_ml"] is not None


def test_a_price_cached_before_prices_were_kept_is_said_so_and_never_borrowed_from_the_schedule():
    old = {k: v for k, v in _parsed().items() if k not in PRICE_KEYS}
    game = dict(GAME, home_ml=-180.0, away_ml=155.0, spread_line=3.5, total_line=44.5,
                home_spread_odds=-110.0, away_spread_odds=-110.0, over_odds=-110.0, under_odds=-110.0)
    m = no.market_for(_cached(old), game, now=NOW)
    # The line is the-odds-api's, so its price must be too: the schedule's
    # pair sits right there and is not taken.
    assert m["source"] == "the-odds-api" and m["spread"] == 3.5
    assert m["spread_price_home"] is None and m["total_price_over"] is None
    assert m["spread_price_missing"] == m["total_price_missing"] == "fetched before prices were kept"
    assert m["spread_price_books"] is None and m["spread_price_source"] is None


def test_the_schedule_fallback_reads_its_own_posted_prices():
    game = dict(GAME, home_ml=-180.0, away_ml=155.0, spread_line=3.5, total_line=44.5,
                home_spread_odds=-105.0, away_spread_odds=-115.0, over_odds=-110.0, under_odds=np.nan)
    m = no.market_for(no.empty(), game, now=NOW)
    assert m["source"] == "nflverse" and m["spread"] == 3.5
    assert (m["spread_price_home"], m["spread_price_away"]) == (-105.0, -115.0)
    assert m["spread_price_source"] == "nflverse" and m["spread_price_missing"] is None
    assert m["spread_price_books"] is None                     # the schedule does not say how many books
    # One side alone is no pair: neither is shown.
    assert m["total_price_over"] is None and m["total_price_under"] is None
    assert m["total_price_missing"] == "the schedule posts one side only" and m["total_price_source"] is None
    json.dumps(m, allow_nan=False)


def test_the_schedule_fallback_says_why_it_has_no_price():
    none_posted = dict(GAME, home_ml=-180.0, away_ml=155.0, spread_line=3.5, total_line=np.nan,
                       home_spread_odds=np.nan, away_spread_odds=np.nan, over_odds=-110.0, under_odds=-110.0)
    m = no.market_for(no.empty(), none_posted, now=NOW)
    assert m["spread_price_missing"] == "the schedule posts none" and m["spread_price_home"] is None
    # Prices with no line beside them are not a price for anything shown.
    assert m["total"] is None and m["total_price_missing"] == "no line" and m["total_price_over"] is None
    # A schedule row read before load_games kept the columns: none posted.
    m = no.market_for(no.empty(), dict(GAME, home_ml=-180.0, away_ml=155.0, spread_line=3.5, total_line=44.5))
    assert m["spread_price_missing"] == m["total_price_missing"] == "the schedule posts none"


def test_the_prices_change_none_of_the_markets_other_fields():
    """What the pick reads (home_prob, spread, total, source) is what it was."""
    e = _parsed()
    m = no.market_for(_cached(e), GAME, now=NOW)
    assert set(m) - PRICE_KEYS == {"source", "as_of", "books", "home_prob", "home_ml", "away_ml", "best_home_ml",
                                   "best_home_book", "best_away_ml", "best_away_book", "spread", "total",
                                   "opening_spread", "opening_total"}
    game = dict(GAME, home_ml=-400.0, away_ml=320.0, spread_line=9.5, total_line=44.5,
                home_spread_odds=-110.0, away_spread_odds=-110.0, over_odds=-110.0, under_odds=-110.0)
    fb = no.market_for(no.empty(), game)
    assert {k: fb[k] for k in set(fb) - PRICE_KEYS} == {
        "source": "nflverse", "as_of": None, "books": None, "home_prob": no._no_vig(-400.0, 320.0),
        "home_ml": -400.0, "away_ml": 320.0, "spread": 9.5, "total": 44.5}


def test_a_logged_market_from_before_shows_the_prices_missing_and_the_log_is_untouched():
    logged = {"source": "nflverse", "spread": 3.5, "total": None, "home_prob": 0.6}
    shown = no.with_price_fields(logged)
    assert shown["spread_price_missing"] == "fetched before prices were kept" and shown["spread_price_home"] is None
    assert shown["total_price_missing"] == "no line"
    assert "spread_price_missing" not in logged                 # a copy: the log is never changed
    has = no.market_for(no.empty(), dict(GAME, home_ml=-180.0, away_ml=155.0, spread_line=3.5, total_line=44.5,
                                         home_spread_odds=-110.0, away_spread_odds=-110.0))
    assert no.with_price_fields(has) is has and no.with_price_fields(None) is None


# ---------------------------------------------------------------- the schedule

def test_load_games_keeps_the_four_price_columns_and_no_model_reads_them(tmp_path):
    csv = ("game_id,season,week,game_type,gameday,gametime,home_team,away_team,home_score,away_score,div_game,"
           "location,spread_line,total_line,home_moneyline,away_moneyline,home_spread_odds,away_spread_odds,"
           "over_odds,under_odds\n"
           "2026_11_CIN_WAS,2026,11,REG,2026-11-22,13:00,WAS,CIN,,,0,Home,-3.5,51.5,150,-175,-105,-115,-110,-110\n"
           "2026_11_LV_KC,2026,11,REG,2026-11-22,16:25,KC,LV,,,1,Home,7.0,44.5,-300,240,,,-108,\n")
    path = tmp_path / "games.csv"
    path.write_text(csv)
    g = nd.load_games(path)
    assert {"home_spread_odds", "away_spread_odds", "over_odds", "under_odds"} <= set(g.columns)
    was = g[g["home"] == "WAS"].iloc[0]
    m = no.market_for(no.empty(), was)
    assert (m["spread"], m["spread_price_home"], m["spread_price_away"]) == (-3.5, -105.0, -115.0)
    assert (m["total"], m["total_price_over"], m["total_price_under"]) == (51.5, -110.0, -110.0)
    kc = no.market_for(no.empty(), g[g["home"] == "KC"].iloc[0])
    assert kc["spread_price_missing"] == "the schedule posts none"
    assert kc["total_price_missing"] == "the schedule posts one side only"
    # Not a model input.
    assert not {"home_spread_odds", "away_spread_odds", "over_odds", "under_odds"} & \
        set(nf.FEATURES + nf.TOTAL_FEATURES)


def test_the_snapshot_schedule_prices_reach_the_fallback(games):
    priced = games[games["spread_line"].notna() & games["home_spread_odds"].notna()
                   & games["away_spread_odds"].notna()]
    assert len(priced) > 1000
    row = priced.iloc[-1]
    m = no.market_for(no.empty(), row)
    assert m["source"] == "nflverse" and m["spread"] == row["spread_line"]
    assert (m["spread_price_home"], m["spread_price_away"]) == (row["home_spread_odds"], row["away_spread_odds"])


# ---------------------------------------------------------------- week.json

def _events_for(slate):
    """A synthetic reply pricing every other upcoming game of the slate:
    two books on both sides of 2.5 / 47.5, one on 3 / 48.5."""
    out = []
    for g in slate.iloc[::2].itertuples(index=False):
        home, away = nd.TEAMS[g.home][0], nd.TEAMS[g.away][0]
        books = []
        for title, sp, to in (("DraftKings", ((-2.5, -110), (2.5, -110)), ((47.5, -105), (47.5, -115))),
                              ("FanDuel", ((-2.5, -112), (2.5, -108)), ((47.5, -110), (47.5, -110))),
                              ("BetMGM", ((-3.0, -105), (3.0, -115)), ((48.5, -110), (48.5, -110)))):
            books.append({"key": title.lower(), "title": title, "markets": [
                {"key": "h2h", "outcomes": [{"name": home, "price": -140}, {"name": away, "price": 120}]},
                {"key": "spreads", "outcomes": [{"name": home, "point": sp[0][0], "price": sp[0][1]},
                                                {"name": away, "point": sp[1][0], "price": sp[1][1]}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": to[0][0], "price": to[0][1]},
                                               {"name": "Under", "point": to[1][0], "price": to[1][1]}]}]})
        out.append({"home_team": home, "away_team": away, "commence_time": f"{g.kickoff:%Y-%m-%dT%H:%M:%SZ}",
                    "bookmakers": books})
    return out


@pytest.fixture(scope="module")
def week_run(tmp_path_factory, games):
    """One real run on the snapshot (as test_nfl_predict's), with a cache
    holding a synthetic reply for half the week's games: no key, no call."""
    out = tmp_path_factory.mktemp("app")
    now = games[games["played"]]["kickoff"].iloc[-1] + pd.Timedelta(hours=6)
    season, week, _ = npd.current_week(games, now)
    slate = games[(games["season"] == season) & (games["week"] == week) & (games["kickoff"] > now)]
    cache = no.empty()
    parsed, unknown = no.parse_events(_events_for(slate))
    assert not unknown and parsed
    no.merge(cache, parsed, now=now.tz_localize("UTC").to_pydatetime() - timedelta(hours=1))
    no.save(cache, out / "odds_cache.json")
    npd.run(sync=False, now=now, out_dir=out, log_path=out / "log.json", log=lambda *_: None,
            cache_path=out / "odds_cache.json", api_key="")
    return json.loads((out / "week.json").read_text()), json.loads((out / "log.json").read_text()), set(parsed)


def test_week_json_carries_the_prices_beside_every_line_with_source_or_reason(week_run):
    week, log, priced = week_run
    upcoming = [g for g in week["games"] if g["status"] == "upcoming" and g.get("market")]
    assert upcoming
    api = [g for g in upcoming if g["market"]["source"] == "the-odds-api"]
    assert api and len(api) == len(priced)
    for g in api:
        m = g["market"]
        assert (m["spread"], m["total"]) == (2.5, 47.5)
        assert m["spread_price_home"] == no.consensus_price([-110, -112]) and m["spread_price_books"] == 2
        assert m["total_price_under"] == no.consensus_price([-115, -110]) and m["total_price_books"] == 2
        assert m["spread_price_source"] == m["total_price_source"] == "the-odds-api"
    for g in upcoming:
        m = g["market"]
        assert PRICE_KEYS <= set(m)
        for market, (a, b) in no.PRICE_SIDES.items():
            if m[f"{market}_price_missing"] is None:
                assert m[f"{market}_price_{a}"] is not None and m[f"{market}_price_{b}"] is not None
                assert m[f"{market}_price_source"] == m["source"]          # the line's own source
            else:
                assert m[f"{market}_price_missing"] in no.PRICE_MISSING.values()
                assert m[f"{market}_price_{a}"] is None and m[f"{market}_price_source"] is None
        # The logged market is the one shown, prices and all.
        assert log["picks"][g["game_id"]]["market"] == m
