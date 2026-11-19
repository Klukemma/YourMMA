"""The over/under 2.5 prices on every match entry, the log and every card
that reads the log (locked, kept, kicked off, final).

WHY: the same-day parlays' TICKETS v2 (parlays/engine/sameday.py) price an
over/under 2.5 leg live from prices.avg.o25/u25 - the pre-match average its
history prices it at - so the entry must carry it, None (never NaN) where
football-data.co.uk posts none, and keep it where the league withholds over
2.5 (the withholding is of a chance; a posted price is a fact). The legs are
graded on the total goals, so the log's final must keep both scores.

Runs on the made-up league of test_soccer_predict (its fixture and helpers):
no download, no model."""

import json

import numpy as np
import pandas as pd
import pytest

import soccer_leagues as slg
import soccer_predict as sp
from test_soccer_predict import (AC, FRIDAY, LS, _fixtures, _strict, both, by_id, espn, fdl, posted_ac,  # noqa: F401
                                 run, weekend)

PRICED = {"avg": {"h": 1.8, "d": 3.6, "a": 4.5, "o25": 1.8, "u25": 2.0},
          "best": {"h": 1.9, "d": 3.8, "a": 4.8, "o25": 1.85, "u25": 2.1}}
NONE_POSTED = {"avg": dict.fromkeys(sp.PRICE_KEYS), "best": dict.fromkeys(sp.PRICE_KEYS)}


def _priced(drop=()):
    """football-data.co.uk's fixtures with Arsenal v Chelsea's best over/under
    prices posted too (Leeds v Stoke: no price of any kind)."""
    fx = _fixtures()
    i = fx.index[fx["match_id"] == AC][0]
    fx.loc[i, ["max_o25", "max_u25"]] = [1.85, 2.1]
    return fx[~fx["match_id"].isin(drop)].reset_index(drop=True)


def test_prices_of_is_none_never_nan_where_a_price_is_not_posted():
    row = pd.Series({"avg_h": 2.1, "avg_d": np.nan, "avg_a": 3.9, "avg_o25": 1.95, "avg_u25": float("inf"),
                     "max_h": np.float64(2.2)})
    out = sp.prices_of(row)
    assert out == {"avg": {"h": 2.1, "d": None, "a": 3.9, "o25": 1.95, "u25": None},
                   "best": {"h": 2.2, "d": None, "a": None, "o25": None, "u25": None}}
    assert all(type(v) is float for side in out.values() for v in side.values() if v is not None)
    # A row with no price columns at all (a feed-only match): every price None.
    assert sp.prices_of(pd.Series({"league": "E0"})) == NONE_POSTED
    json.dumps(out, allow_nan=False)


def test_every_entry_carries_the_over_under_prices_and_none_where_not_posted(run):
    m, _, _, log = run(FRIDAY, fixtures=_priced())
    ups = by_id(m)
    assert ups[AC]["prices"] == PRICED
    assert ups[LS]["prices"] == NONE_POSTED and ups[LS]["market"] is None
    # The file itself: strict JSON, the missing prices null.
    raw = _strict((run.dir / "matches.json").read_text())
    assert by_id(raw)[LS]["prices"]["avg"]["o25"] is None
    assert log["picks"][AC]["prices"] == PRICED and log["picks"][LS]["prices"] == NONE_POSTED


def test_the_prices_are_kept_where_the_league_withholds_over_2_5(run, monkeypatch):
    own = slg.withheld
    monkeypatch.setattr(slg, "withheld", lambda code, source: tuple(own(code, source)) + ("o25",))
    m, _, _, log = run(FRIDAY, fixtures=_priced())
    e = by_id(m)[AC]
    # The chance is withheld and said so; the posted price is a fact, kept.
    assert e["market_o25"] is None and e["p_o25"] is None and "o25" in e["scores"]["withheld"]
    assert e["prices"] == PRICED and log["picks"][AC]["prices"] == PRICED


def test_a_kept_card_shows_the_logged_prices(run):
    """football-data.co.uk stops listing a match it had priced while the
    feeds still list it: the card keeps the priced pick, its prices with it."""
    run(FRIDAY, fixtures=_priced(), feed=weekend())
    m, _, _, sat = run("2026-10-31 07:17", fixtures=_priced(drop=(AC,)))
    e = by_id(m)[AC]
    assert e.get("priced_kept") and e["prices"] == PRICED == sat["picks"][AC]["prices"]


def test_a_locked_card_shows_the_logged_prices(run):
    """football-data.co.uk and ESPN have Arsenal v Chelsea at 15:00,
    fixturedownload at 12:00: the pick closes at the earlier listing, and the
    card in between - locked, not kicked off - shows the prices it was logged
    with."""
    feed = [espn("Arsenal", "Chelsea", "2026-10-31 15:00"), fdl("Arsenal", "Chelsea", "2026-10-31 12:00")] + \
        both("Leeds", "Stoke", "2026-11-01 16:30")
    _, _, _, fri = run(FRIDAY, fixtures=_priced(), feed=feed)
    assert fri["picks"][AC]["prices"] == PRICED and fri["picks"][AC]["freeze_at"] == "2026-10-31T12:00:00Z"
    m, _, _, sat = run("2026-10-31 13:00")
    e = by_id(m)[AC]
    assert e["locked"] and e["status"] == "upcoming" and e["prices"] == PRICED == sat["picks"][AC]["prices"]


def test_a_kicked_off_and_a_final_card_show_the_logged_prices_and_the_final_both_scores(run):
    run(FRIDAY, fixtures=_priced())
    m, _, _, _ = run("2026-10-31 15:30")
    started = by_id(m)[AC]
    assert started["status"] == "started" and started["prices"] == PRICED
    m, _, _, after = run("2026-11-02 09:00", results=posted_ac(0, 2))
    final = by_id(m)[AC]
    assert final["status"] == "final" and final["prices"] == PRICED
    # The total goals an over/under leg is graded on: both scores, in the log.
    p = after["picks"][AC]
    assert (p["final"]["home"], p["final"]["away"]) == (0, 2) and final["score"] == {"home": 0, "away": 2}
    assert p["prices"] == PRICED
