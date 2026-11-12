"""The underdog ticket's measurement (experiments/sameday_backtest.py) and its
inputs (legs_history.py's legs_dogs_* files): what is published reproduces
from the files by sameday's own functions, the dogs are the market's, and
the json says nothing it cannot back.

WHY these tests: the record on the page is only worth what its rebuild is.
The likeliest ticket's numbers must still come out of the legs files after
the soccer legs were rebuilt; a dog row must sit on the same event, key and
day as the pick it opposes (the fill leaves the dog's event out by key); a
dog is never a draw and always the side the market makes less likely; the
live salt's tickets must be what build_dog gives live; and a NaN in the json
would break the page."""

import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "experiments"))

import sameday as sdy  # noqa: E402
import sameday_backtest as sb  # noqa: E402

DATA = ENGINE.parents[0] / "data"
SPORTS = {"ufc": "UFC", "nfl": "NFL", "soccer": "Soccer"}


@pytest.fixture(scope="module")
def published():
    return json.loads((DATA / "sameday_backtest.json").read_text())


@pytest.fixture(scope="module")
def files():
    out = {}
    for name in SPORTS:
        out[name] = (pd.read_csv(DATA / f"legs_{name}.csv.gz"), pd.read_csv(DATA / f"legs_dogs_{name}.csv.gz"))
    return out


@pytest.fixture(scope="module")
def days():
    """Every day's legs and underdog candidates, as the backtest reads them."""
    legs, end, _ = sb.load()
    dogs = sb.load_dogs(end)
    L = {d: [l for l in (sb.to_leg(r) for r in g.itertuples()) if l] for d, g in legs.groupby("day")}
    D = {d: [x for x in (sb.to_dog(r) for r in g.itertuples()) if x] for d, g in dogs.groupby("day")}
    return L, D


def test_the_likeliest_ticket_reproduces_from_the_legs_files_to_the_published_numbers(published, days):
    L, _ = days
    tickets = []
    for day in sorted(L):
        t = sdy.build(L[day], "p")
        if t is not None:
            t["day"] = day
            tickets.append(t)
    per = published["periods"]
    for name, (lo, hi) in per.items():
        mine = sdy.summarise([t for t in tickets if lo <= t["day"] <= hi])
        assert mine == published["results"]["same_day"][name], name


def test_every_dog_row_sits_on_its_pick_sides_event_key_and_day(files):
    for name, (picks, dogs) in files.items():
        assert not dogs["key"].duplicated().any()
        m = dogs.merge(picks, on="key", how="left", suffixes=("", "_pick"), indicator=True)
        assert (m["_merge"] == "both").all(), name
        assert (m["day"] == m["day_pick"]).all() and (m["event"] == m["event_pick"]).all(), name
        # an event with no dog row is one the market makes level: equal prices
        level = picks[~picks["key"].isin(dogs["key"])]
        if name in ("ufc", "nfl"):
            assert (level["p_market"] == 0.5).all(), name
        assert len(level) < 0.03 * len(picks), name           # UFC 86 of 4,387, NFL 13, soccer 20


def test_the_lot_key_is_sameday_dog_lot_key(files):
    for name, (_, dogs) in files.items():
        for r in dogs.sample(300, random_state=1).itertuples():
            if name == "nfl":
                want = sdy.dog_lot_key("NFL", game_id=r.key)
            elif name == "soccer":
                want = sdy.dog_lot_key("Soccer", match_id=r.key)
            else:
                want = sdy.dog_lot_key("UFC", corners=[r.pick, r.opponent])
            assert r.lot_key == want, (name, r.key)


def test_no_dog_is_a_draw_and_every_dog_is_one_of_the_two_teams(files):
    _, dogs = files["soccer"]
    assert not (dogs["pick"] == "Draw").any()
    teams = dogs["event"].str.split(" v ", n=1, expand=True)
    assert ((dogs["pick"] == teams[0]) | (dogs["pick"] == teams[1])).all()
    assert ((dogs["opponent"] == teams[0]) | (dogs["opponent"] == teams[1])).all()


def test_every_dog_is_the_markets_underdog_at_the_same_snapshot(files):
    for name, (picks, dogs) in files.items():
        assert (dogs["p_market"] < 0.5).all(), name
        assert dogs["p"].between(0, 1).all() and (dogs["decimal"] > 1).all(), name
        if name == "soccer":
            continue
        # two sides: where the dog is not the pick it is the other side of the same price pair
        m = dogs.merge(picks, on="key", suffixes=("", "_pick"))
        other = m[m["pick"] != m["pick_pick"]]
        assert (other["opponent"] == other["pick_pick"]).all(), name
        assert ((other["p_market"] + other["p_market_pick"] - 1).abs() < 2e-6).all(), name
        assert ((other["p"] + other["p_pick"] - 1).abs() < 2e-6).all(), name
        # and the overround read from either side agrees: one snapshot
        o_dog = 1 / (other["decimal"] * other["p_market"])
        o_pick = 1 / (other["decimal_pick"] * other["p_market_pick"])
        assert ((o_dog - o_pick).abs() < 1e-4).all(), name
        same = m[m["pick"] == m["pick_pick"]]
        assert (same["p_market"] < 0.5).all() and (same["p"] >= 0.5).all(), name   # the blend picked the dog


def test_the_live_salt_dog_tickets_reproduce_from_build_dog(days):
    L, D = days
    t = pd.read_csv(DATA / "sameday_dog_tickets.csv.gz")
    live = t[t["rule"] == "dog"].set_index("day")
    lo, hi = sb.DOG_PERIODS["all"]
    window = [d for d in sorted(L) if lo <= d <= hi]
    rng = random.Random(11)
    with_ticket = rng.sample([d for d in window if d in live.index], 60)
    without = rng.sample([d for d in window if d not in live.index], 20)
    for day in with_ticket + without:
        got = sdy.build_dog(L.get(day, []), D.get(day, []))
        if day not in live.index:
            assert got is None, day
            continue
        row = live.loc[day]
        assert "|".join(l["key"] for l in got["legs"]) == row["keys"], day
        assert got["dog_key"] == row["dog_key"] and got["dog_rank"] == row["dog_rank"], day
        assert math.isclose(got["multiplier"], row["multiplier"], abs_tol=1e-4), day
        assert sdy.settle(got)["status"] == row["status"], day
    for rule, order, obj in (("dog_gap", "gap", "p"), ("dog_likeliest", "likeliest", "p"),
                             ("dog_market_only", "lottery", "p_market")):
        rows = t[t["rule"] == rule].set_index("day")
        for day in rng.sample(list(rows.index), 15):
            got = sdy.build_dog(L[day], D[day], order=order, objective=obj)
            assert "|".join(l["key"] for l in got["legs"]) == rows.loc[day, "keys"], (rule, day)


def test_the_published_live_record_is_the_tickets_files(published):
    t = pd.read_csv(DATA / "sameday_dog_tickets.csv.gz")
    for name, (lo, hi) in sb.DOG_PERIODS.items():
        live = t[(t["rule"] == "dog") & t["day"].between(lo, hi)]
        s = published["dog"]["results"]["live"][name]
        assert s["tickets"] == len(live) and s["wins"] == int((live["status"] == "won").sum()), name
        assert s["roi"] == round(float(live["profit"].mean()), 4), name
        assert s["implied_roi"] == pytest.approx(float((live["p_market"] * live["multiplier"] - 1).mean()), abs=2e-4)


def test_the_json_has_no_nan_and_keeps_its_labels_and_caveats(published):
    def no_constant(c):
        raise ValueError(f"{c} in the json")
    text = (DATA / "sameday_backtest.json").read_text()
    json.loads(text, parse_constant=no_constant)
    json.dumps(published, allow_nan=False)
    # every key the page reads is still there
    for k in ("rule", "periods", "results", "splits", "caveats"):
        assert k in published
    assert set(published["periods"]) == {"all", "first", "last_3_years"}
    for name in ("same_day", "market_only"):
        assert set(published["results"][name]) == {"all", "first", "last_3_years"}
    dog = published["dog"]
    assert dog["version"] == sdy.DOG_VERSION and dog["salt"] == sdy.DOG_SALT
    assert dog["rule"]["weight"] == "shipped chance" and dog["rule"]["floor"] == sdy.DOG_FLOOR
    assert dog["periods"] == {k: list(v) for k, v in sb.DOG_PERIODS.items()}
    r = dog["results"]
    assert set(r) == {"live", "salts", "gap", "likeliest_dog", "market_only", "paired", "both"}
    assert r["gap"]["label"] == "largest model gap" and r["likeliest_dog"]["label"] == "likeliest underdog"
    assert "value" not in r["gap"]["label"]
    for period in sb.DOG_PERIODS:
        assert r["salts"][period]["spread"]["label"] == "how much the draw alone moves it, same games"
        assert r["salts"][period]["salts"] == 1000
    # the headline is the live salt's record, never the salts' mean
    live = r["live"]["all"]
    assert dog["headline"].startswith("At the live salt") and f"{live['wins']} won" in dog["headline"]
    for k, v in dog["splits"].items():
        assert k == "note" or "cannot move the band, the floor, the weight or the sports in v1" in v["note"]
    text = " ".join(dog["caveats"])
    for must in ("hard to beat", "Prior exposure", "not a confidence interval", "opening lines",
                 "closing moneyline", "generously", "more margin on underdogs", "Live, a UFC leg"):
        assert must in text, must
    assert any("hard to beat" in c for c in published["caveats"])
