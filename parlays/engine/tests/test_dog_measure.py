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


# ---------------------------------------------------------------- the review's fixes (2026-11-13)

def _tickets():
    t = pd.read_csv(DATA / "sameday_tickets.csv.gz")
    d = pd.read_csv(DATA / "sameday_dog_tickets.csv.gz")
    lk = t[t["rule"] == "same_day"].set_index("day")
    return d[d["rule"] == "dog"].set_index("day"), lk


def test_both_held_and_paired_rows_carry_the_prices_and_the_salts_beside_the_live_salt(published):
    """WHY: the live salt drew above 992 of the 1,000, so a both-held or
    paired row with the live salt's number alone lets its luck stand for
    the rule. Each must carry its price-implied return and the salts' mean,
    and the files must back them."""
    dog, lk = _tickets()
    for name, (lo, hi) in sb.DOG_PERIODS.items():
        b = published["dog"]["results"]["both"][name]
        p = published["dog"]["results"]["paired"][name]
        days = [d for d in dog.index if lo <= d <= hi and d in lk.index]
        assert b["days"] == p["days"] == len(days), name
        imp_d = dog.loc[days, "p_market"] * dog.loc[days, "multiplier"] - 1
        imp_l = lk.loc[days, "p_market"] * lk.loc[days, "multiplier"] - 1
        assert b["implied_roi"] == pytest.approx(float(((imp_d + imp_l) / 2).mean()), abs=2e-4), name
        assert p["implied_roi_diff"] == pytest.approx(float((imp_d - imp_l).mean()), abs=2e-4), name
        prof = (dog.loc[days, "profit"] + lk.loc[days, "profit"]) / 2
        assert b["roi"] == pytest.approx(float(prof.mean()), abs=1e-4), name
        # the salts' both-held is the salts' paired difference, halved, plus the likeliest's own
        assert b["salts_roi"] == pytest.approx(p["salts_roi_diff"] / 2 + p["likeliest"]["roi"], abs=2e-4), name
        assert b["salts_roi_low"] <= b["salts_roi"] <= b["salts_roi_high"], name
        assert p["salts_roi_diff_low"] <= p["salts_roi_diff"] <= p["salts_roi_diff_high"], name
        for k in ("implied_roi", "salts_roi", "salts_roi_low", "salts_roi_high", "salts_implied_roi"):
            assert isinstance(b[k], float), (name, k)


def test_the_draw_word_comes_from_the_rank():
    of = 1000
    word = lambda below, equal=0: sb.draw_word({"roi_below": below, "roi_equal": equal, "of": of})
    assert word(992) == "drew well"
    assert word(300) == "drew about average"           # a fabricated rank below 500: never "well"
    assert word(120) == "drew badly"
    assert word(500) == word(510) == "drew about average"


def test_a_fabricated_rank_below_500_changes_the_caveat_not_just_the_number(published, days):
    """The caveat's verdict word follows the rank: with the live salt put at
    300 of 1,000 the caveat no longer says it drew well."""
    import copy
    dog = copy.deepcopy(published["dog"])
    for name in sb.DOG_PERIODS:
        dog["results"]["salts"][name]["live_rank"].update(roi_below=300, roi_equal=0)
    legs, end, _ = sb.load()
    out = " ".join(sb.dog_caveats(dog, legs, sb.load_dogs(end), {"live": []}))
    assert "drew about average" in out and "drew well" not in out and "above 300 of the 1,000" in out
    for name in sb.DOG_PERIODS:
        dog["results"]["salts"][name]["live_rank"].update(roi_below=90)
    out = " ".join(sb.dog_caveats(dog, legs, sb.load_dogs(end), {"live": []}))
    assert "drew badly" in out and "drew well" not in out
    real = published["dog"]["results"]["salts"]["all"]["live_rank"]
    assert sb.draw_word(real) in " ".join(published["dog"]["caveats"])


def test_the_payout_words_come_from_the_overrounds(published, monkeypatch):
    assert sb.payout_words(0.028, 0.040) == ("lower than", "less")
    assert sb.payout_words(0.040, 0.028) == ("higher than", "more")
    assert sb.payout_words(0.030, 0.0304)[1] == "about the same"
    assert sb.payout_words(0.030, None) is None
    legs, end, ends = sb.load()
    res = published["results"]
    for live, want, not_want in ((0.020, "pays more than", "pays less"), (0.060, "pays less than", "pays more"),
                                 (None, "the live figure is missing", "pays less")):
        monkeypatch.setattr(sb, "nfl_live_overround", lambda live=live: (live, 0 if live is None else 12))
        text = " ".join(sb.caveats(res, None, ends, end, legs))
        assert want in text and not_want not in text, live


def test_soccer_has_its_equal_margin_reading_and_no_missing_sentence(published, days):
    text = " ".join(published["dog"]["caveats"])
    assert "Soccer's split is missing" not in text
    legs, end, _ = sb.load()
    em = sb.equal_margin(legs, sb.load_dogs(end), "Soccer", *sb.DOG_PERIODS["all"])
    assert em["outcomes"] == 3 and em["dogs"] > 1000
    assert f"the {em['dogs']:,} soccer dogs {em['chance_equal_margin']:.3f}" in text
    # the n=3 reading from the dog row alone: the whole book is 1 / (decimal x p_market)
    d = sb.load_dogs(end)
    d = d[(d["sport"] == "Soccer") & d["decimal"].between(*sdy.DOG_PRICE)]
    row = d.iloc[0]
    over = 1 / (row["decimal"] * row["p_market"]) - 1
    assert 0 < over < 0.2
    # the word "generously" is the data's, sport by sport
    fake = {"UFC": {"dogs": 1, "chance_equal_margin": 0.30, "chance_proportional": 0.31},
            "Soccer": {"dogs": 1, "chance_equal_margin": 0.32, "chance_proportional": 0.31}}
    assert sb.generous_words(fake).endswith("the UFC dogs generously and the soccer dogs no more generously "
                                            "than the equal margin")


def test_ufc_fights_without_a_winner_are_counted_from_the_archive_in_both_blocks(published):
    c, why = sb.ufc_no_winner(*sb.DOG_PERIODS["all"])
    assert why is None and c["no_winner"] == c["draws"] + c["no_contests"] and c["no_winner"] > 0
    assert c["in_legs"] == 0                       # parlay_legs keeps decided fights only
    ufc = pd.read_csv(DATA / "legs_ufc.csv.gz")
    assert not (ufc["result"] == "void").any()     # so no UFC leg is ever void here
    want = f"{c['no_winner']} of the {c['fights']:,} UFC fights"
    assert any(want in x for x in published["dog"]["caveats"])
    lo, hi = published["periods"]["all"]
    c2, _ = sb.ufc_no_winner(lo, hi)
    assert any(f"{c2['no_winner']} of the {c2['fights']:,} UFC fights" in x for x in published["caveats"])
    assert "UFC draw or" not in (sb.__doc__ or "")


def test_every_caveat_that_shows_the_live_draw_shows_the_draws_mean_and_the_prices(published):
    """The key fact the page rule rests on: a caveat line that gives the
    live draw's record also gives the 1,000 draws' mean and the
    price-implied return - and none says "live salt", the page's word is
    "draw"."""
    sl = published["dog"]["results"]["salts"]["all"]
    lv = published["dog"]["results"]["live"]["all"]
    defined = [c for c in published["dog"]["caveats"] if "its seed" in c]
    assert len(defined) == 1 and "live salt" in defined[0]          # the salt is defined once
    for c in published["dog"]["caveats"]:
        assert c in defined or ("live salt" not in c and "simulation salts" not in c), c
        if "at the live draw" in c.lower():
            assert "1,000 simulation draws" in c and "by the prices" in c, c
    text = " ".join(published["dog"]["caveats"])
    assert sb._pct(sl["roi"], 1) in text and sb._pct(lv["implied_roi"], 1) in text


def test_the_readmes_figures_are_the_jsons(published):
    """The README quotes the backtest; it is typed by hand, so this test is
    what keeps a figure or a verdict word in it from going stale when the
    backtest is re-run. Line breaks in the README do not count."""
    text = " ".join((DATA.parent / "README.md").read_text().split())
    pc = lambda x: sb._pct(x, 1)
    s, r3, mo = (published["results"][k][p] for k, p in
                 (("same_day", "all"), ("same_day", "last_3_years"), ("market_only", "all")))
    want = [f"{s['tickets']:,} tickets, {s['wins']} won ({s['hit_rate']:.1%} against the picks' "
            f"{s['expected_hit_rate']:.1%}), {pc(s['roi'])} a ticket ({pc(s['roi_low'])} to {pc(s['roi_high'])}); "
            f"last 3 years {pc(r3['roi'])} ({pc(r3['roi_low'])} to {pc(r3['roi_high'])}); the control {pc(mo['roi'])}"]
    r = published["dog"]["results"]
    lv, sl, p, b = r["live"]["all"], r["salts"]["all"], r["paired"]["all"], r["both"]["all"]
    rank = sl["live_rank"]
    want += [
        f"`{published['dog']['salt']}`) {lv['tickets']:,} tickets, {lv['wins']} won ({lv['hit_rate']:.1%} against "
        f"the prices' {lv['market_hit_rate']:.1%}), {pc(lv['roi'])} a ticket ({pc(lv['roi_low'])} to "
        f"{pc(lv['roi_high'])}); over the 1,000 simulation draws {pc(sl['roi'])} ({pc(sl['roi_low'])} to "
        f"{pc(sl['roi_high'])}); by the prices {pc(lv['implied_roi'])}.",
        f"The live draw {sb.draw_word(rank)}: it ranks above {rank['roi_below']:,} of the {rank['of']:,}",
        f"on the {p['days']:,} days both exist, it won {abs(p['hit_rate_diff']) * 100:.1f} points "
        f"{'less' if p['hit_rate_diff'] < 0 else 'more'} often",
        f"holding both returned {pc(b['roi'])} a unit at the live draw, {pc(b['salts_roi'])} over the draws "
        f"({pc(b['salts_roi_low'])} to {pc(b['salts_roi_high'])}) and {pc(b['implied_roi'])} by the prices",
    ]
    c, _ = sb.ufc_no_winner(*sb.DOG_PERIODS["all"])
    want.append(f"({c['no_winner']} of {c['fights']:,} in the window)")
    for w in want:
        assert w in text, w
