"""Tickets v2's measurement (experiments/sameday_backtest.py, the json's "v2")
and its legs (legs_history.py --v2): what is published reproduces from the
files by sameday's own v2 builders, the legs are what sameday.py's LEGS v2
says, and v1 is left exactly as it was measured.

WHY these tests: v2 is measured beside v1, on new files, in the same json.
A rebuild that moved a v1 file or a v1 number would change a record that
was published before v2 existed; a leg file with two legs on one event
would let a ticket hold a match's winner and its over/under together; a
record without its price-implied return would let a lucky draw stand for
the rule; and a NaN breaks the page."""

import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "experiments"))

import sameday as sdy  # noqa: E402
import sameday_backtest as sb  # noqa: E402

ROOT = ENGINE.parents[1]
DATA = ENGINE.parents[0] / "data"

# The v1 legs files as v1 was measured on them (sha256 of the committed bytes
# when v2 was added, 2026-11-16). v2 writes new files only.
V1_SHA256 = {
    "legs_ufc.csv.gz": "fcb4a3f892851e7f90b5552e3286bfc26b7903a9a36910b7039be3fecbd4ed30",
    "legs_nfl.csv.gz": "e53eb5c9f19310b834fcdd64120dfef3556dc89afc192683373522f80e74c0f7",
    "legs_soccer.csv.gz": "13337a20105892a4d9745081c4b3758aaa255829470b3bf969e13d72f93258b6",
    "legs_dogs_ufc.csv.gz": "fcf7deeb285b9b7334821ed4084c7db02c501e64798973c523bc0b8ceab024e5",
    "legs_dogs_nfl.csv.gz": "9c028c9e7eea0c6acdef14014a90125e40f18fd009bbbae5b254fc1cd926f55c",
    "legs_dogs_soccer.csv.gz": "c125fb95fd5bba7dbccef7573e30239f6fe516ab89c31fa8ac1daae5d43c7b35",
}
V2_FILES = ("legs_v2_soccer", "legs_dogs_v2_soccer", "legs_v2_ou25", "legs_v2_nfl_lines", "legs_v2_ufc_method")


@pytest.fixture(scope="module")
def published():
    return json.loads((DATA / "sameday_backtest.json").read_text())


@pytest.fixture(scope="module")
def v2(published):
    return published["v2"]


@pytest.fixture(scope="module")
def files():
    return {name: pd.read_csv(DATA / f"{name}.csv.gz") for name in V2_FILES + ("legs_ufc", "legs_nfl", "legs_soccer",
                                                                                "legs_dogs_soccer")}


@pytest.fixture(scope="module")
def tickets():
    return pd.read_csv(DATA / "sameday_v2_tickets.csv.gz")


@pytest.fixture(scope="module")
def frames():
    return sb.load_v2()


# ---------------------------------------------------------------- v1 untouched

def test_v1_files_are_byte_identical_and_v2_was_measured_beside_them(v2):
    for name, want in V1_SHA256.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == want, name
    assert v2["v1_files_sha256"] == V1_SHA256


def test_v1_numbers_are_unchanged_by_v2(published, v2):
    """The json's v1 keys are exactly those v2 was measured beside (v2
    recorded their sha256), so adding v2 moved no v1 number; and the v1
    likeliest record still reproduces from the v1 files (test_dog_measure
    holds every period)."""
    assert sb.v1_fingerprint(published) == v2["v1_sha256"]
    assert set(published) == {"generated", "rule", "periods", "days", "leg_tables_end", "results", "splits",
                              "caveats", "dog", "v2"}
    v1 = published["dog"]["results"]["live"]["all"]
    assert v2["activity"]["v1"]["all"]["tickets"] == published["results"]["same_day"]["all"]["tickets"] + v1["tickets"]


def test_the_likeliest_v2_over_the_v1_pool_is_the_v1_ticket(frames):
    """On v1's pool (UFC, NFL and the top five) the v2 legs and builder give
    v1's ticket: the v2 constructor changes nothing the search reads."""
    lk = pd.read_csv(DATA / "sameday_tickets.csv.gz")
    lk = lk[lk["rule"] == "same_day"].set_index("day")
    rng = random.Random(5)
    days = rng.sample(sorted(lk.index[lk.index >= "2016-08-01"]), 40)
    by = sb.v2_days(frames, days)
    for day in days:
        W = [l for l in by[day][0] if l["sport"] != "Soccer" or l.get("group") == "top"]
        t = sdy.build_likeliest_v2(W)
        assert t is not None and math.isclose(t["multiplier"], lk.loc[day, "multiplier"], abs_tol=1e-4), day
        assert "|".join(f"{l['sport']}:{l['pick']}@{l['decimal']:.3f}" for l in t["legs"]) == lk.loc[day, "legs"], day


# ---------------------------------------------------------------- the legs

def test_keys_are_one_per_event_and_each_event_sits_on_one_day(files):
    s, d, o = files["legs_v2_soccer"], files["legs_dogs_v2_soccer"], files["legs_v2_ou25"]
    assert not s["key"].duplicated().any() and not d["key"].duplicated().any() and not o["key"].duplicated().any()
    assert set(d["key"]) <= set(s["key"])
    n = files["legs_v2_nfl_lines"]
    assert not n.duplicated(["key", "bet"]).any() and set(n["bet"]) == {"SPREAD", "TOTAL"}
    m = files["legs_v2_ufc_method"]
    assert not m.duplicated(["key", "corner", "bet_detail"]).any()
    assert (m.groupby("key").size() == 6).all()
    day = {}
    for name in V2_FILES + ("legs_ufc", "legs_nfl", "legs_soccer"):
        for k, dd in files[name].groupby("key")["day"].unique().items():
            assert len(dd) == 1, (name, k)
            assert day.setdefault(k, dd[0]) == dd[0], (name, k)
    # the events are the v1 files' (same key, same event name)
    for name, v1 in (("legs_v2_ufc_method", "legs_ufc"), ("legs_v2_nfl_lines", "legs_nfl")):
        x = files[name].drop_duplicates("key").merge(files[v1], on="key", suffixes=("", "_v1"))
        assert (x["event"] == x["event_v1"]).all(), name
    assert set(m["key"]) <= set(files["legs_ufc"]["key"])


def test_the_top_five_1x2_rows_are_the_v1_soccer_file(files):
    v1 = files["legs_soccer"]
    top = files["legs_v2_soccer"]
    top = top[top["group"] == "top"][list(v1.columns)].sort_values(["day", "key"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(top, v1.sort_values(["day", "key"]).reset_index(drop=True), check_dtype=False)
    dv1 = files["legs_dogs_soccer"]
    dtop = files["legs_dogs_v2_soccer"]
    dtop = dtop[dtop["group"] == "top"][list(dv1.columns)].sort_values(["day", "key"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(dtop, dv1.sort_values(["day", "key"]).reset_index(drop=True), check_dtype=False)


def test_soccer_leagues_prices_and_over_under_are_the_apps():
    """Every league in the app gives 1X2 legs, the extra leagues at the close
    (no pre-match price in their history), the rest pre-match; over/under
    2.5 only where the league ships it from the market."""
    code = ("import json, soccer_leagues as slg, soccer_data as sd; a = slg.in_app(); "
            "print(json.dumps({'in_app': a, 'extra': [c for c in a if c in sd.EXTRA], "
            "'ou': [c for c in a if 'o25' not in slg.withheld(c, 'market')]}))")
    out = json.loads(subprocess.run([sys.executable, "-c", code], cwd=ROOT / "soccer" / "engine", check=True,
                                    capture_output=True, text=True).stdout)
    s = pd.read_csv(DATA / "legs_v2_soccer.csv.gz")
    assert set(s["league"]) == set(out["in_app"])
    assert (s.loc[s["league"].isin(out["extra"]), "price_source"] == "close").all()
    assert (s.loc[~s["league"].isin(out["extra"]), "price_source"] == "pre").all()
    o = pd.read_csv(DATA / "legs_v2_ou25.csv.gz")
    assert set(o["league"]) == set(out["ou"]) and (o["price_source"] == "pre").all()
    assert (o["p"] > 0.5).all() and (o["p"] == o["p_market"]).all()            # level prices give no leg
    assert set(o["pick"]) == {"Over", "Under"} and (o["bet"] == "OU25").all()


def test_nfl_lines_side_price_and_grading(files):
    n = files["legs_v2_nfl_lines"]
    g = pd.read_csv(ROOT / "nfl" / "data" / "games.csv.gz", low_memory=False).set_index("game_id")
    level = n[n["side_source"] == "model"]
    assert (level["p"] == 0.5).all() and len(level) > 0
    assert (n.loc[n["side_source"] == "market", "p"] > 0.5).all()
    spreads = n[n["bet"] == "SPREAD"]
    assert not (g.loc[spreads["key"], "spread_line"] == 0).any()                 # a zero spread gives no leg
    for r in n.sample(400, random_state=3).itertuples():
        x = g.loc[r.key]
        if r.bet == "SPREAD":
            home = r.pick == sb_franchise(x["home_team"])
            margin = x["result"] - x["spread_line"]
            odds = x["home_spread_odds"] if home else x["away_spread_odds"]
            want = "void" if margin == 0 else ("won" if (margin > 0) == home else "lost")
        else:
            over = r.pick == "Over"
            diff = x["total"] - x["total_line"]
            odds = x["over_odds"] if over else x["under_odds"]
            want = "void" if diff == 0 else ("won" if (diff > 0) == over else "lost")
        assert r.result == want, r.key
        assert math.isclose(r.decimal, 1 + (odds / 100 if odds > 0 else 100 / -odds), abs_tol=1e-5), r.key


def sb_franchise(code):
    sys.path.insert(0, str(ROOT / "nfl" / "engine"))
    import nfl_data as nd
    return nd.franchise(code)


def test_method_legs_chance_estimate_and_grading(files):
    m = files["legs_v2_ufc_method"]
    assert m["day"].min() >= "2016-08-01" and m["day"].max() <= sdy.PROPS_HISTORY_END
    assert ((m["p"] - m["p_win"] * m["share"]).abs() < 2e-6).all()
    assert ((m.groupby(["key", "corner"])["share"].sum() - 1).abs() < 1e-5).all()
    assert ((m.groupby("key")["p_market"].sum() - 1).abs() < 1e-5).all()        # the six-way no-vig book
    assert set(m["bet_detail"]) == set(sdy.METHOD_TYPES)
    # the estimate: refitted on earlier years only, missing (with its reason) for 2016
    est = m[m["est_decimal"].notna()]
    assert (est["est_fit_last"] < est["day"].str[:4] + "-01-01").all()
    miss = m[m["est_decimal"].isna()]
    assert miss["est_missing"].notna().all()
    assert (m.loc[m["day"] < "2017-01-01", "est_decimal"].isna()).all()
    assert miss["day"].str[:4].value_counts().index[0] == "2016"
    # graded by method: at most one prop of a fight wins, and only its winner's
    won = m[m["result"] == "won"]
    assert not won["key"].duplicated().any()
    ufc = files["legs_ufc"].set_index("key")
    for r in won.sample(200, random_state=2).itertuples():
        pick_won = ufc.loc[r.key, "result"] == "won"
        assert (ufc.loc[r.key, "pick"] == r.pick) == pick_won, r.key
    for r in m.sample(300, random_state=4).itertuples():
        corners = [r.pick, r.opponent]
        assert r.lot_key == sdy.method_lot_key(corners, r.pick, r.bet_detail)


# ---------------------------------------------------------------- the json

def test_the_json_has_no_nan_and_keeps_its_labels(v2):
    def no_constant(c):
        raise ValueError(f"{c} in the json")
    json.loads((DATA / "sameday_backtest.json").read_text(), parse_constant=no_constant)
    assert v2["version"] == sdy.V2_VERSION
    assert v2["rule"]["dog_salt"] == sdy.DOG_SALT_V2 and v2["rule"]["props_salt"] == sdy.PROPS_SALT
    assert v2["rule"]["props_live_band"] == list(sdy.PROPS_LIVE_BAND)
    assert v2["periods"] == {k: list(x) for k, x in sb.DOG_PERIODS.items()}
    assert v2["props_periods"]["all"][1] == sdy.PROPS_HISTORY_END
    assert v2["props_periods"]["last_3_years"] == [sb.DOG_PERIODS["last_3_years"][0], sdy.PROPS_HISTORY_END]
    assert set(v2["kinds"]) == set(sb.V2_KINDS)
    for k in ("underdog", "props_as_live", "props_as_priced"):
        for v in sb.V2_VARIANTS:
            for p, st in v2["kinds"][k]["salts"][v].items():
                if st.get("tickets_per_salt"):
                    assert st["salts"] == 1000 and st["spread"]["label"] == sb.SPREAD_LABEL, (k, v, p)
    for k, b in v2["kinds"].items():
        for name, sp in b["splits"].items():
            assert name == "note" or sp["note"] == sb.V2_SPLIT_NOTE, (k, name)
    assert v2["headline"].startswith("PROPS as live (the headline)")


def _records(x, path=()):
    if isinstance(x, dict):
        if "tickets" in x and ("roi" in x or x.get("tickets") == 0) and "by_kind" not in x:
            yield path, x
        if "tickets_per_salt" in x:
            yield path, x
        for k, v in x.items():
            yield from _records(v, path + (k,))


def test_every_record_has_its_price_implied_return(v2):
    n = 0
    for path, r in _records(v2["kinds"]):
        if "legs_by" in "/".join(path):
            continue
        n += 1
        assert "implied_roi" in r, path
        if r.get("tickets") or r.get("tickets_per_salt"):
            assert isinstance(r["implied_roi"], float), path
    assert n > 100


def test_every_record_is_reported_with_and_without_the_extra_leagues(v2):
    for k in sb.V2_KINDS:
        live = v2["kinds"][k]["live"]
        assert set(live) == set(sb.V2_VARIANTS), k
        for v in sb.V2_VARIANTS:
            assert set(live[v]) == {"all", "first", "last_3_years"}, (k, v)
            for p, s in live[v].items():
                assert {"days", "days_with_ticket", "share_of_days", "tickets_per_day"} <= set(s), (k, v, p)
                assert s["days"] == sb.calendar_days(v2["kinds"][k]["periods"][p]), (k, v, p)


def test_the_published_records_are_the_tickets_files(v2, tickets):
    for k in sb.V2_KINDS:
        for v in sb.V2_VARIANTS:
            for p, (lo, hi) in v2["kinds"][k]["periods"].items():
                t = tickets[(tickets["kind"] == k) & (tickets["variant"] == v) & tickets["day"].between(lo, hi)]
                s = v2["kinds"][k]["live"][v][p]
                assert s["tickets"] == len(t) and s.get("wins", 0) == int((t["status"] == "won").sum()), (k, v, p)
                if len(t):
                    assert s["roi"] == round(float(t["profit"].mean()), 4), (k, v, p)
                    assert s["implied_roi"] == pytest.approx(float((t["p_market"] * t["multiplier"] - 1).mean()),
                                                             abs=2e-4)
    for _, t in tickets.sample(500, random_state=1).iterrows():
        keys = t["keys"].split("|")
        assert len(keys) == len(set(keys)) == t["n_legs"]                        # one leg per event


def _day_legs(by, day, variant):
    W, O, D = by.get(day, ([], [], []))
    if variant == "without_extra_leagues":
        W, O, D = sb._not_extra(W), sb._not_extra(O), sb._not_extra(D)
    return W, O, D


def _build(kind, W, O, D):
    M = sorted((l for l in O if l["bet"] == "METHOD"), key=lambda l: (l["key"], l["lot_key"]))
    if kind == "likeliest":
        return sdy.build_likeliest_v2(W)
    if kind == "likeliest_market_only":
        return sdy.build_likeliest_v2(W, "p_market")
    if kind == "mixed":
        return sdy.build_mixed(W + O)
    if kind == "mixed_market_only":
        return sdy.build_mixed(W + O, "p_market")
    if kind == "underdog":
        return sdy.build_underdog_v2(W, D)
    if kind == "props_as_priced":
        return sdy.build_props(W, M) if M else None
    if kind == "props_as_live":
        M = [m for m in M if m.get("est_decimal") is not None]
        return sdy.build_props(W, M, band=sdy.PROPS_LIVE_BAND, price_key="est_decimal") if M else None
    raise ValueError(kind)


def test_a_sample_of_v2_tickets_reproduces_from_sameday_builders(tickets, frames):
    """Each kind's published tickets, and days it published none, rebuilt
    from the legs files by sameday's v2 builders: the same events, markets
    and selections, multiplier and result."""
    rng = random.Random(17)
    want = {}
    for (k, v), g in tickets.groupby(["kind", "variant"]):
        g = g.set_index("day")
        pick = rng.sample(list(g.index), min(6, len(g)))
        lo, hi = (sb.PROPS_PERIODS if k.startswith("props") else sb.V2_PERIODS)["all"]
        span = [d for d in sorted(frames["winners"]["day"].unique()) if lo <= d <= hi and d not in g.index]
        if k.startswith("props"):
            ufc_days = set(frames["others"].loc[frames["others"]["bet"] == "METHOD", "day"])
            span = [d for d in span if d in ufc_days]
        none = rng.sample(span, min(2, len(span)))
        want[(k, v)] = [(d, g.loc[d]) for d in pick] + [(d, None) for d in none]
    days = sorted({d for rows in want.values() for d, _ in rows})
    by = sb.v2_days(frames, days)
    for (k, v), rows in want.items():
        for day, row in rows:
            got = _build(k, *_day_legs(by, day, v))
            if row is None:
                assert got is None, (k, v, day)
                continue
            assert got is not None, (k, v, day)
            assert "|".join(l["key"] for l in got["legs"]) == row["keys"], (k, v, day)
            assert "|".join(f"{l.get('bet') or '1X2'}:{l.get('bet_detail')}:{l['pick']}" for l in got["legs"]) \
                == row["markets"], (k, v, day)
            assert math.isclose(got["multiplier"], row["multiplier"], abs_tol=1e-4), (k, v, day)
            assert sdy.settle(got)["status"] == row["status"], (k, v, day)


def test_as_live_pays_the_real_prop_and_searched_the_estimate(tickets, frames):
    t = tickets[(tickets["kind"] == "props_as_live") & (tickets["variant"] == "all_leagues")]
    lo, hi = sdy.PROPS_LIVE_BAND
    assert t["searched_multiplier"].between(lo - 1e-9, hi + 1e-9).all()
    assert (t["day"] >= "2017-01-01").all()                                      # no estimate for 2016
    p = tickets[(tickets["kind"] == "props_as_priced") & (tickets["variant"] == "all_leagues")]
    assert p["multiplier"].between(*sdy.BAND).all()
    assert t["markets"].str.count("METHOD:").eq(1).all() and p["markets"].str.count("METHOD:").eq(1).all()


def test_the_caveats_say_what_the_rule_and_the_project_require(v2):
    text = " ".join(v2["caveats"])
    for must in ("hard to beat", "Prior exposure", "estimate", "method_price.json", "not a confidence interval",
                 sdy.PROPS_HISTORY_END, "closing", "Expected (sameday.py)", "descriptive"):
        assert must in text, must
    doc = json.loads((ROOT / "engine" / "data" / "method_price.json").read_text())
    assert f"{doc['test']['all_single_props']:.1%}" in text
    assert sb._doc_block("PRIOR EXPOSURE") in text
    for k in ("underdog", "props_as_live"):
        rank = v2["kinds"][k]["salts"]["all_leagues"]["all"]["live_rank"]
        assert f"{sb.draw_word(rank)} (above {rank['roi_below']:,} of {rank['of']:,})" in text, k


def test_the_readmes_v2_figures_are_the_jsons(v2):
    """The README's v2 paragraphs quote the json; this keeps them from going
    stale when the backtest is re-run. Line breaks do not count."""
    text = " ".join((DATA.parent / "README.md").read_text().split())
    pc = lambda x: sb._pct(x, 1)
    K = v2["kinds"]
    L, U, X = (K[k]["live"]["all_leagues"]["all"] for k in ("likeliest", "underdog", "mixed"))
    PL, PP = K["props_as_live"]["live"]["all_leagues"]["all"], K["props_as_priced"]["live"]["all_leagues"]["all"]
    US, PS = K["underdog"]["salts"]["all_leagues"]["all"], K["props_as_live"]["salts"]["all_leagues"]["all"]
    a, a1 = v2["activity"]["all_leagues"]["all"], v2["activity"]["v1"]["all"]
    want = [
        f"{a['tickets']:,} tickets ({a['tickets_per_day']:.2f} a day, a ticket on {a['share_of_days']:.1%} of days) "
        f"against v1's {a1['tickets']:,} ({a1['tickets_per_day']:.2f} a day)",
        f"LIKELIEST v2 {L['tickets']:,} tickets, {L['wins']:,} won, {pc(L['roi'])} a ticket ({pc(L['roi_low'])} to "
        f"{pc(L['roi_high'])}), by the prices {pc(L['implied_roi'])}",
        f"MIXED {X['tickets']:,}, {pc(X['roi'])} ({pc(X['roi_low'])} to {pc(X['roi_high'])}), by the prices "
        f"{pc(X['implied_roi'])}",
        f"UNDERDOG v2 {U['tickets']:,} at the live draw, {pc(U['roi'])}; over the 1,000 simulation draws "
        f"{pc(US['roi'])} ({pc(US['roi_low'])} to {pc(US['roi_high'])}), by the prices {pc(U['implied_roi'])}",
        f"PROPS as live (the headline) {PL['tickets']:,} tickets, {PL['wins']:,} won, {pc(PL['roi'])} a ticket "
        f"({pc(PL['roi_low'])} to {pc(PL['roi_high'])}); over the 1,000 simulation draws {pc(PS['roi'])}; by the "
        f"prices {pc(PL['implied_roi'])}; as priced {pc(PP['roi'])}, by the prices {pc(PP['implied_roi'])}",
    ]
    for w in want:
        assert w in text, w
