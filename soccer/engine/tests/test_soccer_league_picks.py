"""Every league's production pieces, on made-up leagues: the price each
league is picked on, the pick (recalibrated on matches dated before the
league-season only, the raw price where its recalibration lost, the model's
own or none where no price), the league's base rates walking forward on
dates, the model over leagues of different calendars, what a card withholds
- and the per-league decisions, held to the rules that made them."""

import math

import numpy as np
import pandas as pd
import pytest

import soccer_backtest as bt
import soccer_data as sd
import soccer_leagues as sl
import soccer_model as smod
import soccer_scores as ssc
from experiments import shots_ratings as sr

SMALL = dict(min_train=60, min_played=2)          # shots_ratings on small made-up leagues
CLUBS = [f"Club {k}" for k in range(16)]


def _season(code, season, start, teams, rng, price="pre", shots=True, spacing=7):
    """One double round robin of `teams` from `start`, results following
    each club's strength, priced as `price` says ("pre": pre-match and
    closing averages and over/under, as a European division; "close": the
    closing average only, as an extra league's history; None: no price)."""
    strength = dict(zip(teams, np.linspace(-0.45, 0.45, len(teams))))
    rows = []
    pairs = [(h, a) for h in teams for a in teams if h != a]
    rng.shuffle(pairs)
    per_day = max(1, len(teams) // 2)
    for k, (h, a) in enumerate(pairs):
        day = pd.Timestamp(start) + pd.Timedelta(days=spacing * (k // per_day))
        lam = math.exp(0.32 + strength[h] - strength[a])
        nu = math.exp(0.05 + strength[a] - strength[h])
        hg, ag = int(rng.poisson(lam)), int(rng.poisson(nu))
        ph, pd_, pa, po = (float(v[0]) for v in sr.score_probs(np.array([lam]), np.array([nu]), -0.06))
        r = {"match_id": sd.match_id(code, day.date(), h, a), "league": code, "season": season,
             "date": day.strftime("%Y-%m-%d"), "time": "15:00", "kickoff": day + pd.Timedelta(hours=15),
             "home": h, "away": a, "hg": hg, "ag": ag, "result": "H" if hg > ag else "A" if hg < ag else "D",
             "hr": 0, "ar": 0}
        if shots:
            r.update(hs=hg * 4 + int(rng.integers(3, 9)), **{"as": ag * 4 + int(rng.integers(2, 8))},
                     hst=hg + int(rng.integers(1, 4)), ast=ag + int(rng.integers(0, 3)))
        # a market with a favourite-longshot lean, so a recalibration has something to fit
        q = np.array([ph, pd_, pa]) ** 0.92
        q /= q.sum()
        if price == "pre":
            r.update(avg_h=0.94 / q[0], avg_d=0.94 / q[1], avg_a=0.94 / q[2], avg_o25=0.94 / po,
                     avg_u25=0.94 / (1 - po), avgc_h=0.95 / q[0], avgc_d=0.95 / q[1], avgc_a=0.95 / q[2])
        elif price == "close":
            r.update(avgc_h=0.95 / q[0], avgc_d=0.95 / q[1], avgc_a=0.95 / q[2])
        rows.append(r)
    return rows


def _frame(spec, seed=0):
    """spec: [(code, seasons, month-day the season starts, teams, price, shots)]."""
    rng = np.random.default_rng(seed)
    rows = []
    for code, seasons, md, teams, price, shots in spec:
        for s in seasons:
            rows += _season(code, s, f"{s}-{md}", teams, rng, price, shots)
    return sd.frame(rows).sort_values(["date", "league", "home"]).reset_index(drop=True)


def _preds(frame, seed=1):
    """A stand-in for the model's own chances (anything finite)."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([4, 2.5, 3], len(frame))
    return pd.DataFrame(p, index=frame.index, columns=["p_h", "p_d", "p_a"])


@pytest.fixture(scope="module")
def world():
    """Two top leagues and a second tier priced pre-match; an extra league on
    the calendar year, priced at the close only."""
    return _frame([("E0", range(2005, 2012), "08-10", CLUBS[:10], "pre", True),
                   ("SP1", range(2005, 2012), "08-17", CLUBS[:10], "pre", True),
                   ("B1", range(2005, 2012), "07-25", CLUBS[6:16], "pre", False),
                   ("NOR", range(2005, 2012), "03-20", CLUBS[:10], "close", False)])


# ------------------------------------------------------------- the price each league is picked on

def test_each_league_is_priced_on_what_its_history_has():
    f = _frame([("E0", [2010], "08-10", CLUBS[:4], "pre", True), ("NOR", [2010], "03-20", CLUBS[:4], "close", False)])
    one = f.index[f["league"] == "E0"][1]
    f.loc[one, ["avg_h", "avg_d", "avg_a"]] = np.nan         # a European match with only the close: no price
    nor = f.index[f["league"] == "NOR"]
    f.loc[nor[0], ["avg_h", "avg_d", "avg_a"]] = [2.0, 3.4, 3.9]   # an extra league's live fixture: pre-match
    q, kind = smod.market_price(f)
    pre, close = bt.no_vig(f, "pre"), bt.no_vig(f, "close")
    assert kind[one] is None and q.loc[one].isna().all()
    e0 = f.index[(f["league"] == "E0") & (f.index != one)]
    assert (kind[e0] == "pre").all() and np.array_equal(q.loc[e0].to_numpy(), pre.loc[e0].to_numpy())
    assert kind[nor[0]] == "pre" and np.array_equal(q.loc[nor[0]].to_numpy(), pre.loc[nor[0]].to_numpy())
    assert (kind[nor[1:]] == "close").all() and np.array_equal(q.loc[nor[1:]].to_numpy(), close.loc[nor[1:]].to_numpy())


# ------------------------------------------------------------- the pick

def test_the_top_fives_weights_for_the_top_five_are_the_shipped_pick_to_the_bit(world, monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 300)
    preds = _preds(world)
    old, w_old = smod.pick(world, preds)
    new, w_new = smod.pick_leagues(world, preds, list(sd.TOP), "top5")
    top = world["league"].isin(sd.TOP) & ~old["model_only"]
    assert top.sum() > 1000
    assert np.array_equal(old.loc[top, ["p_h", "p_d", "p_a"]].to_numpy(), new.loc[top, ["p_h", "p_d", "p_a"]].to_numpy())
    assert (new.loc[top, "basis"] == "recalibrated").all()
    for (lg, s), w in w_new.items():
        if lg in sd.TOP and w is not None:
            assert w == w_old[s]


@pytest.mark.parametrize("option", smod.WEIGHT_OPTIONS)
def test_a_league_seasons_weights_see_only_matches_dated_before_its_first_match(world, option, monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    preds = _preds(world)
    leagues = ["E0", "SP1", "B1", "NOR"]
    ref, w_ref = smod.pick_leagues(world, preds, leagues, option)
    for lg, season in (("B1", 2009), ("NOR", 2010), ("SP1", 2008)):
        first = world.loc[(world["league"] == lg) & (world["season"] == season), "date"].min()
        later = world["date"] >= first
        moved = world.copy()
        rng = np.random.default_rng(5)
        moved.loc[later, "result"] = rng.choice(list("HDA"), int(later.sum()))
        rows = (world["league"] == lg) & (world["season"] == season)
        new, w_new = smod.pick_leagues(moved, preds, leagues, option)
        assert w_new[(lg, season)] == w_ref[(lg, season)], (option, lg, season)
        assert np.array_equal(new.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy(), ref.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy())
    assert any(w is not None for (lg, s), w in w_ref.items() if lg == "NOR" and s == 2010)


def test_a_match_filed_under_an_earlier_season_but_played_later_is_not_in_the_later_seasons_weights(monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 100)
    f = _frame([("ARG", range(2008, 2013), "02-10", CLUBS[:8], "close", False)])
    late = f.index[f["season"] == 2010][-1]
    # season 2010's last match was played after season 2011 had begun (ARG 2013/14's, played 29/01/2015)
    f.loc[late, "date"] = "2011-03-01"
    f = f.sort_values("date").reset_index(drop=True)
    late = f.index[(f["season"] == 2010) & (f["date"] == "2011-03-01")][0]
    preds = _preds(f)
    ref, w_ref = smod.pick_leagues(f, preds, ["ARG"], "per_league")
    flip = f.copy()
    flip.loc[late, "result"] = {"H": "A", "A": "D", "D": "H"}[f.loc[late, "result"]]
    _, w_new = smod.pick_leagues(flip, preds, ["ARG"], "per_league")
    assert w_new[("ARG", 2011)] == w_ref[("ARG", 2011)] and w_new[("ARG", 2012)] != w_ref[("ARG", 2012)]


def test_pooling_per_price_type_never_crosses_from_the_close_to_the_pre_match(world, monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    preds = _preds(world)
    leagues = ["E0", "SP1", "B1", "NOR"]
    ref, w_ref = smod.pick_leagues(world, preds, leagues, "per_price")
    for lg in ("E0", "NOR"):
        other = world["league"] != lg if lg == "NOR" else world["league"] == "NOR"
        moved = world.copy()
        moved.loc[other, "result"] = np.random.default_rng(2).choice(list("HDA"), int(other.sum()))
        _, w_new = smod.pick_leagues(moved, preds, leagues, "per_price")
        own = [k for k in w_ref if (k[0] == "NOR") == (lg == "NOR")]
        assert own and all(w_new[k] == w_ref[k] for k in own)


def test_the_raw_price_where_recalibration_lost_and_the_model_or_nothing_where_unpriced(world, monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    f = world.copy()
    unpriced = np.concatenate([f.index[(f["season"] == 2011) & (f["league"] == lg)][:10] for lg in ("E0", "SP1", "B1", "NOR")])
    f.loc[unpriced, [c for c in f.columns if c.startswith("avg")]] = np.nan
    preds = _preds(f)
    out, w = smod.pick_leagues(f, preds, ["E0", "SP1", "B1", "NOR"], "per_price", raw=["B1"],
                               model_alone=["E0", "NOR"])
    q, _ = smod.market_price(f)
    b1 = (f["league"] == "B1") & q.notna().all(axis=1)
    assert np.array_equal(out.loc[b1, ["p_h", "p_d", "p_a"]].to_numpy(), q.loc[b1].to_numpy())
    assert (out.loc[b1, "basis"] == "raw").all() and all(v is None for (lg, s), v in w.items() if lg == "B1")
    for i in unpriced:
        lg = f.loc[i, "league"]
        assert out.loc[i, "model_only"]
        if lg in ("E0", "NOR"):
            assert out.loc[i, "basis"] == "model" and np.array_equal(out.loc[i, ["p_h", "p_d", "p_a"]].to_numpy(float),
                                                                    preds.loc[i].to_numpy(float))
        else:
            assert out.loc[i, "basis"] == "unpriced" and out.loc[i, ["p_h", "p_d", "p_a"]].isna().all()
    rec = (f["league"] == "SP1") & (f["season"] == 2010)
    assert (out.loc[rec, "basis"] == "recalibrated").all()
    assert out.loc[f["league"] == "D1", ["p_h"]].isna().all().all()        # not asked for: nothing


def test_an_unknown_weight_option_is_refused(world):
    with pytest.raises(ValueError):
        smod.pick_leagues(world, _preds(world), ["E0"], "per_country")


# ------------------------------------------------------------- the base rates

def test_a_leagues_base_rates_come_from_matches_dated_before_its_season(world):
    f = _frame([("ARG", range(2008, 2012), "02-10", CLUBS[:8], "close", False)])
    late = f.index[f["season"] == 2009][-1]
    f.loc[late, "date"] = "2010-06-01"                 # filed under 2009, played in 2010's season
    base = ssc.baseline(f, ["ARG"])
    hg, ag = f["hg"].to_numpy(), f["ag"].to_numpy()
    for s in (2009, 2010, 2011):
        rows = f["season"] == s
        first = f.loc[rows, "date"].min()
        train = (f["date"] < first).to_numpy()
        want = [(hg[train] > ag[train]).mean(), (hg[train] == ag[train]).mean(), (hg[train] < ag[train]).mean()]
        assert np.allclose(base.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy(), want)
    assert base.loc[f["season"] == 2008].isna().all().all()                 # its first season: none
    # where seasons never overlap (the top five) the cut on dates is the cut on seasons
    top = ssc.baseline(world)
    e0 = world["league"] == "E0"
    for s in (2007, 2010):
        rows = e0 & (world["season"] == s)
        train = e0 & (world["season"] < s)
        assert np.isclose(top.loc[rows, "btts"].iloc[0], ((world.loc[train, "hg"] > 0) & (world.loc[train, "ag"] > 0)).mean())


# ------------------------------------------------------------- the model over every league

def test_the_shipped_model_is_untouched_by_other_leagues_and_calendars(world):
    ten = world[world["league"].isin(["E0", "SP1"])]
    ref = sr.predict_walk_forward(ten, score_params=True, **SMALL)
    alone = sr.predict_leagues(ten, list(sd.TOP), score_params=True, england="top-two", **SMALL)
    every = sr.predict_leagues(world, ["E0", "SP1", "B1", "NOR"], score_params=True, england="top-two", **SMALL)
    cols = ["p_h", "p_d", "p_a", "p_o25", "lam", "nu", "rho"]
    assert np.array_equal(ref[cols].to_numpy(), alone[cols].to_numpy(), equal_nan=True)
    assert np.array_equal(ref[cols].to_numpy(), every.loc[ten.index, cols].to_numpy(), equal_nan=True)
    for lg in ("B1", "NOR"):                            # goals only (no shots), its own pool and group
        got = every.loc[world["league"] == lg, "p_h"]
        assert got.notna().mean() > 0.5
    p = every[["p_h", "p_d", "p_a"]].dropna()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_the_rating_passes_keep_calendars_and_englands_pool_apart():
    every = list(sd.ALL_LEAGUES)
    top, aug, cal = sr.passes(every, "all")
    assert set(top[0]) == {"E0", "E1", "E2", "E3", "EC", "SP1", "SP2", "I1", "I2", "D1", "D2", "F1", "F2"} and top[1] == {}
    assert set(cal[0]) == set(sr.CALENDAR_PASS) and not set(aug[0]) & set(cal[0])
    two = sr.passes(every, "top-two")
    assert set(two[0][0]) == set(sd.LEAGUES) and {"E2", "E3", "EC"} <= set(two[1][0])
    assert two[1][1] == {"E2": "England (lower)", "E3": "England (lower)", "EC": "England (lower)"}
    assert sum(len(g) for g, _ in two) == 38


def test_the_model_over_every_league_sees_only_earlier_days(world):
    def predict(df):
        return sr.predict_leagues(df, ["E0", "SP1", "B1", "NOR"], **SMALL)
    shared = world.groupby("date")["league"].nunique()
    for year in ("2009", "2010"):
        cut = shared[(shared >= 2) & shared.index.str.startswith(year)].index[0]
        assert bt.check_point_in_time(predict, world, cut)


def test_a_club_rated_in_two_pools_shows_the_rating_of_the_league_it_played_in_last():
    # England's pool split (E0-E1 apart from E2-EC): a club up from League One
    # is rated in both passes; the table shows its Championship rating.
    up, down = CLUBS[8], CLUBS[0]
    e1, e2 = CLUBS[:8], CLUBS[8:16]
    f = _frame([("E1", [2008], "08-10", e1, "pre", True), ("E2", [2008], "08-10", e2, "pre", True),
                ("E1", [2009], "08-10", [up] + e1[1:], "pre", True),
                ("E2", [2009], "08-10", [down] + e2[1:], "pre", True)])
    p = sr.predict_leagues(f, ["E1", "E2"], england="top-two", **SMALL)
    r = p.attrs["ratings"]
    assert r[up]["league"] == "E1" and r[down]["league"] == "E2"


def test_a_match_filed_under_an_earlier_season_never_restarts_its_clubs():
    f = _frame([("ARG", [2010, 2011], "02-10", CLUBS[:8], None, False)])
    late = f.index[f["season"] == 2010][-1]
    club = f.loc[late, "home"]
    f.loc[late, "date"] = "2011-04-01"                 # played after 2011 began
    f = f.sort_values("date").reset_index(drop=True)
    feats = sr.rate(f)
    after = f.index[(f["date"] > "2011-04-01") & (f["season"] == 2011)
                    & ((f["home"] == club) | (f["away"] == club))][0]
    before = ((f["date"] < f.loc[after, "date"]) & (f["season"] == 2011)
              & ((f["home"] == club) | (f["away"] == club))).sum()
    side = "h" if f.loc[after, "home"] == club else "a"
    # this season's matches so far, the late 2010 one among them - never reset to none
    assert feats.loc[after, f"gp_{side}"] == before + 1


def test_a_club_past_a_double_round_robin_has_nothing_at_stake():
    teams = CLUBS[:8]
    f = _frame([("SC0", [2010], "08-01", teams, "pre", True)])
    last = pd.Timestamp(f["date"].max())
    extra = []
    for k, (h, a) in enumerate([(teams[0], teams[1]), (teams[2], teams[3])]):
        day = (last + pd.Timedelta(days=7 * (k + 1))).strftime("%Y-%m-%d")
        extra.append(dict(f.iloc[0], match_id=f"split{k}", date=day, home=h, away=a))
    f = pd.concat([f, pd.DataFrame(extra)], ignore_index=True)
    feats = sr.rate(f)
    split = f["match_id"].str.startswith("split")
    assert (feats.loc[split, ["up_h", "rel_h", "dead_h", "up_a", "rel_a", "dead_a"]] == 0).all().all()
    assert feats.loc[split, "rem_h"].isna().all()
    last_round = f.index[f["date"] == f.loc[~split, "date"].max()]
    assert feats.loc[last_round, "rem_h"].notna().all()


# ------------------------------------------------------------- what a card withholds

def test_a_card_withholds_exactly_the_numbers_its_league_did_not_ship():
    mat = ssc.dc_matrix(1.5, 1.1, -0.05)[0]
    full = ssc.card(mat, "model", ssc.NO_PRICE)
    assert full == ssc.card(mat, "model", ssc.NO_PRICE, withhold=()) and "withheld" not in full
    part = ssc.card(mat, "model", ssc.NO_PRICE, withhold=("btts", "exact"))
    assert part["btts"] is None and part["scores"] is None and part["other"] is None
    assert part["withheld"] == ["btts", "exact"]
    for k in ("o15", "o25", "o35", "matrix", "source", "reason"):
        assert part[k] == full[k]
    with pytest.raises(ValueError):
        ssc.card(mat, "model", ssc.NO_PRICE, withhold=("corners",))


# ------------------------------------------------------------- the rules

def test_the_rules_on_made_up_numbers():
    assert sl.rule_current(45, 0) and not sl.rule_current(46, 0) and sl.rule_current(None, 3)
    assert not sl.rule_current(None, 0)
    assert sl.rule_pick(0.99, 0.99) == "recalibrated" and sl.rule_pick(0.991, 0.99) == "raw"
    assert sl.rule_pick(None, 0.99) == "raw"
    good, bad = {"model": 1.0, "base": 1.05}, {"model": 1.06, "base": 1.05}
    assert sl.rule_model_alone(good, good) and not sl.rule_model_alone(good, bad)
    assert not sl.rule_model_alone(bad, good) and not sl.rule_model_alone({}, good)
    win, lose = {"log_loss": 0.6, "base": 0.62}, {"log_loss": 0.63, "base": 0.62}
    dev = {"btts": win, "o15": win, "o25": lose, "o35": win, "exact": win}
    conf = {"btts": win, "o15": lose, "o25": win, "o35": win, "exact": win}
    assert sl.rule_numbers(dev, conf) == ["btts", "o35", "exact"]
    assert sl.rule_numbers(dev, {}) == []


def test_a_current_season_is_a_recent_result_or_a_posted_fixture():
    f = _frame([("NOR", [2026], "03-20", CLUBS[:6], "close", False), ("RUS", [2026], "07-24", CLUBS[6:12], "close", False)])
    f = f[(f["league"] == "NOR") | (f["date"] <= "2026-08-02")]
    none = f.iloc[:0]
    nor = sl.season_check(f, none, "NOR", today=f.loc[f["league"] == "NOR", "date"].max())
    assert nor["ok"] and nor["days_since"] == 0
    rus = sl.season_check(f, none, "RUS", today="2026-10-30")
    assert not rus["ok"] and rus["last_result"] <= "2026-08-02" and rus["days_since"] >= 89
    posted = f[f["league"] == "RUS"].tail(1).assign(date="2026-10-31", hg=np.nan, ag=np.nan, result=None)
    assert sl.season_check(f, posted, "RUS", today="2026-10-30")["ok"]
    stale = posted.assign(date="2026-09-01")                      # a fixture left over from before: not posted now
    assert not sl.season_check(f, stale, "RUS", today="2026-10-30")["ok"]
    assert "RUS" not in sl.in_app() and "RUS" not in sl.in_app(f, none, "2026-10-30")
    assert "RUS" in sl.in_app(f, posted, "2026-10-30")


# ------------------------------------------------------------- the decisions as shipped

def test_every_shipped_decision_follows_from_the_numbers_stored_beside_it():
    assert set(sl.SHIPPED) == set(sd.ALL_LEAGUES)
    assert sl.ENGLAND_POOL in ("all", "top-two") and sl.PICK_WEIGHTS in smod.WEIGHT_OPTIONS
    for code, r in sl.SHIPPED.items():
        m = r["measured"]
        assert r["in_app"] == sl.rule_current(m["season_check"]["days_since"], m["season_check"]["fixtures_posted"])
        assert r["pick"] == sl.rule_pick(m["dev"]["pick_recalibrated"], m["dev"]["market"])
        assert r["model_alone"] == sl.rule_model_alone(m["dev"], m["confirm"])
        for source in sl.SOURCES:
            got = r["numbers"][source]
            if code in sd.EXTRA and source == "market":
                assert got is None                     # no over/under price: never the market's matrix
                assert sl.withheld(code, source) == sl.NUMBERS
                continue
            assert got == sl.rule_numbers(m["scores"]["dev"][source], m["scores"]["confirm"][source])
            assert sl.withheld(code, source) == tuple(k for k in sl.NUMBERS if k not in got)
        assert sl.group_of(code) == r["group"]
        if code in sd.EXTRA:
            assert any("closing" in c for c in r["caveats"])
    assert sl.raw_leagues() == [c for c in sl.in_app() if sl.SHIPPED[c]["pick"] == "raw"]
    assert sl.model_alone() == [c for c in sl.in_app() if sl.SHIPPED[c]["model_alone"]]


def test_the_band_rates_ship_for_every_group_in_the_app():
    for code in sl.in_app():
        for basis in ("pick", "model"):
            for band, _ in ssc.BANDS:
                r = sl.band_rate(code, band, basis)
                assert r is not None and r["picks"] > 0 and 0 < r["hit_rate"] < 1
