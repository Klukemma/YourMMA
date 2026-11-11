"""Every league's production pieces, on made-up leagues: the price each
league is picked on, the pick (recalibrated on matches dated before the
league-season only, the raw price where its recalibration lost, the model's
own or none where no price), the league's base rates walking forward on
dates, the model over leagues of different calendars, what a card withholds
and the over 2.5 beside it, the Record tab and the Method tab per league on
its own periods and price - and the per-league decisions, held to the rules
that made them."""

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
    assert top.sum() >= 900                            # five seasons of two leagues recalibrated
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


def test_a_groups_regression_is_fitted_on_every_league_of_the_group_in_the_frame():
    # which leagues get numbers never changes the numbers they get: the extra
    # leagues' regression learns from RUS's matches while RUS is not shown
    f = _frame([("NOR", range(2005, 2011), "03-20", CLUBS[:8], "close", False),
                ("SWE", range(2005, 2011), "04-01", CLUBS[8:16], "close", False)])
    both = sr.predict_leagues(f, ["NOR", "SWE"], score_params=True, **SMALL)
    one = sr.predict_leagues(f, ["NOR"], score_params=True, **SMALL)
    nor, swe = f["league"] == "NOR", f["league"] == "SWE"
    cols = ["p_h", "p_d", "p_a", "p_o25", "lam", "nu", "rho"]
    assert both.loc[nor, "p_h"].notna().sum() > 100
    assert np.array_equal(both.loc[nor, cols].to_numpy(), one.loc[nor, cols].to_numpy(), equal_nan=True)
    assert one.loc[swe, cols].isna().all().all()
    alone = sr.predict_leagues(f[nor], ["NOR"], score_params=True, **SMALL)
    assert not np.array_equal(alone[cols].to_numpy(), one.loc[nor, cols].to_numpy(), equal_nan=True)


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


def _with_play_offs(f, code):
    """Each season of `code` followed by a two-legged play-off of its last
    club against one from outside the league, filed under the same season
    after its regular season (as the extra leagues' files have them)."""
    extra = []
    for s, g in f[f["league"] == code].groupby("season"):
        last = pd.Timestamp(g["date"].max())
        bottom = g["home"].iloc[0]
        for k, (h, a) in enumerate([(f"Outsider {s}", bottom), (bottom, f"Outsider {s}")]):
            day = (last + pd.Timedelta(days=4 * (k + 1))).strftime("%Y-%m-%d")
            extra.append(dict(g.iloc[0], match_id=f"po{s}{k}", date=day, home=h, away=a,
                              kickoff=pd.Timestamp(day) + pd.Timedelta(hours=15)))
    return pd.concat([f, pd.DataFrame(extra)]).sort_values(["date", "league", "home"]).reset_index(drop=True)


def test_a_play_off_club_filed_after_the_season_is_unknown_to_the_earlier_matches():
    # FIN 2026: a play-off side with two matches in Oct / Nov was in the roster
    # of every match of the season, so the late-season stakes (games left, the
    # drop and place boundaries of a 13-club division) moved the predictions of
    # July-September with rows that did not exist then. Cut before the play-off,
    # nothing earlier may move; the play-off club reads nothing at stake.
    f = _with_play_offs(_frame([("FIN", range(2005, 2012), "04-10", CLUBS[:12], "close", False),
                                ("NOR", range(2005, 2012), "03-20", CLUBS[:10], "close", False)]), "FIN")
    po = f["match_id"].str.startswith("po")
    cut = f.loc[po & (f["season"] == 2011), "date"].min()
    before = f["date"] < cut
    feats, upto_feats = sr.rate(f), sr.rate(f[before])
    cols = ["up_h", "up_a", "rel_h", "rel_a", "dead_h", "dead_a", "rem_h", "rem_a", "eta_h_0", "eta_a_0"]
    assert np.array_equal(feats.loc[before, cols].to_numpy(), upto_feats[cols].to_numpy(), equal_nan=True)
    assert (feats.loc[po, ["up_h", "rel_h", "dead_h", "up_a", "rel_a", "dead_a"]] == 0).all().all()
    late = (f["league"] == "FIN") & (f["season"] == 2011) & ~po
    assert (feats.loc[late, "up_h"] > 0).any()                 # the stakes are read, on the 12 clubs
    full = sr.predict_leagues(f, ["FIN", "NOR"], **SMALL)
    upto = sr.predict_leagues(f[before], ["FIN", "NOR"], **SMALL)
    a = full.loc[before, ["p_h", "p_d", "p_a", "p_o25"]].to_numpy()
    assert np.isfinite(a[late[before].to_numpy()]).all()
    assert np.array_equal(a, upto[["p_h", "p_d", "p_a", "p_o25"]].to_numpy(), equal_nan=True)
    # A second spelling that reaches 5 matches only later (NOR 2023's HamKam):
    # cut at its 4th, it is from outside on the day - and so, read on the day,
    # it is in the full frame too. Read from the whole season it was in the
    # roster all along (a 13-club division from the first day).
    f = _frame([("FIN", range(2005, 2012), "04-10", CLUBS[:12], "close", False),
                ("NOR", range(2005, 2012), "03-20", CLUBS[:10], "close", False)])
    mine = f.index[(f["league"] == "FIN") & (f["season"] == 2011)
                   & ((f["home"] == CLUBS[3]) | (f["away"] == CLUBS[3]))][-6:]
    for side in ("home", "away"):
        f.loc[mine, side] = f.loc[mine, side].replace(CLUBS[3], "Club 3 FC")
    before = f["date"] < f.loc[mine[3], "date"]
    full = sr.predict_leagues(f, ["FIN", "NOR"], **SMALL)
    upto = sr.predict_leagues(f[before], ["FIN", "NOR"], **SMALL)
    assert np.array_equal(full.loc[before, ["p_h", "p_d", "p_a", "p_o25"]].to_numpy(),
                          upto[["p_h", "p_d", "p_a", "p_o25"]].to_numpy(), equal_nan=True)


# ------------------------------------------------------------- the pipeline reads the decisions

def test_the_pipeline_calls_read_the_measured_decisions(world, monkeypatch):
    seen = {}

    def spy(frame, leagues, score_params=False, england="all", **params):
        seen["england"], seen["leagues"] = england, leagues
        return "preds"
    monkeypatch.setattr(sr, "predict_leagues", spy)
    shipped = {c: dict(in_app=c != "SP1", pick="raw" if c == "B1" else "recalibrated", model_alone=c != "E0")
               for c in ("E0", "SP1", "B1", "NOR")}
    monkeypatch.setattr(sl, "SHIPPED", shipped)           # the world's four leagues, as if all measured
    for pool in ("all", "top-two"):
        monkeypatch.setattr(sl, "ENGLAND_POOL", pool)
        assert smod.predict(world, leagues=sl.MODEL_LEAGUES) == "preds" and seen["england"] == pool
    assert smod.predict(world, leagues=["E0"], england="all") == "preds" and seen["england"] == "all"
    monkeypatch.setattr(sl, "ENGLAND_POOL", None)
    with pytest.raises(ValueError):
        smod.predict(world, leagues=["E0"])
    monkeypatch.undo()

    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    preds = _preds(world)
    monkeypatch.setattr(sl, "SHIPPED", shipped)
    monkeypatch.setattr(sl, "PICK_WEIGHTS", "per_price")
    got, w_got = smod.shipped_pick(world, preds)
    want, w_want = smod.pick_leagues(world, preds, ["E0", "B1", "NOR"], "per_price", raw=["B1"],
                                     model_alone=["B1", "NOR"])
    assert sl.in_app() == ["E0", "B1", "NOR"] and sl.raw_leagues() == ["B1"] and sl.model_alone() == ["B1", "NOR"]
    assert got.equals(want) and w_got == w_want
    assert got.loc[world["league"] == "SP1", "p_h"].isna().all()           # not in the app: nothing
    # a league back in the data (RUS's file resuming) is picked by its own decisions
    every, _ = smod.shipped_pick(world, preds, ["E0", "SP1", "B1", "NOR"])
    want, _ = smod.pick_leagues(world, preds, ["E0", "SP1", "B1", "NOR"], "per_price", raw=["B1"],
                                model_alone=["SP1", "B1", "NOR"])
    assert every.equals(want)
    with pytest.raises(ValueError):
        smod.shipped_pick(world, preds, ["E0", "D1"])                    # D1 not measured here
    monkeypatch.setattr(sl, "PICK_WEIGHTS", None)
    with pytest.raises(ValueError):
        smod.shipped_pick(world, preds)


def test_the_model_and_the_pick_refuse_a_frame_without_every_measured_league(world, monkeypatch):
    # a league left out of the frame moves the numbers of the leagues it trains
    # with (RUS's matches: the extra leagues' regression and closing-price weights)
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    preds = _preds(world)
    shipped = {c: dict(in_app=c != "SP1", pick="recalibrated", model_alone=True) for c in ("E0", "SP1", "B1", "NOR")}
    monkeypatch.setattr(sl, "SHIPPED", shipped)
    monkeypatch.setattr(sl, "PICK_WEIGHTS", "per_price")
    smod.require_every_league(world)
    without = world[world["league"] != "SP1"]                # SP1 is not in the app, but it was measured
    for call in (lambda: smod.shipped_pick(without, preds.loc[without.index]),
                 lambda: smod.predict(without, leagues=["E0", "B1", "NOR"])):
        with pytest.raises(ValueError, match="SP1"):
            call()
    # and it is not a formality: the weights pooled over the European divisions move
    _, w_all = smod.shipped_pick(world, preds)
    _, w_without = smod.pick_leagues(without, preds.loc[without.index], ["E0", "B1", "NOR"], "per_price",
                                     model_alone=["E0", "B1", "NOR"])
    assert w_all[("E0", 2010)] != w_without[("E0", 2010)]
    assert w_all[("NOR", 2010)] == w_without[("NOR", 2010)]          # the closing-price pool has no SP1
    monkeypatch.undo()
    with pytest.raises(ValueError):
        smod.shipped_pick(world, preds)                     # the real decisions: 38 leagues, the world has 4


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


# ------------------------------------------------------------- the experiment's own machinery

def _small_experiment(monkeypatch):
    """experiments/leagues.py on the made-up world: shorter periods, smaller
    minimums (the procedure itself unchanged)."""
    from experiments import leagues as L
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    for k, v in SMALL.items():
        monkeypatch.setitem(sr.DEFAULTS, k, v)
    monkeypatch.setattr(bt, "DEV", (2007, 2009))
    monkeypatch.setattr(bt, "CONFIRM", (2010, 2011))
    monkeypatch.setattr(bt, "EXTRA_DEV", (2007, 2009))
    monkeypatch.setattr(bt, "EXTRA_CONFIRM", (2010, 2011))
    return L


def _shared_day(world, year):
    days = world.groupby("date")["league"].nunique()
    return days[(days >= 2) & days.index.str.startswith(year)].index[0]


def test_the_experiments_point_in_time_check_passes_and_bites(world, monkeypatch):
    L = _small_experiment(monkeypatch)
    cuts = (_shared_day(world, "2008"), _shared_day(world, "2010"))
    out = L.run(pit=True, log=lambda *a: None, frame=world, fixtures=world.iloc[:0], cuts=cuts, timing=False)
    for r in out["point_in_time"]:
        assert r["rows_on_cut_day"] > 0 and r["numbers_compared"] > 10000
        assert r["unmoved_on_or_before_cut"] and r["unmoved_before_cut_prices_from_cut_scrambled"]
        assert r["unmoved_before_cut_rows_from_cut_dropped"]
    # every decision is its rule on the numbers beside it, and the block the module ships is that
    for code in ("E0", "SP1", "B1", "NOR"):
        r, d = out["leagues"][code], out["decisions"][code]
        assert r["dev"]["matches"] > 0 and r["confirm"]["matches"] > 0
        assert d["pick"] == sl.rule_pick(r["dev"]["pick_recalibrated"], r["dev"]["market"])
        assert d["model_only"] == sl.rule_model_alone(r["dev"], r["confirm"])
        assert not d["in_app"]                                # made-up seasons long over: no current season
    assert out["decisions"]["NOR"]["numbers"]["market"] is None
    assert not out["decisions"]["D1"]["in_app"]               # not in the frame: not measured, never shipped
    import json
    ns = {}
    exec(L.shipped_block(json.loads(json.dumps(out, default=lambda x: x.item() if hasattr(x, "item") else str(x)))), ns)
    assert set(ns["SHIPPED"]) == set(sd.ALL_LEAGUES) and ns["PICK_WEIGHTS"] == out["protocol"]["weights"]
    assert ns["SHIPPED"]["E0"]["measured"]["dev"]["pick"] == out["leagues"]["E0"]["dev"]["pick"]

    # a pick that peeks at results to come is caught
    honest = smod.pick_leagues

    def peeking(frame, preds, leagues, option="top5", **kw):
        out, w = honest(frame, preds, leagues, option, **kw)
        out[["p_h", "p_d", "p_a"]] *= 1 + 1e-9 * (frame["result"] == "H").mean()
        return out, w
    monkeypatch.setattr(smod, "pick_leagues", peeking)
    ref = sr.predict_leagues(world, list(sd.ALL_LEAGUES), score_params=True, england="top-two")
    caught = L.point_in_time(world, cuts[0], "top-two", out["protocol"]["weights"], [], reference=ref)
    assert not caught["unmoved_on_or_before_cut"]


def test_the_experiments_point_in_time_check_catches_a_roster_that_knows_later_clubs(world, monkeypatch):
    # Scrambling results and prices leaves every row in place, so it cannot see
    # a fact read from the rows themselves; the run on the frame cut at the cut
    # day can. A model whose stakes read the clubs of the whole league-season
    # (as the roster did until 2026-11-11) is caught by that run alone.
    L = _small_experiment(monkeypatch)
    f = _with_play_offs(world, "NOR")
    cut = f.loc[f["match_id"].str.startswith("po") & (f["season"] == 2011), "date"].min()
    honest = L.point_in_time(f, cut, "top-two", "per_price", [])
    assert honest["unmoved_before_cut_rows_from_cut_dropped"] and honest["unmoved_on_or_before_cut"]
    rate = sr.rate

    def whole_season(matches, **params):
        out = rate(matches, **params)
        clubs = pd.concat([matches[["league", "season", "home"]].set_axis(["l", "s", "c"], axis=1),
                           matches[["league", "season", "away"]].set_axis(["l", "s", "c"], axis=1)])
        n = clubs.groupby(["l", "s"])["c"].nunique()
        out["up_h"] = out["up_h"] + 1e-6 * n.reindex(pd.MultiIndex.from_frame(matches[["league", "season"]])).to_numpy()
        return out
    monkeypatch.setattr(sr, "rate", whole_season)
    caught = L.point_in_time(f, cut, "top-two", "per_price", [])
    assert caught["unmoved_on_or_before_cut"] and caught["unmoved_before_cut_prices_from_cut_scrambled"]
    assert not caught["unmoved_before_cut_rows_from_cut_dropped"]


# ------------------------------------------------------------- the decisions as shipped

def test_every_shipped_decision_follows_from_the_numbers_stored_beside_it():
    assert set(sl.SHIPPED) == set(sd.ALL_LEAGUES)
    assert sl.ENGLAND_POOL in ("all", "top-two") and sl.PICK_WEIGHTS in smod.WEIGHT_OPTIONS
    for code, r in sl.SHIPPED.items():
        m = r["measured"]
        assert r["in_app"] == (sl.rule_current(m["season_check"]["days_since"], m["season_check"]["fixtures_posted"])
                               and m["dev"]["matches"] > 0 and m["confirm"]["matches"] > 0)
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


# ------------------------------------------------------------- goals: the over 2.5 beside the card

def _with_numbers(monkeypatch, **numbers):
    """sl.SHIPPED with `numbers` = {code: {source: [names]}} in place of the
    measured ones (every other decision as shipped)."""
    import copy
    shipped = copy.deepcopy(sl.SHIPPED)
    for code, by_source in numbers.items():
        shipped[code]["numbers"].update(by_source)
    monkeypatch.setattr(sl, "SHIPPED", shipped)


def test_goals_is_the_over_under_price_or_the_models_and_never_a_number_its_league_withheld(world, monkeypatch):
    f = world.copy()
    e0 = f.index[f["league"] == "E0"]
    f.loc[e0[:12], ["avg_o25", "avg_u25"]] = np.nan          # twelve E0 matches with no over/under posted
    preds = _preds(f)
    preds["p_o25"] = np.random.default_rng(3).uniform(0.3, 0.7, len(f))
    q = bt.no_vig_total(f)
    posted = q.notna()
    got = smod.goals(f, preds, ["E0", "B1", "NOR"])
    for lg in ("E0", "B1"):
        rows = (f["league"] == lg) & posted
        assert rows.sum() > 100 and np.array_equal(got[rows].to_numpy(), q[rows].to_numpy())
    model_rows = (f["league"] == "E0") & ~posted
    assert model_rows.sum() == 12 and np.array_equal(got[model_rows].to_numpy(), preds.loc[model_rows, "p_o25"].to_numpy())
    assert got[f["league"] == "SP1"].isna().all()               # not asked for: nothing, not the model's
    # NOR has no over/under price, and its model's over 2.5 did not beat its base rate in both periods
    nor = f["league"] == "NOR"
    assert "o25" in sl.withheld("NOR", "model") and not posted[nor].any() and got[nor].isna().all()

    # where the model's over 2.5 ships, the extra league shows the model's own
    _with_numbers(monkeypatch, NOR={"model": ["o15", "o25"]})
    got = smod.goals(f, preds, ["E0", "B1", "NOR"])
    assert np.array_equal(got[nor].to_numpy(), preds.loc[nor, "p_o25"].to_numpy())
    # where the market's over 2.5 is withheld (SC2's on the run of 2026-11-02), the posted
    # price is not shown - and the model's is not put in its place
    _with_numbers(monkeypatch, B1={"market": ["btts", "o15"]})
    got = smod.goals(f, preds, ["E0", "B1", "NOR"])
    b1 = f["league"] == "B1"
    assert posted[b1].all() and got[b1].isna().all()
    assert np.array_equal(got[(f["league"] == "E0") & posted].to_numpy(), q[(f["league"] == "E0") & posted].to_numpy())
    monkeypatch.undo()

    # the default is the top five, every one shipping over 2.5 from both sources: as it was
    for code in sd.TOP:
        assert "o25" not in sl.withheld(code, "market") and "o25" not in sl.withheld(code, "model")
    old = preds["p_o25"].copy()
    top = f["league"].isin(sd.TOP)
    old[posted & top] = q[posted & top]
    got = smod.goals(f, preds)
    assert np.array_equal(got[top].to_numpy(), old[top].to_numpy(), equal_nan=True)
    assert got[~top].isna().all()


# ------------------------------------------------------------- the Record tab, every league

def _short_periods(monkeypatch):
    """The world's seasons run 2005-2011: periods that fit, the extra
    leagues' different from the European divisions'."""
    monkeypatch.setattr(bt, "DEV", (2007, 2008))
    monkeypatch.setattr(bt, "CONFIRM", (2009, 2011))
    monkeypatch.setattr(bt, "EXTRA_DEV", (2008, 2009))
    monkeypatch.setattr(bt, "EXTRA_CONFIRM", (2010, 2011))


def test_each_league_is_scored_on_its_own_price_and_the_default_is_the_top_five(world):
    assert bt.scored(world).equals(world["league"].isin(sd.TOP) & bt.no_vig(world, "pre").notna().all(axis=1)
                                   & world["result"].isin(bt.RESULTS))
    every = bt.scored(world, ["E0", "SP1", "B1", "NOR"])
    nor = world["league"] == "NOR"
    assert every[nor].all() and not bt.scored(world)[nor].any()       # priced at the close only
    assert every.equals(world["result"].isin(bt.RESULTS))
    assert smod.market_price is bt.market_price


def test_the_record_is_the_measurement_that_decided_each_league(world, monkeypatch):
    # record() on the matches experiments/leagues.py scored is its league_rows
    # and group_rows, number for number: the Record tab shows what shipped
    from experiments import leagues as L
    _short_periods(monkeypatch)
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    leagues = ["E0", "SP1", "B1", "NOR"]
    preds = _preds(world)
    preds.loc[preds.index[::17], ["p_h", "p_d", "p_a"]] = np.nan        # the model missing on some matches
    pick, _ = smod.pick_leagues(world, preds, leagues, "per_price", raw=["B1"])
    base = ssc.baseline(world, leagues)
    rec = bt.record(world, pick, preds, leagues)
    q, kind = smod.market_price(world)
    want = L.league_rows(world, q, kind, preds, base, pick, pick, {c: None for c in L.ALL})
    names = {"matches": "matches", "log_loss": "pick", "market_log_loss": "market", "model_log_loss": "model",
             "base_log_loss": "base", "accuracy": "acc_pick", "market_accuracy": "acc_market",
             "model_accuracy": "acc_model", "seasons": "seasons"}
    assert [r["code"] for r in rec["by_league"]] == leagues
    for r in rec["by_league"]:
        w = want[r["code"]]
        assert (r["name"], r["country"], r["group"], r["price"]) == (w["name"], w["country"], w["group"], w["price"])
        for key in ("dev", "confirm"):
            assert r[key]["matches"] > 50
            assert {a: r[key][a] for a in names} == {a: w[key][b] for a, b in names.items()}
            assert r["periods"][key] == w[key]["seasons"]
    assert rec["by_league"][3]["periods"] == {"dev": [2008, 2009], "confirm": [2010, 2011]}    # NOR: its own
    groups = L.group_rows(world, q, preds, base, pick, {c: {"in_app": c in leagues} for c in L.ALL})
    for g, r in rec["by_group"].items():
        for key in ("dev", "confirm"):
            w = groups[g][key]
            assert r[key]["matches"] == w["matches"] and r[key]["log_loss"] == w["pick"]
            assert (r[key]["market_log_loss"], r[key]["model_log_loss"], r[key]["base_log_loss"]) == \
                (w["market"], w["model"], w["base"])
            assert (r[key]["accuracy"], r[key]["market_accuracy"], r[key]["model_accuracy"]) == \
                (w["acc_pick"], w["acc_market"], w["acc_model"])
    assert list(rec["by_group"]) == ["top", "europe", "extra"]
    assert rec["by_group"]["top"]["codes"] == ["E0", "SP1"] and rec["by_group"]["extra"]["periods"]["dev"] == [2008, 2009]
    for key in ("dev", "confirm"):           # pooled: every league on its own periods, each match once
        assert rec["pooled"][key]["matches"] == sum(r[key]["matches"] for r in rec["by_league"])
    # NOR's 2007 is in the European divisions' DEV but not in its own: never scored
    nor07 = (world["league"] == "NOR") & (world["season"] == 2007)
    moved = world.copy()
    moved.loc[nor07, "result"] = "D"
    again = bt.record(moved, pick, preds, leagues, base=base)
    assert again == rec


def test_a_league_with_nothing_to_score_in_a_period_shows_nothing(world, monkeypatch):
    _short_periods(monkeypatch)
    monkeypatch.setattr(bt, "EXTRA_CONFIRM", (2030, 2031))
    preds = _preds(world)
    rec = bt.record(world, preds, preds, ["E0", "NOR"])
    nor = rec["by_league"][1]
    assert nor["confirm"] == {"seasons": [2030, 2031], "matches": 0} and nor["dev"]["matches"] > 0
    # a pick with no number where the league was priced is a caller's mistake, never a NaN on the tab
    partial, _ = smod.pick_leagues(world, preds, ["E0"], "per_price")
    with pytest.raises(ValueError, match="NOR"):
        bt.record(world, partial, preds, ["E0", "NOR"])


def test_report_scores_every_league_on_its_own_periods_and_price(world, monkeypatch):
    _short_periods(monkeypatch)
    preds = _preds(world)
    preds["p_o25"] = 0.5
    leagues = ["E0", "SP1", "B1", "NOR"]
    rep = bt.report(world, preds, "model", leagues=leagues)["model"]
    q, _ = bt.market_price(world)
    for code in leagues:
        for k, key in enumerate(("dev", "confirm")):
            lo, hi = bt.periods_for(code)[k]
            sel = (world["league"] == code) & world["season"].between(lo, hi)
            got = rep["by_league"][code][key]
            assert got["matches"] == sel.sum()
            assert got["market_log_loss"] == bt.log_loss(q[sel], world.loc[sel, "result"])
            assert got["log_loss"] == bt.log_loss(preds[sel], world.loc[sel, "result"])
            assert ("over_2_5" in got) == (code != "NOR")            # no over/under price in an extra league
    assert rep["by_league"]["NOR"]["price"] == "closing" and rep["by_league"]["B1"]["price"] == "pre-match"
    assert set(rep["by_group"]) == {"top", "europe", "extra"}
    assert rep["dev"]["matches"] == sum(rep["by_league"][c]["dev"]["matches"] for c in leagues)
    # without leagues: the top five, on the periods given, as before
    old = bt.report(world, preds, "model", periods=((2007, 2008),))["model"]["2007-2008"]
    sel = world["league"].isin(sd.TOP) & world["season"].between(2007, 2008)
    pre = bt.no_vig(world, "pre")
    assert old["matches"] == sel.sum() and old["market_log_loss"] == bt.log_loss(pre[sel], world.loc[sel, "result"])
    assert old["closing"]["matches"] == sel.sum() and "by_league" not in old


def test_the_latest_weights_are_each_leagues_newest_season(world, monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 150)
    preds = _preds(world)
    _, w = smod.pick_leagues(world, preds, ["E0", "B1", "NOR"], "per_price", raw=["B1"])
    got = smod.latest_weights(w)
    assert list(got) == ["E0", "B1", "NOR"] and got["B1"] is None          # the raw price: no weights
    for c in ("E0", "NOR"):
        assert got[c] == {"season": 2011, "w": [round(v, 4) for v in w[(c, 2011)]]}
    assert smod.latest_weights(w, ["NOR", "SP1"]) == {"NOR": got["NOR"], "SP1": None}


@pytest.mark.snapshot
def test_the_shipped_weights_are_what_the_snapshot_fits():
    # A snapshot check: the pick's weights need no model, and every in-app
    # league's newest season reproduces what soccer_leagues stores beside it.
    frame = sd.load_matches(leagues="all")
    preds = pd.DataFrame(np.nan, index=frame.index, columns=["p_h", "p_d", "p_a"])
    _, w = smod.shipped_pick(frame, preds)
    got = smod.latest_weights(w, sl.in_app())
    assert got == {c: sl.SHIPPED[c]["weights"] for c in sl.in_app()}


# ------------------------------------------------------------- the cards and the Method tab, every league

@pytest.fixture(scope="module")
def modelled(world):
    """The model over the world's four leagues, with its score parameters,
    and the pick on its prices (weights pooled per price type)."""
    preds = sr.predict_leagues(world, ["E0", "SP1", "B1", "NOR"], score_params=True, england="top-two", **SMALL)
    keep = smod.MIN_TRAIN
    smod.MIN_TRAIN = 150
    try:
        pick, _ = smod.pick_leagues(world, preds, ["E0", "SP1", "B1", "NOR"], "per_price", raw=["B1"])
    finally:
        smod.MIN_TRAIN = keep
    return preds, pick


def test_every_leagues_cards_come_from_the_source_it_has_and_withhold_what_it_did_not_ship(world, modelled):
    preds, pick = modelled
    idx, mats, info = ssc.walk(world, preds, pick, leagues=["B1", "NOR"])
    lg = world.loc[idx, "league"].to_numpy()
    assert set(lg) == {"B1", "NOR"} and len(idx) == int((world["league"].isin(["B1", "NOR"]) & preds["p_h"].notna()).sum())
    nor = lg == "NOR"
    # an extra league is priced 1X2 only (its history: the close): the model's matrix, said why
    assert (info["source"][nor] == ssc.SOURCE_MODEL).all() and (info["reason"][nor] == ssc.NO_OVER_UNDER).all()
    assert (info["source"][~nor] == ssc.SOURCE_MARKET).mean() > 0.9
    shown = {"B1": 0, "NOR": 0}
    for k in range(len(idx)):
        code, src = lg[k], info["source"].iloc[k]
        c = ssc.card(mats[k], src, info["reason"].iloc[k], info["miss"].iloc[k], info["miss_on"].iloc[k],
                     withhold=sl.withheld(code, src))
        if c is None:
            continue
        shown[code] += 1
        for name in sl.NUMBERS:
            gone = name in sl.withheld(code, src)
            assert (name in c.get("withheld", [])) == gone
            assert (c["scores"] is None if name == "exact" else c[name] is None) == gone
    assert shown["B1"] > 100 and shown["NOR"] > 100
    assert "o25" in sl.withheld("NOR", "model") and "btts" in sl.withheld("B1", "market")


def test_the_method_tab_measures_every_league_on_its_own_periods(world, modelled, monkeypatch):
    _short_periods(monkeypatch)
    preds, pick = modelled
    leagues = ["E0", "SP1", "B1", "NOR"]
    walked = ssc.walk(world, preds, pick, leagues=leagues)
    with pytest.raises(ValueError, match="different periods"):
        ssc.measure(world, preds, pick, walked)                     # the extra league's years are not the others'
    europe = ssc.measure(world, preds, pick, walked, leagues=["E0", "SP1", "B1"])
    top = ssc.measure(world, preds, pick, walked, leagues=["E0", "SP1"])
    assert list(europe["periods"]) == ["2007-2008", "2009-2011"]
    for key in europe["periods"]:
        assert europe["periods"][key]["matches"] > top["periods"][key]["matches"] > 50     # B1 is in, not dropped
    # a walk over the top five alone, with no leagues said: as it always was
    alone = ssc.walk(world, preds, pick)
    assert ssc.measure(world, preds, pick, alone) == ssc.measure(world, preds, pick, alone, periods=(bt.DEV, bt.CONFIRM),
                                                                 leagues=list(sd.TOP)) == top
    nor = ssc.measure(world, preds, pick, walked, leagues=["NOR"])
    assert list(nor["periods"]) == ["2008-2009", "2010-2011"]
    for key, r in nor["periods"].items():
        lo, hi = map(int, key.split("-"))
        n = int(((world["league"] == "NOR") & world["season"].between(lo, hi)).sum())
        assert 0 < r["matches"] <= n and r["market"] == 0 and r["model"][ssc.NO_OVER_UNDER] == r["matches"]
        assert sum(nor["bands"]["pick"][key][b]["picks"] for b, _ in ssc.BANDS) == n     # priced at the close
    groups = ssc.by_group(world, preds, pick, walked, leagues)
    assert list(groups) == ["top", "europe", "extra"]
    assert groups["top"] == top and groups["extra"] == nor
    assert groups["europe"] == ssc.measure(world, preds, pick, walked, leagues=["B1"])
