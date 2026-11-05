"""The gradient-boosted stack candidate (experiments/gbm_stack.py): its own
features read only earlier days, and with no trees it is the pick itself.
(The whole candidate's scramble test runs on the real history in the
module's own run: python experiments/gbm_stack.py.)"""

import math

import numpy as np
import pandas as pd

import soccer_data as sd
from experiments import gbm_stack as gs

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford", "Burnley", "Watford"]


def _double_round_robin():
    """Every pairing home and away, four matches a round, each club once a round."""
    n, fixed, rest = len(TEAMS), TEAMS[0], TEAMS[1:]
    first = []
    for r in range(n - 1):
        order = [fixed] + rest[r:] + rest[:r]
        first += [(order[i], order[n - 1 - i]) for i in range(n // 2)]
    return first + [(a, h) for h, a in first]


def _league(seasons=range(2005, 2009), seed=0):
    """A made-up top league: results follow strength, prices follow it too."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-0.4, 0.4, len(TEAMS))))
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-10")
        for k, (h, a) in enumerate(_double_round_robin()):
            date = day + pd.Timedelta(days=7 * (k // 4))          # four matches a day, a club once
            ph = 1 / (1 + math.exp(-(0.3 + 2 * (strength[h] - strength[a]))))
            p = np.array([0.75 * ph, 0.25, 0.75 * (1 - ph)])
            r = rng.choice(3, p=p / p.sum())
            hg, ag = [(2, 0), (1, 1), (0, 1)][r]
            rows.append({"match_id": sd.match_id("E0", date.date(), h, a), "league": "E0", "season": season,
                         "date": date.strftime("%Y-%m-%d"), "time": "15:00", "kickoff": date,
                         "home": h, "away": a, "hg": hg, "ag": ag, "result": "HDA"[r],
                         "hs": hg * 4 + 5, "as": ag * 4 + 4, "hst": hg + 2, "ast": ag + 1,
                         "avg_h": 0.95 / p[0], "avg_d": 0.95 / p[1], "avg_a": 0.95 / p[2],
                         "max_h": 1.0 / p[0], "max_d": 1.0 / p[1], "max_a": 1.0 / p[2]})
    return sd.frame(rows)


def _own_features(m):
    q, _ = gs._no_vig(m, ["avg_h", "avg_d", "avg_a"])
    f = gs._form(m, q)
    out = pd.DataFrame(f, index=m.index)
    out["elo_d"] = gs._elo(m)
    lb = gs._league_bias(m, q, half=40.0)
    out["lbias_h"], out["lbias_d"] = lb[:, 0], lb[:, 1]
    return out


def test_form_elo_and_league_bias_never_see_the_day_or_later():
    m = _league()
    cut = sorted(m["date"].unique())[30]          # mid-season, four matches that day
    later = m["date"] >= cut
    s = m.copy()
    rng = np.random.default_rng(3)
    for c in ("hg", "ag", "hs", "as", "hst", "ast"):
        s.loc[later, c] = rng.integers(0, 9, int(later.sum()))
    s.loc[later, "result"] = np.where(s.loc[later, "hg"] > s.loc[later, "ag"], "H",
                                      np.where(s.loc[later, "hg"] < s.loc[later, "ag"], "A", "D"))
    for c in ("avg_h", "avg_d", "avg_a"):
        s.loc[m["date"] > cut, c] = rng.uniform(1.2, 9.0, int((m["date"] > cut).sum()))
    a, b = _own_features(m), _own_features(s)
    upto = (m["date"] <= cut).to_numpy()
    assert np.allclose(a[upto].to_numpy(float), b[upto].to_numpy(float), equal_nan=True)
    assert not np.allclose(a[~upto].to_numpy(float), b[~upto].to_numpy(float), equal_nan=True)
    # the league's market miss is read once 50 matches' weight has been seen, and is NaN before
    assert a["lbias_h"].isna().iloc[0] and a["lbias_h"].notna().iloc[-1]


def test_form_reads_the_clubs_previous_matches():
    m = _league(seasons=[2005])
    f = pd.DataFrame(gs._form(m, gs._no_vig(m, ["avg_h", "avg_d", "avg_a"])[0]), index=m.index)
    club = "Arsenal"
    rows = m.index[(m["home"] == club) | (m["away"] == club)]
    gd = np.where(m.loc[rows, "home"] == club, m.loc[rows, "hg"] - m.loc[rows, "ag"], m.loc[rows, "ag"] - m.loc[rows, "hg"])
    k = rows[6]
    side = "h" if m.loc[k, "home"] == club else "a"
    assert math.isclose(f.loc[k, f"gd5_{side}"], gd[1:6].astype(float).mean())
    assert math.isnan(f.loc[rows[0], "gd5_h" if m.loc[rows[0], "home"] == club else "gd5_a"])


def test_with_no_trees_the_candidate_is_the_pick():
    rng = np.random.default_rng(0)
    m = _league(seasons=range(2005, 2012))
    q = pd.DataFrame(gs._no_vig(m, ["avg_h", "avg_d", "avg_a"])[0], index=m.index, columns=["p_h", "p_d", "p_a"])
    pick = q * np.array([1.02, 0.97, 1.0])
    pick = pick.div(pick.sum(axis=1), axis=0)
    preds = q.copy()
    preds.loc[m["season"] == 2005] = np.nan
    X = pd.DataFrame(rng.normal(size=(len(m), len(gs.FEATURE_SETS["all"]))), index=m.index,
                     columns=gs.FEATURE_SETS["all"])
    inp = dict(m=m, preds=preds, pick=pick, pick_ext=pick, X=X)
    old = gs.MIN_TRAIN
    gs.MIN_TRAIN = 50
    try:
        cfg = dict(gs.CHOSEN, rounds_by="fixed", fallback_rounds=0)
        p, used = gs.walk_forward(inp, cfg)
    finally:
        gs.MIN_TRAIN = old
    ok = np.isfinite(p[:, 0])
    assert set(used.values()) == {0}
    assert (m.loc[ok, "season"] >= 2007).all() and ok[(m["season"] >= 2007).to_numpy()].all()
    assert np.allclose(p[ok], pick.to_numpy()[ok], atol=1e-12)
