"""The NFL models: who wins, by how much, how many points, and the blend
with the market.

  winner   logistic regression on nfl_features.FEATURES -> P(home wins)
  margin   ridge regression on the same features -> expected home margin
  total    ridge regression on nfl_features.TOTAL_FEATURES -> expected points
  blend    logistic regression on [logit model, logit market, logit spread],
           fitted on the model's OUT-OF-SAMPLE predictions for earlier seasons,
           so it learns how far to trust the model next to the price and not
           how well the model fits games it was trained on. The spread enters
           beside the moneyline because the two prices are not the same
           number: the spread through a normal curve carries information the
           vig-free moneyline does not (measured in both periods)

Every one is fitted on whole seasons before the one it predicts - the same
rule nfl_backtest measures - so the number the app shows for a game this
week comes from exactly the procedure whose record the app also shows.
"""

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nfl_backtest as bt
import nfl_data as nd
import nfl_elo as ne
import nfl_features as nf

ELO = ne.EloParams(k=20.0, hfa=48.0, revert=1 / 3)
WINNER_C = 0.1
# An L1 (lasso) winner. scikit-learn 1.8 spells it l1_ratio=1 and deprecates
# penalty= (gone in 1.10); before 1.8 l1_ratio is ignored without
# penalty="l1", so the spelling follows the installed version.
_L1 = ({"l1_ratio": 1.0} if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
       else {"penalty": "l1"})
RIDGE_ALPHA = 10.0


def feature_frame(games=None, team_games=None, qb_games=None):
    games = nd.load_games() if games is None else games
    team_games = nd.load_team_games() if team_games is None else team_games
    qb_games = nd.load_qb_games() if qb_games is None else qb_games
    frame = nf.build(games, team_games, qb_games, ELO)
    frame["market_prob"] = bt.market_prob(frame)
    return frame


def _x(frame, cols):
    return frame[cols].astype(float).fillna(0.0).to_numpy()


def fit_winner(train):
    """L1, not L2: the features overlap (epa_diff is off_epa_diff minus
    def_epa_diff; rest_diff, byes and short weeks say the same thing), and a
    lasso keeps one of each and drops the noise terms where a ridge spreads
    weight over all of them. saga with a fixed seed is deterministic."""
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=WINNER_C, solver="saga", max_iter=10000, random_state=0, **_L1))
    model.fit(_x(train, nf.FEATURES), (train["result"] > 0).astype(int))
    return model


def fit_margin(train):
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(_x(train, nf.FEATURES), train["result"].astype(float))
    return model


def fit_total(train):
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(_x(train, nf.TOTAL_FEATURES), train["total"].astype(float))
    return model


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def spread_prob(spread):
    """The home side's chance by the closing spread through a normal curve."""
    return norm.cdf(np.asarray(spread, float) / bt.SPREAD_SD)


def _blend_x(model_p, market_p, spread_p):
    # No spread: the moneyline stands in for it, as the market price stands
    # in for a missing moneyline (bt.market_prob) - so the two columns are
    # the same number wherever only one price is known.
    spread_p = np.where(np.isfinite(spread_p), spread_p, market_p)
    return np.column_stack([_logit(model_p), _logit(market_p), _logit(spread_p)])


def fit_blend(model_p, market_p, y, spread_p=None):
    model_p, market_p = np.asarray(model_p, float), np.asarray(market_p, float)
    spread_p = market_p if spread_p is None else np.asarray(spread_p, float)
    ok = np.isfinite(model_p) & np.isfinite(market_p)
    blend = LogisticRegression(C=1.0, max_iter=1000)
    blend.fit(_blend_x(model_p[ok], market_p[ok], spread_p[ok]), np.asarray(y)[ok])
    return blend


def apply_blend(blend, model_p, market_p, spread_p=None):
    """The blend where a market price exists, the model alone where not."""
    model_p = np.asarray(model_p, float)
    market_p = np.asarray(market_p, float)
    spread_p = market_p if spread_p is None else np.asarray(spread_p, float)
    out = model_p.copy()
    ok = np.isfinite(model_p) & np.isfinite(market_p)
    if blend is not None and ok.any():
        out[ok] = blend.predict_proba(_blend_x(model_p[ok], market_p[ok], spread_p[ok]))[:, 1]
    return out


class SeasonModels:
    """Everything fitted to predict one season, from the seasons before it."""

    def __init__(self, winner, margin, total, blend, trained_on):
        self.winner, self.margin, self.total, self.blend = winner, margin, total, blend
        self.trained_on = trained_on

    def predict(self, rows, market_p=None, spread_p=None):
        """model_prob, margin, total_pred and blend_prob for `rows`. The blend
        reads `market_p` (home win probability) and `spread_p` (the same by
        the spread, spread_prob) where given, else the rows' own
        market_prob and spread_line. A caller that passes a price but no
        spread gets the price in both places."""
        if market_p is None:
            market_p = rows["market_prob"]
            if spread_p is None:
                spread_p = spread_prob(rows["spread_line"])
        model_p = self.winner.predict_proba(_x(rows, nf.FEATURES))[:, 1]
        return pd.DataFrame({
            "model_prob": model_p,
            "margin": self.margin.predict(_x(rows, nf.FEATURES)),
            "total_pred": self.total.predict(_x(rows, nf.TOTAL_FEATURES)),
            "blend_prob": apply_blend(self.blend, model_p, np.asarray(market_p, float),
                                      None if spread_p is None else np.asarray(spread_p, float)),
        }, index=rows.index)


def fit_season(frame, season, oos_model_prob):
    """The models that predict `season`: fitted on every decided game from
    TRAIN_FROM to the season before, the blend on earlier seasons'
    out-of-sample model probabilities (`oos_model_prob`, aligned to frame)."""
    ok = bt.decided(frame)
    before = frame["season"].between(bt.TRAIN_FROM, season - 1)
    train = frame[ok & before]
    played = frame[frame["played"] & before]
    history = ok & (frame["season"] < season) & oos_model_prob.notna() & frame["market_prob"].notna()
    blend = None
    if history.sum() >= 500:
        blend = fit_blend(oos_model_prob[history].to_numpy(float),
                          frame.loc[history, "market_prob"].to_numpy(float),
                          (frame.loc[history, "result"] > 0).astype(int).to_numpy(),
                          spread_prob(frame.loc[history, "spread_line"]))
    return SeasonModels(fit_winner(train), fit_margin(played), fit_total(played), blend, len(train))


def walk_forward(frame, start=bt.TRAIN_FROM + 1, end=None, keep_models=False):
    """Out-of-sample winner, margin and total predictions plus the blend,
    season by season, for every game from `start` (played or not).

    `start` defaults to the first season with a season before it to train
    on, not to DEV: the blend learns from these out-of-sample predictions,
    and started at DEV it has none for 2006 and too few for 2007, which
    were then reported as the raw model under the blend's name. The
    reports read DEV and CONFIRM only, so the earlier seasons are history
    for the blend and nothing else.

    With keep_models, also returns {season: SeasonModels} - the live week is
    predicted by the very object the backtest used for its season."""
    end = end or int(frame["season"].max())
    out = pd.DataFrame(index=frame.index, columns=["model_prob", "margin", "total_pred", "blend_prob"], dtype=float)
    models = {}
    for season in range(start, end + 1):
        test = frame["season"] == season
        if not test.any():
            continue
        fitted = fit_season(frame, season, out["model_prob"])
        out.loc[test] = fitted.predict(frame[test])
        if keep_models:
            models[season] = fitted
    return (out, models) if keep_models else out


def backtest(frame=None, preds=None):
    """Everything the app's Method and Record tabs report about the model."""
    frame = feature_frame() if frame is None else frame
    preds = walk_forward(frame) if preds is None else preds
    report = {}
    report.update(bt.winner_report(frame, frame["elo_prob"], "elo"))
    report.update(bt.winner_report(frame, preds["model_prob"].astype(float), "model"))
    report.update(bt.winner_report(frame, preds["blend_prob"].astype(float), "blend"))
    spread = bt.line_report(frame, preds["margin"].astype(float), frame["spread_line"], frame["result"])
    total = bt.line_report(frame, preds["total_pred"].astype(float), frame["total_line"], frame["total"])
    blend = preds["blend_prob"].astype(float)
    return {
        "winner": report,
        "tiers": bt.tier_report(frame, blend),
        "value": bt.value_gate(frame, preds["model_prob"].astype(float)),
        "spread": spread, "spread_enabled_at": bt.line_gate(spread),
        "total": total, "total_enabled_at": bt.line_gate(total),
        "calibration": bt.calibration(frame, blend),
        "by_season": bt.by_season(frame, blend),
        "periods": {"dev": list(bt.DEV), "confirm": list(bt.CONFIRM), "train_from": bt.TRAIN_FROM},
    }
