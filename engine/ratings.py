"""TrueSkill rating construction for the UFC dataset.

The dataset ships with pre-fight ratings (r_mu_pre, r_sigma_pre, r_mmr_pre and
the blue-corner equivalents) but not the code that produced them. This module
reconstructs that step so new fights pulled from Kaggle can be appended with
ratings on the same scale.

Two facts were established by measuring the shipped columns, not assumed:

1. mmr_pre == mu_pre - 3 * sigma_pre, exactly (slope 1.0, intercept 0.0,
   max error 0.0 across all 8,229 rows). MMR is the TrueSkill conservative
   rating, not a separate Elo.

2. The ratings were built with beta close to 5.0, NOT the 4.17 that
   predict_card.py uses for bayesian_win_prob. Fitting the update against
   13,725 consecutive-bout pairs puts beta at 5.0-5.1 and tau at 0.2-0.6,
   which reproduces 97% of updates within 0.05 of mu. At beta=4.17 the median
   error is 17x larger.

Exact bit-for-bit reproduction is not achievable from this file alone: fights
on the same date have no recorded within-event order, and the "dedup" step in
the filename appears to have happened after the ratings were computed, so some
bouts that shaped them are no longer present. Hence extend() below, which
continues from each fighter's last known rating rather than recomputing
history and silently shifting every value the model was trained on.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class TrueSkillConfig:
    """Parameters for the rating update.

    Defaults are the empirical fit against the shipped columns, not the
    TrueSkill library defaults. See the module docstring.
    """
    mu0: float = 25.0
    sigma0: float = 8.333333
    beta: float = 5.0
    tau: float = 0.2
    min_sigma: float = 0.5

    @property
    def mmr0(self):
        return self.mu0 - 3 * self.sigma0


DEFAULT = TrueSkillConfig()


def conservative_rating(mu, sigma):
    """The MMR the dataset stores: mu - 3*sigma."""
    return mu - 3 * sigma


def rate_1v1(win_mu, win_sigma, lose_mu, lose_sigma, cfg=DEFAULT):
    """One TrueSkill update. Returns (win_mu, win_sigma, lose_mu, lose_sigma)."""
    w2 = win_sigma ** 2 + cfg.tau ** 2
    l2 = lose_sigma ** 2 + cfg.tau ** 2
    c2 = 2 * cfg.beta ** 2 + w2 + l2
    c = np.sqrt(c2)
    t = (win_mu - lose_mu) / c
    v = norm.pdf(t) / max(norm.cdf(t), 1e-12)
    w = v * (v + t)
    return (
        win_mu + (w2 / c) * v,
        max(np.sqrt(max(w2 * (1 - (w2 / c2) * w), 1e-6)), cfg.min_sigma),
        lose_mu - (l2 / c) * v,
        max(np.sqrt(max(l2 * (1 - (l2 / c2) * w), 1e-6)), cfg.min_sigma),
    )


def _is_decisive(winner, red, blue):
    """True when the bout has a real winner. Draws and NCs leave ratings alone."""
    return isinstance(winner, str) and winner in (red, blue)


def replay(df, cfg=DEFAULT, state=None):
    """Walk fights in chronological order, recording each fighter's pre-fight rating.

    Args:
        df: fights with date, r_name, b_name, winner. Must be sorted by date.
        cfg: TrueSkillConfig
        state: optional {name: (mu, sigma)} to continue from

    Returns:
        (DataFrame with the six *_pre columns filled, final state dict)
    """
    state = dict(state or {})
    cols = {k: [] for k in ('r_mu_pre', 'r_sigma_pre', 'b_mu_pre', 'b_sigma_pre')}

    for row in df.itertuples(index=False):
        red, blue, winner = row.r_name, row.b_name, row.winner
        r_mu, r_sig = state.get(red, (cfg.mu0, cfg.sigma0))
        b_mu, b_sig = state.get(blue, (cfg.mu0, cfg.sigma0))

        cols['r_mu_pre'].append(r_mu)
        cols['r_sigma_pre'].append(r_sig)
        cols['b_mu_pre'].append(b_mu)
        cols['b_sigma_pre'].append(b_sig)

        if not _is_decisive(winner, red, blue):
            continue
        if winner == red:
            nr_mu, nr_sig, nb_mu, nb_sig = rate_1v1(r_mu, r_sig, b_mu, b_sig, cfg)
        else:
            nb_mu, nb_sig, nr_mu, nr_sig = rate_1v1(b_mu, b_sig, r_mu, r_sig, cfg)
        state[red] = (nr_mu, nr_sig)
        state[blue] = (nb_mu, nb_sig)

    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    out['r_mmr_pre'] = conservative_rating(out['r_mu_pre'], out['r_sigma_pre'])
    out['b_mmr_pre'] = conservative_rating(out['b_mu_pre'], out['b_sigma_pre'])
    return out, state


def state_from_existing(df, cfg=DEFAULT):
    """Recover each fighter's current rating from a dataset that already has ratings.

    Takes the fighter's most recent bout, reads its recorded pre-fight rating,
    and applies one update for that bout's result. This continues the existing
    rating series instead of recomputing it, so values the model was trained on
    do not move.
    """
    df = df.sort_values('date', kind='mergesort')
    latest = {}
    for pos, row in enumerate(df.itertuples(index=False)):
        for corner, name in (('r', row.r_name), ('b', row.b_name)):
            if isinstance(name, str) and name:
                latest[name] = (pos, corner)

    rows = list(df.itertuples(index=False))
    state = {}
    for name, (pos, corner) in latest.items():
        row = rows[pos]
        red, blue, winner = row.r_name, row.b_name, row.winner
        r_mu, r_sig = float(row.r_mu_pre), float(row.r_sigma_pre)
        b_mu, b_sig = float(row.b_mu_pre), float(row.b_sigma_pre)
        if any(pd.isna(x) for x in (r_mu, r_sig, b_mu, b_sig)):
            continue
        if not _is_decisive(winner, red, blue):
            state[name] = (r_mu, r_sig) if corner == 'r' else (b_mu, b_sig)
            continue
        if winner == red:
            nr_mu, nr_sig, nb_mu, nb_sig = rate_1v1(r_mu, r_sig, b_mu, b_sig, cfg)
        else:
            nb_mu, nb_sig, nr_mu, nr_sig = rate_1v1(b_mu, b_sig, r_mu, r_sig, cfg)
        state[name] = (nr_mu, nr_sig) if corner == 'r' else (nb_mu, nb_sig)
    return state


def extend(existing, new_fights, cfg=DEFAULT):
    """Append new fights to a rated dataset, rating them from where it left off.

    Existing rows are returned untouched - their ratings are exactly what the
    model was trained on. Only the new rows get freshly computed ratings.
    """
    if new_fights is None or len(new_fights) == 0:
        return existing.copy()
    state = state_from_existing(existing, cfg)
    new_sorted = new_fights.sort_values('date', kind='mergesort').reset_index(drop=True)
    rated_new, _ = replay(new_sorted, cfg, state=state)
    combined = pd.concat([existing, rated_new], ignore_index=True)
    return combined.sort_values('date', kind='mergesort').reset_index(drop=True)
