"""True Fight Weight: how heavy a fighter actually is when the bell rings.

The division limit says everyone at lightweight fights at 155. The measured
fight-night weights say otherwise: median regain about 10%, the middle half
7-13%, individuals from -1% to +19% (fight_night_weights.csv). At 155 that
spreads fighters from about 166 to 175 lb, and further at the tails.

The estimator (design 2.2): pull every fighter toward the division norm, and
trust their own history in proportion to how much of it there is.

    r_hat = mu(limit, sex) + w_n * e_bar          w_n = n / (n + k)

    mu      the division-by-sex MEAN regain over labels published before the
            fight (a cell needs MIN_CELL labels, else the division pooled
            over sexes, else every label)
    e_bar   the mean of the fighter's own earlier published residuals,
            regain minus mu at that fight's class, so a class change carries
    n       how many earlier measurements they have; k is chosen by
            forward-chained validation and fixed before anything is marked

A measurement may inform a fight only if it was PUBLISHED strictly before
that fight (published_at < fight date). A fight's own measurement is taken
on fight day and published after the bout, so it never informs that fight,
and nothing from the same event does either.

A measurement is a division cut only when the fighter made a division
limit: a catch-weight or missed-weight row (OFF_LIMIT: the archive or the
world harvest says catch weight, or the official weight is more than
ALLOWANCE_LB from the limit) is kept in the CSV but is never a fighter's
history, never enters a norm, and is never scored - the variable does not
ship to such fights either. Heavyweight is a ceiling, not a target, and is
excluded from the bout test for the same reason the experiment masks it.

Validation (design 2.4) is forward-chained by event: labels published before
the event predict it. Fighters WITH an earlier measurement and WITHOUT one
are reported separately, against the division-by-sex median and the earlier
regain alone; on bouts where both opponents were measured, the error of the
estimated gap against the weigh-in-only gap, and how often the estimate
picks the heavier man. The gate (2.4.5) needs MIN_GATE_EVENTS events: with
fewer, an event bootstrap cannot reach zero and would pass on a sample it
cannot judge, so it reports "too few events to test" instead. The old ridge
frame features (height, reach, age, class drop) stay only if validation
earns them. Leave-one-event-out is printed as the design's secondary check.

    python engine/true_fight_weight.py --validate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from fight_night_weights import with_published_at
from name_resolution import canonical_winners, norm_name

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
MEASURED = ENGINE / "data" / "fight_night_weights.csv"
WORLD = ENGINE / "data" / "world_bouts.csv.gz"

LIMITS = {"strawweight": 115, "flyweight": 125, "bantamweight": 135,
          "featherweight": 145, "lightweight": 155, "welterweight": 170,
          "middleweight": 185, "light heavyweight": 205, "heavyweight": 265}
HEAVYWEIGHT = 265
HEAVYWEIGHT_FLOOR = 206.0   # an official weight above this is a heavyweight
ALLOWANCE_LB = 1.5          # further than this from the limit: not a division cut
FRAME_FEATURES = ["height_z", "reach_z", "age", "came_down"]
K_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
DEFAULT_K = 1.0        # until forward validation has seen MIN_K_ROWS repeats
MIN_K_ROWS = 3
MIN_K_IDENTIFY = 30    # with-history rows before a chosen k means anything
MIN_CELL = 5           # labels a (limit, sex) cell needs to stand on its own
MIN_GATE_EVENTS = 6    # events the bout gate needs before an interval can judge
MAX_EVENT_SHARE = 0.5  # no single event may carry more than this of the gain
DRAWS = 5000

# Label corrections. The CSV stays as harvested; these apply on load.
#   DESIGN_FIXES  the design's quality check (1.5): Wikipedia's Bellator 300
#                 list names Brent Primus's numbers as Renato Moicano's.
#   SPELLING      THIS SESSION's label-quality change, not in the design:
#                 two spellings as the UFC archive / the world harvest have
#                 them (Josh Quinlan fought Danny Barlow at UFC 298; Alexandr
#                 Shabliy fought at both Bellator CS4 and PFL San Diego).
#                 They join a fighter's measurements across events, so they
#                 change which rows have history; the validation reports the
#                 result with and without them.
#   SUSPECT       design 1.5: Talbott and Barcelos carry identical numbers in
#                 every source, a likely transcription duplicate. Flagged,
#                 not dropped: they stay in the division norms but are never
#                 a fighter's history and are not scored.
DESIGN_FIXES = {("2023-10-07", "Renato Moicano"): "Brent Primus"}
SPELLING = {("2024-02-17", "Josh Quinland"): "Josh Quinlan",
            ("2024-09-07", "Alexander Shabily"): "Alexander Shabliy"}
SUSPECT = {("2025-01-18", "Payton Talbott"), ("2025-01-18", "Raoni Barcelos")}


def limit_of(division):
    d = str(division).lower()
    if "catch" in d:
        return np.nan
    for name in sorted(LIMITS, key=len, reverse=True):   # "light heavyweight"
        if name in d:                                    # before "heavyweight"
            return LIMITS[name]
    return np.nan


def class_of(official):
    """The limit an official weight was made for: above HEAVYWEIGHT_FLOOR
    it is heavyweight (265 is a ceiling, 233 lb is not a light heavyweight
    missing by 28), else the nearest limit."""
    official = float(official)
    if official > HEAVYWEIGHT_FLOOR:
        return HEAVYWEIGHT
    return min((lim for lim in LIMITS.values() if lim != HEAVYWEIGHT),
               key=lambda lim: abs(lim - official))


def nearest_limit(official):
    return class_of(official)


def is_women(division):
    return "women" in str(division).lower()


def is_catch(division):
    return "catch" in str(division).lower()


# ---------------------------------------------------------------------------
# archive: per-fighter bouts, for sex, class and the frame features
# ---------------------------------------------------------------------------

class Archive:
    """Per-fighter bouts from the UFC archive, read as of a date."""

    def __init__(self, frame):
        a = canonical_winners(frame).copy()
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        a["limit"] = a["division"].map(limit_of)
        a["women"] = a["division"].map(is_women)
        a["catch"] = a["division"].map(is_catch)
        rows = []
        for side in ("r", "b"):
            rows.append(pd.DataFrame({
                "fighter": a[f"{side}_name"].map(norm_name), "date": a["date"],
                "limit": a["limit"], "women": a["women"], "catch": a["catch"],
                "height": a.get(f"{side}_height"), "reach": a.get(f"{side}_reach"),
                "dob": pd.to_datetime(a.get(f"{side}_dob"), errors="coerce")}))
        self.long = pd.concat(rows).dropna(subset=["date"]).sort_values("date")
        self.bouts = {k: g for k, g in self.long.groupby("fighter")}
        self._norms = {}

    def norms(self, date):
        """Height and reach median/std per limit, from bouts BEFORE date.

        A height is a physical constant, but the set of fighters whose
        heights define "typical" grows with time, so the norm is as-of too
        (design 4.2, rule 3).
        """
        date = pd.Timestamp(date)
        if date not in self._norms:
            before = self.long[self.long["date"] < date]
            self._norms[date] = before.groupby("limit")[["height", "reach"]].agg(
                ["median", "std"]) if len(before) else None
        return self._norms[date]

    def sex(self, fighter):
        """'w' or 'm' from the fighter's divisions; '?' if not in the archive."""
        g = self.bouts.get(norm_name(fighter))
        if g is None or not len(g):
            return "?"
        return "w" if bool(g["women"].any()) else "m"

    def _on(self, fighter, date):
        g = self.bouts.get(norm_name(fighter))
        if g is None:
            return None
        same = g[g["date"] == pd.Timestamp(date)]
        return same if len(same) else None

    def limit_on(self, fighter, date):
        """The limit of the fighter's archive bout on that date, else NaN."""
        same = self._on(fighter, date)
        if same is None or not same["limit"].notna().any():
            return np.nan
        return float(same["limit"].dropna().iloc[0])

    def catch_on(self, fighter, date):
        """True when the archive has the bout on that date as a catch weight."""
        same = self._on(fighter, date)
        return bool(same is not None and same["catch"].any())

    def features(self, fighter, date, limit):
        """Frame features for one fighter at one fight, as of the date.

        Height and reach are the values recorded at the fighter's last
        archive bout BEFORE the date; only when nothing earlier exists is
        the earliest later record used (a physical constant, but two
        fighters sharing a name would otherwise lend each other a frame).
        Age is from the date of birth; came_down from earlier limits only.
        """
        date = pd.Timestamp(date)
        g = self.bouts.get(norm_name(fighter))
        height = reach = age = np.nan
        came_down = 0.0
        if g is not None and len(g):
            before, later = g[g["date"] < date], g[g["date"] >= date]
            for column in ("height", "reach"):
                early = before[column].dropna()
                late = later[column].dropna()
                value = early.iloc[-1] if len(early) else (late.iloc[0] if len(late) else np.nan)
                if column == "height":
                    height = value
                else:
                    reach = value
            dob = g["dob"].dropna()
            if len(dob):
                age = (date - dob.iloc[0]).days / 365.25
            if len(before) and before["limit"].notna().any():
                came_down = max(0.0, float(before["limit"].max()) - float(limit))
        norms = self.norms(date)

        def z(value, column):
            if norms is None or limit not in norms.index or not np.isfinite(value):
                return 0.0
            med, sd = norms.loc[limit, (column, "median")], norms.loc[limit, (column, "std")]
            return float((value - med) / sd) if sd and np.isfinite(sd) else 0.0
        return {"height_z": z(height, "height"), "reach_z": z(reach, "reach"),
                "age": float(age) if np.isfinite(age) else 30.0,
                "came_down": came_down}


# ---------------------------------------------------------------------------
# world harvest: sex and the contracted class for fighters outside the UFC
# ---------------------------------------------------------------------------

class World:
    """Per-fighter bouts from the world harvest (Bellator, PFL, ...), for the
    sex and the contracted class of fighters the UFC archive has never seen.
    A Bellator woman is 'Women's Flyweight' on her event page; a 138-lb bout
    is 'Catchweight (138 lb)'. Sex is knowable before every fight; '?' is a
    join gap, so it is filled from here before anything falls back."""

    def __init__(self, frame):
        f = frame.copy()
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f["weight"] = f.get("weight", pd.Series(np.nan, index=f.index)).astype(str)
        self.frame = f
        rows = []
        for side in ("winner", "loser"):
            rows.append(pd.DataFrame({"fighter": f[side].map(norm_name),
                                      "date": f["date"], "weight": f["weight"]}))
        long = pd.concat(rows).dropna(subset=["date"])
        self.bouts = {k: g for k, g in long.groupby("fighter")}

    def sex(self, fighter):
        g = self.bouts.get(norm_name(fighter))
        if g is None:
            return "?"
        w = g["weight"].str.lower()
        if w.str.contains("women").any():
            return "w"
        if any(np.isfinite(limit_of(x)) for x in w):
            return "m"
        return "?"

    def _on(self, fighter, date):
        g = self.bouts.get(norm_name(fighter))
        if g is None:
            return None
        same = g[g["date"] == pd.Timestamp(date)]
        return same if len(same) else None

    def limit_on(self, fighter, date):
        same = self._on(fighter, date)
        if same is None:
            return np.nan
        limits = [limit_of(x) for x in same["weight"]]
        limits = [x for x in limits if np.isfinite(x)]
        return float(limits[0]) if limits else np.nan

    def catch_on(self, fighter, date):
        same = self._on(fighter, date)
        return bool(same is not None and same["weight"].map(is_catch).any())


def _world_frame(world):
    return world.frame if isinstance(world, World) else world


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def load_labels(path=MEASURED, archive=None, world=None, spelling=True):
    """Measured regains with published_at, class, sex, flags and a name key.

    DESIGN_FIXES always apply; SPELLING only with spelling=True (the
    validation reports both). SUSPECT rows are flagged, kept.
    Class is the archive's limit on that date, else the world harvest's,
    else the limit the official weight was made for. off_limit marks a
    catch-weight or missed-weight row (either source says catch weight, or
    the official weight is more than ALLOWANCE_LB from a non-heavyweight
    limit): not a division cut, so never history, norm or scored row.
    Sex comes from the archive, else the world harvest, else the opponent
    in the same bout (pair_bouts), and is '?' only when none of them knows.
    """
    m = pd.read_csv(path, dtype={"published_at": str})
    m = with_published_at(m)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["published_at"] = pd.to_datetime(m["published_at"], errors="coerce")
    renames = {**DESIGN_FIXES, **SPELLING} if spelling else dict(DESIGN_FIXES)
    day = m["date"].dt.strftime("%Y-%m-%d")
    m["fighter"] = [renames.get((d, f), f) for d, f in zip(day, m["fighter"])]
    m["suspect"] = [(d, f) in SUSPECT for d, f in zip(day, m["fighter"])]
    m["key"] = m["fighter"].map(norm_name)
    m["limit"] = m["official_lbs"].map(class_of).astype(float)
    m["off_limit"] = False
    m["sex"] = "?"
    assigned = np.zeros(len(m), bool)
    for source in (s for s in (archive, world) if s is not None):
        on_date = np.array([source.limit_on(f, d) for f, d in zip(m["fighter"], m["date"])])
        take = np.isfinite(on_date) & ~assigned
        m.loc[take, "limit"] = on_date[take]
        assigned |= take
        m["off_limit"] |= np.array([source.catch_on(f, d) for f, d in zip(m["fighter"], m["date"])])
        unknown = (m["sex"] == "?").to_numpy()
        m.loc[unknown, "sex"] = [source.sex(f) for f in m.loc[unknown, "fighter"]]
    off = (m["limit"] != HEAVYWEIGHT) & ((m["official_lbs"] - m["limit"]).abs() > ALLOWANCE_LB)
    m["off_limit"] |= off.to_numpy()
    # Sex is shared within a bout: an unknown opponent of a known one.
    for i, j in pair_bouts(m, world):
        a, b = m.at[i, "sex"], m.at[j, "sex"]
        if a == "?" and b in ("m", "w"):
            m.at[i, "sex"] = b
        elif b == "?" and a in ("m", "w"):
            m.at[j, "sex"] = a
    m.attrs["suspect_flagged"] = int(m["suspect"].sum())
    m.attrs["off_limit_flagged"] = int(m["off_limit"].sum())
    m.attrs["spelling"] = bool(spelling)
    return m.sort_values(["published_at", "date", "event"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------

def shrinkage(n, k):
    """w_n = n / (n + k): 0 with no history, toward 1 as it grows."""
    n = float(n)
    if n <= 0:
        return 0.0
    if k == np.inf:
        return 0.0
    return n / (n + float(k))


class RegainEstimator:
    """r_hat = mu(limit, sex) + w_n * e_bar from labels PUBLISHED before a date.

    Rows flagged off_limit are never available; rows flagged suspect enter
    the norms but are never a fighter's history.
    """

    def __init__(self, labels, k=DEFAULT_K, min_cell=MIN_CELL):
        need = {"key", "date", "published_at", "limit", "sex", "gain_pct"}
        missing = need - set(labels.columns)
        if missing:
            raise ValueError(f"labels lack {sorted(missing)}")
        labels = labels.copy()
        for flag in ("off_limit", "suspect"):
            if flag not in labels.columns:
                labels[flag] = False
            labels[flag] = labels[flag].fillna(False).astype(bool)
        self.labels = labels.sort_values("published_at").reset_index(drop=True)
        self.k = k
        self.min_cell = min_cell
        self._tables = {}

    def available(self, date):
        """Labels published strictly before the date that are division cuts."""
        return self.labels[(self.labels["published_at"] < pd.Timestamp(date))
                           & ~self.labels["off_limit"]]

    def table(self, date):
        """{('cell'|'limit'|'all', ...): (mean, median, n)} as of the date."""
        date = pd.Timestamp(date)
        if date not in self._tables:
            avail = self.available(date)
            t = {}
            if len(avail):
                for (lim, sex), g in avail.groupby(["limit", "sex"]):
                    t[("cell", float(lim), sex)] = (g["gain_pct"].mean(),
                                                   g["gain_pct"].median(), len(g))
                for lim, g in avail.groupby("limit"):
                    t[("limit", float(lim))] = (g["gain_pct"].mean(),
                                                g["gain_pct"].median(), len(g))
                t[("all",)] = (avail["gain_pct"].mean(), avail["gain_pct"].median(),
                               len(avail))
            self._tables[date] = t
        return self._tables[date]

    def mu(self, date, limit, sex, stat="mean"):
        """(division norm, level): sex cell, else division, else everything.

        NaN when nothing has been published before the date.
        """
        t = self.table(date)
        which = 0 if stat == "mean" else 1
        for key in ((("cell", float(limit), sex) if sex in ("m", "w") else None),
                    ("limit", float(limit)), ("all",)):
            if key and key in t and t[key][2] >= self.min_cell:
                return float(t[key][which]), key[0]
        if ("all",) in t:
            return float(t[("all",)][which]), "all"
        return np.nan, "none"

    def history(self, key, date):
        """The fighter's own division-cut labels published before the date."""
        avail = self.available(date)
        return avail[(avail["key"] == key) & ~avail["suspect"]]

    def estimate(self, key, date, limit, sex, k=None):
        """r_hat and its parts for one fighter at one fight."""
        k = self.k if k is None else k
        mu, level = self.mu(date, limit, sex)
        out = {"r_hat": np.nan, "mu": mu, "level": level, "e_bar": 0.0,
               "n": 0, "w": 0.0, "prior_alone": np.nan}
        if not np.isfinite(mu):
            return out
        mine = self.history(key, date)
        residuals = [float(g) - self.mu(date, lim, s)[0]
                     for g, lim, s in zip(mine["gain_pct"], mine["limit"], mine["sex"])]
        n = len(residuals)
        e_bar = float(np.mean(residuals)) if n else 0.0
        w = shrinkage(n, k)
        out.update({"r_hat": mu + w * e_bar, "e_bar": e_bar, "n": n, "w": w,
                    "prior_alone": float(mine["gain_pct"].mean()) if n else np.nan})
        return out


def fight_night_lbs(official, r_hat):
    return float(official) * (1.0 + float(r_hat) / 100.0)


# ---------------------------------------------------------------------------
# frame features: the ridge component, kept only if validation earns it
# ---------------------------------------------------------------------------

class FrameRidge:
    """Ridge on the frame features, fitted to residuals from mu, as of a date."""

    def __init__(self, estimator, archive, alpha=3.0):
        self.estimator = estimator
        self.archive = archive
        self.alpha = alpha
        self._models = {}

    def _x(self, row):
        f = self.archive.features(row.fighter, row.date, row.limit)
        return [f[c] for c in FRAME_FEATURES]

    def model(self, date):
        date = pd.Timestamp(date)
        if date not in self._models:
            from sklearn.linear_model import Ridge
            train = self.estimator.available(date)
            model = None
            if len(train) >= 20:
                X = np.array([self._x(r) for r in train.itertuples()])
                y = np.array([float(g) - self.estimator.mu(date, lim, s)[0]
                              for g, lim, s in zip(train["gain_pct"], train["limit"],
                                                   train["sex"])])
                model = Ridge(alpha=self.alpha).fit(X, y)
            self._models[date] = model
        return self._models[date]

    def adjustment(self, fighter, date, limit, as_of=None):
        """beta . x for one fighter at one fight, 0 before 20 labels exist.

        The frame is read as of the fight date; the fit as of `as_of`
        (the fight date forward-chained, everything for leave-one-event-out).
        """
        model = self.model(date if as_of is None else as_of)
        if model is None:
            return 0.0
        f = self.archive.features(fighter, date, limit)
        return float(model.predict(np.array([[f[c] for c in FRAME_FEATURES]]))[0])


# ---------------------------------------------------------------------------
# forward-chained validation
# ---------------------------------------------------------------------------

def _row(est, r, as_of, date, event, ridge, k_grid):
    """One scored row: the truth, the norms, the earlier regain alone, the
    model at every k, and the parts the bout test needs. `as_of` is the
    date the labels are read as of (the fight date forward-chained)."""
    base = est.estimate(r.key, as_of, r.limit, r.sex)
    row = {"date": date, "event": event, "key": r.key, "fighter": r.fighter,
           "published_at": r.published_at,
           "official_lbs": r.official_lbs, "fight_night_lbs": r.fight_night_lbs,
           "gain_pct": r.gain_pct, "limit": r.limit, "sex": r.sex,
           "n": base["n"], "has_history": base["n"] > 0,
           "level": base["level"], "mu_mean": base["mu"],
           "mu_median": est.mu(as_of, r.limit, r.sex, "median")[0],
           "mu_pooled_mean": est.mu(as_of, r.limit, "?")[0],
           "mu_pooled_median": est.mu(as_of, r.limit, "?", "median")[0],
           "e_bar": base["e_bar"], "prior_alone": base["prior_alone"]}
    beta = ridge.adjustment(r.fighter, date, r.limit, as_of=as_of) if ridge else 0.0
    for k in k_grid:
        e = est.estimate(r.key, as_of, r.limit, r.sex, k=k)
        row[f"k={k}"] = e["r_hat"]
        row[f"frame_k={k}"] = e["r_hat"] + beta
    return row


def _scorable(labels):
    """Rows the validation scores: trusted labels of a division cut below
    heavyweight - the fights the variable can ship to, the experiment's mask.
    Heavyweight and off-limit rows still feed the norms."""
    off = labels["off_limit"] if "off_limit" in labels.columns else False
    sus = labels["suspect"] if "suspect" in labels.columns else False
    heavy = (labels["limit"] == HEAVYWEIGHT).to_numpy()
    return labels[~(np.asarray(off, bool) | np.asarray(sus, bool) | heavy)]


def forward_rows(labels, archive=None, k_grid=K_GRID):
    """One row per scorable label predicted by everything published before
    its event (off-limit and suspect rows are not scored)."""
    est = RegainEstimator(labels)
    ridge = FrameRidge(est, archive) if archive is not None else None
    rows = []
    for (date, event), test in _scorable(labels).groupby(["date", "event"], sort=True):
        if not len(est.available(date)):
            continue                                   # nothing public yet
        for r in test.itertuples():
            rows.append(_row(est, r, date, date, event, ridge, k_grid))
    return pd.DataFrame(rows)


def loeo_rows(labels, archive=None, k_grid=K_GRID):
    """Leave-one-event-out: every OTHER event's labels predict this one,
    later events included. Design 2.4(1)'s secondary check - more training
    data than deployment ever has, so it is not point in time and not the
    gate."""
    rows = []
    for (date, event), test in _scorable(labels).groupby(["date", "event"], sort=True):
        others = labels[labels["event"] != event]
        if not len(others):
            continue
        est = RegainEstimator(others)
        ridge = FrameRidge(est, archive) if archive is not None else None
        for r in test.itertuples():
            rows.append(_row(est, r, pd.Timestamp.max, date, event, ridge, k_grid))
    return pd.DataFrame(rows)


def k_for(rows, date, k_grid=K_GRID, min_rows=MIN_K_ROWS, default=DEFAULT_K):
    """The k a fight on `date` may use: the one that minimises the forward
    error on fighters with history whose labels were PUBLISHED before the
    date; DEFAULT_K until MIN_K_ROWS such rows exist. Ties go to the
    smaller k."""
    if not len(rows):
        return default
    earlier = rows[(rows["published_at"] < pd.Timestamp(date)) & rows["has_history"]]
    if len(earlier) < min_rows:
        return default
    errors = {k_: np.abs(earlier[f"k={k_}"] - earlier["gain_pct"]).mean() for k_ in k_grid}
    return min(k_grid, key=lambda k_: (errors[k_], k_grid.index(k_)))


def nested_k(rows, k_grid=K_GRID, min_rows=MIN_K_ROWS, default=DEFAULT_K):
    """k for every event from the repeats of EARLIER events only (k_for).

    Returns the rows with 'k_used' and 'nested' (and 'frame_nested').
    """
    rows = rows.sort_values("date").copy()
    rows["k_used"] = default
    rows["nested"] = np.nan
    rows["frame_nested"] = np.nan
    for date in sorted(rows["date"].unique()):
        k = k_for(rows, date, k_grid, min_rows, default)
        here = rows["date"] == date
        rows.loc[here, "k_used"] = k
        rows.loc[here, "nested"] = rows.loc[here, f"k={k}"]
        rows.loc[here, "frame_nested"] = rows.loc[here, f"frame_k={k}"]
    return rows


def _mae(pred, rows):
    err = np.abs(pred - rows["gain_pct"])
    lb = err * rows["official_lbs"] / 100.0
    return {"mae_pts": float(err.mean()), "mae_lb": float(lb.mean()), "n": int(err.notna().sum())}


def group_report(rows, k_grid=K_GRID):
    """MAE tables for fighters with and without an earlier measurement."""
    out = {}
    for name, part in (("with_history", rows[rows["has_history"]]),
                       ("without_history", rows[~rows["has_history"]])):
        table = {"n": int(len(part)), "events": int(part["event"].nunique())}
        if len(part):
            table["division_median"] = _mae(part["mu_median"], part)
            table["division_mean"] = _mae(part["mu_mean"], part)
            if name == "with_history":
                table["earlier_regain_alone"] = _mae(part["prior_alone"], part)
                table["by_k"] = {str(k): _mae(part[f"k={k}"], part) for k in k_grid}
                table["frame_by_k"] = {str(k): _mae(part[f"frame_k={k}"], part)
                                       for k in k_grid}
            table["model_nested_k"] = _mae(part["nested"], part)
            table["model_frame_nested_k"] = _mae(part["frame_nested"], part)
            table["k_used"] = sorted(set(float(v) for v in part["k_used"]))
            if name == "with_history":
                # The pre-registered baseline against the shrinkage model,
                # on the only rows where they differ. Negative: the raw
                # earlier regain is the better estimator.
                table["shrinkage_gain_over_earlier_regain_pts"] = float(
                    table["earlier_regain_alone"]["mae_pts"] - table["model_nested_k"]["mae_pts"])
                table["k_identified"] = bool(len(part) >= MIN_K_IDENTIFY)
                table["k_identified_reason"] = (
                    f"{len(part)} rows with history; {MIN_K_IDENTIFY} needed before a "
                    f"chosen k means more than noise" if len(part) < MIN_K_IDENTIFY
                    else f"{len(part)} rows with history")
        out[name] = table
    return out


def _event_interval(values, events, alpha=0.05, seed=0):
    values, events = np.asarray(values, float), np.asarray(events)
    unique = np.unique(events)
    if len(unique) < 2:
        return [float("nan"), float("nan")]
    index = {e: np.flatnonzero(events == e) for e in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(unique), len(unique))
        means[i] = values[np.concatenate([index[unique[j]] for j in pick])].mean()
    return [float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2)))]


def _by_event(values, events):
    """Per-event (n, mean, sum) and the leave-one-event-out intervals: what
    an event-bootstrap result over a handful of events actually rests on."""
    frame = pd.DataFrame({"v": np.asarray(values, float), "e": np.asarray(events)})
    per = {str(e): {"n": int(len(g)), "mean": float(g["v"].mean()), "sum": float(g["v"].sum())}
           for e, g in frame.groupby("e")}
    total = float(frame["v"].sum())
    # The largest event's share of the total gain; above 1 when the other
    # events net negative, NaN when there is no gain to share.
    share = max(t["sum"] for t in per.values()) / total if total > 0 else float("nan")
    loeo = {}
    for e in per:
        rest = frame[frame["e"] != e]
        loeo[e] = {"n": int(len(rest)), "mean": float(rest["v"].mean()) if len(rest) else float("nan"),
                   "interval": _event_interval(rest["v"], rest["e"]) if len(rest) else [float("nan")] * 2}
    return {"per_event": per, "max_event_share": float(share), "leave_one_event_out": loeo}


def frame_verdict(rows):
    """Do the ridge frame features earn their place?

    They would matter most for fighters WITHOUT history, where the model is
    otherwise the division mean. Earned only if they lower the forward
    error there and the event-bootstrap interval of the per-row gain
    excludes zero.
    """
    part = rows[~rows["has_history"]]
    if len(part) < 20:
        return {"earned": False, "reason": "too few rows", "n": int(len(part))}
    gain = (np.abs(part["nested"] - part["gain_pct"])
            - np.abs(part["frame_nested"] - part["gain_pct"]))
    low, high = _event_interval(gain, part["event"])
    earned = bool(gain.mean() > 0 and low > 0)
    return {"earned": earned, "n": int(len(part)), "mean_gain_pts": float(gain.mean()),
            "interval": [low, high],
            "reason": "lowers the error and the interval excludes zero" if earned
            else "does not clearly lower the error for fighters without history"}


# ---------------------------------------------------------------------------
# bouts where both opponents were measured
# ---------------------------------------------------------------------------

def pair_bouts(labels, world=None):
    """(red row, blue row) index pairs within an event.

    The world harvest names the opponents where it has the bout. The rows
    it leaves are paired as the Wikipedia lists are written, consecutively,
    but only inside a run of unpaired rows of EVEN length (a harvest pair
    that is not adjacent, or a missing row, leaves an odd run whose
    alignment is unknown) and only when the two official weights are within
    12 lb or both are heavyweight.
    """
    world = _world_frame(world)
    pairs, used = [], set()
    by_date = {}
    if world is not None:
        w = world.dropna(subset=["winner", "loser"])
        for d, a, b in zip(pd.to_datetime(w["date"], errors="coerce"), w["winner"], w["loser"]):
            if pd.notna(d):
                by_date.setdefault(d, set()).add(frozenset((norm_name(a), norm_name(b))))
    for (date, _event), g in labels.groupby(["date", "event"], sort=True):
        idx = list(g.index)
        keys = {i: labels.at[i, "key"] for i in idx}
        known = by_date.get(pd.Timestamp(date), set())
        for i in idx:
            if i in used:
                continue
            for j in idx:
                if j != i and j not in used and frozenset((keys[i], keys[j])) in known:
                    pairs.append((i, j))
                    used.update((i, j))
                    break
        runs, run = [], []
        for i in idx:                       # runs of consecutive unpaired rows
            if i in used:
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(i)
        if run:
            runs.append(run)
        for run in runs:
            if len(run) % 2:
                continue                    # alignment unknown
            for a, b in zip(run[0::2], run[1::2]):
                wa, wb = labels.at[a, "official_lbs"], labels.at[b, "official_lbs"]
                if abs(wa - wb) <= 12 or (wa > HEAVYWEIGHT_FLOOR and wb > HEAVYWEIGHT_FLOOR):
                    pairs.append((a, b))
                    used.update((a, b))
    return pairs


def _estimate_from_row(r, pooled, stat="mean"):
    """The row's estimate, at its own level or (pooled) the division level
    shared with an opponent whose level differs."""
    if stat == "median":
        return r.mu_pooled_median if pooled else r.mu_median
    if pooled:
        return r.mu_pooled_mean + shrinkage(r.n, r.k_used) * r.e_bar
    return r.nested


def bout_report(rows, labels, world=None):
    """Gap error against the weigh-in-only gap, and who is picked heavier.

    Only bouts the variable could ship to: a division limit on both sides
    (no catch weight or missed weight) and not heavyweight - the mask the
    experiment applies. Two opponents must be judged at the same norm level:
    where one has a sex cell and the other only the pooled division, both
    use the pooled division, so no gap is an artefact of the join.
    """
    by_key = {(r.date, r.key): r for r in rows.itertuples()}
    bouts = []
    excluded = {"heavyweight": 0, "off_limit": 0, "pooled_level": 0}
    for i, j in pair_bouts(labels, world):
        a, b = labels.loc[i], labels.loc[j]
        if bool(a.get("off_limit", False)) or bool(b.get("off_limit", False)):
            excluded["off_limit"] += 1
            continue
        ra, rb = by_key.get((a.date, a.key)), by_key.get((b.date, b.key))
        if ra is None or rb is None:
            continue                        # not scored: first event or suspect
        if a.limit == HEAVYWEIGHT or b.limit == HEAVYWEIGHT:
            excluded["heavyweight"] += 1
            continue
        pooled = ra.level != rb.level
        excluded["pooled_level"] += int(pooled)
        measured = a.fight_night_lbs - b.fight_night_lbs
        bouts.append({
            "date": a.date, "event": a.event, "red": a.fighter, "blue": b.fighter,
            "limit": a.limit, "either_history": bool(ra.has_history or rb.has_history),
            "pooled_level": pooled,
            "measured": float(measured),
            "weigh_in_only": float(a.official_lbs - b.official_lbs),
            "division_median": fight_night_lbs(a.official_lbs, _estimate_from_row(ra, pooled, "median"))
            - fight_night_lbs(b.official_lbs, _estimate_from_row(rb, pooled, "median")),
            "model": fight_night_lbs(a.official_lbs, _estimate_from_row(ra, pooled))
            - fight_night_lbs(b.official_lbs, _estimate_from_row(rb, pooled))})
    bouts = pd.DataFrame(bouts)
    out = {"bouts": int(len(bouts)), "events": int(bouts["event"].nunique()) if len(bouts) else 0,
           "excluded": excluded}
    if not len(bouts):
        return out
    for name, part in (("all", bouts), ("either_history", bouts[bouts["either_history"]])):
        table = {"n": int(len(part)), "events": int(part["event"].nunique()) if len(part) else 0}
        if len(part):
            for col in ("weigh_in_only", "division_median", "model"):
                err = (part[col] - part["measured"]).abs()
                decided = part[part[col].abs() > 1e-9]
                agree = np.sign(decided[col]) == np.sign(decided["measured"])
                table[col] = {"gap_mae_lb": float(err.mean()),
                              "picks_heavier": float(agree.mean()) if len(decided) else float("nan"),
                              "decided": int(len(decided))}
            improve = ((part["weigh_in_only"] - part["measured"]).abs()
                       - (part["model"] - part["measured"]).abs())
            table["model_vs_weigh_in_only"] = dict(
                {"mean_gain_lb": float(improve.mean()),
                 "interval": _event_interval(improve, part["event"]),
                 "events": int(part["event"].nunique())},
                **_by_event(improve, part["event"]))
            if len(part) >= 3 and part["model"].std() > 0:
                table["rank_corr"] = float(part[["model", "measured"]].corr("spearman").iloc[0, 1])
                slope = np.polyfit(part["model"], part["measured"], 1)[0]
                table["calibration_slope"] = float(slope)
            if name == "either_history":
                table["rows"] = [{"date": str(r.date.date()), "event": r.event, "red": r.red,
                                  "blue": r.blue, "measured": round(r.measured, 2),
                                  "weigh_in_only": round(r.weigh_in_only, 2),
                                  "model": round(r.model, 2)} for r in part.itertuples()]
        out[name] = table
    return out


def gate(report, min_events=MIN_GATE_EVENTS, max_share=MAX_EVENT_SHARE):
    """Design 2.4 (5): the gap error must beat weigh-in-only, interval > 0.

    An event bootstrap over a handful of events reaches zero only when a
    whole event is negative, so with fewer than `min_events` events, or
    with one event carrying more than `max_share` of the gain, the test has
    no power to fail and reports "too few events to test" rather than PASSED.
    """
    part = report.get("bouts", {}).get("either_history", {})
    test = part.get("model_vs_weigh_in_only")
    events = part.get("events", 0)
    if not test or events < 2:
        return {"passed": False, "testable": False,
                "reason": "too few bouts with a measured history to test"}
    out = {"mean_gain_lb": test["mean_gain_lb"], "interval": test["interval"],
           "events": events, "bouts": part.get("n", 0),
           "max_event_share": test.get("max_event_share", float("nan"))}
    low = test["interval"][0]
    beats = bool(test["mean_gain_lb"] > 0 and np.isfinite(low) and low > 0)
    if events < min_events:
        out.update({"passed": False, "testable": False,
                    "reason": f"too few events to test: {events} events, {min_events} needed "
                              f"before an event bootstrap can reach zero (the interval "
                              f"{'excludes' if beats else 'includes'} zero but cannot judge)"})
        return out
    share = out["max_event_share"]
    if np.isfinite(share) and share > max_share:
        out.update({"passed": False, "testable": False,
                    "reason": f"one event carries {share:.0%} of the gain (limit {max_share:.0%})"})
        return out
    out.update({"passed": beats, "testable": True,
                "reason": "beats the weigh-in-only gap, interval excludes zero" if beats
                else "does not beat the weigh-in-only gap beyond the event bootstrap"})
    return out


def loeo_report(labels, archive=None):
    """The secondary leave-one-event-out tables (design 2.4(1))."""
    rows = loeo_rows(labels, archive)
    if not len(rows):
        return {"n": 0}
    k = chosen_k(labels)
    rows["k_used"] = k
    rows["nested"] = rows[f"k={k}"]
    rows["frame_nested"] = rows[f"frame_k={k}"]
    out = {"k": float(k), "note": "every other event predicts each one, later events included: "
                                  "not point in time, not the gate"}
    out.update(group_report(rows))
    return out


def validate(labels, archive=None, world=None, loeo=True):
    rows = nested_k(forward_rows(labels, archive))
    report = {"labels": int(len(labels)), "events": int(labels["event"].nunique()),
              "suspect_flagged": int(labels.attrs.get("suspect_flagged",
                                                      labels.get("suspect", pd.Series(dtype=bool)).sum())),
              "off_limit_flagged": int(labels.attrs.get("off_limit_flagged",
                                                        labels.get("off_limit", pd.Series(dtype=bool)).sum())),
              "heavyweight_rows": int((labels["limit"] == HEAVYWEIGHT).sum()),
              "spelling_corrections": bool(labels.attrs.get("spelling", True)),
              "predicted_rows": int(len(rows)),
              "predicted_events": int(rows["event"].nunique()) if len(rows) else 0,
              "first_event_training_only": True,
              "groups": group_report(rows),
              "frame_features": frame_verdict(rows) if archive is not None and len(rows) else
              {"earned": False, "reason": "no archive"},
              "bouts": bout_report(rows, labels, world) if len(rows) else {}}
    report["gate"] = gate(report)
    report["k_last_event"] = float(rows["k_used"].iloc[-1]) if len(rows) else DEFAULT_K
    report["k_final"] = float(k_for(rows, pd.Timestamp.max)) if len(rows) else DEFAULT_K
    report["k_identified"] = bool(report["groups"]["with_history"].get("k_identified", False))
    if loeo:
        report["leave_one_event_out"] = loeo_report(labels, archive)
    return report, rows


def chosen_k(labels):
    """The k forward validation would use for a fight after every label."""
    rows = forward_rows(labels)
    return k_for(rows, pd.Timestamp.max) if len(rows) else DEFAULT_K


def spelling_sensitivity(path, archive=None, world=None):
    """The validation WITHOUT this session's spelling corrections, in brief:
    the design's corrections alone. Printed next to the main result so the
    reader sees what the joins across events rest on."""
    labels = load_labels(path, archive, world, spelling=False)
    report, _ = validate(labels, archive, world, loeo=False)
    g = report["groups"]["with_history"]
    e = report["bouts"].get("either_history", {})
    v = e.get("model_vs_weigh_in_only", {})
    return {"labels": int(len(labels)), "with_history_rows": g["n"], "with_history_events": g["events"],
            "earlier_regain_alone_pts": g.get("earlier_regain_alone", {}).get("mae_pts"),
            "model_nested_k_pts": g.get("model_nested_k", {}).get("mae_pts"),
            "either_history_bouts": e.get("n", 0), "either_history_events": e.get("events", 0),
            "model_vs_weigh_in_only_lb": v.get("mean_gain_lb"), "interval": v.get("interval"),
            "gate": report["gate"]}


def full_validation(archive, world, path=MEASURED):
    """Labels, the validation report (with the spelling sensitivity) and the
    forward rows, as both the CLI and the experiment use them."""
    labels = load_labels(path, archive, world)
    report, rows = validate(labels, archive, world)
    report["without_spelling_corrections"] = spelling_sensitivity(path, archive, world)
    return labels, report, rows


def _print_report(report):
    g = report["groups"]
    print(f"  {report['labels']} labels, {report['events']} events; "
          f"{report['off_limit_flagged']} catch/missed-weight rows flagged (never history, "
          f"norm or scored), {report['suspect_flagged']} suspect rows flagged (never history, "
          f"not scored), {report['heavyweight_rows']} heavyweight rows (norms only: the "
          f"variable never ships to heavyweight)")
    print(f"  {report['predicted_rows']} rows of {report['predicted_events']} events "
          f"predicted forward-chained (the first event has nothing before it)")
    for name in ("with_history", "without_history"):
        t = g[name]
        print(f"\n  fighters {name.replace('_', ' ')}: {t['n']} rows, {t['events']} events")
        if not t["n"]:
            continue
        for label in ("division_median", "division_mean", "earlier_regain_alone",
                      "model_nested_k", "model_frame_nested_k"):
            if label in t:
                m = t[label]
                print(f"    {label:<24}{m['mae_pts']:6.2f} pts  {m['mae_lb']:6.2f} lb")
        if "by_k" in t:
            print("    model by k (descriptive, k chosen on these rows):")
            for k, m in t["by_k"].items():
                print(f"      k={k:<6}{m['mae_pts']:6.2f} pts   with frame {t['frame_by_k'][k]['mae_pts']:6.2f}")
            gain = t["shrinkage_gain_over_earlier_regain_pts"]
            print(f"    shrinkage vs the earlier regain alone: {gain:+.2f} pts "
                  f"({'the raw earlier regain is the better estimator' if gain < 0 else 'shrinkage helps'})")
            print(f"    k identified: {'yes' if t['k_identified'] else 'NO'} - {t['k_identified_reason']}")
        print(f"    k used, nested: {t['k_used']}")
    f = report["frame_features"]
    print(f"\n  frame features (height, reach, age, class drop): "
          f"{'EARNED' if f.get('earned') else 'not earned'} - {f.get('reason')}"
          + (f"; gain {f['mean_gain_pts']:+.2f} pts [{f['interval'][0]:+.2f}, {f['interval'][1]:+.2f}]"
             if "mean_gain_pts" in f else ""))
    b = report["bouts"]
    ex = b.get("excluded", {})
    print(f"\n  bouts with both opponents measured, a division limit, not heavyweight: "
          f"{b.get('bouts', 0)} in {b.get('events', 0)} events "
          f"(excluded: {ex.get('heavyweight', 0)} heavyweight, {ex.get('off_limit', 0)} catch/missed "
          f"weight; {ex.get('pooled_level', 0)} judged at the pooled division level)")
    for name in ("all", "either_history"):
        t = b.get(name)
        if not t:
            continue
        print(f"    {name}: {t['n']} bouts, {t['events']} events")
        for col in ("weigh_in_only", "division_median", "model"):
            if col in t:
                m = t[col]
                print(f"      {col:<18} gap error {m['gap_mae_lb']:5.2f} lb, picks the heavier man "
                      f"{m['picks_heavier']:.0%} of {m['decided']} decided")
        if "model_vs_weigh_in_only" in t:
            v = t["model_vs_weigh_in_only"]
            share = v["max_event_share"]
            print(f"      model vs weigh-in-only: {v['mean_gain_lb']:+.2f} lb "
                  f"[{v['interval'][0]:+.2f}, {v['interval'][1]:+.2f}] over {v['events']} events; "
                  f"largest single event's share of the gain "
                  f"{f'{share:.0%}' if np.isfinite(share) else 'n/a (no net gain)'}")
            if name == "either_history":
                for e, p in v["per_event"].items():
                    lo = v["leave_one_event_out"][e]
                    print(f"        {e[:40]:<40} {p['n']:2d} bouts, mean {p['mean']:+.2f} lb; "
                          f"without it {lo['mean']:+.2f} [{lo['interval'][0]:+.2f}, {lo['interval'][1]:+.2f}]")
        if "rank_corr" in t:
            print(f"      rank corr {t['rank_corr']:+.2f}, calibration slope {t['calibration_slope']:.2f}")
    print(f"\n  k used at the last validated event: {report['k_last_event']}; "
          f"k for the next fight (every published repeat): {report['k_final']}"
          f"{'' if report['k_identified'] else ' - NOT identified, see above'}")
    gt = report["gate"]
    print(f"  GATE: {'PASSED' if gt['passed'] else ('NOT TESTABLE' if not gt.get('testable', True) else 'NOT PASSED')}"
          f" - {gt['reason']}")
    w = report.get("without_spelling_corrections")
    if w:
        wg = w["gate"]
        pts = (f"earlier regain alone {w['earlier_regain_alone_pts']:.2f} vs model "
               f"{w['model_nested_k_pts']:.2f} pts" if w["model_nested_k_pts"] is not None
               else "no rows with history")
        lb = (f"model vs weigh-in-only {w['model_vs_weigh_in_only_lb']:+.2f} lb "
              f"[{w['interval'][0]:+.2f}, {w['interval'][1]:+.2f}]"
              if w["model_vs_weigh_in_only_lb"] is not None else "no bouts with history")
        print(f"  without this session's spelling corrections (design fixes only): "
              f"with-history {w['with_history_rows']} rows / {w['with_history_events']} events, {pts}; "
              f"either-history {w['either_history_bouts']} bouts / {w['either_history_events']} events, "
              f"{lb}; gate "
              f"{'PASSED' if wg['passed'] else ('NOT TESTABLE' if not wg.get('testable', True) else 'NOT PASSED')}")
    lo = report.get("leave_one_event_out")
    if lo and lo.get("n", 1):
        print(f"\n  secondary, leave-one-event-out ({lo['note']}; k={lo['k']}):")
        for name in ("with_history", "without_history"):
            t = lo.get(name, {})
            if not t.get("n"):
                continue
            parts = [f"{lab} {t[lab]['mae_pts']:.2f}" for lab in
                     ("division_median", "earlier_regain_alone", "model_nested_k", "model_frame_nested_k")
                     if lab in t]
            print(f"    {name.replace('_', ' '):<16} {t['n']:3d} rows: " + " | ".join(parts) + " pts")


def load_world(path=WORLD):
    if not Path(path).exists():
        return None
    return World(pd.read_csv(path, usecols=["date", "weight", "winner", "loser"], low_memory=False))


def load_archive(path=ARCHIVE):
    return Archive(pd.read_csv(path, usecols=[
        "date", "division", "r_name", "b_name", "winner", "r_height",
        "b_height", "r_reach", "b_reach", "r_dob", "b_dob"], low_memory=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args(argv)
    archive = load_archive()
    world = load_world()
    labels, report, rows = full_validation(archive, world)
    print(f"  {len(labels)} measured regains, {int((labels['sex'] != '?').sum())} with a known sex "
          f"(archive, world harvest or opponent), {labels['event'].nunique()} events")
    _print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
