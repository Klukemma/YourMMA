"""METHOD ESTIMATOR: a price for "wins by KO/TKO / submission / decision" where no book is on file.

WHY THIS EXISTS. Tickets v2 (parlays/engine/sameday.py, "TICKETS v2") offer
method props - each corner's "wins by KO/TKO", "wins by submission" and
"wins by decision" - and history prices them (data/method_odds.csv, through
2024-12-07). Live there is no prop price at all: the odds api answers the
method markets with INVALID_MARKET (experiments/method_doubt.py), so a prop
leg on a live card can only carry an ESTIMATE of the price a book would
quote. This module is that estimate, built only from what a bettor holds
before the bell, and measured against real prop prices before it is shown.

WHAT IT READS - market and record only, never a result:
  p        the moneyline's no-vig chance of the side (proportional de-vig,
           parlay_profiles.devig; history: the closing line in odds.csv, the
           same upstream snapshot as the props; live: the card's pair).
  division the weight class, men's and women's apart (one indicator each;
           catch weight, open weight and anything unnamed read as none).
  rounds   five rounds scheduled or three.
  records  each corner's UFC record strictly before the fight
           (parlay_profiles.facts): its own KO and submission win rates, the
           opponent's KO and submission loss rates, the opponent's own KO and
           submission win rates (its finish rates), and both fight counts.
           Each rate is per fight, shrunk toward the fitted pool's rate by
           `prior_fights` fights' worth of it, so a debut reads as the pool
           and a twenty-fight veteran mostly as himself.
The model's own method view (method_probs, p_finish) is NOT an input:
experiments/method_blend.json measured the market at AUC 0.660 against the
model's 0.594 on finish-or-distance, and its forward blend gave the model
no weight from 2017 on. On their dev split the tickets v2 scouts found that
age, reach and a decision rate added nothing and that gradient boosting
beat this ridge by at most 0.1 point.

HOW A PRICE IS MADE, per corner and type:
  1. side chance  Q = logistic(b * logit(p)): the six-way market's total
     chance that this corner wins, as a function of the moneyline's. One
     slope and no intercept, so the two corners' chances sum to one and
     swapping the corners swaps the answer.
  2. split        a ridge regression of log(share KO / share decision) and
     log(share submission / share decision) on the inputs above; the shares
     are the six-way market's (its de-vigged chances - price ratios, so the
     split does not depend on how the margin is taken out).
  3. chance       c = Q x share: the estimate of the six-way market's
     de-vigged chance of this prop.
  4. price        1 / (c x (1 + loading(c))). The loading is fitted from the
     prices themselves - log(1 + loading) quadratic in log c, least squares
     on log price, against the fit's own estimates c - so it carries both
     the book's margin and the estimate's lean at that chance. A chance
     outside the fitted range is read at the range's edge: a quadratic
     carried past its data invents margins no book quotes.
The ridge penalty and the prior's weight are chosen by the fit script on a
dev split that never touches the test years (fit 2016-2019, check 2020).

HOW IT IS USED. Live, the frozen fit (data/method_price.json: fitted on
2016-2024, its coefficients hashed with sha256 - load() refuses a file whose
coefficients do not match their hash) prices every prop on the card, and the
card shows it as an estimate beside the measured median error of its leg
type and side (fitted on 2016-2020, tested on 2021-2024; written by
experiments/method_price_fit.py, never typed). In the backtest fit_before(Y)
refits on fights dated before Y only. The rows are read from
experiments/parlay_legs.csv.gz (which joins the props, the moneyline and
the records to each fight) and the archive's division and scheduled rounds;
fit_rows() names every column it reads, and none is a result: the estimator
is fitted to PRICES, never to how fights ended.

NOT AN EDGE. An estimated price says what a book would probably charge. The
six-way market carries about 22% margin across its six outcomes against
about 4% on a moneyline pair, so a prop costs more than a moneyline whatever
its estimate says, and the real price can differ from the estimate by the
measured error or more.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parent
DATA = ENGINE / "data"
COEFFS = DATA / "method_price.json"
LEGS = ENGINE / "experiments" / "parlay_legs.csv.gz"
ARCHIVE = DATA / "UFC_with_mmr_rebuilt_dedup.csv"

# The leg types, in the order parlays/engine/sameday.METHOD_TYPES names them
# (a test holds the two equal); "Decision" is the split's reference.
METHOD_TYPES = ("KO/TKO", "Submission", "Decision")
SIDES = ("red", "blue")
PRICE_COLS = {"KO/TKO": "ko", "Submission": "sub", "Decision": "decision"}

# Every column the fit reads, by file. None is a result: parlay_legs.csv.gz
# also carries winner and method, and the archive the result of every bout;
# all are left unread (a test holds these lists free of RESULT_COLUMNS and
# watches the reads).
LEG_COLUMNS = ("fight_id", "date", "dec_red_close", "dec_blue_close",
               "ko_red", "ko_blue", "sub_red", "sub_blue",
               "decision_red", "decision_blue", "facts_red", "facts_blue")
ARCHIVE_COLUMNS = ("fight_id", "division", "total_rounds")
RESULT_COLUMNS = frozenset({"winner", "method", "finish_round", "finish_time",
                            "finish_round_time", "result", "y", "outcome",
                            "details", "referee", "r_result", "b_result"})

# The weight classes, men's and women's apart. "interim X" reads as X.
DIVISIONS = ("flyweight", "bantamweight", "featherweight", "lightweight",
             "welterweight", "middleweight", "light heavyweight", "heavyweight",
             "women's strawweight", "women's flyweight", "women's bantamweight",
             "women's featherweight")

# (feature, whose record, its count): each a rate per UFC fight before this one.
RECORD_RATES = (("ko_win", "me", "ko_wins"), ("sub_win", "me", "sub_wins"),
                ("opp_ko_loss", "opp", "ko_losses"), ("opp_sub_loss", "opp", "sub_losses"),
                ("opp_ko_win", "opp", "ko_wins"), ("opp_sub_win", "opp", "sub_wins"))

MONEYLINE = ("p", "p2", "p3")
CONTEXT = tuple(f"div:{d}" for d in DIVISIONS) + ("five_rounds",)
RECORDS = tuple(name for name, _, _ in RECORD_RATES) + ("log_fights", "opp_log_fights")
FEATURES = MONEYLINE + CONTEXT + RECORDS

# The baselines the fit script measures beside the estimator, each a subset
# of FEATURES (the scouts' ladder): what each block of inputs is worth.
LADDER = (("constant split", ()),
          ("moneyline alone", MONEYLINE),
          ("+ division and rounds", MONEYLINE + CONTEXT),
          ("+ records (the estimator)", FEATURES))

# The hyperparameter grids the fit script searches on its dev split; the
# chosen pair is written to data/method_price.json and read back from there.
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
PRIOR_GRID = (1.0, 3.0, 5.0, 10.0, 20.0)

# Fewer usable six-way books than this and fit() refuses: a split of 25
# inputs over a couple of cards is not an estimate.
MIN_FIT_FIGHTS = 200

# A chance is kept off 0 and 1 before any logit or log.
_EPS = 1e-6


# ---------------------------------------------------------------- prices

def _pp():
    import parlay_profiles as pp
    return pp


def moneyline_chance(dec_red, dec_blue):
    """Red's no-vig moneyline chance (proportional; parlay_profiles.devig,
    which refuses an overround below 0 or above 15%), or None."""
    pair = _pp().devig(dec_red, dec_blue)
    return None if pair is None else float(pair[0])


def six_way(row):
    """The six props de-vigged together, {"red": {type: q}, "blue": {...}},
    or None unless all six are priced and the set is a believable book
    (parlay_profiles._method_market: an overround of 0-40%)."""
    return _pp()._method_market({f"{PRICE_COLS[t]}_{s}": row.get(f"{PRICE_COLS[t]}_{s}")
                                 for t in METHOD_TYPES for s in SIDES})


def american(decimal):
    """A decimal price as American, rounded as books quote it; None for none."""
    return _pp().american_from_decimal(decimal)


# ---------------------------------------------------------------- inputs

def normal_division(division):
    """A division as one of DIVISIONS, or None (catch weight, open weight,
    a tournament bracket, nothing named)."""
    if division is None or (isinstance(division, float) and math.isnan(division)):
        return None
    text = str(division).replace("’", "'").replace("‘", "'").strip().lower()
    text = " ".join(text.split())
    if text.startswith("interim "):
        text = text[len("interim "):]
    if text.startswith("womens "):
        text = "women's " + text[len("womens "):]
    return text if text in DIVISIONS else None


def _facts(f):
    """A parlay_profiles.facts record (dict or its JSON) as the counts read
    here; anything missing or unreadable counts as 0 - a debut."""
    if isinstance(f, str):
        f = json.loads(f) if f.strip() else {}
    f = f or {}
    out = {}
    for k in ("fights", "ko_wins", "sub_wins", "ko_losses", "sub_losses"):
        try:
            v = float(f.get(k) or 0)
        except (TypeError, ValueError):
            v = 0.0
        out[k] = v if math.isfinite(v) and v >= 0 else 0.0
    return out


def _rounds(r):
    try:
        v = float(r)
    except (TypeError, ValueError):
        return None
    return int(v) if math.isfinite(v) else None


def side_vector(p, division, rounds_scheduled, facts_me, facts_opp, prior, prior_fights):
    """One side's inputs in FEATURES order. `prior` maps each RECORD_RATES
    name to the pooled rate it is shrunk toward."""
    me, opp = _facts(facts_me), _facts(facts_opp)
    div = normal_division(division)
    x = [p, p * p, p * p * p]
    x += [1.0 if div == d else 0.0 for d in DIVISIONS]
    x.append(1.0 if _rounds(rounds_scheduled) == 5 else 0.0)
    for name, who, count in RECORD_RATES:
        f = me if who == "me" else opp
        x.append((f[count] + prior_fights * prior[name]) / (f["fights"] + prior_fights))
    x += [math.log1p(me["fights"]), math.log1p(opp["fights"])]
    return x


def _logit(q):
    q = min(max(float(q), _EPS), 1.0 - _EPS)
    return math.log(q / (1.0 - q))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- rows

def fit_rows(legs_path=LEGS, archive_path=ARCHIVE):
    """The fights a fit can use, oldest first: dicts with fight_id, date,
    dec_red/dec_blue (the closing moneyline), the six props in decimal
    (ko_red ... decision_blue), facts_red/facts_blue (records strictly before
    the fight, as parlay_legs wrote them), division and rounds_scheduled.
    Only LEG_COLUMNS and ARCHIVE_COLUMNS are read - never a result. A fight
    is kept only with a usable moneyline pair and a believable six-way book."""
    import pandas as pd

    legs = pd.read_csv(legs_path, usecols=list(LEG_COLUMNS))
    arch = pd.read_csv(archive_path, usecols=list(ARCHIVE_COLUMNS), low_memory=False)
    arch = arch.drop_duplicates("fight_id").set_index("fight_id")
    rows = []
    for r in legs.to_dict("records"):
        row = {"fight_id": r["fight_id"], "date": str(r["date"])[:10],
               "dec_red": r["dec_red_close"], "dec_blue": r["dec_blue_close"],
               **{f"{PRICE_COLS[t]}_{s}": r[f"{PRICE_COLS[t]}_{s}"]
                  for t in METHOD_TYPES for s in SIDES},
               "facts_red": _facts(r["facts_red"]), "facts_blue": _facts(r["facts_blue"])}
        if moneyline_chance(row["dec_red"], row["dec_blue"]) is None or six_way(row) is None:
            continue
        known = r["fight_id"] in arch.index
        row["division"] = arch.at[r["fight_id"], "division"] if known else None
        row["rounds_scheduled"] = _rounds(arch.at[r["fight_id"], "total_rounds"]) if known else None
        rows.append(row)
    rows.sort(key=lambda x: (x["date"], str(x["fight_id"])))
    return rows


def features_of(row):
    """A fit row as predict() reads it: the inputs only, no prop price."""
    return {"p_red": moneyline_chance(row["dec_red"], row["dec_blue"]),
            "division": row.get("division"), "rounds_scheduled": row.get("rounds_scheduled"),
            "facts_red": row.get("facts_red"), "facts_blue": row.get("facts_blue")}


def usable(rows):
    """The rows a fit or a test can read: a moneyline chance and a six-way book."""
    return [r for r in rows if moneyline_chance(r["dec_red"], r["dec_blue"]) is not None
            and six_way(r) is not None]


# ---------------------------------------------------------------- fit

def _pool_prior(rows):
    """Each RECORD_RATES rate pooled over every corner in `rows` (count per
    fight). The opponent's rates pool the same corners: across the rows each
    corner is the other's opponent exactly once."""
    fights, counts = 0.0, {count: 0.0 for _, _, count in RECORD_RATES}
    for row in rows:
        for side in SIDES:
            f = _facts(row.get(f"facts_{side}"))
            fights += f["fights"]
            # each count once a corner, however many rates read it
            for count in counts:
                counts[count] += f[count]
    return {name: (counts[count] / fights if fights > 0 else 0.0)
            for name, _, count in RECORD_RATES}


def fit(rows, alpha, prior_fights, use=FEATURES):
    """The estimator fitted on `rows` (fit_rows' dicts) with the inputs named
    in `use` (a subset of FEATURES; LADDER's baselines use fewer): a dict of
    plain numbers, json-ready, that predict() reads. Raises ValueError with
    fewer than MIN_FIT_FIGHTS usable six-way books."""
    rows = usable(rows)
    if len(rows) < MIN_FIT_FIGHTS:
        raise ValueError(f"{len(rows)} usable six-way books, fewer than {MIN_FIT_FIGHTS}")
    use = [f for f in FEATURES if f in set(use)]
    cols = [FEATURES.index(f) for f in use]
    prior = _pool_prior(rows)

    xs, lp, lq, ratios = [], [], [], []
    for row in rows:
        p_red, book = moneyline_chance(row["dec_red"], row["dec_blue"]), six_way(row)
        for side, other, p in (("red", "blue", p_red), ("blue", "red", 1.0 - p_red)):
            q = book[side]
            Q = sum(q.values())
            x = side_vector(p, row.get("division"), row.get("rounds_scheduled"),
                            row.get(f"facts_{side}"), row.get(f"facts_{other}"),
                            prior, prior_fights)
            xs.append([x[i] for i in cols])
            lp.append(_logit(p))
            lq.append(_logit(Q))
            ratios.append([math.log(max(q["KO/TKO"], _EPS) / max(q["Decision"], _EPS)),
                           math.log(max(q["Submission"], _EPS) / max(q["Decision"], _EPS))])

    # 1. side chance: logit(Q) = b * logit(p), through the origin
    lp, lq = np.array(lp), np.array(lq)
    slope = float(lp @ lq / (lp @ lp))

    # 2. split: ridge on standardised inputs, two log-ratios against decision
    Y = np.array(ratios)
    y_mean = Y.mean(axis=0)
    if cols:
        X = np.array(xs, dtype=float)
        mean, scale = X.mean(axis=0), X.std(axis=0)
        scale[scale == 0] = 1.0
        Z = (X - mean) / scale
        W = np.linalg.solve(Z.T @ Z + float(alpha) * np.eye(len(cols)), Z.T @ (Y - y_mean))
    else:
        mean, scale, W = np.zeros(0), np.ones(0), np.zeros((0, 2))
    model = {
        "features": use, "method_types": list(METHOD_TYPES),
        "alpha": float(alpha), "prior_fights": float(prior_fights),
        "prior": {k: float(v) for k, v in sorted(prior.items())},
        "side_slope": slope,
        "mean": [float(v) for v in mean], "scale": [float(v) for v in scale],
        "intercept": {"KO/TKO": float(y_mean[0]), "Submission": float(y_mean[1])},
        "weights": {"KO/TKO": [float(v) for v in W[:, 0]],
                    "Submission": [float(v) for v in W[:, 1]]},
        "loading": [0.0, 0.0, 0.0], "loading_range": [math.log(_EPS), 0.0],
        "fit_fights": len(rows), "fit_first": rows[0]["date"], "fit_last": rows[-1]["date"],
    }

    # 3. loading from the prices: log(1 + m) = g0 + g1 L + g2 L^2, L = log c,
    #    c the fit's own estimate (so the curve also absorbs its lean)
    L, target = [], []
    for row in rows:
        est = predict(features_of(row), model)
        for side in SIDES:
            for t in METHOD_TYPES:
                c = est[side][t]["market_chance"]
                L.append(math.log(c))
                target.append(-math.log(_pp()._price(row[f"{PRICE_COLS[t]}_{side}"])) - math.log(c))
    L = np.array(L)
    g = np.linalg.lstsq(np.column_stack([np.ones_like(L), L, L * L]), np.array(target), rcond=None)[0]
    model["loading"] = [float(v) for v in g]
    model["loading_range"] = [float(L.min()), float(L.max())]
    return model


def loading(chance, model):
    """The margin on a prop the estimate rates `chance`: the price is
    1 / (chance x (1 + loading)). Read at the edge of the fitted range of
    log chance outside it."""
    lo, hi = model["loading_range"]
    L = min(max(math.log(min(max(float(chance), _EPS), 1.0)), lo), hi)
    g0, g1, g2 = model["loading"]
    return math.exp(g0 + g1 * L + g2 * L * L) - 1.0


def predict(features, model=None):
    """{"red": {type: {"share", "market_chance", "decimal"}}, "blue": {...}}
    for one fight. features: p_red (red's no-vig moneyline chance), division,
    rounds_scheduled, facts_red, facts_blue (records strictly before the
    fight). `share` is the side's wins going that way, `market_chance` the
    estimate of the six-way market's de-vigged chance of the prop (side
    chance x share), `decimal` the estimated price. model None: the frozen
    fit (load()). Raises ValueError without a moneyline chance in (0, 1)."""
    model = load() if model is None else model
    p_red = features.get("p_red")
    try:
        p_red = float(p_red)
    except (TypeError, ValueError):
        p_red = float("nan")
    if not (math.isfinite(p_red) and 0.0 < p_red < 1.0):
        raise ValueError("no moneyline chance: the estimator reads the market's no-vig chance")
    cols = [FEATURES.index(f) for f in model["features"]]
    mean, scale = np.array(model["mean"]), np.array(model["scale"])
    q_red = _sigmoid(model["side_slope"] * _logit(p_red))
    out = {}
    for side, other, p, Q in (("red", "blue", p_red, q_red), ("blue", "red", 1.0 - p_red, 1.0 - q_red)):
        x = side_vector(p, features.get("division"), features.get("rounds_scheduled"),
                        features.get(f"facts_{side}"), features.get(f"facts_{other}"),
                        model["prior"], model["prior_fights"])
        z = (np.array([x[i] for i in cols]) - mean) / scale
        logits = {t: model["intercept"][t] + float(z @ np.array(model["weights"][t]))
                  for t in ("KO/TKO", "Submission")}
        logits["Decision"] = 0.0
        top = max(logits.values())
        ex = {t: math.exp(v - top) for t, v in logits.items()}
        total = sum(ex.values())
        out[side] = {}
        for t in METHOD_TYPES:
            share = ex[t] / total
            c = Q * share
            out[side][t] = {"share": share, "market_chance": c,
                            "decimal": 1.0 / (c * (1.0 + loading(c, model)))}
    return out


# ---------------------------------------------------------------- frozen

def coefficient_hash(model):
    """sha256 of the coefficients as canonical JSON (sorted keys, no spaces,
    floats as Python writes them, which read back exactly)."""
    text = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FrozenMismatch(RuntimeError):
    """The frozen file's coefficients do not match their recorded hash."""



def load(path=COEFFS):
    """The frozen fit from data/method_price.json, its hash checked: a file
    edited by hand (or half written) is refused, never used."""
    path = Path(path)
    doc = json.loads(path.read_text())
    model = doc["frozen"]["model"]
    if coefficient_hash(model) != doc["frozen"]["sha256"]:
        raise FrozenMismatch(f"{path.name}: the coefficients do not match their sha256")
    return model


def measured(path=COEFFS):
    """The whole measured file (frozen fit, test error table, dev choice,
    coverage), or None when it has not been written or cannot be read."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def chosen(doc=None):
    """(alpha, prior_fights) the fit script chose on its dev split."""
    doc = measured() if doc is None else doc
    c = (doc or {}).get("dev_choice") or {}
    if c.get("alpha") is None or c.get("prior_fights") is None:
        raise ValueError("no dev choice on file: run experiments/method_price_fit.py")
    return float(c["alpha"]), float(c["prior_fights"])


def fit_before(year, rows=None, doc=None):
    """The estimator refitted on fights dated before 1 January `year` only
    (the backtest's yearly refit), at the dev split's hyperparameters; None
    when fewer than MIN_FIT_FIGHTS usable books precede it. The fit rows start
    in 2016 (parlay_legs.csv.gz), so fit_before(2016) is None: a 2016 card
    has no earlier props here to be priced from."""
    rows = fit_rows() if rows is None else rows
    cut = f"{int(year):04d}-01-01"
    alpha, prior_fights = chosen(doc)
    try:
        return fit([r for r in rows if str(r["date"])[:10] < cut], alpha, prior_fights)
    except ValueError:
        return None


# ---------------------------------------------------------------- error

def side_of(p):
    """"fav" when the side's no-vig moneyline chance is at least 0.5, else
    "dog" (parlay_profiles' favourite rule)."""
    return "fav" if p >= 0.5 else "dog"


def errors(rows, model):
    """One entry per prop in the usable `rows`: (type, "fav"/"dog", the
    estimate, the real price, |estimate - price| / price)."""
    out = []
    for row in usable(rows):
        feats = features_of(row)
        est = predict(feats, model)
        for side, p in (("red", feats["p_red"]), ("blue", 1.0 - feats["p_red"])):
            for t in METHOD_TYPES:
                d = _pp()._price(row[f"{PRICE_COLS[t]}_{side}"])
                e = est[side][t]["decimal"]
                out.append((t, side_of(p), e, d, abs(e - d) / d))
    return out


def error_table(errs):
    """{"by_leg": {type: {"fav": median, "dog": median}}, "n": {...},
    "all_single_props": median} of errors()' relative errors."""
    by, n = {}, {}
    for t in METHOD_TYPES:
        by[t], n[t] = {}, {}
        for s in ("fav", "dog"):
            e = [x[-1] for x in errs if x[0] == t and x[1] == s]
            by[t][s] = float(np.median(e)) if e else None
            n[t][s] = len(e)
    allv = [x[-1] for x in errs]
    return {"by_leg": by, "n": n, "all_single_props": float(np.median(allv)) if allv else None,
            "props": len(allv)}
