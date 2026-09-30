"""Rate every fighter in every promotion, and predict fights the UFC model
cannot: the Contender Series, and anyone without UFC history.

The UFC model reads UFCStats - strikes, takedowns, control time - which
exist for no other promotion. What exists for all of them is the result:
who beat whom, how, and when. So this is a results model, deliberately
simple:

  RATING. An Elo rating per fighter, updated bout by bout in date order
  across every promotion in world_bouts.csv.gz. Strength flows along shared
  opponents, so a Cage Warriors champion's wins over men who later won in
  the UFC are worth more than a 10-0 record against nobody. A new fighter
  starts at BASE; the step size shrinks with experience.

  RECORD FEATURES, as of the morning of the bout and never after it:
  bouts, win rate (shrunk toward .5 for small samples), finish rate, KO
  losses, current streak, days since last bout, and the average rating of
  opponents beaten.

  MODEL. Logistic regression on the differences, fitted walk-forward: every
  year is predicted by a model trained only on the years before it, and
  that is the only evaluation reported. Scored on all bouts, on non-UFC
  bouts, and on Contender Series bouts alone.

  SHERDOG (--contender). Prospects' regional careers, from Sherdog
  (harvest_sherdog.py), for both fighters of every Contender Series bout.
  A record is used only from the day its owner first appeared on the
  Contender Series - the day anyone would have looked them up - and only
  its bouts dated before the bout being predicted. Scored per event, each
  from a model fitted on bouts before that event, with and without Sherdog.

    python engine/world_model.py --evaluate
    python engine/world_model.py --contender
    python engine/world_model.py --predict "Fighter A" "Fighter B"
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from name_resolution import norm_name

BOUTS = ENGINE / "data" / "world_bouts.csv.gz"
OUT = ENGINE / "data" / "world_model_eval.json"
SHERDOG = ENGINE / "data" / "sherdog_records.jsonl.gz"
CONTENDER_OUT = ENGINE / "data" / "world_model_contender.json"
BASE = 1500.0
K_NEW, K_OLD, K_SETTLE = 64.0, 24.0, 12    # K falls from K_NEW to K_OLD
                                            # over K_SETTLE bouts
PRIOR_BOUTS = 4                             # win-rate shrinkage strength
FEATURES = ["elo_diff", "exp_diff", "winrate_diff", "finish_diff",
            "ko_loss_diff", "streak_diff", "layoff_diff", "beaten_diff",
            "exp_min"]
FINISH = ("ko", "tko", "submission")


def _k(bouts):
    share = min(bouts, K_SETTLE) / K_SETTLE
    return K_NEW + (K_OLD - K_NEW) * share


def _expect(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


class State:
    """Everything known about one fighter, as of their last bout."""
    __slots__ = ("elo", "bouts", "wins", "finishes", "ko_losses", "streak",
                 "last", "beaten")

    def __init__(self):
        self.elo, self.bouts, self.wins, self.finishes = BASE, 0, 0, 0
        self.ko_losses, self.streak, self.last = 0, 0, None
        self.beaten = []

    def features(self, date):
        winrate = (self.wins + PRIOR_BOUTS * 0.5) / (self.bouts + PRIOR_BOUTS)
        finish = (self.finishes + 1) / (self.wins + 2)
        layoff = ((date - self.last).days if self.last is not None
                  else 365) / 365.0
        beaten = float(np.mean(self.beaten[-10:])) if self.beaten else BASE
        return {"elo": self.elo, "exp": math.log1p(self.bouts),
                "winrate": winrate, "finish": finish,
                "ko_loss": self.ko_losses, "streak": self.streak,
                "layoff": min(layoff, 5.0), "beaten": beaten}


def _orient(key):
    """A fixed, name-independent coin: which fighter is 'A' in a row.

    The winner is always listed first on Wikipedia; left there, the label
    would be 1 for every row and the model would learn the page layout.
    """
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0


def load(path=BOUTS):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "winner", "loser"])
    # A bout dated after today is a typo on the page it came from (two were
    # found, "2027" and "2028", in fighters' own record tables).
    frame = frame[frame["date"] <= pd.Timestamp.today().normalize()]
    frame = frame[frame["result"].isin(["win", "draw", "nc"])]
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def results_only(frame):
    """Bouts from event and results pages only.

    THE LEAK THIS CLOSES. A fighter's own article - with their whole
    regional record in it - usually exists BECAUSE they went on to succeed.
    Bouts known only from those tables therefore favour the fighters who
    later won: at a Contender Series bout the eventual winner shows ten
    recorded fights and the loser two, and "more recorded fights" quietly
    encodes the future. Event and results pages list every bout on a card,
    whoever later became famous, so their coverage does not depend on the
    outcome being predicted.
    """
    return frame[~frame["source"].astype(str).str.startswith("record:")]


def build(frame, births=None):
    """One row per decided bout, with both fighters' pre-bout features.

    Walks the bouts in date order. Features are read BEFORE the bout
    updates anything, so no row ever sees its own result.

    births: {normalised name: birth date} - when given, rows carry age_diff
    (decades, 0 unless BOTH ages are known).
    """
    states = defaultdict(State)
    rows = []
    for bout in frame.itertuples():
        w, l = norm_name(bout.winner), norm_name(bout.loser)
        if not w or not l or w == l:
            continue
        a_is_winner = _orient(f"{bout.date.date()}|{min(w, l)}|{max(w, l)}")
        a, b = (w, l) if a_is_winner else (l, w)
        fa, fb = states[a].features(bout.date), states[b].features(bout.date)
        method = str(bout.method).lower()
        if bout.result == "win":
            rows.append({
                "date": bout.date, "event": bout.event, "a": a, "b": b,
                "y": 1 if a_is_winner else 0,
                "elo_diff": (fa["elo"] - fb["elo"]) / 100.0,
                "exp_diff": fa["exp"] - fb["exp"],
                "exp_min": min(fa["exp"], fb["exp"]),
                "winrate_diff": fa["winrate"] - fb["winrate"],
                "finish_diff": fa["finish"] - fb["finish"],
                "ko_loss_diff": fa["ko_loss"] - fb["ko_loss"],
                "streak_diff": fa["streak"] - fb["streak"],
                "layoff_diff": fa["layoff"] - fb["layoff"],
                "beaten_diff": (fa["beaten"] - fb["beaten"]) / 100.0,
                "p_elo": _expect(fa["elo"], fb["elo"]),
                "age_diff": _age_diff(births, a, b),
                "dwcs": "contender series" in str(bout.event).lower()
                        or "contender series" in str(bout.source).lower(),
                "ufc": str(bout.event).upper().startswith("UFC"),
            })
        # --- update, after the row is written -------------------------
        sw, sl = states[w], states[l]
        if bout.result == "nc":
            for s in (sw, sl):
                s.last = bout.date
            continue
        score = 0.5 if bout.result == "draw" else 1.0
        ew = _expect(sw.elo, sl.elo)
        dw = _k(sw.bouts) * (score - ew)
        dl = _k(sl.bouts) * ((1 - score) - (1 - ew))
        loser_elo_before = sl.elo
        sw.elo += dw
        sl.elo += dl
        for s in (sw, sl):
            s.bouts += 1
            s.last = bout.date
        if bout.result == "win":
            sw.wins += 1
            sw.streak = sw.streak + 1 if sw.streak >= 0 else 1
            sl.streak = sl.streak - 1 if sl.streak <= 0 else -1
            sw.beaten.append(loser_elo_before)
            if any(m in method for m in FINISH):
                sw.finishes += 1
            if "ko" in method:
                sl.ko_losses += 1
    return pd.DataFrame(rows), states


def sherdog_bouts(records):
    """Sherdog records as bouts, each stamped with when it became knowable.

    records: rows of sherdog_records.jsonl.gz. The record's owner is named
    as the Contender Series page names them (the name the rest of the world
    frame uses); an opponent who is also a looked-up fighter is named the
    same way, through their Sherdog ID. A bout between two looked-up
    fighters is in both records and kept once.
    """
    by_id = {r["id"]: r["target"] for r in records if r.get("id")}
    rows, seen = [], set()
    for r in records:
        if not r.get("id") or not r.get("first_contender"):
            continue
        owner = r["target"]
        for bout in r.get("record", []):
            other = by_id.get(bout["opponent_id"], bout["opponent"])
            key = (bout["date"], frozenset((norm_name(owner), norm_name(other))))
            if key in seen:
                continue
            seen.add(key)
            won = bout["result"] != "loss"
            result = bout["result"] if bout["result"] in ("draw", "nc") else "win"
            rows.append({"date": bout["date"], "event": bout["event"],
                         "winner": owner if won else other,
                         "loser": other if won else owner,
                         "result": result, "method": bout["method"],
                         "round": bout["round"], "time": bout["time"],
                         "source": f"sherdog:{owner}",
                         "disclosed": r["first_contender"]})
    frame = pd.DataFrame(rows, columns=["date", "event", "winner", "loser",
                                        "result", "method", "round", "time",
                                        "source", "disclosed"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["disclosed"] = pd.to_datetime(frame["disclosed"], errors="coerce")
    # When two records disagree on a date by a day (time zones), a bout can
    # survive the exact-date check twice; the world-frame check below uses
    # a window and catches the rest against event pages.
    return frame.dropna(subset=["date", "disclosed"])


def not_in(extra, frame, days=2):
    """Rows of `extra` that are not already a bout of `frame` - same pair of
    fighters within `days` days. Event pages win: they are the label."""
    dates = defaultdict(list)
    for w, l, d in zip(frame["winner"].map(norm_name),
                       frame["loser"].map(norm_name), frame["date"]):
        dates[frozenset((w, l))].append(d)
    keep = []
    window = pd.Timedelta(days=days)
    for w, l, d in zip(extra["winner"].map(norm_name),
                       extra["loser"].map(norm_name), extra["date"]):
        keep.append(not any(abs(d - x) <= window
                            for x in dates.get(frozenset((w, l)), ())))
    return extra[np.array(keep, dtype=bool)] if len(extra) else extra


def gated(world, extra, when):
    """The world as known the day before `when`: its bouts up to `when`,
    plus the extra bouts disclosed by then and dated before it."""
    part = extra[(extra["disclosed"] <= when) & (extra["date"] < when)]
    frame = pd.concat([world[world["date"] <= when],
                       part.drop(columns="disclosed")], ignore_index=True)
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def _fit_predict(rows, when, features=FEATURES):
    from sklearn.linear_model import LogisticRegression
    train = rows[rows["date"] < when]
    test = rows[(rows["date"] == when) & rows["dwcs"]]
    if len(train) < 2000 or test.empty:
        return test.assign(p_model=np.nan).iloc[0:0]
    model = LogisticRegression(C=1.0, max_iter=1000).fit(train[features],
                                                         train["y"])
    return test.assign(p_model=model.predict_proba(test[features])[:, 1])


def evaluate_contender(world, extra, first_year=2017, births=None):
    """Every Contender Series bout predicted by the same fitting protocol:
    the world alone, the world plus Sherdog as known then, and - when
    birth dates are given - that plus the age gap.

    births: {normalised name: (birth date, date it became known)}.
    """
    rows_base, _ = build(world)
    days = sorted(rows_base.loc[rows_base["dwcs"]
                                & (rows_base["date"].dt.year >= first_year),
                                "date"].unique())
    extra = not_in(extra, world)
    base, with_sherdog, with_age = [], [], []
    for when in days:
        when = pd.Timestamp(when)
        base.append(_fit_predict(rows_base[rows_base["date"] <= when], when))
        known = {k: born for k, (born, disclosed) in (births or {}).items()
                 if disclosed <= when}
        rows, _ = build(gated(world, extra, when), known)
        with_sherdog.append(_fit_predict(rows, when))
        if births:
            with_age.append(_fit_predict(rows, when, FEATURES + ["age_diff"]))
    return (pd.concat(base), pd.concat(with_sherdog),
            pd.concat(with_age) if with_age else None)


def births_of(records):
    """{normalised name: (birth date, first Contender Series date)}."""
    out = {}
    for r in records:
        born = (r.get("bio") or {}).get("birth_date")
        if r.get("id") and born and r.get("first_contender"):
            out[norm_name(r["target"])] = (pd.Timestamp(born),
                                           pd.Timestamp(r["first_contender"]))
    return out


def paired(base, other, draws=2000, seed=0):
    """Log-loss difference (other minus base) on the bouts both scored, with
    a bootstrap interval that resamples whole events."""
    key = ["date", "a", "b"]
    m = base[key + ["y", "p_model", "p_elo"]].merge(
        other[key + ["p_model", "p_elo"]], on=key, suffixes=("_base", "_new"))
    m = m.dropna(subset=["p_model_base", "p_model_new"])
    def loss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(m["y"] * np.log(p) + (1 - m["y"]) * np.log(1 - p))
    diff = (loss(m["p_model_new"]) - loss(m["p_model_base"])).to_numpy()
    events = m["date"].to_numpy()
    groups = [np.flatnonzero(events == e) for e in np.unique(events)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        means.append(diff[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": int(len(m)), "events": len(groups),
            "logloss_diff": float(diff.mean()), "lo": float(lo),
            "hi": float(hi)}


def load_sherdog(path=SHERDOG):
    import gzip
    if not path.exists():
        return []
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contender_report(world):
    records = load_sherdog()
    if not records:
        print("  no Sherdog records yet - run harvest_sherdog.py")
        return 1
    extra = sherdog_bouts(records)
    found = sum(1 for r in records if r.get("id"))
    verified = sum(1 for r in records if r.get("verified"))
    print(f"\n  Sherdog: {len(records)} fighters looked up, {found} found, "
          f"{verified} verified against their Contender Series bout; "
          f"{len(extra):,} bouts ({len(not_in(extra, world)):,} not already "
          f"on an event page)\n")
    births = births_of(records)
    base, new, aged = evaluate_contender(world, extra, births=births)
    results = [score(base, base["dwcs"], "world only"),
               score(new, new["dwcs"], "world + Sherdog (as known then)")]
    if aged is not None:
        results.append(score(aged, aged["dwcs"], "  + age gap (Sherdog birth dates)"))
    print(f"  {'Contender Series':<34}{'n':>6}{'logloss':>10}{'elo only':>10}"
          f"{'coin':>8}{'acc':>8}")
    for r in results:
        print(f"  {r['label']:<34}{r['n']:>6}{r['logloss_model']:>10.4f}"
              f"{r['logloss_elo']:>10.4f}{r['logloss_coin']:>8.4f}"
              f"{r['accuracy_model']:>8.3f}")
    gap = paired(base, new)
    print(f"\n  with Sherdog minus without: {gap['logloss_diff']:+.4f} log loss "
          f"on {gap['n']} bouts, 95% interval (by event) "
          f"[{gap['lo']:+.4f}, {gap['hi']:+.4f}] - negative is better")
    age_gap = paired(new, aged) if aged is not None else None
    if age_gap:
        print(f"  adding the age gap: {age_gap['logloss_diff']:+.4f} "
              f"[{age_gap['lo']:+.4f}, {age_gap['hi']:+.4f}] "
              f"({len(births)} birth dates)")
    covered = {r["target"] for r in records if r.get("verified")}
    both = new[new["a"].isin({norm_name(n) for n in covered})
               & new["b"].isin({norm_name(n) for n in covered})]
    if len(both):
        r = score(both, both["dwcs"], "both fighters verified")
        print(f"  bouts with both fighters verified: n={r['n']}, log loss "
              f"{r['logloss_model']:.4f}, accuracy {r['accuracy_model']:.3f}")
    CONTENDER_OUT.write_text(json.dumps({"results": results, "paired": gap,
                                         "paired_age": age_gap,
                                         "looked_up": len(records),
                                         "found": found,
                                         "verified": verified}, indent=1))
    print(f"\n  wrote {CONTENDER_OUT.name}. Each event predicted by a model "
          f"fitted only on bouts before it.")
    return 0


def _age_diff(births, a, b):
    if not births or a not in births or b not in births:
        return 0.0
    years = (births[b] - births[a]).days / 365.25    # a older -> positive
    return float(np.clip(years, -15, 15)) / 10.0


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward(rows, first_year=2012, min_train=2000):
    """Out-of-sample predictions: each year from a model of earlier years."""
    from sklearn.linear_model import LogisticRegression
    rows = rows.copy()
    rows["year"] = rows["date"].dt.year
    rows["p_model"] = np.nan
    for year in sorted(rows["year"].unique()):
        if year < first_year:
            continue
        train = rows[rows["year"] < year]
        test = rows["year"] == year
        if len(train) < min_train or not test.any():
            continue
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(train[FEATURES], train["y"])
        rows.loc[test, "p_model"] = model.predict_proba(
            rows.loc[test, FEATURES])[:, 1]
    return rows


def score(rows, mask, label):
    part = rows[mask & rows["p_model"].notna()]
    if part.empty:
        return {"label": label, "n": 0}
    y = part["y"].to_numpy()
    out = {"label": label, "n": int(len(part)),
           "logloss_model": _logloss(y, part["p_model"].to_numpy()),
           "logloss_elo": _logloss(y, part["p_elo"].to_numpy()),
           "logloss_coin": math.log(2),
           "accuracy_model": float(((part["p_model"] > .5) == (y == 1)).mean()),
           "accuracy_elo": float(((part["p_elo"] > .5) == (y == 1)).mean())}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--predict", nargs=2, metavar=("A", "B"))
    parser.add_argument("--contender", action="store_true",
                        help="Contender Series with and without Sherdog "
                             "records, as known at each event")
    parser.add_argument("--all-sources", action="store_true",
                        help="also use bouts known only from fighters' own "
                             "record tables (leaks - see RESULTS_ONLY)")
    args = parser.parse_args(argv)

    frame = load()
    if not args.all_sources:
        frame = results_only(frame)
    if args.contender:
        return contender_report(frame)
    rows, states = build(frame)
    if args.predict:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0, max_iter=1000).fit(rows[FEATURES],
                                                             rows["y"])
        today = pd.Timestamp.today().normalize()
        a, b = (norm_name(n) for n in args.predict)
        fa, fb = states[a].features(today), states[b].features(today)
        x = pd.DataFrame([{
            "elo_diff": (fa["elo"] - fb["elo"]) / 100.0,
            "exp_diff": fa["exp"] - fb["exp"],
            "exp_min": min(fa["exp"], fb["exp"]),
            "winrate_diff": fa["winrate"] - fb["winrate"],
            "finish_diff": fa["finish"] - fb["finish"],
            "ko_loss_diff": fa["ko_loss"] - fb["ko_loss"],
            "streak_diff": fa["streak"] - fb["streak"],
            "layoff_diff": fa["layoff"] - fb["layoff"],
            "beaten_diff": (fa["beaten"] - fb["beaten"]) / 100.0}])
        p = float(model.predict_proba(x[FEATURES])[0, 1])
        for name, s in ((args.predict[0], states[a]), (args.predict[1], states[b])):
            print(f"  {name:<28} elo {s.elo:7.1f}  bouts {s.bouts:3d}  "
                  f"wins {s.wins:3d}")
        print(f"\n  P({args.predict[0]} wins) = {p:.3f}")
        if min(states[a].bouts, states[b].bouts) == 0:
            print("  one fighter has no recorded bouts - this is a guess "
                  "about an unknown, not a rating")
        return 0

    rows = walk_forward(rows)
    results = [score(rows, rows["p_model"].notna(), "all bouts"),
               score(rows, ~rows["ufc"], "non-UFC bouts"),
               score(rows, rows["ufc"], "UFC bouts"),
               score(rows, rows["dwcs"], "Contender Series bouts")]
    print(f"\n  {len(frame):,} bouts, {len(rows):,} decided, "
          f"{len(states):,} fighters rated\n")
    print(f"  {'':<26}{'n':>7}{'logloss':>10}{'elo only':>10}{'coin':>8}"
          f"{'acc':>8}{'acc elo':>9}")
    for r in results:
        if not r["n"]:
            print(f"  {r['label']:<26}{0:>7}")
            continue
        print(f"  {r['label']:<26}{r['n']:>7}{r['logloss_model']:>10.4f}"
              f"{r['logloss_elo']:>10.4f}{r['logloss_coin']:>8.4f}"
              f"{r['accuracy_model']:>8.3f}{r['accuracy_elo']:>9.3f}")
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {OUT.name}. Every number is out of sample: each year "
          f"predicted from earlier years only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
