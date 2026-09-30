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

    python engine/world_model.py --evaluate
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


def build(frame):
    """One row per decided bout, with both fighters' pre-bout features.

    Walks the bouts in date order. Features are read BEFORE the bout
    updates anything, so no row ever sees its own result.
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
    parser.add_argument("--all-sources", action="store_true",
                        help="also use bouts known only from fighters' own "
                             "record tables (leaks - see RESULTS_ONLY)")
    args = parser.parse_args(argv)

    frame = load()
    if not args.all_sources:
        frame = results_only(frame)
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
