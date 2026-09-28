"""Is the method model behind its market the way the winner model was?

The winner model turned out to be four points behind the closing line, and
blending the two bought six points of accuracy - the largest single gain in
this project, for arithmetic. The method model has never been compared with a
market at all, because nobody had one: the live api prices no method market
on this plan, and experiments/method_doubt.py concluded from that there was no
second opinion to read.

There is. The historical dataset carries method-of-victory props on about
5,200 fights, found by check_line_history.py while it was looking for an
opening price. This asks the same question of them that market_blend.py asked
of the moneyline, in the same shape, so the two answers are comparable:

    the model alone        walk-forward, each year from earlier years only
    the market alone       de-vigged from the six prop prices
    blended                weight chosen from EARLIER years and applied
                           forward, never fitted on what it is judged on

Reported on FINISH-OR-DISTANCE, because that is the call the app now leads
with and the one a three-way argmax obscures.

THE MARGIN IS THE CATCH. A moneyline pair implies about 104% between the two
sides. These six prices imply 122% between them, so the bookmaker's margin is
five times wider and removing it proportionally across six outcomes is a
cruder operation than splitting two. A market this expensive to bet may also
be a market priced with less care. The measurement is the point; the caveat
travels with it.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.method_model import (CLASSES, CONFIRM_FROM,
                                      FIRST_PREDICTED_YEAR, level_columns,
                                      rate_columns, walk_forward)
from name_resolution import norm_name

METHOD_ODDS = ENGINE / "data" / "method_odds.csv"
PROP_COLUMNS = ("dec_a", "dec_b", "ko_a", "ko_b", "sub_a", "sub_b")
WEIGHTS = np.linspace(0, 1, 21)


def implied(american):
    """What one American price implies, margin included."""
    american = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(american > 0, 100.0 / (american + 100.0),
                        -american / (-american + 100.0))


def market_finish_probability(frame):
    """P(finish) from the six props, with the margin removed.

    The six outcomes cover the whole space between them, so they de-vig
    together: scale all six to sum to one and add up the two that are not a
    decision. Proportional de-vigging across six is cruder than across two,
    and this is where that crudeness lives.
    """
    prices = np.column_stack([implied(frame[c]) for c in PROP_COLUMNS])
    total = prices.sum(axis=1)
    fair = prices / total[:, None]
    return 1.0 - (fair[:, 0] + fair[:, 1]), total


def pair_key(a, b, date):
    return tuple(sorted((norm_name(a), norm_name(b)))) + (pd.Timestamp(date).date(),)


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/method_blend.json")
    if not METHOD_ODDS.exists():
        sys.exit(f"{METHOD_ODDS} is not here. Run the fetch-method-odds mode.")

    print("building features...")
    import predict_card as engine

    from feature_inventory import finish_level_names, method_rate_names

    added = set(method_rate_names()) | set(finish_level_names())
    mask = engine.ufc_valid["target_method"].isin(CLASSES).values
    full = engine.X_valid.reset_index(drop=True)[mask]
    base = full[[c for c in full.columns if c not in added]].reset_index(drop=True)

    meta = engine.ufc_valid.reset_index(drop=True)[mask].copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
    years = meta["date"].dt.year
    order = {c: i for i, c in enumerate(CLASSES)}
    y = meta["target_method"].map(order).to_numpy()

    valid = engine.ufc["target_win"].notna().to_numpy()

    def aligned(frame):
        return (frame.reset_index(drop=True)[valid]
                .reset_index(drop=True)[mask].reset_index(drop=True))

    X = pd.concat([base, aligned(level_columns(engine)),
                   aligned(rate_columns(engine))], axis=1).to_numpy()

    print("walking the method model forward...")
    proba, idx = walk_forward(X, y, years, weighted=True)
    model_finish = 1.0 - proba[:, 0]
    predicted = meta.iloc[idx].reset_index(drop=True)

    # --- join to the market ------------------------------------------------
    props = pd.read_csv(METHOD_ODDS)
    props["date"] = pd.to_datetime(props["date"], errors="coerce")
    props = props.dropna(subset=list(PROP_COLUMNS))
    market, overround = market_finish_probability(props)
    priced = {pair_key(a, b, d): (p, o) for a, b, d, p, o
              in zip(props.fighter_a, props.fighter_b, props.date,
                     market, overround)}

    rows = []
    for position, fight in predicted.iterrows():
        hit = priced.get(pair_key(fight["r_name"], fight["b_name"], fight["date"]))
        if hit is None:
            continue
        rows.append({
            "year": int(fight["date"].year),
            "model": float(model_finish[position]),
            "market": float(hit[0]),
            "overround": float(hit[1]),
            "finished": float(y[idx][position] != 0),
        })
    d = pd.DataFrame(rows)
    print(f"\n  {len(predicted):,} walk-forward predictions, "
          f"{len(d):,} of them priced ({d.year.min()}-{d.year.max()})")
    print(f"  the six prices imply {d.overround.mean():.1%} between them "
          f"(a moneyline pair implies about 104%)")
    print(f"  really finished {d.finished.mean():.1%}; "
          f"model said {d.model.mean():.1%}, market said {d.market.mean():.1%}")

    def report(label, p, truth):
        return {
            "label": label,
            "n": int(len(truth)),
            "accuracy": float(((p > 0.5) == (truth == 1)).mean()),
            "auc": float(roc_auc_score(truth, p)),
            "brier": float(brier_score_loss(truth, np.clip(p, 0, 1))),
            "log_loss": float(log_loss(truth, np.clip(p, 1e-6, 1 - 1e-6))),
        }

    # --- the weight, chosen forward ---------------------------------------
    picked, blended, truth_out, years_out = [], [], [], []
    for year in sorted(d.year.unique()):
        past, now = d[d.year < year], d[d.year == year]
        if len(past) < 300 or len(now) < 30:
            continue
        scores = [((( (1 - w) * past.model + w * past.market) > 0.5)
                   == (past.finished == 1)).mean() for w in WEIGHTS]
        w = float(WEIGHTS[int(np.argmax(scores))])
        picked.append((int(year), w))
        blended.append(((1 - w) * now.model + w * now.market).to_numpy())
        truth_out.append(now.finished.to_numpy())
        years_out.append(np.full(len(now), year))

    if not blended:
        sys.exit("Too few priced fights to choose a weight forward.")
    bp = np.concatenate(blended)
    bt = np.concatenate(truth_out)
    by = np.concatenate(years_out)
    print("\n  weight chosen from earlier years only: "
          + ", ".join(f"{yr}:{w:.2f}" for yr, w in picked))

    scored = d[d.year.isin(np.unique(by))]
    results = [report("model alone", scored.model.to_numpy(),
                      scored.finished.to_numpy()),
               report("market alone", scored.market.to_numpy(),
                      scored.finished.to_numpy()),
               report("blended, forward", bp, bt)]

    confirm = by >= CONFIRM_FROM
    if confirm.sum() > 200:
        c = scored[scored.year >= CONFIRM_FROM]
        results += [report("model, confirm", c.model.to_numpy(),
                           c.finished.to_numpy()),
                    report("market, confirm", c.market.to_numpy(),
                           c.finished.to_numpy()),
                    report("blended, confirm", bp[confirm], bt[confirm])]

    print(f"\n  {'':<20}{'n':>7}{'accuracy':>10}{'AUC':>8}{'Brier':>9}{'log loss':>10}")
    print("  " + "-" * 64)
    for r in results:
        print(f"  {r['label']:<20}{r['n']:>7}{r['accuracy']:>10.1%}"
              f"{r['auc']:>8.3f}{r['brier']:>9.4f}{r['log_loss']:>10.4f}")

    model = next(r for r in results if r["label"].startswith("model alone"))
    blend = next(r for r in results if r["label"].startswith("blended, forward"))
    gain = blend["accuracy"] - model["accuracy"]
    # Paired bootstrap, because the arms are scored on the same fights and a
    # 4,000-fight sample wobbles by about a point on its own.
    rng = np.random.default_rng(0)
    model_hits = (scored.model.to_numpy() > 0.5) == (scored.finished.to_numpy() == 1)
    blend_hits = (bp > 0.5) == (bt == 1)
    n = min(len(model_hits), len(blend_hits))
    diffs = [float(blend_hits[i].mean() - model_hits[i].mean())
             for i in (rng.integers(0, n, n) for _ in range(4000))]
    low, high = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  blending is worth {gain:+.1%} on accuracy; "
          f"paired bootstrap 95% [{low:+.1%}, {high:+.1%}]")
    print("  " + ("The interval excludes zero." if low > 0 or high < 0
                  else "THE INTERVAL CROSSES ZERO - it is not established."))

    out = ENGINE / "experiments" / "method_blend.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "priced": int(len(d)),
        "overround": float(d.overround.mean()),
        "weights": picked,
        "results": results,
        "accuracy_gain": float(gain),
        "gain_interval": [float(low), float(high)],
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
