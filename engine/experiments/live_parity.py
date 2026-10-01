"""Does the live card see the same features the model was trained on?

Every accuracy figure this project quotes is measured on training-style
rows: features built point-in-time by the training pipeline. The phone's
card is built by a different path - a per-fighter snapshot read from each
fighter's LAST archived bout. An audit found that snapshot copying the last
bout's PRE-fight values (so the most recent fight is missing from striking,
takedown, finish and opponent-quality numbers), several features forced to
zero live, and interaction terms that are always zero on a one-row frame.

This measures what that costs, directly. For each of the last N events:

  train  the full archive is loaded once; each fight's feature row is taken
         exactly as training built it
  live   the archive is cut to the bouts BEFORE the event, the engine is
         loaded on that (so it knows what it would have known the night
         before), and each fight is predicted through the live path; the
         feature row it hands the winner model is captured

Both rows go through the SAME fitted models (the cut archive's). The gap
between the two probabilities is what the live path changes; the per-
feature table says where. Nothing here changes a prediction.

    python engine/experiments/live_parity.py [--events 4]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
OUT = Path(__file__).with_suffix(".json")


def _quiet_env(tmp, csv=None):
    env = dict(os.environ)
    env["APP_DATA_DIR"] = str(tmp / "app")
    env["PREDICTIONS_LOG"] = str(tmp / "predictions.json")
    env.pop("ODDS_API_KEY", None)
    if csv:
        env["UFC_CSV"] = str(csv)
    return env


def train_child(dates, out):
    """Training-style feature rows for the fights on `dates`."""
    import predict_card as engine
    meta = engine.ufc_valid.reset_index(drop=True)
    X = engine.X_valid_winner.reset_index(drop=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    day = pd.to_datetime(meta["date"]).dt.strftime("%Y-%m-%d")
    rows = []
    for i in np.flatnonzero(day.isin(dates).to_numpy()):
        # A bout the sync filed under a placeholder id was built as a
        # debut under 'new_<hex>'; the live path builds the fighter under
        # the real id, so that training row is not reproducible by design
        # (pending_rows._canon_ids) and the report leaves it out.
        placeholder = any(str(meta.loc[i, c]).startswith(("new_", "unk_"))
                          for c in ("r_id", "b_id"))
        rows.append({"date": day[i], "red": meta.loc[i, "r_name"],
                     "blue": meta.loc[i, "b_name"],
                     "won": float(engine.y_win[i]),
                     "is_5rnd": int(meta.loc[i, "total_rounds"] == 5),
                     "is_title": int(bool(meta.loc[i, "title_fight"])),
                     "division": str(meta.loc[i, "division"]),
                     "placeholder": bool(placeholder),
                     "features": {c: float(X.loc[i, c]) for c in X.columns}})
    Path(out).write_text(json.dumps(rows))


def live_child(date, train_path, out):
    """Live feature rows (and both probabilities) for one event."""
    import predict_card as engine
    seen = []

    class Recorder:
        def __init__(self, inner):
            self.inner = inner

        def transform(self, frame):
            seen.append(frame.copy())
            return self.inner.transform(frame)

    scaler = engine.scaler_winner
    engine.scaler_winner = Recorder(scaler)

    def prob(frame):
        xs = scaler.transform(frame[engine.feature_cols_winner])
        ens = (engine.lr_prod.predict_proba(xs)[:, 1]
               + engine.rf_prod.predict_proba(xs)[:, 1]
               + engine.xgb_prod.predict_proba(xs)[:, 1]) / 3
        return engine.platt_prod.predict_proba(ens.reshape(-1, 1))[:, 1]

    rows = []
    for fight in json.loads(Path(train_path).read_text()):
        if fight["date"] != date:
            continue
        seen.clear()
        try:
            pred = engine.predict_fight_prod(fight["red"], fight["blue"],
                                             event_date=date,
                                             is_5rnd=fight["is_5rnd"],
                                             is_title=fight["is_title"],
                                             context={"division": fight.get("division")})
        except Exception as err:            # a crash, not a refusal
            rows.append({**fight, "live": None, "error": f"crash: {err}"[:200]})
            continue
        if pred.get("status") == "NO_DATA":  # refused with its reason: a
            rows.append({**fight, "live": None,  # debut, a shared name...
                         "error": str(pred.get("reason"))[:200]})
            continue
        if not seen:
            rows.append({**fight, "live": None, "error": "no prediction"})
            continue
        live = seen[-1].iloc[0].to_dict()
        trained = pd.DataFrame([fight["features"]])
        rows.append({**fight, "live": {k: float(v) for k, v in live.items()},
                     "p_live": float(prob(pd.DataFrame([live]))[0]),
                     "p_train": float(prob(trained)[0])})
    Path(out).write_text(json.dumps(rows))


def _run(args, env):
    result = subprocess.run([sys.executable, __file__, *args], env=env,
                            cwd=ENGINE.parent, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-2000:], result.stderr[-3000:])
        raise SystemExit(f"child {args[:2]} failed")


def _logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def report(rows):
    placeholder = [r for r in rows if r.get("placeholder")]
    rows = [r for r in rows if not r.get("placeholder")]
    scored = [r for r in rows if r.get("live")]
    refused = [r for r in rows if not r.get("live")]
    print(f"\n  {len(rows)} fights, {len(scored)} predicted live"
          + (f", {len(placeholder)} filed under a placeholder id left out"
             if placeholder else ""))
    for r in refused:
        print(f"    refused: {r['red']} vs {r['blue']} ({r['date']}): "
              f"{r.get('error')}")
    if not scored:
        return {}
    gap = np.array([r["p_live"] - r["p_train"] for r in scored])
    y = [r["won"] for r in scored]
    summary = {
        "fights": len(rows), "scored": len(scored),
        "mean_abs_gap": float(np.mean(np.abs(gap))),
        "share_gap_over_5pts": float(np.mean(np.abs(gap) > 0.05)),
        "share_pick_flipped": float(np.mean(
            [(r["p_live"] > .5) != (r["p_train"] > .5) for r in scored])),
        "logloss_live": _logloss(y, [r["p_live"] for r in scored]),
        "logloss_train_features": _logloss(y, [r["p_train"] for r in scored]),
        "accuracy_live": float(np.mean([(r["p_live"] > .5) == (r["won"] == 1)
                                        for r in scored])),
        "accuracy_train_features": float(np.mean(
            [(r["p_train"] > .5) == (r["won"] == 1) for r in scored]))}
    print(f"  live minus training-style probability: mean |gap| "
          f"{summary['mean_abs_gap']:.3f}; {summary['share_gap_over_5pts']:.0%} "
          f"of fights move 5+ points; {summary['share_pick_flipped']:.0%} "
          f"change the pick")
    print(f"  on these fights: log loss live {summary['logloss_live']:.4f} vs "
          f"training-style {summary['logloss_train_features']:.4f}; accuracy "
          f"{summary['accuracy_live']:.3f} vs "
          f"{summary['accuracy_train_features']:.3f} (few fights - the gap "
          f"above is the measurement, these are colour)")
    features = sorted(scored[0]["features"])
    table = []
    for name in features:
        a = np.array([r["live"].get(name, np.nan) for r in scored])
        b = np.array([r["features"][name] for r in scored])
        differs = ~np.isclose(a, b, atol=1e-6, equal_nan=True)
        sd = np.nanstd(b) or 1.0
        table.append({"feature": name, "share_differs": float(differs.mean()),
                      "live_zero_train_not": float(np.mean((a == 0) & (b != 0))),
                      "mean_abs_diff_sd": float(np.nanmean(np.abs(a - b)) / sd)})
    table.sort(key=lambda t: -t["share_differs"] * (t["mean_abs_diff_sd"] + 1e-9))
    print(f"\n  {'feature':<32}{'differs':>9}{'0 live':>8}{'|diff| in sd':>14}")
    for t in table:
        if t["share_differs"] > 0:
            print(f"  {t['feature']:<32}{t['share_differs']:>9.0%}"
                  f"{t['live_zero_train_not']:>8.0%}{t['mean_abs_diff_sd']:>14.2f}")
    same = [t["feature"] for t in table if t["share_differs"] == 0]
    print(f"\n  {len(same)} of {len(table)} features match on every fight")
    return {"summary": summary, "features": table}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--child", choices=["train", "live"])
    parser.add_argument("--dates")
    parser.add_argument("--train")
    parser.add_argument("--out")
    parser.add_argument("--work", help="keep the replays here and reuse "
                                       "those already done (each is a full "
                                       "engine load, ~10 minutes)")
    args = parser.parse_args(argv)
    if args.child == "train":
        return train_child(args.dates.split(","), args.out) or 0
    if args.child == "live":
        return live_child(args.dates, args.train, args.out) or 0

    archive = pd.read_csv(ARCHIVE, low_memory=False)
    days = pd.to_datetime(archive["date"], errors="coerce")
    dates = sorted(days.dropna().dt.strftime("%Y-%m-%d").unique())[-args.events:]
    print(f"  events: {', '.join(dates)}")
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(args.work) if args.work else Path(scratch)
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "app").mkdir(exist_ok=True)
        train_path = tmp / f"train_{dates[0]}_{dates[-1]}.json"
        if not train_path.exists():
            print("  loading the full archive (training-style rows)...",
                  flush=True)
            _run(["--child", "train", "--dates", ",".join(dates), "--out",
                  str(train_path)], _quiet_env(tmp))
        rows = []
        for date in dates:
            out = tmp / f"live_{date}.json"
            if not out.exists():
                cut = tmp / f"cut_{date}.csv"
                archive[days < pd.Timestamp(date)].to_csv(cut, index=False)
                print(f"  {date}: loading the archive as of the night "
                      f"before...", flush=True)
                _run(["--child", "live", "--dates", date, "--train",
                      str(train_path), "--out", str(out)],
                     _quiet_env(tmp, cut))
                cut.unlink()
            rows += json.loads(out.read_text())
    result = report(rows)
    OUT.write_text(json.dumps({"events": dates, **result,
                               "fights": [{k: v for k, v in r.items()
                                           if k not in ("features", "live")}
                                          for r in rows]}, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
