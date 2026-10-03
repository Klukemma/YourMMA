"""Predict a past UFC event as the engine would have the night before.

The archive is cut to the bouts strictly BEFORE the event's date, the
engine is loaded on that cut (so every model, calibrator and feature knows
only what was known the night before), and each bout on the card goes
through predict_fight_prod - the same call the phone's card uses, with the
shipped corner-symmetric recipe. No odds are fetched: the probabilities are
the model's own. The real results are then read from the full archive and
the card is graded.

    python engine/experiments/replay_event.py "UFC 322"
    python engine/experiments/replay_event.py --date 2025-11-15

Writes replay_<date>.json beside this file. One engine load, ~10-20 min.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_parity import ARCHIVE, _quiet_env  # noqa: E402

HERE = Path(__file__).resolve().parent


def card_of(archive, event=None, date=None):
    """(date, rows of the event) chosen by name or by date."""
    days = pd.to_datetime(archive["date"], errors="coerce")
    if event:
        names = archive["event_name"].astype(str)
        hit = names.str.lower().str.startswith(event.lower() + ":") | (
            names.str.lower() == event.lower())
        if not hit.any():
            hit = names.str.contains(event, case=False, regex=False)
        ids = archive.loc[hit, "event_id"].unique()
        if len(ids) != 1:
            raise SystemExit(f"'{event}' matches {len(ids)} events: "
                             f"{sorted(archive.loc[hit, 'event_name'].unique())[:8]}")
        rows = archive[archive["event_id"] == ids[0]]
    else:
        rows = archive[days.dt.strftime("%Y-%m-%d") == date]
        if rows.empty:
            raise SystemExit(f"no bouts on {date}")
    day = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d").unique()
    if len(day) != 1:
        raise SystemExit(f"the event spans dates {list(day)}")
    return day[0], rows


def child(card_path, out):
    """Runs inside the engine loaded on the cut archive."""
    import predict_card as engine
    rows = []
    for fight in json.loads(Path(card_path).read_text()):
        try:
            pred = engine.predict_fight_prod(
                fight["red"], fight["blue"], event_date=fight["date"],
                is_5rnd=fight["is_5rnd"], is_title=fight["is_title"],
                context={"division": fight["division"]})
        except Exception as err:            # a crash, not a refusal
            rows.append({**fight, "error": f"crash: {err}"[:300]})
            continue
        if pred.get("status") == "NO_DATA":
            rows.append({**fight, "error": str(pred.get("reason"))[:300]})
            continue
        rows.append({**fight,
                     "p_red": float(pred["red_win_prob_model"]),
                     "pick": pred["winner"],
                     "method": pred["method"],
                     "method_probs": {k: float(v) for k, v in pred["method_probs"].items()},
                     "p_finish": float(pred["p_finish"]),
                     "round": pred["round"],
                     "confidence": float(pred["confidence"])})
    Path(out).write_text(json.dumps(rows))


def method_class(method):
    m = str(method).upper()
    if m.startswith(("KO", "TKO")) or "KO/TKO" in m:
        return "KO/TKO"
    if m.startswith("SUB"):
        return "Submission"
    if "DEC" in m:
        return "Decision"
    return str(method)


def grade(rows):
    scored = [r for r in rows if "p_red" in r and r["red_won"] is not None]
    if not scored:
        return {"scored": 0}
    y = np.array([r["red_won"] for r in scored], float)
    p = np.clip(np.array([r["p_red"] for r in scored]), 1e-6, 1 - 1e-6)
    correct = [(r["p_red"] > 0.5) == bool(r["red_won"]) for r in scored]
    method_hits = [r["method"] == r["actual_method"] for r in scored]
    return {"scored": len(scored), "correct": int(sum(correct)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)),
            "method_correct": int(sum(method_hits))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("event", nargs="?", help='e.g. "UFC 322"')
    parser.add_argument("--date")
    parser.add_argument("--child")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child, args.out) or 0
    if not (args.event or args.date):
        parser.error("name an event or a --date")

    archive = pd.read_csv(ARCHIVE, low_memory=False)
    date, rows = card_of(archive, args.event, args.date)
    days = pd.to_datetime(archive["date"], errors="coerce")
    name = str(rows["event_name"].iloc[0])
    card = []
    order = rows.assign(_t=-rows["title_fight"].fillna(0).astype(int),
                        _5=-rows["total_rounds"].fillna(3))
    for _, r in order.sort_values(["_t", "_5"], kind="stable").iterrows():
        winner = r["winner"]
        red_won = (1 if winner == r["r_name"] else 0 if winner == r["b_name"]
                   else None)
        card.append({"date": date, "red": r["r_name"], "blue": r["b_name"],
                     "division": str(r.get("division")),
                     "is_5rnd": int(r["total_rounds"] == 5),
                     "is_title": int(bool(r.get("title_fight"))),
                     "red_won": red_won,
                     "actual_winner": None if pd.isna(winner) else winner,
                     "actual_method": method_class(r["method"]),
                     "actual_detail": str(r["method"])})
    print(f"  {name} ({date}): {len(card)} bouts; engine loaded on the "
          f"{int((days < pd.Timestamp(date)).sum()):,} bouts before it")
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        (tmp / "app").mkdir()
        cut = tmp / "cut.csv"
        archive[days < pd.Timestamp(date)].to_csv(cut, index=False)
        card_path, out = tmp / "card.json", tmp / "out.json"
        card_path.write_text(json.dumps(card))
        result = subprocess.run([sys.executable, __file__, "--child",
                                 str(card_path), "--out", str(out)],
                                env=_quiet_env(tmp, cut), cwd=ENGINE.parent,
                                capture_output=True, text=True)
        if result.returncode:
            print(result.stdout[-2000:], result.stderr[-3000:])
            raise SystemExit("the engine load failed")
        preds = json.loads(out.read_text())

    print(f"\n  {'fight':<46} {'pick':<22} {'prob':>5}  {'method':<11} result")
    for r in preds:
        fight = f"{r['red']} vs {r['blue']}"
        if "p_red" not in r:
            print(f"  {fight:<46} REFUSED: {r['error'][:60]}")
            continue
        p = max(r["p_red"], 1 - r["p_red"])
        right = ("no contest" if r["red_won"] is None else
                 "RIGHT" if (r["p_red"] > 0.5) == bool(r["red_won"]) else "wrong")
        print(f"  {fight:<46} {r['pick']:<22} {p:>5.0%}  {r['method']:<11} "
              f"{right}: {r['actual_winner']} by {r['actual_detail']}")
    summary = grade(preds)
    if summary["scored"]:
        print(f"\n  winners {summary['correct']}/{summary['scored']}, "
              f"methods {summary['method_correct']}/{summary['scored']}, "
              f"log loss {summary['log_loss']:.3f}, Brier {summary['brier']:.3f}")
    path = HERE / f"replay_{date}.json"
    path.write_text(json.dumps({"event": name, "date": date, "summary": summary,
                                "fights": preds}, indent=1))
    print(f"  wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
