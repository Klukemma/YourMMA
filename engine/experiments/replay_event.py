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
    python engine/experiments/replay_event.py --card ufc332.json

A --card file predicts an event the archive does not hold yet: a JSON
object {"event", "date", "fights": [{"red", "blue", "division", "is_5rnd",
"is_title", optional "odds_red"/"odds_blue" (American, as quoted before the
event), optional "winner"/"method" once known}]}. Where both odds are
given, the market-blended probability (market_blend.blend, as production
blends) is reported beside the model's own.

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
            "method_correct": int(sum(method_hits)),
            **priced_grade(scored)}


def priced_grade(scored):
    """Model, market and blend on the fights with both prices."""
    priced = [r for r in scored if r.get("p_blend") is not None]
    if not priced:
        return {}
    y = np.array([r["red_won"] for r in priced], float)
    out = {"priced": len(priced)}
    for key, col in (("model", "p_red"), ("blend", "p_blend"), ("market", "p_market")):
        p = np.clip(np.array([r[col] for r in priced]), 1e-6, 1 - 1e-6)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        hits = int(sum((q > 0.5) == bool(t) for q, t in zip(p, y)))
        if key == "model":
            out.update(model_correct_priced=hits, model_log_loss_priced=ll)
        else:
            out.update({f"{key}_correct": hits, f"{key}_log_loss": ll})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("event", nargs="?", help='e.g. "UFC 322"')
    parser.add_argument("--date")
    parser.add_argument("--card", help="a typed card for an event the "
                                       "archive does not hold yet")
    parser.add_argument("--show", help="re-print a replay_<date>.json")
    parser.add_argument("--child")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child, args.out) or 0
    if args.show:
        saved = json.loads(Path(args.show).read_text())
        print(f"  {saved['event']} ({saved['date']})")
        show(saved["fights"])
        return 0
    if not (args.event or args.date or args.card):
        parser.error("name an event, a --date or a --card")

    archive = pd.read_csv(ARCHIVE, low_memory=False)
    days = pd.to_datetime(archive["date"], errors="coerce")
    if args.card:
        name, date, card = typed_card(json.loads(Path(args.card).read_text()))
        if (days >= pd.Timestamp(date)).any():
            print(f"  note: the archive holds bouts on or after {date}; "
                  f"they are cut, as for any replay")
        return run(archive, days, name, date, card)
    date, rows = card_of(archive, args.event, args.date)
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
    return run(archive, days, name, date, card)


def typed_card(spec):
    """(name, date, card rows) from a --card file."""
    date = str(pd.Timestamp(spec["date"]).date())
    card = []
    for f in spec["fights"]:
        winner = f.get("winner")
        card.append({"date": date, "red": f["red"], "blue": f["blue"],
                     "division": f["division"],
                     "is_5rnd": int(bool(f.get("is_5rnd"))),
                     "is_title": int(bool(f.get("is_title"))),
                     "odds_red": f.get("odds_red"), "odds_blue": f.get("odds_blue"),
                     "red_won": (1 if winner == f["red"] else 0 if winner == f["blue"]
                                 else None),
                     "actual_winner": winner,
                     "actual_method": method_class(f.get("method")) if winner else None,
                     "actual_detail": f.get("method")})
    return spec["event"], date, card


def with_market(preds):
    """The 0.75 market blend production applies, where both odds are known."""
    import market_blend
    for r in preds:
        if "p_red" in r and r.get("odds_red") is not None and r.get("odds_blue") is not None:
            market = market_blend.devig(r["odds_red"], r["odds_blue"])
            r["p_market"] = float(market)
            r["p_blend"] = float(market_blend.blend(r["p_red"], market))
    return preds


def show(preds):
    """Print the card: the model's own pick and, where priced, the blended
    pick production shows; grade both against any known results."""
    print(f"\n  {'fight':<44} {'model pick':<22} {'model':>5}  "
          f"{'blend pick':<22} {'blend':>5} {'mkt':>5}  {'method':<10} result")
    for r in preds:
        fight = f"{r['red']} vs {r['blue']}"
        if "p_red" not in r:
            print(f"  {fight:<44} REFUSED: {r['error'][:70]}")
            continue
        def pick(q):
            return (r["red"], q) if q > 0.5 else (r["blue"], 1 - q)
        m_pick, m_p = pick(r["p_red"])
        blend = r.get("p_blend")
        b_pick, b_p = pick(blend) if blend is not None else ("(no price)", None)
        mkt = r.get("p_market")
        mkt_p = None if mkt is None else (mkt if b_pick == r["red"] else 1 - mkt)
        if r["red_won"] is None:
            tail = "not graded"
        else:
            mark = lambda name: "RIGHT" if name == r["actual_winner"] else "wrong"
            tail = (f"model {mark(m_pick)}"
                    + (f", blend {mark(b_pick)}" if blend is not None else "")
                    + f": {r['actual_winner']} by {r['actual_detail']}")
        fmt = lambda q: "" if q is None else f"{q:.0%}"
        print(f"  {fight:<44} {m_pick:<22} {fmt(m_p):>5}  {b_pick:<22} "
              f"{fmt(b_p):>5} {fmt(mkt_p):>5}  {r['method']:<10} {tail}")
    summary = grade(preds)
    if summary["scored"]:
        print(f"\n  model: winners {summary['correct']}/{summary['scored']}, "
              f"methods {summary['method_correct']}/{summary['scored']}, "
              f"log loss {summary['log_loss']:.3f}, Brier {summary['brier']:.3f}")
        if summary.get("priced"):
            print(f"  on the {summary['priced']} priced fights: model "
                  f"{summary['model_correct_priced']}/{summary['priced']} (log loss "
                  f"{summary['model_log_loss_priced']:.3f}), blend "
                  f"{summary['blend_correct']}/{summary['priced']} "
                  f"({summary['blend_log_loss']:.3f}), market "
                  f"{summary['market_correct']}/{summary['priced']} "
                  f"({summary['market_log_loss']:.3f})")
    return summary


def run(archive, days, name, date, card):
    print(f"  {name} ({date}): {len(card)} bouts; engine loaded on the "
          f"{int((days < pd.Timestamp(date)).sum()):,} bouts before it")
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        (tmp / "app").mkdir()
        cut = tmp / "cut.csv"
        archive[days < pd.Timestamp(date)].to_csv(cut, index=False)
        card_path, out = tmp / "card.json", tmp / "out.json"
        card_path.write_text(json.dumps(card))
        # Loading the engine flushes cached lines of finished fights into
        # data/odds.csv (the scheduled run does that and commits it); a
        # replay changes nothing in the repository, so the file is put back.
        odds_csv = ENGINE / "data" / "odds.csv"
        before = odds_csv.read_bytes() if odds_csv.exists() else None
        try:
            result = subprocess.run([sys.executable, __file__, "--child",
                                     str(card_path), "--out", str(out)],
                                    env=_quiet_env(tmp, cut), cwd=ENGINE.parent,
                                    capture_output=True, text=True)
        finally:
            if before is not None:
                odds_csv.write_bytes(before)
        if result.returncode:
            print(result.stdout[-2000:], result.stderr[-3000:])
            raise SystemExit("the engine load failed")
        preds = with_market(json.loads(out.read_text()))

    summary = show(preds)
    path = HERE / f"replay_{date}.json"
    path.write_text(json.dumps({"event": name, "date": date, "summary": summary,
                                "fights": preds}, indent=1))
    print(f"  wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
