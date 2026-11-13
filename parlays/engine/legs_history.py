"""The three sports' historical legs, one row per event, for the backtest.

Each row is that sport's SHIPPED pick, point in time, at the price on file:

  UFC     engine/experiments/parlay_legs.csv.gz - the walk-forward model
          (the shipped recipe) and each card's single price snapshot; the
          pick is the blended side (market_blend at the shipped weight).
  NFL     nfl_model.walk_forward over nfl_model.feature_frame - each season
          fitted on earlier seasons; the pick is the blend's (home at 0.5 or
          more), priced at the pick side's closing moneyline (nflverse).
  Soccer  soccer_model.pick over the top five leagues - the market
          recalibrated with weights fitted on earlier seasons only - priced
          at the pre-match average for the pick (never the closing price).
          Built from main's soccer engine (`--soccer-engine`), because this
          branch carries unfinished soccer work that main does not ship.

Writes parlays/data/legs_{ufc,nfl,soccer}.csv.gz with: day (ET), key, event,
pick, opponent, p, p_model, p_market, decimal, price_source, result
(won / lost / void). Runs offline in under a minute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "parlays" / "data"
COLS = ["day", "key", "event", "pick", "opponent", "p", "p_model", "p_market",
        "decimal", "price_source", "result"]


def _devig(a, b):
    ia, ib = 1.0 / a, 1.0 / b
    return ia / (ia + ib)


def _dec(american):
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def ufc():
    sys.path.insert(0, str(ROOT / "engine"))
    import market_blend
    d = pd.read_csv(ROOT / "engine" / "experiments" / "parlay_legs.csv.gz")
    d = d[d["dec_red"].notna() & d["dec_blue"].notna() & d["winner"].ne("unresolved")]
    rows = []
    for r in d.itertuples():
        over = 1.0 / r.dec_red + 1.0 / r.dec_blue - 1.0
        if over < 0 or over > 0.15:            # the same believable-book check parlay_profiles uses
            continue
        mk_red = _devig(r.dec_red, r.dec_blue)
        pb_red = market_blend.blend(r.p_model, mk_red)
        red = pb_red >= 0.5
        pick, opp = (r.red, r.blue) if red else (r.blue, r.red)
        winner = None if pd.isna(r.winner) else r.winner
        result = "void" if winner is None else ("won" if winner == ("red" if red else "blue") else "lost")
        rows.append({"day": r.date, "key": r.fight_id, "event": r.event, "pick": pick, "opponent": opp,
                     "p": pb_red if red else 1 - pb_red,
                     "p_model": r.p_model if red else 1 - r.p_model,
                     "p_market": mk_red if red else 1 - mk_red,
                     "decimal": r.dec_red if red else r.dec_blue,
                     "price_source": r.price_source, "result": result})
    return pd.DataFrame(rows, columns=COLS)


def nfl():
    sys.path.insert(0, str(ROOT / "nfl" / "engine"))
    import nfl_data as nd
    import nfl_model as nm
    frame = nm.feature_frame()
    wf = nm.walk_forward(frame)
    f = frame.join(wf)
    f = f[(f["season"] >= 2016) & f["played"].astype(bool) & f["home_ml"].notna() & f["away_ml"].notna()
          & f["blend_prob"].notna()]
    rows = []
    for r in f.itertuples():
        dh, da = _dec(r.home_ml), _dec(r.away_ml)
        mk_home = _devig(dh, da)
        home = r.blend_prob >= 0.5
        res = r.result
        result = "void" if res == 0 else ("won" if (res > 0) == home else "lost")
        rows.append({"day": nd.eastern_day(r.kickoff).strftime("%Y-%m-%d") if hasattr(nd.eastern_day(r.kickoff), "strftime") else str(nd.eastern_day(r.kickoff))[:10],
                     "key": r.game_id, "event": f"{r.away} @ {r.home}",
                     "pick": r.home if home else r.away, "opponent": r.away if home else r.home,
                     "p": r.blend_prob if home else 1 - r.blend_prob,
                     "p_model": r.model_prob if home else 1 - r.model_prob,
                     "p_market": mk_home if home else 1 - mk_home,
                     "decimal": dh if home else da, "price_source": "close", "result": result})
    return pd.DataFrame(rows, columns=COLS)


def soccer(engine_dir):
    """Main's soccer engine at `engine_dir` (its own soccer_data reads the
    data directory beside it)."""
    sys.path.insert(0, str(engine_dir))
    import soccer_backtest as bt
    import soccer_data as sd
    import soccer_model as smod
    m = sd.load_matches()
    m = m[m["league"].isin(sd.TOP)].reset_index(drop=True)
    nan = pd.DataFrame(np.nan, index=m.index, columns=["p_h", "p_d", "p_a"])
    p, _ = smod.pick(m, nan)
    q = bt.no_vig(m, "pre")
    rows = []
    for i, r in m.iterrows():
        if bool(p.at[i, "model_only"]) or pd.isna(r["result"]) or r["season"] < 2016:
            continue
        probs = {"H": p.at[i, "p_h"], "D": p.at[i, "p_d"], "A": p.at[i, "p_a"]}
        if any(pd.isna(v) for v in probs.values()):
            continue
        k = max(probs, key=probs.get)
        dec = r[f"avg_{k.lower()}"]
        if pd.isna(dec):
            continue
        rows.append({"day": str(r["date"])[:10], "key": r["match_id"], "event": f"{r['home']} v {r['away']}",
                     "pick": {"H": r["home"], "D": "Draw", "A": r["away"]}[k],
                     "opponent": {"H": r["away"], "D": f"{r['home']} v {r['away']}", "A": r["home"]}[k],
                     "p": probs[k], "p_model": np.nan,
                     "p_market": q.at[i, f"p_{k.lower()}"] if f"p_{k.lower()}" in q else np.nan,
                     "decimal": float(dec), "price_source": "pre",
                     "result": "won" if r["result"] == k else "lost"})
    return pd.DataFrame(rows, columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soccer-engine", required=True,
                    help="main's soccer/engine (e.g. from git archive origin/main soccer)")
    ap.add_argument("--only", choices=("ufc", "nfl", "soccer"),
                    help="build one sport in this process (each sport runs in its own process: "
                         "the UFC and soccer engines both have a package named experiments)")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.only is None:
        import subprocess
        for name in ("ufc", "nfl", "soccer"):
            subprocess.run([sys.executable, __file__, "--soccer-engine", args.soccer_engine,
                            "--only", name], check=True)
        return
    build = {"ufc": ufc, "nfl": nfl, "soccer": lambda: soccer(Path(args.soccer_engine))}[args.only]
    df = build().sort_values(["day", "key"]).reset_index(drop=True)
    df.to_csv(DATA / f"legs_{args.only}.csv.gz", index=False, float_format="%.6f")
    print(f"{args.only:7s} {len(df):6,d} legs  {df['day'].min()} .. {df['day'].max()}  "
          f"won {np.mean(df['result'] == 'won'):.3f}  mean p {df['p'].mean():.3f}")


if __name__ == "__main__":
    main()
