"""Opening lines against closing lines, and what weight each earns in the blend.

odds.csv holds CLOSING moneylines and has none for 2025. The one source
found that prices 2025 (data/opening_odds.csv, sync_kaggle
fetch-opening-odds) carries OPENING lines. This measures, on every fight
priced both ways, with the walk-forward model probabilities of
experiments/model_compare.py (each year fitted on earlier years only):

  - how good each price is as a forecast (log loss, accuracy)
  - the blend weight on the market that each earns (the live card reads
    prices days before the bell, nearer an opening line than a closing one,
    while MARKET_WEIGHT = 0.75 was fitted on closing lines)
  - whether the line moves toward the model between open and close

    python engine/experiments/opening_lines.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments import model_compare as mc

OPENING = ENGINE / "data" / "opening_odds.csv"
OUT = Path(__file__).with_suffix(".json")
WEIGHTS = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)


def _ll(y, p):
    return float(np.mean(mc.log_loss_rows(y, p)))


def main():
    base = pd.read_csv(Path(__file__).with_name("model_compare_baseline.csv"))
    base["p_close"] = mc.market_probabilities(base, mc.load_prices())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "open.csv"
        pd.read_csv(OPENING).rename(columns={"open_a": "odds_a",
                                             "open_b": "odds_b"}).to_csv(path, index=False)
        base["p_open"] = mc.market_probabilities(base, mc.load_prices(path))
    both = base[base["p_open"].notna() & base["p_close"].notna()].copy()
    y = both["y"].to_numpy()
    out = {"fights_both": int(len(both)),
           "open_priced_2025": int((base["p_open"].notna() & (base["year"] == 2025)).sum())}
    for name, col in (("model", "p_model"), ("opening", "p_open"), ("closing", "p_close")):
        out[name] = {"log_loss": _ll(y, both[col]),
                     "accuracy": float(((both[col] > .5) == (y == 1)).mean())}
    out["blend_log_loss"] = {
        col: {str(w): _ll(y, (1 - w) * both["p_model"] + w * both[col]) for w in WEIGHTS}
        for col in ("p_open", "p_close")}
    move = both["p_close"] - both["p_open"]
    moved = move.abs() > 0.01
    toward = np.sign(move) == np.sign(both["p_model"] - both["p_open"])
    out["mean_abs_move"] = float(move.abs().mean())
    out["moved_toward_model"] = float((toward & moved).sum() / moved.sum())
    for name in ("model", "opening", "closing"):
        print(f"  {name:8} log loss {out[name]['log_loss']:.4f}  "
              f"accuracy {out[name]['accuracy']:.3f}")
    for col, by in out["blend_log_loss"].items():
        print(f"  blend with {col:8}" + "".join(f"  w={w}: {v:.4f}" for w, v in by.items()))
    print(f"  {out['fights_both']} fights priced both ways; the line moves "
          f"{out['mean_abs_move']:.3f} on average and toward the model "
          f"{out['moved_toward_model']:.1%} of the times it moves")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
