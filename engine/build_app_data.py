"""Build the app's fighter and performance data, without running the model.

Split from the card export on purpose. The card needs the trained model and
takes twenty minutes; this needs only the dataset and the files CI already
commits, so it runs in seconds and can be regenerated whenever the data moves.

    python3 build_app_data.py [--out ../app/data]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

import app_export as ax
import career_stats as cs
import matchup as mu

UFC_CSV = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
HISTORY = ENGINE / "data" / "prediction_history.json"
EXPERIMENTS = ENGINE / "experiments"
DEFAULT_OUT = ENGINE.parent / "app" / "data"

# What the app needs per fighter: enough to search, compare and simulate.
# Deliberately not every column - this ships to a phone over mobile data.
EXPORTED_COLUMNS = (
    "cd_bouts", "cd_wins", "cd_losses", "cd_win_rate", "cd_minutes",
    "cd_slpm", "cd_str_acc", "cd_sapm", "cd_str_def",
    "cd_td_per15", "cd_td_acc", "cd_td_def", "cd_sub_per15", "cd_kd_per15",
    "cd_ctrl_share", "cd_opp_ctrl_share", "cd_opp_sub_per15",
    "cd_opp_kd_per15", "cd_opp_head_per15",
    "cd_ko_for_per15", "cd_ko_against_per15",
    "cd_sub_for_per15", "cd_sub_against_per15",
    "cd_sig_atmpted", "cd_opp_sig_atmpted", "cd_td_atmpted",
    "cd_opp_td_atmpted",
    "cd_head_share", "cd_body_share", "cd_leg_share",
    "cd_dist_share", "cd_clinch_share", "cd_ground_share",
)


def _read_csv(name):
    path = EXPERIMENTS / name
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def latest_names(df):
    """One display name per fighter id, taken from their most recent bout.

    Names change - a fighter drops a nickname, a scrape fixes a spelling - and
    the newest spelling is the one someone types into a phone.
    """
    names = {}
    for corner in cs.CORNERS:
        block = df[[f"{corner}_id", f"{corner}_name"]].dropna()
        for fighter_id, name in block.itertuples(index=False):
            names[fighter_id] = name
    return names


def physical(df):
    """Height, reach and weight per fighter, from their latest bout.

    These are static attributes rather than accumulated ones, so they do not
    belong in career_stats - it emits what a career adds up to, and a body
    does not add up.
    """
    out = {}
    for corner in cs.CORNERS:
        columns = [f"{corner}_id", f"{corner}_height", f"{corner}_reach",
                   f"{corner}_weight"]
        if not all(c in df.columns for c in columns):
            continue
        for row in df[columns].itertuples(index=False):
            fighter_id = row[0]
            if fighter_id != fighter_id:
                continue
            out[fighter_id] = {"height_cm": row[1], "reach_cm": row[2],
                               "weight_kg": row[3]}
    return out


def build(out_dir=DEFAULT_OUT):
    df = pd.read_csv(UFC_CSV, low_memory=False).sort_values("date")
    df = df.reset_index(drop=True)

    final = cs.final_stats(df)
    columns = [c for c in EXPORTED_COLUMNS if c in final.columns]
    fighters = ax.fighters_payload(final, columns)

    # Height, reach and weight ride along as three more columns, so the app
    # has one shape to read rather than two.
    body = physical(df)
    extra = ("height_cm", "reach_cm", "weight_kg")
    fighters["columns"] = list(fighters["columns"]) + list(extra)
    for row, fighter_id in zip(fighters["values"], fighters["ids"]):
        measurements = body.get(fighter_id, {})
        row.extend(ax._clean(measurements.get(key)) for key in extra)

    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else {}
    performance = ax.performance_payload(
        history,
        auc_by_year=_read_csv("auc_by_year.csv"),
        backtest=_read_csv("historical_backtest.csv"),
        strategies=STRATEGIES)
    performance["dataset_end"] = str(df["date"].max())[:10]
    performance["bouts"] = int(len(df))

    written = ax.write_all(out_dir, {
        "fighters": fighters,
        "names": ax.names_payload(latest_names(df)),
        "performance": performance,
        "constants": ax.constants_payload(mu),
    })
    for path in written:
        print(f"  wrote {path} ({path.stat().st_size/1024:.0f} KB)")
    return written


# From the walk-forward backtest on honest point-in-time statistics, confirm
# period only (2020 onward). The tune-period figures are deliberately absent:
# they were used to choose how the strategies work and are not a forecast.
STRATEGIES = [
    {"name": "MODEL", "description": "Bet every confident pick",
     "bets": 2339, "hit_rate": 0.616, "roi": -0.018,
     "interval": [-0.053, 0.017]},
    {"name": "FADE", "description": "Bet against the picks the second model "
                                    "flags as likely wrong",
     "bets": 565, "hit_rate": 0.632, "roi": -0.017,
     "interval": [-0.083, 0.047]},
    {"name": "COMBINED", "description": "Model picks, minus the flagged ones",
     "bets": 2339, "hit_rate": 0.680, "roi": 0.003,
     "interval": [-0.026, 0.032]},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
