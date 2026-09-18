"""Detect features that carry information from after the fight they describe.

The leakage audits in predict_card check that rolling windows use shift(1),
which is the right check for a rolling window and catches nothing here. The
profile columns are not rolling: str_acc, splm, sapm, str_def, td_avg, td_def
and sub_avg come from the upstream fighter table, which publishes a fighter's
career averages as of the data pull, and those are joined onto every bout that
fighter ever had.

So a 2014 fight is described by a striking accuracy computed over 2014 to 2026,
including the fight being predicted and every fight after it.

The signature is simple and unmistakable once looked for: a genuine
point-in-time statistic changes as a fighter's career progresses, and a career
total does not. 80 to 90 per cent of fighters with five or more bouts carry a
single value for their whole career in these columns.

This matters beyond tidiness. The walk-forward backtest returned 68.6% and
+16.2% over 5,943 priced bets, rising to 75% by 2024, which is not a plausible
return against a liquid market. The one year that behaves normally is 2026, at
61.5% and -3.2% - and 2026 is the year with almost no career left to leak,
because the data ends in August of it.
"""

import os
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
UFC_CSV = Path(os.environ.get(
    "UFC_CSV", ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"))

MIN_FIGHTS = 5
CONSTANT_SHARE = 0.5


def constant_across_career(df, column, corner="r", min_fights=MIN_FIGHTS):
    """Share of established fighters whose value never changes.

    A point-in-time statistic moves as a career progresses. One that never
    moves was computed over the whole career and is visible to every fight in
    it, including the ones it should not be.
    """
    name_col = f"{corner}_name"
    if column not in df.columns or name_col not in df.columns:
        return None
    present = df[df[column].notna()]
    counts = present[name_col].value_counts()
    established = counts[counts >= min_fights].index
    if len(established) == 0:
        return None
    distinct = present[present[name_col].isin(established)] \
        .groupby(name_col)[column].nunique()
    return {
        "column": column,
        "fighters": int(len(distinct)),
        "constant_share": float((distinct == 1).mean()),
        "median_distinct": float(distinct.median()),
    }


def scan(df, columns, corner="r"):
    """Every column that looks like a career total rather than a snapshot."""
    findings = []
    for column in columns:
        row = constant_across_career(df, column, corner)
        if row and row["constant_share"] >= CONSTANT_SHARE:
            findings.append(row)
    return sorted(findings, key=lambda r: -r["constant_share"])


PROFILE_COLUMNS = ["r_splm", "r_str_acc", "r_sapm", "r_str_def",
                   "r_td_avg", "r_td_def", "r_td_avg_acc", "r_sub_avg"]


def report(findings):
    print("=" * 74)
    print(f"FEATURES CARRYING THE WHOLE CAREER: {len(findings)}")
    print("  a value that never changes was computed over every fight,")
    print("  including the one being predicted and the ones after it")
    print("=" * 74)
    for row in findings:
        print(f"  {row['column']:<16} constant for {row['constant_share']:>6.1%} "
              f"of {row['fighters']:,} fighters with {MIN_FIGHTS}+ bouts "
              f"(median {row['median_distinct']:.0f} distinct values)")
    if not findings:
        print("  none")


def main():
    df = pd.read_csv(UFC_CSV, low_memory=False)
    report(scan(df, PROFILE_COLUMNS))


if __name__ == "__main__":
    main()
