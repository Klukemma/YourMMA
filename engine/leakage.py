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

WHAT THE FIRST VERSION OF THIS MODULE MISSED, AND WHY.

Fixing the eight profile columns moved the walk-forward AUC by nothing at all:
2015-2025 stayed at ~0.77 and 2026 FELL to 0.604. A fix that changes nothing is
evidence the diagnosis was incomplete, so the scan was repeated over every
numeric column instead of the eight this module had been told to look at.

r_wins is constant across 98.1% of careers and r_losses across 97.8% - a
stronger signature than any of the eight. They are the fighter's LIFETIME
professional record as of the data pull, which the pipeline turned into
winrate_diff and exp_diff. On its own that one feature scores:

    year   2015   2018   2021   2024   2025   2026
    AUC    0.639  0.672  0.725  0.803  0.778  0.583

Rising for a decade and then collapsing in the only year with no future left to
leak. The point-in-time record built from prior bouts alone scores 0.53 to 0.63
in EVERY year, 2026 included, which is what a real effect looks like.

THE PRE-UFC RECORD CANNOT BE RECOVERED SAFELY, and the attempt is recorded here
so it is not retried. career_total minus every UFC result in the dataset should
leave the record a fighter brought with them, which IS known before their debut
and would be legitimate. It reconstructs to a median 10-2, exactly right. But
the profile totals are pulled at one moment and some are stale, so for a
fighter whose recent bouts postdate the pull the subtraction removes wins that
were never in the total. That error is anti-correlated with winning, and the
reconstructed feature scores AUC 0.335 in 2026 - inverted, and worst in exactly
the period predictions are made for. It is not used.
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

# A HAND-WRITTEN LIST IS WHY THIS MODULE MISSED THE WORST LEAK IN THE DATASET.
# PROFILE_COLUMNS named the eight striking and grappling rates and nothing
# else, so r_wins and r_losses - the fighter's LIFETIME record as of the data
# pull, 98.1% and 97.8% constant across careers - were never examined. The win
# rate built from them scores AUC 0.80 in 2024 and 0.58 in 2026, and carried
# the whole apparent skill of the model. scan_all() therefore checks EVERY
# numeric column and the caller must justify each exemption below.
#
# A fighter's frame is genuinely fixed. Being constant across a career is the
# correct behaviour for these, not a leak: nobody's reach changes between
# fights. They are the only columns allowed to be constant without comment.
STATIC_ATTRIBUTES = {
    "r_height": "a fighter's height does not change between bouts",
    "r_reach": "a fighter's reach does not change between bouts",
    "r_weight": "a division label, constant by definition - see matchup.py",
    "r_draws": "near-universally 0; kept only so the exemption is explicit",
}


def scan_all(df, corner="r", exempt=None):
    """Every numeric column that looks like a career total, not a snapshot.

    Exhaustive by construction. A new leaking column added to the dataset
    upstream is caught the next time this runs, without anyone remembering to
    add it to a list - which is exactly the failure that let r_wins through.
    """
    exempt = STATIC_ATTRIBUTES if exempt is None else exempt
    columns = [c for c in df.columns
               if c.startswith(f"{corner}_")
               and c not in exempt
               and pd.api.types.is_numeric_dtype(df[c])]
    return scan(df, columns, corner)


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
    findings = scan_all(df)
    report(findings)
    print()
    print(f"  exempt as genuinely static: "
          f"{', '.join(sorted(STATIC_ATTRIBUTES))}")
    return findings


if __name__ == "__main__":
    main()
