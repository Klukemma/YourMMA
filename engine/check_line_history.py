"""Does any source we can reach carry an OPENING line as well as a closing one?

The question this answers. A closing line is the market's final word and the
card already blends 75% of it. The MOVE from open to close is something else:
it is what the market learned in the fortnight before the bell - the injury
report, the late replacement, the camp news - priced by people with money at
stake. If that movement predicts outcomes beyond where the line ended up, it
is the cheapest version of the soft-information feature an AI researcher would
otherwise be built to chase.

None of it is measurable without paired prices, and right now there are none:

    odds.csv          6,565 fights, ONE price each
    odds_cache.json   30 fights, first and latest, all captured on one day
                      because the cache only started persisting yesterday

So before building anything, this asks whether the upstream historical dataset
has an opening price hiding in a column nobody has looked at. Two answers and
two different projects follow from them:

    it does      the measurement can run this week on thousands of fights
    it does not  the cache has to collect open/close itself, which means
                 polling through fight week and waiting months

This prints columns and counts, never a schema guessed at from a name: a
column called RedOdds might be opening or closing and the distinction is the
whole point, so where both exist it checks that they actually differ.

    python engine/check_line_history.py        (needs Kaggle credentials)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent

# The dataset sync_kaggle already pulls historical moneylines from.
DATASET = "valihameed/ufc-stats"

# Words that would appear in a column holding a price or a movement. Kept
# broad: the point is to see everything odds-shaped, not to confirm a guess.
INTERESTING = ("odd", "open", "close", "line", "price", "money", "implied",
               "ev", "expect", "dec_", "moneyline")


def download(into):
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(into),
         "--unzip"], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"download failed: {result.stderr.strip()[:300]}")
    return sorted(Path(into).rglob("*.csv"))


def report(frame, name):
    columns = [c for c in frame.columns
               if any(word in c.lower() for word in INTERESTING)]
    print(f"\n  {name}   {len(frame):,} rows x {len(frame.columns)} columns")
    if not columns:
        print("    nothing odds-shaped")
        return []
    print(f"    {'column':<28}{'non-null':>10}{'sample values'}")
    print("    " + "-" * 68)
    for column in columns:
        series = frame[column]
        filled = int(series.notna().sum())
        sample = ", ".join(str(v) for v in series.dropna().head(3).tolist())
        print(f"    {column:<28}{filled:>10}  {sample[:34]}")
    return columns


def main():
    with tempfile.TemporaryDirectory() as tmp:
        files = download(Path(tmp))
        print("=" * 76)
        print("IS THERE AN OPENING LINE ANYWHERE IN THE HISTORICAL DATA?")
        print("=" * 76)

        found = {}
        for path in files:
            frame = pd.read_csv(path, low_memory=False)
            columns = report(frame, path.name)
            if columns:
                found[path.name] = (frame, columns)

        print("\n" + "=" * 76)
        opening = {}
        for name, (frame, columns) in found.items():
            hits = [c for c in columns
                    if "open" in c.lower() or "close" in c.lower()]
            if hits:
                opening[name] = (frame, hits)

        if not opening:
            print("  NO OPENING OR CLOSING COLUMN ANYWHERE.")
            print("  Every price in reach is a single number per fight, so the")
            print("  movement cannot be measured from history. The odds cache")
            print("  has to collect it: poll through fight week, and the")
            print("  measurement becomes possible once a few hundred fights")
            print("  have both ends recorded.")
            return

        for name, (frame, hits) in opening.items():
            print(f"  {name} HAS: {hits}")
            # A pair of columns that never disagree is one column twice.
            pairs = [(a, b) for a in hits for b in hits
                     if "open" in a.lower() and "close" in b.lower()]
            for a, b in pairs:
                both = frame[[a, b]].dropna()
                if both.empty:
                    print(f"    {a} and {b} never both present")
                    continue
                differ = (both[a] != both[b]).mean()
                print(f"    {a} vs {b}: both present on {len(both):,} fights, "
                      f"they differ on {differ:.1%}")
                if differ < 0.05:
                    print("      -> they are effectively the same column; this "
                            "is not a movement")
                else:
                    print("      -> A REAL MOVEMENT. The measurement can run "
                          "on these.")


if __name__ == "__main__":
    main()
