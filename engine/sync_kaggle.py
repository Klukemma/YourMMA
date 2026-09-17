"""Pull the UFC dataset from Kaggle and merge new fights into the local CSV.

    python engine/sync_kaggle.py inspect     # download, report schema, change nothing
    python engine/sync_kaggle.py sync        # merge new fights and rate them
    python engine/sync_kaggle.py sync --dry-run

Source: https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025

Auth: set KAGGLE_API_TOKEN, or put the token in ~/.kaggle/access_token.
Never commit the token - this repository is public.

New fights are rated by continuing each fighter's existing TrueSkill series
(see ratings.extend), so rows already in the file keep the exact values the
model was trained on.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ratings import DEFAULT, extend

DATASET = "neelagiriaditya/ufc-datasets-1994-2025"
ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = ENGINE_DIR / "data"
LOCAL_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))

# Columns the model cannot work without.
KEY_COLUMNS = ["date", "event_name", "r_name", "b_name", "winner"]
RATING_COLUMNS = ["r_mu_pre", "r_sigma_pre", "r_mmr_pre",
                  "b_mu_pre", "b_sigma_pre", "b_mmr_pre"]


def _have_token():
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or (Path.home() / ".kaggle" / "access_token").exists()


def download(dest):
    """Download and unzip the dataset. Returns the directory it landed in."""
    if not _have_token():
        sys.exit("No Kaggle credentials. Set KAGGLE_API_TOKEN or write "
                 "~/.kaggle/access_token, then retry.")
    if shutil.which("kaggle") is None:
        sys.exit("The kaggle CLI is not installed. pip install kaggle")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Kaggle download failed:\n{result.stdout}\n{result.stderr}")
    files = sorted(dest.rglob("*.csv"))
    if not files:
        sys.exit(f"No CSV found in the download. Contents: {list(dest.iterdir())}")
    return files


def series_is_datelike(series):
    """Cheap check: does this column parse as dates?"""
    sample = series.dropna().head(20)
    if sample.empty or sample.dtype.kind in 'if':
        return False
    try:
        return pd.to_datetime(sample, errors='coerce', format='mixed').notna().mean() > 0.8
    except Exception:
        return False


def cmd_inspect(args):
    """Dump enough of the upstream schema to design a join against our layout.

    The upstream dataset is relational (event / fight / fighter / round tables)
    while the local file is one wide row per bout, so this prints every column
    of every table with sample values, not just a column diff.
    """
    with tempfile.TemporaryDirectory() as tmp:
        files = download(Path(tmp))
        local = pd.read_csv(LOCAL_CSV, low_memory=False, nrows=3)

        print(f"\nLOCAL: {LOCAL_CSV.name}")
        print(f"  {len(local.columns)} columns, one row per bout")
        print(f"  key columns: {KEY_COLUMNS}")

        for f in sorted(files):
            df = pd.read_csv(f, low_memory=False)
            print("\n" + "=" * 72)
            print(f"{f.name}   {len(df):,} rows x {len(df.columns)} columns")
            print("-" * 72)
            for col in df.columns:
                sr = df[col]
                sample = [str(x)[:26] for x in sr.dropna().head(2).tolist()]
                print(f"  {col:<26} {str(sr.dtype):<8} nulls={sr.isna().mean():5.1%} "
                      f"uniq={sr.nunique():<7} {sample}")
            ids = [c for c in df.columns if c.lower().endswith('id')]
            dates = [c for c in df.columns if 'date' in c.lower() or series_is_datelike(df[c])]
            if ids:
                print(f"  -> id-like   : {ids}")
            if dates:
                print(f"  -> date-like : {dates}")

        print("\n" + "=" * 72)
        print("Next: decide which tables join to reproduce one wide row per bout.")
        print("COLUMN_MAP alone is not enough for a relational source.")


def _canon(col):
    """Reduce a column name to a comparable token set.

    Upstream and local disagree on wording, not meaning:
    r_total_sig_str_landed_head  vs  r_head_landed
    """
    c = col.lower()
    for noise in ('_total', 'total_', 'seconds', '_no'):
        c = c.replace(noise, '_')
    c = (c.replace('atmp', 'atmpted').replace('atmptedted', 'atmpted')
           .replace('success', 'landed').replace('sig_str', 'sig')
           .replace('significant', 'sig').replace('distance', 'dist'))
    return frozenset(t for t in c.split('_') if t and t not in ('str',))


def cmd_propose_map(args):
    """Emit a candidate COLUMN_MAP by matching upstream columns to local ones.

    Prints only the proposal and the unmatched local columns, so the output is
    reviewable. Nothing is written - the mapping is committed by hand after
    checking it.
    """
    local = pd.read_csv(LOCAL_CSV, low_memory=False, nrows=5)
    local_cols = list(local.columns)

    with tempfile.TemporaryDirectory() as tmp:
        files = {f.name: pd.read_csv(f, low_memory=False, nrows=200)
                 for f in download(Path(tmp))}

    wide = files.get('master.csv')
    if wide is None:
        sys.exit(f"master.csv not found. Got: {list(files)}")

    remote_cols = list(wide.columns)
    remote_by_canon = {}
    for rc in remote_cols:
        remote_by_canon.setdefault(_canon(rc), []).append(rc)

    exact, canon, unmatched = {}, {}, []
    for lc in local_cols:
        if lc in remote_cols:
            exact[lc] = lc
            continue
        hits = remote_by_canon.get(_canon(lc), [])
        if len(hits) == 1:
            canon[hits[0]] = lc
        else:
            unmatched.append((lc, hits))

    print(f"\nmaster.csv: {len(wide.columns)} columns   local: {len(local_cols)} columns")
    print(f"identical names      : {len(exact)}")
    print(f"matched by meaning   : {len(canon)}")
    print(f"unmatched local cols : {len(unmatched)}")

    print("\n# --- proposed COLUMN_MAP (kaggle_name -> local_name) ---")
    print("COLUMN_MAP = {")
    for rc, lc in sorted(canon.items(), key=lambda kv: kv[1]):
        print(f'    {rc!r}: {lc!r},')
    print("}")

    print("\n# --- local columns with no confident source ---")
    for lc, hits in unmatched:
        note = f"  ambiguous: {hits}" if hits else ""
        print(f"    {lc}{note}")

    print("\n# --- upstream columns we would not use ---")
    used = set(exact) | set(canon)
    print(sorted(c for c in remote_cols if c not in used))

    for name in ('fighter.csv', 'event.csv'):
        if name in files:
            print(f"\n# --- {name} columns ---")
            print(sorted(files[name].columns))


# Fill this in once `inspect` shows the upstream names. {kaggle_name: local_name}
COLUMN_MAP = {}


def _normalise(df):
    if COLUMN_MAP:
        df = df.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def _fight_key(df):
    """Identity for a bout, so re-running sync cannot duplicate rows."""
    return (df["date"].dt.strftime("%Y-%m-%d") + "|"
            + df["r_name"].astype(str).str.strip().str.lower() + "|"
            + df["b_name"].astype(str).str.strip().str.lower())


def cmd_sync(args):
    local = pd.read_csv(LOCAL_CSV, low_memory=False)
    local["date"] = pd.to_datetime(local["date"], errors="coerce")
    print(f"Local: {len(local):,} fights through {local['date'].max().date()}")

    with tempfile.TemporaryDirectory() as tmp:
        files = download(Path(tmp))
        frames = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            if COLUMN_MAP:
                df = df.rename(columns=COLUMN_MAP)
            if all(c in df.columns for c in KEY_COLUMNS):
                frames.append(_normalise(df))
            else:
                missing = [c for c in KEY_COLUMNS if c not in df.columns]
                print(f"  skipping {f.name}: missing {missing}")
        if not frames:
            sys.exit("No downloaded file had the required columns. Run `inspect` "
                     "and fill in COLUMN_MAP.")
        remote = pd.concat(frames, ignore_index=True)

    print(f"Kaggle: {len(remote):,} fights through {remote['date'].max().date()}")

    known = set(_fight_key(local))
    new = remote[~_fight_key(remote).isin(known)].copy()
    # Only bouts after our last date - avoids back-filling gaps that the dedup
    # step deliberately removed.
    new = new[new["date"] > local["date"].max()]
    print(f"New fights: {len(new):,}")
    if new.empty:
        print("Already up to date.")
        return

    print(f"  {new['date'].min().date()} -> {new['date'].max().date()}")
    for ev, n in new.groupby("event_name").size().sort_values(ascending=False).items():
        print(f"    {n:3d}  {ev}")

    missing_ratings = [c for c in RATING_COLUMNS if c not in new.columns or new[c].isna().all()]
    if missing_ratings:
        print(f"\nRating columns absent upstream (expected): computing with "
              f"TrueSkill beta={DEFAULT.beta}, tau={DEFAULT.tau}")

    combined = extend(local, new)
    print(f"Combined: {len(combined):,} fights through {combined['date'].max().date()}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    backup = LOCAL_CSV.with_suffix(f".backup-{local['date'].max().date()}.csv")
    if not backup.exists():
        shutil.copy2(LOCAL_CSV, backup)
        print(f"Backed up to {backup.name}")
    combined.to_csv(LOCAL_CSV, index=False)
    print(f"Wrote {LOCAL_CSV}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect", help="download and report the upstream schema")
    sub.add_parser("propose-map", help="suggest a COLUMN_MAP from upstream to local")
    s = sub.add_parser("sync", help="merge new fights into the local CSV")
    s.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()
    {"inspect": cmd_inspect, "propose-map": cmd_propose_map,
     "sync": cmd_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
