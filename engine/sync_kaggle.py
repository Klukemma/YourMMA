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


def describe(path):
    df = pd.read_csv(path, low_memory=False, nrows=5000)
    return {"path": path, "columns": list(df.columns), "sample_rows": len(df)}


def cmd_inspect(args):
    with tempfile.TemporaryDirectory() as tmp:
        files = download(Path(tmp))
        local = pd.read_csv(LOCAL_CSV, low_memory=False, nrows=5)
        local_cols = set(local.columns)

        print(f"\nLocal file: {LOCAL_CSV.name} ({len(local.columns)} columns)\n")
        for f in files:
            info = describe(f)
            remote_cols = set(info["columns"])
            shared = local_cols & remote_cols
            print("=" * 70)
            print(f"{f.name}: {len(info['columns'])} columns")
            print(f"  shared with local : {len(shared)}")
            print(f"  missing key cols  : {[c for c in KEY_COLUMNS if c not in remote_cols] or 'none'}")
            only_remote = sorted(remote_cols - local_cols)
            only_local = sorted(local_cols - remote_cols)
            if only_remote:
                print(f"  only in Kaggle    : {only_remote[:15]}{' ...' if len(only_remote) > 15 else ''}")
            if only_local:
                print(f"  only in local     : {only_local[:15]}{' ...' if len(only_local) > 15 else ''}")
        print("\nIf key columns are named differently upstream, add a mapping to "
              "COLUMN_MAP in this file before running sync.")


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
    s = sub.add_parser("sync", help="merge new fights into the local CSV")
    s.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()
    {"inspect": cmd_inspect, "sync": cmd_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
