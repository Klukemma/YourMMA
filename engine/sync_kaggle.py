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
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema_map
from ratings import DEFAULT, extend
from transform import carry_forward_records, transform

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


# Positional strike columns: upstream keeps a "sig" token that local drops.
#   local     r_head_landed
#   upstream  r_total_sig_str_landed_head
_POSITIONS = {'head', 'body', 'leg', 'dist', 'clinch', 'ground'}

# Tokens that carry no meaning for matching, only house style.
_NOISE = {'total', 'str', 'fighter', 'inches', 'lbs', 'seconds', 'no'}

_SYNONYM = {
    'atmp': 'atmpted', 'attempted': 'atmpted', 'attempts': 'atmpted',
    'success': 'landed', 'distance': 'dist', 'slpm': 'splm',
    'percent': 'per', 'pct': 'per', 'rnd': 'round',
}


def _canon(col):
    """Reduce a column name to a comparable token set."""
    tokens = [_SYNONYM.get(t, t) for t in str(col).lower().split('_')]
    tokens = [t for t in tokens if t and t not in _NOISE]
    # A positional breakdown column is already unambiguous without "sig".
    if _POSITIONS & set(tokens):
        tokens = [t for t in tokens if t != 'sig']
    return frozenset(tokens)


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

    print("\n# --- every master.csv column, for writing the map by hand ---")
    print(sorted(remote_cols))

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


def _load_upstream(tmpdir):
    """Download and return {filename: DataFrame}."""
    return {f.name: pd.read_csv(f, low_memory=False) for f in download(Path(tmpdir))}


def _excluded_fight_ids(tables):
    """fight_ids upstream flagged as incompletely parsed.

    scrape_error.csv records PartialFightParse failures. Importing those rows
    would add bouts with missing round data, which reads as a real zero to
    every downstream feature.
    """
    errors = tables.get('scrape_error.csv')
    if errors is None or 'entity_id' not in errors.columns:
        return set()
    unresolved = errors
    if 'resolved' in errors.columns:
        unresolved = errors[errors['resolved'].fillna(0).astype(int) == 0]
    if 'entity_type' in unresolved.columns:
        unresolved = unresolved[unresolved['entity_type'].astype(str) == 'fight']
    return set(unresolved['entity_id'].dropna().astype(str))



# Ratings sit on a continuous per-fighter series and the win/loss records are
# point-in-time, both computed from local history rather than from any single
# upstream row. A repair must not touch either.
PROTECTED_FROM_REPAIR = (set(schema_map.RATING_COLUMNS)
                         | set(schema_map.CARRIED_FORWARD)
                         | {"date", "r_name", "b_name"})

# Absence of this column is how a bout with no scraped statistics announces
# itself: it is recorded for every fight that has any statistics at all.
STAT_PROBE = "r_total_str_landed"


def blank_stat_rows(local, probe=STAT_PROBE):
    """Boolean mask of bouts whose measured statistics never arrived."""
    if probe not in local.columns:
        return pd.Series(False, index=local.index)
    return local[probe].isna()


def repair_blank_stats(local, fresh, probe=STAT_PROBE):
    """Fill cells that are empty locally but present upstream.

    cmd_sync only ever appends: it takes the rows it has not seen and whose
    date is past the local cutoff, so a bout published before its statistics
    were scraped keeps its blanks permanently, even once upstream backfills
    them. That is how 145 fights between 2025-09-13 and 2025-12-06 ended up
    with no striking, takedown or control data at all.

    Only null cells are filled. A value that is already present is never
    overwritten, so nothing the ratings or any earlier analysis depend on can
    move underneath them.

    Returns (repaired, report).
    """
    mask = blank_stat_rows(local, probe)
    report = {"blank_rows": int(mask.sum()), "matched": 0,
              "cells_filled": 0, "rows_repaired": 0, "unmatched": []}
    if not mask.any():
        return local, report

    upstream = fresh.set_index(_fight_key(fresh))
    upstream = upstream[~upstream.index.duplicated(keep="first")]

    repaired = local.copy()
    keys = _fight_key(local)
    fillable = [c for c in local.columns
                if c in fresh.columns and c not in PROTECTED_FROM_REPAIR]

    for idx in local.index[mask]:
        key = keys.loc[idx]
        if key not in upstream.index:
            report["unmatched"].append(key)
            continue
        report["matched"] += 1
        source = upstream.loc[key]
        filled = 0
        for col in fillable:
            if pd.isna(repaired.at[idx, col]) and not pd.isna(source[col]):
                repaired.at[idx, col] = source[col]
                filled += 1
        report["cells_filled"] += filled
        if filled and not pd.isna(repaired.at[idx, probe]):
            report["rows_repaired"] += 1

    return repaired, report


def cmd_repair(args):
    """Re-fetch upstream and fill in bouts whose statistics never arrived."""
    local = pd.read_csv(LOCAL_CSV, low_memory=False)
    local["date"] = pd.to_datetime(local["date"], errors="coerce")

    mask = blank_stat_rows(local)
    print(f"Local: {len(local):,} fights, {int(mask.sum()):,} with no statistics")
    if not mask.any():
        print("Nothing to repair.")
        return
    blank_dates = local.loc[mask, "date"]
    print(f"  spanning {blank_dates.min().date()} -> {blank_dates.max().date()}")

    with tempfile.TemporaryDirectory() as tmp:
        tables = _load_upstream(tmp)
    master = tables.get("master.csv")
    if master is None:
        sys.exit(f"master.csv not found upstream. Got: {sorted(tables)}")

    fresh = transform(master, tables.get("fighter.csv"), tables.get("round.csv"))
    print(f"Kaggle: {len(fresh):,} fights through {fresh['date'].max().date()}")

    repaired, report = repair_blank_stats(local, fresh)
    print("")
    print(f"Blank rows        : {report['blank_rows']:,}")
    print(f"Matched upstream  : {report['matched']:,}")
    print(f"Fully repaired    : {report['rows_repaired']:,}")
    print(f"Cells filled      : {report['cells_filled']:,}")
    if report["unmatched"]:
        print(f"Not found upstream: {len(report['unmatched']):,}")
        for key in report["unmatched"][:10]:
            print(f"    {key}")

    if report["cells_filled"] == 0:
        print("")
        print("Upstream has no data for these bouts either; nothing written.")
        return

    still = int(blank_stat_rows(repaired).sum())
    print(f"Still blank after : {still:,}")

    if args.dry_run:
        print("")
        print("--dry-run: nothing written.")
        return

    backup = LOCAL_CSV.with_suffix(".backup-before-repair.csv")
    if not backup.exists():
        shutil.copy2(LOCAL_CSV, backup)
        print(f"Backed up to {backup.name}")
    repaired.to_csv(LOCAL_CSV, index=False)
    print(f"Wrote {LOCAL_CSV}")


def cmd_sync(args):
    local = pd.read_csv(LOCAL_CSV, low_memory=False)
    local['date'] = pd.to_datetime(local['date'], errors='coerce')
    cutoff = local['date'].max()
    print(f"Local:  {len(local):,} fights through {cutoff.date()}")

    with tempfile.TemporaryDirectory() as tmp:
        tables = _load_upstream(tmp)

    master = tables.get('master.csv')
    if master is None:
        sys.exit(f"master.csv not found upstream. Got: {sorted(tables)}")

    rows = transform(master, tables.get('fighter.csv'), tables.get('round.csv'))
    print(f"Kaggle: {len(rows):,} fights through {rows['date'].max().date()}")

    skip = _excluded_fight_ids(tables)
    if skip and 'fight_id' in rows.columns:
        bad = rows['fight_id'].astype(str).isin(skip)
        if bad.any():
            print(f"  excluding {bad.sum():,} fights flagged as partially parsed upstream")
            rows = rows[~bad]

    known = set(_fight_key(local))
    new = rows[~_fight_key(rows).isin(known)]
    new = new[new['date'] > cutoff].copy()
    print(f"New fights: {len(new):,}")
    if new.empty:
        print("Already up to date.")
        return

    print(f"  {new['date'].min().date()} -> {new['date'].max().date()}")
    counts = new.groupby('event_name').size().sort_values(ascending=False)
    for event, n in counts.items():
        print(f"    {n:3d}  {event}")

    new = carry_forward_records(local, new)

    # Ratings continue each fighter's existing series; existing rows never move.
    combined = extend(local, new)
    combined = combined.reindex(columns=list(local.columns))
    print(f"Combined: {len(combined):,} fights through {combined['date'].max().date()}")

    blank = [c for c in local.columns if combined.tail(len(new))[c].isna().all()]
    if blank:
        print(f"  columns with no data in the new rows: {blank}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    backup = LOCAL_CSV.with_suffix(f".backup-{cutoff.date()}.csv")
    if not backup.exists():
        shutil.copy2(LOCAL_CSV, backup)
        print(f"Backed up to {backup.name}")
    combined.to_csv(LOCAL_CSV, index=False)
    print(f"Wrote {LOCAL_CSV}")


def cmd_search_odds(args):
    """Look for a Kaggle dataset carrying historical UFC betting odds.

    The Odds API serves upcoming events only, so it cannot price fights that
    have already happened. Without a historical source there is no way to
    compute ROI on the 2026 predictions.
    """
    queries = ['ufc odds', 'ufc betting odds', 'mma odds', 'ufc moneyline',
               'ufc betting', 'ufc fights odds']
    seen = {}
    for q in queries:
        print(f"\n--- searching: {q!r} ---")
        result = subprocess.run(['kaggle', 'datasets', 'list', '-s', q, '--csv'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  failed: {result.stderr.strip()[:200]}")
            continue
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        if len(lines) < 2:
            print("  no results")
            continue
        import csv as _csv
        for row in _csv.DictReader(lines):
            ref = row.get('ref')
            if ref and ref not in seen:
                seen[ref] = row
                print(f"  {ref}")
                print(f"      {row.get('title','')[:70]}  "
                      f"size={row.get('size','?')}  updated={row.get('lastUpdated','')[:10]}")

    print(f"\n{'='*70}")
    print(f"{len(seen)} candidate datasets. Inspect one with:")
    print("  kaggle datasets files <ref>")


# Shortlisted by `search-odds`. The rest of the 23 results end in 2024/2025,
# before the period we need to price.
ODDS_CANDIDATES = [
    'p0p0xyz/ufc-fights-ml-with-odds-csv',            # "ml" = moneyline, updated 2026-01-14
    'oliviersportsdata/ufc-multimarket-sample-2025',  # updated 2026-06-24
    'martnoisrodgz/ufc-events-fight-results-2026',    # 2026 events
    'juanpez24/ufc-fight-outcome-prediction-1994-2025',
]


def cmd_inspect_odds(args):
    """Download each candidate and report whether it can price our predictions."""
    history = json.loads((DATA_DIR / 'prediction_history.json').read_text())
    need = pd.to_datetime([p['event_date'] for p in history['predictions']])
    print(f"predictions to price: {need.min().date()} -> {need.max().date()}\n")

    for ref in ODDS_CANDIDATES:
        print("=" * 72)
        print(ref)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', ref, '-p', tmp, '--unzip'],
                capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  download failed: {result.stderr.strip()[:160]}")
                continue
            for f in sorted(Path(tmp).rglob('*.csv')):
                try:
                    df = pd.read_csv(f, low_memory=False)
                except Exception as exc:
                    print(f"  {f.name}: unreadable ({exc})")
                    continue
                odds_cols = [c for c in df.columns if any(
                    k in c.lower() for k in ('odd', 'moneyline', 'ml_', 'line', 'price', 'book'))]
                date_cols = [c for c in df.columns if 'date' in c.lower()]
                print(f"  {f.name}: {len(df):,} rows x {len(df.columns)} cols")
                print(f"    odds-like : {odds_cols[:8] or 'NONE'}")
                if not odds_cols:
                    continue
                for dc in date_cols[:1]:
                    parsed = pd.to_datetime(df[dc], errors='coerce')
                    if parsed.notna().any():
                        print(f"    {dc}: {parsed.min().date()} -> {parsed.max().date()}")
                        covers = ((parsed >= need.min()) & (parsed <= need.max())).sum()
                        print(f"    rows inside our window: {covers:,}")
                print(f"    columns   : {sorted(df.columns)[:24]}")


ODDS_DATASET = 'martnoisrodgz/ufc-events-fight-results-2026'
ODDS_FILE = DATA_DIR / 'odds.csv'


def detect_odds_format(values):
    """American (-150, +130) or decimal (1.67, 2.30)?

    Guessing wrong silently inverts every payout, so this decides from the
    values and says which it picked.
    """
    v = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if v.empty:
        return 'unknown'
    if (v.abs() >= 100).mean() > 0.8:
        return 'american'
    if ((v > 1.0) & (v < 60)).mean() > 0.8:
        return 'decimal'
    return 'unknown'


def decimal_to_american(d):
    d = float(d)
    if d <= 1.0:
        return None
    return round((d - 1) * 100) if d >= 2.0 else -round(100 / (d - 1))


def cmd_fetch_odds(args):
    """Pull historical odds and write them in the schema roi.py expects."""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', ODDS_DATASET, '-p', tmp, '--unzip'],
            capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"download failed: {result.stderr.strip()[:300]}")
        files = sorted(Path(tmp).rglob('*.csv'))
        frames = [pd.read_csv(f, low_memory=False) for f in files]

    df = next((f for f in frames if {'RedOdds', 'BlueOdds', 'RedCorner',
                                     'BlueCorner', 'Date'} <= set(f.columns)), None)
    if df is None:
        sys.exit(f"no file had the expected columns. Got: "
                 f"{[sorted(f.columns)[:8] for f in frames]}")

    fmt_red = detect_odds_format(df['RedOdds'])
    fmt_blue = detect_odds_format(df['BlueOdds'])
    print(f"odds format: RedOdds={fmt_red}, BlueOdds={fmt_blue}")
    print(f"sample: {df[['RedOdds', 'BlueOdds']].head(3).to_dict('records')}")
    if fmt_red != fmt_blue or fmt_red == 'unknown':
        sys.exit(f"cannot determine the odds format ({fmt_red}/{fmt_blue}); "
                 f"refusing to guess, since guessing inverts every payout")

    out = pd.DataFrame({
        'date': pd.to_datetime(df['Date'], errors='coerce'),
        'fighter_a': df['RedCorner'].astype(str).str.strip(),
        'fighter_b': df['BlueCorner'].astype(str).str.strip(),
        'odds_a': pd.to_numeric(df['RedOdds'], errors='coerce'),
        'odds_b': pd.to_numeric(df['BlueOdds'], errors='coerce'),
    })
    if fmt_red == 'decimal':
        print("converting decimal odds to American")
        out['odds_a'] = out['odds_a'].map(lambda d: decimal_to_american(d) if pd.notna(d) else None)
        out['odds_b'] = out['odds_b'].map(lambda d: decimal_to_american(d) if pd.notna(d) else None)

    before = len(out)
    out = out.dropna(subset=['date', 'odds_a', 'odds_b'])
    print(f"rows: {before} -> {len(out)} with a date and both prices")
    if len(out):
        print(f"range: {out['date'].min().date()} -> {out['date'].max().date()}")
    ODDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ODDS_FILE, index=False)
    print(f"wrote {ODDS_FILE}")


# Non-UFC promotions and general MMA databases. The model knows nothing about
# a fighter until their second UFC bout, which is when it is weakest, so records
# from everywhere else are the gap worth filling.
MMA_QUERIES = [
    'bellator', 'pfl mma', 'one championship mma', 'rizin mma',
    'cage warriors', 'ksw mma', 'invicta fc',
    'mma fighters dataset', 'mma fight results', 'sherdog',
    'professional mma records', 'mixed martial arts dataset',
]


def cmd_search_mma(args):
    """Find non-UFC fight data on Kaggle.

    Reports rows and last-updated for each hit so a big, current database is
    distinguishable from a one-event scrape, which the titles alone do not tell
    you.
    """
    import csv as _csv
    seen = {}
    for query in MMA_QUERIES:
        result = subprocess.run(['kaggle', 'datasets', 'list', '-s', query, '--csv'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {query!r}: failed - {result.stderr.strip()[:120]}")
            continue
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        for row in _csv.DictReader(lines):
            ref = row.get('ref')
            if not ref or ref in seen:
                continue
            seen[ref] = row
            title = (row.get('title') or '')[:58]
            try:
                size = int(row.get('size') or 0)
            except ValueError:
                size = 0
            print(f"  {ref}")
            print(f"      {title}")
            print(f"      size={size:>10,}  updated={(row.get('lastUpdated') or '')[:10]}"
                  f"  votes={row.get('voteCount', '?')}  query={query!r}")

    print(f"\n{'=' * 72}")
    print(f"{len(seen)} distinct datasets.")
    big = {k: v for k, v in seen.items()
           if (v.get('size') or '0').isdigit() and int(v['size']) > 200_000}
    print(f"{len(big)} over 200KB, i.e. plausibly a real database rather than "
          f"one event:")
    for ref in sorted(big):
        print(f"  {ref}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect", help="download and report the upstream schema")
    sub.add_parser("propose-map", help="suggest a COLUMN_MAP from upstream to local")
    sub.add_parser("search-odds", help="look for a Kaggle dataset with historical odds")
    sub.add_parser("inspect-odds", help="check whether shortlisted odds datasets cover our window")
    sub.add_parser("search-mma", help="find non-UFC fight data on Kaggle")
    sub.add_parser("fetch-odds", help="download historical odds into data/odds.csv")
    r = sub.add_parser("repair", help="refill bouts whose statistics never arrived")
    r.add_argument("--dry-run", action="store_true", help="report without writing")
    s = sub.add_parser("sync", help="merge new fights into the local CSV")
    s.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()
    {"inspect": cmd_inspect, "propose-map": cmd_propose_map,
     "search-odds": cmd_search_odds, "inspect-odds": cmd_inspect_odds,
     "search-mma": cmd_search_mma,
     "fetch-odds": cmd_fetch_odds, "repair": cmd_repair,
     "sync": cmd_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
