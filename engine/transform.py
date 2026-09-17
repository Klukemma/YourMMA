"""Turn upstream Kaggle tables into rows in our local schema.

Reads master.csv (one row per bout) plus fighter.csv (career profiles) and
produces a frame with our 130 columns, minus the rating columns which
ratings.extend() fills in afterwards.

Nothing here guesses at a column name - see schema_map.py for why the mapping
is declared by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema_map as sm


def _to_seconds(value):
    """'4:31' -> 271. Already-numeric values pass through."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if ':' not in text:
        try:
            return float(text)
        except ValueError:
            return np.nan
    parts = text.split(':')
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return np.nan
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


def _rounds_from_format(value):
    """'3 Rnd (5-5-5)' -> 3. Falls back to counting the per-round segment."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    head = text.strip().split()
    if head and head[0].isdigit():
        return float(head[0])
    if '(' in text and ')' in text:
        inner = text[text.index('(') + 1:text.index(')')]
        segments = [s for s in inner.replace('-', ' ').split() if s]
        if segments:
            return float(len(segments))
    if 'no time limit' in text.lower():
        return 1.0
    return np.nan


def _safe_ratio(numerator, denominator, scale=100.0):
    """Percentage, with 0/0 left as NaN rather than becoming 0 or inf."""
    num = pd.to_numeric(numerator, errors='coerce')
    den = pd.to_numeric(denominator, errors='coerce')
    out = np.where(den > 0, num / den * scale, np.nan)
    return pd.Series(out, index=num.index).round(2)


def _is_title_fight(weight_class):
    if pd.isna(weight_class):
        return 0
    return int('title' in str(weight_class).lower())


def _clean_division(weight_class):
    """'UFC Lightweight Title Bout' -> 'lightweight', matching local style."""
    if pd.isna(weight_class):
        return np.nan
    text = str(weight_class).lower()
    for noise in ('ufc ', ' title', ' bout', 'interim ', 'ultimate fighter ',
                  'tournament ', 'championship'):
        text = text.replace(noise, ' ')
    return ' '.join(text.split()).strip() or np.nan


def transform(master, fighters=None):
    """Build local-schema rows from upstream tables.

    Args:
        master: master.csv as a DataFrame
        fighters: fighter.csv as a DataFrame, for career profile columns

    Returns:
        DataFrame in the local schema. Rating columns are absent - they are
        filled by ratings.extend() once the rows are in date order.
    """
    missing = [c for c in ('winner_id', 'r_fighter_id', 'b_fighter_id')
               if c not in master.columns]
    if missing:
        raise ValueError(f"master.csv is missing {missing}; cannot resolve winners")

    out = pd.DataFrame(index=master.index)

    # 1. Straight renames.
    for upstream, local in sm.COLUMN_MAP.items():
        if upstream in master.columns:
            out[local] = master[upstream]

    # 2. Unit conversions.
    if 'finish_time' in master.columns:
        out['match_time_sec'] = master['finish_time'].map(_to_seconds)
    if 'time_format' in master.columns:
        out['total_rounds'] = master['time_format'].map(_rounds_from_format)
    if 'rounds_fought' in master.columns:
        out['finish_round'] = pd.to_numeric(master['rounds_fought'], errors='coerce')

    # 3. Winner: whichever corner's id matches winner_id. A draw or no-contest
    #    leaves winner_id unmatched, and the result stays blank rather than
    #    defaulting to a corner.
    winner_id = master['winner_id']
    out['winner'] = np.select(
        [winner_id.eq(master['r_fighter_id']) & winner_id.notna(),
         winner_id.eq(master['b_fighter_id']) & winner_id.notna()],
        [master.get('r_fighter_name'), master.get('b_fighter_name')],
        default=np.nan,
    )

    # 4. Title fights and division, both read from the weight class text.
    if 'weight_class' in master.columns:
        out['title_fight'] = master['weight_class'].map(_is_title_fight)
        out['division'] = master['weight_class'].map(_clean_division)

    # 5. Career profiles joined from fighter.csv.
    if fighters is not None and 'fighter_id' in fighters.columns:
        profile = fighters.drop_duplicates('fighter_id').set_index('fighter_id')
        for corner in ('r', 'b'):
            id_col = f'{corner}_fighter_id'
            if id_col not in master.columns:
                continue
            for local_col, upstream_col in sm.profile_columns(corner).items():
                if upstream_col in profile.columns:
                    out[local_col] = master[id_col].map(profile[upstream_col])

    # 6. Percentages computed from landed/attempted, not read.
    for corner in ('r', 'b'):
        for local_col, (landed, attempted) in sm.accuracy_columns(corner).items():
            if landed in out.columns and attempted in out.columns:
                out[local_col] = _safe_ratio(out[landed], out[attempted])
        for local_col, (part, whole) in sm.share_columns(corner).items():
            if part in out.columns and whole in out.columns:
                out[local_col] = _safe_ratio(out[part], out[whole])

    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'], errors='coerce')
    return out


def _as_number(value):
    """Coerce a cell to a float, or None if it is not one.

    Cells are not always numbers: the dataset shipped one row where a
    name lookup matched two fighters and wrote a whole Series repr into the
    field (see data_quality.py). A bad cell should read as unknown, not stop
    the sync.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def carry_forward_records(existing, new_rows):
    """Fill win/loss/draw records, which upstream does not carry.

    These are the fighter's overall professional record, including bouts
    outside the UFC, so they cannot be recomputed from this dataset. Each
    fighter's last known record is carried forward from the existing file and
    incremented by results we observe. Fighters with no history are left blank
    rather than assumed to be 0-0.
    """
    record = {}
    existing_sorted = existing.sort_values('date', kind='mergesort')
    for row in existing_sorted.itertuples(index=False):
        for corner in ('r', 'b'):
            name = getattr(row, f'{corner}_name', None)
            if not isinstance(name, str) or not name:
                continue
            wins = _as_number(getattr(row, f'{corner}_wins', None))
            losses = _as_number(getattr(row, f'{corner}_losses', None))
            draws = _as_number(getattr(row, f'{corner}_draws', None))
            if wins is not None:
                record[name] = [wins, losses or 0.0, draws or 0.0]

    out = new_rows.sort_values('date', kind='mergesort').copy()
    cols = {f'{c}_{k}': [] for c in ('r', 'b') for k in ('wins', 'losses', 'draws')}
    for row in out.itertuples(index=False):
        red, blue, winner = row.r_name, row.b_name, getattr(row, 'winner', None)
        for corner, name in (('r', red), ('b', blue)):
            current = record.get(name)
            for idx, key in enumerate(('wins', 'losses', 'draws')):
                cols[f'{corner}_{key}'].append(current[idx] if current else np.nan)
        # Update after recording pre-fight values.
        for name in (red, blue):
            if name not in record:
                continue
            if not isinstance(winner, str) or winner not in (red, blue):
                record[name][2] += 1           # draw or no contest
            elif name == winner:
                record[name][0] += 1
            else:
                record[name][1] += 1
    for col, values in cols.items():
        out[col] = values
    return out
