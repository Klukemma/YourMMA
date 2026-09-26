"""Detect and repair cells that hold a pandas Series repr instead of a value.

Whatever built the dataset looked fighters up by name. When a name matched two
different people it wrote the whole matched Series into the cell:

    r_wins = "name_norm\\nbruno silva    14\\nbruno silva    23\\nName: wins, dtype: int64"

Those cells read as strings, so pd.to_numeric turns them into NaN and the row
quietly becomes missing data rather than raising anything.
"""

import re

import pandas as pd

SERIES_REPR = re.compile(r"Name:\s*\w+,\s*dtype:", re.MULTILINE)


def is_series_repr(value):
    """True if this cell holds a printed Series rather than a value."""
    return isinstance(value, str) and bool(SERIES_REPR.search(value))


def parse_series_repr(text):
    """Pull the values out of a printed Series, in order.

    "name_norm\\nbruno silva    14\\nbruno silva    23\\nName: wins, dtype: int64"
    -> ['14', '23']
    """
    values = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith('Name:') or '  ' not in line:
            continue
        values.append(line.rsplit('  ', 1)[-1].strip())
    return [v for v in values if v]


def find_corrupt_cells(df):
    """[(row_index, column)] for every cell holding a Series repr."""
    found = []
    for col in df.columns:
        # pandas 2 gives text columns dtype object; pandas 3 gives them str.
        # Checking for object alone silently skips every corrupt cell on
        # pandas 3, so test for text rather than for a specific dtype.
        if not (pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])):
            continue
        mask = df[col].map(is_series_repr).fillna(False).astype(bool)
        found.extend((idx, col) for idx in df.index[mask])
    return found


def describe_corruption(df):
    """Group corrupt cells by row, for reporting."""
    by_row = {}
    for idx, col in find_corrupt_cells(df):
        by_row.setdefault(idx, []).append(col)
    return by_row
