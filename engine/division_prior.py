"""How often does THIS division finish? Point in time, from earlier fights only.

Heavyweight finishes 63.9% of the time and women's strawweight 33.6% - a
thirty-point spread, and the largest single effect measured on the method
question so far. The model currently gets one binary flag, is_heavyweight,
and nothing else about the division.

THIS FEATURE IS BUILT FROM THE TARGET, which is the dangerous kind. A finish
rate computed over the whole dataset and joined onto every bout would tell
each fight how often fights like it end early INCLUDING ITSELF, and the model
would learn to read its own answer. That is the shape of both leaks this
project has already found and removed.

So it is accumulated the way career_stats accumulates a record: in date order,
each fight seeing only the fights that came before it. Fight number one in a
division gets the league prior, not its own result. The test suite asserts
that shuffling the outcomes of LATER fights cannot change an EARLIER fight's
value - which is the only property that matters and the one a whole-dataset
groupby would fail.

Shrunk toward the league rate, because a division with four bouts on the
books has a finish rate that is mostly noise:

    prior = (finishes + k * league) / (bouts + k)

k is the number of fights at which a division's own rate and the league rate
are weighted equally. Set from split-half reliability rather than by taste:
see K_FINISH below.
"""

import numpy as np
import pandas as pd

# Interim titles are the same division. A catchweight bout is not a division
# at all and is left to the league prior rather than pooled with anything.
_STRIP = ("interim ", "ufc ")

# Fights at which a division's own rate counts as much as the league's.
# Measured by split-half reliability on 2011-2019 in experiments/
# method_model.py: the correlation between two random halves of a division's
# fights reaches 0.5 at roughly this many bouts.
K_FINISH = 40.0

LEAGUE_FALLBACK = 0.5   # before any fight has been seen


def normalise_division(value):
    """Interim and non-interim are one division; blanks are their own bucket."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    for prefix in _STRIP:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def is_finish(method):
    """True for a finish, False for a decision, None for anything else.

    A disqualification or a no-contest is neither, and counting it as either
    would move a division's rate for a reason that has nothing to do with how
    its fights are fought.
    """
    if method is None or (isinstance(method, float) and np.isnan(method)):
        return None
    text = str(method).upper()
    if "KO" in text or "TKO" in text or "SUB" in text:
        return True
    if any(word in text for word in
           ("DEC", "UNANIMOUS", "SPLIT", "MAJORITY")):
        return False
    return None


def division_finish_prior(df, *, k=K_FINISH, strict=True, date_column="date",
                          division_column="division", method_column="method"):
    """Shrunk finish rate of each fight's division, from earlier fights only.

    Returns a float Series aligned to `df`'s index. Every value is computed
    from bouts strictly before that fight's own, so a row can never see its
    own result or any later one.

    Ties on date are resolved by taking the whole day as "earlier than the
    next day" rather than ordering within it: two fights on the same card
    cannot inform each other, and pretending one preceded the other would
    leak a few hours of the future into the other.
    """
    dates = pd.to_datetime(df[date_column], errors="coerce")
    # A row whose date will not parse has no place in the ordering, and the
    # grouping below would drop it and hand back NaN with nothing said. That
    # is how a feature quietly becomes missing for part of the data.
    unparsed = int(dates.isna().sum())
    if unparsed and strict:
        raise ValueError(
            f"{unparsed} row(s) have a date that will not parse; a prior "
            f"accumulated in date order cannot place them. Fix the dates, or "
            f"pass strict=False to accept NaN for those rows.")
    division = df[division_column].map(normalise_division)
    finished = df[method_column].map(is_finish)

    order = np.lexsort((division.to_numpy(), dates.to_numpy()))
    out = pd.Series(np.nan, index=df.index, dtype=float)

    league_n = league_f = 0.0
    per_division = {}
    # Fights of one day are settled together, after every one of them has been
    # given the state that existed before the day began.
    for _, block in pd.Series(order).groupby(dates.to_numpy()[order], sort=True):
        rows = [df.index[i] for i in block]
        league = (league_f / league_n) if league_n else LEAGUE_FALLBACK
        for row in rows:
            div = division.loc[row]
            seen_n, seen_f = per_division.get(div, (0.0, 0.0))
            out.loc[row] = (seen_f + k * league) / (seen_n + k)
        for row in rows:
            was = finished.loc[row]
            if was is None:
                continue
            div = division.loc[row]
            seen_n, seen_f = per_division.get(div, (0.0, 0.0))
            per_division[div] = (seen_n + 1.0, seen_f + (1.0 if was else 0.0))
            league_n += 1.0
            league_f += 1.0 if was else 0.0
    return out


def final_priors(df, *, k=K_FINISH, **columns):
    """Each division's prior as it stands after the last fight in `df`.

    The live counterpart of the accumulated column: a fight tomorrow gets the
    state the dataset ends in, the way career_stats.final_stats() gives a
    fighter the state they ended their last bout in. Returns
    ({division: prior}, league_rate).

    Computed by appending one dummy row per division dated after everything
    else and reading what the accumulation hands it, rather than by repeating
    the shrinkage formula here - a second copy of that arithmetic is a second
    thing to keep in step.
    """
    date_column = columns.get("date_column", "date")
    division_column = columns.get("division_column", "division")
    method_column = columns.get("method_column", "method")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    divisions = sorted({d for d in df[division_column].map(normalise_division)
                        if d})
    after = (dates.max() + pd.Timedelta(days=1)) if len(dates) else pd.Timestamp("2000-01-01")

    probe = pd.DataFrame({
        date_column: [after] * len(divisions),
        division_column: divisions,
        method_column: [None] * len(divisions),
    })
    combined = pd.concat(
        [df[[date_column, division_column, method_column]], probe],
        ignore_index=True)
    out = division_finish_prior(combined, k=k, date_column=date_column,
                                division_column=division_column,
                                method_column=method_column)
    priors = {d: float(v) for d, v in zip(divisions, out.iloc[-len(divisions):])}

    finished = df[method_column].map(is_finish).dropna()
    league = float(finished.mean()) if len(finished) else LEAGUE_FALLBACK
    return priors, league


def prior_for(red_division, blue_division, priors, league):
    """The prior for a fight between two fighters' most recent divisions.

    They usually agree. When they do not - a catchweight, or someone moving
    up - there is no one division this fight belongs to, and the league rate
    is the honest answer rather than an arbitrary pick between the two.
    """
    red = normalise_division(red_division)
    blue = normalise_division(blue_division)
    if red and red == blue and red in priors:
        return priors[red]
    return league
