"""Career-to-date fighter statistics computed from strictly-prior bouts only.

This module exists to replace eight leaking columns.

r_splm, r_str_acc, r_sapm, r_str_def, r_td_avg, r_td_avg_acc, r_td_def and
r_sub_avg (and their b_ twins) come from an upstream fighter-profile table that
publishes CAREER AVERAGES AS OF THE DATA PULL. They are joined onto every bout
a fighter ever had, so a 2014 fight is described by a striking accuracy
computed over 2014-2026 - including the fight being predicted and every fight
after it. engine/leakage.py measures the signature: 80-90% of fighters with
five or more bouts carry a single value for their entire career.

That leak inflated the walk-forward backtest to 68.6% and +16.2% over 5,943
priced bets, rising to 75% by 2024. The only honest year is 2026 (61.5%,
-3.2%), because the data ends in August 2026 and there is almost no future
career left to leak.

What this module computes instead: for every bout and each corner, the same
statistics UFCStats publishes, truncated at that bout's date. The guarantee is
structural rather than checked afterwards - a single chronological pass keeps
running totals per fighter, each row reads the accumulator BEFORE any of its
own date's bouts are folded in, so no future row can reach a past one.

MEASURED ON engine/data/UFC_with_mmr_rebuilt_dedup.csv (8,587 bouts,
2000-11-17 to 2026-08-08, 17,174 fighter-appearances, 2,617 fighter ids):

  correctness
    verify_no_lookahead      0 mismatches over all 17,174 fighter-appearances
                             x 23 columns (the exhaustive run, 77s)
    truncation invariance    0 mismatches over 5,208 rows at cutoff 2020-01-01
    UFCStats agreement       Jon Jones, whole career: SLpM 4.3834 / SApM 2.2365
                             / TD per 15 1.8918 / TD def 0.9500, against a
                             published 4.38 / 2.24 / 1.89 / 95.0%
  coverage
    89.6% of rows have red-corner prior history, 75.3% have it for both
    corners, 45.3% have three or more prior bouts on both sides
    10.4% of red corners are UFC debuts; 24.7% of rows have a debutant
    with history present: cd_td_acc is undefined on 8.6% of red corners (663
    rows, no career takedown attempt), cd_td_def on 512, cd_str_acc on 3
  runtime
    0.05s for the full 8,587-row pass

SCALE WARNING. Accuracies, defences and distribution shares are FRACTIONS in
[0, 1]: cd_str_acc reads 0.465 where the leaked r_str_acc reads 46.5. That is a
deliberate 100x departure from the columns being replaced, chosen so every
ratio in the module shares one scale. Trees will not care; a regularised linear
model will, quietly.

THE FIX WILL LOOK LIKE A REGRESSION. These features will score below the
leaked ones, because the leaked ones were reading the answer. The backtest
should fall from 68.6% / +16.2% toward the honest 2026 figure of 61.5% /
-3.2%. That drop is the module working.

This module never reads winner, winner_id or method, so it carries no outcome
information at all and cannot leak the target by a second route.
"""

import numpy as np
import pandas as pd


# --- constants -------------------------------------------------------------

# A championship-length round under the UFC Unified Rules, in force for every
# event in this file (earliest bout 2000-11-17, UFC 28, the first card fought
# under them). Validated on the data rather than assumed: under the duration
# derived below, control time exceeds fight duration on 0 of 8,587 rows (2,015
# rows violated it before the fix, and r_ctrl + b_ctrl violated it on 3,777),
# and every decision lands exactly on a round multiple - 3-round decisions max
# 900.0s, 5-round 1500.0s, 2-round 600.0s.
ROUND_SECONDS = 300.0

# SLpM and SApM are per-minute by the UFCStats definition this module
# reproduces. Confirmed by matching Jon Jones' published SLpM 4.38 and SApM
# 2.24 to two decimals from his career totals.
SECONDS_PER_MINUTE = 60.0

# Three five-minute rounds: the UFCStats normalisation for "TD Avg." and
# "Sub. Avg.". Confirmed against the published profile columns - Jon Jones
# TD/15 computed 1.89 vs published 1.89, Charles Oliveira sub/15 2.59 vs 2.6,
# Max Holloway 0.28 vs 0.3.
PER_15_MINUTES = 15.0

# The threshold separating the two meanings of match_time_sec. Derived, not
# chosen: 8,486 rows are <= 300 (every round-1 finish maxes at exactly 300.0)
# and all 101 rows above 300 fall inside ((finish_round-1)*300, finish_round*300],
# which is where and only where a cumulative reading can land. No value is
# ambiguous, because a cumulative time in round 2 or later cannot be <= 300.
MATCH_TIME_CUMULATIVE_ABOVE = ROUND_SECONDS

# Default for reliability_mask only; career_stats never applies it. Taken from
# the measured collapse of rate instability with sample size - the standard
# deviation of prior-only cd_slpm is 3.87 at one prior bout, 2.06 at two, 1.70
# at three, 1.40 at five and 1.23 beyond, against a population mean of 3.67
# (both corners pooled, 17,174 appearances). Three bouts is where the spread
# first falls below half the one-bout figure.
RELIABLE_BOUTS = 3

# Default sample for verify_no_lookahead. Chosen for runtime, not statistics:
# 400 brute-force recomputations run in about half a second, keeping the check
# inside a normal pytest run, while sample=None exhaustively checks all 17,174
# fighter-appearances for a release audit.
VERIFY_SAMPLE_DEFAULT = 400


# The emitted columns, in order, with the unit of each. Anything reading these
# downstream should read this dict rather than hardcode a list.
#
# No smoothing, shrinkage, prior or regularisation constant exists anywhere in
# this module. Every number below is a ratio of observed counts, and thin
# evidence is reported through cd_bouts, cd_minutes and the four attempt
# denominators rather than hidden by pulling the estimate toward a mean.
CAREER_COLUMNS = {
    "cd_bouts": "prior UFC bouts (count; 0 on a debut)",
    "cd_minutes": "prior UFC fight time (minutes)",
    "cd_slpm": "significant strikes landed per minute",
    "cd_str_acc": "significant striking accuracy (FRACTION in [0,1], not %)",
    "cd_sapm": "significant strikes absorbed per minute",
    "cd_str_def": "significant striking defence (FRACTION in [0,1], not %)",
    "cd_td_per15": "takedowns landed per 15 minutes",
    "cd_td_acc": "takedown accuracy (FRACTION in [0,1], not %)",
    "cd_td_def": "takedown defence (FRACTION in [0,1], not %)",
    "cd_sub_per15": "submission attempts per 15 minutes",
    "cd_kd_per15": "knockdowns per 15 minutes",
    "cd_ctrl_share": "control seconds / fight seconds (FRACTION in [0,1])",
    "cd_head_share": "share of landed significant strikes to the head [0,1]",
    "cd_body_share": "share of landed significant strikes to the body [0,1]",
    "cd_leg_share": "share of landed significant strikes to the legs [0,1]",
    "cd_dist_share": "share of landed significant strikes at distance [0,1]",
    "cd_clinch_share": "share of landed significant strikes in the clinch [0,1]",
    "cd_ground_share": "share of landed significant strikes on the ground [0,1]",
    "cd_sig_atmpted": "evidence behind cd_str_acc (attempts)",
    "cd_opp_sig_atmpted": "evidence behind cd_str_def (opponent attempts)",
    "cd_td_atmpted": "evidence behind cd_td_acc (attempts)",
    "cd_opp_td_atmpted": "evidence behind cd_td_def (opponent attempts)",
    "cd_breakdown_bouts": "bouts contributing to the six distribution shares",
}

CORNERS = ("r", "b")

# Per-corner count columns the accumulator reads. Defensive statistics are read
# off the OPPOSING corner of the same row - one row carries both fighters'
# counts, so strikes absorbed and takedowns stuffed need no join.
_COUNT_COLUMNS = (
    "sig_str_landed", "sig_str_atmpted",
    "td_landed", "td_atmpted",
    "ctrl", "kd", "sub_att",
    "head_landed", "body_landed", "leg_landed",
    "dist_landed", "clinch_landed", "ground_landed",
)

_REQUIRED_SHARED = ("date", "match_time_sec", "finish_round", "total_rounds")

# Every ratio is accumulated as a numerator/denominator PAIR over exactly the
# bouts where both operands are present. That is what lets the 145 bouts with a
# missing strike breakdown still count toward SLpM while being excluded from
# takedown accuracy: a bout is skipped only by the ratios whose operands it is
# missing, never by the others, and no ratio is ever computed from mismatched
# evidence.
_PAIRS = (
    ("slpm", "sig_str_landed", "dur"),
    ("sapm", "opp_sig_str_landed", "dur"),
    ("td15", "td_landed", "dur"),
    ("sub15", "sub_att", "dur"),
    ("kd15", "kd", "dur"),
    ("ctrl", "ctrl", "dur"),
    ("str_acc", "sig_str_landed", "sig_str_atmpted"),
    ("str_def", "opp_sig_str_landed", "opp_sig_str_atmpted"),
    ("td_acc", "td_landed", "td_atmpted"),
    ("td_def", "opp_td_landed", "opp_td_atmpted"),
)

_BREAKDOWN = ("head", "body", "leg", "dist", "clinch", "ground")

# Layout of the running-total vector held per fighter.
_STATE_FIELDS = (
    ("bouts", "sec")
    + tuple(f"{name}_{half}" for name, _, _ in _PAIRS for half in ("num", "den"))
    + tuple(f"bd_{part}" for part in _BREAKDOWN)
    + ("bd_total", "bd_bouts")
)
_STATE_INDEX = {field: i for i, field in enumerate(_STATE_FIELDS)}
_STATE_WIDTH = len(_STATE_FIELDS)


# --- public helpers --------------------------------------------------------

def career_columns(corner=None):
    """The exact column names career_stats emits, for one corner or both.

    Lets feature_inventory declare against this module without importing the
    dataset or hardcoding a list that drifts out of step with it.
    """
    if corner is None:
        return [f"{c}_{name}" for c in CORNERS for name in CAREER_COLUMNS]
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {CORNERS}, got {corner!r}")
    return [f"{corner}_{name}" for name in CAREER_COLUMNS]


def fight_elapsed_seconds(df):
    """Actual fight duration in seconds, repairing match_time_sec.

    That column carries two different meanings in the same place. On 8,486 of
    8,587 rows it is the clock WITHIN THE FINAL ROUND - every round-1 finish
    maxes at exactly 300.0 - and on 101 rows it is total elapsed time, with
    values 321-1500 every one of which lands inside its own round's cumulative
    window. Read at face value, as a per-minute rate needs to, it understates
    most fights by up to twenty minutes.

    Public and separately tested because every rate in this module divides by
    it AND the lookahead verifier shares it, so it is the one place a mistake
    could be confirmed leak-free while being wrong. Its evidence therefore has
    to come from outside the career statistics: control time now fits inside
    the fight on 8,587 of 8,587 rows (2,015 violations before the repair) and
    every decision lands exactly on its round limit.
    """
    match_time = pd.to_numeric(df["match_time_sec"], errors="coerce")
    finish_round = pd.to_numeric(df["finish_round"], errors="coerce")
    cumulative = (finish_round - 1.0) * ROUND_SECONDS + match_time
    elapsed = match_time.where(match_time > MATCH_TIME_CUMULATIVE_ABOVE,
                               cumulative)
    return elapsed.astype(float)


def duration_violations(df, elapsed=None):
    """Rows whose derived duration cannot be real, as a boolean Series.

    The > 300 rule in fight_elapsed_seconds assumes five-minute rounds. A sync
    that ever includes a different format - or a pre-2000 bout, where rounds
    ran ten and fifteen minutes - would be silently misread and produce a
    plausible wrong number rather than an error. A duration must fall in
    (0, total_rounds * 300]; 0 of 8,587 rows violate that today, and
    career_stats(strict=True) refuses to run on one that does.
    """
    if elapsed is None:
        elapsed = fight_elapsed_seconds(df)
    limit = pd.to_numeric(df["total_rounds"], errors="coerce") * ROUND_SECONDS
    return ~((elapsed > 0) & (elapsed <= limit)) & elapsed.notna()


def reliability_mask(careers, *, min_bouts=RELIABLE_BOUTS, both_corners=True):
    """Rows where prior history is deep enough for the rates to mean anything.

    A helper the caller chooses to apply. career_stats itself never blanks a
    rate on thin evidence, because discarding a real one-bout measurement is a
    different error from reporting it with its sample size attached - and the
    sample size is attached, in cd_bouts and cd_minutes.

    both_corners=True requires both fighters to qualify (45.3% of rows at the
    default of three, 75.3% at one); False requires only one of them to.
    """
    red = pd.to_numeric(careers["r_cd_bouts"], errors="coerce") >= min_bouts
    blue = pd.to_numeric(careers["b_cd_bouts"], errors="coerce") >= min_bouts
    return (red & blue) if both_corners else (red | blue)


# --- the accumulator -------------------------------------------------------

def _require_columns(df, strict):
    needed = list(_REQUIRED_SHARED)
    for corner in CORNERS:
        needed.append(f"{corner}_id")
        needed.extend(f"{corner}_{col}" for col in _COUNT_COLUMNS)
    missing = [col for col in needed if col not in df.columns]
    if missing and strict:
        # An all-NaN output column reads downstream as "no fighter has any
        # history", which is the exact failure this module exists to remove.
        raise KeyError("career_stats is missing required columns: "
                       + ", ".join(missing))
    return missing


def _corner_counts(df, corner, elapsed):
    """Every count one corner contributes to its own history, as float arrays."""
    other = "b" if corner == "r" else "r"
    counts = {"dur": elapsed.to_numpy(dtype=float)}
    for col in _COUNT_COLUMNS:
        counts[col] = pd.to_numeric(
            df.get(f"{corner}_{col}"), errors="coerce").to_numpy(dtype=float)
    # Absorbed strikes and stuffed takedowns are the OPPONENT's counts in the
    # same bout, which is why no join is needed to get defensive statistics.
    for col in ("sig_str_landed", "sig_str_atmpted", "td_landed", "td_atmpted"):
        counts[f"opp_{col}"] = pd.to_numeric(
            df.get(f"{other}_{col}"), errors="coerce").to_numpy(dtype=float)
    return counts


def _contributions(counts, n):
    """What one bout adds to a fighter's running totals: an (n, K) array."""
    contrib = np.zeros((n, _STATE_WIDTH), dtype=float)
    contrib[:, _STATE_INDEX["bouts"]] = 1.0

    dur = counts["dur"]
    contrib[:, _STATE_INDEX["sec"]] = np.where(np.isfinite(dur), dur, 0.0)

    for name, num_field, den_field in _PAIRS:
        num, den = counts[num_field], counts[den_field]
        both = np.isfinite(num) & np.isfinite(den)
        contrib[:, _STATE_INDEX[f"{name}_num"]] = np.where(both, num, 0.0)
        contrib[:, _STATE_INDEX[f"{name}_den"]] = np.where(both, den, 0.0)

    # The breakdown is all-or-nothing per bout: on 145 bouts in 2025 all six
    # parts are null for BOTH corners at once - a whole-bout stat outage - and
    # a share computed from a partial breakdown would be a ratio of two
    # different things.
    parts = [counts[f"{part}_landed"] for part in _BREAKDOWN]
    present = np.logical_and.reduce([np.isfinite(p) for p in parts])
    for part, values in zip(_BREAKDOWN, parts):
        contrib[:, _STATE_INDEX[f"bd_{part}"]] = np.where(present, values, 0.0)
    total = parts[0] + parts[1] + parts[2]  # head + body + leg
    contrib[:, _STATE_INDEX["bd_total"]] = np.where(present, total, 0.0)
    contrib[:, _STATE_INDEX["bd_bouts"]] = np.where(present, 1.0, 0.0)
    return contrib


def _safe_ratio(num, den):
    """num / den, NaN wherever the denominator is unusable.

    Never 0.0, never 1.0, never a population mean. A fighter who has never
    attempted a takedown has an UNDEFINED takedown accuracy; filling 0 would
    assert someone who attempts them and always fails.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(den.shape, np.nan, dtype=float)
    usable = den > 0  # excludes 0, negatives and NaN in one comparison
    out[usable] = num[usable] / den[usable]
    return out


def _derive(state):
    """Turn a block of running totals into the 23 published statistics."""
    col = {field: state[:, i] for field, i in _STATE_INDEX.items()}

    def per_minute(name):
        return _safe_ratio(col[f"{name}_num"],
                           col[f"{name}_den"] / SECONDS_PER_MINUTE)

    def per_15(name):
        return PER_15_MINUTES * per_minute(name)

    out = {
        "cd_bouts": col["bouts"],
        "cd_minutes": col["sec"] / SECONDS_PER_MINUTE,
        "cd_slpm": per_minute("slpm"),
        "cd_str_acc": _safe_ratio(col["str_acc_num"], col["str_acc_den"]),
        "cd_sapm": per_minute("sapm"),
        "cd_str_def": 1.0 - _safe_ratio(col["str_def_num"], col["str_def_den"]),
        "cd_td_per15": per_15("td15"),
        "cd_td_acc": _safe_ratio(col["td_acc_num"], col["td_acc_den"]),
        "cd_td_def": 1.0 - _safe_ratio(col["td_def_num"], col["td_def_den"]),
        "cd_sub_per15": per_15("sub15"),
        "cd_kd_per15": per_15("kd15"),
        "cd_ctrl_share": _safe_ratio(col["ctrl_num"], col["ctrl_den"]),
        "cd_sig_atmpted": col["str_acc_den"],
        "cd_opp_sig_atmpted": col["str_def_den"],
        "cd_td_atmpted": col["td_acc_den"],
        "cd_opp_td_atmpted": col["td_def_den"],
        "cd_breakdown_bouts": col["bd_bouts"],
    }
    # head+body+leg and distance+clinch+ground are two partitions of the same
    # landed total - verified exactly equal on all 8,442 rows carrying a
    # breakdown - so both triples divide by it.
    for part in _BREAKDOWN:
        out[f"cd_{part}_share"] = _safe_ratio(col[f"bd_{part}"], col["bd_total"])

    # A debut has no history at all, so every rate is NaN and cd_bouts is 0.
    # Those are different facts and the caller must be able to tell them apart:
    # a 0.0 SLpM asserts a fighter who has never landed a strike, which is the
    # inversion feature_spec.py exists to prevent.
    return {name: out[name] for name in CAREER_COLUMNS}


def career_stats(df, *, strict=True):
    """Point-in-time career statistics for both corners of every bout.

    For each fighter, the history is every bout of theirs whose date is
    STRICTLY BEFORE this bout's, in whichever corner they occupied, aggregated
    as a ratio of career totals (which is what UFCStats publishes, and what
    reproduces Jon Jones' 4.38 SLpM exactly) rather than a mean of per-fight
    rates.

    Ties on date are mutually blind: two bouts on the same card see nothing of
    each other. That is the only defensible rule here - the data has no
    intra-day timestamp and fight_id is a content hash carrying no chronology -
    and it costs nothing today, because no fighter in the file has two bouts on
    one date. Should a same-day tournament ever appear, the rule degrades to
    understating history rather than to a lookahead.

    Pure: no file I/O, no mutation of df, no dependence on row order or index
    type. Returns a DataFrame on df.index with the 46 r_/b_ prefixed columns.
    """
    missing = _require_columns(df, strict)
    n = len(df)
    index = df.index

    if missing:
        empty = {name: np.full(n, np.nan) for name in CAREER_COLUMNS}
        return pd.DataFrame(
            {f"{c}_{k}": v for c in CORNERS for k, v in empty.items()},
            index=index)

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        # There is no safe fallback: a row that cannot be placed in time cannot
        # be made point-in-time safe, in strict mode or out of it.
        raise ValueError(
            f"{int(dates.isna().sum())} rows have an unparseable date; "
            "career statistics cannot be ordered without one")

    elapsed = fight_elapsed_seconds(df)
    bad = duration_violations(df, elapsed)
    if bad.any():
        if strict:
            raise ValueError(
                f"{int(bad.sum())} rows have a fight duration outside "
                "(0, total_rounds * 300]; the match_time_sec repair assumes "
                f"five-minute rounds. First offending index: {index[bad][0]!r}")
        # Out of strict mode an impossible duration is simply unknown, so those
        # bouts drop out of every rate denominator instead of scaling it wrong.
        elapsed = elapsed.mask(bad)

    fighter_ids = np.concatenate(
        [df[f"{c}_id"].to_numpy(dtype=object) for c in CORNERS])
    contrib = np.vstack(
        [_contributions(_corner_counts(df, c, elapsed), n) for c in CORNERS])
    when = np.concatenate([dates.to_numpy(dtype="datetime64[ns]")] * len(CORNERS))

    # One chronological pass. Each appearance reads the accumulator before any
    # of its own date's bouts are folded in, which is the whole leak-freeness
    # guarantee: an emitted value is a function of totals that were complete
    # before the current bout existed in them.
    order = np.argsort(when, kind="stable")
    state = np.zeros((len(order), _STATE_WIDTH), dtype=float)
    totals = {}
    position = 0
    while position < len(order):
        # Buffer a whole date and flush it only once every row on that date has
        # read its state. This is what makes same-card bouts mutually blind by
        # construction rather than by luck (up to 25 bouts share a date).
        end = position
        day = when[order[position]]
        while end < len(order) and when[order[end]] == day:
            end += 1
        block = order[position:end]
        for appearance in block:
            running = totals.get(fighter_ids[appearance])
            if running is not None:
                state[appearance] = running
        for appearance in block:
            key = fighter_ids[appearance]
            running = totals.get(key)
            if running is None:
                totals[key] = contrib[appearance].copy()
            else:
                running += contrib[appearance]
        position = end

    out = {}
    for offset, corner in enumerate(CORNERS):
        block = state[offset * n:(offset + 1) * n]
        for name, values in _derive(block).items():
            out[f"{corner}_{name}"] = values
    return pd.DataFrame(out, index=index)


# --- the proofs ------------------------------------------------------------

def _appearances(df, elapsed):
    """One row per fighter-appearance, for the brute-force recomputation."""
    frames = []
    for corner in CORNERS:
        counts = _corner_counts(df, corner, elapsed)
        frame = pd.DataFrame(counts)
        frame["fighter_id"] = df[f"{corner}_id"].to_numpy(dtype=object)
        frame["date"] = pd.to_datetime(df["date"]).to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _naive_stats(history):
    """Every statistic recomputed by plain filter-and-aggregate.

    Deliberately a different algorithm from the accumulator - and written out
    longhand rather than sharing _derive - so that a bug in the running totals
    cannot hide behind the same arithmetic on both sides of the comparison.
    """
    def paired(num_field, den_field):
        num = history[num_field]
        den = history[den_field]
        both = num.notna() & den.notna()
        return float(num[both].sum()), float(den[both].sum())

    def divide(num, den):
        return num / den if den > 0 else float("nan")

    seconds = float(history["dur"].fillna(0.0).sum())
    stats = {"cd_bouts": float(len(history)),
             "cd_minutes": seconds / SECONDS_PER_MINUTE}

    rates = {"cd_slpm": "sig_str_landed", "cd_sapm": "opp_sig_str_landed"}
    for out_name, field in rates.items():
        num, den = paired(field, "dur")
        stats[out_name] = divide(num, den / SECONDS_PER_MINUTE)
    for out_name, field in {"cd_td_per15": "td_landed",
                            "cd_sub_per15": "sub_att",
                            "cd_kd_per15": "kd"}.items():
        num, den = paired(field, "dur")
        stats[out_name] = PER_15_MINUTES * divide(num, den / SECONDS_PER_MINUTE)

    ctrl_num, ctrl_den = paired("ctrl", "dur")
    stats["cd_ctrl_share"] = divide(ctrl_num, ctrl_den)

    acc_num, acc_den = paired("sig_str_landed", "sig_str_atmpted")
    stats["cd_str_acc"] = divide(acc_num, acc_den)
    stats["cd_sig_atmpted"] = acc_den

    def_num, def_den = paired("opp_sig_str_landed", "opp_sig_str_atmpted")
    stats["cd_str_def"] = 1.0 - divide(def_num, def_den)
    stats["cd_opp_sig_atmpted"] = def_den

    tda_num, tda_den = paired("td_landed", "td_atmpted")
    stats["cd_td_acc"] = divide(tda_num, tda_den)
    stats["cd_td_atmpted"] = tda_den

    tdd_num, tdd_den = paired("opp_td_landed", "opp_td_atmpted")
    stats["cd_td_def"] = 1.0 - divide(tdd_num, tdd_den)
    stats["cd_opp_td_atmpted"] = tdd_den

    breakdown = history[[f"{part}_landed" for part in _BREAKDOWN]].dropna()
    stats["cd_breakdown_bouts"] = float(len(breakdown))
    total = float(breakdown[["head_landed", "body_landed",
                             "leg_landed"]].to_numpy().sum())
    for part in _BREAKDOWN:
        stats[f"cd_{part}_share"] = divide(
            float(breakdown[f"{part}_landed"].sum()), total)
    return stats


def _equal(left, right, rtol):
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(np.isclose(left, right, rtol=rtol, atol=0.0))


def verify_no_lookahead(df, careers=None, *, sample=VERIFY_SAMPLE_DEFAULT,
                        seed=0, rtol=1e-9):
    """Prove every value at fight N comes from that fighter's first N-1 bouts.

    For each sampled (row, corner) this filters that fighter's appearances to
    the ones dated strictly before the bout and aggregates them from scratch,
    then compares all 23 statistics against what the accumulator emitted. The
    two paths share only fight_elapsed_seconds - which is why that function
    carries its own evidence.

    sample=None checks all 17,174 fighter-appearances; the 400 default runs in
    about half a second, which keeps it inside a normal pytest run.
    """
    if careers is None:
        careers = career_stats(df)
    elapsed = fight_elapsed_seconds(df)
    appearances = _appearances(df, elapsed)
    by_fighter = {fid: block for fid, block
                  in appearances.groupby("fighter_id", sort=False)}
    dates = pd.to_datetime(df["date"])

    targets = [(pos, corner) for corner in CORNERS for pos in range(len(df))]
    if sample is not None and sample < len(targets):
        picked = np.random.default_rng(seed).choice(
            len(targets), size=sample, replace=False)
        targets = [targets[i] for i in picked]

    failures = []
    for pos, corner in targets:
        label = df.index[pos]
        fighter = df[f"{corner}_id"].iloc[pos]
        block = by_fighter.get(fighter)
        history = block[block["date"] < dates.iloc[pos]]
        expected = _naive_stats(history)
        for name in CAREER_COLUMNS:
            actual = careers[f"{corner}_{name}"].iloc[pos]
            if not _equal(expected[name], actual, rtol):
                failures.append({"index": label, "corner": corner,
                                 "column": name, "expected": expected[name],
                                 "actual": actual})
    return {"checked": len(targets), "mismatches": len(failures),
            "columns_checked": len(CAREER_COLUMNS), "failures": failures[:20]}


def verify_truncation_invariance(df, cutoff, careers=None):
    """Prove that data after `cutoff` changes nothing before it.

    The cheaper and stronger proof, and the one that directly contradicts the
    defect this module exists to fix: recompute on only the bouts before the
    cutoff and every value for those rows must be bit-identical to the
    full-data run. A joined career total fails this the moment a fighter has
    one later bout; a point-in-time statistic cannot fail it at all.
    """
    if careers is None:
        careers = career_stats(df)
    dates = pd.to_datetime(df["date"])
    before = dates < pd.Timestamp(cutoff)
    truncated = career_stats(df[before])

    failures = []
    for column in career_columns():
        full_values = careers.loc[before, column]
        cut_values = truncated[column]
        differs = ~((full_values.values == cut_values.values)
                    | (pd.isna(full_values.values) & pd.isna(cut_values.values)))
        for label in full_values.index[differs]:
            failures.append({"index": label, "column": column,
                             "full": full_values.loc[label],
                             "truncated": cut_values.loc[label]})
    return {"rows": int(before.sum()), "mismatches": len(failures),
            "columns_checked": len(career_columns()),
            "failures": failures[:20]}
