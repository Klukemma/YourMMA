"""A live model row is a training row that has not been played yet.

These run the real training pipeline (feature_frame.build, about 12 s a
pass) over the real archive - not the 10-minute engine import - and check
the properties that make a pending row equal to the row training would
build for the same fight.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import feature_frame
import pending_rows

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


@pytest.fixture(scope="module")
def raw():
    if not ARCHIVE.exists():
        pytest.skip("archive not present")
    return pd.read_csv(ARCHIVE, low_memory=False)


@pytest.fixture(scope="module")
def full(raw):
    return feature_frame.build(raw, verbose=False)


@pytest.fixture(scope="module")
def last_event(raw):
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = days.max()
    return when, raw[days < when], raw[days == when]


def _fights(event, when):
    return [dict(red=r.r_name, blue=r.b_name, red_id=r.r_id, blue_id=r.b_id,
                 date=when, division=r.division, total_rounds=r.total_rounds,
                 is_title=bool(r.title_fight)) for r in event.itertuples()]


@pytest.fixture(scope="module")
def live(full, last_event):
    when, before, event = last_event
    pend = pending_rows.build_pending(before, _fights(event, when))
    built = feature_frame.build(before, pending=pend,
                                centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    return built, pend


def test_frozen_centres_reproduce_training_exactly(raw, full):
    again = feature_frame.build(raw, centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    pd.testing.assert_frame_equal(again["X"], full["X"], check_exact=True)


def test_archive_rows_are_untouched_by_pending_rows(full, live, last_event):
    when = last_event[0]
    built, _ = live
    n = built["N_ARCHIVE"]
    part = built["X"].iloc[:n].set_index(
        built["ufc"]["fight_id"].iloc[:n].astype(str))
    train = full["X"].set_index(full["ufc"]["fight_id"].astype(str))
    train = train.loc[part.index]
    assert (full["ufc"]["date"] < when).sum() == n
    assert np.array_equal(part.to_numpy(), train.to_numpy())


def test_a_pending_row_equals_the_row_training_built(full, live, last_event):
    """Every fight of the last archived event, rebuilt from the bouts before
    it plus a pending row, against its training row - all features.

    A debutant has no earlier row to take a height or reach from, so their
    physical columns can differ; the live path refuses debutants anyway.
    Everything else must match."""
    when, before, event = last_event
    built, _ = live
    cols = full["feature_cols"]
    n = built["N_ARCHIVE"]
    known = set(before["r_id"]) | set(before["b_id"])
    fu, fx = full["ufc"], full["X"]
    checked = 0
    for i in range(n, len(built["ufc"])):
        r_id, b_id = built["ufc"].loc[i, ["r_id", "b_id"]]
        if r_id not in known or b_id not in known:
            continue
        j = fu.index[(fu["date"] == when) & (fu["r_id"] == r_id)
                     & (fu["b_id"] == b_id)][0]
        mine = built["X"].loc[i, cols].to_numpy(float)
        theirs = fx.loc[j, cols].to_numpy(float)
        bad = [c for c, a, b in zip(cols, mine, theirs)
               if not np.isclose(a, b, rtol=1e-9, atol=1e-12)]
        assert bad == [], f"{fu.loc[j, 'r_name']} vs {fu.loc[j, 'b_name']}: {bad}"
        checked += 1
    assert checked >= 5


def test_a_bout_reads_only_the_contract_columns_of_its_own_row(raw, full,
                                                               last_event):
    """Blank every non-CONTRACT column of the last event (results, methods,
    in-bout counts, ids of the event) and its features must not move. A new
    feature that reads the current bout's own result fails here."""
    when = last_event[0]
    days = pd.to_datetime(raw["date"], errors="coerce")
    blanked = raw.copy()
    on = (days == when).to_numpy()
    keep = set(pending_rows.CONTRACT) | {"fight_id", "event_id", "event_name"}
    for col in blanked.columns:
        if col not in keep:
            blanked.loc[on, col] = np.nan
    again = feature_frame.build(blanked, centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    mask = (full["ufc"]["date"] == when).to_numpy()
    assert np.allclose(again["X"][mask].to_numpy(), full["X"][mask].to_numpy(),
                       rtol=0, atol=0, equal_nan=True)


def test_pending_rows_carry_no_outcome(live):
    _, pend = live
    for col in ("winner", "method", "finish_round", "match_time_sec"):
        if col in pend.columns:
            assert pend[col].isna().all(), col


def test_pending_rows_must_come_after_the_archive(raw, last_event):
    when, before, event = last_event
    early = _fights(event, when)[:1]
    early[0]["date"] = pd.Timestamp("2010-01-01")
    pend = pending_rows.build_pending(before, early)
    with pytest.raises(ValueError, match="after every archive bout"):
        feature_frame.build(before, pending=pend, verbose=False)


def test_a_fighter_twice_among_pending_rows_is_refused(last_event):
    when, before, event = last_event
    f = _fights(event, when)[:1]
    pend = pending_rows.build_pending(before, f + f)
    with pytest.raises(ValueError, match="twice"):
        feature_frame.build(before, pending=pend, verbose=False)


def test_division_is_inferred_only_when_both_fighters_agree(last_event):
    when, before, event = last_event
    f = _fights(event, when)[:1]
    f[0]["division"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pend = pending_rows.build_pending(before, f)
    divs = pending_rows.last_division(before)
    r_div, b_div = divs.get(f[0]["red_id"]), divs.get(f[0]["blue_id"])
    got = pend["division"].iloc[0]
    assert got == r_div if r_div == b_div else pd.isna(got)


def test_dob_is_normalised_to_the_archive_format():
    assert pending_rows._normal_dob("Jan 05, 1990") == "1990/01/05"
    assert pending_rows._normal_dob("1990/01/05") == "1990/01/05"
    assert pd.isna(pending_rows._normal_dob(None))
