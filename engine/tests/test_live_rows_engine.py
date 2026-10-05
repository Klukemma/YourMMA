"""The live-row logic predict_card runs for its card, without the engine.

live_rows.LiveRows holds what the engine gives it - the archive, the
training matrix by fight id, the frozen centres - so every way a card fight
can go wrong is driven here over the archive cut at the end of 2006 (a
pipeline pass in about a second) instead of through the ten-minute import:
a corner with no bout before the date, the same fighter in both corners, a
name two fighters share, a fight given no division whose fighters last
fought in different ones, a fighter in two fights, the cache keying the
card prefetch and the later prediction share, and a pipeline that stops
reproducing the training rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import feature_frame
import pending_rows
from live_rows import LiveBuildDrift, LiveRows, NoLiveRow

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
CUT = pd.Timestamp("2007-01-01")


@pytest.fixture(scope="module")
def raw():
    if not ARCHIVE.exists():
        pytest.skip("archive not present")
    return pd.read_csv(ARCHIVE, low_memory=False)


@pytest.fixture(scope="module")
def small(raw):
    """The archive through 2006, built as training would build it."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    cut = raw[days < CUT].reset_index(drop=True)
    return cut, feature_frame.build(cut, verbose=False)


def _live(cut, built, build=feature_frame.build):
    return LiveRows(cut, built["X"].set_index(built["ufc"]["fight_id"].astype(str)),
                    built["INTERACTION_CENTRES"], build=build)


@pytest.fixture
def live(small):
    return _live(*small)


def _event(cut, when):
    """The archive event on `when` as FIGHT_CARD-shaped tuples with a
    division context, and the archive before it."""
    days = pd.to_datetime(cut["date"], errors="coerce")
    event = cut[days == when]
    fights = [(r.r_name, r.b_name, r.total_rounds == 5, bool(r.title_fight),
               {"division": r.division}) for r in event.itertuples()]
    return fights, cut[days < when]


@pytest.fixture(scope="module")
def last(small):
    cut, _ = small
    when = pd.to_datetime(cut["date"], errors="coerce").max()
    fights, before = _event(cut, when)
    return when, fights, before


def _known(before, fights):
    ids = pending_rows.fighter_ids(before)
    return [f for f in fights if all(pending_rows.norm_name(n) in ids
                                     for n in f[:2])]


def _reference(before, when, fights, centres):
    """{(red, blue): row} straight from pending_rows and feature_frame, the
    fights built together in this order - what LiveRows must serve for each.
    (Not the training rows of the 2006 cut: the historical rating rebuild
    differs from ratings.extend in the third decimal there, and the
    class-limit weight rule from 2006 listed weights; the last real event
    is proven equal to training in test_live_rows.py.)"""
    spec = [dict(red=r, blue=b, date=when, is_5rnd=is_5rnd, is_title=is_title,
                 division=(context or {}).get("division"))
            for r, b, is_5rnd, is_title, context in fights]
    pend = pending_rows.build_pending(before, spec)
    built = feature_frame.build(before, pending=pend, centres=centres, verbose=False)
    n = built["N_ARCHIVE"]
    return {(r, b): built["X"].iloc[n + i] for i, (r, b, *_) in enumerate(fights)}


def _same(mine, theirs):
    return np.allclose(mine.to_numpy(float), theirs.to_numpy(float),
                       rtol=1e-9, atol=1e-12, equal_nan=True)


# --- the cache: one pass for a card, hit again by the prediction ------------

def test_a_card_is_one_pass_and_every_later_call_hits_it(live, last):
    when, fights, before = last
    assert len(fights) >= 5
    live.prepare(fights, when)
    assert live.passes == 1
    assert len(live.rows) == len(fights) and not live.refused
    for red, blue, is_5rnd, is_title, context in fights:
        for date in (when, str(when.date()), when.to_pydatetime()):
            row = live.row(red, blue, date, is_5rnd, is_title, context)
            assert row.shape == (1, len(live.train.columns))
    assert live.passes == 1


def test_rows_come_back_in_card_order(small, live, last):
    """The card prepared reversed, every row checked against the pipeline's
    row for the same fight built in card order: a positional slip would
    hand one fight another's row."""
    cut, built = small
    when, fights, before = last
    fights = _known(before, fights)
    assert len(fights) >= 3
    reference = _reference(before, when, fights, live.centres)
    live.prepare(fights[::-1], when)
    assert live.passes == 1
    for red, blue, is_5rnd, is_title, context in fights:
        mine = live.row(red, blue, when, is_5rnd, is_title, context).iloc[0]
        assert _same(mine, reference[(red, blue)]), (red, blue)
    rows = [live.row(r, b, when, f, t, c).iloc[0].to_numpy(float)
            for r, b, f, t, c in fights]
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            assert not np.allclose(a, b, equal_nan=True), "two fights, one row"


def test_the_division_spelling_does_not_split_the_cache(live, last):
    when, fights, before = last
    red, blue, is_5rnd, is_title, context = _known(before, fights)[0]
    division = context["division"]
    live.prepare([(red, blue, is_5rnd, is_title, {"division": division.title()})], when)
    for spelling in (division, division.upper(), f" {division} ",
                     division.replace("'", "’")):
        live.row(red, blue, when, is_5rnd, is_title, {"division": spelling})
        live.row(red, blue, when, is_5rnd, is_title, {"weight_class": spelling})
    assert live.passes == 1
    assert LiveRows.key("A", "B", when, 0, 0, "Women’s Bantamweight") == \
        LiveRows.key("A", "B", str(when.date()), False, False, "women's bantamweight")


def test_event_date_none_keys_prepare_and_row_alike(small):
    cut, built = small
    live = _live(cut, built)
    when = pd.to_datetime(cut["date"], errors="coerce").max()
    fights, before = _event(cut, when)
    red, blue, is_5rnd, is_title, context = _known(before, fights)[0]
    live.prepare([(red, blue, is_5rnd, is_title, context)], None)
    live.row(red, blue, None, is_5rnd, is_title, context)
    assert live.passes == 1
    assert LiveRows.key(red, blue, None, 0, 0, None) == \
        LiveRows.key(red, blue, pd.Timestamp.today().normalize(), 0, 0, None)


def test_a_fighter_in_two_fights_takes_two_passes(live, last):
    when, fights, before = last
    (a, b, _, _, ab), (c, d, _, _, cd) = _known(before, fights)[:2]
    live.prepare([(a, b, 0, 0, ab), (a, d, 0, 0, ab), (c, d, 0, 0, cd)], when)
    assert live.passes == 2
    assert len(live.rows) == 3 and not live.refused


# --- refusals: reported, never raised out of the card -------------------------

def test_a_corner_with_no_bout_before_the_date_is_refused_before_any_build(small, live, last):
    """An unknown name, and a debutant the engine knows from the archive
    (his only bout is the event itself, so fighter_stats holds him) but who
    has no bout before the date: both refused with the reason, no pass run."""
    when, fights, before = last
    red = _known(before, fights)[0][0]
    with pytest.raises(NoLiveRow, match="no bout before 2006-12-30 for 'Nobody Here'"):
        live.row(red, "Nobody Here", when)
    cut, _ = small
    days = pd.to_datetime(cut["date"], errors="coerce")
    for day in sorted(days.dropna().unique(), reverse=True):
        event, earlier = _event(cut, day)
        ids = pending_rows.fighter_ids(earlier)
        debutants = [n for f in event for n in f[:2]
                     if pending_rows.norm_name(n) not in ids]
        if debutants:
            break
    veteran = next(n for f in event for n in f[:2] if pending_rows.norm_name(n) in ids)
    with pytest.raises(NoLiveRow, match=f"no bout before {pd.Timestamp(day).date()} for {debutants[0]!r}"):
        live.row(veteran, debutants[0], day)
    assert live.passes == 0 and not live.rows


def test_the_same_fighter_in_both_corners_is_refused_before_any_build(live, last):
    when, fights, before = last
    red = _known(before, fights)[0][0]
    with pytest.raises(NoLiveRow, match="same fighter"):
        live.row(red, red, when)
    with pytest.raises(NoLiveRow, match="same fighter"):
        live.row(red, red.upper(), when)
    assert live.passes == 0


def test_a_refused_fight_does_not_stop_the_rest_of_the_card(live, last):
    when, fights, before = last
    known = _known(before, fights)
    red = known[0][0]
    live.prepare(known + [(red, "Nobody Here"), (red, red)], when)
    assert live.passes == 1
    assert len(live.rows) == len(known)
    assert len(live.refused) == 2


def test_a_name_two_fighters_share_is_refused_unless_the_division_tells(small):
    """A fighter's first bout filed under a second id makes his name two
    fighters: the division settles it when it can, and the refusal says
    so when it cannot."""
    cut, _ = small
    when = pd.to_datetime(cut["date"], errors="coerce").max()
    fights, before = _event(cut, when)
    red, blue, is_5rnd, is_title, context = _known(before, fights)[0]
    twin = cut.copy()
    days = pd.to_datetime(twin["date"], errors="coerce")
    mine = (twin["r_name"] == red) | (twin["b_name"] == red)
    assert mine.sum() >= 3, "pick a fighter with history"
    first = days[mine].idxmin()
    corner = "r" if twin.loc[first, "r_name"] == red else "b"
    twin.loc[first, f"{corner}_id"] = "ffffffffffffffff"
    twin.loc[first, "division"] = "catch weight"
    built = feature_frame.build(twin, verbose=False)
    live = _live(twin, built)
    with pytest.raises(NoLiveRow, match="two fighters"):
        live.row(red, blue, when, is_5rnd, is_title, None)
    with pytest.raises(NoLiveRow, match="does not tell them apart"):
        live.row(red, blue, when, is_5rnd, is_title, {"division": "super heavyweight"})
    assert live.passes == 0
    row = live.row(red, blue, when, is_5rnd, is_title, context)
    assert live.passes == 1
    before = twin[days < when]
    reference = _reference(before, when, [(red, blue, is_5rnd, is_title, context)],
                           live.centres)
    assert _same(row.iloc[0], reference[(red, blue)])
    assert pending_rows.resolve_fighter(
        red, context["division"], pending_rows.fighter_candidates(before),
        pending_rows.division_history(before)) != "ffffffffffffffff"


# --- the archive part of a live build must be the training rows --------------

def _drifting(build):
    """feature_frame.build with one archive cell moved."""
    def drift(raw, **kw):
        built = build(raw, **kw)
        built["X"] = built["X"].copy()
        built["X"].iloc[0, built["X"].columns.get_loc("mu_diff")] += 1e-9
        return built
    return drift


def test_check_append_invariance_names_the_column(small, live, last):
    cut, built = small
    when, fights, before = last
    red, blue, *_ = _known(before, fights)[0]
    pend = pending_rows.build_pending(before, [dict(red=red, blue=blue, date=when)])
    good = feature_frame.build(before, pending=pend, centres=live.centres, verbose=False)
    live.check_append_invariance(good, good["N_ARCHIVE"])
    bad = _drifting(feature_frame.build)(before, pending=pend, centres=live.centres,
                                         verbose=False)
    with pytest.raises(LiveBuildDrift, match="1 columns \\(mu_diff\\)"):
        live.check_append_invariance(bad, bad["N_ARCHIVE"])


def test_a_pass_whose_archive_rows_drift_from_training_is_refused(small, last, capsys):
    cut, built = small
    when, fights, before = last
    live = _live(cut, built, build=_drifting(feature_frame.build))
    known = _known(before, fights)
    live.prepare(known, when)
    assert live.passes == 1 and not live.rows
    assert len(live.refused) == len(known)
    for red, blue, is_5rnd, is_title, context in known:
        with pytest.raises(NoLiveRow, match="differs from training.*mu_diff"):
            live.row(red, blue, when, is_5rnd, is_title, context)
    assert live.passes == 1
    assert "WARNING: live build differs from training" in capsys.readouterr().out


def test_a_pass_that_hands_back_rows_out_of_order_is_refused(small, last, capsys):
    """Rows are read back by position, so a pipeline that reordered the
    pending rows would hand one fight another's row: every row of that
    pass is refused."""
    cut, built = small
    when, fights, before = last

    def reordered(raw, **kw):
        out = feature_frame.build(raw, **kw)
        n = out["N_ARCHIVE"]
        ufc = out["ufc"].copy()
        col = ufc.columns.get_loc("fight_id")
        ufc.iloc[n:, col] = ufc["fight_id"].iloc[n:].tolist()[::-1]
        out["ufc"] = ufc
        return out

    live = _live(cut, built, build=reordered)
    known = _known(before, fights)
    live.prepare(known, when)
    assert live.passes == 1 and not live.rows
    red, blue, is_5rnd, is_title, context = known[0]
    with pytest.raises(NoLiveRow, match="reordered"):
        live.row(red, blue, when, is_5rnd, is_title, context)
    assert "WARNING: pending rows came back reordered" in capsys.readouterr().out


# --- the date ---------------------------------------------------------------------

def test_an_event_before_the_archive_end_sees_only_earlier_bouts(small):
    """Replaying a past event: the row is built from the bouts before that
    date, and a fighter whose bouts all come later is refused."""
    cut, built = small
    days = pd.to_datetime(cut["date"], errors="coerce")
    when = sorted(days.dropna().unique())[-6]
    fights, before = _event(cut, when)
    live = _live(cut, built)
    known = _known(before, fights)
    assert known
    red, blue, is_5rnd, is_title, context = known[0]
    row = live.row(red, blue, when, is_5rnd, is_title, context)
    reference = _reference(before, when, [known[0]], live.centres)
    assert _same(row.iloc[0], reference[(red, blue)])
    # built from the bouts before the date: a later archive row of either
    # fighter cannot have been read, because the cut never held it
    assert len(live.raw[live.dates < when]) == len(before)
    later = cut[days > when]
    newcomer = next(n for n in pd.concat([later["r_name"], later["b_name"]])
                    if pending_rows.norm_name(n) not in pending_rows.fighter_ids(before))
    with pytest.raises(NoLiveRow, match=f"no bout before {pd.Timestamp(when).date()}"):
        live.row(red, newcomer, when)


def _by_last_division(before, fights):
    """Pairs of known fighters on the card: (a, b) with the same last
    division, and (a, b) with different ones."""
    known = _known(before, fights)
    divisions = pending_rows.last_division(before)
    ids = pending_rows.fighter_ids(before)
    last = lambda n: divisions.get(ids[pending_rows.norm_name(n)])
    names = list(dict.fromkeys(n for f in known for n in f[:2]))
    agree = next((a, b) for a in names for b in names if a != b and last(a) == last(b))
    differ = next((a, b) for a in names for b in names if a != b and last(a) != last(b))
    return agree, differ, last


def test_a_missing_division_the_fighters_disagree_on_is_refused_before_any_build(small, last):
    """No training row has an empty division, so a fight given none whose
    fighters last fought in different classes has no training-shaped row:
    refused with both divisions in the reason, no pass run - and built
    once the card says which class it is."""
    cut, built = small
    when, fights, before = last
    _, (a, b), last_of = _by_last_division(before, fights)
    live = _live(cut, built)
    with pytest.raises(NoLiveRow, match=(
            f"no division given and their last bouts differ "
            f"\\({last_of(a)} / {last_of(b)}\\) - give the division")):
        live.row(a, b, when)
    assert live.passes == 0 and not live.rows
    assert LiveRows.key(a, b, when, False, False, None) in live.refused
    live.row(a, b, when, context={"division": last_of(a)})
    assert live.passes == 1


def test_a_missing_division_is_the_one_both_fighters_last_fought_in(small, last):
    cut, built = small
    when, fights, before = last
    (a, b), _, last_of = _by_last_division(before, fights)
    live = _live(cut, built)
    row = live.row(a, b, when)
    assert live.passes == 1 and not live.refused
    reference = _reference(before, when, [(a, b, False, False,
                                           {"division": last_of(a)})], live.centres)
    assert _same(row.iloc[0], reference[(a, b)])
    assert LiveRows.key(a, b, when, False, False, None) in live.rows


# --- the card prefetch --------------------------------------------------------------

def test_prepare_card_builds_what_the_check_accepts_as_it_resolves_it(live, last):
    """predict_card's prefetch: names go through the engine's resolver, a
    fight it refuses is left out, and the prediction's later call for the
    resolved spelling hits the same entry."""
    when, fights, before = last
    known = _known(before, fights)
    spelled = {n.lower(): n for f in known for n in f[:2]}

    def check(red, blue):
        if red.lower() not in spelled or blue.lower() not in spelled:
            return {}, [f"UNKNOWN: {red} / {blue}"], []
        return {"red": (spelled[red.lower()], {}), "blue": (spelled[blue.lower()], {})}, [], []

    card = [(r.lower(), b.lower(), is_5rnd, is_title) for r, b, is_5rnd, is_title, _ in known]
    card.append(("nobody here", known[0][1].lower()))
    contexts = {i: f[4] for i, f in enumerate(known)}
    contexts[0] = dict(contexts[0], red_home=True, cage_size="small")
    specs = live.prepare_card(card, when, contexts, check=check)
    assert len(specs) == len(known) and live.passes == 1
    assert specs[0]["division"] == known[0][4]["division"]
    for red, blue, is_5rnd, is_title, context in known:
        live.row(red, blue, when, is_5rnd, is_title, {"division": context["division"]})
    assert live.passes == 1
    assert not live.refused
