"""A live model row is a training row that has not been played yet.

These run the real training pipeline (feature_frame.build, about 12 s a
pass) over the real archive - not the 10-minute engine import - and check
the properties that make a pending row equal to the row training would
build for the same fight. The fights are given the way the live path gives
them: names, a date, a division, is_5rnd and is_title - never ids.
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
    """The event's fights in the shape the live path sends: names, date,
    division, rounds and title - no ids."""
    return [dict(red=r.r_name, blue=r.b_name, date=when, division=r.division,
                 is_5rnd=r.total_rounds == 5, is_title=bool(r.title_fight))
            for r in event.itertuples()]


def _buildable(before, fights):
    """The fights whose corners both have a bout before the event; the
    others are debutants, which the live path refuses."""
    candidates = pending_rows.fighter_candidates(before)
    history = pending_rows.division_history(before)
    out = []
    for f in fights:
        try:
            for who in ("red", "blue"):
                pending_rows.resolve_fighter(f[who], f["division"], candidates, history)
        except KeyError:
            continue
        out.append(f)
    return out


@pytest.fixture(scope="module")
def live(full, last_event):
    """The last event rebuilt live - the card REVERSED, so the positional
    contract prepare() relies on (rows come back in the order the fights
    went in) is what this fixture proves."""
    when, before, event = last_event
    fights = _buildable(before, _fights(event, when))[::-1]
    pend = pending_rows.build_pending(before, fights)
    built = feature_frame.build(before, pending=pend,
                                centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    n = built["N_ARCHIVE"]
    assert built["ufc"]["fight_id"].iloc[n:].tolist() == pend["fight_id"].tolist()
    for i, f in enumerate(fights):
        row = built["ufc"].iloc[n + i]
        assert (row["r_name"], row["b_name"]) == (f["red"], f["blue"])
        assert (pend.loc[i, "r_name"], pend.loc[i, "b_name"]) == (f["red"], f["blue"])
    return built, pend, fights


def test_frozen_centres_reproduce_training_exactly(raw, full):
    again = feature_frame.build(raw, centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    pd.testing.assert_frame_equal(again["X"], full["X"], check_exact=True)


def test_archive_rows_are_untouched_by_pending_rows(full, live, last_event):
    when = last_event[0]
    built, _, _ = live
    n = built["N_ARCHIVE"]
    part = built["X"].iloc[:n].set_index(
        built["ufc"]["fight_id"].iloc[:n].astype(str))
    train = full["X"].set_index(full["ufc"]["fight_id"].astype(str))
    train = train.loc[part.index]
    assert (full["ufc"]["date"] < when).sum() == n
    assert np.array_equal(part.to_numpy(), train.to_numpy())


def test_a_pending_row_equals_the_row_training_built(full, live, last_event):
    """Every fight of the last archived event, rebuilt by NAME from the
    bouts before it plus a pending row, against its training row - all
    features, and the ids the archive gave the bout."""
    when, before, event = last_event
    built, pend, fights = live
    cols = full["feature_cols"]
    n = built["N_ARCHIVE"]
    fu, fx = full["ufc"], full["X"]
    checked = 0
    for i, f in enumerate(fights):
        r_id, b_id = built["ufc"].loc[n + i, ["r_id", "b_id"]]
        j = fu.index[(fu["date"] == when) & (fu["r_name"] == f["red"])
                     & (fu["b_name"] == f["blue"])][0]
        assert (fu.loc[j, "r_id"], fu.loc[j, "b_id"]) == (r_id, b_id)
        mine = built["X"].loc[n + i, cols].to_numpy(float)
        theirs = fx.loc[j, cols].to_numpy(float)
        bad = [c for c, a, b in zip(cols, mine, theirs)
               if not np.isclose(a, b, rtol=1e-9, atol=1e-12)]
        assert bad == [], f"{f['red']} vs {f['blue']}: {bad}"
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
    _, pend, _ = live
    for col in ("winner", "method", "finish_round", "match_time_sec"):
        if col in pend.columns:
            assert pend[col].isna().all(), col


def test_pending_rows_must_come_after_the_archive(raw, last_event):
    when, before, event = last_event
    early = _buildable(before, _fights(event, when))[:1]
    early[0]["date"] = pd.Timestamp("2010-01-01")
    pend = pending_rows.build_pending(before, early)
    with pytest.raises(ValueError, match="after every archive bout"):
        feature_frame.build(before, pending=pend, verbose=False)


def test_a_fighter_twice_among_pending_rows_is_refused(last_event):
    when, before, event = last_event
    f = _buildable(before, _fights(event, when))[:1]
    pend = pending_rows.build_pending(before, f + f)
    with pytest.raises(ValueError, match="twice"):
        feature_frame.build(before, pending=pend, verbose=False)


def test_the_same_fighter_in_both_corners_is_refused_by_the_pipeline(last_event):
    """build_pending cannot tell a typo from a fight; the pipeline can."""
    when, before, event = last_event
    f = _buildable(before, _fights(event, when))[0]
    pend = pending_rows.build_pending(before, [dict(f, blue=f["red"])])
    assert pend["r_id"].iloc[0] == pend["b_id"].iloc[0]
    with pytest.raises(ValueError, match="twice"):
        feature_frame.build(before, pending=pend, verbose=False)


# --- the division ----------------------------------------------------------

def test_division_is_inferred_when_both_fighters_last_fought_in_it(last_event):
    when, before, event = last_event
    f = _buildable(before, _fights(event, when))[0]
    divs = pending_rows.last_division(before)
    f["division"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)     # none allowed here
        pend = pending_rows.build_pending(before, [f])
    assert divs[pend["r_id"].iloc[0]] == divs[pend["b_id"].iloc[0]]
    assert pend["division"].iloc[0] == divs[pend["r_id"].iloc[0]]


def test_division_is_left_empty_with_a_warning_when_their_last_bouts_differ(last_event):
    when, before, event = last_event
    f = _buildable(before, _fights(event, when))[0]
    f["division"] = None
    moved = before.copy()
    days = pd.to_datetime(moved["date"], errors="coerce")
    mine = (moved["r_name"] == f["red"]) | (moved["b_name"] == f["red"])
    latest = days[mine].idxmax()
    moved.loc[latest, "division"] = "lightweight"
    with pytest.warns(UserWarning, match="last bouts differ"):
        pend = pending_rows.build_pending(moved, [f])
    assert pd.isna(pend["division"].iloc[0])


def test_a_curly_apostrophe_division_is_read_as_the_archive_spells_it(raw):
    """Duda Santana's last listed weight (65.77) is not the women's
    bantamweight limit, so the class lookup is what sets her row's weight -
    and the lookup used to miss a curly apostrophe."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = pd.Timestamp("2026-10-10")
    before = raw[days < when]
    pend = pending_rows.build_pending(before, [dict(
        red="Duda Santana", blue="Ailin Perez", date=when,
        division="Women’s Bantamweight")])
    assert pend["division"].iloc[0] == "women's bantamweight"
    assert pend["r_weight"].iloc[0] == pending_rows.DIVISION_KG["women's bantamweight"]
    assert pending_rows.normal_division(" Light Heavyweight ") == "light heavyweight"
    assert pending_rows.normal_division(None) is None
    assert pending_rows.normal_division(np.nan) is None


# --- the fighters ------------------------------------------------------------

def test_an_exponent_shaped_id_survives_a_one_fight_build(raw, full):
    """Nathaniel Wood's id is '329e403448756217'. Parsed alone, a CSV
    reader takes it for a float and overflows it to inf - and the pipeline
    then finds no career for him. The live path builds one fight at a time
    on a cache miss, so this is the single-fight case."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = pd.Timestamp("2026-10-10")
    before = raw[days < when]
    pend = pending_rows.build_pending(before, [dict(
        red="Nathaniel Wood", blue="Losene Keita", date=when,
        division="bantamweight")])
    assert pend["r_id"].iloc[0] == "329e403448756217"
    assert pend["r_id"].dtype == before["r_id"].dtype
    bouts = int(((before["r_id"] == "329e403448756217")
                 | (before["b_id"] == "329e403448756217")).sum())
    assert bouts >= 10
    built = feature_frame.build(before, pending=pend,
                                centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    row = built["ufc"].iloc[built["N_ARCHIVE"]]
    assert row["r_id"] == "329e403448756217"
    assert row["r_cd_bouts"] == bouts


def test_a_name_two_fighters_share_is_settled_by_the_division(raw):
    """'Bruno Silva' is a middleweight and a flyweight. The archive filed
    one of the flyweight's bouts under the middleweight's id, so at this
    cut both are 'flyweight' last; who fights in the division decides."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = pd.Timestamp("2026-03-14")
    before = raw[days < when]
    assert "bruno silva" not in pending_rows.fighter_ids(before)
    assert sorted(pending_rows.fighter_candidates(before)["bruno silva"]) == [
        "12ebd7d157e91701", "294aa73dbf37d281"]
    fly = pending_rows.build_pending(before, [dict(
        red="Charles Johnson", blue="Bruno Silva", date=when, division="flyweight")])
    assert fly["b_id"].iloc[0] == "294aa73dbf37d281"
    assert fly["b_id"].iloc[0] == raw[(days == when) & (raw["r_name"] == "Charles Johnson")]["b_id"].iloc[0]
    middle = pending_rows.build_pending(before, [dict(
        red="Chris Weidman", blue="Bruno Silva", date=when, division="Middleweight")])
    assert middle["b_id"].iloc[0] == "12ebd7d157e91701"
    with pytest.raises(KeyError, match="two fighters"):
        pending_rows.build_pending(before, [dict(
            red="Charles Johnson", blue="Bruno Silva", date=when)])
    with pytest.raises(KeyError, match="does not tell them apart"):
        pending_rows.build_pending(before, [dict(
            red="Charles Johnson", blue="Bruno Silva", date=when, division="welterweight")])


def test_a_shared_name_is_refused_when_the_two_division_signals_disagree(raw):
    """Had the flyweight Bruno Silva's latest bout been his one
    bantamweight one, the latest-bout rule alone would name the
    middleweight (whose latest row reads 'flyweight', the misfiled bout)
    for a flyweight fight, while the bout count names the flyweight: the
    two signals disagree, and the fight is refused saying so."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = pd.Timestamp("2026-03-14")
    moved = raw[days < when].copy()
    fly = "294aa73dbf37d281"
    mine = (moved["r_id"] == fly) | (moved["b_id"] == fly)
    latest = days[days < when][mine].idxmax()
    assert moved.loc[latest, "division"] == "flyweight"
    moved.loc[latest, "division"] = "bantamweight"
    candidates = pending_rows.fighter_candidates(moved)
    history = pending_rows.division_history(moved)
    assert history["12ebd7d157e91701"][-1] == "flyweight"       # the misfiled bout
    with pytest.raises(KeyError, match="does not settle it - the latest bout there "
                                       "is 12ebd7d157e91701's, most bouts there are "
                                       f"{fly}'s"):
        pending_rows.resolve_fighter("Bruno Silva", "flyweight", candidates, history)
    # the middleweight is still the only one who fights at middleweight
    assert pending_rows.resolve_fighter("Bruno Silva", "middleweight",
                                        candidates, history) == "12ebd7d157e91701"
    # and the unmodified archive still settles flyweight by the bout count
    before = raw[days < when]
    assert pending_rows.resolve_fighter(
        "Bruno Silva", "flyweight", pending_rows.fighter_candidates(before),
        pending_rows.division_history(before)) == fly


def test_a_placeholder_id_is_read_as_the_id_the_next_row_will_carry(raw, full):
    """The sync wrote 'new_955da1675ad58a50' for Josh Hokit's first bout
    and '955da1675ad58a50' from his second on. Training built his second
    bout as a debut (the real id had no earlier row); the pending row for
    it must carry the same id and the same debut-style career."""
    ids = pending_rows.fighter_ids(raw)
    assert not any(str(v).startswith(("new_", "unk_")) for v in ids.values())
    assert ids["josh hokit"] == "955da1675ad58a50"
    assert ids["waldo cortes acosta"] == "fc08099550072fe4"
    assert ids["ethyn ewing"] == "d776b5814f2400b4"
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = pd.Timestamp("2026-01-24")
    before = raw[days < when]
    fu = full["ufc"]
    j = fu.index[(fu["date"] == when) & (fu["r_name"] == "Josh Hokit")][0]
    pend = pending_rows.build_pending(before, [dict(
        red="Josh Hokit", blue=fu.loc[j, "b_name"], date=when,
        division=fu.loc[j, "division"])])
    assert pend["r_id"].iloc[0] == fu.loc[j, "r_id"] == "955da1675ad58a50"
    built = feature_frame.build(before, pending=pend,
                                centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    row = built["ufc"].iloc[built["N_ARCHIVE"]]
    for col in ("r_cd_bouts", "r_exp", "r_wins", "r_losses"):
        assert row[col] == fu.loc[j, col], col
    assert row["r_cd_bouts"] == 0
    # The whole vector: every other feature equals training. What differs
    # is profile-derived only - his placeholder row lists 208 cm, no dob
    # and a heavyweight's last listed weight where the next scrape wrote
    # 185 cm, 1997-11-12 and 106.59; Freeman's reach and stance were
    # rewritten for the bout too. Nothing before the date can know those.
    bad = _differing(built, full, j)
    assert bad and set(bad) <= PROFILE_DERIVED, sorted(set(bad) - PROFILE_DERIVED)
    assert pend["r_height"].iloc[0] == 208.28 and pd.isna(pend["r_dob"].iloc[0])


# Features that read a corner's profile (height, reach, dob, stance, listed
# weight) and nothing else - the only ones a pending row built from a
# placeholder row can differ from its training row on.
PROFILE_DERIVED = {
    "height_known", "height_level", "height_diff", "ape_index_diff",
    "age_known", "age_level", "age_diff", "prime_wc_diff",
    "reach_known", "reach_level", "reach_diff",
    "southpaw_diff", "stance_mismatch",
    "mx_size_known", "mx_size_adv", "mx_mass_adv"}


def _differing(built, full, j):
    """The feature columns on which the first pending row of `built`
    differs from training row `j`."""
    cols = full["feature_cols"]
    mine = built["X"].loc[built["N_ARCHIVE"], cols].to_numpy(float)
    theirs = full["X"].loc[j, cols].to_numpy(float)
    return [c for c, a, b in zip(cols, mine, theirs)
            if not (np.isclose(a, b, rtol=1e-9, atol=1e-12)
                    or (np.isnan(a) and np.isnan(b)))]


def test_a_fighter_whose_only_row_is_a_placeholder_is_built_with_an_unknown_profile(raw, full):
    """Santiago Luna's one archive row before 2026-02-28 is a placeholder
    with a 381 cm height and no dob. HEIGHT_CM drops the height, so his
    pending row has no height and no age: the pipeline's own unknown
    branch (height_known = 0, age_known = 0), as training built every row
    of a fighter it had no profile for. The next scrape knew both, so the
    training row for the bout differs on exactly the ten height- and
    age-derived features and nothing else - the gap, pinned."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = pd.Timestamp("2026-02-28")
    before = raw[days < when]
    luna = before[(before["r_name"] == "Santiago Luna") | (before["b_name"] == "Santiago Luna")]
    assert len(luna) == 1 and str(luna["r_id"].iloc[0]).startswith("new_")
    assert float(luna["r_height"].iloc[0]) > pending_rows.HEIGHT_CM[1]
    fu = full["ufc"]
    j = fu.index[(fu["date"] == when) & (fu["r_name"] == "Santiago Luna")][0]
    pend = pending_rows.build_pending(before, [dict(
        red="Santiago Luna", blue=fu.loc[j, "b_name"], date=when,
        division=fu.loc[j, "division"])])
    assert pend["r_id"].iloc[0] == fu.loc[j, "r_id"]
    assert pd.isna(pend["r_height"].iloc[0]) and pd.isna(pend["r_dob"].iloc[0])
    built = feature_frame.build(before, pending=pend,
                                centres=full["INTERACTION_CENTRES"],
                                verbose=False)
    row = built["X"].iloc[built["N_ARCHIVE"]]
    assert row["height_known"] == 0 and row["age_known"] == 0
    assert set(_differing(built, full, j)) == {
        "height_known", "height_level", "height_diff", "ape_index_diff",
        "age_known", "age_level", "age_diff", "prime_wc_diff",
        "mx_size_known", "mx_size_adv"}


def test_a_units_bug_height_is_not_inherited(raw):
    """17 rows of late 2025 list heights of 234-381 cm. The latest valid
    scrape, or nothing, is what a pending row gets."""
    prof = pending_rows.profiles(raw)
    assert not (prof["height"] > pending_rows.HEIGHT_CM[1]).any()
    assert not (prof["height"] < pending_rows.HEIGHT_CM[0]).any()
    ids = pending_rows.fighter_ids(raw)
    sezinando = ids["rodrigo sezinando"]          # his only scrape says 381
    assert sezinando not in prof["height"].index
    assert prof["reach"].get(sezinando) is not None


def test_dob_is_normalised_to_the_archive_format():
    assert pending_rows._normal_dob("Jan 05, 1990") == "1990/01/05"
    assert pending_rows._normal_dob("1990/01/05") == "1990/01/05"
    assert pd.isna(pending_rows._normal_dob(None))
