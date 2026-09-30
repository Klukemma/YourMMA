"""Injury history must never let a fight see what was learned from it.

Every test here is a version of one question: for a fight on date D, is
anything visible that was not public before D? "Fought with a torn ACL",
revealed after a loss, is the case that matters most. Read as known
beforehand it predicts the loss it was disclosed because of.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import injury_history as ih

ARCHIVE = pd.DataFrame([
    # Brown fights Evans, withdraws from UFC 200 vs Green (not in the
    # archive: the bout never happened), then meets Green six months later.
    ("UFC 196: McGregor vs Diaz", "2016-03-05", "Carl Brown", "Dan Evans",
     "KO/TKO", "Carl Brown"),
    ("UFC 200: Tate vs Nunes", "2016-07-09", "Tom Hall", "Ed Green",
     "Decision - Unanimous", "Ed Green"),
    ("UFC on Fox 23: Shevchenko vs Pena", "2017-01-28", "Ed Green",
     "Carl Brown", "KO/TKO", "Ed Green"),
    ("UFC 210: Cormier vs Johnson 2", "2017-04-08", "Carl Brown",
     "Leslie Smith", "Decision - Split", "Carl Brown"),
    ("UFC 220: Miocic vs Ngannou", "2018-01-20", "Anthony Smith",
     "Rob Doctor", "TKO - Doctor's Stoppage", "Anthony Smith"),
], columns=["event_name", "date", "r_name", "b_name", "method", "winner"])


@pytest.fixture
def archive():
    return ih.Archive(ARCHIVE)


def sentence(sid, source, subject, subject_date="", text="..."):
    return {"id": sid, "source": source, "subject": subject,
            "subject_date": subject_date, "page": subject, "text": text}


def record(sid, **fields):
    base = {"sentence_id": sid, "fighter": "Carl Brown", "kind": "withdrawal",
            "condition": "knee injury", "body_part": "knee",
            "disclosed": "before", "certainty": "stated"}
    base.update(fields)
    return base


# --- dates ------------------------------------------------------------------

def test_date_windows():
    day = ih.parse_date("2016-07-09")
    assert day[2] == "day" and day[0] == day[1] == pd.Timestamp("2016-07-09")
    month = ih.parse_date("2016-06")
    assert month[2] == "month" and month[1] == pd.Timestamp("2016-06-30")
    assert ih.parse_date("June 2016")[1] == pd.Timestamp("2016-06-30")
    assert ih.parse_date("2016")[1] == pd.Timestamp("2016-12-31")
    assert ih.parse_date("July 9, 2016")[0] == pd.Timestamp("2016-07-09")
    assert ih.parse_date("") is None and ih.parse_date(None) is None


def test_before_a_dated_bout_is_known_the_day_before():
    window = ih.parse_date("2016-07-09")
    assert ih.known_by("before", window) == pd.Timestamp("2016-07-08")


def test_before_a_month_is_known_only_at_its_end():
    # "In June he pulled out" - the worst case is June 30.
    assert ih.known_by("before", ih.parse_date("2016-06")) == \
        pd.Timestamp("2016-06-30")


@pytest.mark.parametrize("disclosed", ["at", "after"])
def test_in_or_after_a_bout_is_known_from_that_bout_not_before(disclosed):
    window = ih.parse_date("2017-01-28")
    assert ih.known_by(disclosed, window) == pd.Timestamp("2017-01-28")


def test_a_stated_disclosure_date_can_only_push_later():
    window = ih.parse_date("2017-01-28")
    assert ih.known_by("after", window, ih.parse_date("2019")) == \
        pd.Timestamp("2019-12-31")
    assert ih.known_by("after", window, ih.parse_date("2016")) == \
        pd.Timestamp("2017-01-28")


# --- building ---------------------------------------------------------------

def test_a_withdrawal_on_an_event_page_takes_the_event_date(archive):
    s = [sentence("a", "event", "UFC 200: Tate vs Nunes", "2016-07-09")]
    out = ih.build([record("a")], s, archive)
    row = out.iloc[0]
    assert row.fighter == "Carl Brown" and row.matched
    assert row.anchor_source == "event_page"
    assert row.known_by == "2016-07-08"


def test_a_post_fight_revelation_is_invisible_to_that_fight(archive):
    s = [sentence("b", "fighter", "Carl Brown")]
    rec = record("b", kind="fought_hurt", condition="torn ACL",
                 disclosed="after", anchor_event="UFC on Fox 23: Shevchenko vs Pena",
                 anchor_opponent="Ed Green")
    history = ih.build([rec], s, archive)
    assert history.iloc[0].known_by == "2017-01-28"
    # Not before the fight it describes, not on its day...
    assert ih.known_injuries(history, "Carl Brown", "2017-01-28").empty
    # ...only for the fight after it.
    assert len(ih.known_injuries(history, "Carl Brown", "2017-04-08")) == 1


def test_a_later_admission_with_no_date_is_left_out(archive):
    s = [sentence("c", "fighter", "Carl Brown")]
    rec = record("c", kind="fought_hurt", disclosed="after",
                 later_disclosure=True,
                 anchor_event="UFC on Fox 23: Shevchenko vs Pena")
    history = ih.build([rec], s, archive)
    assert history.iloc[0].vague
    assert ih.known_injuries(history, "Carl Brown", "2030-01-01").empty
    assert len(ih.known_injuries(history, "Carl Brown", "2030-01-01",
                                 include_vague=True)) == 1


def test_unclear_timing_is_vague(archive):
    s = [sentence("d", "fighter", "Carl Brown")]
    history = ih.build([record("d", disclosed="unclear",
                               anchor_date="2016-07-09")], s, archive)
    assert history.iloc[0].vague


def test_a_withdrawn_bout_is_never_dated_by_a_later_rematch(archive):
    # Brown pulled out of Green at UFC 200 (July 2016) and fought Green in
    # January 2017. Looking the withdrawal up by opponent would date it to
    # the rematch - six months late - and a reader could not tell.
    s = [sentence("e", "fighter", "Carl Brown")]
    rec = record("e", anchor_opponent="Ed Green", anchor_date="2016-07-09")
    row = ih.build([rec], s, archive).iloc[0]
    assert row.anchor_date == "2016-07-09"
    assert row.anchor_source == "text_date"


def test_an_archive_event_beats_the_readers_date_and_flags_disagreement(archive):
    s = [sentence("f", "fighter", "Carl Brown")]
    rec = record("f", anchor_event="UFC 200", anchor_date="2016-08-20")
    row = ih.build([rec], s, archive).iloc[0]
    assert row.anchor_date == "2016-07-09"
    assert row.anchor_source == "archive_event"
    assert row.conflict


def test_a_bout_that_happened_can_be_found_by_opponent(archive):
    s = [sentence("g", "fighter", "Carl Brown")]
    rec = record("g", kind="in_fight", disclosed="at", anchor_opponent="Smith",
                 anchor_date="2017")
    row = ih.build([rec], s, archive).iloc[0]
    assert row.anchor_date == "2017-04-08" and row.anchor_source == "archive_bout"


# --- names --------------------------------------------------------------------

def test_a_surname_resolves_within_the_pages_people(archive):
    assert archive.name("Green", pool=["Carl Brown", "Ed Green"]) == \
        ("Ed Green", True)


def test_a_different_first_name_is_a_different_person(archive):
    # Leslie Smith fought Brown; "Anthony Smith" in Brown's page is not her.
    got = archive.name("Anthony Smithers",
                       pool=["Carl Brown", "Leslie Smith"])
    assert got == ("Anthony Smithers", False)
    assert archive.name("Anthony Smith", pool=["Leslie Smith"]) == \
        ("Anthony Smith", True)   # an archive fighter in his own right


def test_an_unknown_person_stays_unmatched_and_unusable(archive):
    s = [sentence("h", "event", "UFC 200: Tate vs Nunes", "2016-07-09")]
    history = ih.build([record("h", fighter="Jimmy Nobody")], s, archive)
    assert not history.iloc[0].matched
    assert ih.known_injuries(history, "Jimmy Nobody", "2030-01-01").empty


# --- merging ------------------------------------------------------------------

def test_one_withdrawal_told_twice_is_one_injury(archive):
    s = [sentence("i", "event", "UFC 200: Tate vs Nunes", "2016-07-09"),
         sentence("j", "fighter", "Carl Brown")]
    recs = [record("i"), record("j", anchor_event="UFC 200")]
    history = ih.build(recs, s, archive)
    assert len(history) == 1
    assert history.iloc[0].n_sources == 2


def test_two_stories_about_when_keep_the_later(archive):
    s = [sentence("k", "fighter", "Carl Brown"),
         sentence("l", "event", "UFC on Fox 23: Shevchenko vs Pena",
                  "2017-01-28")]
    recs = [record("k", kind="fought_hurt", disclosed="before",
                   anchor_event="UFC on Fox 23: Shevchenko vs Pena"),
            record("l", kind="fought_hurt", disclosed="after")]
    history = ih.build(recs, s, archive)
    assert len(history) == 1
    row = history.iloc[0]
    assert row.known_by == "2017-01-28" and row.conflict
    assert ih.known_injuries(history, "Carl Brown", "2017-01-28").empty


# --- the archive's own stoppages ------------------------------------------------

def test_a_doctor_stoppage_is_the_losers_in_fight_injury():
    out = ih.archive_stoppages(ARCHIVE)
    assert list(out.fighter) == ["Rob Doctor"]
    assert out.iloc[0].known_by == "2018-01-20"
    assert ih.known_injuries(out, "Rob Doctor", "2018-01-20").empty
    assert len(ih.known_injuries(out, "Rob Doctor", "2018-01-21")) == 1


def test_the_reader_batches_cover_every_sentence_once():
    sentences = [{"id": str(i), "text": "x"} for i in range(125)]
    got = [s["id"] for k in range(3) for s in ih.batch(sentences, k, 60)]
    assert got == [str(i) for i in range(125)]
