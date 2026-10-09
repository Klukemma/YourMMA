"""Tests for grading logged predictions against real results."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from grade import (categorise_method, deduplicate, find_result, grade,
                   load_results)


@pytest.fixture
def results(tmp_path):
    csv = tmp_path / "f.csv"
    pd.DataFrame({
        'date': ['2026-01-24', '2026-02-01', '2026-03-01'],
        'r_name': ['Justin Gaethje', 'Alice One', 'Carl Three'],
        'b_name': ['Paddy Pimblett', 'Bob Two', 'Dan Four'],
        'winner': ['Justin Gaethje', 'Bob Two', ''],
        'method': ['Decision - Unanimous', 'KO/TKO', 'Overturned'],
    }).to_csv(csv, index=False)
    return load_results(csv)


def _pred(red, blue, winner, date, prob=0.6, method='Decision', stamp='2026-01-01T00:00:00'):
    return {'red_corner': red, 'blue_corner': blue, 'predicted_winner': winner,
            'event_date': date, 'win_probability': prob, 'method_predicted': method,
            'event_name': 'Test', 'timestamp': stamp}


@pytest.mark.parametrize("raw,expected", [
    ('Decision - Unanimous', 'Decision'), ('Decision - Split', 'Decision'),
    ('Decision - Majority', 'Decision'), ('KO/TKO', 'KO/TKO'),
    ("TKO - Doctor's Stoppage", 'KO/TKO'), ('Submission', 'Submission'),
    ('Overturned', None), ('Could Not Continue', None), (None, None),
])
def test_method_categories(raw, expected):
    assert categorise_method(raw) == expected


def test_grades_a_correct_and_an_incorrect_pick(results):
    preds = [_pred('Justin Gaethje', 'Paddy Pimblett', 'Justin Gaethje', '2026-01-24'),
             _pred('Alice One', 'Bob Two', 'Alice One', '2026-02-01')]
    graded, ungraded = grade(preds, results)
    assert [r['correct'] for r in graded] == [True, False]
    assert ungraded == []


def test_matches_regardless_of_corner_order(results):
    """The logged red corner may be the recorded blue corner."""
    preds = [_pred('Paddy Pimblett', 'Justin Gaethje', 'Justin Gaethje', '2026-01-24')]
    graded, _ = grade(preds, results)
    assert graded[0]['correct'] is True


def test_rescheduled_card_still_matches(results):
    """Cards move. Names plus a nearby date is the key, not the date alone."""
    preds = [_pred('Justin Gaethje', 'Paddy Pimblett', 'Justin Gaethje', '2026-01-20')]
    graded, ungraded = grade(preds, results)
    assert len(graded) == 1 and ungraded == []


def test_far_off_date_does_not_match(results):
    preds = [_pred('Justin Gaethje', 'Paddy Pimblett', 'Justin Gaethje', '2025-06-01')]
    graded, ungraded = grade(preds, results)
    assert graded == [] and len(ungraded) == 1


def test_a_bout_that_never_happened_is_not_graded(results):
    preds = [_pred('Ghost One', 'Ghost Two', 'Ghost One', '2026-01-24')]
    graded, ungraded = grade(preds, results)
    assert graded == []
    assert ungraded[0][1] == 'no matching bout in the dataset'


def test_draw_is_excluded_not_counted_wrong(results):
    """A no-contest is not a failed prediction."""
    preds = [_pred('Carl Three', 'Dan Four', 'Carl Three', '2026-03-01')]
    graded, ungraded = grade(preds, results)
    assert graded == []
    assert ungraded[0][1] == 'draw or no contest'


def test_duplicate_predictions_count_once():
    """Re-running a card logs it again; the bout must not gain extra weight."""
    preds = [
        _pred('A One', 'B Two', 'A One', '2026-01-24', stamp='2026-01-01T00:00:00'),
        _pred('A One', 'B Two', 'B Two', '2026-01-24', stamp='2026-01-20T00:00:00'),
        _pred('A One', 'B Two', 'A One', '2026-01-24', stamp='2026-01-10T00:00:00'),
    ]
    unique = deduplicate(preds)
    assert len(unique) == 1
    assert unique[0]['predicted_winner'] == 'B Two', "should keep the most recent"


def test_deduplicate_keeps_distinct_bouts():
    preds = [_pred('A One', 'B Two', 'A One', '2026-01-24'),
             _pred('C Three', 'D Four', 'C Three', '2026-01-24')]
    assert len(deduplicate(preds)) == 2


def test_the_shipped_history_grades(): 
    """End to end against the real files."""
    history = json.loads((ENGINE / "data" / "prediction_history.json").read_text())
    unique = deduplicate(history['predictions'])
    index = load_results(ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv")
    graded, ungraded = grade(unique, index)
    assert len(graded) > 100, "most logged bouts should now be gradeable"
    assert all(isinstance(r['correct'], bool) for r in graded)


# --- deduplication on the matched bout ------------------------------------

def test_same_bout_logged_under_two_event_names_counts_once():
    """The real case: one card logged as both "UFC 325 - Australia" and
    "UFC 325 - Sydney, Australia", with different event dates. Keying on what
    was logged leaves both; keying on the matched bout does not."""
    from grade import deduplicate_graded
    rows = [
        {'red_corner': 'A One', 'blue_corner': 'B Two', 'event_name': 'UFC 325 - Australia',
         'actual_date': pd.Timestamp('2026-01-31'), 'timestamp': '2026-01-20T00:00:00',
         'correct': True},
        {'red_corner': 'A One', 'blue_corner': 'B Two',
         'event_name': 'UFC 325 - Sydney, Australia',
         'actual_date': pd.Timestamp('2026-01-31'), 'timestamp': '2026-01-28T00:00:00',
         'correct': False},
    ]
    out = deduplicate_graded(rows)
    assert len(out) == 1
    assert out[0]['event_name'] == 'UFC 325 - Sydney, Australia', "keep the most recent"


def test_dedup_ignores_which_corner_is_which():
    from grade import deduplicate_graded
    rows = [
        {'red_corner': 'A', 'blue_corner': 'B', 'actual_date': pd.Timestamp('2026-01-31'),
         'timestamp': '1', 'correct': True},
        {'red_corner': 'B', 'blue_corner': 'A', 'actual_date': pd.Timestamp('2026-01-31'),
         'timestamp': '2', 'correct': True},
    ]
    assert len(deduplicate_graded(rows)) == 1


def test_a_rematch_on_a_different_date_is_kept():
    """Two fighters can meet twice. Those are distinct bouts."""
    from grade import deduplicate_graded
    rows = [
        {'red_corner': 'A', 'blue_corner': 'B', 'actual_date': pd.Timestamp('2026-01-31'),
         'timestamp': '1', 'correct': True},
        {'red_corner': 'A', 'blue_corner': 'B', 'actual_date': pd.Timestamp('2026-06-20'),
         'timestamp': '2', 'correct': False},
    ]
    assert len(deduplicate_graded(rows)) == 2


def test_picks_are_logged_only_up_to_the_day_of_the_card():
    """A pick made after the card is not a prediction, and grading keeps the
    newest pick for a bout - so a late one would replace the real one."""
    from datetime import datetime, timezone
    from grade import picks_may_be_logged
    at = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    assert picks_may_be_logged("2026-10-10", now=at("2026-10-09T15:00:00"))
    assert picks_may_be_logged("2026-10-10", now=at("2026-10-10T20:00:00"))
    assert not picks_may_be_logged("2026-10-10", now=at("2026-10-11T02:00:00"))
    assert not picks_may_be_logged("not a date", now=at("2026-10-09T15:00:00"))


def test_the_engine_checks_before_it_logs():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "predict_card.py").read_text()
    guard = source.index("_grade.picks_may_be_logged(")
    log = source.index("log_card_predictions(predictions_to_log, EVENT_NAME, EVENT_DATE")
    assert guard < log
    # the guard nearest the call is the one that decides it
    assert source.rindex("_grade.picks_may_be_logged(", 0, log) > source.index("PREDICTION LOGGING")



def test_with_a_start_time_the_cutoff_is_the_first_bell():
    """A Vegas card is dated 10-10 locally and its first bell is on 10-11 UTC:
    a run just before it logs, one just after does not - by the start time,
    not by the date."""
    from datetime import datetime, timezone
    from grade import picks_may_be_logged
    at = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    start = "2026-10-11T00:00:00Z"
    assert picks_may_be_logged("2026-10-10", now=at("2026-10-10T23:59:00"), first_start=start)
    assert not picks_may_be_logged("2026-10-10", now=at("2026-10-11T00:01:00"), first_start=start)


def test_a_pick_logged_after_its_card_never_counts():
    """The log holds September re-runs of a June card; graded as the newest
    pick for each bout, they replaced the real picks and flipped results."""
    real = {'red_corner': 'Manel Kape', 'blue_corner': 'Kyoji Horiguchi',
            'event_date': '2026-06-20', 'predicted_winner': 'Kyoji Horiguchi',
            'timestamp': '2026-06-20T12:00:00'}
    late = dict(real, predicted_winner='Manel Kape', timestamp='2026-09-18T10:00:00')
    kept = deduplicate([real, late])
    assert [p['predicted_winner'] for p in kept] == ['Kyoji Horiguchi']


def test_a_pick_with_nothing_to_show_it_was_late_is_kept():
    from grade import made_before_the_card
    assert made_before_the_card({'event_date': '2026-06-20'})
    assert made_before_the_card({'timestamp': 'garbled', 'event_date': '2026-06-20'})



def test_grading_applies_the_cutoff_the_engine_logged_under():
    """A pick logged at 00:30Z before a 01:00Z first bell passed the guard,
    then the date rule dropped it in grading and an older pick was graded."""
    early = {'red_corner': 'Brendan Allen', 'blue_corner': 'Christian Leroy Duncan',
             'event_date': '2026-10-10', 'predicted_winner': 'Brendan Allen',
             'timestamp': '2026-10-09T18:00:00', 'first_start': '2026-10-11T01:00:00Z'}
    fight_night = dict(early, predicted_winner='Christian Leroy Duncan',
                       timestamp='2026-10-11T00:30:00')
    kept = deduplicate([early, fight_night])
    assert [p['predicted_winner'] for p in kept] == ['Christian Leroy Duncan']
    after_bell = dict(fight_night, timestamp='2026-10-11T01:30:00')
    assert [p['predicted_winner'] for p in deduplicate([early, after_bell])] == ['Brendan Allen']


def test_the_engine_checks_the_cutoff_before_buying_a_line():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "predict_card.py").read_text()
    guard = source.index("if _grade.picks_may_be_logged(EVENT_DATE, first_start=_card_first_start()):")
    assert guard < source.index("_should_fetch, _why = _odds_cache.decide(")
    assert "first_start=_first_start)" in source
