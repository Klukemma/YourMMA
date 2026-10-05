"""Mentality signals may only come from fights before the one predicted."""

import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from mentality import MentalityIndex

COLS = ["date", "r_name", "b_name", "r_kd", "b_kd", "winner", "method",
        "finish_round", "match_time_sec", "title_fight", "r_sig_str_atmpted",
        "b_sig_str_atmpted", "r_td_atmpted", "b_td_atmpted", "r_mu_pre",
        "b_mu_pre"]


def archive(rows):
    return pd.DataFrame(rows, columns=COLS)


# As the archive stores them: match_time_sec is the time in the FINAL round.
# Ann: dropped and finished (2019), dropped and came back to win (2020),
# then three straight wins with a rising rating (2021-22), then a title.
A = archive([
    ("2019-01-01", "Ann", "X1", 0, 1, "X1", "KO/TKO", 1, 200, 0, 50, 40, 0, 0,
     20, 22),
    ("2020-01-01", "Ann", "X2", 1, 1, "Ann", "KO/TKO", 2, 100, 0, 60, 40, 0, 0,
     21, 22),
    ("2021-01-01", "Ann", "X3", 0, 0, "Ann", "Decision - Unanimous", 3, 300, 0,
     140, 90, 10, 0, 22, 22),
    ("2021-06-01", "Ann", "X4", 0, 0, "Ann", "U-DEC", 3, 300, 0,
     150, 90, 0, 0, 23, 22),
    ("2022-01-01", "Ann", "X5", 1, 0, "Ann", "KO/TKO", 1, 300, 1, 60, 30, 0, 0,
     24, 22),
])


def test_nothing_from_the_fight_itself_or_after():
    m = MentalityIndex(A)
    before_first = m.at("Ann", "2019-01-01")
    assert before_first["fights"] == 0 and before_first["dropped"] == 0
    assert m.at("Ann", "2019-06-01")["dropped"] == 1


def test_being_dropped_recovering_and_coming_back():
    got = MentalityIndex(A).at("Ann", "2020-06-01")
    assert (got["dropped"], got["recovered"], got["comebacks"]) == (2, 1, 1)


def test_rising_is_a_streak_with_no_title_yet_and_the_rating_climbing():
    m = MentalityIndex(A)
    assert m.at("Ann", "2021-12-01")["rising"]          # 3 straight, no title
    after_title = m.at("Ann", "2022-06-01")
    assert after_title["champion_stage"] == 1
    assert not after_title["rising"]                   # a champion is not "rising"


def test_confidence_is_counted_separately_from_hunger():
    got = MentalityIndex(A).at("Ann", "2022-06-01")
    assert got["dominant_streak"] == 4 and got["streak_finishes"] == 2
    assert not got["rising"]


def test_a_long_layoff_is_not_rising():
    got = MentalityIndex(A).at("Ann", "2023-06-01")    # 17 months out
    assert not got["rising"] and got["layoff_days"] > 500


def test_quit_losses_come_from_results_pages_only():
    world = pd.DataFrame([
        ("2018-05-01", "Y", "Ann", "win", "Submission (punches)", "2018 in LFA"),
        ("2018-06-01", "Z", "Ann", "win", "TKO (retirement)", "record:Z"),
    ], columns=["date", "winner", "loser", "result", "method", "source"])
    m = MentalityIndex(A, world)
    assert m.at("Ann", "2019-01-01")["quit_losses"] == 1   # record row ignored
    assert m.at("Ann", "2018-05-01")["quit_losses"] == 0   # not before itself


def test_pace_uses_the_whole_fight_not_the_final_round():
    # Fights 2-5 last 6.7, 15, 15 and 5 minutes: output 9, 10, 10, 12 a
    # minute. Last two against the two before: 11 / 9.5.
    got = MentalityIndex(A).at("Ann", "2023-01-01")
    assert abs(got["pace_trend"] - 11 / 9.5) < 1e-6


def test_every_spelling_of_a_decision_is_a_decision():
    got = MentalityIndex(A).at("Ann", "2022-06-01")
    # Wins in the streak: U-DEC, Decision, KO -> two decisions, one finish
    # of the four (the 2020 KO is the fourth).
    assert got["streak_finishes"] == 2


def test_a_kneebar_is_not_quitting_but_tapping_to_punches_is():
    world = pd.DataFrame([
        ("2018-01-01", "Y", "Ann", "win", "Submission (kneebar)", "2018 in X"),
        ("2018-02-01", "Y", "Ann", "win", "Verbal Submission (armbar)", "2018 in X"),
        ("2018-03-01", "Y", "Ann (c)", "win", "Submission (punches)", "2018 in X"),
    ], columns=["date", "winner", "loser", "result", "method", "source"])
    # The third also carries a title tag, which must not hide it.
    assert MentalityIndex(A, world).at("Ann", "2019-01-01")["quit_losses"] == 1


def test_a_winner_spelled_differently_from_the_corner_still_won():
    b = archive([("2020-01-01", "Waldo Cortes Acosta", "X", 0, 0,
                  "Waldo Cortes-Acosta", "KO/TKO", 1, 100, 0, 10, 5, 0, 0, 20, 20)])
    got = MentalityIndex(b).at("Waldo Cortes Acosta", "2020-02-01")
    assert got["dominant_streak"] == 1


def test_a_repaired_row_that_stores_total_seconds_is_not_counted_twice():
    # Late-2025 rows store the whole fight: a 3-round decision reads 900.
    b = archive([
        ("2025-10-01", "Bo", "X", 0, 0, "Bo", "U-DEC", 3, 900, 0, 150, 90, 0, 0,
         20, 20)])
    fight = MentalityIndex(b).fights["bo"][0]
    assert abs(fight["pace"] - 150 / 15) < 1e-9       # 15 minutes, not 25
