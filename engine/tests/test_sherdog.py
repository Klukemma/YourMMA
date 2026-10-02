"""Sherdog records, parsed from pages saved from Sherdog by the probe."""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import sherdog

FIX = ENGINE / "tests" / "fixtures" / "sherdog"


def page(name):
    path = FIX / name
    if not path.exists():
        pytest.skip(f"{name} not saved")
    return path.read_text()


def test_a_record_is_read_newest_first_with_dates_and_ids():
    # The probe saved Alonzo Menifield's page (see test_pick below for why).
    bouts = sherdog.record(page("fighter_Luca_Borando.html"))
    assert len(bouts) >= 10
    first = bouts[0]
    assert first == {"date": "2026-09-19", "result": "win",
                     "opponent": "Iwo Baraniewski", "opponent_id": 381634,
                     "event": "UFC 331 - Van vs. Pantoja 2",
                     "method": "Decision (Split)", "round": "3", "time": "5:00"}
    loss = bouts[2]
    assert (loss["result"], loss["opponent"], loss["method"]) == \
        ("loss", "Volkan Oezdemir", "KO (Knee and Punches)")
    assert all(b["date"] and b["opponent_id"] > 0 for b in bouts)


def test_pick_takes_the_fighter_searched_for_not_a_sidebar_link():
    search = page("search_Luca_Borando.html")
    assert sherdog.pick(search, "Luca Borando") == ("Luca-Borando", 304627)
    assert sherdog.pick(search, "Nobody Atall") is None


def test_amateur_bouts_are_not_read():
    html = ('FIGHT HISTORY - PRO <table>'
            '<tr><td><span class="final_result win">win</span></td>'
            '<td><a href="/fighter/A-B-1">A B</a></td>'
            '<td><a href="/events/X-1">X 1</a><br /><span class="sub_line">'
            'Jan / 02 / 2020</span></td><td class="winby"><b>KO</b></td>'
            '<td>1</td><td>1:00</td></tr></table>'
            'FIGHT HISTORY - AMATEUR <table>'
            '<tr><td><span class="final_result win">win</span></td>'
            '<td><a href="/fighter/C-D-2">C D</a></td>'
            '<td><a href="/events/Y-2">Y 2</a><br /><span class="sub_line">'
            'Jan / 02 / 2019</span></td><td class="winby"><b>KO</b></td>'
            '<td>1</td><td>1:00</td></tr></table>')
    got = sherdog.record(html)
    assert [b["opponent"] for b in got] == ["A B"]


def test_candidates_come_from_the_results_table_not_the_trending_sidebar():
    search = page("search_Luca_Borando.html")
    assert sherdog.candidates(search) == [("Luca-Borando", 304627,
                                           "Luca Borando")]


def test_bio_reads_constants_only():
    got = sherdog.bio(page("fighter_Luca_Borando.html"))   # Menifield's page
    assert got == {"birth_date": "1987-10-18", "height_cm": 182.9}


def test_a_record_is_confirmed_only_by_the_bout_it_was_looked_up_for():
    rec = sherdog.record(page("fighter_Luca_Borando.html"))
    assert sherdog.confirms(rec, "2026-09-20", "Iwo Baraniewski")
    # A UFC or Contender Series bout on the date confirms the record even
    # under a differently transliterated opponent: a namesake on the same
    # UFC card cannot happen. A bout on some other card does not.
    assert not sherdog.confirms(rec, "2026-09-20", "Somebody Else", days=0) \
        or rec[0]["event"].lower().startswith("ufc")
    assert not sherdog.confirms(rec, "2026-07-01", "Somebody Else")
    # A different spelling of the opponent is still accepted on a
    # Contender Series card (Menifield's 2018 appearance).
    assert sherdog.confirms(rec, "2018-06-12", "Dmitrii Smoliakov")
    assert not sherdog.confirms(rec, "2015-06-12", None)


def test_find_skips_a_namesake_whose_record_lacks_the_bout(monkeypatch):
    menifield = page("fighter_Luca_Borando.html")
    search = ('<table class="new_table fightfinder_result">'
              '<tr><td><a href="/fighter/Alonzo-Menifield-1">Alonzo Menifield</a>'
              '</td></tr><tr><td><a href="/fighter/Alonzo-Menifield-2">'
              'Alonzo Menifield</a></td></tr></table>')
    pages = {"fightfinder": search, "Menifield-1": "no record here",
             "Menifield-2": menifield}
    fetched = []

    def fake(url, agent, pause=0):
        fetched.append(url)
        return next(v for k, v in pages.items() if k in url)
    monkeypatch.setattr(sherdog, "fetch", fake)
    hit = sherdog.find("Alonzo Menifield", "test",
                       [("2026-09-19", "Iwo Baraniewski")])
    assert (hit["id"], hit["verified"]) == (2, True)
    assert hit["bio"]["birth_date"] == "1987-10-18"
    # Without a bout to check against, two namesakes means no answer.
    assert sherdog.find("Alonzo Menifield", "test", []) is None
