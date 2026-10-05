"""The injury harvester, offline: what it keeps, what it drops, whom it finds.

The harvest runs on the Actions runner, where a mistake costs a twenty-minute
round trip and is found only by reading its output. These pin the parts that
decide what text reaches the readers: which sentences count, how a record
table row survives, and which article is a fighter's own.
"""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import harvest_injuries as hi

FIGHTER_PAGE = """'''Carl Brown''' (born 1990) is an American [[mixed martial arts|mixed martial artist]].

==Mixed martial arts career==
===Ultimate Fighting Championship===
Brown made his debut against [[Dan Evans]] on March 5, 2016, at [[UFC 196]]. He won via knockout in the first round.

Brown was expected to face [[Ed Green]] on July 9, 2016, at [[UFC 200]]. However, Brown pulled out of the fight in late June citing a knee injury and was replaced by [[Tom Hall]].

Brown faced [[Ed Green]] on {{dts|2017|01|28}} at UFC on Fox 23. He lost by TKO after a knee to the head in the second round. After the fight, Brown revealed that he had fought with a torn ACL.<ref>{{cite web|title=Brown fought with torn ACL}}</ref>

==Personal life==
Brown is married and has two children.

==Mixed martial arts record==
{| class="wikitable sortable"
|-
!Res.
!Record
!Opponent
!Method
!Event
!Date
|-
|Loss
|align=center|5–1
|Ed Green
|TKO (knee injury)
|UFC on Fox 23
|{{dts|2017|01|28}}
|-
|Win
|align=center|5–0
|Dan Evans
|KO (punch)
|UFC 196
|{{dts|2016|March|5}}
|}

==References==
{{reflist}}
* Brown suffered a broken hand, according to a source.
"""


def by_text(found, needle):
    return [h for h in found if needle in h["text"]]


def test_a_withdrawal_sentence_is_kept_with_the_bout_it_belongs_to():
    found = hi.hits(FIGHTER_PAGE)
    hit = by_text(found, "knee injury and was replaced")
    assert len(hit) == 1
    hit = hit[0]
    assert hit["form"] == "prose"
    assert hit["section"] == "Ultimate Fighting Championship"
    # The sentence before it names the bout and its date; that is what makes
    # the withdrawal placeable.
    assert "July 9, 2016" in hit["context"]
    assert "UFC 200" in hit["context"]
    # The hint is the latest date named BEFORE the hit: the bout it cancels.
    assert hit["last_date"] == "July 9, 2016"
    assert hit["last_event"] == "UFC 200"


def test_a_post_fight_revelation_is_kept_and_its_bout_date_survives():
    found = hi.hits(FIGHTER_PAGE)
    hit = by_text(found, "torn ACL")
    assert len(hit) == 1
    # {{dts}} is rewritten before templates are stripped, or the date of the
    # bout it describes would vanish from the context.
    assert "2017-01-28" in hit[0]["context"]


def test_a_record_row_is_one_line_with_its_date():
    rows = [h for h in hi.hits(FIGHTER_PAGE) if h["form"] == "record_row"]
    assert len(rows) == 1
    row = rows[0]["text"]
    for part in ("Loss", "5–1", "Ed Green", "TKO (knee injury)",
                 "UFC on Fox 23", "2017-01-28"):
        assert part in row
    assert rows[0]["last_date"] == "2017-01-28"


def test_a_row_with_no_injury_is_not_kept():
    rows = [h["text"] for h in hi.hits(FIGHTER_PAGE)
            if h["form"] == "record_row"]
    assert not any("Dan Evans" in r for r in rows)


def test_a_knee_strike_is_not_an_injury():
    assert not by_text(hi.hits(FIGHTER_PAGE), "knee to the head")


def test_reference_sections_are_not_read():
    assert not by_text(hi.hits(FIGHTER_PAGE), "broken hand")


def test_citation_titles_are_not_read_as_prose():
    # The ref's title also says "torn ACL"; only the sentence itself counts.
    assert len(by_text(hi.hits(FIGHTER_PAGE), "torn ACL")) == 1


@pytest.mark.parametrize("sentence", [
    "Smith withdrew due to a staph infection.",
    "Jones was hospitalized during his weight cut.",
    "He tested positive for COVID-19 and was removed from the card.",
    "Silva was forced out of the bout with a back problem.",
    "The bout was stopped by the ringside doctor due to a cut.",
    "Pérez underwent surgery on his shoulder.",
    "He suffered a torn meniscus in training.",
    "Lee could not continue after the second round.",
    "Kim was not cleared by the athletic commission's doctors.",
])
def test_the_injury_and_illness_phrasings_are_caught(sentence):
    assert hi.INJURY.search(sentence), sentence


@pytest.mark.parametrize("sentence", [
    "He won via knee to the body in the second round.",
    "Brown landed a spinning back elbow.",
    "She secured an armbar in the first round.",
])
def test_ordinary_fight_description_is_not_caught(sentence):
    assert not hi.INJURY.search(sentence), sentence


def test_date_templates_become_iso():
    assert hi.keep_dates("{{dts|2012|08|11}}") == "2012-08-11"
    assert hi.keep_dates("{{dts|2016|March|5}}") == "2016-03-05"
    assert hi.keep_dates("{{Start date|2019|7|6|df=y}}") == "2019-07-06"


@pytest.mark.parametrize("text,event", [
    ("He faced Jones at UFC 196 in Las Vegas.", "UFC 196"),
    ("at UFC Fight Night 81.", "UFC Fight Night 81"),
    ("at UFC on Fox 23 in Denver.", "UFC on Fox 23"),
    ("at UFC Fight Night: Holloway vs. Kattar.",
     "UFC Fight Night: Holloway vs. Kattar"),
])
def test_event_hints(text, event):
    assert hi.events_in(text)[-1] == event


# --- whose page is it -------------------------------------------------------

def test_fighter_titles_try_the_specific_forms_first():
    assert hi.fighter_candidates("Michael Johnson") == [
        "Michael Johnson (fighter)", "Michael Johnson (mixed martial artist)",
        "Michael Johnson"]


def test_a_page_naming_the_fighters_opponents_is_theirs():
    page = "He has fought Tony Ferguson, Nate Diaz and Beneil Dariush."
    assert hi.opponent_hits(page, ["Tony Ferguson", "Nate Diaz",
                                   "Clay Guida"]) == 2
    assert hi.opponent_hits("An American sprinter.", ["Tony Ferguson"]) == 0


def test_resolution_prefers_the_page_that_names_the_opponents(monkeypatch):
    sprinter = "Michael Johnson is an American sprinter and mixed martial arts fan."
    fighter = ("Michael Johnson is an American mixed martial artist. He beat "
               "Tony Ferguson and lost to Nate Diaz.")
    monkeypatch.setattr(hi, "existing", lambda titles: {
        "Michael Johnson (fighter)": "Michael Johnson (fighter)",
        "Michael Johnson": "Michael Johnson"})
    monkeypatch.setattr(hi, "fetch_pages", lambda titles: {
        "Michael Johnson (fighter)": fighter, "Michael Johnson": sprinter})
    monkeypatch.setattr(hi, "search_fighter", lambda q: {})
    got = hi.resolve_fighters(["Michael Johnson"],
                              {"Michael Johnson": {"Tony Ferguson",
                                                   "Nate Diaz"}},
                              verbose=False)
    assert got["Michael Johnson"][0] == "Michael Johnson (fighter)"


def test_a_search_hit_needs_two_opponents_named(monkeypatch):
    monkeypatch.setattr(hi, "existing", lambda titles: {})
    monkeypatch.setattr(hi, "fetch_pages", lambda titles: {})
    monkeypatch.setattr(hi, "search_fighter", lambda q: {
        "Someone Else": "A mixed martial artist who once fought a Ferguson."})
    got = hi.resolve_fighters(["Carl Brown"],
                              {"Carl Brown": {"Tony Ferguson", "Nate Diaz"}},
                              verbose=False)
    assert got == {}


def test_existing_follows_redirects(monkeypatch):
    monkeypatch.setattr(hi, "get", lambda params: {"query": {
        "redirects": [{"from": "Jose Aldo", "to": "José Aldo"}],
        "pages": {"1": {"title": "José Aldo"},
                  "-1": {"title": "Nobody Here", "missing": ""}}}})
    assert hi.existing(["Jose Aldo", "Nobody Here"]) == {
        "Jose Aldo": "José Aldo"}


def test_fetch_pages_follows_continuation(monkeypatch):
    calls = []

    def fake_get(params):
        calls.append(params)
        if "rvcontinue" not in params:
            return {"continue": {"rvcontinue": "2|99", "continue": "||"},
                    "query": {"pages": {
                        "1": {"title": "A", "revisions": [
                            {"slots": {"main": {"*": "text A"}}}]},
                        "2": {"title": "B"}}}}
        return {"query": {"pages": {
            "1": {"title": "A"},
            "2": {"title": "B", "revisions": [
                {"slots": {"main": {"*": "text B"}}}]}}}}

    monkeypatch.setattr(hi, "get", fake_get)
    assert hi.fetch_pages(["A", "B"]) == {"A": "text A", "B": "text B"}
    assert len(calls) == 2


def test_records_carry_a_stable_id_and_their_page():
    first = hi.records_for("fighter", "Carl Brown", "", "Carl Brown",
                           FIGHTER_PAGE)
    again = hi.records_for("fighter", "Carl Brown", "", "Carl Brown",
                           FIGHTER_PAGE)
    assert [r["id"] for r in first] == [r["id"] for r in again]
    assert len({r["id"] for r in first}) == len(first)
    assert all(r["page"] == "Carl Brown" for r in first)


# --- the first full harvest's failures, pinned ------------------------------

TEMPLATE_RECORD = """==Mixed martial arts record==
{{MMA record start}}
|-
|{{no2}}Loss
|align=center|12–4
|Tom Nolan
|TKO (shoulder injury)
|UFC 250
|{{dts|2020|06|06}}
|align=center|1
|align=center|2:13
|Las Vegas, Nevada, United States
|
|-
|{{yes2}}Win
|align=center|12–3
|Dan Evans
|Decision (unanimous)
|UFC 240
|{{dts|2019|07|27}}
|align=center|3
|align=center|5:00
|Edmonton, Alberta, Canada
|
{{end}}
"""


def test_the_mma_record_template_is_read_as_rows():
    found = hi.hits(TEMPLATE_RECORD)
    rows = [h for h in found if h["form"] == "record_row"]
    assert len(rows) == 1
    for part in ("Loss", "Tom Nolan", "TKO (shoulder injury)", "UFC 250",
                 "2020-06-06"):
        assert part in rows[0]["text"]
    # And no cell leaks out as a "sentence" of its own.
    assert not [h for h in found if h["form"] == "prose"]


@pytest.mark.parametrize("title", [
    "2025 in UFC", "List of Canadian UFC fighters",
    "List of deaths due to the COVID-19 pandemic",
    "The Ultimate Fighter: China", "UFC 200",
    "List of male mixed martial artists"])
def test_pages_about_many_fighters_are_nobodys_page(title):
    assert not hi.mma_page("a mixed martial artist", title)


@pytest.mark.parametrize("fighter,title,expected", [
    ("Aaron Wilkinson", "Michael Johnson (fighter)", False),
    ("Aliaskhab Khizriev", "Abusupiyan Magomedov", False),
    ("Alex Reyes", "Alex Caceres", False),
    ("Joe Brammer", "Aaron Riley", False),
    ("Alatengheili", "Alateng Heili", True),
    ("Abus Magomedov", "Abusupiyan Magomedov", True),
    ("Ariane da Silva", "Ariane Lipski da Silva", True),
    ("AJ Fletcher", "AJ Fletcher", True),
    ("Bruno Silva", "Bruno Silva (fighter, born 1989)", True),
    ("Carlos Silva", "Bruno Silva (fighter, born 1989)", False),
])
def test_a_search_result_must_carry_the_fighters_name(fighter, title,
                                                        expected):
    assert hi.title_matches(fighter, title) is expected


def test_an_opponents_page_is_not_taken_for_the_fighters(monkeypatch):
    # The page names both of Wilkinson's opponents - because it is one of
    # them - and the first harvest took it.
    monkeypatch.setattr(hi, "existing", lambda titles: {})
    monkeypatch.setattr(hi, "fetch_pages", lambda titles: {})
    monkeypatch.setattr(hi, "search_fighter", lambda q: {
        "Michael Johnson (fighter)": "A mixed martial artist who fought "
                                     "Tony Ferguson and Nate Diaz."})
    got = hi.resolve_fighters(["Aaron Wilkinson"],
                              {"Aaron Wilkinson": {"Tony Ferguson",
                                                   "Nate Diaz"}},
                              verbose=False)
    assert got == {}
