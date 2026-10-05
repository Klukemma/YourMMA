"""Historical intel is kept only when provably published before the fight.

The failure this exists to prevent is not a crash. It is "X reveals he fought
with a torn ACL", published the week after X lost, being filed as pre-fight
injury news. Losers disclose far more than winners, so every such record
teaches the model that injury news predicts losing - perfectly, in testing,
and not at all live.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import intel_backfill as bf
import verify_intel_dates as vd


def record(**over):
    base = {"fighter": "Raquel Pennington", "event_date": "2019-07-20",
            "event_name": "UFC Fight Night: Dos Anjos vs. Edwards",
            "kind": "injury", "confidence": "reported",
            "note": "reported knee trouble in the final weeks of camp",
            "source": "https://www.bloodyelbow.com/2019/07/12/story",
            "gathered": "2026-09-29T10:00:00Z"}
    base.update(over)
    return base


# --- dates read from the URL, never from a summary -------------------------

def test_a_full_date_in_the_path_pins_one_day():
    assert bf.url_date("https://x.com/2019/07/12/story") == \
        (date(2019, 7, 12), date(2019, 7, 12))


def test_a_hyphenated_date_is_read():
    assert bf.url_date("https://x.com/news/2019-07-12-story") == \
        (date(2019, 7, 12), date(2019, 7, 12))


def test_a_month_path_spans_the_whole_month():
    assert bf.url_date("https://mmajunkie.usatoday.com/2019/07/story") == \
        (date(2019, 7, 1), date(2019, 7, 31))


def test_a_url_with_no_date_gives_none():
    assert bf.url_date("https://www.sherdog.com/news/news/Story-123456") is None


def test_an_impossible_date_is_not_read_as_one():
    assert bf.url_date("https://x.com/2019/13/45/story") is None


# --- the verdict --------------------------------------------------------------

def test_a_story_from_the_week_before_is_accepted():
    outcome, how, published = bf.verdict(record())
    assert (outcome, how, published) == ("accept", "url", date(2019, 7, 12))


def test_a_story_from_the_week_after_is_rejected():
    """The whole reason this module exists."""
    outcome, reason, _ = bf.verdict(
        record(source="https://x.com/2019/07/25/reveals-he-fought-injured"))
    assert outcome == "reject" and "after the fight" in reason


def test_a_story_from_fight_day_is_rejected():
    """An American fight night's results carry the fight's own date."""
    outcome, _, _ = bf.verdict(record(source="https://x.com/2019/07/20/story"))
    assert outcome == "reject"


def test_the_day_before_is_accepted():
    """Weigh-in day. Legitimately pre-fight, and the market has it too."""
    outcome, _, _ = bf.verdict(record(source="https://x.com/2019/07/19/story"))
    assert outcome == "accept"


def test_a_month_path_wholly_before_the_event_is_accepted():
    outcome, how, _ = bf.verdict(record(source="https://x.com/2019/06/story"))
    assert (outcome, how) == ("accept", "url")


def test_a_month_path_containing_the_event_waits_for_the_page():
    """A July URL for a 20 July fight could be the 12th or the 25th."""
    outcome, _, _ = bf.verdict(record(source="https://x.com/2019/07/story"))
    assert outcome == "pending"


def test_a_month_path_after_the_event_is_rejected_outright():
    outcome, _, _ = bf.verdict(record(source="https://x.com/2019/08/story"))
    assert outcome == "reject"


def test_no_date_anywhere_waits_for_the_page():
    outcome, _, _ = bf.verdict(
        record(source="https://www.sherdog.com/news/news/Story-123456"))
    assert outcome == "pending"


def test_a_claimed_date_alone_verifies_nothing():
    """The researcher's date is exactly what a model gets confidently wrong."""
    outcome, _, _ = bf.verdict(record(
        source="https://www.sherdog.com/news/news/Story-123456",
        published_claimed="2019-07-10"))
    assert outcome == "pending"


def test_a_claimed_date_that_disagrees_with_the_url_is_rejected():
    """If the summary has the date wrong, it may have the content wrong."""
    outcome, reason, _ = bf.verdict(record(published_claimed="2019-06-01"))
    assert outcome == "reject" and "claimed" in reason


def test_a_page_date_decides_what_the_url_could_not():
    outcome, how, published = bf.verdict(
        record(source="https://x.com/2019/07/story"), verified="2019-07-15")
    assert (outcome, how, published) == ("accept", "page", date(2019, 7, 15))


def test_a_page_date_after_the_fight_is_rejected():
    outcome, _, _ = bf.verdict(
        record(source="https://x.com/2019/07/story"), verified="2019-07-22")
    assert outcome == "reject"


# --- field validation is the live scout's -----------------------------------

def test_no_source_is_rejected_before_any_date_is_read():
    accepted, pending, rejected = bf.classify([record(source="")])
    assert not accepted and not pending and "source" in rejected[0].lower()


def test_an_invented_kind_is_rejected():
    _, _, rejected = bf.classify([record(kind="vibes")])
    assert "unknown kind" in rejected[0]


def test_classify_splits_the_three_ways():
    accepted, pending, rejected = bf.classify([
        record(),
        record(fighter="Irene Aldana",
               source="https://www.sherdog.com/news/news/Story-1"),
        record(fighter="Other Fighter",
               source="https://x.com/2019/07/28/after"),
    ])
    assert (len(accepted), len(pending), len(rejected)) == (1, 1, 1)
    assert accepted[0]["verified_by"] == "url"


# --- the sample and the log ------------------------------------------------

@pytest.fixture
def archive():
    rows = []
    for year in (2008, 2012, 2016, 2020, 2024):
        for i in range(10):
            rows.append({"date": f"{year}-06-{i + 1:02d}",
                         "event_name": f"UFC {year}-{i}",
                         "r_name": f"Red {year} {i}",
                         "b_name": f"Blue {year} {i}"})
    return pd.DataFrame(rows)


def test_the_sample_is_reproducible(archive):
    assert bf.sample_fights(archive, 5, seed=3) == \
        bf.sample_fights(archive, 5, seed=3)


def test_a_different_seed_draws_a_different_sample(archive):
    assert bf.sample_fights(archive, 5, seed=3) != \
        bf.sample_fights(archive, 5, seed=4)


def test_the_sample_starts_in_2010(archive):
    fights = bf.sample_fights(archive, 40, seed=0)
    assert all(f["event_date"] >= "2010" for f in fights)
    assert len(fights) == 40


def test_the_brief_covers_both_corners(archive):
    plan = bf.brief(bf.sample_fights(archive, 1, seed=0))
    assert len(plan) == 2
    assert plan[0]["fighter"] == plan[1]["opponent"]


def test_every_searched_fighter_is_logged_even_with_nothing_found(tmp_path,
                                                                  monkeypatch):
    """The empty searches are the denominator. Without them there is no
    finding rate and no control group."""
    monkeypatch.setattr(bf, "SCOUTED", tmp_path / "s.csv")
    monkeypatch.setattr(bf, "PENDING", tmp_path / "p.csv")
    monkeypatch.setattr(bf, "LOG", tmp_path / "l.csv")
    result = bf.ingest({
        "searched": [{"event_date": "2019-07-20", "fighter": "Raquel Pennington"},
                     {"event_date": "2019-07-20", "fighter": "Irene Aldana"}],
        "findings": [record()],
    })
    log = pd.read_csv(tmp_path / "l.csv")
    assert result["accepted"] == 1 and len(log) == 2
    clean = log[log.fighter == "Irene Aldana"].iloc[0]
    assert clean.found == 0 and clean.accepted == 0


def test_ingesting_twice_does_not_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "SCOUTED", tmp_path / "s.csv")
    monkeypatch.setattr(bf, "PENDING", tmp_path / "p.csv")
    monkeypatch.setattr(bf, "LOG", tmp_path / "l.csv")
    payload = {"searched": [{"event_date": "2019-07-20",
                             "fighter": "Raquel Pennington"}],
               "findings": [record()]}
    bf.ingest(payload)
    second = bf.ingest(payload)
    assert second["accepted"] == 0 and second["logged"] == 0


# --- reading a page's own publication date ---------------------------------

def test_article_published_time_is_read():
    html = '<meta property="article:published_time" content="2019-07-12T09:00:00Z">'
    assert vd.published_from_html(html) == date(2019, 7, 12)


def test_attribute_order_does_not_matter():
    html = '<meta content="2019-07-12" name="pubdate">'
    assert vd.published_from_html(html) == date(2019, 7, 12)


def test_json_ld_date_published_is_read():
    html = '<script type="application/ld+json">{"datePublished": "2019-07-12"}</script>'
    assert vd.published_from_html(html) == date(2019, 7, 12)


def test_a_modified_date_is_never_taken_for_a_published_one():
    """An article updated after the fight carries a post-fight modified date.
    Reading it would reject stories that were genuinely pre-fight."""
    html = ('<meta property="article:modified_time" content="2019-07-25">'
            '<meta property="article:published_time" content="2019-07-12">')
    assert vd.published_from_html(html) == date(2019, 7, 12)


def test_a_page_with_only_a_modified_date_declares_nothing():
    html = '<meta property="article:modified_time" content="2019-07-25">'
    assert vd.published_from_html(html) is None


def test_settle_accepts_rejects_and_keeps_unfetchable_pending():
    rows = [record(source="https://a.com/x"),
            record(fighter="Irene Aldana", source="https://b.com/y"),
            record(fighter="Third Person", source="https://c.com/z"),
            record(fighter="Fourth Person", source="https://d.com/w")]
    pages = {
        "https://a.com/x": '<meta property="article:published_time" content="2019-07-10">',
        "https://b.com/y": '<meta property="article:published_time" content="2019-07-24">',
        "https://c.com/z": "<html>no date here</html>",
    }

    def fetcher(url):
        if url not in pages:
            raise OSError("timeout")
        return pages[url]

    accepted, still, rejected = vd.settle(rows, fetcher=fetcher, pause=0)
    assert [r["fighter"] for r in accepted] == ["Raquel Pennington"]
    assert accepted[0]["verified_by"] == "page"
    assert [r["fighter"] for r in still] == ["Fourth Person"]
    assert len(rejected) == 2
