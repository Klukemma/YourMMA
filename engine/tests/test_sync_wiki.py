"""sync_wiki: results from Wikipedia while the Kaggle archive lags.

Written against a real event page (tests/fixtures/wiki/UFC_300.wiki) and the
archive's own rows for that event, so "the bridge agrees with the archive" is
checked on a card both hold rather than asserted.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import pending_rows
import ratings
import sync_kaggle
import sync_wiki as sw
from name_resolution import norm_name

UFC_300 = ENGINE / "tests" / "fixtures" / "wiki" / "UFC_300.wiki"
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


@pytest.fixture(scope="module")
def ufc300():
    return sw.parse_event(UFC_300.read_text(), "UFC 300")


@pytest.fixture(scope="module")
def archive():
    a = pd.read_csv(ARCHIVE, low_memory=False, dtype=pending_rows.TEXT_COLUMNS)
    a["date"] = pd.to_datetime(a["date"])
    return a[~sw.is_provisional(a)].reset_index(drop=True)


# --- reading a page ----------------------------------------------------------

def test_reads_the_event_and_every_bout_in_page_order(ufc300):
    event, bouts = ufc300
    assert event["name"] == "UFC 300: Pereira vs. Hill"
    assert event["date"] == "2024-04-13"
    assert event["location"] == "Paradise, Nevada, USA"
    assert len(bouts) == 13
    assert (bouts[0]["first"], bouts[0]["second"]) == ("Alex Pereira (c)", "Jamahal Hill")
    assert [b["order"] for b in bouts] == list(range(13))


def test_title_and_scheduled_rounds(ufc300):
    _, bouts = ufc300
    by = {sw.ascii_name(b["first"]): b for b in bouts}
    assert sw.is_title(by["Alex Pereira"]) and sw.scheduled_rounds(by["Alex Pereira"]) == 5
    assert sw.is_title(by["Zhang Weili"])
    # Gaethje carries (c) for the symbolic BMF belt: five rounds, no title.
    assert not sw.is_title(by["Max Holloway"])
    assert sw.scheduled_rounds(by["Max Holloway"]) == 5
    assert sw.scheduled_rounds(by["Arman Tsarukyan"]) == 3


def test_a_bout_not_yet_fought_is_left_out():
    page = """{{Infobox MMA event|name=UFC 999|promotion=[[Ultimate Fighting Championship]]
|date={{start date|2030|01|05}}}}
{{MMAevent card|Main card}}
{{MMAevent bout|Lightweight|[[A One]]|vs.|[[B Two]]|||||}}
{{MMAevent bout|Lightweight|[[C Three]]|def.|[[D Four]]|KO (punch)|1|0:30|}}
{{MMAevent card|Exhibition}}
{{MMAevent bout|Lightweight|[[E Five]]|def.|[[F Six]]|KO (punch)|1|0:30|}}"""
    _, bouts = sw.parse_event(page)
    assert [(b["first"], b["second"]) for b in bouts] == [("C Three", "D Four")]


@pytest.mark.parametrize("text,label", [
    ("KO (punches)", "KO/TKO"),
    ("TKO (corner stoppage)", "KO/TKO"),
    ("TKO (doctor stoppage)", "TKO - Doctor's Stoppage"),
    ("Submission (rear-naked choke)", "Submission"),
    ("Technical Submission (arm-triangle choke)", "Submission"),
    ("Decision (unanimous) (30–27, 30–27, 30–27)", "Decision - Unanimous"),
    ("Decision (split) (28–29, 29–28, 29–28)", "Decision - Split"),
    ("Draw (majority) (29–28, 28–28, 28–28)", "Decision - Majority"),
    ("No Contest (accidental eye poke)", "Could Not Continue"),
    ("No Contest (overturned)", "Overturned"),
    ("DQ (illegal knee)", "DQ"),
])
def test_methods_read_as_the_archive_labels_them(text, label):
    assert sw.method_label(text) == label


def test_time_is_seconds_into_the_final_round():
    assert sw.seconds("4:25") == 265.0
    assert sw.seconds("5:00") == 300.0
    assert np.isnan(sw.seconds(""))


def test_divisions():
    assert sw.division_of("Women's Strawweight") == "women's strawweight"
    assert sw.division_of("Women’s Flyweight") == "women's flyweight"
    assert sw.division_of("Light Heavyweight") == "light heavyweight"
    assert sw.division_of("Catchweight (147.5 lb)") is None


def test_a_catchweight_is_filed_where_ufcstats_files_it():
    # The fighters' own division when their last bouts agree...
    assert sw.catchweight_division("Catchweight (128 lb)", ["flyweight", "flyweight"]) == "flyweight"
    # ...otherwise the heaviest division at or under the agreed weight.
    assert sw.catchweight_division("Catchweight (147.5 lb)", ["featherweight", None]) == "featherweight"
    assert sw.catchweight_division("Catchweight (130 lb)",
                                   ["women's flyweight", None]) == "women's flyweight"
    assert sw.catchweight_division("Catchweight", [None, None]) == "catch weight"


# --- names -------------------------------------------------------------------

def _tiny_archive():
    rows = [("2024-01-06", "Diego Ferreira", "d1", "Rafa Garcia", "g1", "lightweight"),
            ("2024-03-02", "King Green", "k1", "Jim Miller", "m1", "lightweight"),
            ("2024-05-04", "Bruno Silva", "s1", "Ana Lee", "a1", "flyweight"),
            ("2024-05-04", "Bruno Silva", "s2", "Joe Hill", "h1", "middleweight")]
    return pd.DataFrame([{"date": d, "r_name": r, "r_id": ri, "b_name": b, "b_id": bi,
                          "division": div} for d, r, ri, b, bi, div in rows])


def test_an_alias_is_learned_from_a_bout_both_sources_hold():
    archive = _tiny_archive()
    wiki = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-06"), "winner": "Carlos Diego Ferreira", "loser": "Rafa García"},
        {"date": pd.Timestamp("2024-03-02"), "winner": "Bobby Green", "loser": "Jim Miller"}])
    aliases = sw.NameBridge.learn(archive, wiki)
    assert aliases == {"carlos diego ferreira": "Diego Ferreira", "bobby green": "King Green"}


def test_resolution_order():
    archive = _tiny_archive()
    names = sw.NameBridge(archive, pd.DataFrame([
        {"date": pd.Timestamp("2024-03-02"), "winner": "Bobby Green", "loser": "Jim Miller"}]),
        active_since=pd.Timestamp("2020-01-01"))
    assert names.resolve("Jim Miller", "lightweight")[:2] == ("Jim Miller", "m1")
    assert names.resolve("Bobby Green", "lightweight") == ("King Green", "k1", "learned")
    assert names.resolve("Diego Ferreira Jr.", "lightweight") == ("Diego Ferreira", "d1", "subset")
    assert names.resolve("Brand New", "lightweight") == (None, None, "new")
    # Two fighters share the name: the division decides, and none decides nothing.
    assert names.resolve("Bruno Silva", "middleweight")[1] == "s2"
    assert names.resolve("Bruno Silva", None) == (None, None, "ambiguous")


# --- rows --------------------------------------------------------------------

def test_replaying_ufc_300_reproduces_the_archive(archive, ufc300):
    """The bridge, given the archive as it stood the day before UFC 300 and the
    event's page, agrees with the archive's own rows: same fighters, same
    winners, same method, round and time - and the pre-fight ratings a Kaggle
    sync would have stored for them (ratings.extend over the same rows; the
    2024 rows themselves predate the sync and came from a rating run that
    cannot be reproduced bit for bit, see ratings.py)."""
    before = archive[archive["date"] < "2024-04-13"].reset_index(drop=True)
    truth = archive[archive["date"] == "2024-04-13"]
    rated = ratings.extend(before, truth.drop(columns=[c for c in truth.columns
                                                       if c.endswith(("mu_pre", "sigma_pre", "mmr_pre"))]))
    truth = rated[rated["date"] == "2024-04-13"]
    wiki = sw.wiki_history()
    names = sw.NameBridge(before, wiki[wiki["date"] < "2024-04-13"])
    rows, report = sw.bridge_rows(before, [ufc300], names)
    assert len(rows) == len(truth) == 13
    # Bobby Green is "King Green" in the archive, learned from earlier cards;
    # Kayla Harrison made her UFC debut here.
    assert report["new"] == ["2024-04-13 Kayla Harrison (Women's Bantamweight)"]
    assert report["skipped"] == []
    assert sw.is_provisional(rows).all()
    harrison = rows[(rows["r_name"] == "Kayla Harrison") | (rows["b_name"] == "Kayla Harrison")].iloc[0]
    assert sw.debut_id("Kayla Harrison") in (harrison["r_id"], harrison["b_id"])
    for _, r in rows.iterrows():
        known = [i for i in (r["r_id"], r["b_id"]) if not i.startswith(sw.PROVISIONAL)]
        t = truth[truth["r_id"].isin(known) | truth["b_id"].isin(known)]
        assert len(t) == 1, (r["r_name"], r["b_name"])
        t = t.iloc[0]
        if not r["winner_id"].startswith(sw.PROVISIONAL):
            assert r["winner_id"] == t["winner_id"]
        assert r["winner"] in (r["r_name"], r["b_name"])
        assert r["method"] == t["method"]
        assert r["finish_round"] == t["finish_round"]
        assert r["match_time_sec"] == t["match_time_sec"]
        assert r["division"] == t["division"]
        assert r["total_rounds"] == t["total_rounds"]
        for c in "rb":
            other = "b" if c == "r" else "r"
            tc = "r" if (r[f"{c}_id"] == t["r_id"] or r[f"{other}_id"] == t["b_id"]) else "b"
            assert r[f"{c}_mu_pre"] == pytest.approx(t[f"{tc}_mu_pre"], abs=1e-9)
            assert r[f"{c}_sigma_pre"] == pytest.approx(t[f"{tc}_sigma_pre"], abs=1e-9)


def test_red_is_the_higher_rated_fighter_and_the_winner_stays(archive, ufc300):
    before = archive[archive["date"] < "2024-04-13"].reset_index(drop=True)
    rows, _ = sw.bridge_rows(before, [ufc300], sw.NameBridge(before))
    assert (rows["r_mmr_pre"] >= rows["b_mmr_pre"]).all()
    assert (rows["r_mmr_pre"] == rows["r_mu_pre"] - 3 * rows["r_sigma_pre"]).all()
    pereira = rows[(rows["r_name"] == "Alex Pereira") | (rows["b_name"] == "Alex Pereira")].iloc[0]
    assert pereira["winner"] == "Alex Pereira"


def test_merge_replaces_old_provisional_rows_and_is_repeatable(archive, ufc300):
    before = archive[archive["date"] < "2024-04-13"].reset_index(drop=True)
    names = sw.NameBridge(before)
    first, rows, _ = sw.merge(before, [ufc300], names)
    second, rows2, _ = sw.merge(first, [ufc300], names)
    assert len(first) == len(before) + 13 == len(second)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    # Rows already in the archive never move.
    pd.testing.assert_frame_equal(first.iloc[:len(before)].reset_index(drop=True),
                                  before.reset_index(drop=True), check_dtype=False)


def test_an_event_the_archive_already_holds_is_not_added_again(archive, ufc300):
    before = archive[archive["date"] <= "2024-04-13"].reset_index(drop=True)
    combined, rows, _ = sw.merge(before, [ufc300], sw.NameBridge(before))
    assert len(rows) == 0 and len(combined) == len(before)


# --- the Kaggle sync around them -------------------------------------------

def test_repair_never_fills_a_provisional_row():
    frame = pd.DataFrame({"fight_id": ["abc0123456789def", "wiki_0123456789ab"],
                          sync_kaggle.STAT_PROBE: [np.nan, np.nan]})
    assert sync_kaggle.blank_stat_rows(frame).tolist() == [True, False]


def test_the_kaggle_sync_drops_provisional_rows_before_it_merges():
    text = (ENGINE / "sync_kaggle.py").read_text()
    body = text[text.index("def cmd_sync"):text.index("def cmd_search_odds")]
    assert body.index("is_provisional") < body.index("cutoff = local['date'].max()")


def test_the_workflow_runs_the_wiki_sync_after_the_kaggle_one():
    text = (ENGINE.parent / ".github" / "workflows" / "update-dataset.yml").read_text()
    sync = text[text.rindex("python engine/sync_kaggle.py sync"):]
    assert "python engine/sync_wiki.py sync" in sync[:800]
    assert "tee -a sync.log" in sync[:800]


def test_fetch_keeps_finished_ufc_events_after_the_archive(monkeypatch):
    """Only events dated after the archive and before today: an event on
    today's date may be in progress, its page half filled in."""
    page = UFC_300.read_text()
    upcoming = page.replace("def.", "vs.")
    monkeypatch.setattr(sw, "discover", lambda years: ["UFC 300", "UFC 301"])
    import harvest_injuries
    monkeypatch.setattr(harvest_injuries, "fetch_pages",
                        lambda titles: {"UFC 300": page, "UFC 301": upcoming})
    got = sw.fetch_events(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-20"))
    assert [e["name"] for e, _ in got] == ["UFC 300: Pereira vs. Hill"]
    assert sw.fetch_events(pd.Timestamp("2024-04-13"), pd.Timestamp("2024-04-20")) == []
    assert sw.fetch_events(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-13")) == []
