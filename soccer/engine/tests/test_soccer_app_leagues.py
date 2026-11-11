"""The league picker is in the app, remembers its choice on the device, and
the app still reads every field the five-league data had."""

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
JS = (APP / "app.js").read_text()


def test_the_picker_is_shared_by_three_tabs_and_remembered_per_device():
    assert 'const LEAGUES_KEY = "yoursoccer.leagues"' in JS
    assert "DESK.store" in JS
    # Matches, Tables and Record each draw the same picker.
    assert len(re.findall(r'picker\(w, choice, "(matches|tables|record)"', JS)) == 3


def test_the_app_reads_the_new_league_blocks_and_still_reads_the_old_fields():
    for field in ("league_order", "leagues", "groups", "bands"):
        assert re.search(rf"\b(d|data|M|MATCHES|w)\.{field}\b|\[\"{field}\"\]|\b{field}\b", JS), field
    # The five-league payload's fields are still read, so today's files draw as before.
    for field in ("caveats", "matches", "results", "season_label", "prices_at", "data_through"):
        assert field in JS, field


def test_no_pick_until_priced_is_said_in_words():
    assert "No pick until the price is posted" in JS or "no pick until" in JS.lower()
