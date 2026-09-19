"""Tests for the odds-key check.

The API cannot be reached from the sandbox this was written in, so every
response it can return is constructed here instead. That is the point: the
value of the check is what it SAYS when something is wrong, and the two cases
worth getting right - a rejected key and a spent quota - are exactly the two
nobody can reproduce on demand.
"""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import check_odds


def test_a_good_response_reports_the_card_and_the_quota():
    payload = [{
        "home_team": "Alex Pereira", "away_team": "Magomed Ankalaev",
        "bookmakers": [
            {"title": "DraftKings", "markets": [{"key": "h2h"}]},
            {"title": "FanDuel", "markets": [{"key": "h2h"}]},
        ],
    }]
    result = check_odds.describe(
        200, {"x-requests-remaining": "487", "x-requests-used": "13"}, payload)
    assert result["events"] == 1
    assert result["books"] == 2
    assert result["remaining"] == "487"
    assert result["fighters"] == ["Alex Pereira vs Magomed Ankalaev"]
    assert "problem" not in result


def test_a_rejected_key_says_so_in_words_someone_can_act_on():
    result = check_odds.describe(401, {}, {})
    assert "rejected" in result["problem"].lower()


def test_a_spent_quota_is_distinguished_from_a_bad_key():
    """Both leave the card with no prices, and the fixes are unrelated."""
    result = check_odds.describe(429, {}, {})
    assert "quota" in result["problem"].lower()
    assert "resets" in result["problem"].lower()


def test_a_working_key_with_no_upcoming_card_is_not_reported_as_a_fault():
    """Between events there is genuinely nothing priced. Calling that a
    failure would send someone hunting a bug that is not there."""
    result = check_odds.describe(
        200, {"x-requests-remaining": "500"}, [])
    assert result["events"] == 0
    assert "normal" in result["problem"].lower()


def test_an_unexpected_status_is_carried_through_rather_than_swallowed():
    result = check_odds.describe(503, {}, {})
    assert "503" in result["problem"]


def test_the_request_costs_one_credit():
    """A credit is markets x regions. Asking for three markets across two
    regions would cost six, turning the free 500 into 83 calls."""
    assert check_odds.PARAMS["regions"] == "us"
    assert check_odds.PARAMS["markets"] == "h2h"
    assert len(check_odds.PARAMS["regions"].split(",")) == 1
    assert len(check_odds.PARAMS["markets"].split(",")) == 1


def test_the_check_asks_for_the_same_thing_the_engine_does():
    """A check that verifies a different request than the one the engine makes
    can pass while the engine still gets nothing."""
    source = (ENGINE / "predict_card.py").read_text()
    block = source[source.index("def fetch_mma_odds"):]
    block = block[:block.index("try:")]
    for key, value in check_odds.PARAMS.items():
        assert f"'{key}': '{value}'" in block, (key, value)
    assert check_odds.ENDPOINT in block


def test_no_key_exits_with_instructions_rather_than_a_traceback(monkeypatch,
                                                                capsys):
    monkeypatch.setenv("ODDS_API_KEY", "")
    assert check_odds.main() == 2
    printed = capsys.readouterr().out
    assert "the-odds-api.com" in printed
    assert "Secrets and variables" in printed


def test_the_key_is_never_printed_in_an_error():
    """requests puts the whole URL, query string included, into its exception
    text. A connection failure printed the key in clear."""
    error = ("HTTPSConnectionPool(host='api.the-odds-api.com', port=443): "
             "Max retries exceeded with url: /v4/sports/x/odds"
             "?apiKey=EXAMPLE-KEY-NOT-REAL&regions=us (Caused by ProxyError)")
    cleaned = check_odds.redact(error, "EXAMPLE-KEY-NOT-REAL")
    assert "EXAMPLE-KEY-NOT-REAL" not in cleaned
    assert "<redacted>" in cleaned
    assert "ProxyError" in cleaned, "the useful part must survive"


def test_redacting_without_a_key_changes_nothing():
    assert check_odds.redact("plain message", "") == "plain message"
    assert check_odds.redact("plain message", None) == "plain message"


def test_the_engine_redacts_on_its_network_path_too():
    """check_odds is not the only thing that calls the API."""
    source = (ENGINE / "predict_card.py").read_text()
    block = source[source.index("def fetch_mma_odds"):]
    block = block[:block.index("def match_fighter_to_odds")]
    assert "<redacted>" in block, "predict_card prints the raw exception"


def test_a_rejected_key_triggers_a_probe_rather_than_a_dead_end():
    """Several services ship under near-identical names and issue different
    key shapes. A 401 says the key is wrong HERE, not that it is wrong."""
    result = check_odds.describe(401, {}, {})
    assert result.get("probe") is True
    assert "similar names" in result["problem"]


def test_the_probe_covers_both_auth_styles_and_the_lookalike_services():
    """A key passed the wrong way looks exactly like a key that is invalid."""
    labels = [p[0] for p in check_odds.PROBES]
    styles = {p[2] for p in check_odds.PROBES}
    assert styles == {"query", "header"}
    joined = " ".join(labels)
    for service in ("the-odds-api.com", "odds-api.io", "theoddsapi.com"):
        assert service in joined


def test_the_probe_never_prints_the_key_even_when_a_request_explodes():
    """The probe touches four hosts, so it has four more chances to leak."""
    source = (ENGINE / "check_odds.py").read_text()
    block = source[source.index("def probe("):source.index("def describe(")]
    assert "redact(error, key)" in block, "probe prints a raw exception"


def test_a_probe_result_names_the_service_that_accepted_the_key(capsys):
    check_odds.report({
        "status": 401, "remaining": None, "used": None, "events": 0,
        "books": 0, "fighters": [], "problem": "rejected here",
        "probes": [("the-odds-api.com (query param)", 401),
                   ("odds-api.io (query param)", 200)],
    })
    printed = capsys.readouterr().out
    assert "THIS ONE" in printed
    assert "odds-api.io" in printed


def test_a_key_that_works_nowhere_says_what_to_check(capsys):
    check_odds.report({
        "status": 401, "remaining": None, "used": None, "events": 0,
        "books": 0, "fighters": [], "problem": "rejected here",
        "probes": [("the-odds-api.com (query param)", 401),
                   ("odds-api.io (query param)", 401)],
    })
    printed = capsys.readouterr().out
    assert "None of them" in printed
    assert "confirmed" in printed or "newline" in printed
