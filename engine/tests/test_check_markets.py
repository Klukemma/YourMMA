"""The market probe must never print the key, and must not lie about absence.

It spends a credit to answer one question: is there a distance market on this
plan? A probe that reports "available" for an endpoint that accepted the
request but returned nothing would send us building a wrongness layer on a
market nobody prices - which is the same mistake as reading a public
endpoint's 200 as proof a key works, made once already in check_odds.py.
"""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import check_markets as cm


def test_the_key_never_survives_redaction():
    key = "f172d86a5026fa0498c259469b3ee8ec"
    body = f"error: bad apiKey={key} on request"
    assert key not in cm.redact(body, key)
    assert "<redacted>" in cm.redact(body, key)


def test_a_url_encoded_key_is_redacted_too():
    key = "abc/def+ghi"
    import urllib.parse
    body = f"failed at ?apiKey={urllib.parse.quote(key)}"
    assert urllib.parse.quote(key) not in cm.redact(body, key)


def test_redaction_survives_an_empty_key():
    assert cm.redact("nothing to hide", "") == "nothing to hide"
    assert cm.redact("nothing to hide", None) == "nothing to hide"


def test_the_moneyline_is_asked_as_a_control():
    """Without it, "the others returned nothing" cannot be told apart from
    "the request shape is wrong"."""
    assert cm.MARKETS[0][0] == "h2h"


def test_the_distance_market_is_among_those_asked():
    names = {m for m, _ in cm.MARKETS}
    assert names & {"fight_goes_the_distance", "fight_to_go_distance", "totals"}


def test_every_market_carries_a_reason():
    for market, why in cm.MARKETS:
        assert why and len(why) > 8, market


def test_refuses_without_a_key_rather_than_calling_anonymously():
    import os
    saved = os.environ.pop("ODDS_API_KEY", None)
    try:
        with pytest.raises(SystemExit):
            cm.main()
    finally:
        if saved is not None:
            os.environ["ODDS_API_KEY"] = saved
