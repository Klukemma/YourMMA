"""Tests for provider selection.

The engine was hardcoded to one odds service. When the key turned out not to
belong to it, the tempting move was to guess which provider it WAS and rewrite
for that - the same move that once produced a matcher pricing the wrong
fighter. These tests are mostly about refusing to let a guess pass as a fact.
"""

import os
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import odds_provider as op


def test_the_default_is_the_one_provider_whose_shape_was_actually_seen():
    assert op.DEFAULT == "the-odds-api"
    assert op.PROVIDERS[op.DEFAULT]["verified"] is True


def test_an_unknown_provider_names_the_real_choices(monkeypatch):
    """A typo must not silently fall back to the default and call the wrong
    service with a key that belongs elsewhere."""
    monkeypatch.setenv("ODDS_PROVIDER", "nonsense")
    with pytest.raises(KeyError, match="the-odds-api"):
        op.selected()


def test_an_empty_setting_falls_back_rather_than_failing(monkeypatch):
    monkeypatch.setenv("ODDS_PROVIDER", "  ")
    assert op.name() == op.DEFAULT


def test_the_key_travels_the_way_each_provider_wants_it():
    url, params, headers = op.request_for("K", op.PROVIDERS["the-odds-api"])
    assert params["apiKey"] == "K" and not headers

    bearer = {"base": "https://x/y", "auth": ("bearer", "Authorization"),
              "sport": None, "extras": {}, "paths": {}, "verified": False}
    _, params, headers = op.request_for("K", bearer)
    assert headers["Authorization"] == "Bearer K"
    assert "K" not in str(params)

    header = {"base": "https://x/y", "auth": ("header", "x-api-key"),
              "sport": None, "extras": {}, "paths": {}, "verified": False}
    _, params, headers = op.request_for("K", header)
    assert headers["x-api-key"] == "K"


def test_the_sport_parameter_is_only_sent_when_the_provider_needs_one():
    _, params, _ = op.request_for("K", op.PROVIDERS["the-odds-api"])
    assert "sport" not in params      # it is in the URL for this one
    _, params, _ = op.request_for("K", op.PROVIDERS["odds-api-io"])
    assert params["sport"] == "mma"


def test_an_unverified_provider_says_so_out_loud():
    """It will parse to nothing and show a card with no prices, which looks
    exactly like a quiet day at the bookmakers."""
    said = []
    fired = op.warn_if_unverified(op.PROVIDERS["odds-api-io"], echo=said.append)
    assert fired is True
    assert "UNVERIFIED" in said[0]


def test_a_verified_provider_stays_quiet():
    said = []
    assert op.warn_if_unverified(op.PROVIDERS["the-odds-api"],
                                 echo=said.append) is False
    assert said == []


def test_no_provider_claims_to_be_verified_without_field_paths():
    """verified=True asserts a real reply was read. Empty paths prove it was
    not, and the flag would then be a claim nobody checked."""
    for provider_name, provider in op.PROVIDERS.items():
        if provider["verified"]:
            assert provider["paths"], f"{provider_name} claims verified but " \
                                      "carries no field paths"
