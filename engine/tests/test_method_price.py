"""The METHOD ESTIMATOR (engine/method_price.py) and the card's method legs.

A live card has no method prop price, so tickets v2 price a prop leg by this
estimate. What must hold, because a reader cannot check any of it from the
phone:

  FROZEN       data/method_price.json's coefficients match their sha256, and
               are what the fit script fits on 2016-2024; its error table is
               what the script measures (written, never typed).
  DETERMINISM  the same inputs give the same estimate, fit after fit.
  NO FUTURE    the fit reads no result column, a yearly refit reads only
               earlier fights, and a card's records stop before its date.
  THE CARD     the card export adds each fight's six method legs and changes
               nothing else; a leg that cannot be estimated says why.

The card's export block is lifted out of predict_card rather than imported:
importing that module runs the whole pipeline (as test_median_through_cache
does).
"""

import ast
import copy
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
ROOT = ENGINE.parent
sys.path.insert(0, str(ENGINE))

import app_export as ax  # noqa: E402
import method_price as mp  # noqa: E402
import parlay_profiles as pp  # noqa: E402

SOURCE = (ENGINE / "predict_card.py").read_text()
DATA_ON_DISK = mp.LEGS.exists() and mp.ARCHIVE.exists()


# ---------------------------------------------------------------- fixtures

def _book(p_red, split_red, split_blue, overround=1.22, slope=0.85):
    """Six prop prices (decimal) for a fight whose market gives red `p_red`."""
    z = slope * math.log(p_red / (1 - p_red))
    q_red = 1 / (1 + math.exp(-z))
    out = {}
    for side, q, split in (("red", q_red, split_red), ("blue", 1 - q_red, split_blue)):
        for t, col in mp.PRICE_COLS.items():
            out[f"{col}_{side}"] = 1.0 / (q * split[t] * overround)
    return out


def _facts(rng):
    n = rng.randint(0, 20)
    ko, sub = rng.randint(0, n // 2), rng.randint(0, n // 3)
    kol, subl = rng.randint(0, max(0, n - ko - sub) // 2), rng.randint(0, max(0, n - ko - sub) // 3)
    return {"fights": n, "wins": ko + sub, "losses": kol + subl, "ko_wins": ko,
            "sub_wins": sub, "ko_losses": kol, "sub_losses": subl}


def synthetic_rows(years=range(2016, 2020), per_year=120, seed=3):
    """Fit rows with a known market: shares lean with the records and the
    division, so the ridge has something to find."""
    rng = random.Random(seed)
    rows = []
    for y in years:
        for i in range(per_year):
            p_red = rng.uniform(0.2, 0.8)
            fr, fb = _facts(rng), _facts(rng)
            div = rng.choice(mp.DIVISIONS + ("catch weight",))

            def split(f, o):
                ko = 0.3 + 0.3 * f["ko_wins"] / (f["fights"] + 3) + (0.15 if div == "heavyweight" else 0)
                sub = 0.15 + 0.3 * f["sub_wins"] / (f["fights"] + 3) + 0.1 * o["sub_losses"] / (o["fights"] + 3)
                dec = 0.55 + (0.1 if "women" in div else 0)
                s = ko + sub + dec
                return {"KO/TKO": ko / s, "Submission": sub / s, "Decision": dec / s}

            dec_red = 1 / (p_red * 1.04)
            dec_blue = 1 / ((1 - p_red) * 1.04)
            rows.append({"fight_id": f"f{y}{i:04d}", "date": f"{y}-{1 + i % 12:02d}-{1 + i % 27:02d}",
                         "dec_red": dec_red, "dec_blue": dec_blue,
                         **_book(p_red, split(fr, fb), split(fb, fr)),
                         "facts_red": fr, "facts_blue": fb, "division": div,
                         "rounds_scheduled": 5 if i % 9 == 0 else 3})
    rows.sort(key=lambda r: (r["date"], r["fight_id"]))
    return rows


DOC = {"dev_choice": {"alpha": 1.0, "prior_fights": 3.0}}


@pytest.fixture(scope="module")
def frozen_doc():
    if not mp.COEFFS.exists():
        pytest.skip("data/method_price.json not written")
    return json.loads(mp.COEFFS.read_text())


@pytest.fixture(scope="module")
def real_rows():
    if not DATA_ON_DISK:
        pytest.skip("parlay_legs.csv.gz or the archive not on disk")
    return mp.fit_rows()


# ---------------------------------------------------------------- frozen

def test_the_frozen_coefficients_match_their_hash(frozen_doc):
    assert mp.coefficient_hash(frozen_doc["frozen"]["model"]) == frozen_doc["frozen"]["sha256"]
    assert mp.load() == frozen_doc["frozen"]["model"]


def test_a_hand_edited_coefficient_is_refused(frozen_doc, tmp_path):
    """One digit changed and the file is no longer the fit that was measured."""
    doc = copy.deepcopy(frozen_doc)
    doc["frozen"]["model"]["weights"]["KO/TKO"][0] += 1e-9
    path = tmp_path / "method_price.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(mp.FrozenMismatch):
        mp.load(path)
    est = ax.method_estimator(path)
    assert est["model"] is None and "hash" in est["missing"]
    missing = ax.method_estimator(tmp_path / "absent.json")
    assert missing["model"] is None and "not been fitted" in missing["missing"]


def test_the_frozen_fit_is_fitted_on_2016_2024_at_the_dev_choice(frozen_doc):
    m = frozen_doc["frozen"]["model"]
    assert frozen_doc["frozen"]["fit"] == ["2016-01-01", "2024-12-31"]
    assert m["fit_first"] >= "2016-01-01" and m["fit_last"] <= "2024-12-31"
    assert (m["alpha"], m["prior_fights"]) == mp.chosen(frozen_doc)
    assert m["features"] == list(mp.FEATURES)
    assert frozen_doc["test"]["fit"] == ["2016-01-01", "2020-12-31"]
    assert frozen_doc["test"]["test"] == ["2021-01-01", "2024-12-31"]
    assert frozen_doc["dev_choice"]["check"] == ["2020-01-01", "2020-12-31"]


def test_the_file_is_what_the_script_writes(frozen_doc, real_rows):
    """Refit from the data on disk: the frozen estimates and the measured
    error table are the script's, to floating-point noise - nothing typed.
    A change to parlay_legs.csv.gz (an alias fixed, a row re-matched) fails
    here until experiments/method_price_fit.py is run again: an estimate is
    never shipped from a fit the file no longer measures."""
    # Loaded from its path: engine/experiments on sys.path would shadow
    # engine/parlay_profiles.py with the experiment of the same name.
    spec = importlib.util.spec_from_file_location(
        "method_price_fit", ENGINE / "experiments" / "method_price_fit.py")
    fit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_script)
    doc = fit_script.run(real_rows)
    assert doc["dev_choice"]["alpha"] == frozen_doc["dev_choice"]["alpha"]
    assert doc["dev_choice"]["prior_fights"] == frozen_doc["dev_choice"]["prior_fights"]
    for t in mp.METHOD_TYPES:
        for s in ("fav", "dog"):
            assert doc["test"]["by_leg"][t][s] == pytest.approx(frozen_doc["test"]["by_leg"][t][s], abs=1e-9)
    assert doc["test"]["all_single_props"] == pytest.approx(frozen_doc["test"]["all_single_props"], abs=1e-9)
    refit, frozen = doc["frozen"]["model"], frozen_doc["frozen"]["model"]
    for row in real_rows[::97]:
        a, b = mp.predict(mp.features_of(row), refit), mp.predict(mp.features_of(row), frozen)
        for side in mp.SIDES:
            for t in mp.METHOD_TYPES:
                assert a[side][t]["decimal"] == pytest.approx(b[side][t]["decimal"], rel=1e-9)


def test_the_measured_error_is_near_the_scouts(frozen_doc):
    """The scouts measured about 20% over all single props on the same split
    (scratchpad scout_facts.md: 20.5%); a rebuild far from that is a
    different estimator, not this one."""
    t = frozen_doc["test"]
    assert 0.15 < t["all_single_props"] < 0.25
    ladder = {s["inputs"]: s["all_single_props"] for s in t["ladder"]}
    assert ladder["+ records (the estimator)"] < ladder["+ division and rounds"] < ladder["moneyline alone"]


def test_the_method_types_are_sameday_s():
    tree = ast.parse((ROOT / "parlays" / "engine" / "sameday.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "METHOD_TYPES" for t in node.targets):
            assert ast.literal_eval(node.value) == mp.METHOD_TYPES
            return
    pytest.fail("sameday.py names no METHOD_TYPES")


# ---------------------------------------------------------------- determinism

def test_estimates_are_deterministic():
    rows = synthetic_rows()
    a, b = mp.fit(rows, 1.0, 3.0), mp.fit(copy.deepcopy(rows), 1.0, 3.0)
    assert a == b and mp.coefficient_hash(a) == mp.coefficient_hash(b)
    feats = mp.features_of(rows[5])
    assert mp.predict(feats, a) == mp.predict(copy.deepcopy(feats), a)
    # through JSON and back: the same numbers, the same hash
    c = json.loads(json.dumps(a))
    assert mp.coefficient_hash(c) == mp.coefficient_hash(a)
    assert mp.predict(feats, c) == mp.predict(feats, a)


def test_swapping_the_corners_swaps_the_estimate():
    m = mp.fit(synthetic_rows(), 1.0, 3.0)
    f = mp.features_of(synthetic_rows()[11])
    g = dict(f, p_red=1 - f["p_red"], facts_red=f["facts_blue"], facts_blue=f["facts_red"])
    a, b = mp.predict(f, m), mp.predict(g, m)
    for t in mp.METHOD_TYPES:
        assert a["red"][t]["decimal"] == pytest.approx(b["blue"][t]["decimal"], rel=1e-12)
        assert a["blue"][t]["decimal"] == pytest.approx(b["red"][t]["decimal"], rel=1e-12)


def test_the_six_chances_are_one_book():
    m = mp.fit(synthetic_rows(), 1.0, 3.0)
    e = mp.predict(mp.features_of(synthetic_rows()[2]), m)
    assert sum(e[s][t]["market_chance"] for s in mp.SIDES for t in mp.METHOD_TYPES) == pytest.approx(1.0)
    for s in mp.SIDES:
        assert sum(e[s][t]["share"] for t in mp.METHOD_TYPES) == pytest.approx(1.0)


def test_the_prior_pools_each_corner_once():
    """A rate two features read (own KO wins, the opponent's KO wins) is one
    pooled rate: the corners' KO wins over their fights, counted once."""
    rows = synthetic_rows()
    m = mp.fit(rows, 1.0, 3.0)
    fights = sum(r[f"facts_{s}"]["fights"] for r in rows for s in mp.SIDES)
    for name, _, count in mp.RECORD_RATES:
        pooled = sum(r[f"facts_{s}"][count] for r in rows for s in mp.SIDES) / fights
        assert m["prior"][name] == pytest.approx(pooled)


def test_the_loading_is_read_at_the_edge_outside_its_range():
    m = mp.fit(synthetic_rows(), 1.0, 3.0)
    lo, hi = m["loading_range"]
    assert mp.loading(1e-9, m) == mp.loading(math.exp(lo), m)
    assert mp.loading(0.999, m) == mp.loading(math.exp(hi), m)


def test_the_estimator_learns_a_known_market():
    """On a synthetic market it can represent, the estimate lands close."""
    rows = synthetic_rows(years=range(2016, 2022))
    m = mp.fit([r for r in rows if r["date"] < "2021"], 1.0, 3.0)
    err = mp.error_table(mp.errors([r for r in rows if r["date"] >= "2021"], m))
    assert err["all_single_props"] < 0.05


def test_no_moneyline_no_estimate():
    m = mp.fit(synthetic_rows(), 1.0, 3.0)
    for p in (None, float("nan"), 0.0, 1.0, "x"):
        with pytest.raises(ValueError):
            mp.predict({"p_red": p, "division": "lightweight"}, m)


def test_too_few_books_is_no_fit():
    with pytest.raises(ValueError):
        mp.fit(synthetic_rows(years=[2016], per_year=mp.MIN_FIT_FIGHTS - 1), 1.0, 3.0)


def test_divisions_read_as_the_archive_spells_them():
    assert mp.normal_division("Women’s Bantamweight") == "women's bantamweight"
    assert mp.normal_division("interim heavyweight") == "heavyweight"
    assert mp.normal_division("Light Heavyweight") == "light heavyweight"
    assert mp.normal_division("catch weight") is None
    assert mp.normal_division(None) is None and mp.normal_division(float("nan")) is None


# ---------------------------------------------------------------- no future

def test_the_fit_names_no_result_column():
    assert not set(mp.LEG_COLUMNS) & mp.RESULT_COLUMNS
    assert not set(mp.ARCHIVE_COLUMNS) & mp.RESULT_COLUMNS


def test_fit_rows_never_reads_a_result(tmp_path, monkeypatch):
    """The fit files hold winner and method; fit_rows asks for neither, and
    scrambling them changes nothing it returns."""
    legs = pd.DataFrame([{
        "fight_id": f"x{i}", "date": "2018-03-03", "event": "E", "red": "A", "blue": "B",
        "dec_red_close": 1.6, "dec_blue_close": 2.5,
        "ko_red": 3.0, "ko_blue": 5.0, "sub_red": 8.0, "sub_blue": 12.0,
        "decision_red": 3.2, "decision_blue": 6.5,
        "facts_red": json.dumps({"fights": 4, "ko_wins": 2}), "facts_blue": "{}",
        "winner": "red", "method": "KO/TKO"} for i in range(3)])
    arch = pd.DataFrame([{"fight_id": f"x{i}", "division": "lightweight", "total_rounds": 3,
                          "winner": "A", "method": "KO/TKO", "finish_round": 1} for i in range(3)])
    lp, ap = tmp_path / "legs.csv.gz", tmp_path / "arch.csv"
    legs.to_csv(lp, index=False)
    arch.to_csv(ap, index=False)
    asked = []
    real = pd.read_csv

    def watched(path, *args, **kwargs):
        asked.append(kwargs.get("usecols"))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", watched)
    first = mp.fit_rows(lp, ap)
    assert len(asked) == 2 and all(u is not None for u in asked)
    assert not {c for u in asked for c in u} & mp.RESULT_COLUMNS
    assert first and not {k for r in first for k in r} & mp.RESULT_COLUMNS
    legs.assign(winner="blue", method="Submission").to_csv(lp, index=False)
    arch.assign(winner="B", method="Decision", finish_round=3).to_csv(ap, index=False)
    assert mp.fit_rows(lp, ap) == first


def test_the_fit_script_reads_through_fit_rows_only():
    text = (ENGINE / "experiments" / "method_price_fit.py").read_text()
    assert "read_csv" not in text and "open(" not in text
    assert "mp.fit_rows()" in text


def test_a_yearly_refit_reads_only_earlier_fights():
    rows = synthetic_rows(years=range(2016, 2021))
    a = mp.fit_before(2019, rows, DOC)
    assert a["fit_last"] < "2019-01-01"
    later = [dict(r, ko_red=r["ko_red"] * 1.3, sub_blue=r["sub_blue"] * 0.8)
             if r["date"] >= "2019-01-01" else r for r in rows]
    later += [dict(r, fight_id=r["fight_id"] + "b") for r in rows if r["date"] >= "2019"]
    assert mp.fit_before(2019, later, DOC) == a


def test_no_refit_without_enough_earlier_books():
    """Rows from 2016, 120 books a year: nothing precedes 2016, one year is
    fewer than MIN_FIT_FIGHTS, two years are enough."""
    rows = synthetic_rows()
    assert mp.MIN_FIT_FIGHTS > 120
    assert mp.fit_before(2016, rows, DOC) is None
    assert mp.fit_before(2017, rows, DOC) is None
    assert mp.fit_before(2018, rows, DOC)["fit_fights"] == 240


def _archive(extra=()):
    rows = [("Alpha One", "Beta Two", "2026-01-10", "Alpha One", "KO/TKO"),
            ("Gamma Three", "Alpha One", "2026-03-10", "Alpha One", "SUB"),
            ("Beta Two", "Delta Four", "2026-05-10", "Delta Four", "U-DEC")]
    rows += list(extra)
    return pd.DataFrame(rows, columns=["r_name", "b_name", "date", "winner", "method"])


CARD_FIGHT = {"number": 1, "red": "Alpha One", "blue": "Beta Two", "pick": "Alpha One",
              "win_prob": 0.66, "market_prob": 0.62, "rounds_scheduled": 3}


@pytest.fixture(scope="module")
def estimator():
    est = ax.method_estimator()
    if est["model"] is None:
        pytest.skip(est["missing"])
    return est


def test_a_card_reads_records_strictly_before_its_date(estimator):
    """A bout on the card's own date (this very fight, finished) or after it
    must not move a single leg."""
    before = ax.method_legs_payload(CARD_FIGHT, "lightweight", _archive(), "2026-11-14", estimator)
    after = ax.method_legs_payload(CARD_FIGHT, "lightweight", _archive([
        ("Alpha One", "Beta Two", "2026-11-14", "Alpha One", "KO/TKO"),
        ("Beta Two", "Epsilon Five", "2026-12-01", "Beta Two", "SUB")]), "2026-11-14", estimator)
    assert before == after
    moved = ax.method_legs_payload(CARD_FIGHT, "lightweight", _archive([
        ("Alpha One", "Zeta Six", "2026-11-13", "Alpha One", "KO/TKO")]), "2026-11-14", estimator)
    assert moved != before        # a bout the day before is history, and does count


# ---------------------------------------------------------------- the legs

def test_the_six_legs(estimator):
    legs = ax.method_legs_payload(CARD_FIGHT, "lightweight", _archive(), "2026-11-14", estimator)
    assert [(l["corner"], l["type"]) for l in legs] == [(c, t) for c in mp.SIDES for t in mp.METHOD_TYPES]
    assert {l["source"] for l in legs} == {"estimated"}
    assert all(l["missing_reason"] is None for l in legs)
    assert [l["fighter"] for l in legs] == ["Alpha One"] * 3 + ["Beta Two"] * 3
    # chance: the shipped win chance times the estimator's share of its wins
    est = mp.predict({"p_red": 0.62, "division": "lightweight", "rounds_scheduled": 3,
                      "facts_red": pp.facts(_archive(), "Alpha One", "2026-11-14"),
                      "facts_blue": pp.facts(_archive(), "Beta Two", "2026-11-14")},
                     estimator["model"])
    for l in legs:
        shipped = 0.66 if l["corner"] == "red" else 0.34
        e = est[l["corner"]][l["type"]]
        assert l["chance"] == pytest.approx(shipped * e["share"], abs=1e-6)
        assert l["market_chance"] == pytest.approx(e["market_chance"], abs=1e-6)
        assert l["est_decimal"] == pytest.approx(e["decimal"], abs=1e-6)
        assert l["est_american"] == mp.american(l["est_decimal"])
        side = "fav" if l["corner"] == "red" else "dog"
        assert l["error_pct"] == round(100 * estimator["measured"]["test"]["by_leg"][l["type"]][side], 1)
    for corner, shipped in (("red", 0.66), ("blue", 0.34)):
        assert sum(l["chance"] for l in legs if l["corner"] == corner) == pytest.approx(shipped, abs=1e-5)


def test_the_pick_s_side_is_read_from_the_pick(estimator):
    """Picking blue hands the same corner chances in: the legs do not move."""
    as_blue = dict(CARD_FIGHT, pick="Beta Two", win_prob=0.34, market_prob=0.38)
    assert (ax.method_legs_payload(as_blue, "lightweight", _archive(), "2026-11-14", estimator)
            == ax.method_legs_payload(CARD_FIGHT, "lightweight", _archive(), "2026-11-14", estimator))


@pytest.mark.parametrize("change, division, archive, reason", [
    ({"market_prob": None}, "lightweight", True, "No moneyline pair"),
    ({"win_prob": None}, "lightweight", True, "No shipped win chance"),
    ({}, None, True, "No division named"),
    ({}, "lightweight", False, "records could not be read"),
])
def test_a_leg_that_cannot_be_estimated_says_why(estimator, change, division, archive, reason):
    legs = ax.method_legs_payload(dict(CARD_FIGHT, **change), division,
                                  _archive() if archive else None, "2026-11-14", estimator)
    assert len(legs) == 6
    for l in legs:
        assert reason in l["missing_reason"]
        assert l["chance"] is l["est_decimal"] is l["est_american"] is l["error_pct"] is None
        assert l["source"] == "estimated" and l["fighter"] in ("Alpha One", "Beta Two")


def test_a_catch_weight_is_named_and_estimated(estimator):
    legs = ax.method_legs_payload(CARD_FIGHT, "catch weight", _archive(), "2026-11-14", estimator)
    assert all(l["est_decimal"] for l in legs)


# ---------------------------------------------------------------- the card

def _lift_block():
    start = SOURCE.index("_method_estimator = _app_export.method_estimator()")
    end = SOURCE.index("# END OF THE METHOD LEGS")
    return SOURCE[start:end]


def _card():
    fights = [dict(CARD_FIGHT, odds=-160, odds_other=140, method_probs={"KO/TKO": 0.3}),
              {"number": 2, "red": "Gamma Three", "blue": "Delta Four", "pick": "Delta Four",
               "win_prob": 0.55, "market_prob": None, "rounds_scheduled": 5}]
    return ax.card_payload("Card", "2026-11-14", fights, profiles_measured=None)


def _run_block(card, app_export=ax, contexts=None, archive_ready=True):
    import live_rows
    ns = {"_app_export": app_export, "_app_card": card, "_live_rows": live_rows,
          "FIGHT_CONTEXTS": {0: {"division": "lightweight"}, 1: {"division": "women's flyweight"}}
          if contexts is None else contexts,
          "_form_archive": _archive(), "_form_ready": archive_ready, "EVENT_DATE": "2026-11-14"}
    exec(_lift_block(), ns)
    return card


def _without_legs(card):
    out = copy.deepcopy(card)
    for f in out["fights"]:
        f.pop("method_legs", None)
    return out


def test_the_card_export_adds_method_legs_and_changes_nothing_else(estimator):
    plain = _card()
    with_legs = _run_block(copy.deepcopy(plain))
    assert _without_legs(with_legs) == plain
    assert set(with_legs) == set(plain)
    assert all(set(f) - set(p) == {"method_legs"} for f, p in zip(with_legs["fights"], plain["fights"]))
    first, second = (f["method_legs"] for f in with_legs["fights"])
    assert all(l["est_decimal"] for l in first)
    assert all("No moneyline pair" in l["missing_reason"] for l in second)
    json.dumps(with_legs, allow_nan=False)


def test_the_card_reads_each_fight_s_own_division(estimator):
    """FIGHT_CONTEXTS is keyed by FIGHT_CARD index; a card fight's number is
    that index plus one."""
    a = _run_block(_card(), contexts={0: {"division": "heavyweight"}})["fights"][0]["method_legs"]
    b = _run_block(_card(), contexts={0: {"division": "women's strawweight"}})["fights"][0]["method_legs"]
    c = _run_block(_card(), contexts={1: {"division": "heavyweight"}})["fights"][0]["method_legs"]
    assert a != b
    assert all("No division named" in l["missing_reason"] for l in c)


def test_a_failure_never_costs_the_card(estimator):
    class Broken:
        method_estimator = staticmethod(ax.method_estimator)
        method_legs_unbuilt = staticmethod(ax.method_legs_unbuilt)

        @staticmethod
        def method_legs_payload(*args, **kwargs):
            raise RuntimeError("boom")

    plain = _card()
    card = _run_block(copy.deepcopy(plain), app_export=Broken)
    assert _without_legs(card) == plain
    for f in card["fights"]:
        assert [l["missing_reason"] for l in f["method_legs"]] == ["Not estimated on this run (RuntimeError)."] * 6


def test_no_records_on_the_run_is_a_reason_not_a_debut(estimator):
    card = _run_block(_card(), archive_ready=False)
    assert all("records could not be read" in l["missing_reason"]
               for f in card["fights"] for l in f["method_legs"] if f["market_prob"] is not None)


def test_card_payload_itself_is_unchanged():
    """method_legs is written onto the card after card_payload, as the median
    pair is: card_payload's own fields are exactly what they were."""
    assert "method_legs" not in _card()["fights"][0]
    assert "_card_fight['method_legs'] = _app_export.method_legs_payload(" in SOURCE
    block = SOURCE.index("_method_estimator = _app_export.method_estimator()")
    assert SOURCE.index("_app_card = _app_export.card_payload(") < block
    assert block < SOURCE.index("for _path in _app_export.write_all(APP_DATA_DIR, {'card': _app_card}):")
