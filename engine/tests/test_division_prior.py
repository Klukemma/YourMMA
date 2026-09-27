"""The division prior is built from the target, so it is guarded like one.

Heavyweight finishes 63.9% of the time and women's strawweight 33.6%. Telling
the model that is worth thirty points of signal - and telling it by way of a
whole-dataset groupby would tell each fight how often fights like it end
early including itself, which is a target leak and is how this project's two
worst bugs both looked.

The test that matters is the last one: changing the outcome of a LATER fight
must not move an EARLIER fight's value. A groupby over the whole frame fails
it. Nothing else here would catch that.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from division_prior import (K_FINISH, LEAGUE_FALLBACK, division_finish_prior,
                            final_priors, prior_for,
                            is_finish, normalise_division)


def frame(rows):
    return pd.DataFrame(rows, columns=["date", "division", "method"])


# --- the pieces ------------------------------------------------------------

def test_interim_is_the_same_division():
    assert normalise_division("Interim Heavyweight") == "heavyweight"
    assert normalise_division("heavyweight") == "heavyweight"


def test_a_blank_division_does_not_crash_or_pool_with_a_real_one():
    assert normalise_division(None) == ""
    assert normalise_division(np.nan) == ""
    assert normalise_division("  ") == ""


def test_finishes_decisions_and_neither():
    assert is_finish("KO/TKO") is True
    assert is_finish("Submission") is True
    assert is_finish("U-DEC") is False
    assert is_finish("Split Decision") is False
    # A disqualification is neither, and counting it either way moves a
    # division's rate for a reason unrelated to how its fights are fought.
    assert is_finish("DQ") is None
    assert is_finish("Overturned") is None
    assert is_finish(None) is None


# --- the accumulation ------------------------------------------------------

def test_the_first_fight_gets_the_league_prior_not_its_own_result():
    out = division_finish_prior(frame([("2020-01-01", "heavyweight", "KO/TKO")]))
    assert out.iloc[0] == pytest.approx(LEAGUE_FALLBACK)


def test_a_division_drifts_toward_its_own_rate_as_bouts_accumulate():
    """The league has to differ from the division or there is nothing to
    drift away from: shrinking an all-finish division toward an all-finish
    league correctly gives 1.0 and demonstrates nothing."""
    rows = []
    for day in pd.date_range("2020-01-01", periods=60, freq="D"):
        # Four decisions elsewhere for every heavyweight finish, so the league
        # sits near 20% and heavyweight's own rate is 100%.
        stamp = day.strftime("%Y-%m-%d")
        rows.append((stamp, "heavyweight", "KO/TKO"))
        for _ in range(4):
            rows.append((stamp, "lightweight", "U-DEC"))
    out = division_finish_prior(frame(rows))
    heavy = out[[i for i, r in enumerate(rows) if r[1] == "heavyweight"]]

    assert heavy.iloc[0] == pytest.approx(LEAGUE_FALLBACK)
    assert heavy.iloc[-1] > heavy.iloc[5] > heavy.iloc[1]
    # Pulled well above the league it is shrunk toward...
    league = out[[i for i, r in enumerate(rows) if r[1] == "lightweight"]].iloc[-1]
    assert heavy.iloc[-1] > league + 0.3
    # ...and still short of its own raw 100%, because k=40 still holds it back.
    assert heavy.iloc[-1] < 0.95


def test_divisions_do_not_contaminate_each_other():
    """A newcomer division gets the league, not the division next door."""
    rows = [(f"2020-01-{d:02d}", "heavyweight", "KO/TKO") for d in range(1, 29)]
    rows += [(f"2020-01-{d:02d}", "lightweight", "U-DEC") for d in range(1, 29)]
    rows.append(("2020-02-01", "women's strawweight", "U-DEC"))
    out = division_finish_prior(frame(rows))

    heavy_last = out[[i for i, r in enumerate(rows)
                      if r[1] == "heavyweight"]].iloc[-1]
    straw = out.iloc[-1]
    league = 28 / 56.0            # half the settled fights were finishes
    assert straw == pytest.approx(league, abs=0.02)
    assert straw < heavy_last - 0.1


def test_two_fights_on_one_card_cannot_inform_each_other():
    """Ordering within a day would leak a few hours of the future."""
    rows = [("2020-01-01", "heavyweight", "KO/TKO"),
            ("2020-01-01", "heavyweight", "U-DEC")]
    out = division_finish_prior(frame(rows))
    assert out.iloc[0] == out.iloc[1] == pytest.approx(LEAGUE_FALLBACK)


def test_a_dq_moves_nothing():
    rows = [("2020-01-01", "heavyweight", "DQ"),
            ("2020-02-01", "heavyweight", "KO/TKO")]
    out = division_finish_prior(frame(rows))
    assert out.iloc[1] == pytest.approx(LEAGUE_FALLBACK)


# --- the one that matters --------------------------------------------------

def test_a_later_result_cannot_change_an_earlier_fight():
    """The leak test. A whole-dataset groupby passes everything above and
    fails this, which is exactly how the career-total leak survived."""
    rows = [("2020-01-01", "heavyweight", "U-DEC"),
            ("2020-02-01", "heavyweight", "U-DEC"),
            ("2020-03-01", "heavyweight", "U-DEC"),
            ("2020-04-01", "heavyweight", "U-DEC")]
    before = division_finish_prior(frame(rows))

    flipped = list(rows)
    flipped[-1] = ("2020-04-01", "heavyweight", "KO/TKO")
    after = division_finish_prior(frame(flipped))

    assert list(before)[:-1] == pytest.approx(list(after)[:-1])


def test_shuffling_every_later_outcome_leaves_the_first_half_untouched():
    rng = np.random.default_rng(3)
    rows = [(f"2020-01-{d:02d}", "lightweight",
             "KO/TKO" if rng.random() < 0.5 else "U-DEC") for d in range(1, 29)]
    before = division_finish_prior(frame(rows))

    shuffled = rows[:14] + [(d, div, "KO/TKO" if m == "U-DEC" else "U-DEC")
                            for d, div, m in rows[14:]]
    after = division_finish_prior(frame(shuffled))
    assert list(before)[:14] == pytest.approx(list(after)[:14])


def test_the_whole_dataset_shortcut_would_fail_that_test():
    """Mutation, written out: the tempting one-liner, shown failing."""
    rows = [("2020-01-01", "heavyweight", "U-DEC"),
            ("2020-02-01", "heavyweight", "U-DEC"),
            ("2020-03-01", "heavyweight", "KO/TKO")]
    df = frame(rows)

    def leaky(d):
        fin = d["method"].map(is_finish).astype(float)
        return d.groupby("division")["method"].transform(
            lambda s: fin.loc[s.index].mean())

    before = leaky(df)
    flipped = frame(rows[:2] + [("2020-03-01", "heavyweight", "U-DEC")])
    after = leaky(flipped)
    assert before.iloc[0] != after.iloc[0], (
        "if this passes, the shortcut is no longer leaky and this file "
        "is guarding against nothing")


def test_it_runs_on_the_real_dataset_and_ranks_the_divisions_sensibly():
    csv = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
    if not csv.exists():
        pytest.skip("dataset not present")
    df = pd.read_csv(csv, low_memory=False, usecols=["date", "division", "method"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    prior = division_finish_prior(df)

    assert prior.notna().all()
    assert (prior.between(0, 1)).all()

    div = df["division"].map(normalise_division)
    recent = df["date"] >= "2020-01-01"
    heavy = prior[recent & (div == "heavyweight")].mean()
    straw = prior[recent & (div == "women's strawweight")].mean()
    assert heavy > straw + 0.15, (heavy, straw)


def test_an_unparseable_date_is_refused_rather_than_silently_dropped():
    """It has no place in an ordering, and the grouping would hand back NaN
    with nothing said - which is how a feature quietly goes missing for part
    of the data. This was a real silent NaN before the check existed."""
    rows = [("2020-01-01", "heavyweight", "KO/TKO"),
            ("2020-01-99", "heavyweight", "U-DEC")]
    with pytest.raises(ValueError, match="will not parse"):
        division_finish_prior(frame(rows))

    relaxed = division_finish_prior(frame(rows), strict=False)
    assert np.isnan(relaxed.iloc[1])


# --- the live counterpart --------------------------------------------------
# A prediction is for a fight that is not in the frame, so it cannot read the
# accumulated column and has to be handed the state the accumulation ended in
# - the same shape as career_stats.final_stats().

def test_final_priors_match_the_accumulated_column_at_the_end():
    """Two definitions of one number is how they drift apart. These are the
    same arithmetic, so a fight on the day after the data ends gets exactly
    what the last fight of the data would have been given."""
    rows = [(f"2020-01-{d:02d}", "heavyweight", "KO/TKO") for d in range(1, 15)]
    rows += [(f"2020-01-{d:02d}", "lightweight", "U-DEC") for d in range(1, 15)]
    df = frame(rows)

    priors, league = final_priors(df)
    extended = frame(rows + [("2020-02-01", "heavyweight", "U-DEC")])
    accumulated = division_finish_prior(extended).iloc[-1]
    assert priors["heavyweight"] == pytest.approx(accumulated)
    assert league == pytest.approx(0.5)


def test_a_division_ranking_survives_the_round_trip():
    rows = [(f"2020-01-{d:02d}", "heavyweight", "KO/TKO") for d in range(1, 29)]
    rows += [(f"2020-01-{d:02d}", "women's strawweight", "U-DEC")
             for d in range(1, 29)]
    priors, _ = final_priors(frame(rows))
    assert priors["heavyweight"] > priors["women's strawweight"] + 0.3


def test_a_mismatched_pairing_gets_the_league_not_a_coin_toss():
    """A catchweight, or someone moving up, belongs to no single division."""
    priors = {"heavyweight": 0.9, "flyweight": 0.2}
    assert prior_for("heavyweight", "flyweight", priors, 0.5) == 0.5
    assert prior_for("heavyweight", "heavyweight", priors, 0.5) == 0.9
    # Interim and non-interim are the same division and do agree.
    assert prior_for("Interim Heavyweight", "heavyweight", priors, 0.5) == 0.9


def test_an_unknown_or_missing_division_falls_back_to_the_league():
    priors = {"heavyweight": 0.9}
    assert prior_for(None, None, priors, 0.47) == 0.47
    assert prior_for("", "", priors, 0.47) == 0.47
    assert prior_for("catch weight", "catch weight", priors, 0.47) == 0.47


def test_the_live_prior_is_a_real_probability_on_the_real_dataset():
    csv = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
    if not csv.exists():
        pytest.skip("dataset not present")
    df = pd.read_csv(csv, low_memory=False, usecols=["date", "division", "method"])
    priors, league = final_priors(df)
    assert 0.3 < league < 0.7
    assert all(0 < v < 1 for v in priors.values())
    assert priors["heavyweight"] > priors["women's strawweight"] + 0.2
