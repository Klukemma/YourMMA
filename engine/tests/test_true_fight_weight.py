"""True Fight Weight: a measurement informs a fight only once PUBLISHED
before it; catch weights and heavyweight are not division cuts; shrinkage
toward the division norm; the gate's power rule; the pre-registered arms
and a placebo that is not the real arm in disguise."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import true_fight_weight as tfw
from experiments import true_fight_weight_edge as edge

ARCHIVE = pd.DataFrame([
    ("2019-01-01", "welterweight", "Ann", "X", "Ann", 180, 170, 185, 175, "1990-01-01", "1990-01-01"),
    ("2021-01-01", "lightweight", "Ann", "Y", "Ann", 180, 170, 185, 175, "1990-01-01", "1990-01-01"),
    ("2022-01-01", "lightweight", "Bea", "Z", "Bea", 170, 170, 172, 175, "1995-01-01", "1990-01-01"),
    ("2022-01-01", "women's flyweight", "Cat", "Dee", "Cat", 165, 165, 168, 168, "1996-01-01", "1996-01-01"),
    ("2022-08-13", "catch weight", "Quin", "Witt", "Quin", 180, 180, 185, 185, "1995-01-01", "1995-01-01"),
    ("2023-06-01", "lightweight", "Ann", "Y", "Ann", 160, 170, 185, 175, "1990-01-01", "1990-01-01"),
], columns=["date", "division", "r_name", "b_name", "winner", "r_height",
            "b_height", "r_reach", "b_reach", "r_dob", "b_dob"])

WORLD = pd.DataFrame({
    "date": ["2023-10-07", "2023-10-07", "2023-10-07", "2024-01-01"],
    "weight": ["Women's Flyweight", "Catchweight (138 lb)", "Lightweight", "Welterweight"],
    "winner": ["Jena", "Bobby", "Usman", "Lorenz"],
    "loser": ["Ilara", "Alberto", "Brent", "Levan"]})


def labels(rows):
    """Measured regains: (date, published_at, fighter, limit, sex, gain_pct)."""
    frame = pd.DataFrame(rows, columns=["date", "published_at", "fighter", "limit",
                                        "sex", "gain_pct"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["published_at"] = pd.to_datetime(frame["published_at"])
    frame["key"] = frame["fighter"].str.lower()
    frame["event"] = "E " + frame["date"].dt.strftime("%Y-%m-%d")
    frame["official_lbs"] = frame["limit"].astype(float)
    frame["fight_night_lbs"] = frame["official_lbs"] * (1 + frame["gain_pct"] / 100)
    frame["off_limit"] = False
    frame["suspect"] = False
    return frame


# Ten lightweights measured in January 2023, published a week later; Ann
# measured again in June 2023 and again in 2024.
BASE = [("2023-01-01", "2023-01-08", f"f{i}", 155, "m", 8.0 + i) for i in range(10)]
ANN = [("2023-01-01", "2023-01-08", "Ann", 155, "m", 16.0),
       ("2023-06-01", "2023-06-08", "Ann", 155, "m", 15.0)]


def test_division_limits_and_the_class_an_official_weight_was_made_for():
    assert tfw.limit_of("Light Heavyweight") == 205
    assert tfw.limit_of("Heavyweight") == 265
    assert tfw.limit_of("Women's Strawweight") == 115
    assert np.isnan(tfw.limit_of("Catch Weight")) and np.isnan(tfw.limit_of("Catchweight (160 lb)"))
    assert tfw.class_of(154.8) == 155 and tfw.class_of(178.2) == 185
    # 233.8 lb is a heavyweight, not a light heavyweight 29 lb over.
    assert tfw.class_of(233.8) == 265 and tfw.class_of(235.0) == 265
    assert tfw.class_of(206.0) == 205 and tfw.class_of(206.5) == 265


# ---------------------------------------------------------------------------
# point in time
# ---------------------------------------------------------------------------

def test_a_measurement_counts_only_from_its_publication_date():
    est = tfw.RegainEstimator(labels(BASE + ANN), k=1.0)
    # Fight day itself: the check is not public, nothing is.
    assert est.table("2023-01-01") == {}
    assert np.isnan(est.estimate("ann", "2023-01-01", 155, "m")["r_hat"])
    # Between the event and publication: still nothing.
    assert np.isnan(est.estimate("ann", "2023-01-07", 155, "m")["r_hat"])
    # The publication day itself is not "before" it.
    assert np.isnan(est.estimate("ann", "2023-01-08", 155, "m")["r_hat"])
    # The day after: the division norm exists and Ann has one measurement.
    after = est.estimate("ann", "2023-01-09", 155, "m")
    assert after["n"] == 1 and np.isfinite(after["r_hat"])


def test_a_fights_own_and_same_event_measurements_never_inform_it():
    est = tfw.RegainEstimator(labels(BASE + ANN), k=1.0)
    # Ann's June measurement is on the books, published June 8. A fight on
    # June 1 (its own) and June 5 (same week) see only January.
    for day in ("2023-06-01", "2023-06-05", "2023-06-08"):
        assert est.estimate("ann", day, 155, "m")["n"] == 1
    assert est.estimate("ann", "2023-06-09", 155, "m")["n"] == 2
    # And nobody else's same-event numbers enter mu before publication.
    assert est.mu("2023-06-05", 155, "m")[0] == pytest.approx(np.mean([8.0 + i for i in range(10)] + [16.0]))


def test_forward_rows_use_only_earlier_published_labels():
    rows = tfw.forward_rows(labels(BASE + ANN))
    # The first event has nothing before it, so it is training only.
    assert set(rows["date"].dt.strftime("%Y-%m-%d")) == {"2023-06-01"}
    ann = rows[rows["key"] == "ann"].iloc[0]
    assert ann["has_history"] and ann["n"] == 1
    mu = np.mean([8.0 + i for i in range(10)] + [16.0])
    assert ann["mu_mean"] == pytest.approx(mu)
    # k=0 trusts her own residual fully: mu + (16 - mu) = 16.
    assert ann["k=0.0"] == pytest.approx(16.0)
    assert ann["k=inf"] == pytest.approx(mu)


def test_k_is_chosen_only_from_errors_already_published():
    rows = tfw.forward_rows(labels(BASE + ANN + [
        ("2024-01-01", "2024-01-08", "Ann", 155, "m", 15.5),
        ("2024-01-01", "2024-01-08", "f1", 155, "m", 9.0),
        ("2024-01-01", "2024-01-08", "f2", 155, "m", 10.0)]))
    # Before any repeat is published: the default.
    assert tfw.k_for(rows, "2023-06-01") == tfw.DEFAULT_K
    # The June repeat is one row, fewer than MIN_K_ROWS: still the default.
    assert tfw.k_for(rows, "2023-12-01") == tfw.DEFAULT_K
    # January 2024 adds three repeats, published Jan 8: not usable on Jan 5.
    assert tfw.k_for(rows, "2024-01-05") == tfw.DEFAULT_K
    assert tfw.k_for(rows, "2024-01-09") in tfw.K_GRID


def test_k_for_the_next_fight_sees_every_published_repeat_not_just_the_last_event():
    rows = tfw.nested_k(tfw.forward_rows(labels(BASE + ANN + [
        ("2024-01-01", "2024-01-08", "Ann", 155, "m", 15.5),
        ("2024-01-01", "2024-01-08", "f1", 155, "m", 9.0),
        ("2024-01-01", "2024-01-08", "f2", 155, "m", 10.0)])))
    report, _ = tfw.validate(labels(BASE + ANN + [
        ("2024-01-01", "2024-01-08", "Ann", 155, "m", 15.5),
        ("2024-01-01", "2024-01-08", "f1", 155, "m", 9.0),
        ("2024-01-01", "2024-01-08", "f2", 155, "m", 10.0)]), loeo=False)
    # The last event (Jan 2024) could only use the one June repeat: default
    # k. The next fight may use all four repeats: the chosen k.
    assert report["k_last_event"] == tfw.DEFAULT_K == rows["k_used"].iloc[-1]
    assert report["k_final"] == tfw.k_for(rows, pd.Timestamp.max)
    assert not report["k_identified"]       # 4 rows, far from MIN_K_IDENTIFY


# ---------------------------------------------------------------------------
# shrinkage and the division norm
# ---------------------------------------------------------------------------

def test_shrinkage_grows_with_history_and_shrinks_with_k():
    assert tfw.shrinkage(0, 1.0) == 0.0
    assert tfw.shrinkage(1, 1.0) == 0.5
    assert tfw.shrinkage(3, 1.0) == 0.75
    assert tfw.shrinkage(1, 0.0) == 1.0
    assert tfw.shrinkage(5, np.inf) == 0.0


def test_estimate_pulls_toward_the_norm_in_proportion_to_history():
    est = tfw.RegainEstimator(labels(BASE + ANN))
    mu = est.mu("2023-07-01", 155, "m")[0]
    e_bar = np.mean([16.0 - mu, 15.0 - mu])
    full = est.estimate("ann", "2023-07-01", 155, "m", k=0.0)
    half = est.estimate("ann", "2023-07-01", 155, "m", k=2.0)
    none = est.estimate("ann", "2023-07-01", 155, "m", k=np.inf)
    assert full["r_hat"] == pytest.approx(mu + e_bar)
    assert half["r_hat"] == pytest.approx(mu + 0.5 * e_bar)
    assert none["r_hat"] == pytest.approx(mu)
    # A stranger gets the norm whatever k is.
    assert est.estimate("nobody", "2023-07-01", 155, "m", k=0.0)["r_hat"] == pytest.approx(mu)


def test_a_residual_carries_across_a_class_change():
    rows = BASE + [("2023-01-01", "2023-01-08", f"w{i}", 170, "m", 6.0 + i) for i in range(5)]
    rows += [("2023-01-01", "2023-01-08", "Ann", 170, "m", 11.0)]       # above 170's norm
    est = tfw.RegainEstimator(labels(rows), k=0.0)
    # Her residual is against 170's mean (her own row included, as it would
    # be in deployment: 8.5), and it carries to 155 unchanged.
    over = 11.0 - est.mu("2023-03-01", 170, "m")[0]
    assert over == pytest.approx(2.5)
    at_155 = est.estimate("ann", "2023-03-01", 155, "m")
    assert at_155["e_bar"] == pytest.approx(over)
    assert at_155["r_hat"] == pytest.approx(est.mu("2023-03-01", 155, "m")[0] + over)


def test_thin_sex_cells_fall_back_to_the_division_then_everything():
    rows = BASE + [("2023-01-01", "2023-01-08", "Cat", 155, "w", 4.0)]
    est = tfw.RegainEstimator(labels(rows))
    # One woman at 155: the cell is thin, so she gets 155 pooled.
    assert est.mu("2023-02-01", 155, "w")[1] == "limit"
    assert est.mu("2023-02-01", 155, "m")[1] == "cell"
    # Unknown sex never uses a sex cell; a class with no labels uses all.
    assert est.mu("2023-02-01", 155, "?")[1] == "limit"
    assert est.mu("2023-02-01", 125, "m")[1] == "all"


# ---------------------------------------------------------------------------
# catch weights, suspect rows, heavyweight: not division cuts
# ---------------------------------------------------------------------------

def test_a_catch_weight_label_is_never_history_or_norm_but_stays_on_file():
    rows = labels(BASE + [("2023-03-01", "2023-03-08", "Ann", 170, "m", 5.0)])
    rows.loc[rows["fighter"] == "Ann", ["official_lbs", "off_limit"]] = [175.0, True]
    est = tfw.RegainEstimator(rows)
    # A 175-lb catch-weight regain is not a 170 cut: no history at 170.
    later = est.estimate("ann", "2023-05-01", 170, "m", k=0.0)
    assert later["n"] == 0 and later["level"] == "all"
    assert ("limit", 170.0) not in est.table("2023-05-01")
    assert len(est.labels) == 11                   # kept, flagged
    # Nor is it scored, although a second event's row otherwise would be.
    assert len(tfw.forward_rows(rows)) == 0
    rows["off_limit"] = False
    assert list(tfw.forward_rows(rows)["key"]) == ["ann"]


def test_a_suspect_label_feeds_the_norm_but_is_never_history_and_is_not_scored():
    rows = labels(BASE + ANN)
    rows.loc[(rows["fighter"] == "Ann") & (rows["date"] == "2023-01-01"), "suspect"] = True
    est = tfw.RegainEstimator(rows)
    # Her January number still counts in the 155 norm...
    assert est.table("2023-02-01")[("cell", 155.0, "m")][2] == 11
    # ...but is not her history, so June sees nothing of hers.
    assert est.estimate("ann", "2023-06-01", 155, "m")["n"] == 0
    scored = tfw.forward_rows(rows)
    assert list(scored["key"]) == ["ann"] and not scored["has_history"].iloc[0]


def test_load_labels_flags_catch_weights_and_suspects_and_fills_sex(tmp_path):
    path = tmp_path / "w.csv"
    pd.DataFrame([
        ("2022-08-13", "UFC Vera", "Quin", 175.0, 185.0, 10.0, 5.71, "V"),       # archive: catch weight
        ("2022-08-13", "UFC Vera", "Witt", 179.5, 191.0, 11.5, 6.41, "V"),
        ("2023-10-07", "Bellator 300", "Renato Moicano", 154.8, 180.6, 25.8, 16.67, "B300"),
        ("2023-10-07", "Bellator 300", "Usman", 154.8, 173.2, 18.4, 11.89, "B300"),
        ("2023-10-07", "Bellator 300", "Jena", 125.4, 135.4, 10.0, 7.97, "B300"),  # world: women's
        ("2023-10-07", "Bellator 300", "Ilara", 125.6, 136.0, 10.4, 8.28, "B300"),
        ("2023-10-07", "Bellator 300", "Bobby", 138.2, 147.6, 9.4, 6.8, "B300"),   # world: catchweight
        ("2023-10-07", "Bellator 300", "Alberto", 137.6, 153.4, 15.8, 11.48, "B300"),
        ("2023-10-07", "Bellator 300", "Slim", 233.8, 233.4, -0.4, -0.17, "B300"),
        ("2023-10-07", "Bellator 300", "Big", 265.0, 284.0, 19.0, 7.17, "B300"),
        ("2025-01-18", "UFC 311", "Payton Talbott", 135.5, 158.2, 22.7, 16.75, "U311"),
        ("2025-01-18", "UFC 311", "Raoni Barcelos", 135.5, 158.2, 22.7, 16.75, "U311"),
        ("2025-01-18", "UFC 311", "Ann", 155.0, 170.0, 15.0, 9.68, "U311"),
        ("2025-01-18", "UFC 311", "Stranger", 156.0, 170.0, 14.0, 8.97, "U311"),
    ], columns=["date", "event", "fighter", "official_lbs", "fight_night_lbs",
                "gain_lbs", "gain_pct", "source"]).to_csv(path, index=False)
    got = tfw.load_labels(path, tfw.Archive(ARCHIVE), tfw.World(WORLD))
    by = got.set_index("fighter")
    assert "Brent Primus" in by.index and "Renato Moicano" not in by.index
    # Suspects are flagged and kept, not dropped.
    assert len(got) == 14 and got.attrs["suspect_flagged"] == 2
    assert by.loc["Payton Talbott", "suspect"] and by.loc["Raoni Barcelos", "suspect"]
    # Catch weights: the archive's division, the world's class, or the
    # official weight more than ALLOWANCE_LB from the limit.
    assert by.loc["Quin", "off_limit"] and by.loc["Witt", "off_limit"]
    assert by.loc["Bobby", "off_limit"] and by.loc["Alberto", "off_limit"]
    assert not by.loc["Usman", "off_limit"] and not by.loc["Stranger", "off_limit"]
    # 233.8 lb is a heavyweight, never flagged.
    assert by.loc["Slim", "limit"] == 265 and not by.loc["Slim", "off_limit"]
    assert list(got["published_at"].dt.strftime("%Y-%m-%d").unique()) == \
        ["2022-08-20", "2023-10-14", "2025-01-25"]
    # Sex: the archive, else the world harvest, else the bout's opponent.
    assert by.loc["Ann", "sex"] == "m" and by.loc["Quin", "sex"] == "m"
    assert by.loc["Jena", "sex"] == "w" and by.loc["Ilara", "sex"] == "w"
    assert by.loc["Usman", "sex"] == "m"
    assert by.loc["Stranger", "sex"] == "m"          # Ann's opponent (adjacent)


def test_spelling_corrections_are_this_sessions_and_can_be_switched_off(tmp_path):
    path = tmp_path / "w.csv"
    pd.DataFrame([
        ("2024-02-17", "UFC 298", "Josh Quinland", 169.0, 187.8, 18.8, 11.12, "U"),
        ("2024-09-07", "BCS4", "Alexander Shabily", 154.8, 174.0, 19.2, 12.4, "B"),
    ], columns=["date", "event", "fighter", "official_lbs", "fight_night_lbs",
                "gain_lbs", "gain_pct", "source"]).to_csv(path, index=False)
    assert list(tfw.load_labels(path)["fighter"]) == ["Josh Quinlan", "Alexander Shabliy"]
    assert list(tfw.load_labels(path, spelling=False)["fighter"]) == ["Josh Quinland", "Alexander Shabily"]
    assert set(tfw.SPELLING).isdisjoint(tfw.DESIGN_FIXES)


def test_two_strangers_at_the_same_limit_get_the_same_norm_whatever_source_knows_them(tmp_path):
    # One in the UFC archive (Ann), one in nothing but the Wikipedia list
    # (Stranger), the same bout: the opponent rule gives both a sex, so
    # both read the same cell and the model gap between them is zero.
    path = tmp_path / "w.csv"
    rows = [("2023-01-01", "E1", f"f{i}", 155.0, 170.0, 15.0, 9.68, "s") for i in range(6)]
    rows += [("2023-06-01", "E2", "Ann", 155.0, 172.0, 17.0, 10.97, "s"),
             ("2023-06-01", "E2", "Stranger", 155.0, 168.0, 13.0, 8.39, "s")]
    pd.DataFrame(rows, columns=["date", "event", "fighter", "official_lbs", "fight_night_lbs",
                                "gain_lbs", "gain_pct", "source"]).to_csv(path, index=False)
    got = tfw.load_labels(path, tfw.Archive(ARCHIVE))
    assert set(got.loc[got["event"] == "E2", "sex"]) == {"m"}
    report, rows_ = tfw.validate(got, loeo=False)
    assert report["bouts"]["excluded"]["pooled_level"] == 0
    ann, stranger = (rows_.set_index("key").loc[k] for k in ("ann", "stranger"))
    assert ann["level"] == stranger["level"] and ann["mu_mean"] == pytest.approx(stranger["mu_mean"])


def test_a_mixed_level_pair_is_judged_at_the_pooled_division_for_both():
    rows = labels(BASE + [("2023-01-01", "2023-01-08", "Cat", 155, "w", 4.0)]
                  + [("2023-06-01", "2023-06-08", "P", 155, "m", 10.0),
                     ("2023-06-01", "2023-06-08", "Q", 155, "?", 10.0)])
    report, _ = tfw.validate(rows, loeo=False)
    b = report["bouts"]
    assert b["excluded"]["pooled_level"] == 1 and b["bouts"] == 1
    # Two strangers judged at one level: no model gap, nothing "decided".
    assert b["all"]["model"]["decided"] == 0


def test_heavyweight_bouts_are_outside_the_gate_and_the_scored_rows():
    rows = labels(BASE + [("2023-01-01", "2023-01-08", "Hok", 265, "m", 0.4),
                          ("2023-01-01", "2023-01-08", "Smi", 265, "m", 1.1),
                          ("2023-06-01", "2023-06-08", "Hok", 265, "m", 0.6),
                          ("2023-06-01", "2023-06-08", "Ros", 265, "m", 2.9)]
                  + ANN[1:] + [("2023-06-01", "2023-06-08", "f1", 155, "m", 9.5)])
    rows.loc[rows["limit"] == 265, "official_lbs"] = [246.0, 247.8, 242.2, 264.8]
    report, scored = tfw.validate(rows, loeo=False)
    assert "hok" not in set(scored["key"])
    assert report["bouts"]["excluded"]["heavyweight"] == 0   # not even paired into the count
    assert report["bouts"]["bouts"] == 1                      # Ann vs f1 only
    assert all(r["red"] != "Hok" for r in report["bouts"]["either_history"].get("rows", []))


# ---------------------------------------------------------------------------
# labels and the archive
# ---------------------------------------------------------------------------

def test_frame_features_read_only_earlier_bouts():
    a = tfw.Archive(ARCHIVE)
    assert a.features("Ann", "2019-01-01", 170)["came_down"] == 0
    assert a.features("Ann", "2021-01-01", 155)["came_down"] == 15
    # Before any lightweight bout exists the norm is undefined: z is 0.
    assert a.features("Ann", "2020-06-01", 155)["height_z"] == 0.0
    assert a.features("Ann", "2022-06-01", 155)["height_z"] > \
        a.features("Bea", "2022-06-01", 155)["height_z"]
    assert a.sex("Cat") == "w" and a.sex("Ann") == "m" and a.sex("Nobody") == "?"
    # Ann's archive height drops to 160 at her June 2023 bout (another Ann,
    # or a bad record): a fight in 2022 sees 180, as of its date, not 160.
    assert a.features("Ann", "2022-06-01", 155)["height_z"] == \
        a.features("Ann", "2022-06-01", 155)["height_z"] > 0
    june = a.features("Ann", "2023-06-01", 155)
    assert june["height_z"] == a.features("Ann", "2022-06-01", 155)["height_z"]   # still the 180 record
    assert a.catch_on("Quin", "2022-08-13") and not a.catch_on("Ann", "2021-01-01")


def test_world_harvest_gives_sex_class_and_catch_weight():
    w = tfw.World(WORLD)
    assert w.sex("Jena") == "w" and w.sex("Usman") == "m" and w.sex("Nobody") == "?"
    assert w.limit_on("Usman", "2023-10-07") == 155 and np.isnan(w.limit_on("Bobby", "2023-10-07"))
    assert w.catch_on("Bobby", "2023-10-07") and not w.catch_on("Usman", "2023-10-07")


# ---------------------------------------------------------------------------
# bouts and the gate
# ---------------------------------------------------------------------------

def test_bouts_pair_by_the_world_harvest_then_by_adjacent_rows_in_even_runs():
    rows = labels([("2023-01-01", "2023-01-08", "A", 155, "m", 10.0),
                   ("2023-01-01", "2023-01-08", "B", 155, "m", 12.0),
                   ("2023-01-01", "2023-01-08", "C", 185, "m", 8.0),
                   ("2023-01-01", "2023-01-08", "F", 170, "m", 8.0),
                   ("2023-01-01", "2023-01-08", "D", 145, "m", 9.0),
                   ("2023-01-01", "2023-01-08", "E", 145, "m", 9.0)])
    rows["official_lbs"] = [155.0, 155.5, 185.0, 170.0, 145.0, 145.5]
    world = pd.DataFrame({"date": ["2023-01-01"], "winner": ["D"], "loser": ["E"]})
    pairs = tfw.pair_bouts(rows, world)
    assert (4, 5) in pairs and (0, 1) in pairs
    # C (185) is next to F (170) in the run but 15 lb apart: no bout.
    assert all(2 not in p and 3 not in p for p in pairs)
    # Without F the run A-B-C is odd, its alignment unknown: only D-E.
    assert tfw.pair_bouts(rows.drop(index=3), world) == [(4, 5)]


def test_a_non_adjacent_harvest_pair_does_not_shift_the_pairing_of_the_rest():
    rows = labels([("2023-01-01", "2023-01-08", n, 155, "m", 10.0) for n in "ABCDEF"])
    rows["official_lbs"] = 155.0
    # The harvest says A fought C. B is then alone between them; D, E, F
    # are an odd run whose alignment is unknown: nobody is guessed.
    world = pd.DataFrame({"date": ["2023-01-01"], "winner": ["A"], "loser": ["C"]})
    assert tfw.pair_bouts(rows, world) == [(0, 2)]
    # With G added, D-E-F-G is an even run and pairs as written.
    rows2 = labels([("2023-01-01", "2023-01-08", n, 155, "m", 10.0) for n in "ABCDEFG"])
    rows2["official_lbs"] = 155.0
    assert tfw.pair_bouts(rows2, world) == [(0, 2), (3, 4), (5, 6)]


def _gate_report(events, mean, low, share=0.3):
    return {"bouts": {"either_history": {
        "n": 2 * events, "events": events,
        "model_vs_weigh_in_only": {"mean_gain_lb": mean, "interval": [low, mean + 1],
                                   "max_event_share": share}}}}


def test_gate_needs_enough_events_before_an_interval_can_judge():
    assert not tfw.gate({"bouts": {}})["passed"]
    few = tfw.gate(_gate_report(tfw.MIN_GATE_EVENTS - 1, 2.0, 1.0))
    assert not few["passed"] and not few["testable"] and "too few events" in few["reason"]
    enough = tfw.gate(_gate_report(tfw.MIN_GATE_EVENTS, 2.0, 1.0))
    assert enough["passed"] and enough["testable"]
    assert not tfw.gate(_gate_report(tfw.MIN_GATE_EVENTS, 2.0, -1.0))["passed"]
    # One event carrying most of the gain is not evidence either.
    lopsided = tfw.gate(_gate_report(tfw.MIN_GATE_EVENTS, 2.0, 1.0, share=0.8))
    assert not lopsided["passed"] and not lopsided["testable"]


def test_bout_report_shows_what_the_interval_rests_on():
    rows = labels(BASE + ANN[1:] + [("2023-06-01", "2023-06-08", "f1", 155, "m", 9.5),
                                    ("2024-01-01", "2024-01-08", "Ann", 155, "m", 15.5),
                                    ("2024-01-01", "2024-01-08", "f2", 155, "m", 10.0)])
    report, _ = tfw.validate(rows, loeo=False)
    v = report["bouts"]["either_history"]["model_vs_weigh_in_only"]
    assert set(v["per_event"]) == {"E 2023-06-01", "E 2024-01-01"}
    assert set(v["leave_one_event_out"]) == set(v["per_event"])
    assert len(report["bouts"]["either_history"]["rows"]) == 2


def test_leave_one_event_out_is_a_secondary_table_not_point_in_time():
    rows = labels(BASE + ANN + [("2024-01-01", "2024-01-08", "Ann", 155, "m", 15.5),
                                ("2024-01-01", "2024-01-08", "f1", 155, "m", 9.0)])
    lo = tfw.loeo_report(rows)
    # Every event is predicted, the first included, because the others
    # (later ones too) train it.
    assert lo["with_history"]["n"] + lo["without_history"]["n"] == len(rows)
    forward = tfw.forward_rows(rows)
    assert len(forward) < len(rows)


# ---------------------------------------------------------------------------
# the experiment: estimates per fight, frozen thresholds, arm marking
# ---------------------------------------------------------------------------

def fights():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-12-01", "2023-01-05", "2023-02-01", "2023-02-01",
                                "2023-03-01", "2023-03-01"]),
        "event": ["E0", "E1", "E2", "E2", "E3", "E3"],
        "division": ["lightweight", "lightweight", "lightweight", "heavyweight",
                     "lightweight", "catch weight"],
        "red_raw": ["Ann", "Ann", "Ann", "Big", "Pat", "Pat"],
        "blue_raw": ["Zed", "Zed", "Zed", "Ben", "Zed", "Zed"],
        "red": ["ann", "ann", "ann", "big", "pat", "pat"],
        "blue": ["zed", "zed", "zed", "ben", "zed", "zed"],
        "won": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "p_market": [0.5, 0.6, 0.55, 0.5, 0.4, 0.5],
        "p_blend": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]})


def test_estimates_respect_publication_and_the_mask():
    est = edge.estimates(fights(), labels(BASE + ANN))
    # Before the first publication (Jan 8 2023) nothing is estimable.
    assert not est.loc[0, "eligible"] and not est.loc[1, "eligible"]
    # February: Ann has one published measurement, Zed none.
    assert est.loc[2, "eligible"] and est.loc[2, "n_red"] == 1 and est.loc[2, "n_blue"] == 0
    assert est.loc[2, "delta_regain"] > 0
    # Heavyweight and catchweight are masked out.
    assert not est.loc[3, "eligible"] and not est.loc[5, "eligible"]
    # Two strangers: the same norm, no regain edge.
    assert est.loc[4, "eligible"] and est.loc[4, "delta_regain"] == pytest.approx(0.0)
    assert est.loc[4, "delta_pct"] == pytest.approx(0.0)


def test_a_catch_weight_measurement_gives_no_edge_in_the_experiment():
    rows = labels(BASE + [("2023-01-01", "2023-01-08", "Ann", 170, "m", 5.0)])
    rows.loc[rows["fighter"] == "Ann", ["official_lbs", "off_limit"]] = [175.0, True]
    est = edge.estimates(fights(), rows)
    assert est.loc[2, "eligible"] and est.loc[2, "n_red"] == 0
    assert est.loc[2, "delta_regain"] == pytest.approx(0.0)


def test_missed_weight_pounds_enter_the_full_gap_only():
    frame = fights()
    missed = {(pd.Timestamp("2023-03-01").date(), "pat"): 3.0}
    est = edge.estimates(frame, labels(BASE + ANN), missed=missed)
    assert est.loc[4, "delta_regain"] == pytest.approx(0.0)
    assert est.loc[4, "delta_pct"] > 0


def test_threshold_is_frozen_on_the_first_half_by_date():
    values = [1, 2, 3, 4, 10, 20, 30, 40]
    dates = pd.date_range("2023-01-01", periods=8)
    cut, n = edge.freeze_threshold(values, dates)
    assert n == 4 and cut == pytest.approx(np.quantile([1, 2, 3, 4], 2 / 3))
    assert np.isnan(edge.freeze_threshold([1.0], dates[:1])[0])


def _est_frame():
    est = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"] * 4 + ["2023-03-01"] * 2),
        "event": ["E2"] * 4 + ["E3"] * 2,
        "eligible": [True, True, True, False, True, True],
        "n_red": [1, 0, 1, 1, 0, 0], "n_blue": [0, 1, 0, 0, 0, 0],
        "delta_regain": [3.0, -2.0, 0.5, 9.0, 0.0, 0.0],
        "delta_pct": [3.5, -2.5, 0.5, 9.0, 0.0, 0.0],
        "delta_frame": [np.nan, np.nan, np.nan, np.nan, 1.0, -1.0]})
    est.index.name = "fight"
    return est


class Fight:
    def __init__(self, i):
        self.Index = i


def test_arms_mark_the_side_with_the_edge():
    cuts = {"regain_edge_top_third": (2.0, 2), "tfw_gap_top_third": (2.0, 2),
            "frame_only_gap_top_third": (0.5, 1)}
    pops = edge.populations(_est_frame(), cuts)
    assert edge.PLACEBO not in pops
    assert pops["regain_edge_top_third"](Fight(0)) == "red"
    assert pops["regain_edge_top_third"](Fight(1)) == "blue"
    assert pops["regain_edge_top_third"](Fight(2)) is None      # below the cut
    assert pops["regain_edge_top_third"](Fight(3)) is None      # not eligible
    assert pops["regain_edge_top_third"](Fight(4)) is None      # no history
    assert pops["tfw_gap_top_third"](Fight(0)) == "red"
    # Arm 3 only where neither fighter has a measurement.
    assert pops["frame_only_gap_top_third"](Fight(0)) is None
    assert pops["frame_only_gap_top_third"](Fight(4)) == "red"
    assert pops["frame_only_gap_top_third"](Fight(5)) == "blue"


def test_placebo_hands_each_residual_to_another_fighter_of_the_same_event():
    rows = labels(BASE + ANN + [("2023-06-01", "2023-06-08", "f3", 155, "m", 6.0)])
    placebo = edge.placebo_labels(rows, seed=1)
    # Same rows, same dates, same numbers per event; only the keys moved.
    for (d, e), g in rows.groupby(["date", "event"]):
        p = placebo[(placebo["date"] == d) & (placebo["event"] == e)]
        assert sorted(g["gain_pct"]) == sorted(p["gain_pct"])
        assert sorted(g["key"]) == sorted(p["key"])
    assert (placebo["key"] != rows["key"]).any()
    # Off-limit and suspect rows keep their keys.
    flagged = rows.copy()
    flagged.loc[flagged["key"] == "f0", "off_limit"] = True
    assert (edge.placebo_labels(flagged, seed=1).set_index("key").loc["f0", "gain_pct"]
            == rows.set_index("key").loc["f0", "gain_pct"])


def test_placebo_keeps_the_coverage_and_the_population_but_not_the_values():
    frame = fights()
    real = edge.estimates(frame, labels(BASE + ANN))
    # Use a permutation that moves Ann's January number to f0: Ann then
    # carries f0's residual, the fight is still "Ann has a measurement".
    placebo = edge.estimates(frame, edge.placebo_labels(labels(BASE + ANN), seed=3))
    assert list(placebo["eligible"]) == list(real["eligible"])
    assert list(placebo["n_red"]) == list(real["n_red"])
    assert list(placebo["n_blue"]) == list(real["n_blue"])
    moved = np.isfinite(real["delta_regain"]) & (real["delta_regain"] != 0)
    assert moved.any()
    assert (placebo.loc[moved, "delta_regain"] != real.loc[moved, "delta_regain"]).any()
    cuts = {"regain_edge_top_third": (0.1, 1), "tfw_gap_top_third": (0.1, 1),
            "frame_only_gap_top_third": (0.5, 1)}
    pops = edge.populations(real, cuts, placebo)
    # The placebo is marked over arm 1's population only, at its threshold.
    assert pops[edge.PLACEBO](Fight(4)) is None and pops[edge.PLACEBO](Fight(3)) is None
    assert pops[edge.PLACEBO](Fight(2)) in ("red", "blue", None)


def test_overlap_reports_shared_rows_and_the_unmeasured_marked_side():
    est = _est_frame()
    rows = pd.DataFrame({"fight": [0, 0, 1, 1, 2, 2], "side": ["red", "blue"] * 3})
    marks = {"regain_edge_top_third": np.array([True, False, False, True, False, False]),
             "other": np.array([True, False, True, False, False, False])}
    got = edge.overlap(marks, rows, est)
    assert got["other"]["shared_with_arm1"] == 1 and got["other"]["marked"] == 2
    # Fight 1 marked blue: blue has the measurement; fight 1 marked red: not.
    assert got["regain_edge_top_third"]["marked_side_unmeasured"] == 0
    assert got["other"]["marked_side_unmeasured"] == 1


def test_clustered_logit_recovers_a_planted_coefficient():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    m = rng.normal(size=n)
    events = rng.integers(0, 200, n)
    p = 1 / (1 + np.exp(-(0.2 + 1.0 * m + 0.5 * x)))
    y = (rng.random(n) < p).astype(float)
    beta, se = edge.clustered_logit(y, np.column_stack([np.ones(n), m, x]), events)
    assert beta[2] == pytest.approx(0.5, abs=0.15) and 0 < se[2] < 0.1
