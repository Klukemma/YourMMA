"""Rows for fights not yet fought, in the archive's own layout.

A live prediction is a training row that has not been played yet. This
builds that row: the columns known before the bell, and nothing else -
every result, method, round and per-bout count is left empty, so nothing
the pipeline computes for it can read an outcome. feature_frame.build puts
it through the same code as every training row.

The CONTRACT columns are the only ones a bout's own row feeds into its
features (measured: blanking every other column of an archived event leaves
all of its model features bitwise unchanged - tests/test_live_rows.py):

    date, division, title_fight, total_rounds       from the card
    r_/b_name, r_/b_id                              the fighters
    r_/b_dob, height, reach, weight, stance         latest known values
    r_/b_mu_pre, sigma_pre, mmr_pre                 ratings.extend - exactly
                                                    what a sync stores for
                                                    the bout when it lands

    build_pending(archive_before, fights) -> DataFrame
"""

import io
import warnings

import numpy as np
import pandas as pd

import ratings
from name_resolution import norm_name

PROFILE = ("dob", "height", "reach", "weight", "stance")
# r_weight is the fighter's listed weight, re-scraped with each bout, and it
# follows them between classes. For a fight in a standard division the
# class limit predicts what the next archived row will hold better than the
# fighter's last listed weight (95.2% vs 92.4% exact over the last 25
# events); heavyweights and catch weights keep their own.
DIVISION_KG = {"flyweight": 56.70, "bantamweight": 61.23,
               "featherweight": 65.77, "lightweight": 70.31,
               "welterweight": 77.11, "middleweight": 83.91,
               "light heavyweight": 92.99, "women's strawweight": 52.16,
               "women's flyweight": 56.70, "women's bantamweight": 61.23,
               "women's featherweight": 65.77}
RATING = ("mu_pre", "sigma_pre", "mmr_pre")
CONTRACT = (["date", "division", "title_fight", "total_rounds"]
            + [f"{c}_{p}" for c in "rb"
               for p in ("name", "id") + PROFILE + RATING])


def _normal_dob(value):
    """A dob in the archive's '%Y/%m/%d', whatever form the scrape used.

    Training parses the whole dob column with one inferred format, taken
    from '1963/06/22'; a 'Mon DD, YYYY' value silently becomes NaT there.
    The latest scrape of a fighter is written in '%Y/%m/%d', so that is
    what their next training row will hold.
    """
    if pd.isna(value):
        return np.nan
    when = pd.to_datetime(str(value), errors="coerce", format="mixed")
    return np.nan if pd.isna(when) else when.strftime("%Y/%m/%d")


def profiles(before):
    """{column: Series by fighter id} - latest non-null value per fighter,
    and the latest spelling of their name."""
    order = pd.to_datetime(before["date"], errors="coerce")
    parts = []
    for c in "rb":
        part = before[[f"{c}_id", f"{c}_name"] + [f"{c}_{p}" for p in PROFILE]].copy()
        part.columns = ["id", "name"] + list(PROFILE)
        part["_when"] = order.values
        parts.append(part)
    long = pd.concat(parts).sort_values("_when", kind="mergesort")
    out = {col: long.dropna(subset=[col]).groupby("id")[col].last()
           for col in ("name",) + PROFILE}
    out["dob"] = out["dob"].map(_normal_dob)
    return out


def fighter_ids(before):
    """{normalised name: id at that name's most recent appearance}."""
    order = pd.to_datetime(before["date"], errors="coerce")
    long = pd.concat([
        pd.DataFrame({"name": before[f"{c}_name"], "id": before[f"{c}_id"],
                      "_when": order.values}) for c in "rb"])
    long = long.dropna(subset=["name", "id"]).sort_values("_when", kind="mergesort")
    long["key"] = long["name"].map(norm_name)
    return long.groupby("key")["id"].last().to_dict()


def last_division(before):
    order = pd.to_datetime(before["date"], errors="coerce")
    long = pd.concat([
        pd.DataFrame({"id": before[f"{c}_id"], "division": before["division"],
                      "_when": order.values}) for c in "rb"])
    long = long.dropna(subset=["id", "division"]).sort_values("_when", kind="mergesort")
    return long.groupby("id")["division"].last().to_dict()


def build_pending(before, fights):
    """One row per fight, in `before`'s columns.

    fights: dicts with red, blue (archive spellings or close to them), date,
    and optionally division, is_5rnd, is_title, red_id, blue_id. A missing
    division is taken from the fighters' last bouts when both agree, and
    left empty with a warning when they do not.
    """
    ids = fighter_ids(before)
    prof = profiles(before)
    divisions = last_division(before)
    rows = []
    for i, fight in enumerate(fights):
        row = {c: np.nan for c in before.columns}
        corner_ids = {}
        for c, who in (("r", "red"), ("b", "blue")):
            fid = fight.get(f"{who}_id")
            if fid is None:
                fid = ids.get(norm_name(fight[who]))
            if fid is None:
                raise KeyError(f"no archive id for {fight[who]!r}")
            corner_ids[c] = fid
            row[f"{c}_id"] = fid
            row[f"{c}_name"] = prof["name"].get(fid, fight[who])
            for p in PROFILE:
                row[f"{c}_{p}"] = prof[p].get(fid, np.nan)
        division = fight.get("division")
        if not division:
            r_div, b_div = (divisions.get(corner_ids["r"]),
                            divisions.get(corner_ids["b"]))
            division = r_div if r_div == b_div else np.nan
            if r_div != b_div:
                warnings.warn(f"{fight['red']} vs {fight['blue']}: no division "
                              f"given and their last bouts differ "
                              f"({r_div} / {b_div}) - left empty")
        if isinstance(division, str) and division.lower() in DIVISION_KG:
            for c in "rb":
                row[f"{c}_weight"] = DIVISION_KG[division.lower()]
        rounds = fight.get("total_rounds")
        if rounds is None:
            rounds = 5.0 if fight.get("is_5rnd") else 3.0
        row.update({"event_id": "pending", "event_name": "pending",
                    "fight_id": f"pending-{i}",
                    "date": pd.Timestamp(fight["date"]).strftime("%Y-%m-%d"),
                    "division": division, "total_rounds": float(rounds),
                    "title_fight": int(bool(fight.get("is_title", False)))})
        rows.append(row)
    pend = pd.DataFrame(rows, columns=before.columns)
    rated = ratings.extend(before, pend)
    rated = (rated[rated["event_id"] == "pending"].set_index("fight_id")
             .loc[pend["fight_id"]].reset_index())
    for c in "rb":
        for r in RATING:
            pend[f"{c}_{r}"] = rated[f"{c}_{r}"].values
    # Through CSV text and back, as every archive row was, so each value is
    # parsed exactly as the archive's would be.
    return pd.read_csv(io.StringIO(pend.to_csv(index=False)), low_memory=False)
