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

The fighter behind a name is found the way the archive's next row will name
them (fighter_candidates, resolve_fighter): the sync's 'new_<hex>' and
'unk_<name>' placeholders are read as the real id they stand for, and a name
two fighters share is settled by the division of the fight, or refused.
"""

import io
import re
import unicodedata
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
# The archive's text columns that a one-row CSV parse would otherwise read
# as numbers: an id of the form digits-e-digits ('329e403448756217',
# Nathaniel Wood) parses as a float and overflows to inf, and the fighter
# loses his whole career in that pass.
TEXT_COLUMNS = {"r_id": str, "b_id": str, "event_id": str, "fight_id": str}
# A listed height outside this band is a units bug of the scrape (the
# archive holds 234-381 cm for 17 rows of late 2025, every one on a
# placeholder-id row), not a fighter. A corner left with no height - or no
# dob, which a placeholder row never carries - is built through the
# pipeline's own unknown branch (height_known = 0, age_known = 0, the
# diffs 0), as training built every row of a fighter it had no profile
# for: the archive before the date does not know the value, and the row
# says so rather than carrying 381 cm. The next scrape usually does know
# it, so the training row for that bout can differ on the profile-derived
# columns (Santiago Luna, 2026-02-28: 10 of 151 features, all height- and
# age-derived; tests/test_live_rows.py pins the set).
HEIGHT_CM = (140.0, 230.0)
_PLACEHOLDER = re.compile(r"^(new_|unk_)")


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


def normal_division(division):
    """The archive's spelling of a division: lower case, straight
    apostrophe, no surrounding space. None when there is none.

    The archive never holds a curly apostrophe, so "women’s bantamweight"
    typed on a card used to miss DIVISION_KG and fall back to the fighter's
    last listed weight.
    """
    if division is None or (isinstance(division, float) and np.isnan(division)):
        return None
    text = unicodedata.normalize("NFKC", str(division))
    text = text.replace("’", "'").replace("‘", "'").strip().lower()
    return text or None


def _canon_ids(before):
    """{id as the archive holds it: the id the fighter's next row will hold}.

    A sync that cannot match a fighter writes a placeholder, 'new_<hex>'
    or 'unk_<name>'; once they are matched, the archive holds the real
    '<hex>' for them from then on, while the placeholder rows keep theirs.
    Measured over the 31 placeholder ids: 27 have a later real-id row for
    the same name, and every one of those 27 holds the id the placeholder
    stands for (the other 4 fighters have not fought again); the real id
    can also precede the placeholder (Zach Reese's had 5 bouts before the
    sync failed on him twice). Read as the real id, a pending row carries
    what the next row will, and gets the career columns training built
    for that fighter at that point - debut-style when the real id had no
    earlier bout, which is what the training row holds (Josh Hokit,
    2026-01-24). Two things this cannot repair:

    - the profile values a placeholder row carries can differ from what
      the next scrape writes (Hokit's lists 208 cm and no dob; his real
      row 185 cm and 1997-11-12): the profile-derived features of his
      pending row differ from training, and nothing before the date can
      say which is right (tests/test_live_rows.py pins the set);
    - a replay AT a placeholder date is not that training row by design:
      the archive built that bout under the placeholder (career columns
      of a debut), the pending row builds it under the real id (Reese vs
      McVey, 2025-11-08: 77 features). The replay experiments leave those
      bouts out.
    """
    long = pd.concat([pd.DataFrame({"id": before[f"{c}_id"],
                                    "name": before[f"{c}_name"]})
                      for c in "rb"]).dropna()
    long["id"] = long["id"].astype(str)
    real = long[~long["id"].str.match(_PLACEHOLDER)]
    by_name = (real.assign(key=real["name"].map(norm_name))
               .groupby("key")["id"].agg(lambda s: sorted(set(s))))
    out = {}
    for fid in long["id"].unique():
        if fid.startswith("new_"):
            out[fid] = fid[4:]
        elif fid.startswith("unk_"):
            twins = by_name.get(norm_name(fid[4:]), [])
            out[fid] = twins[0] if len(twins) == 1 else fid
        else:
            out[fid] = fid
    return out


def _long(before, columns):
    """Both corners stacked, dated, ids canonical: id, name, the per-corner
    `columns` (by suffix) and _when, oldest first (stable)."""
    canon = _canon_ids(before)
    order = pd.to_datetime(before["date"], errors="coerce")
    parts = []
    for c in "rb":
        part = before[[f"{c}_id", f"{c}_name"] + [f"{c}_{p}" for p in columns]].copy()
        part.columns = ["id", "name"] + list(columns)
        part["_when"] = order.values
        parts.append(part)
    long = pd.concat(parts)
    long["id"] = long["id"].map(lambda i: canon.get(i, i) if pd.notna(i) else i)
    return long.sort_values("_when", kind="mergesort")


def profiles(before):
    """{column: Series by fighter id} - latest non-null value per fighter,
    and the latest spelling of their name.

    A height outside HEIGHT_CM is a scrape's units bug and is skipped, so
    the previous valid scrape (or nothing) is what the row inherits."""
    long = _long(before, PROFILE)
    height = pd.to_numeric(long["height"], errors="coerce")
    long.loc[(height < HEIGHT_CM[0]) | (height > HEIGHT_CM[1]), "height"] = np.nan
    out = {col: long.dropna(subset=[col]).groupby("id")[col].last()
           for col in ("name",) + PROFILE}
    out["dob"] = out["dob"].map(_normal_dob)
    return out


def fighter_candidates(before):
    """{normalised name: [every id the name has had, most recent last]}.

    One id for nearly every name. 'Bruno Silva' is two fighters (a
    middleweight and a flyweight, both spelled the same), and the archive
    cannot say which one a card means: resolve_fighter settles that by the
    division of the fight."""
    long = _long(before, ()).dropna(subset=["name", "id"])
    long["key"] = long["name"].map(norm_name)
    return (long.groupby("key")["id"]
            .agg(lambda s: list(dict.fromkeys(s))).to_dict())


def fighter_ids(before):
    """{normalised name: id} for every name that is one fighter in the
    archive. A name two fighters share is left out: see resolve_fighter."""
    return {key: ids[0] for key, ids in fighter_candidates(before).items()
            if len(ids) == 1}


def division_history(before):
    """{fighter id: [division of each bout, oldest first]}."""
    canon = _canon_ids(before)
    order = pd.to_datetime(before["date"], errors="coerce")
    long = pd.concat([
        pd.DataFrame({"id": before[f"{c}_id"], "division": before["division"],
                      "_when": order.values}) for c in "rb"])
    long = long.dropna(subset=["id", "division"]).sort_values("_when", kind="mergesort")
    long["id"] = long["id"].map(lambda i: canon.get(i, i))
    return long.groupby("id")["division"].agg(list).to_dict()


def last_division(before):
    """{fighter id: division of their latest bout}."""
    return {i: h[-1] for i, h in division_history(before).items()}


def resolve_fighter(name, division, candidates, history):
    """The archive id for `name` in a bout at `division`.

    The only id the name has had, when there is one. When two fighters
    share it, the fight's division tells them apart through two signals -
    the one fighter whose latest bout was there, and the one fighter who
    has fought there most - and an id is returned only when the signals
    agree, or when one speaks and the other is silent (no single fighter,
    or none at all). The archive files one of the flyweight Bruno Silva's
    bouts under the middleweight's id, so for a while both are 'flyweight'
    last and the bout count decides; had the flyweight's latest bout been
    his one bantamweight one, the latest-bout rule alone would have named
    the middleweight for a flyweight fight. A KeyError says so when no
    division is given, when neither signal speaks, or when they disagree -
    the row of the other man would be a silent lie."""
    ids = candidates.get(norm_name(name))
    if not ids:
        raise KeyError(f"no archive id for {name!r}")
    if len(ids) == 1:
        return ids[0]
    want = normal_division(division)
    by_latest = by_count = None
    if want:
        latest = [i for i in ids
                  if normal_division((history.get(i) or [None])[-1]) == want]
        if len(latest) == 1:
            by_latest = latest[0]
        bouts = {i: sum(normal_division(d) == want for d in history.get(i, []))
                 for i in ids}
        most = max(bouts.values())
        if most and list(bouts.values()).count(most) == 1:
            by_count = max(bouts, key=bouts.get)
        said = {i for i in (by_latest, by_count) if i}
        if len(said) == 1:
            return said.pop()
    howmany = "two" if len(ids) == 2 else str(len(ids))
    record = "; ".join(
        f"{i}: " + ", ".join(f"{d} x{n}" for d, n in
                             pd.Series(history.get(i, [])).value_counts().items())
        for i in ids)
    if not want:
        hint = "give the division or red_id/blue_id"
    elif by_latest and by_count:
        hint = (f"the division ({want}) does not settle it - the latest bout "
                f"there is {by_latest}'s, most bouts there are {by_count}'s; "
                f"give red_id/blue_id")
    else:
        hint = f"the division ({want}) does not tell them apart; give red_id/blue_id"
    raise KeyError(f"{name!r} is {howmany} fighters in the archive ({record}); {hint}")


def build_pending(before, fights):
    """One row per fight, in `before`'s columns.

    fights: dicts with red, blue (archive spellings or close to them), date,
    and optionally division, is_5rnd, is_title, red_id, blue_id. A missing
    division is taken from the fighters' last bouts when both agree, and
    left empty with a warning when they do not - a row no training row
    looks like, so the live path (live_rows.py) refuses such a fight
    before it gets here rather than build it.
    """
    candidates = fighter_candidates(before)
    canon = _canon_ids(before)
    prof = profiles(before)
    history = division_history(before)
    divisions = {i: h[-1] for i, h in history.items()}
    rows = []
    for i, fight in enumerate(fights):
        row = {c: np.nan for c in before.columns}
        division = normal_division(fight.get("division"))
        corner_ids = {}
        for c, who in (("r", "red"), ("b", "blue")):
            fid = fight.get(f"{who}_id")
            if fid is None:
                fid = resolve_fighter(fight[who], division, candidates, history)
            fid = canon.get(fid, fid)
            corner_ids[c] = fid
            row[f"{c}_id"] = fid
            row[f"{c}_name"] = prof["name"].get(fid, fight[who])
            for p in PROFILE:
                row[f"{c}_{p}"] = prof[p].get(fid, np.nan)
        if not division:
            r_div, b_div = (divisions.get(corner_ids["r"]),
                            divisions.get(corner_ids["b"]))
            division = r_div if r_div == b_div else np.nan
            if r_div != b_div:
                warnings.warn(f"{fight['red']} vs {fight['blue']}: no division "
                              f"given and their last bouts differ "
                              f"({r_div} / {b_div}) - left empty")
        if isinstance(division, str) and division in DIVISION_KG:
            for c in "rb":
                row[f"{c}_weight"] = DIVISION_KG[division]
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
    rated = (rated[rated["event_id"] == "pending"].copy().set_index("fight_id")
             .loc[pend["fight_id"]].reset_index())
    for c in "rb":
        for r in RATING:
            pend[f"{c}_{r}"] = rated[f"{c}_{r}"].values
    # Through CSV text and back, as every archive row was, so each value is
    # parsed exactly as the archive's would be - with the text columns
    # pinned, because a one-row parse has no second value to tell it that
    # '329e403448756217' is an id and not a float.
    return pd.read_csv(io.StringIO(pend.to_csv(index=False)), low_memory=False,
                       dtype=TEXT_COLUMNS)
