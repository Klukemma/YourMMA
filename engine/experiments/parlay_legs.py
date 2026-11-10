"""One row per fight for the parlay backtest - only what was known before the bell.

experiments/parlay_profiles.py scores the parlay profiles on these rows.
This file only assembles them, and every column a ticket is CHOSEN from is
something a bettor could have had before the fight:

  p_model   the shipped recipe's walk-forward probability that the red corner
            wins (model_compare_baseline_symmetric.csv: year Y predicted by
            models fitted on fights before Y only). Read from the CSV, not
            through model_compare.predictions(): that call re-hashes the
            feature matrix, and any archive change since the fit sends it
            into a sixteen-year refit. The cache IS the measured artefact.
  prices    both snapshots, each fight: the CLOSING moneyline (odds.csv,
            dec_red_close / dec_blue_close) and the OPENING line
            (opening_odds.csv, dec_red_open / dec_blue_open), blank where the
            snapshot has no usable pair - odds.csv skips all of 2025. Both
            are quoted before the fight, but they are not the same price (an
            opening line is quoted days before the close), and a ticket that
            took one leg at the close and another at the open could never
            have been bought. So each CARD is priced from ONE snapshot: the
            one that prices more of its fights, ties to the close
            (card_snapshot). dec_red / dec_blue / price_source are that
            snapshot's price for every fight on the card; a fight priced only
            in the other snapshot is left unpriced there ("off_snapshot" in
            the coverage) - its price is never borrowed from the other one.
  method    the KO/TKO, submission and decision props (method_odds.csv,
            American -> decimal), matched the same way. Their history stops
            on 2024-12-07.
  facts     each corner's UFC record strictly before the fight's date
            (parlay_profiles.facts) - the route a long leg's reason names.

  winner    "red" / "blue" / None (draw, no contest, overturned) and method
  method    (parlay_profiles.method_class), from the archive by fight_id -
            used to SETTLE a ticket, never to choose one.

MATCHING. A price row names two fighters and a date; the corner order is
arbitrary. Names are compared after name_resolution.norm_name and
data/fighter_aliases.json. The date may be the fight's or one day either
side, because the Odds API dates a row by its UTC start and a US evening
card starts on the next UTC day. The passes run in order over every fight
- exact date and names, then a date one day off, then (only for a date
inside that tolerance, and only when the OTHER corner matches exactly and
one row alone qualifies) the first initial plus surname - so a weaker
match can never take a row a stronger one would have claimed. Each row is
used once: one price cannot settle two fights. A pair the core module's
devig refuses (an overround below 0 or above 15%: a stale or mistyped
line) is rejected and counted, and that snapshot's columns are left blank.
Unpriced is shown as unpriced - no price is ever interpolated.

THE FIGHTS. The walk-forward set holds decided fights only: draws and no
contests were never scored, so they are not in the cache, cannot be legs,
and the void rule never fires here. Their count is reported (and the price
rows that belong to them, so "unmatched" means a row no fight explains).

    python engine/experiments/parlay_legs.py
"""

import json
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import parlay_profiles as pp  # noqa: E402
from name_resolution import load_aliases, norm_name  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "model_compare_baseline_symmetric.csv"
DATA = ENGINE / "data"
ODDS = DATA / "odds.csv"
OPENING = DATA / "opening_odds.csv"
METHOD_ODDS = DATA / "method_odds.csv"
ARCHIVE = DATA / "UFC_with_mmr_rebuilt_dedup.csv"
ALIASES = DATA / "fighter_aliases.json"
OUT = HERE / "parlay_legs.csv.gz"

START = "2016-01-01"
DATE_TOLERANCE_DAYS = 1
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

SNAPSHOTS = ("close", "open")        # in order of preference on a tie

COLUMNS = ["fight_id", "date", "event", "number", "red", "blue", "p_model",
           "price_source", "dec_red", "dec_blue",
           "dec_red_close", "dec_blue_close", "dec_red_open", "dec_blue_open",
           "ko_red", "ko_blue",
           "sub_red", "sub_blue", "decision_red", "decision_blue",
           "winner", "method", "facts_red", "facts_blue"]


# ---------------------------------------------------------------- names

def canonical(name, aliases):
    """norm_name, through the alias file when it names this spelling."""
    n = norm_name(name)
    alias = aliases.get(n)
    return norm_name(alias) if alias else n


def initial_surname(n):
    """"j smith" from "john smith" (a generational suffix is not a surname);
    None for a single name, which has nothing to abbreviate."""
    toks = [t for t in n.split() if t not in _SUFFIXES]
    if len(toks) < 2:
        return None
    return f"{toks[0][0]} {toks[-1]}"


# ---------------------------------------------------------------- price files

class Book:
    """One price file indexed by (date, name, name).

    pairs: ((col_a, col_b), ...) - the columns read per row, each a pair of
    one value per fighter, handed back oriented to (red, blue)."""

    def __init__(self, path, pairs, aliases, label):
        frame = pd.read_csv(path)
        self.label, self.pairs = label, pairs
        self.rows = []
        self.exact = defaultdict(list)
        self.loose = defaultdict(list)
        for r in frame.to_dict("records"):
            d = pd.to_datetime(r["date"], errors="coerce")
            if pd.isna(d):
                continue
            a, b = canonical(r["fighter_a"], aliases), canonical(r["fighter_b"], aliases)
            idx = len(self.rows)
            self.rows.append({"date": d.date(), "a": a, "b": b,
                              "values": [(r[ca], r[cb]) for ca, cb in pairs]})
            self.exact[(d.date(), a, b)].append(idx)
            for pos, n in (("a", a), ("b", b)):
                key = initial_surname(n)
                if key:
                    self.loose[(d.date(), key)].append((idx, pos))
        self.used = {}

    def _take(self, idx, swapped, how, owner):
        self.used[idx] = owner
        vals = self.rows[idx]["values"]
        oriented = [(vb, va) if swapped else (va, vb) for va, vb in vals]
        return {"row": idx, "how": how, "values": oriented}

    def exact_match(self, day, red, blue, how, owner):
        for idx in self.exact.get((day, red, blue), []):
            if idx not in self.used:
                return self._take(idx, False, how, owner)
        for idx in self.exact.get((day, blue, red), []):
            if idx not in self.used:
                return self._take(idx, True, how, owner)
        return None

    def loose_match(self, day, red, blue, owner):
        """One corner by initial + surname, the other exactly, one row only."""
        found = set()
        for name, other, corner in ((red, blue, "red"), (blue, red, "blue")):
            key = initial_surname(name)
            if not key:
                continue
            for idx, pos in self.loose.get((day, key), []):
                if idx in self.used:
                    continue
                row = self.rows[idx]
                partner = row["b"] if pos == "a" else row["a"]
                if partner != other:
                    continue
                # the row's `pos` fighter is this corner
                swapped = (pos == "a") != (corner == "red")
                found.add((idx, swapped))
        if len(found) != 1:
            return None
        idx, swapped = found.pop()
        return self._take(idx, swapped, "initial", owner)

    def unused_by_year(self, lo, hi):
        """Rows dated inside [lo, hi] that no fight claimed, by year."""
        out = defaultdict(int)
        for idx, r in enumerate(self.rows):
            if lo <= r["date"] <= hi and idx not in self.used:
                out[r["date"].year] += 1
        return out

    def used_by(self, owner_kind, lo, hi):
        out = defaultdict(int)
        for idx, owner in self.used.items():
            d = self.rows[idx]["date"]
            if lo <= d <= hi and owner[0] == owner_kind:
                out[d.year] += 1
        return out


def match_all(book, fights):
    """{fight index: match} over the passes, strongest first.
    fights: [(owner, date, red_canon, blue_canon)]"""
    got = {}
    offsets = [o for k in range(1, DATE_TOLERANCE_DAYS + 1) for o in (-k, k)]
    for i, (owner, day, red, blue) in enumerate(fights):
        m = book.exact_match(day, red, blue, "exact", owner)
        if m:
            got[i] = m
    for i, (owner, day, red, blue) in enumerate(fights):
        if i in got:
            continue
        for off in offsets:
            m = book.exact_match(day + timedelta(days=off), red, blue, "shifted", owner)
            if m:
                got[i] = m
                break
    for i, (owner, day, red, blue) in enumerate(fights):
        if i in got:
            continue
        for off in [0] + offsets:
            m = book.loose_match(day + timedelta(days=off), red, blue, owner)
            if m:
                got[i] = m
                break
    return got


def alias_gaps(book, fights, got, lo, hi):
    """Orphan rows (inside [lo, hi], claimed by no fight) that share exactly
    one corner with a fight this book left unpriced, inside the date
    tolerance. Read by eye they are one fighter under two names - a married
    name, a family-name-first order, a nickname - which only an entry in
    data/fighter_aliases.json can join; counted so the gap is a number."""
    unpriced = defaultdict(list)
    for i, (_, day, red, blue) in enumerate(fights):
        if i not in got:
            unpriced[day].append({red, blue})
    out = defaultdict(int)
    for idx, r in enumerate(book.rows):
        if not (lo <= r["date"] <= hi) or idx in book.used:
            continue
        names = {r["a"], r["b"]}
        for off in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
            day = r["date"] + timedelta(days=off)
            if any(len(names & f) == 1 for f in unpriced.get(day, [])):
                out[r["date"].year] += 1
                break
    return out


# ---------------------------------------------------------------- the rows

def _results(archive, aliases):
    """{fight_id: (winner side or None, method class, resolved?)}."""
    out = {}
    for r in archive.to_dict("records"):
        w = r.get("winner")
        method = pp.method_class(r.get("method"))
        if not isinstance(w, str) or not w.strip():
            out[r["fight_id"]] = (None, method, True)
            continue
        wn = canonical(w, aliases)
        if wn == canonical(r["r_name"], aliases):
            out[r["fight_id"]] = ("red", method, True)
        elif wn == canonical(r["b_name"], aliases):
            out[r["fight_id"]] = ("blue", method, True)
        else:
            out[r["fight_id"]] = (None, method, False)
    return out


def _american(x):
    return pp.decimal_from_american(x)


def snapshot_pair(row, snap):
    """(dec_red, dec_blue) a row carries for `snap`, or None when that
    snapshot has no usable pair (missing, NaN after a CSV round trip, or one
    devig refuses)."""
    dr = pp._price(row.get(f"dec_red_{snap}"))
    db = pp._price(row.get(f"dec_blue_{snap}"))
    return (dr, db) if pp.devig(dr, db) is not None else None


def card_snapshot(rows):
    """The one snapshot a card is priced from: the one that prices more of
    its fights, the close on a tie; None when neither prices any."""
    counts = {s: sum(snapshot_pair(r, s) is not None for r in rows) for s in SNAPSHOTS}
    best = max(SNAPSHOTS, key=lambda s: counts[s])      # max keeps the first on a tie
    return best if counts[best] else None


def price_cards(rows):
    """Set price_source, dec_red and dec_blue on every row (in place) from
    its card's snapshot, so no ticket can mix two. Cards are (event, date),
    the backtest's grouping. Returns, row by row, "priced", "off_snapshot"
    (a usable price only in the snapshot its card did not use) or
    "unpriced"."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["event"], r["date"])].append(i)
    status = [None] * len(rows)
    for idx in groups.values():
        snap = card_snapshot([rows[i] for i in idx])
        for i in idx:
            r = rows[i]
            pair = snapshot_pair(r, snap) if snap else None
            r["price_source"] = snap if pair else None
            r["dec_red"], r["dec_blue"] = pair if pair else (None, None)
            if pair:
                status[i] = "priced"
            elif any(snapshot_pair(r, s) for s in SNAPSHOTS):
                status[i] = "off_snapshot"
            else:
                status[i] = "unpriced"
    return status


def build(cache_path=CACHE, odds_path=ODDS, opening_path=OPENING,
          method_path=METHOD_ODDS, archive_path=ARCHIVE, aliases_path=ALIASES,
          start=START):
    """(rows frame in COLUMNS order, coverage dict)."""
    aliases = load_aliases(aliases_path)
    cache = pd.read_csv(cache_path)
    cache = cache[cache["date"] >= start].reset_index(drop=True)
    cache["number"] = cache.groupby(["event", "date"], sort=False).cumcount() + 1
    end = cache["date"].max()
    archive = pd.read_csv(archive_path, low_memory=False,
                          usecols=["fight_id", "date", "event_name", "r_name", "b_name",
                                   "winner", "method"])
    archive["date"] = archive["date"].astype(str).str[:10]

    close = Book(odds_path, (("odds_a", "odds_b"),), aliases, "close")
    opening = Book(opening_path, (("open_a", "open_b"),), aliases, "open")
    method = Book(method_path, (("ko_a", "ko_b"), ("sub_a", "sub_b"), ("dec_a", "dec_b")),
                  aliases, "method")

    days = pd.to_datetime(cache["date"]).dt.date.tolist()
    reds = [canonical(n, aliases) for n in cache["red"]]
    blues = [canonical(n, aliases) for n in cache["blue"]]
    fights = [(("cache", fid), d, r, b) for fid, d, r, b in zip(cache["fight_id"], days, reds, blues)]

    # archive fights in the same span the walk-forward never scored (draws,
    # no contests): matched only so their rows are not reported as orphans
    in_cache = set(cache["fight_id"])
    extra = archive[(archive["date"] >= start) & (archive["date"] <= end)
                    & ~archive["fight_id"].isin(in_cache)]
    extra_fights = [(("excluded", fid), pd.Timestamp(d).date(), canonical(r, aliases),
                     canonical(b, aliases))
                    for fid, d, r, b in zip(extra["fight_id"], extra["date"],
                                            extra["r_name"], extra["b_name"])]

    matched = {}
    for book in (close, opening, method):
        matched[book.label] = match_all(book, fights)
        match_all(book, extra_fights)

    results = _results(archive, aliases)
    by_name = defaultdict(list)
    for i, (r, b) in enumerate(zip(archive["r_name"], archive["b_name"])):
        by_name[r].append(i)
        by_name[b].append(i)
    hist_cols = archive[["r_name", "b_name", "date", "winner", "method"]]
    history = {n: hist_cols.iloc[ix] for n, ix in by_name.items()}
    empty_hist = hist_cols.iloc[0:0]

    rows, cov = [], defaultdict(lambda: defaultdict(int))
    for i, f in enumerate(cache.to_dict("records")):
        year = f["date"][:4]
        c = cov[year]
        c["fights"] += 1
        pairs = {}
        for label in SNAPSHOTS:
            m = matched[label].get(i)
            if not m:
                continue
            c[f"{label}_matched"] += 1
            c[f"{label}_{m['how']}"] += 1
            (a_red, a_blue), = m["values"]
            dr, db = _american(a_red), _american(a_blue)
            if pp.devig(dr, db) is None:
                c[f"{label}_rejected"] += 1
                continue
            pairs[label] = (dr, db)

        mprice = {}
        m = matched["method"].get(i)
        if m:
            c["method_matched"] += 1
            c[f"method_{m['how']}"] += 1
            (ko_r, ko_b), (sub_r, sub_b), (dc_r, dc_b) = m["values"]
            mprice = {"ko_red": _american(ko_r), "ko_blue": _american(ko_b),
                      "sub_red": _american(sub_r), "sub_blue": _american(sub_b),
                      "decision_red": _american(dc_r), "decision_blue": _american(dc_b)}
            if pp._method_market(mprice) is not None:
                c["method_priced"] += 1
            elif all(mprice.values()):
                c["method_rejected"] += 1

        winner, how, resolved = results.get(f["fight_id"], (None, None, False))
        if not resolved:
            c["result_unresolved"] += 1
        elif winner is None:
            c["result_void"] += 1
        elif (winner == "red") != (f["y"] == 1):
            c["result_disagrees_with_cache"] += 1

        facts = {side: pp.facts(history.get(f[side], empty_hist), f[side], f["date"])
                 for side in ("red", "blue")}
        rows.append({
            "fight_id": f["fight_id"], "date": f["date"], "event": f["event"],
            "number": int(f["number"]), "red": f["red"], "blue": f["blue"],
            "p_model": float(f["p_model"]),
            # the card's snapshot, set by price_cards once every row is in
            "price_source": None, "dec_red": None, "dec_blue": None,
            **{f"dec_{side}_{snap}": (pairs[snap][k] if snap in pairs else None)
               for snap in SNAPSHOTS for k, side in enumerate(("red", "blue"))},
            "ko_red": mprice.get("ko_red"), "ko_blue": mprice.get("ko_blue"),
            "sub_red": mprice.get("sub_red"), "sub_blue": mprice.get("sub_blue"),
            "decision_red": mprice.get("decision_red"),
            "decision_blue": mprice.get("decision_blue"),
            "winner": winner if resolved else "unresolved", "method": how,
            "facts_red": json.dumps(facts["red"], separators=(",", ":")),
            "facts_blue": json.dumps(facts["blue"], separators=(",", ":")),
        })

    for r, state in zip(rows, price_cards(rows)):
        cov[r["date"][:4]][f"priced_{r['price_source']}" if state == "priced" else state] += 1

    lo = pd.Timestamp(start).date() - timedelta(days=DATE_TOLERANCE_DAYS)
    hi = pd.Timestamp(end).date() + timedelta(days=DATE_TOLERANCE_DAYS)
    for book in (close, opening, method):
        for y, n in book.unused_by_year(lo, hi).items():
            cov[str(y)][f"{book.label}_rows_unmatched"] += n
        for y, n in alias_gaps(book, fights, matched[book.label], lo, hi).items():
            cov[str(y)][f"{book.label}_rows_alias_gap"] += n
        for y, n in book.used_by("excluded", lo, hi).items():
            cov[str(y)][f"{book.label}_rows_excluded_fight"] += n
    for y, n in extra.groupby(extra["date"].str[:4]).size().items():
        cov[y]["excluded_fights"] += int(n)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    by_year = {y: dict(sorted(v.items())) for y, v in sorted(cov.items())}
    total = defaultdict(int)
    for v in by_year.values():
        for k, n in v.items():
            total[k] += n
    coverage = {"first": str(cache["date"].min()), "last": str(end),
                "date_tolerance_days": DATE_TOLERANCE_DAYS,
                "aliases": len(aliases),
                "total": dict(sorted(total.items())), "by_year": by_year}
    return frame, coverage


def coverage_table(coverage):
    """The coverage by year, as text."""
    def pair(c, a, b):
        return f"{c.get(a, 0)}/{c.get(b, 0)}"

    lines = [f"  {'year':<6}{'fights':>7}{'close':>7}{'open':>7}{'off-snap':>9}{'unpriced':>9}"
             f"{'rej c/o':>9}{'+-1 day':>8}{'initial':>8}{'method':>8}{'draw/NC':>8}"
             f"{'orphan c/o/m':>14}{'alias gap c/m':>15}"]
    for y, c in list(coverage["by_year"].items()) + [("total", coverage["total"])]:
        g = c.get
        lines.append(
            f"  {y:<6}{g('fights', 0):>7}{g('priced_close', 0):>7}{g('priced_open', 0):>7}"
            f"{g('off_snapshot', 0):>9}{g('unpriced', 0):>9}{pair(c, 'close_rejected', 'open_rejected'):>9}"
            f"{g('close_shifted', 0) + g('open_shifted', 0):>8}"
            f"{g('close_initial', 0) + g('open_initial', 0):>8}{g('method_priced', 0):>8}"
            f"{g('excluded_fights', 0):>8}"
            f"{pair(c, 'close_rows_unmatched', 'open_rows_unmatched') + '/' + str(g('method_rows_unmatched', 0)):>14}"
            f"{pair(c, 'close_rows_alias_gap', 'method_rows_alias_gap'):>15}")
    lines += [
        "  close/open = fights priced from that snapshot (each card uses one: the one",
        "  pricing more of its fights, ties to the close); off-snap = fights priced only",
        "  in the snapshot their card did not use (no leg); rej c/o = matched pairs devig",
        "  refused (overround < 0 or > 15%); +-1 day / initial = price rows matched on a",
        "  shifted date / on initial + surname; method = fights with all six method",
        "  props; draw/NC = archive fights the walk-forward never scored; orphan = price",
        "  rows in the span no fight claimed; alias gap = orphans sharing one corner",
        "  with an unpriced fight (one fighter, two names: fighter_aliases.json)."]
    return "\n".join(lines)


def write(frame, path=OUT):
    frame.to_csv(path, index=False, compression="gzip", float_format="%.6g")


def main():
    frame, coverage = build()
    write(frame)
    print(coverage_table(coverage))
    print(f"  wrote {OUT.name}: {len(frame):,} fights {coverage['first']} .. {coverage['last']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
