"""Live model rows: the upcoming fight through the training pipeline.

A live prediction is a training row that has not been played yet. The
fight becomes a pending row (pending_rows.py) after the archive cut the
day before the event, and feature_frame.build - the code that built every
training row - computes its features, with the interaction centres frozen
at the training values. The live path used to rebuild the same features by
hand from each fighter's last archived row; replaying the last four events,
that moved the winner probability a mean 15.6 points from the training-
style row for the same fight (experiments/live_parity.py).

This is the logic predict_card.py runs for its card, in a class that holds
the three things it needs - the archive as read, the training matrix by
fight id, the frozen centres - so it can be built over any archive and
exercised in a second (tests/test_live_rows_engine.py) rather than only
through the ten-minute engine import.

    live = LiveRows(raw, X_by_fight_id, centres)
    live.prepare(fights, event_date)          # one pass per set of fights
    live.row(red, blue, event_date, ...)      # the row, or NoLiveRow

A fight that cannot be built is refused, never guessed: a corner with no
bout before the date, a name two fighters share that the division cannot
settle, the same fighter in both corners, a fight given no division whose
fighters last fought in different ones (no training row has an empty
division, so that row would be a shape the model never saw), or a pass
whose archive rows no longer equal the training rows. row() raises
NoLiveRow with the reason, and the engine reports the fight as NO DATA.
"""

import numpy as np
import pandas as pd

import feature_frame
import pending_rows
from name_resolution import norm_name


class NoLiveRow(KeyError):
    """The fight has no training-pipeline row; args[0] says why."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


class LiveBuildDrift(RuntimeError):
    """The archive part of a live build is not the training rows."""


def division_of(context):
    """The division a FIGHT_CONTEXTS entry names, or None."""
    if not context:
        return None
    return context.get("division") or context.get("weight_class")


def event_day(event_date=None):
    """The event as a Timestamp; today when the caller gives none."""
    return (pd.Timestamp(event_date) if event_date
            else pd.Timestamp.today().normalize())


def _spec(fight):
    """One fight as a dict: red, blue, is_5rnd, is_title, division.

    Accepts (red, blue[, is_5rnd[, is_title[, context]]]) tuples and dicts
    with those keys, where the division may sit in `context` or be given
    directly."""
    if not isinstance(fight, dict):
        fight = dict(zip(("red", "blue", "is_5rnd", "is_title", "context"),
                         fight))
    division = fight.get("division") or division_of(fight.get("context"))
    return {"red": fight["red"], "blue": fight["blue"],
            "is_5rnd": bool(fight.get("is_5rnd")),
            "is_title": bool(fight.get("is_title")),
            "division": pending_rows.normal_division(division)}


class LiveRows:
    def __init__(self, raw, train_x_by_fight, centres, build=feature_frame.build):
        """raw: the archive as read (dates as text). train_x_by_fight: the
        training X indexed by fight_id as text, which every live pass must
        reproduce bit for bit on its archive rows. centres: the interaction
        centres training fitted. build: feature_frame.build, or a stand-in
        for tests."""
        self.raw = raw
        self.dates = pd.to_datetime(raw["date"], errors="coerce")
        self.train = train_x_by_fight
        self.centres = centres
        self.build = build
        self.rows = {}              # key -> one-row frame of feature_cols
        self.refused = {}           # key -> why there is no row
        self.passes = 0             # pipeline passes run, for tests

    @staticmethod
    def key(red, blue, event_date, is_5rnd, is_title, division):
        """The cache key: the same fight asked for in any spelling of its
        division or form of its date lands on the same entry."""
        return (str(event_day(event_date).date()), norm_name(red),
                norm_name(blue), bool(is_5rnd), bool(is_title),
                pending_rows.normal_division(division) or "")

    # ---- building ----------------------------------------------------------
    def prepare(self, fights, event_date=None):
        """Build the model rows for fights on one date, in as few passes as
        the rule "no fighter twice in one pass" allows (about 12 s a pass
        over the full archive). A fight that cannot be built is recorded
        with its reason and row() refuses it.

        fights: (red, blue, is_5rnd, is_title[, context]) tuples or dicts
        with those keys (or a 'division'), names as the archive spells them.
        """
        when = event_day(event_date)
        wanted = []
        for fight in fights:
            f = _spec(fight)
            key = self.key(f["red"], f["blue"], when, f["is_5rnd"],
                           f["is_title"], f["division"])
            if (key in self.rows or key in self.refused
                    or any(k == key for k, _ in wanted)):
                continue
            wanted.append((key, f))
        if not wanted:
            return
        before = self.raw[self.dates < when]
        candidates = pending_rows.fighter_candidates(before)
        history = pending_rows.division_history(before)
        passes = []                 # each pass holds no fighter twice
        for key, f in wanted:
            try:
                r_id, b_id = self._ids(f, when, candidates, history)
                division = f["division"] or self._division(f, r_id, b_id, history)
            except NoLiveRow as err:
                self.refused[key] = str(err)
                continue
            f = dict(f, red_id=r_id, blue_id=b_id, division=division)
            for batch in passes:
                if not {r_id, b_id} & batch["ids"]:
                    batch["fights"].append((key, f))
                    batch["ids"] |= {r_id, b_id}
                    break
            else:
                passes.append({"fights": [(key, f)], "ids": {r_id, b_id}})
        for batch in passes:
            self._build_pass(before, when, batch["fights"])

    def _ids(self, f, when, candidates, history):
        """Both corners' archive ids before `when`, or NoLiveRow saying why
        there is no row for this fight."""
        ids = []
        for who in ("red", "blue"):
            name = f[who]
            if norm_name(name) not in candidates:
                raise NoLiveRow(f"no bout before {when.date()} for {name!r}")
            try:
                ids.append(pending_rows.resolve_fighter(
                    name, f["division"], candidates, history))
            except KeyError as err:
                raise NoLiveRow(str(err.args[0])) from None
        if ids[0] == ids[1]:
            raise NoLiveRow(f"both corners are the same fighter "
                            f"({f['red']!r} / {f['blue']!r}, id {ids[0]})")
        return ids

    @staticmethod
    def _division(f, r_id, b_id, history):
        """The division of a fight given none: the one both fighters last
        fought in. When their last bouts differ, nothing says which class
        this fight is in, and a row with no division is a shape no training
        row has (weight_class 'unknown', is_womens 0, both weights from
        the last listed value): NoLiveRow, asking for the division."""
        last = {who: pending_rows.normal_division((history.get(i) or [None])[-1])
                for who, i in (("red", r_id), ("blue", b_id))}
        if last["red"] and last["red"] == last["blue"]:
            return last["red"]
        raise NoLiveRow(f"no division given and their last bouts differ "
                        f"({last['red'] or 'none'} / {last['blue'] or 'none'}) "
                        f"- give the division")

    def _build_pass(self, before, when, fights):
        spec = [{"red": f["red"], "blue": f["blue"], "date": when,
                 "red_id": f["red_id"], "blue_id": f["blue_id"],
                 "is_5rnd": f["is_5rnd"], "is_title": f["is_title"],
                 "division": f["division"]} for _, f in fights]
        pend = pending_rows.build_pending(before, spec)
        assert pend["division"].notna().all(), "every fight here has a division"
        built = self.build(before, pending=pend, centres=self.centres,
                           verbose=False)
        self.passes += 1
        n = built["N_ARCHIVE"]
        try:
            self.check_append_invariance(built, n)
            # The pending rows come back in the order they went in: the
            # pipeline sorts before the concat, never after.
            order = built["ufc"]["fight_id"].iloc[n:].astype(str).tolist()
            if order != pend["fight_id"].astype(str).tolist():
                raise LiveBuildDrift(f"pending rows came back reordered "
                                     f"({order}) - no row can be trusted")
        except LiveBuildDrift as err:
            print(f"    WARNING: {err}")
            for key, _ in fights:
                self.refused[key] = str(err)
            return
        rows = built["X"].iloc[n:]
        for (key, _), (_, row) in zip(fights, rows.iterrows()):
            self.rows[key] = row.to_frame().T.reset_index(drop=True)

    def check_append_invariance(self, built, n):
        """The archive part of a live build must equal the training rows,
        bit for bit. If it does not, something in the pipeline has started
        to read across rows, and no live row from it is the training
        definition: LiveBuildDrift, naming the columns."""
        part = built["X"].iloc[:n]
        index = built["ufc"]["fight_id"].iloc[:n].astype(str).values
        train = self.train.loc[index, part.columns]
        mine, theirs = part.to_numpy(float), train.to_numpy(float)
        same = (mine == theirs) | (np.isnan(mine) & np.isnan(theirs))
        if not same.all():
            bad = sorted(set(part.columns[~same.all(axis=0)]))
            raise LiveBuildDrift(
                f"live build differs from training on {len(bad)} columns "
                f"({', '.join(bad[:5])}) - the live rows are not the "
                f"training definition")

    # ---- serving -----------------------------------------------------------
    def row(self, red, blue, event_date=None, is_5rnd=False, is_title=False,
            context=None):
        """The fight's one-row frame of model features, built now if the
        card prefetch did not; NoLiveRow with the reason when it cannot be."""
        when = event_day(event_date)
        division = division_of(context)
        key = self.key(red, blue, when, is_5rnd, is_title, division)
        if key not in self.rows and key not in self.refused:
            self.prepare([{"red": red, "blue": blue, "is_5rnd": is_5rnd,
                           "is_title": is_title, "division": division}], when)
        if key in self.rows:
            return self.rows[key]
        raise NoLiveRow(self.refused.get(
            key, f"no live row for {red!r} vs {blue!r} on {when.date()}"))

    def prepare_card(self, fights, event_date=None, contexts=None, check=None,
                     swapped=False):
        """The prefetch for a card: every fight `check` accepts, in one
        prepare() call, so the whole card costs as few passes as possible.

        fights: (red, blue[, is_5rnd[, is_title]]) as FIGHT_CARD holds them.
        contexts: {index: context} as FIGHT_CONTEXTS; only the division is
        read here. check(red, blue) -> (resolved, problems, warnings) as
        predict_card._check_fighters: a fight with problems is left out and
        refused by the prediction as before; the resolved spellings are what
        the prediction asks for, so they are what is built. swapped=True
        also builds every fight with the corners exchanged - the row a
        symmetrised prediction averages with - which costs the card a
        second pass, since the swapped fight holds the same two fighters.
        Returns the specs prepared (the swapped ones after the listed)."""
        specs = []
        for i, fight in enumerate(fights):
            red, blue = fight[0], fight[1]
            if check is not None:
                resolved, problems, _ = check(red, blue)
                if problems:
                    continue
                red, blue = resolved["red"][0], resolved["blue"][0]
            specs.append({"red": red, "blue": blue,
                          "is_5rnd": fight[2] if len(fight) > 2 else False,
                          "is_title": fight[3] if len(fight) > 3 else False,
                          "division": division_of((contexts or {}).get(i))})
        if swapped:
            specs = specs + [dict(s, red=s["blue"], blue=s["red"]) for s in specs]
        self.prepare(specs, event_date)
        return specs
