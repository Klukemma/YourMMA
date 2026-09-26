"""Feature definitions, declared once and built from the declaration.

Three things went wrong with the hand-written version and all three are
structural rather than careless:

1. The fill rule was inverted on fifteen features.

       acc_diff = r_str_acc.fillna(0) - b_str_acc.fillna(0)

   Striking accuracy is a percentage around 45%. Filling a *missing* value
   with 0 asserts the fighter lands 0% of strikes - the worst fighter who has
   ever competed - so every debutant handed their opponent a fake maximal
   advantage. age_diff was the one feature that filled the *difference*
   instead, which is the correct neutral, so the right pattern was already in
   the file and applied once out of sixteen times.

   Here an operand is never filled. The difference is computed from raw values
   and is NaN when either side is unknown; the matrix turns that into 0, the
   neutral value, and a companion _known flag tells the model which it is.

2. Defaults sat on the wrong scale. height and reach were filled with 70,
   meaning inches, against columns holding centimetres - a 110-unit fake gap on
   9.4% of fights, against a real reach spread of 8.3.

3. A feature could be listed for training without being produced at prediction
   time. Adding two features to the list and not to the hand-written
   prediction dict killed a whole run with a KeyError, ninety seconds in.

The spec below is the single source of truth: the same declaration builds the
training matrix and a single fight, so the two cannot drift apart.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --- primitives ------------------------------------------------------------

def to_number(values):
    """Numeric view of a column, with unparseable entries as NaN."""
    return pd.to_numeric(values, errors="coerce")


def paired_diff(red, blue):
    """red - blue, with no operand filled.

    Unknown minus anything is unknown. That is the whole point: a missing
    value must not masquerade as the worst possible value.
    """
    return to_number(red) - to_number(blue)


def paired_level(red, blue):
    """The mean of the two, i.e. how good this fight is at the attribute.

    A difference says who is better. It cannot say whether two elite strikers
    or two poor ones are meeting, and those are different fights.
    """
    return (to_number(red) + to_number(blue)) / 2.0


def paired_known(red, blue):
    """1.0 when both sides are present, else 0.0.

    Without this a neutral difference is indistinguishable from an evenly
    matched pair, and the model has no way to discount the former.
    """
    return (to_number(red).notna() & to_number(blue).notna()).astype(float)


def ratio(numerator, denominator, floor=1e-6):
    """numerator / denominator, NaN where the denominator is unusable."""
    num, den = to_number(numerator), to_number(denominator)
    safe = den.where(den.abs() > floor)
    return num / safe


def interaction(attack, defence):
    """One fighter's offence against the other's defence.

    MMA is a matchup sport and a difference cannot express it: takedown
    defence matters enormously against a wrestler and not at all against a
    kickboxer. Both operands are centred first so the product is an
    interaction rather than a disguised main effect.
    """
    a, d = to_number(attack), to_number(defence)
    return (a - a.mean()) * (d - d.mean())


# --- declaration -----------------------------------------------------------

@dataclass(frozen=True)
class Paired:
    """One attribute measured on both fighters.

    Emits `<name>_diff` always, plus `<name>_level` and `<name>_known` when
    asked. `why` is required: a feature nobody can justify in one line is one
    nobody will notice has died.
    """

    name: str
    red: str
    blue: str
    why: str
    level: bool = False
    known: bool = False

    def build(self, df):
        out = {f"{self.name}_diff": paired_diff(df[self.red], df[self.blue])}
        if self.level:
            out[f"{self.name}_level"] = paired_level(df[self.red], df[self.blue])
        if self.known:
            out[f"{self.name}_known"] = paired_known(df[self.red], df[self.blue])
        return out

    @property
    def emits(self):
        names = [f"{self.name}_diff"]
        if self.level:
            names.append(f"{self.name}_level")
        if self.known:
            names.append(f"{self.name}_known")
        return names


@dataclass(frozen=True)
class Interaction:
    """A crossing of one fighter's offence with the other's defence."""

    name: str
    red_attack: str
    blue_defence: str
    why: str

    def build(self, df):
        return {self.name: interaction(df[self.red_attack], df[self.blue_defence])}

    @property
    def emits(self):
        return [self.name]


@dataclass(frozen=True)
class Derived:
    """A feature with its own function, for anything the shapes above miss."""

    name: str
    fn: object
    why: str
    needs: tuple = field(default_factory=tuple)

    def build(self, df):
        return {self.name: self.fn(df)}

    @property
    def emits(self):
        return [self.name]


def build_all(specs, df):
    """Apply every spec in order and return one DataFrame.

    Each spec sees the input frame plus everything built before it, so a
    feature can be defined in terms of an earlier one - the trajectory
    differences are built from the trajectory columns above them.

    That ordering matters for more than convenience. Without it a spec reading
    an earlier spec's output would silently fall through to a stale column of
    the same name left on the input frame, which is exactly how three dead
    trajectory features would have survived this rebuild.

    Building training rows and a single fight through the same code is what
    stops the two definitions drifting apart.
    """
    working = df.copy()
    out = {}
    for spec in specs:
        produced = spec.build(working)
        for name, values in produced.items():
            working[name] = values
        out.update(produced)
    return pd.DataFrame(out, index=df.index)


def emitted_names(specs):
    names = []
    for spec in specs:
        names.extend(spec.emits)
    return names
