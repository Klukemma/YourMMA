"""Find constants that no longer match the scale of the data they touch.

Five Elo-era constants survived the switch to TrueSkill unnoticed, and the
worst of them was invisible by reading:

    loss_scale = 0.3 if opp_mmr_val > 1600 else (0.5 if opp_mmr_val > 1550 ...)

No TrueSkill rating can reach 1450, so every branch was false and the whole
adjustment returned 1.0 for every fight ever scored. Nothing crashed, no value
looked odd, and the line reads perfectly sensibly. Only comparing the literal
against the column's actual range shows it.

This does that comparison for the whole file. Two checks:

  DEAD THRESHOLD   a literal compared against a column, sitting outside the
                   range that column ever takes, so the comparison has a
                   constant answer and the branch it guards is unreachable.

  ABSURD DEFAULT   a fillna or .get default outside the column's range, so a
                   missing value becomes a number the data could never produce.
                   fillna(1500) on a rating spanning -25..+31 is a 60x outlier
                   that reads as an ordinary default.

Both are heuristics over an AST, so they only see columns that exist in the
dataset - a feature built in memory has no range to check against. They report
what to look at, not what is certainly wrong.

    python engine/audit_features.py
"""

import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
DATA_DIR = ENGINE / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
DEFAULT_TARGET = ENGINE / "predict_card.py"

# A default this far outside the observed range is worth a look even when it is
# technically reachable, because it is almost always a leftover from a rescale.
RANGE_PAD = 0.0


# Findings a human has looked at and accepted, as (column, threshold): reason.
# A defensive guard against impossible data is unreachable on purpose; a
# constant left over from a rescale is not. The difference cannot be decided
# automatically, so it is written down here rather than guessed.
KNOWN_BENIGN = {
    ("match_time_sec", 0.0):
        "guard against a zero or negative fight duration before dividing by it",
}


def is_benign(finding):
    key = (finding["column"], finding.get("threshold", finding.get("default")))
    return key in KNOWN_BENIGN


def column_ranges(csv_path=None):
    """{column: (min, max)} for every numeric column in the dataset."""
    df = pd.read_csv(csv_path or UFC_CSV, low_memory=False)
    ranges = {}
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values):
            ranges[col] = (float(values.min()), float(values.max()))
    return ranges


def _string_literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return None if inner is None else -inner
    return None


def column_of(node):
    """The dataset column a node reads, if it plainly reads one.

    Recognises df['col'], df.get('col'), df.get('col', default) and
    df['col'].fillna(...), which is how every access in this codebase is
    written.
    """
    if isinstance(node, ast.Subscript):
        return _string_literal(node.slice)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "get" and node.args:
            return _string_literal(node.args[0])
        if node.func.attr in ("fillna", "astype", "clip", "abs"):
            return column_of(node.func.value)
    return None


def default_of(node):
    """The fallback value a node supplies for a missing entry, if any."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "fillna" and node.args:
            return _number(node.args[0])
        if node.func.attr == "get" and len(node.args) > 1:
            return _number(node.args[1])
    return None


class _Collector(ast.NodeVisitor):
    """Track variables that hold a column, then check comparisons against them."""

    def __init__(self, ranges):
        self.ranges = ranges
        self.dead = []
        self.defaults = []
        self._aliases = {}

    def visit_Assign(self, node):
        col = column_of(node.value)
        if col and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self._aliases[node.targets[0].id] = col
        self._record_default(node.value)
        self.generic_visit(node)

    def _record_default(self, node):
        col, default = column_of(node), default_of(node)
        if col and default is not None and col in self.ranges:
            low, high = self.ranges[col]
            if not (low - RANGE_PAD <= default <= high + RANGE_PAD):
                self.defaults.append({
                    "line": node.lineno, "column": col, "default": default,
                    "range": (low, high)})

    def visit_Call(self, node):
        self._record_default(node)
        self.generic_visit(node)

    def _resolve(self, node):
        if isinstance(node, ast.Name):
            return self._aliases.get(node.id)
        return column_of(node)

    def visit_Compare(self, node):
        col = self._resolve(node.left)
        if col and col in self.ranges:
            low, high = self.ranges[col]
            for op, comparator in zip(node.ops, node.comparators):
                threshold = _number(comparator)
                if threshold is None:
                    continue
                if isinstance(op, (ast.Gt, ast.GtE)) and threshold > high:
                    self.dead.append(self._finding(node, col, threshold, "never above"))
                elif isinstance(op, (ast.Lt, ast.LtE)) and threshold < low:
                    self.dead.append(self._finding(node, col, threshold, "never below"))
        self.generic_visit(node)

    def _finding(self, node, col, threshold, why):
        low, high = self.ranges[col]
        return {"line": node.lineno, "column": col, "threshold": threshold,
                "range": (low, high), "why": why}


def audit(path=None, ranges=None, include_benign=False):
    """Returns (dead_thresholds, absurd_defaults), accepted findings removed."""
    path = Path(path or DEFAULT_TARGET)
    ranges = ranges if ranges is not None else column_ranges()
    collector = _Collector(ranges)
    collector.visit(ast.parse(path.read_text()))
    if include_benign:
        return collector.dead, collector.defaults
    return ([f for f in collector.dead if not is_benign(f)],
            [f for f in collector.defaults if not is_benign(f)])



# A per-fight prediction builds its features as a hand-written dict, which has
# to match the training feature list exactly. Nothing enforced that, so adding
# mu_sum and mu_diff_z to the list broke every prediction with a KeyError - and
# only when the pipeline was actually run, ten minutes in.
MIN_KEYS_FOR_FEATURE_DICT = 20
FEATURE_DICT_MATCH = 0.5


def feature_group_lists(tree):
    """{name: [feature names]} for every `something_features = [...]` list."""
    groups = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.endswith("_features"):
            continue
        if not isinstance(node.value, ast.List):
            continue
        names = [_string_literal(e) for e in node.value.elts]
        groups[target.id] = [n for n in names if n]
    return groups


def declared_features(tree):
    """Every feature name reachable from a `feature_cols*` assignment."""
    groups = feature_group_lists(tree)
    wanted = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("feature_cols"):
            continue
        for part in ast.walk(node.value):
            if isinstance(part, ast.Name) and part.id in groups:
                wanted.update(groups[part.id])
    return wanted


def prediction_feature_dicts(tree, known):
    """Dict literals that look like a hand-built feature row."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {_string_literal(k) for k in node.keys if k is not None}
        keys.discard(None)
        if len(keys) < MIN_KEYS_FOR_FEATURE_DICT:
            continue
        overlap = len(keys & known) / len(keys)
        if overlap >= FEATURE_DICT_MATCH:
            found.append((node.lineno, keys))
    return found


def missing_prediction_features(path=None):
    """Features the model is trained on that a prediction never supplies.

    Returns [{line, missing}] - one entry per hand-built feature dict.
    """
    path = Path(path or DEFAULT_TARGET)
    tree = ast.parse(path.read_text())
    wanted = declared_features(tree)
    if not wanted:
        return []
    out = []
    for lineno, keys in prediction_feature_dicts(tree, wanted):
        missing = sorted(wanted - keys)
        if missing:
            out.append({"line": lineno, "missing": missing})
    return out


def report(dead, defaults, missing=None):
    print("=" * 72)
    print(f"DEAD THRESHOLDS: {len(dead)}")
    print("  a literal the column never reaches, so the branch cannot run")
    print("=" * 72)
    for f in dead:
        low, high = f["range"]
        print(f"  line {f['line']:>5}  {f['column']:<22} compared to {f['threshold']:>12,.2f}"
              f"   ({f['why']}; column runs {low:,.2f} to {high:,.2f})")
    if not dead:
        print("  none")

    print()
    print("=" * 72)
    print(f"ABSURD DEFAULTS: {len(defaults)}")
    print("  a fallback outside the range the column can actually take")
    print("=" * 72)
    for f in defaults:
        low, high = f["range"]
        print(f"  line {f['line']:>5}  {f['column']:<22} defaults to {f['default']:>12,.2f}"
              f"   (column runs {low:,.2f} to {high:,.2f})")
    if not defaults:
        print("  none")

    missing = missing or []
    print()
    print("=" * 72)
    print(f"FEATURES A PREDICTION CANNOT SUPPLY: {len(missing)}")
    print("  trained on it, but the per-fight dict never sets it")
    print("=" * 72)
    for f in missing:
        print(f"  line {f['line']:>5}  missing {len(f['missing'])}: "
              f"{', '.join(f['missing'])}")
    if not missing:
        print("  none")

    return len(dead) + len(defaults) + len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=str(DEFAULT_TARGET))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when anything is found")
    args = ap.parse_args()
    dead, defaults = audit(args.target)
    total = report(dead, defaults, missing_prediction_features(args.target))
    if args.strict and total:
        sys.exit(1)


if __name__ == "__main__":
    main()
