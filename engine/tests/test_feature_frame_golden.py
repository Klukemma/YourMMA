"""feature_frame.build must keep producing the rows it was proven to produce.

Every accuracy and backtest figure this project quotes is measured on rows
this pipeline builds, and a live prediction is one of its rows. An edit to
a fillna, a window or a threshold changes every training row with nothing
else failing - the bitwise-identical proof when the pipeline moved out of
predict_card.py was done once, by hand. This freezes it: the archive
through 2006 (fixtures/archive_through_2006.csv - a copy, so syncs and
rating rebuilds cannot move it; it builds in about a second and reaches
every modern branch but the women's and clinch columns) and, beside it,
the matrix it builds (feature_frame_golden_X.npz) with the feature list,
a hash of the matrix and each column's sum (feature_frame_golden.json).

The gate is a per-cell comparison at rtol 1e-9 / atol 1e-12, NaN for NaN:
a changed fillna, window or threshold fails it, a numpy or pandas release
that reorders a summation and moves last bits does not (the requirements
are unpinned, and the runner runs this suite before every dataset
commit). The exact hash is reported beside it: a run where only the hash
moved says so in a warning.

An intended feature change updates the golden deliberately, in the same
diff, with:

    python engine/tests/test_feature_frame_golden.py --regenerate
"""

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

FIXTURE = ENGINE / "tests" / "fixtures" / "archive_through_2006.csv"
GOLDEN = ENGINE / "tests" / "fixtures" / "feature_frame_golden.json"
GOLDEN_X = ENGINE / "tests" / "fixtures" / "feature_frame_golden_X.npz"
RTOL, ATOL = 1e-9, 1e-12


def describe(built):
    X = built["X"]
    values = X.to_numpy(float)
    return {
        "fixture": FIXTURE.name,
        "rows_in": None,
        "shape": list(values.shape),
        "feature_cols": list(built["feature_cols"]),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "column_sums": {c: float(np.nansum(values[:, i]))
                        for i, c in enumerate(X.columns)},
    }


def build_fixture():
    import feature_frame
    raw = pd.read_csv(FIXTURE, low_memory=False)
    built = feature_frame.build(raw, verbose=False)
    out = describe(built)
    out["rows_in"] = len(raw)
    return out, built["X"].to_numpy(float)


def regenerate():
    out, X = build_fixture()
    GOLDEN.write_text(json.dumps(out, indent=1))
    np.savez_compressed(GOLDEN_X, X=X)
    print(f"wrote {GOLDEN} and {GOLDEN_X.name}")


def same_cells(got, want):
    """Which cells agree: close, or NaN in both."""
    return (np.isclose(got, want, rtol=RTOL, atol=ATOL)
            | (np.isnan(got) & np.isnan(want)))


def test_feature_frame_builds_the_golden_rows():
    if not (FIXTURE.exists() and GOLDEN.exists() and GOLDEN_X.exists()):
        pytest.skip("golden fixture not present")
    want = json.loads(GOLDEN.read_text())
    want_X = np.load(GOLDEN_X)["X"]
    got, got_X = build_fixture()
    assert got["rows_in"] == want["rows_in"]
    assert got["feature_cols"] == want["feature_cols"], (
        "the feature list changed; if that is intended, regenerate the golden")
    assert got["shape"] == want["shape"] == list(want_X.shape)
    same = same_cells(got_X, want_X)
    if not same.all():
        moved = {c: int((~same[:, i]).sum())
                 for i, c in enumerate(want["feature_cols"]) if not same[:, i].all()}
        lines = [f"  {c}: {n} of {len(got_X)} rows, column sum "
                 f"{want['column_sums'][c]:.6f} -> {got['column_sums'][c]:.6f}"
                 for c, n in moved.items()]
        pytest.fail("feature_frame.build no longer produces the golden rows "
                    f"({len(moved)} columns moved beyond rtol {RTOL} / atol "
                    f"{ATOL}):\n" + "\n".join(lines)
                    + "\n  if the change is intended: python engine/tests/"
                    "test_feature_frame_golden.py --regenerate")
    if got["sha256"] != want["sha256"]:
        warnings.warn("feature_frame.build reproduces the golden rows within "
                      f"tolerance but not bit for bit (hash {want['sha256'][:12]} "
                      f"-> {got['sha256'][:12]}): a library release moving last "
                      "bits, most likely; nothing to fix unless a value moved")


def test_the_golden_gate_fails_on_a_moved_value_and_not_on_a_last_bit():
    want = np.array([[1.0, np.nan, 0.0], [2.0, 3.0, -1.0]])
    assert same_cells(want * (1 + 1e-15), want).all()
    moved = want.copy()
    moved[0, 2] = 1e-9
    assert not same_cells(moved, want).all()
    assert not same_cells(np.where(np.isnan(want), 0.0, want), want).all()


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        print(__doc__)
