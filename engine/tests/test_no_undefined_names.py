"""Undefined names in the network-only scripts fail here, not on the runner.

check_fight_changes.py's main() cannot run in tests - it needs Wikipedia - so
a constant renamed by a find-and-replace that matched nothing sailed through
every test and crashed the probe on the runner with a NameError, one CI round
later. Static analysis catches that class of mistake without any network.

predict_card.py is the other blind spot: it binds the whole training
pipeline with `globals().update(feature_frame.build(ufc))`, which no static
check can see, so pyflakes reports every pipeline name it uses as undefined
and a rename inside feature_frame._build would surface as a NameError ten
minutes into a runner job. One test here builds the pipeline over the
frozen 2006 fixture (a second) and holds predict_card's undefined names to
the names that build really exports. Another holds predict_card's 3,900
lines of module scope to one more rule pyflakes does not enforce: a name
bound by an import is never rebound further down - `_live_rows = []` once
shadowed the live_rows module in the three functions that catch its
NoLiveRow, so after the import every refusal raised AttributeError.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
FIXTURE = ENGINE / "tests" / "fixtures" / "archive_through_2006.csv"
NETWORK_ONLY = ["check_fight_changes.py", "fetch_fight_changes.py",
                "harvest_injuries.py", "probe_world.py", "harvest_world.py", "fight_night_weights.py", "probe_sherdog.py",
                "harvest_sherdog.py", "sherdog.py", "probe_ufcstats.py",
                "intel_backfill.py", "verify_intel_dates.py",
                "intel_search.py", "fight_intel.py",
                "experiments/short_notice_weight.py"]
# The live path: only reachable at runtime through the engine import.
LIVE_PATH = ["live_rows.py", "pending_rows.py", "feature_frame.py"]

pyflakes = pytest.importorskip("pyflakes")


def _undefined(script):
    result = subprocess.run([sys.executable, "-m", "pyflakes",
                             str(ENGINE / script)],
                            capture_output=True, text=True)
    return [line for line in result.stdout.splitlines()
            if "undefined name" in line]


@pytest.mark.parametrize("script", NETWORK_ONLY + LIVE_PATH)
def test_no_undefined_names(script):
    problems = _undefined(script)
    assert not problems, "\n".join(problems)


def _undefined_names(script):
    return sorted({line.split("undefined name ")[1].strip("'")
                   for line in _undefined(script)})


def test_predict_card_uses_only_names_the_pipeline_binds():
    """Against the real export - build() returns the locals of _build, so
    a name bound only on a path the archive does not take, or deleted
    before the return, is absent here as it would be in the engine."""
    import feature_frame
    exported = set(feature_frame.build(pd.read_csv(FIXTURE, low_memory=False),
                                       verbose=False))
    assert len(exported) > 100, "the pipeline exports hundreds of names"
    undefined = _undefined_names("predict_card.py")
    assert undefined, "predict_card reads the pipeline's names; none seen"
    missing = [n for n in undefined if n not in exported]
    assert missing == [], (
        f"predict_card.py uses names feature_frame.build does not export: "
        f"{missing}")


def rebound_imports(source):
    """{imported name: (import line, [lines that rebind it])} over a
    module's own scope - the bodies of its if/else, for, with and try
    blocks included, its functions, classes, lambdas and comprehensions
    (scopes of their own) not."""
    tree = ast.parse(source)
    own_scope = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                 ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    imports, stores = {}, {}

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, own_scope):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    imports.setdefault(alias.asname or alias.name.split(".")[0],
                                       child.lineno)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                stores.setdefault(child.id, []).append(child.lineno)
            walk(child)

    walk(tree)
    return {name: (line, stores[name]) for name, line in imports.items()
            if name in stores}


def test_rebound_imports_sees_a_module_scope_store_inside_a_block():
    source = ("import live_rows as _live_rows\n"
              "def f():\n    _live_rows = 1\n"       # its own scope: fine
              "if True:\n    _live_rows = []\n")     # module scope: not
    assert rebound_imports(source) == {"_live_rows": (1, [5])}
    assert rebound_imports("import os\nx = [os for os in range(2)]\n") == {}


@pytest.mark.parametrize("script", ["predict_card.py"])
def test_no_import_is_rebound_at_module_scope(script):
    rebound = rebound_imports((ENGINE / script).read_text())
    assert rebound == {}, "\n".join(
        f"{script}:{lines} rebinds {name!r}, imported at line {line}: every "
        f"function reading {name}.<attr> sees the new value at call time"
        for name, (line, lines) in rebound.items())
