"""The runner is thrown away; what it does not commit, it loses.

This file exists because of a bug that cost real money and real data. The odds
cache was written to hold prices so the same fight is never bought twice, and
the workflow committed only app/data/. So every CI run re-fetched every price,
and the cache's entire purpose went unserved for as long as it existed - with
nothing failing, because a cache that is always empty behaves exactly like a
cache that is always missing.

The second cost is worse than the credits. A moneyline only exists while the
market is open; once a fight starts the bookmakers pull it. A price fetched at
20:39 and not committed is not re-bought later, it is gone - which is how a
card that was fully priced an hour before the bell ended up with one price.

So: if a step writes a file the next run needs, the workflow has to commit it.
"""

import re
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import odds_cache

WORKFLOW = ENGINE.parent / ".github" / "workflows" / "update-dataset.yml"

# Every file a run writes that a LATER run reads. Anything here must survive
# the runner being destroyed, which means being committed.
MUST_PERSIST = (
    "app/data/",                     # what the phone reads
    "engine/data/odds_cache.json",   # prices already paid for
    "engine/data/odds.csv",          # settled lines the backtest scores against
    "engine/data/method_odds.csv",   # the market's price on how a fight ends
)


@pytest.fixture(scope="module")
def workflow():
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def commit_step(workflow):
    """The commands the step runs, with comments stripped.

    Comments are stripped because this file explains the bug it guards against
    by name, and a test that searched the whole step would match the
    explanation rather than the code."""
    start = workflow.index("- name: Commit the app data")
    rest = workflow[start + 10:]
    end = rest.find("\n      - name:")
    body = rest[:end if end != -1 else len(rest)]
    return "\n".join(line for line in body.splitlines()
                     if not line.lstrip().startswith("#"))


def test_the_commit_step_stages_everything_a_later_run_needs(commit_step):
    for path in MUST_PERSIST:
        assert path in commit_step, (
            f"{path} is written by a run and read by the next one, but the "
            f"workflow never commits it, so the runner takes it to the grave")


def test_the_change_check_covers_the_same_paths(commit_step):
    """A guard that only watches app/data/ exits before committing the odds
    on any run where the card happens not to change - staging a path the guard
    does not watch is the same as not staging it.

    The assertion is on the `git status` line itself, not on the step body:
    the staged paths are written out just above it, so searching the body
    would find them there and pass whatever the guard actually checks."""
    line = next(l for l in commit_step.splitlines()
                if "git status --porcelain" in l)
    staged = next(l for l in commit_step.splitlines() if l.strip().startswith("PATHS="))
    variable = staged.split("=", 1)[0].strip()
    if f"${variable}" in line:
        return  # the guard watches exactly what is staged, by construction
    for path in MUST_PERSIST:
        assert path in line, (
            f"{path} is staged but the change check does not watch it, so a "
            f"run where only that file moved commits nothing")


def test_the_cache_path_the_workflow_commits_is_the_one_the_code_writes():
    """Two spellings of the same path is how this breaks again."""
    relative = odds_cache.CACHE.relative_to(ENGINE.parent).as_posix()
    assert relative in MUST_PERSIST
    assert relative in WORKFLOW.read_text()


def test_the_odds_csv_path_the_workflow_commits_is_the_one_the_code_writes():
    relative = odds_cache.ODDS_CSV.relative_to(ENGINE.parent).as_posix()
    assert relative in MUST_PERSIST
    assert relative in WORKFLOW.read_text()


def test_the_priced_count_reads_a_field_the_export_actually_writes(commit_step):
    """The old count grepped for `best_odds`, which is the engine's internal
    name and not what card.json carries, so it printed 0 on a fully priced
    card and put that in the commit log."""
    assert "best_odds" not in commit_step
    assert "f.get('odds')" in commit_step


def test_card_json_carries_odds_under_the_name_the_count_reads():
    """Pins the count to the schema rather than to a hopeful string."""
    import app_export as ax
    payload = ax.card_payload("E", "2026-01-01",
                              [{"number": 1, "red": "A", "blue": "B",
                                "pick": "A", "odds": -150}])
    assert payload["fights"][0]["odds"] == -150


def test_the_commit_step_never_adds_a_path_that_may_not_exist():
    """`git add` on a missing pathspec is fatal. A mode that does not create
    every tracked data file - the scout's pending queue when nothing is
    pending - would fail at the commit step with its work done and unsaved,
    which is what happened on the first check-fight-changes run after the
    scout's files were added to the list."""
    text = (Path(__file__).resolve().parents[2] / ".github" / "workflows"
            / "update-dataset.yml").read_text()
    assert "git add $PATHS" not in text
    assert '[ -e "$p" ]' in text


def test_runs_on_one_branch_queue_instead_of_racing():
    # Two concurrent runs both regenerate app/data/*.json; the loser's rebase
    # conflicts on generated files and its work is thrown away.
    text = WORKFLOW.read_text()
    block = re.search(r"^concurrency:\n((?:  .*\n)+)", text, re.M)
    assert block, "workflow has no top-level concurrency group"
    assert "github.ref" in block.group(1)
    assert "cancel-in-progress: false" in block.group(1)


def test_a_queued_run_checks_out_the_branch_not_its_dispatch_commit():
    # Queueing only helps if the second run starts from what the first one
    # pushed; the default checkout is the SHA at dispatch time.
    text = WORKFLOW.read_text()
    step = re.search(r"uses: actions/checkout@v\d+\n(\s+with:\n(?:\s{10,}.*\n)+)",
                     text)
    assert step, "checkout has no 'with:' block"
    assert "ref: ${{ github.ref }}" in step.group(1)
