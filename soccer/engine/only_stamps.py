"""Put back the runner's files whose only change is when the run happened.

    python soccer/engine/only_stamps.py            # in the checkout, before `git status`

Every predict run rewrites matches.json, tables.json and performance.json
("generated"), the feeds' snapshot (each request's answer time, last_run.at)
and the prediction log (each open pick's last_logged), and rebuilds the
standalone page, which embeds the three payloads and its build time. So the
commit step's "Nothing changed." never fired: about 25 runs a week each
committed ~4.6 MB of rewritten files (about 0.5 MB compressed) on days when no
pick, price or result had moved - the standalone page about 45% of it.

WHY THIS IS HONEST. A file is put back only when, with its stamps left out,
it is exactly the file committed: the committed copy then says the same
thing, stamped at the earlier moment it was first true. Nothing is invented
or moved later - an older stamp on identical content is still a true stamp
(a pick last logged earlier, with the same numbers, was logged then; a feed
row first seen earlier was seen then), so point in time holds. The app
prints "generated", so an unchanged page says its age.

  * Nothing but stamps changed anywhere the runner commits: every such file
    is put back, and the page with them - the commit step finds nothing.
  * Something real changed: it is committed, stamps and all - except the
    app's three payloads and the page when the payloads' only change is
    their stamps: the page is a pure function of them (and of index.html and
    app.js, which the runner never changes), so it is rebuilt in effect only
    when its data changed, and the payloads and the page stay one consistent
    snapshot.

A file the commit has never seen, or that is not JSON, is a real change.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOADS = ("soccer/app/data/matches.json", "soccer/app/data/tables.json", "soccer/app/data/performance.json")
PAGE = "soccer/app/yoursoccer.html"
STAMPED = PAYLOADS + ("soccer/data/feed.json", "soccer/data/prediction_history.json")
# The keys that only say when: the payloads' build time, the feeds' answer
# times, a pick's last re-log, matches.json's feed and sync times and a
# download problem's time. Never a kickoff, a price's first-seen time, a
# result's seen_at or graded_at: those change only with what they stamp.
STAMP_KEYS = frozenset({"generated", "at", "feeds_at", "logged_at", "last_logged", "data_synced_at"})
# Maps whose every value is a stamp (soccer_feed's {request: answered at}).
STAMP_MAPS = frozenset({"requests"})


def unstamped(x):
    """x with every stamp left out (STAMP_KEYS, and STAMP_MAPS' values)."""
    if isinstance(x, dict):
        return {k: (sorted(v) if k in STAMP_MAPS and isinstance(v, dict) else unstamped(v))
                for k, v in x.items() if k not in STAMP_KEYS}
    if isinstance(x, list):
        return [unstamped(v) for v in x]
    return x


def _git(root, *args, check=True):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=check)


def committed(root, path):
    """The file as HEAD has it (bytes), or None."""
    r = _git(root, "show", f"HEAD:{path}", check=False)
    return r.stdout if r.returncode == 0 else None


def changed(root, path):
    """Whether the file differs from HEAD (new, removed or modified)."""
    return bool(_git(root, "status", "--porcelain", "--", path).stdout.strip())


def only_stamps(root, path):
    """Whether the file's only change from HEAD is its stamps."""
    old, new = committed(root, path), Path(root, path)
    if old is None or not new.is_file():
        return False
    try:
        return unstamped(json.loads(old)) == unstamped(json.loads(new.read_bytes()))
    except ValueError:
        return False


def restore(root, paths):
    for p in paths:
        _git(root, "checkout", "HEAD", "--", p)


def settle(root=ROOT, paths=None, log=print):
    """Put back what changed only in its stamps (the module's rules) under
    `paths` (the commit step's PATHS). -> the files put back."""
    root = Path(root)
    paths = list(paths or ("soccer/data/", "soccer/app/data/", PAGE))
    out = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths).stdout.decode()
    touched = [line[3:].strip().strip('"') for line in out.splitlines() if line.strip()]
    stamped = {p for p in touched if p in STAMPED and only_stamps(root, p)}
    real = [p for p in touched if p not in stamped and p != PAGE]
    payloads_same = all(p in stamped or not changed(root, p) for p in PAYLOADS)
    if not real:
        back = sorted(stamped | ({PAGE} if PAGE in touched and payloads_same else set()))
    elif payloads_same:
        back = sorted({p for p in stamped if p in PAYLOADS} | ({PAGE} if PAGE in touched else set()))
    else:
        back = []
    restore(root, back)
    for p in back:
        log(f"only its stamps changed, put back: {p}")
    if real:
        log(f"changed: {', '.join(real)}")
    return back


if __name__ == "__main__":
    settle(paths=sys.argv[1:] or None)
