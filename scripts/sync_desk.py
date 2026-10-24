"""Copy the shared Desk design system into both apps.

    python3 scripts/sync_desk.py          # write
    python3 scripts/sync_desk.py --check  # exit 1 if either app has drifted

shared/desk.css goes into each app's index.html and shared/desk.js into each
app's app.js, between marker comments. The copies are inlined rather than
linked so that each app stays one page and one script: the service worker
caches its own directory, and the standalone builders (engine/build_standalone.py,
nfl/engine/nfl_standalone.py) inline exactly those two files into the offline
page. Edit the shared files, never the copies - the tests compare them.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "shared"
APPS = (ROOT / "app", ROOT / "nfl" / "app")

CSS_BEGIN = "/* desk.css:begin - copied from shared/desk.css by scripts/sync_desk.py; edit that file */"
CSS_END = "/* desk.css:end */"
JS_BEGIN = "// desk.js:begin - copied from shared/desk.js by scripts/sync_desk.py; edit that file"
JS_END = "// desk.js:end"

TARGETS = (("index.html", "desk.css", CSS_BEGIN, CSS_END),
           ("app.js", "desk.js", JS_BEGIN, JS_END))


def splice(text, begin, end, body):
    """text with everything between the markers replaced by body."""
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one {begin!r} and one {end!r}")
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body.rstrip()}\n{end}{tail}"


def block(text, begin, end):
    """What sits between the markers."""
    return text.split(begin)[1].split(end)[0].strip("\n")


def expected(name):
    return (SHARED / name).read_text().rstrip()


def drift():
    """[(path, shared file)] for every copy that differs from its source."""
    out = []
    for app in APPS:
        for target, source, begin, end in TARGETS:
            path = app / target
            text = path.read_text()
            if text.count(begin) != 1 or block(text, begin, end).rstrip() != expected(source):
                out.append((path, source))
    return out


def sync():
    for app in APPS:
        for target, source, begin, end in TARGETS:
            path = app / target
            new = splice(path.read_text(), begin, end, expected(source))
            if new != path.read_text():
                path.write_text(new)
                print(f"updated {path.relative_to(ROOT)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        bad = drift()
        for path, source in bad:
            print(f"{path.relative_to(ROOT)} differs from shared/{source}")
        return 1 if bad else 0
    sync()
    return 0


if __name__ == "__main__":
    sys.exit(main())
