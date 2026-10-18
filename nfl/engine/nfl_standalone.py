"""Build one HTML file that is the whole NFL app, data included.

The MMA app's standalone builder (engine/build_standalone.py), for the NFL
app: a page opened from a file:// URL may not fetch its neighbours, so the
phone copy carries everything - the page, the script and the week's data -
and works with aeroplane mode on.

    python3 nfl/engine/nfl_standalone.py

Writes nfl/app/yournfl.html. ONE SOURCE OF TRUTH: index.html and app.js are
read as they are and the data is injected as window.__YOURNFL_DATA__, which
app.js already prefers over fetch, so a change to the app reaches both
builds or neither.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
APP = ENGINE.parent / "app"
DATA = APP / "data"
OUT = APP / "yournfl.html"

PAYLOADS = ("week", "teams", "performance")

NEIGHBOUR_LINKS = re.compile(
    r'^[ \t]*<link rel="(?:manifest|icon|apple-touch-icon)"[^>]*>\n',
    flags=re.MULTILINE)


def drop_neighbour_links(html):
    """A manifest or icon link that resolves to nothing is worse than none."""
    stripped = NEIGHBOUR_LINKS.sub("", html)
    if 'rel="manifest"' in stripped:
        raise ValueError("index.html links a manifest this cannot strip")
    return stripped


def classic(app_js):
    """app.js is loaded as a module but must not use module syntax: inlined,
    it runs as a classic script, where import and export are errors."""
    if re.search(r"^\s*(import|export)\s", app_js, flags=re.MULTILINE):
        raise ValueError("app.js carries module syntax; the standalone build cannot inline it")
    return app_js


def build(out=OUT, data_dir=DATA):
    html = drop_neighbour_links((APP / "index.html").read_text())
    app = classic((APP / "app.js").read_text())
    data, missing = {}, []
    for name in PAYLOADS:
        path = Path(data_dir) / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        data[name] = json.loads(path.read_text())
    if missing:
        raise FileNotFoundError("missing app data: " + ", ".join(missing)
                                + ". Run nfl/engine/nfl_predict.py first.")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # </ inside JSON would close the script block early.
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    inline = (f'<script>window.__YOURNFL_BUILT__ = "{stamp}";\n'
              f"window.__YOURNFL_DATA__ = {blob};</script>\n"
              f"<script>\n{app}\n</script>")
    tag = '<script type="module" src="app.js"></script>'
    if tag not in html:
        raise ValueError("index.html no longer loads app.js the expected way")
    out = Path(out)
    out.write_text(html.replace(tag, inline))
    print(f"wrote {out} ({out.stat().st_size / 1024:,.0f} KB), built {stamp}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()
    build(Path(args.out), Path(args.data))


if __name__ == "__main__":
    main()
