"""Build one HTML file that is the whole app, data included.

The served build fetches data/*.json from a directory. That is right for a
host, and useless on a phone with no connection: a page opened from a file://
URL is not permitted to fetch its own neighbours, and an ES module cannot
import one either. So a phone needs the opposite shape - everything on the
page already, nothing to ask for.

    python3 engine/build_standalone.py

Writes app/yourmma.html. Save it to a phone, open it, Add to Home Screen. It
works with the aeroplane mode on.

ONE SOURCE OF TRUTH. This transforms the served files rather than duplicating
them: index.html, matchup.js and app.js are read as they are, the module
syntax is stripped (a module needs a server; a classic script does not), and
the JSON is injected as window.__YOURMMA_DATA__, which app.js already prefers
over fetch. A change to the app reaches both builds or neither.

The fonts stay linked. Online they load; offline the page falls back to the
system stack it already declares, which is what the fallback is for.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
APP = ENGINE.parent / "app"
DATA = APP / "data"
OUT = APP / "yourmma.html"

# Every payload the app asks for by name, in load().
PAYLOADS = ("card", "fighters", "names", "performance", "constants")


def strip_modules(matchup_js, app_js):
    """Turn two ES modules into one classic script.

    `export` and `import` are syntax errors outside a module, and a module
    cannot be inlined without also being able to resolve ./matchup.js. So the
    exports become plain declarations and the import line goes, which is sound
    only because the two files are concatenated in dependency order.
    """
    matchup = re.sub(r"^export\s+function\s+", "function ", matchup_js,
                     flags=re.MULTILINE)
    if "export" in matchup:
        raise ValueError("matchup.js still carries an export this cannot strip")

    app = re.sub(r'^import\s+\{[^}]*\}\s+from\s+"\./matchup\.js";\s*$', "",
                 app_js, flags=re.MULTILINE)
    if app == app_js:
        raise ValueError("app.js import line did not match; refusing to inline")
    if re.search(r"^\s*(import|export)\s", app, flags=re.MULTILINE):
        raise ValueError("app.js still carries module syntax this cannot strip")
    return matchup, app


def build(out=OUT):
    html = (APP / "index.html").read_text()
    matchup, app = strip_modules((APP / "matchup.js").read_text(),
                                 (APP / "app.js").read_text())

    data, missing = {}, []
    for name in PAYLOADS:
        path = DATA / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        data[name] = json.loads(path.read_text())
    if missing:
        raise FileNotFoundError(
            "missing app data: " + ", ".join(missing)
            + ". Run build_app_data.py, and predict_card.py for the card.")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # </script> inside a JSON string would close this block early. The escape
    # is invisible to JSON.parse and cannot appear in a fighter's name by
    # accident, but a reason string quoting HTML could carry it.
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
    blob = blob.replace("</", "<\\/")

    inline = (
        f'<script>window.__YOURMMA_BUILT__ = "{stamp}";\n'
        f"window.__YOURMMA_DATA__ = {blob};</script>\n"
        f"<script>\n{matchup}\n{app}\n</script>"
    )

    tag = '<script type="module" src="app.js"></script>'
    if tag not in html:
        raise ValueError("index.html no longer loads app.js the expected way")
    html = html.replace(tag, inline)

    out = Path(out)
    out.write_text(html)
    size = out.stat().st_size / 1024
    print(f"wrote {out} ({size:,.0f} KB)")
    print(f"  payloads: {', '.join(sorted(data))}")
    print(f"  built:    {stamp}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
