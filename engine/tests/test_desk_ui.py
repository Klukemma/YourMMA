"""The apps are one family: one stylesheet, one app chrome, each copy
identical to its source in shared/, and an install path a phone can find.

The NFL app installed fine on paper and still read as "only works on the web"
to the person using it: nothing on the page said it could be installed, or
how. So the install offer is part of the page now, on both apps, and this
checks it stays there.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPS = {"mma": ROOT / "app", "nfl": ROOT / "nfl" / "app", "soccer": ROOT / "soccer" / "app"}
sys.path.insert(0, str(ROOT / "scripts"))

import sync_desk  # noqa: E402


def test_every_app_carries_the_shared_design_unchanged():
    """Edit shared/desk.css or desk.js and run scripts/sync_desk.py."""
    assert sync_desk.drift() == []
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_desk.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_the_copies_sit_inside_the_page_and_the_script():
    for app in APPS.values():
        html = (app / "index.html").read_text()
        js = (app / "app.js").read_text()
        style = html.split("<style>")[1].split("</style>")[0]
        assert sync_desk.CSS_BEGIN in style and sync_desk.CSS_END in style
        assert js.count(sync_desk.JS_BEGIN) == 1
        # Nothing styles the page from outside the shared sheet.
        assert style.strip().startswith(sync_desk.CSS_BEGIN)


def test_the_shared_script_has_no_module_syntax():
    """It is inlined into classic scripts by both standalone builders."""
    js = (ROOT / "shared" / "desk.js").read_text()
    assert not re.search(r"^\s*(import|export)\s", js, flags=re.M)


def test_every_app_offers_install_and_explains_it_on_an_iphone():
    js = (ROOT / "shared" / "desk.js").read_text()
    assert "beforeinstallprompt" in js and "e.prompt()" in js
    assert "Add to Home Screen" in js            # Safari never fires the event
    assert "Open it in your browser" in js       # inside another app's viewer
    assert "display-mode: standalone" in js      # no offer once installed
    for app in APPS.values():
        assert "DESK.banner()" in (app / "app.js").read_text()
        assert 'id="menu-btn"' in (app / "index.html").read_text()


def test_the_sport_switch_links_each_app_to_the_others_in_one_order():
    orders = []
    for name, app in APPS.items():
        html = (app / "index.html").read_text()
        nav = re.findall(r'<nav class="switch"[^>]*>.*?</nav>', html, flags=re.S)[0]
        links = re.findall(r'href="([^"]+)"', nav)
        targets = {(app / link / "index.html").resolve() for link in links}
        others = {(a / "index.html").resolve() for n, a in APPS.items() if n != name}
        assert targets == others, (name, links)
        orders.append(re.findall(r'>(MMA|NFL|Soccer)<', nav))
    assert all(o == ["MMA", "NFL", "Soccer"] for o in orders), orders


def test_the_mma_search_does_not_rely_on_a_datalist():
    """iOS shows a datalist badly or not at all; the picker draws its own."""
    js = (APPS["mma"] / "app.js").read_text()
    assert "<datalist" not in js and 'class="suggest"' in js


def test_the_tab_bars_have_the_same_shape():
    shapes = []
    for app in APPS.values():
        html = (app / "index.html").read_text()
        tabs = re.findall(r'<button role="tab" id="tab-(\w+)"', html)
        assert len(tabs) == 4
        assert html.count('<span class="pill"><svg') == 4
        shapes.append(len(tabs))
    assert len(set(shapes)) == 1


def test_every_script_parses_as_the_phone_runs_them():
    """`node --check` let a duplicate top-level declaration through; a
    browser does not, and one SyntaxError blanks every tab. Parse each script
    the way it really runs: the MMA one with its import removed, as the
    standalone builder does, the others as they are."""
    import shutil
    import pytest
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    for name, app in APPS.items():
        src = (app / "app.js").read_text()
        if name == "mma":
            src = re.sub(r'^import\s+\{[^}]*\}\s+from\s+"\./matchup\.js";\s*$', "", src, flags=re.M)
        out = subprocess.run(["node", "-e", "new Function(require('fs').readFileSync(0, 'utf8'))"],
                             input=src, capture_output=True, text=True)
        assert out.returncode == 0, f"{name}: {out.stderr[-400:]}"
