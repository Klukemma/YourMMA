"""Check the app still meets a browser's bar for "Install app".

A phone offers to install a page only if a short list of things is all true at
once, and the failure is silent: the button simply does not appear, with no
console error naming what is missing. That is a bad thing to find out from a
phone, so the list is asserted here.

Chromium's criteria, which is the strictest of the ones that matter:

  - a linked manifest, valid JSON
  - name (or short_name), start_url, and display of standalone/fullscreen/
    minimal-ui
  - an icon of at least 192x192 and one of at least 512x512, both PNG
  - a registered service worker with a fetch handler
  - served over HTTPS (GitHub Pages is; nothing to assert)

iOS ignores all of it and reads apple-touch-icon and the apple-mobile-web-app
meta tags instead, so those are checked too.

The icon dimensions are read out of the PNG headers rather than trusted from
the manifest, because a manifest that claims 512x512 over a 192x192 file is
exactly the kind of mismatch that turns the prompt off.
"""

import json
import re
import struct
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import build_standalone as bs

APP = ENGINE.parent / "app"
INDEX = APP / "index.html"
MANIFEST = APP / "manifest.json"
SW = APP / "sw.js"

INSTALLABLE_DISPLAY = {"standalone", "fullscreen", "minimal-ui"}


def png_size(path):
    """Width and height from the IHDR, and a check that it is really a PNG."""
    head = Path(path).read_bytes()[:26]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert head[12:16] == b"IHDR", f"{path} has no IHDR first"
    width, height = struct.unpack(">II", head[16:24])
    return width, height


@pytest.fixture(scope="module")
def html():
    return INDEX.read_text()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_page_links_the_manifest(html):
    assert re.search(r'<link rel="manifest" href="manifest\.json">', html)


def test_manifest_has_the_required_fields(manifest):
    assert manifest.get("name") or manifest.get("short_name")
    assert manifest["start_url"]
    assert manifest["display"] in INSTALLABLE_DISPLAY


def test_manifest_icons_exist_and_are_the_size_they_claim(manifest):
    for icon in manifest["icons"]:
        path = APP / icon["src"]
        assert path.exists(), f"{icon['src']} is in the manifest but not on disk"
        assert icon["type"] == "image/png"
        claimed = tuple(int(n) for n in icon["sizes"].split("x"))
        assert png_size(path) == claimed, f"{icon['src']} is not {icon['sizes']}"


def test_manifest_covers_both_required_icon_sizes(manifest):
    have = {tuple(int(n) for n in i["sizes"].split("x"))
            for i in manifest["icons"] if "any" in i.get("purpose", "any")}
    assert any(w >= 192 and h >= 192 for w, h in have), "no icon of 192px or more"
    assert any(w >= 512 and h >= 512 for w, h in have), "no icon of 512px or more"


def test_a_maskable_icon_is_offered(manifest):
    """Without one, Android crops the square icon and clips the mark."""
    maskable = [i for i in manifest["icons"] if "maskable" in i.get("purpose", "")]
    assert maskable, "no maskable icon; Android will crop the square one"


def test_start_url_is_inside_the_scope(manifest):
    """A start_url outside the scope disqualifies the manifest outright."""
    scope = manifest.get("scope", ".")
    assert manifest["start_url"].lstrip("./") .startswith(scope.lstrip("./"))


def test_the_page_registers_a_service_worker():
    app_js = (APP / "app.js").read_text()
    assert 'navigator.serviceWorker.register("sw.js")' in app_js
    # file:// has no origin to scope a worker to, so the standalone build must
    # not try; the guard is what keeps it from throwing there.
    assert 'location.protocol.startsWith("http")' in app_js


def test_the_service_worker_handles_fetch():
    """A worker with no fetch handler does not make a site installable."""
    assert SW.exists()
    assert re.search(r'addEventListener\(\s*"fetch"', SW.read_text())


def test_the_service_worker_precaches_files_that_exist():
    """A precache list pointing at a missing file caches a 404 body."""
    source = SW.read_text()
    listed = re.search(r"const SHELL = \[(.*?)\];", source, re.DOTALL)
    assert listed, "SHELL list not found; this test is reading the wrong shape"
    for url in re.findall(r'"([^"]+)"', listed.group(1)):
        if url == "./":
            continue  # the directory, served as index.html
        assert (APP / url).exists(), f"sw.js precaches {url}, which is missing"


def test_data_is_not_served_from_cache_first():
    """Yesterday's card shown as today's is the lie this project avoids."""
    source = SW.read_text()
    assert re.search(r'includes\("/data/"\)\)\s*return networkFirst', source)


def test_ios_add_to_home_screen_tags_are_present(html):
    """iOS never reads the manifest; these are the whole of what it reads."""
    assert 'rel="apple-touch-icon" href="icons/apple-touch-icon.png"' in html
    assert 'name="apple-mobile-web-app-capable" content="yes"' in html
    assert 'name="apple-mobile-web-app-title"' in html
    assert (APP / "icons" / "apple-touch-icon.png").exists()


def test_a_theme_colour_is_declared_for_both_schemes(html):
    schemes = re.findall(r'<meta name="theme-color"[^>]*media="\(prefers-color-scheme: (\w+)\)"',
                         html)
    assert set(schemes) == {"light", "dark"}


def test_standalone_build_drops_links_it_cannot_resolve():
    """A file:// page cannot fetch a manifest, and a broken one is worse than
    none: it is what the browser reads before offering Add to Home Screen."""
    stripped = bs.drop_neighbour_links(INDEX.read_text())
    assert 'rel="manifest"' not in stripped
    assert 'href="icons/' not in stripped
    # Everything else survives: this must not eat the stylesheet or the fonts.
    assert 'rel="stylesheet"' in stripped
    assert 'rel="preconnect"' in stripped
    assert 'name="apple-mobile-web-app-capable"' in stripped


def test_standalone_build_refuses_a_manifest_it_cannot_strip():
    """Mutation guard: if the regex stops matching, the build must fail loudly
    rather than quietly shipping a file:// page linking a manifest."""
    with pytest.raises(ValueError, match="manifest"):
        bs.drop_neighbour_links('<link  rel="manifest" href="manifest.json">')
