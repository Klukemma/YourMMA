"""The phone app: installable, offline, its own cache and storage on the
origin it shares with the MMA app, and its one formula matching Python's."""

import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

import nfl_elo as ne

APP = Path(__file__).resolve().parents[2] / "app"


def _png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_manifest_icons_exist_at_the_sizes_claimed():
    manifest = json.loads((APP / "manifest.json").read_text())
    assert manifest["start_url"] == "." and manifest["scope"] == "."
    purposes = set()
    for icon in manifest["icons"]:
        w, h = (int(x) for x in icon["sizes"].split("x"))
        assert _png_size(APP / icon["src"]) == (w, h)
        purposes.add(icon["purpose"])
    assert {"any", "maskable"} <= purposes
    assert _png_size(APP / "icons" / "apple-touch-icon.png") == (180, 180)


def test_the_page_is_installable_on_both_platforms():
    html = (APP / "index.html").read_text()
    assert '<link rel="manifest" href="manifest.json">' in html
    assert 'rel="apple-touch-icon"' in html and 'apple-mobile-web-app-capable' in html
    assert html.count('name="theme-color"') == 2
    assert '<script type="module" src="app.js"></script>' in html


def test_the_service_worker_keeps_to_its_own_caches():
    sw = (APP / "sw.js").read_text()
    # The MMA worker deletes every "yourmma-" cache on activate.
    assert re.search(r'const CACHE = "yournfl-v\d+"', sw)
    assert 'startsWith("yournfl-")' in sw and "yourmma-" not in sw.split("*/")[-1].replace("yourmma-\" cache", "")
    # The data, and the offline page that carries data inside it, network first.
    assert 'url.pathname.includes("/data/") || url.pathname.endsWith("/yournfl.html")) return networkFirst' in sw
    for path in re.findall(r'^\s+"([^"]+)",?$', sw.split("const SHELL = [")[1].split("];")[0], flags=re.M):
        assert (APP / path).exists() or path == "./", path


def test_the_app_has_its_own_storage_key_and_no_module_syntax():
    js = (APP / "app.js").read_text()
    assert '"yournfl.tab"' in js and "yourmma.tab" not in js
    assert not re.search(r"^\s*(import|export)\s", js, flags=re.M)
    assert 'register("sw.js")' in js and 'location.protocol.startsWith("http")' in js
    assert "window.__YOURNFL_DATA__" in js


def test_every_team_row_shows_the_record_and_last_five():
    js = (APP / "app.js").read_text()
    assert "formLine(g[`${side}_form`])" in js
    assert "formDetail(g.away, g.away_form)" in js and "formDetail(g.home, g.home_form)" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_apps_elo_chance_matches_python():
    js = (APP / "app.js").read_text()
    fn = re.search(r"function eloChance\(a, b\) \{[^}]*\}", js).group(0)
    pairs = [(1650, 1500), (1400, 1600), (1505, 1505), (1720, 1380)]
    script = fn + "\nconsole.log(JSON.stringify(" + json.dumps(pairs) + ".map(([a,b]) => eloChance(a,b))));"
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for (a, b), p in zip(pairs, out):
        assert p == pytest.approx(ne.expected(a - b), abs=1e-12)


def test_no_data_string_can_end_the_offline_pages_script(tmp_path):
    import nfl_standalone
    data = tmp_path / "data"
    data.mkdir()
    for name in nfl_standalone.PAYLOADS:
        (data / f"{name}.json").write_text(json.dumps({"x": "A <!--<script> B </script> C"}))
    html = nfl_standalone.build(tmp_path / "x.html", data).read_text()
    blob = html.split("window.__YOURNFL_DATA__ = ")[1].split(";</script>")[0]
    assert "<" not in blob and json.loads(blob)["week"]["x"] == "A <!--<script> B </script> C"
