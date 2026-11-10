"""The same-day page installs, keeps its own cache, and its runner listens to
the three sports' runners by their real names."""

import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "parlays" / "app"
OTHERS = {"mma": ROOT / "app", "nfl": ROOT / "nfl" / "app", "soccer": ROOT / "soccer" / "app"}


def _png_size(path):
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", head[16:24])


def test_every_manifest_icon_exists_at_its_stated_size():
    man = json.loads((APP / "manifest.json").read_text())
    assert man["start_url"] == "." and man["display"] == "standalone"
    for ic in man["icons"]:
        w, h = _png_size(APP / ic["src"])
        assert f"{w}x{h}" == ic["sizes"], ic
    assert _png_size(APP / "icons" / "apple-touch-icon.png") == (180, 180)


def _cache_prefix(app):
    sw = (app / "sw.js").read_text()
    name = re.search(r'const CACHE = "([^"]+)"', sw).group(1)
    return name.rsplit("-", 1)[0] + "-", sw


def test_its_cache_is_its_own_and_no_app_deletes_another_apps():
    """A worker deletes caches by prefix: two prefixes where one starts the
    other would let one app's worker throw away the other's offline copy."""
    mine, sw = _cache_prefix(APP)
    others = [_cache_prefix(a)[0] for a in OTHERS.values()]
    for o in others:
        assert not mine.startswith(o) and not o.startswith(mine), (mine, o)
    assert f'n.startsWith("{mine}")' in sw
    shell = re.search(r"const SHELL = \[(.*?)\];", sw, re.S).group(1)
    for f in re.findall(r'"([^"]+)"', shell):
        if f not in ("./",):
            assert (APP / f).exists(), f
    assert 'url.pathname.includes("/data/")) return networkFirst' in sw


def test_its_tab_memory_is_its_own():
    keys = [re.search(r'DESK\.tabs\(\[[^\]]*\], BOOT, "([^"]+)"', (a / "app.js").read_text()).group(1)
            for a in [APP, *OTHERS.values()]]
    assert len(set(keys)) == len(keys), keys


def test_the_runner_listens_to_the_sports_runners_by_their_real_names():
    wf = (ROOT / ".github" / "workflows" / "parlays.yml").read_text()
    listed = re.search(r"workflows: \[(.*?)\]", wf).group(1)
    names = re.findall(r'"([^"]+)"', listed)
    real = [re.search(r"^name: (.+)$", (ROOT / ".github" / "workflows" / f).read_text(), re.M).group(1).strip()
            for f in ("update-dataset.yml", "nfl.yml", "soccer.yml")]
    assert sorted(names) == sorted(real)
    assert "python parlays/engine/build_sameday.py" in wf
    assert "parlays/app/data/" in wf and "parlays/data/history.json" in wf


def test_the_page_data_on_file_is_well_formed():
    d = json.loads((APP / "data" / "sameday.json").read_text())
    assert d["day_tz"] == "America/New_York" and d["rule"]["band"] == [3.0, 5.0]
    assert any("hard to beat" in c for c in d["caveats"])
    json.dumps(d, allow_nan=False)
