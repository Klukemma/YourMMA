"""The same-day page installs, keeps its own cache, and its runner listens to
the three sports' runners by their real names."""

import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

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


def test_the_cache_was_bumped_for_the_underdog_ticket_and_the_page_reads_it():
    """sameday.json gained `days` and the underdog ticket (v2 of its shape):
    the worker's cache name moved on with it, so an installed app does not
    keep an app.js that cannot read the new file."""
    sw = (APP / "sw.js").read_text()
    version = int(re.search(r'const CACHE = "yourparlays-v(\d+)"', sw).group(1))
    assert version >= 3  # v3: the page reads the draws' means beside the live draw
    js = (APP / "app.js").read_text()
    assert "d.days" in js and '"underdog"' in js and "cannot_both_win" in js
    # the one plain line on the lottery, and the market chance in words
    assert ("One underdog drawn by lot - likelier ones a little more often - plus favourites to 3-5x. "
            "Each underdog's lot number is fixed for the day; until the ticket locks, a price move can "
            "still change which underdog is drawn.") in js
    assert "by the prices" in js and "Only one underdog fit" in js
    assert "how much the draw alone moves it, same games" in js
    assert "The market is hard to beat." in js
    # an older sameday.json (one ticket a day, no `days`) still reads
    assert "d.tickets || []" in js


def test_the_page_never_shows_a_hand_typed_return_or_a_fixed_draw():
    """Two claims the page once typed by hand: a price-implied loss ("8-9%",
    wrong for the likeliest ticket, about 12%) and a draw "fixed for the day"
    (only each underdog's lot number is; a moved price can change the one
    drawn until the ticket locks - sameday.py's LOTTERY fixes only the ORDER
    of two candidates). Neither may come back."""
    js = (APP / "app.js").read_text()
    assert "8-9%" not in js
    assert "The draw is fixed for the day." not in js
    assert "The same events give the same draw on every run" not in js
    assert "The order of any two underdogs depends only on their own lot numbers and chances" in js
    assert "the market's underdog, usually against the app's pick" in js
    # the underdog's splits are not the apps' picks, and single bets are not tickets
    assert "the apps' chances said" in js and 'unit: "bet"' in js
    # an older data file gets its true reason, not "the backtest has not been run"
    assert "This data file was written before the underdog ticket" in js
    assert "The backtest has not been run." not in js


# The page's pure helpers, run under node: from the number helpers to the
# start of BOOT, with a stand-in for the shared chrome (DESK) and the DOM.
def _helpers(calls):
    js = (APP / "app.js").read_text()
    body = js[js.index("const $ = (id)"):js.index("const BOOT = {")]
    stub = ("const DESK = { esc: (s) => String(s ?? '').replace(/[&<>\"']/g, (c) => "
            "({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;' }[c])),"
            " icon: () => '', cardFold: (a, b, c) => a + b + c };\n"
            "const document = { getElementById: () => null };\n")
    tail = ("\nconst setShape = (d) => { SHAPE = shapeOf(d); return SHAPE; };\n"
            "console.log(JSON.stringify(" + calls + "));")
    out = subprocess.run(["node", "-e", stub + body + tail], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _dog_data(**over):
    salts = {"salts": 1000, "roi": -0.0988, "roi_low": -0.1474, "roi_high": -0.0473, "implied_roi": -0.0849,
             "live_rank": {"roi_below": 992, "roi_equal": 0, "of": 1000}}
    live = {"tickets": 1487, "wins": 397, "hit_rate": 0.267, "roi": -0.0158, "roi_low": -0.0989,
            "roi_high": 0.0698, "implied_roi": -0.0849}
    paired = {"days": 1132, "roi_diff": -0.0492, "roi_diff_low": -0.1771, "roi_diff_high": 0.0772,
              "implied_roi_diff": 0.0346, "salts_roi_diff": -0.1668, "salts_roi_diff_low": -0.2642,
              "salts_roi_diff_high": -0.072}
    both = {"days": 1132, "roi": 0.04, "roi_low": -0.0251, "roi_high": 0.1074, "implied_roi": -0.1019,
            "salts_roi": -0.0188, "salts_roi_low": -0.067, "salts_roi_high": 0.0317, "both_won": 122,
            "cannot_both_win": 102}
    dog = {"rule": {"sim_salts": 1000}, "live": {"all": live}, "salts": {"all": salts},
           "report_only": {"paired": {"all": paired}, "both": {"all": both}}}
    dog.update(over)
    lk = {"rule": {"all": {"tickets": 1245, "wins": 420, "roi": 0.0448, "implied_roi": -0.1173}}}
    return {"days": [], "rule": {"dog": {"sim_salts": 1000}}, "measured": dict(lk, likeliest=lk, underdog=dog)}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_every_live_draw_line_sits_beside_the_draws_mean_and_the_prices():
    """The live draw ranks above 992 of the 1,000 simulation draws: a line
    that shows it alone reads a lucky lot as the rule's worth. So each
    reading - the lottery, the pair with the likeliest ticket, both held -
    carries the mean over the draws and the price-implied return, and the
    rank is a count, never a word."""
    d = json.dumps(_dog_data())
    r = _helpers(f"(() => {{ const d = {d}; setShape(d); const g = d.measured.underdog; return {{"
                 "draw: drawLines(g.live.all, g.salts.all, 'All'),"
                 "paired: pairedLines(g.report_only.paired.all, 'All'),"
                 "both: bothLines(g.report_only.both.all, 'All'),"
                 "prices: pricesSentence() }; })()")
    assert r["draw"][0].startswith("All, at the live draw: 1,487 tickets") and "by the prices −8%" in r["draw"][0]
    assert "over the 1,000 draws" in r["draw"][1].lower() and "−10% a ticket (−15% to −5%)" in r["draw"][1]
    assert "the live draw did better than 992 of 1,000" in r["draw"][1]
    paired = " ".join(r["paired"])
    assert "at the live draw −5% a ticket (−18% to +8%)" in paired
    assert "over the 1,000 draws −17% (−26% to −7%)" in paired and "by the prices +3%" in paired
    both = " ".join(r["both"])
    assert "at the live draw +4% a unit" in both and "over the 1,000 draws −2% (−7% to +3%)" in both
    assert "by the prices −10%" in both
    # computed, never typed: the likeliest's price-implied loss is its own
    assert r["prices"] == ("By the prices, over the backtest's tickets, the likeliest ticket loses about 12% "
                           "and the underdog ticket loses about 8% of the stake on average.")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_missing_reading_is_shown_missing_with_its_reason():
    partial = _dog_data(salts=None, report_only={"paired": {"all": {"days": 10, "roi_diff": -0.01}},
                                                  "both": {"all": {"days": 10, "roi": 0.02}}})
    old = {"tickets": [{"day": "2026-11-09", "status": "none", "legs": []}],
           "measured": {"rule": {"all": {"tickets": 5, "wins": 1, "roi": 0.1}}}}
    r = _helpers(f"(() => {{ const d = {json.dumps(partial)}; setShape(d); const g = d.measured.underdog;"
                 "const out = { draw: drawLines(g.live.all, null, 'All'), paired: pairedLines(g.report_only.paired.all, 'All'),"
                 "both: bothLines(g.report_only.both.all, 'All') };"
                 f"const o = setShape({json.dumps(old)}); out.old = o; out.oldDays = daysOf({json.dumps(old)});"
                 "out.oldPrices = pricesSentence(); out.oldToday = pricesSentence(true);"
                 "out.gain = expects(0.03); return out; })()")
    assert r["draw"][1] == "Over the 1,000 draws: not in this data file."
    assert "over the 1,000 draws: not in this data file; by the prices: not in this data file." in r["paired"]
    assert any("over the 1,000 draws: not in this data file" in x for x in r["both"])
    # an older file: one reason everywhere, and its underdog panel says it
    why = "This data file was written before the underdog ticket; the next run of the page's builder adds it."
    assert r["old"]["dogWhy"] == why and r["old"]["old"] is True and r["old"]["lk"]
    assert r["oldDays"][0]["underdog"]["missing_reason"] == why
    assert "this data file was written before the underdog ticket" in r["oldPrices"]
    assert r["oldToday"] == ""  # the Today note leaves it to the cards, which give the reason
    assert r["gain"] == "gains about 3%"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_no_long_token_from_the_data_is_left_unbreakable():
    """A file path in a reason ran past a 320px card and scrolled the page
    sideways; the shared sheet has no rule for it, so the page breaks it."""
    r = _helpers("[brk('see engine/experiments/parlay_legs.csv.gz now'), brk('a & b <c>'),"
                 "brk('x'.repeat(40)), brk('short words only')]")
    assert r[0] == "see engine/<wbr>experiments/<wbr>parlay_<wbr>legs.<wbr>csv.<wbr>gz now"
    assert r[1] == "a &amp; b &lt;c&gt;"
    assert all(len(part) <= 16 for part in r[2].split("<wbr>"))
    assert r[3] == "short words only"
