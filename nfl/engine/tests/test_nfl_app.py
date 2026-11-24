"""The phone app: installable, offline, its own cache and storage on the
origin it shares with the MMA app, and its one formula matching Python's."""

import json
import os
import re
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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


def test_the_shared_design_is_the_mma_apps_too():
    """The NFL runner runs only these tests, so it checks its own copy of the
    shared stylesheet and app chrome against the source as well."""
    shared = APP.parents[1] / "shared"
    html = (APP / "index.html").read_text()
    js = (APP / "app.js").read_text()
    css_block = html.split("/* desk.css:begin")[1].split("/* desk.css:end */")[0]
    js_block = js.split("// desk.js:begin")[1].split("// desk.js:end")[0]
    assert css_block.split("\n", 1)[1].strip() == (shared / "desk.css").read_text().strip()
    assert js_block.split("\n", 1)[1].strip() == (shared / "desk.js").read_text().strip()
    assert "DESK.banner()" in js and "beforeinstallprompt" in js


# ---------------------------------------------------------------- same-game tickets
#
# The app's tickets are drawn in node from a week.json the engine's own
# functions build: an upcoming game priced from the TD fixture, one with no
# TD price bought, a final game whose logged tickets are graded, and one with
# none logged. The page runs with a stub document; what it would put on the
# screen is read back as each view's HTML.

TICKETS_FIX = Path(__file__).resolve().parent / "fixtures" / "odds"
TICKETS_CLOCK = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
TICKETS_KICKOFF = pd.Timestamp("2026-10-25 20:25")
TICKETS_MARKET = {"source": "the-odds-api", "home_ml": -250.0, "away_ml": 205.0,
                  "spread": 6.5, "spread_price_home": -110.0, "spread_price_away": -110.0,
                  "spread_price_source": "the-odds-api", "total": 47.5, "total_price_over": -108.0,
                  "total_price_under": -112.0, "total_price_source": "the-odds-api"}
TICKETS_HOME = [("kc1", "Tom Runner", 0.48), ("kc2", "Sam Tight", 0.36), ("kc3", "Nicholas Name", 0.22)]
TICKETS_AWAY = [("lv1", "Max Wide", 0.33), ("lv2", "DJ Catcher", 0.27)]
TICKETS_FIRST = {"kc1": 2020, "kc2": 2019, "kc3": 2023, "lv1": 2026, "lv2": 2022}   # lv1: a rookie

RENDER_JS = r"""
const fs = require("fs"), vm = require("vm");
const [appPath, dataPath, storePath] = process.argv.slice(2);
const els = {};
const mk = (id) => ({ id, innerHTML: "", hidden: false, attrs: {}, style: {}, tabIndex: 0, textContent: "",
  setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k] ?? null; },
  removeAttribute(k) { delete this.attrs[k]; }, addEventListener() {}, querySelectorAll() { return []; },
  querySelector() { return null; }, focus() {}, click() {} });
const store = new Map(Object.entries(JSON.parse(fs.readFileSync(storePath, "utf8"))));
const ctx = { console, URLSearchParams, setTimeout, clearTimeout, setInterval: () => 0,
  navigator: { userAgent: "node", platform: "", maxTouchPoints: 0 },
  location: { protocol: "file:", search: "", href: "file:///yournfl.html" },
  document: { getElementById: (id) => els[id] || (els[id] = mk(id)), querySelectorAll: () => [],
    querySelector: () => null, addEventListener() {}, documentElement: mk("html"), readyState: "complete",
    body: { children: [], appendChild() {} }, createElement: () => mk("x") } };
if (process.env.NO_STORAGE) Object.defineProperty(ctx, "localStorage", { get() { throw new Error("blocked"); } });
else ctx.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)) };
Object.assign(ctx, { window: ctx, matchMedia: () => ({ matches: false }), addEventListener() {}, scrollTo() {},
  __YOURNFL_DATA__: { week: JSON.parse(fs.readFileSync(dataPath, "utf8")) } });
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(appPath, "utf8"), ctx, { filename: "app.js" });
(async () => {
  await vm.runInContext("BOOT.week()", ctx);
  await vm.runInContext("BOOT.parlays()", ctx);
  const inputs = JSON.parse(process.env.ODDS_INPUTS || "[]");
  const parsed = vm.runInContext(`(${JSON.stringify(inputs)}).map(parseOdds)`, ctx);
  const stakes = vm.runInContext(`(${process.env.STAKE_INPUTS || "[]"}).map(parseStake)`, ctx);
  process.stdout.write(JSON.stringify({ week: els["view-week"].innerHTML, parlays: els["view-parlays"].innerHTML,
    parsed, stakes }));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _ticket_list(rows):
    return [{"gsis_id": g, "name": n, "position": "WR", "prob": p, "tds_season": 0, "games_season": 0,
             "status": "active", "rank": i + 1, "group": "3+"} for i, (g, n, p) in enumerate(rows)]


def _ticket_game(game_id, home="KC", away="LV", **kw):
    g = {"game_id": game_id, "season": 2026, "week": 7, "game_type": "REG",
         "kickoff": "2026-10-25T20:25:00Z", "status": "upcoming", "home": home, "away": away,
         "home_name": "Kansas City Chiefs", "away_name": "Las Vegas Raiders", "pick": home,
         "pick_prob": 0.7, "home_prob": 0.7, "market": dict(TICKETS_MARKET),
         "home_td_all": _ticket_list(TICKETS_HOME), "away_td_all": _ticket_list(TICKETS_AWAY)}
    g.update(kw)
    return g


def _sgp_week(history="committed"):
    """(week.json, the four games) with same-game tickets, as nfl_predict.run writes them."""
    import nfl_export as nx
    import nfl_odds as no
    import nfl_predict as npd
    import nfl_sgp as sgp
    priced = _ticket_game("2026_07_LV_KC")
    cache = no.empty()
    key, entry = no.parse_td_event(json.loads((TICKETS_FIX / "td_event.json").read_text()))
    no.merge_td(cache, key, entry, now=TICKETS_CLOCK)
    prices = no.td_prices_for(cache, "LV", "KC", TICKETS_KICKOFF, npd._td_names(priced, {}), now=TICKETS_CLOCK)
    priced["tickets"] = npd.game_tickets(priced, priced["market"], prices, TICKETS_FIRST)
    # No TD price bought for this one, and no rookie on either side.
    unpriced = _ticket_game("2026_07_LV_KC2", away_td_all=_ticket_list(TICKETS_AWAY[1:]))
    unpriced["tickets"] = npd.game_tickets(unpriced, unpriced["market"], {}, TICKETS_FIRST)
    final = _ticket_game("2026_06_LV_KC", status="final", score={"home": 27, "away": 20}, week=6,
                         kickoff="2026-10-18T20:25:00Z")
    result = sgp.outcome("KC", "LV", 27, 20, players={"kc1": 1, "kc2": 0, "kc3": 0, "lv1": 0, "lv2": 1})
    final["tickets"] = [sgp.grade_ticket(t, result) for t in npd.game_tickets(final, final["market"], prices,
                                                                             TICKETS_FIRST)]
    final["tickets_graded"] = True
    none = _ticket_game("2026_06_LV_KC2", status="final", score={"home": 10, "away": 13}, week=6,
                        kickoff="2026-10-18T17:00:00Z", tickets=None, tickets_reason=npd.NO_TICKETS_LOGGED)
    log = {"picks": {g["game_id"]: {"kickoff": g["kickoff"], "tickets": g["tickets"],
                                    "tickets_graded": g.get("tickets_graded", False)}
                     for g in (priced, unpriced, final)}}
    hist = nx.load_sgp_history() if history == "committed" else history
    week = {"season": 2026, "week": 7, "label": "Week 7", "games": [final, none, priced, unpriced],
            "caveats": {}, "sgp": json.loads(json.dumps(nx.sgp_block(log, hist)))}
    return json.loads(json.dumps(week)), {"priced": priced, "unpriced": unpriced, "final": final, "none": none}


def _render(tmp_path, week, store=None, no_storage=False, odds_inputs=(), stake_inputs=()):
    (tmp_path / "week.json").write_text(json.dumps(week))
    (tmp_path / "store.json").write_text(json.dumps(store or {}))
    (tmp_path / "render.js").write_text(RENDER_JS)
    env = dict(os.environ, ODDS_INPUTS=json.dumps(list(odds_inputs)), STAKE_INPUTS=json.dumps(list(stake_inputs)))
    if no_storage:
        env["NO_STORAGE"] = "1"
    out = subprocess.run(["node", str(tmp_path / "render.js"), str(APP / "app.js"), str(tmp_path / "week.json"),
                          str(tmp_path / "store.json")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def _signed(x, d=1):
    r = round(x, d)
    return f"{0:.{d}f}" if r == 0 else f"{'+' if r > 0 else '−'}{abs(r):.{d}f}"


def _ticket_html(html, game_id, kind):
    """One ticket card's HTML: from its card to the next."""
    import nfl_sgp as sgp
    marker = f'data-sgp-id="{game_id}|{kind}|{sgp.VERSION}|'
    assert marker in html, (game_id, kind)
    at = html.index(marker)
    starts = [m.start() for m in re.finditer(r'<div class="tkt(?: off)?" data-kind="', html)]
    start = max(i for i in starts if i < at)
    nxt = min((i for i in starts if i > at), default=None)
    return html[start:nxt]


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_the_parlays_tab_is_wired_like_every_other_tab():
    html = (APP / "index.html").read_text()
    js = (APP / "app.js").read_text()
    names = json.loads(re.search(r"DESK\.tabs\((\[[^\]]*\])", js).group(1))
    assert "parlays" in names
    for name in names:
        assert f'id="view-{name}"' in html and f'id="tab-{name}"' in html and f"aria-controls=\"view-{name}\"" in html
        assert re.search(rf"\b{name}\(\) \{{", js), name          # BOOT[name]
    assert 'DESK.sub(esc(sgpTitle(g)), sgpSection(g, d))' in js    # the game's detail


def test_the_tickets_code_types_no_number_the_payload_could_give():
    """Every figure beside a ticket or a meter is read from week.json: no
    percentage, dollar amount or season is typed into the tickets' code."""
    js = (APP / "app.js").read_text()
    block = js.split("// sgp:begin")[1].split("// sgp:end")[0]
    code = re.sub(r"^\s*//.*$", "", block, flags=re.M)
    assert not re.findall(r"\d%|\$\d|\b(?:19|20)\d\d\b", code)
    assert "s.stake" in code and "localStorage" not in code      # the stake from the payload; storage via DESK.store
    assert "fetch(" not in code and "XMLHttpRequest" not in code and "sendBeacon" not in code


@needs_node
def test_each_ticket_shows_its_legs_chance_and_est_payout_from_the_payload(tmp_path):
    import nfl_sgp as sgp
    week, games = _sgp_week()
    out = _render(tmp_path, week)
    s, page = week["sgp"], out["parlays"]
    stake = s["stake"]
    for t in games["priced"]["tickets"]:
        card = _ticket_html(page, "2026_07_LV_KC", t["kind"])
        assert t["built"], t["kind"]
        assert f'<b class="num">{_pct(t["chance"])}</b> <span class="mini">the product of the legs\' chances</span>' in card
        assert f'returns <b class="num">${stake * t["est_payout"]:.2f}</b> <span class="est">est.</span>' in card
        assert f'Fair odds at the product chance: <b class="num">{1 / t["chance"]:.2f}</b>' in card
        for leg in t["legs"]:
            assert f'<span class="num lead">{_pct(leg["chance"], 0)}</span>' in card
            assert f'<span class="mini num"> {leg["decimal"]:.2f}</span> <span class="est">est.</span>' in card
            if leg.get("best_american") is not None:       # a book's own price is not Stake's either
                best = int(leg["best_american"])
                assert f'best {"+" if best > 0 else "−"}{abs(best)} <span class="est">est.</span>' in card
        # Beside the product: the kind's measured hit rate, from the history.
        h = s["history"]
        whole = max(h["periods"].items(), key=lambda p: p[1][1] - p[1][0])[0]
        m = h["kinds"][t["kind"]]["meters"][whole]
        assert f'tickets hit <b>{_pct(m["hit_rate"])}</b>' in card and f'said {_pct(m["mean_chance"])}' in card
    # Every price on the page is marked est.: none is Stake's.
    prices = page.split('<span class="mini num"> ')[1:]
    assert prices and all(chunk.split("</span>", 1)[1].startswith(' <span class="est">est.</span>') for chunk in prices)
    # The rookie ticket is the rookie's TD and his own team's win.
    rookie = _ticket_html(page, "2026_07_LV_KC", "rookie")
    assert "Max Wide" in rookie and '<span class="tag accent">Rookie</span>' in rookie and "LV to win" in rookie
    assert sgp.NO_EDGE.replace("'", "&#39;") in page


@needs_node
def test_a_missing_price_or_an_unbuilt_ticket_says_why(tmp_path):
    import nfl_odds as no
    week, games = _sgp_week()
    page = _render(tmp_path, week)["parlays"]
    for t in games["unpriced"]["tickets"]:
        if not t["built"]:
            off = page.split(f'data-kind="{t["kind"]}"')
            assert any(f"<b>Not built:</b> {t['reason']}" in chunk.replace("&#39;", "'") for chunk in off), t["kind"]
            continue
        card = _ticket_html(page, "2026_07_LV_KC2", t["kind"])
        assert t["est_payout"] is None and f"no est. payout: {t['est_payout_reason']}" in card
        assert f"no price: {no.TD_PRICE_MISSING['none']}" in card
    assert not any(t["built"] for t in games["unpriced"]["tickets"] if t["kind"] == "rookie")
    assert f'<div class="mini">{games["none"]["tickets_reason"]}</div>' in page


@needs_node
def test_a_graded_ticket_shows_each_legs_result_and_the_live_meters_mark_their_roi_est(tmp_path):
    import nfl_sgp as sgp
    week, games = _sgp_week()
    out = _render(tmp_path, week)
    page = out["parlays"]
    small = next(t for t in games["final"]["tickets"] if t["kind"] == "small")
    assert small["result"] == sgp.WON
    card = _ticket_html(page, "2026_06_LV_KC", "small")
    assert '<span class="tag good">Won</span>' in card
    assert f"returned ${week['sgp']['stake'] * small['payout']:.2f}" in card
    live = week["sgp"]["live"]["kinds"]
    for kind in sgp.KINDS:
        m = live[kind]
        meters = page.split(f'<div class="panel sgpm" data-kind="{kind}">')[1].split('<div class="panel sgpm"')[0]
        assert f'<b class="num">{m["won"]}</b> won of <b class="num">{m["tickets"]}</b> built' in meters
        if m["roi"] is not None:
            assert f'ROI <b class="num">{_signed(m["roi"] * 100)}%</b> <span class="est">est.</span>' in meters
        else:
            assert f"no return: {m['roi_reason']}" in meters
    assert any(live[k]["roi"] is not None for k in sgp.KINDS)
    # The history's return is missing, and its reason is the payload's.
    h = week["sgp"]["history"]
    assert h["available"] and f"no return: {h['roi_reason']}" in page
    assert week["sgp"]["caveats"]["bias"] in page
    # The week tab's detail carries the same tickets.
    assert "Parlays · 4 of 4 built" in out["week"] and 'data-sgp-id="2026_07_LV_KC|small|' in out["week"]


@needs_node
def test_stakes_quote_gives_the_exact_payout_and_stays_on_the_phone(tmp_path):
    import nfl_sgp as sgp
    week, games = _sgp_week()
    t = next(x for x in games["priced"]["tickets"] if x["kind"] == "small")
    key = "|".join(["2026_07_LV_KC", "small", sgp.VERSION, "+".join(leg["key"] for leg in t["legs"])])
    page = _render(tmp_path, week, store={f"yournfl.sgp.{key}": json.dumps({"odds": "+450", "stake": "20"})})["parlays"]
    card = _ticket_html(page, "2026_07_LV_KC", "small")
    dec = sgp.decimal_from_american(450)
    assert f'Pays <b class="num">${20 * dec:.2f}</b> · profit <b class="num">+${20 * dec - 20:.2f}</b>' in card
    assert f"implies {_pct(1 / dec)}" in card and 'value="+450"' in card and 'value="20"' in card
    # Untyped, a box offers the payload's stake per ticket; with no storage at all the page still draws.
    other = _ticket_html(page, "2026_07_LV_KC", "medium")
    assert f'value="{week["sgp"]["stake"]:g}"' in other
    blocked = _render(tmp_path, week, no_storage=True)["parlays"]
    assert "Parlays" in blocked and 'value="+450"' not in blocked and "Fair odds at the product chance" in blocked


@needs_node
def test_the_apps_reading_of_typed_odds_matches_python(tmp_path):
    import nfl_sgp as sgp
    week, _ = _sgp_week(history=None)
    signed = ["+450", "-120", "\u2212120", "+100", "-100", "+99", "-99.5", "+1250", "+1,250"]
    plain = ["4.50", "4,5", "1.01", "1", "0.9", "12"]
    bad = ["abc", "4.5x", "++450"]
    parsed = _render(tmp_path, week, odds_inputs=signed + plain + bad + [""])["parsed"]
    for text, p in zip(signed, parsed):
        want = sgp.decimal_from_american(float(text.replace("\u2212", "-").replace(",", "")))
        assert (p.get("decimal") is None) == (want is None) and (want is None or p["decimal"] == pytest.approx(want)), text
    for text, p in zip(plain, parsed[len(signed):]):
        v = float(text.replace(",", "."))
        assert (p.get("decimal") == pytest.approx(v)) if v > 1 else ("error" in p), text
    assert all("error" in p for p in parsed[len(signed) + len(plain):-1]) and parsed[-1] is None


@needs_node
def test_a_typed_amount_reads_a_thousands_comma_as_dollars(tmp_path):
    """The amount box is in dollars: "1,000" is a thousand, not one. Only a
    comma before groups of exactly three digits is a thousands separator;
    any other single comma is a decimal point, and anything else is refused."""
    week, _ = _sgp_week(history=None)
    cases = {"20": 20, "$20": 20, "20.50": 20.5, "10,50": 10.5, "1,5": 1.5, "1,000": 1000, "2,500": 2500,
             "$1,000.50": 1000.5, "12,345,678": 12345678, "0,500": 0.5, "1,00": 1.0,
             "1,0000": 1.0, "1,000,00": None, "1,2,3": None, "0": None, "abc": None, "": None}
    stakes = _render(tmp_path, week, stake_inputs=list(cases))["stakes"]
    for (text, want), got in zip(cases.items(), stakes):
        assert got == (pytest.approx(want) if want is not None else None), text


@needs_node
def test_the_week_tabs_tickets_carry_the_backtests_known_bias(tmp_path):
    """Each card quotes the backtest's hit rate for its kind, so the bias
    the rule says goes where the meter is is said in the game's detail too."""
    week, _ = _sgp_week()
    out = _render(tmp_path, week)
    bias = week["sgp"]["caveats"]["bias"]
    assert "tickets hit <b>" in out["week"]
    assert bias in out["week"] and bias in out["parlays"]


@needs_node
def test_a_chance_bins_edges_are_printed_exactly(tmp_path):
    """A bin edge of 2.5% reads 2.5%, not a rounded 3% that a ticket shown
    at 2.6% would contradict."""
    week, _ = _sgp_week()
    page = _render(tmp_path, week)["parlays"]

    def edge(x):
        d = next(d for d in range(3) if abs(round(x * 100, d) - x * 100) < 1e-9)
        return f"{x * 100:.{d}f}%"
    kinds = week["sgp"]["history"]["kinds"]
    bins = [b for k in kinds.values() for b in k.get("chance_bins") or []]
    assert any(abs(b["hi"] * 1000 - round(b["hi"] * 1000)) < 1e-9 and round(b["hi"] * 1000) % 10 for b in bins)
    for b in bins:
        assert f'<td class="n">{edge(b["lo"])}-{edge(b["hi"])}</td>' in page, b


@needs_node
def test_with_no_backtest_or_no_tickets_the_page_says_why(tmp_path):
    week, _ = _sgp_week(history=None)
    page = _render(tmp_path, week)["parlays"]
    assert week["sgp"]["history"]["reason"] in page
    old = dict(week, games=[{k: v for k, v in g.items() if k not in ("tickets", "tickets_reason")}
                            for g in week["games"]])
    old.pop("sgp")
    out = _render(tmp_path, old)
    assert "before the same-game tickets existed" in out["parlays"] and "Parlays ·" not in out["week"]
