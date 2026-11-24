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
# screen is read back as each view's HTML. The tickets are the live rule's
# (nfl_predict.LIVE_VERSION, sgp-v2): KC is a confident pick, three of its
# and LV's players clear the TD floor, and KC's rookie Sam Tight is the
# pick's team's rookie, so all four are built.

TICKETS_FIX = Path(__file__).resolve().parent / "fixtures" / "odds"
TICKETS_CLOCK = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
TICKETS_KICKOFF = pd.Timestamp("2026-10-25 20:25")
TICKETS_MARKET = {"source": "the-odds-api", "home_ml": -250.0, "away_ml": 205.0,
                  "spread": 6.5, "spread_price_home": -110.0, "spread_price_away": -110.0,
                  "spread_price_source": "the-odds-api", "total": 47.5, "total_price_over": -108.0,
                  "total_price_under": -112.0, "total_price_source": "the-odds-api"}
TICKETS_HOME = [("kc1", "Tom Runner", 0.58), ("kc2", "Sam Tight", 0.49), ("kc3", "Nicholas Name", 0.22)]
TICKETS_AWAY = [("lv1", "Max Wide", 0.46), ("lv2", "DJ Catcher", 0.27)]
TICKETS_FIRST = {"kc1": 2020, "kc2": 2026, "kc3": 2023, "lv1": 2026, "lv2": 2022}   # kc2, lv1: rookies

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
  await vm.runInContext("BOOT.td()", ctx);
  const inputs = JSON.parse(process.env.ODDS_INPUTS || "[]");
  const parsed = vm.runInContext(`(${JSON.stringify(inputs)}).map(parseOdds)`, ctx);
  const stakes = vm.runInContext(`(${process.env.STAKE_INPUTS || "[]"}).map(parseStake)`, ctx);
  process.stdout.write(JSON.stringify({ week: els["view-week"].innerHTML, parlays: els["view-parlays"].innerHTML,
    td: els["view-td"].innerHTML, parsed, stakes }));
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


def _td_fixture_prices(game):
    """The anytime-TD prices nfl_odds matches to a KC-LV game's listed
    players from the fixture event: four priced, one with none under his name."""
    import nfl_odds as no
    import nfl_predict as npd
    cache = no.empty()
    key, entry = no.parse_td_event(json.loads((TICKETS_FIX / "td_event.json").read_text()))
    no.merge_td(cache, key, entry, now=TICKETS_CLOCK)
    return no.td_prices_for(cache, "LV", "KC", TICKETS_KICKOFF, npd._td_names(game, {}), now=TICKETS_CLOCK)


def _sgp_week(history="committed"):
    """(week.json, the four games) with same-game tickets, as nfl_predict.run writes them."""
    import nfl_export as nx
    import nfl_predict as npd
    import nfl_sgp as sgp
    priced = _ticket_game("2026_07_LV_KC")
    prices = _td_fixture_prices(priced)
    priced["tickets"] = npd.game_tickets(priced, priced["market"], prices, TICKETS_FIRST)
    # No TD price bought for this one, and no rookie on either side.
    unpriced = _ticket_game("2026_07_LV_KC2", home_td_all=_ticket_list(TICKETS_HOME[::2]),
                            away_td_all=_ticket_list(TICKETS_AWAY[1:]))
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
    hist = nx.load_sgp_history(nx.SGP2_HISTORY) if history == "committed" else history
    v1 = nx.load_sgp_history() if history == "committed" else None
    week = {"season": 2026, "week": 7, "label": "Week 7", "games": [final, none, priced, unpriced],
            "caveats": {}, "sgp": json.loads(json.dumps(nx.sgp_block(log, hist, v1)))}
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
    import nfl_sgp2 as sgp2
    marker = f'data-sgp-id="{game_id}|{kind}|{sgp2.VERSION}|'
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
    # The rookie ticket is the pick's team's rookie's TD and the pick's win.
    rookie = _ticket_html(page, "2026_07_LV_KC", "rookie")
    assert "Sam Tight" in rookie and '<span class="tag accent">Rookie</span>' in rookie and "KC to win" in rookie
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
    import nfl_sgp2 as sgp2
    week, games = _sgp_week()
    t = next(x for x in games["priced"]["tickets"] if x["kind"] == "small")
    key = "|".join(["2026_07_LV_KC", "small", sgp2.VERSION, "+".join(leg["key"] for leg in t["legs"])])
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
def test_the_est_payouts_correction_is_shown_wherever_v1_calls_it_an_upper_bound(tmp_path):
    """v1's words call the est. payout and the live return an upper bound on
    what Stake pays; legs that pull against each other can be priced above
    the product, so the correction found after the rule was fixed is shown
    beside those words, which stay as they were registered."""
    import nfl_sgp as sgp
    week, _ = _sgp_week()
    out = _render(tmp_path, week)
    cav = week["sgp"]["caveats"]
    assert cav["est"] == sgp.EST_PAYOUT and cav["live"] == sgp.LIVE_ROI and cav["est_erratum"] == sgp.EST_ERRATUM
    esc = lambda x: x.replace("'", "&#39;")   # noqa: E731
    for view in ("week", "parlays"):
        page = out[view]
        assert esc(sgp.EST_PAYOUT) in page, view
        assert page.count(esc(sgp.EST_ERRATUM)) >= page.count(esc(sgp.EST_PAYOUT)), view
    meters = out["parlays"].split('id="sgp-meters"')[1]
    live = meters.index(esc(sgp.LIVE_ROI))
    assert esc(sgp.EST_ERRATUM) in meters[live:]
    assert "upper bound" not in (APP / "app.js").read_text()


@needs_node
def test_a_chance_bins_edges_are_printed_exactly(tmp_path):
    """A bin edge of 2.5% reads 2.5%, not a rounded 3% that a ticket shown
    at 2.6% would contradict. v2's history on file has no such edge among
    its bins (its tickets are the likelier ones), so one is added to a copy
    of it: what is tested is the printing, not the data."""
    import nfl_export as nx
    hist = nx.load_sgp_history(nx.SGP2_HISTORY)
    large = hist["kinds"]["large"]
    large["chance_bins"] = [dict(large["chance_bins"][0], lo=0.0, hi=0.025)] + large["chance_bins"]
    week, _ = _sgp_week(history=hist)
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


# ---------------------------------------------------------------- v2 in the app
#
# The same week grown by the games v2 changes: one not confident (its four
# v2 tickets unbuilt for the one reason), one not confident with v1's tickets
# logged before v1 was retired, and a final whose only tickets are v1's,
# graded - each as nfl_predict.show_tickets shows it, the sgp block built
# from a log of all of them.

def _esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;") \
        .replace("'", "&#39;")


def _cap(x):
    return x[:1].upper() + x[1:]


def _american(x):
    """American odds as app.js's odds() prints them (Math.round, a typographic minus)."""
    import math
    r = math.floor(x + 0.5)
    return f"+{r}" if r > 0 else f"−{abs(r)}" if r < 0 else "0"


def _v2_week():
    import nfl_export as nx
    import nfl_predict as npd
    import nfl_sgp as sgp
    _, base = _sgp_week()
    games = dict(base)
    shaky = _ticket_game("2026_07_NYJ_BUF", home="BUF", away="NYJ", home_prob=0.55, pick="BUF", pick_prob=0.55)
    npd.show_tickets(shaky, npd.game_tickets(shaky, shaky["market"], {}, TICKETS_FIRST))
    both = _ticket_game("2026_07_MIA_NE", home="NE", away="MIA", home_prob=0.4, pick="MIA", pick_prob=0.6)
    npd.show_tickets(both, npd.game_tickets(both, both["market"], {}, TICKETS_FIRST, version=sgp.VERSION)
                     + npd.game_tickets(both, both["market"], {}, TICKETS_FIRST))
    old = _ticket_game("2026_06_SEA_SF", home="SF", away="SEA", status="final", week=6,
                       kickoff="2026-10-18T23:05:00Z", score={"home": 24, "away": 17})
    result = sgp.outcome("SF", "SEA", 24, 17, players={"kc1": 1, "kc2": 0, "kc3": 0, "lv1": 0, "lv2": 0})
    npd.show_tickets(old, [sgp.grade_ticket(t, result) for t in npd.game_tickets(
        old, old["market"], {}, TICKETS_FIRST, version=sgp.VERSION)])
    old["tickets_graded"] = True
    games.update(shaky=shaky, both=both, old=old)
    for g in (games["priced"], games["unpriced"], games["final"]):
        npd.show_tickets(g, g["tickets"])
    log = {"picks": {g["game_id"]: {"kickoff": g["kickoff"], "tickets": (g.get("tickets") or []) + (g.get("tickets_v1") or []),
                                    "tickets_graded": g.get("tickets_graded", False)}
                     for g in games.values() if g.get("tickets") or g.get("tickets_v1")}}
    order = ["final", "old", "none", "priced", "both", "shaky", "unpriced"]
    week = {"season": 2026, "week": 7, "label": "Week 7", "games": [games[k] for k in order], "caveats": {},
            "sgp": nx.sgp_block(log, nx.load_sgp_history(nx.SGP2_HISTORY), nx.load_sgp_history())}
    return json.loads(json.dumps(week)), json.loads(json.dumps(games))


def _card_of(html, game_id, kind, version):
    """One ticket card of a given rule's version: from its card to the next."""
    marker = f'data-sgp-id="{game_id}|{kind}|{version}|'
    assert marker in html, (game_id, kind, version)
    at = html.index(marker)
    starts = [m.start() for m in re.finditer(r'<div class="tkt(?: off)?" data-kind="', html)]
    start = max(i for i in starts if i < at)
    nxt = min((i for i in starts if i > at), default=None)
    return html[start:nxt]


@needs_node
def test_a_game_that_is_not_confident_says_why_once_and_the_confident_ones_come_first(tmp_path):
    import nfl_sgp2 as sgp2
    week, games = _v2_week()
    out = _render(tmp_path, week)
    page, wk = out["parlays"], out["week"]
    reason = sgp2.not_confident(sgp2.CONFIDENT)
    assert games["shaky"]["tickets_reason"] == reason == games["both"]["tickets_reason"]
    assert not any(t["built"] for t in games["shaky"]["tickets"])
    for view in (page, wk):
        # Once for each of the two games, never as four unbuilt tickets.
        assert view.count(f'<div class="mini">{_esc(reason)}</div>') == 2
        assert f"<b>Not built:</b> {_esc(reason)}" not in view
        assert 'data-sgp-id="2026_07_NYJ_BUF|' not in view and 'data-sgp-id="2026_07_MIA_NE|small|sgp-v2|' not in view
    assert "Parlays · none built" in wk and "Parlays · none built · v1 retired" in wk
    # The pick beside the reason, from the tickets the payload logged.
    t = games["shaky"]["tickets"][0]
    assert f'pick</span> <b>{t["pick"]}</b> <span class="num">{_pct(t["pick_prob"])}</span>' in page
    assert f'Pick <b>{t["pick"]}</b> <span class="num">{_pct(t["pick_prob"])}</span> · confident from {_pct(t["confident"], 0)}' in wk
    # Confident games first, then the other cards, then the one-line list.
    confident = page.index('data-sgp-id="2026_07_LV_KC|small|sgp-v2|')
    v1_only = page.index('data-sgp-id="2026_07_MIA_NE|small|sgp-v1|')
    listed = page.index('<div class="card sgpothers">')
    assert confident < v1_only < listed < page.index(f'<div class="mini">{_esc(reason)}</div>', listed)
    n_confident = sum(any(t.get("confident_game") for t in g.get("tickets") or []) for g in week["games"])
    assert page.count('<span class="tag good">Confident</span>') == n_confident
    assert f"{n_confident} of {len(week['games'])} games here are confident" in page


@needs_node
def test_v1s_kept_tickets_are_folded_under_v2s_and_read_beside_v1s_meters(tmp_path):
    import nfl_predict as npd
    import nfl_sgp as sgp
    week, games = _v2_week()
    out = _render(tmp_path, week)
    page = out["parlays"]
    v1 = week["sgp"]["v1"]
    assert v1["retired"] and _esc(v1["retired_note"]) in page and _esc(v1["retired_note"]) in out["week"]
    for key in ("both", "old"):
        kept = games[key]["tickets_v1"]
        assert f"v1 (retired) · {sum(t['built'] for t in kept)} of {len(kept)} built" in page
    # A v1 ticket quotes v1's history, not v2's, and carries no "built by another rule" line.
    h = v1["history"]
    whole = max(h["periods"].items(), key=lambda p: p[1][1] - p[1][0])[0]
    m = h["kinds"]["small"]["meters"][whole]
    card = _card_of(page, "2026_07_MIA_NE", "small", sgp.VERSION)
    assert f'tickets hit <b>{_pct(m["hit_rate"])}</b> of {m["won"] + m["lost"]:,}' in card
    assert "Built by rule" not in card
    # The final with only v1's tickets: the payload's reason, and v1's graded ticket under it.
    assert games["old"]["tickets"] is None and games["old"]["tickets_reason"] == npd.NO_LIVE_TICKETS_LOGGED
    assert f'<div class="mini">{_esc(npd.NO_LIVE_TICKETS_LOGGED)}</div>' in page
    small = next(t for t in games["old"]["tickets_v1"] if t["kind"] == "small")
    chip = {sgp.WON: "Won", sgp.LOST: "Lost"}[small["result"]]
    assert f">{chip}</span>" in _card_of(page, "2026_06_SEA_SF", "small", sgp.VERSION)
    # v1's own meters, folded under v2's: its live record of these logged tickets.
    meters = page.split('id="sgp-meters"')[1]
    assert "v1 (retired) · how its tickets have done" in meters
    for kind in sgp.KINDS:
        lm = v1["live"]["kinds"][kind]
        panel = meters.split(f'<div class="panel sgpm" data-kind="{kind}">')[2].split('<div class="panel sgpm"')[0]
        assert f'<b class="num">{lm["won"]}</b> won of <b class="num">{lm["tickets"]}</b> built' in panel


@needs_node
def test_v2s_meters_carry_the_coverage_and_the_chosen_pair_is_the_payloads(tmp_path):
    week, _ = _v2_week()
    page = _render(tmp_path, week)["parlays"]
    s = week["sgp"]
    h, c = s["history"], s["caveats"]
    whole = max(h["periods"].items(), key=lambda p: p[1][1] - p[1][0])[0]
    for kind in s["kinds"]:
        panel = page.split(f'<div class="panel sgpm" data-kind="{kind}">')[1].split('<div class="panel sgpm"')[0]
        cov = h["kinds"][kind]["coverage"][whole]
        assert f"built in {_pct(cov)} of {h['games'][whole]:,} games" in panel
        m = h["kinds"][kind]["meters"][whole]
        assert f'hit <b>{_pct(m["hit_rate"])}</b> <span class="mini">· product said {_pct(m["mean_chance"])}</span>' in panel
        assert f"no return: {_esc(m['roi_reason'])}" in panel
    assert f", {h['confident_games'][whole]:,} of them confident" in page
    assert f", {s['live']['confident_games']:,} of them confident" in page
    # The pair, the rule's sentence on how it was chosen, and the DEV figures that chose it.
    ch = h["chosen"]
    dev = ch["dev"]["small"]
    assert (s["confident"], s["td_floor"]) == (ch["confident"], ch["td_floor"])
    assert f'the pick\'s chance is <b class="num">{_pct(s["confident"], 0)}</b> or more' in page
    assert f'only at <b class="num">{_pct(s["td_floor"], 0)}</b> or more' in page
    assert _esc(_cap(c["rules"]["chosen"])) in page and _esc(c["edge_v2"]) in page
    assert (f"built in {_pct(dev['coverage'])} of {ch['dev']['games']:,} games and hit {_pct(dev['hit_rate'])} "
            f"({dev['won']} of {dev['won'] + dev['lost']})") in page
    assert f"{ch['cleared']} of {ch['pairs']} pairs built it in {_pct(ch['coverage_min'], 0)} or more" in page
    assert "chose the pair" in page and "untouched" in page
    # Every pair of the grid, per period - the chosen one marked - report-only.
    grid = page.split("Every pair of the grid, and v2 beside v1")[1]
    periods = [n for n in h["periods"] if n != whole]
    table = grid.split('class="tight sgpgrid"')[1].split("</table>")[0]
    assert table.count("<tr") == len(h["grid"]["rows"]) + 1
    for r in h["grid"]["rows"]:
        label = f"{_pct(r['confident'], 0)} · {_pct(r['td_floor'], 0)}"
        row = table.split(f'<td class="nowrap">{label}')[1].split("</tr>")[0]
        for n in periods:
            sm = r["periods"][n]["small"]
            assert f'{_pct(sm["hit_rate"])}<div class="mini">in {_pct(sm["coverage"], 0)}</div>' in row, (label, n)
    chosen = f'<tr class="chosen"><td class="nowrap">{_pct(s["confident"], 0)} · {_pct(s["td_floor"], 0)}'
    assert table.count('<tr class="chosen">') == 1 and chosen in table
    assert _esc(_cap(h["alt_small"]["rule"])) in grid and _esc(_cap(h["v1_same_games"]["note"])) in grid
    # v2's rule sentences in the rule's fold.
    for k, sentence in c["rules"].items():
        assert _esc(sentence) in page, k


def _odds_week():
    """The v2 week with each game's TD lists as nfl_predict.run shows them:
    the top 3 beside the whole list, each player's anytime-TD price - the
    priced game's from the fixture, none bought for the other - and the
    final's logged before the lists kept one."""
    import nfl_export as nx
    import nfl_predict as npd
    week, _ = _v2_week()
    by_id = {g["game_id"]: g for g in week["games"]}
    for gid in ("2026_07_LV_KC", "2026_07_LV_KC2", "2026_06_LV_KC"):
        g = by_id[gid]
        for side in ("home", "away"):
            g[f"{side}_td"] = [dict(p) for p in g[f"{side}_td_all"][:npd.TD_SHOWN]]
    npd.attach_td_prices(by_id["2026_07_LV_KC"], _td_fixture_prices(by_id["2026_07_LV_KC"]))
    npd.attach_td_prices(by_id["2026_07_LV_KC2"], {})
    npd.show_td_prices(by_id["2026_06_LV_KC"])
    week["caveats"] = {"td_odds": nx.CAVEATS["td_odds"]}
    return json.loads(json.dumps(week)), by_id


def _td_row(html, name):
    at = html.index(f"<b>{_esc(name)}</b>")
    start = html.rindex('<li class="tdp">', 0, at)
    end = html.find('<li class="tdp">', at)
    return html[start:end if end > 0 else None]


@needs_node
def test_every_td_scorer_shows_his_odds_beside_our_chance_or_why_not(tmp_path):
    import nfl_odds as no
    import nfl_predict as npd
    week, games = _odds_week()
    out = _render(tmp_path, week)
    td = out["td"]
    cards = td.split('<div class="card tdgame"')[1:]
    shown = [g for g in week["games"] if g.get("home_td") or g.get("away_td")]
    assert len(cards) == len(shown)
    by_card = dict(zip((g["game_id"] for g in shown), cards))
    priced = games["2026_07_LV_KC"]
    for side in ("home", "away"):
        for p in priced[f"{side}_td_all"]:
            row = _td_row(by_card[priced["game_id"]], p["name"])
            assert f'<span class="num lead">{_pct(p["prob"], 0)}</span>' in row
            o = p["price"]
            if o["american"] is not None:
                assert f'<span class="num">{_american(o["american"])}</span> <span class="est">est.</span>' in row
                assert f'books <span class="num">{_pct(o["implied"], 0)}</span>' in row
                assert "no odds" not in row
            else:
                assert f"no odds: {_esc(o['missing'])}" in row and "books" not in row
    assert any(p["price"]["american"] is None for p in priced["home_td_all"])
    assert all(p["price"]["american"] is not None for p in priced["away_td_all"])
    # One reason for a whole list is said once above it, not beside every player.
    for gid, why in (("2026_07_LV_KC2", no.TD_PRICE_MISSING["none"]), ("2026_06_LV_KC", npd.NO_TD_PRICE_LOGGED)):
        card = by_card[gid]
        assert card.count(f'<div class="mini tdnoodds">Odds: {_esc(why)}</div>') == 2, gid
        assert "no odds:" not in card and "books <span" not in card and "tdodds" not in card
    # Every price beside a scorer is marked est., and the note beside the odds is the payload's.
    for view in ("td", "week"):
        chunks = out[view].split('<span class="end tdodds"')[1:]
        assert chunks and all('<span class="est">est.</span>' in c.split("</li>")[0] for c in chunks), view
        assert _esc(week["caveats"]["td_odds"]) in out[view], view
    # A ticket's TD leg says the chance its price implies, as the lists do.
    for t in priced["tickets"]:
        card = _card_of(out["parlays"], priced["game_id"], t["kind"], t["version"])
        for leg in t["legs"]:
            if leg["kind"] == "TD" and leg.get("implied") is not None:
                assert f'books {_pct(leg["implied"], 0)}' in card


@needs_node
def test_a_week_written_before_the_odds_or_v2_still_draws_and_says_so(tmp_path):
    """The committed week.json may be the runner's output from before v2 and
    the TD odds: the app draws it, and says the odds are not in the file."""
    week = json.loads((APP / "data" / "week.json").read_text())
    out = _render(tmp_path, week)
    assert "Parlays" in out["parlays"] and "Touchdowns" in out["td"]
    lists = [p for g in week["games"] for k in ("home_td_all", "away_td_all") for p in g.get(k) or []]
    if not any("price" in p for p in lists):
        assert "written before the touchdown lists carried the books" in out["td"]
    old, _ = _sgp_week()
    assert "written before the touchdown lists carried the books" in _render(tmp_path, old)["td"]


def test_the_odds_and_parlays_tab_code_types_no_number_the_payload_could_give():
    """The odds beside each scorer and the Parlays tab's own lines read every
    figure from week.json, as the tickets' block does."""
    js = (APP / "app.js").read_text()
    odds = js.split("// The books' anytime-TD price beside our chance")[1].split("// One line on the closed card")[0]
    tab = js.split("async parlays() {")[1].split("async teams() {")[0]
    for name, block in (("odds", odds), ("parlays", tab)):
        code = re.sub(r"^\s*//.*$", "", block, flags=re.M)
        assert not re.findall(r"\d%|\$\d|\b(?:19|20)\d\d\b|[+−-]\d{3}\b", code), name
        assert "fetch(" not in code and "localStorage" not in code, name
