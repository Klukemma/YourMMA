// YourNFL Pick Desk: the week's NFL games, the model's picks blended with
// the betting market, each team's form, and the model's own measured record.
//
// Data comes from data/*.json (served build) or from window.__YOURNFL_DATA__
// (the standalone file, nfl/engine/nfl_standalone.py). No module imports:
// the standalone build inlines this file as a classic script.

const $ = (id) => document.getElementById(id);
const pct = (x, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const signed = (x, d = 1) =>
  x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// American odds as a bookmaker prints them.
const odds = (o) => (o == null ? "—" : o > 0 ? `+${Math.round(o)}` : `${Math.round(o)}`);
// A spread from one side's point of view: "KC −7.5", "BUF +3", "PK".
const line = (team, pts) =>
  pts == null ? "—" : pts === 0 ? `${team} PK` : `${team} ${pts > 0 ? "+" : "−"}${Math.abs(pts)}`;

async function load(name) {
  const embedded = window.__YOURNFL_DATA__;
  if (embedded && embedded[name]) return embedded[name];
  const res = await fetch(`data/${name}.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name}.json — ${res.status}`);
  return res.json();
}

function fail(el, error, what) {
  el.innerHTML = `<div class="note err">Could not load ${esc(what)}: ${esc(error.message)}.
    The runner writes these files; run <b>python nfl/engine/nfl_predict.py</b> to build them locally.</div>`;
}

// ---------------------------------------------------------------- tabs

const TABS = ["week", "teams", "record", "method"];
const started = {};
function show(name) {
  for (const t of TABS) {
    $(`view-${t}`).hidden = t !== name;
    $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
  }
  try { localStorage.setItem("yournfl.tab", name); } catch (e) { /* private mode */ }
  if (!started[name]) { started[name] = true; BOOT[name](); }
}

// ---------------------------------------------------------------- form

// A team's record and last five games beside its name, newest on the left:
// green a win, red (ringed) a loss, grey a tie, and the number inside is the
// margin. This season's regular-season record; the last five reach back into
// last season when this one is young. Not a model input.
const RESULT_WORD = { W: "Win", L: "Loss", T: "Tie" };
const RESULT_CLASS = { W: "w", L: "l", T: "t" };
const FORM_LEGEND = `Beside each team: this season's record, then its last five games,
  newest first - green a win, red (ringed) a loss, grey a tie. The number is the
  margin: 7 in green is a seven-point win. Open a game for opponents and scores.`;

const GAME_WORD = { WC: "Wild Card", DIV: "Divisional", CON: "Conference final", SB: "Super Bowl" };
function weekName(b) {
  return b.game_type && b.game_type !== "REG" ? (GAME_WORD[b.game_type] || b.game_type) : `Week ${b.week}`;
}
function gameTip(b) {
  const where = b.neutral ? "vs" : b.home ? "vs" : "at";
  return `${RESULT_WORD[b.result] || b.result} ${b.score} ${where} ${b.opponent} · ${weekName(b)} ${b.season} · ${b.date}`;
}

function formLine(form) {
  // An absent key means the data predates records; null means no games yet.
  if (form === undefined) return "";
  if (!form || !form.record) return `<span class="form-r"><span class="rec none">no games yet</span></span>`;
  const r = form.record;
  const text = `${r.wins}-${r.losses}${r.ties ? `-${r.ties}` : ""}`;
  const dots = (form.last5 || []).map((b) => {
    const cls = RESULT_CLASS[b.result] || "t";
    const tip = esc(gameTip(b));
    return `<span class="res ${cls}" role="img" title="${tip}" aria-label="${tip}">${esc(b.code)}</span>`;
  }).join("");
  return `<span class="form-r"><span class="rec num">${esc(text)}</span><span class="last5">${dots}</span></span>`;
}

function formDetail(code, form) {
  if (!form || !form.last5 || !form.last5.length) return "";
  const items = form.last5.map((b) => {
    const cls = RESULT_CLASS[b.result] || "t";
    const where = b.home || b.neutral ? "vs" : "at";
    return `<li><span class="res ${cls}" aria-hidden="true">${esc(b.code)}</span>
      <div><b>${esc(RESULT_WORD[b.result] || b.result)} ${esc(b.score)}</b> ${where} ${esc(b.opponent)}
      <div class="mini">${esc(weekName(b))} ${esc(b.season)} · ${esc(b.date)}</div></div></li>`;
  }).join("");
  const po = form.playoffs ? ` · playoffs ${form.playoffs.wins}-${form.playoffs.losses}` : "";
  return `<div><div class="eyebrow">${esc(code)} · last 5${esc(po)}</div><ul class="lastfights">${items}</ul></div>`;
}

// ---------------------------------------------------------------- week

const TIER_TONE = { STRONG: "good", SOLID: "good", LEAN: "flat", "TOSS-UP": "flat" };
const STATUS_WORD = { final: "Final", "in progress": "In progress", upcoming: "" };

const kickTime = (iso) => new Date(iso).toLocaleString(undefined,
  { weekday: "short", hour: "numeric", minute: "2-digit" });
const dayKey = (iso) => new Date(iso).toLocaleDateString(undefined,
  { weekday: "long", month: "short", day: "numeric" });

function teamRow(g, side) {
  const code = g[side];
  // The nickname ("Chiefs"), so code, name, record and five circles fit one
  // line at phone width; the full name is the title.
  const full = g[`${side}_name`] || "";
  const name = full.split(" ").slice(-1)[0];
  const lead = side === "home" ? `<span class="at">${g.neutral ? "vs" : "@"}</span>` : "";
  const score = g.score ? `<span class="sc">${esc(g.score[side])}</span>` : "";
  return `<div class="frow ${side}"><span class="team">${lead}<span class="code">${esc(code)}</span>
    <span class="nm" title="${esc(full)}">${esc(name)}</span></span>${score}${formLine(g[`${side}_form`])}</div>`;
}

function marginText(g, m) {
  if (m == null) return "—";
  if (Math.abs(m) < 0.05) return "even";
  return m > 0 ? `${esc(g.home)} by ${Math.abs(m).toFixed(1)}` : `${esc(g.away)} by ${Math.abs(m).toFixed(1)}`;
}

function resultTag(g) {
  if (g.status !== "final" || !g.pick || !g.score) return "";
  const margin = g.score.home - g.score.away;
  if (margin === 0) return `<span class="tag flat">Tie</span>`;
  const won = (margin > 0 ? g.home : g.away) === g.pick;
  return won ? `<span class="tag good">Right</span>` : `<span class="tag bad">Wrong</span>`;
}

function qbText(qb) {
  if (!qb) return "—";
  const src = qb.replaced ? ` (starts for ${qb.replaced}, ruled out)`
    : qb.source === "last starter" ? " (assumed: last starter)" : qb.source === "projected" ? " (projected)" : "";
  const first = qb.starts === 0 ? " · first start" : qb.change ? " · new starter" : "";
  const hurt = qb.injury_status ? ` · <b>${esc(qb.injury_status)} on the injury report</b>` : "";
  return `${esc(qb.name || qb.id)}${esc(src)} · ${signed(qb.rating, 2)} EPA/dropback${esc(first)}${hurt}`;
}

function injuryList(code, list) {
  if (!list || !list.length) return `<div><div class="eyebrow">${esc(code)} injuries</div><div class="mini">None listed out, doubtful or questionable.</div></div>`;
  const items = list.map((i) => `<li><span class="st ${esc(String(i.status).toLowerCase())}">${esc(i.status)}</span>
    <span><b>${esc(i.name)}</b> ${esc(i.position)}${i.injury ? ` · ${esc(i.injury)}` : ""}</span></li>`).join("");
  return `<div><div class="eyebrow">${esc(code)} injuries</div><ul class="inj">${items}</ul></div>`;
}

// ---------------------------------------------------------------- players

// Key players, touchdown chances and news - shown, not applied: none of it
// moves the pick except the starting quarterback.
const AVAIL_WORD = { out: "Out", doubtful: "Doubtful", questionable: "Questionable", active: "" };

function alertLine(a) {
  return `<div class="alert ${esc(a.status)}"><b>${esc(a.team)} ${esc(a.name)}</b>
    <span class="mini">${esc(a.position)} · ${esc(a.role)}</span> - ${esc(a.reason || AVAIL_WORD[a.status] || a.status)}</div>`;
}

function statusBadge(p) {
  return p.status && p.status !== "active"
    ? ` <span class="st ${esc(p.status)}">${esc(AVAIL_WORD[p.status] || p.status)}</span>` : "";
}

function keyPlayers(code, kp) {
  if (!kp) return "";
  const off = (kp.offense || []).map((p) => `<li><span class="rk num">${esc(p.rank)}</span>
    <span><b>${esc(p.name)}</b> ${esc(p.position)}${statusBadge(p)}
    <div class="mini">${signed(p.value, 1)} EPA · ${esc(p.line)}</div></span></li>`).join("");
  const dfn = (kp.defense || []).map((p) => `<li><span class="rk num">D</span>
    <span><b>${esc(p.name)}</b> ${esc(p.position)}${statusBadge(p)}
    <div class="mini">${esc(p.line)}</div></span></li>`).join("");
  if (!off && !dfn) return "";
  return `<div><div class="eyebrow">${esc(code)} · most valuable</div><ul class="kp">${off}${dfn}</ul></div>`;
}

function tdList(code, list) {
  if (!list || !list.length) return "";
  const items = list.map((p) => `<li><span class="num tdp">${pct(p.prob)}</span>
    <span><b>${esc(p.name)}</b> ${esc(p.position)}${statusBadge(p)}</span></li>`).join("");
  return `<div><div class="eyebrow">${esc(code)} · touchdown</div><ul class="kp">${items}</ul></div>`;
}

function ago(iso) {
  const h = (Date.now() - new Date(iso).getTime()) / 3.6e6;
  if (!(h >= 0)) return "";
  return h < 1 ? "just now" : h < 24 ? `${Math.round(h)}h ago` : `${Math.round(h / 24)}d ago`;
}

function newsList(items) {
  if (!items || !items.length) return "";
  const li = items.map((n) => {
    const safe = /^https?:\/\//.test(n.link || "");
    const title = safe ? `<a href="${esc(n.link)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>` : esc(n.title);
    const who = (n.players || []).length ? ` · ${esc(n.players.join(", "))}` : "";
    return `<li>${title}<div class="mini">${esc(n.source)} · ${esc(ago(n.published))}${who}</div></li>`;
  }).join("");
  return `<div class="news"><div class="eyebrow">Player news · shown, not applied</div><ul class="plain news">${li}</ul></div>`;
}

function gameRow(g, d) {
  const p = g.home_prob;
  const awayShare = p == null ? null : 1 - p;
  const bar = p == null
    ? `<div class="corner-bar"><i class="b" style="flex:1"></i><i class="r" style="flex:1"></i></div>`
    : `<div class="corner-bar"><i class="b" style="flex:${awayShare}"></i><i class="r" style="flex:${p}"></i></div>`;
  const statusWord = STATUS_WORD[g.status] || "";
  const status = statusWord
    ? `<span class="status ${g.status === "final" ? "final" : "live"}">${esc(statusWord)}</span>` : "";
  const extra = [g.neutral ? "Neutral site" : "", g.div_game ? "Division" : ""].filter(Boolean).join(" · ");

  const tags = [];
  if (g.tier) tags.push(`<span class="tag ${TIER_TONE[g.tier] || "flat"}" title="Picks in this band won ${pct(g.tier_hit_rate, 1)} of 2016-2025 games">${esc(g.tier)}</span>`);
  const m = g.market || {};
  const pickOdds = g.pick === g.home ? m.home_ml : g.pick === g.away ? m.away_ml : null;
  if (pickOdds != null) tags.push(`<span class="tag flat num">${esc(odds(pickOdds))}</span>`);
  for (const side of ["away", "home"]) {
    const qb = g[`${side}_qb`];
    if (qb && qb.injury_status) tags.push(`<span class="tag warn">${esc(g[side])} QB ${esc(qb.injury_status)}</span>`);
    if (qb && qb.replaced) tags.push(`<span class="tag warn">${esc(g[side])} backup QB</span>`);
  }
  const mvpOut = (g.alerts || []).filter((a) => a.role === "most valuable player" && a.status !== "questionable");
  for (const a of mvpOut) tags.push(`<span class="tag bad">${esc(a.team)} MVP ${esc(AVAIL_WORD[a.status] || a.status)}</span>`);
  if ((g.news || []).length) tags.push(`<span class="tag flat">News ${(g.news || []).length}</span>`);
  const injured = (g.home_injuries || []).length + (g.away_injuries || []).length;
  if (injured) tags.push(`<span class="tag flat">Injuries ${injured}</span>`);
  tags.push(resultTag(g));

  const readout = g.pick
    ? `<span class="pick">${esc(g.pick)}</span><span class="pct num">${pct(g.pick_prob)}</span>${tags.join("")}`
    : `<span class="mini">${esc(g.no_pick_reason || "No pick.")}</span>${tags.join("")}`;

  const sp = g.spread_pick, tp = g.total_pick;
  const kv = [
    ["Pick", g.pick ? `${esc(g.pick)} ${pct(g.pick_prob, 1)} <span class="mini">model and market blended</span>` : "—"],
    ["Band", g.tier ? `${esc(g.tier)} <span class="mini">picks in this band won ${pct(g.tier_hit_rate, 1)} of 2016-2025 games</span>` : "—"],
    ["Model alone", g.model_prob == null ? "—" : `${esc(g.away)} ${pct(1 - g.model_prob, 1)} · ${esc(g.home)} ${pct(g.model_prob, 1)}`],
    ["Market", g.market_prob == null ? "—" : `${esc(g.away)} ${pct(1 - g.market_prob, 1)} · ${esc(g.home)} ${pct(g.market_prob, 1)} <span class="mini">no-vig, ${esc(m.source === "the-odds-api" ? `${m.books ?? "?"} books` : "schedule line")}</span>`],
    ["Moneyline", m.home_ml == null ? "—" : `${esc(g.away)} ${esc(odds(m.away_ml))} · ${esc(g.home)} ${esc(odds(m.home_ml))}`],
    ["Spread", g.market_spread == null ? "—" : `market ${esc(line(g.home, -g.market_spread))} · model ${marginText(g, g.model_spread)}`
      + (sp ? ` <span class="mini">model side ${esc(line(sp.team, sp.line))}, ${num(sp.edge, 1)} pts</span>` : "")],
    ["Total", g.market_total == null ? (g.model_total == null ? "—" : `model ${num(g.model_total, 1)}`)
      : `market ${num(g.market_total, 1)} · model ${num(g.model_total, 1)}` + (tp ? ` <span class="mini">model side ${esc(tp.side)}</span>` : "")],
    ["QB", `${esc(g.away)}: ${qbText(g.away_qb)}<br>${esc(g.home)}: ${qbText(g.home_qb)}`],
    ["Rest", `${esc(g.away)} ${g.away_rest == null ? "—" : Math.round(g.away_rest) + " days"} · ${esc(g.home)} ${g.home_rest == null ? "—" : Math.round(g.home_rest) + " days"}`],
    ["Elo", `${esc(g.away)} ${num(g.elo && g.elo.away, 0)} · ${esc(g.home)} ${num(g.elo && g.elo.home, 0)}`],
    ["Venue", `${esc(g.stadium || "—")}${g.roof ? ` · ${esc(g.roof)}` : ""}`],
  ];
  if (m.as_of) kv.push(["Prices", `${esc(m.as_of.replace("T", " ").replace("Z", " UTC"))}`]);
  if (g.logged_at) kv.push(["Logged", `${esc(g.logged_at.replace("T", " ").replace("Z", " UTC"))} <span class="mini">before kickoff</span>`]);

  const qbAlert = ["away", "home"].some((s) => g[`${s}_qb`] && g[`${s}_qb`].injury_status)
    ? `<div class="note warnbox">${esc(d.caveats.qb)}</div>` : "";

  return `<details class="bout">
    <summary>
      <div class="when"><span class="t">${esc(kickTime(g.kickoff))}</span>${status}<span class="mini">${esc(extra)}</span></div>
      <div class="names">${teamRow(g, "away")}${teamRow(g, "home")}</div>
      ${(g.alerts || []).slice(0, 3).map(alertLine).join("")}
      ${(g.alerts || []).length > 3 ? `<div class="mini">+${g.alerts.length - 3} more in the detail</div>` : ""}
      ${bar}
      <div class="readout">${readout}</div>
    </summary>
    <div class="detail">
      <dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>
      ${qbAlert}
      ${(g.alerts || []).length > 3 ? g.alerts.slice(3).map(alertLine).join("") : ""}
      <div class="two">${keyPlayers(g.away, g.away_players)}${keyPlayers(g.home, g.home_players)}</div>
      ${(g.away_td || g.home_td) ? `<div class="eyebrow">Most likely to score a touchdown</div>
        <div class="two">${tdList(g.away, g.away_td)}${tdList(g.home, g.home_td)}</div>
        <div class="mini">${esc((d.caveats && d.caveats.td) || "")}</div>` : ""}
      ${newsList(g.news)}
      <div class="two">${injuryList(g.away, g.away_injuries)}${injuryList(g.home, g.home_injuries)}</div>
      <div class="two">${formDetail(g.away, g.away_form)}${formDetail(g.home, g.home_form)}</div>
    </div>
  </details>`;
}

function upsetCard(games) {
  const open = games.filter((g) => g.status === "upcoming" && g.pick && g.pick_prob != null)
    .sort((a, b) => a.pick_prob - b.pick_prob).slice(0, 6);
  if (!open.length) return "";
  const items = open.map((g) => {
    const opp = g.pick === g.home ? g.away : g.home;
    const wrong = 1 - g.pick_prob;
    return `<li><span><b>${esc(g.pick)}</b> over ${esc(opp)}</span>
      <span class="num ${wrong >= 0.4 ? "warn" : "flat"}">${pct(wrong)}</span></li>`;
  }).join("");
  return `<div class="card"><h2>Upset watch</h2>
    <div class="mini">Chance each pick is wrong, most doubtful first. The probabilities are calibrated:
    picks the model gave about 60% won about 59% of 2016-2025 games.</div>
    <ul class="plain doubt">${items}</ul></div>`;
}

const BOOT = {
  async week() {
    const el = $("view-week");
    let d;
    try { d = await load("week"); } catch (e) { return fail(el, e, "the week"); }
    $("stamp").textContent = `data to ${d.data_through ?? "—"}`;
    const games = d.games || [];
    const predicted = games.filter((g) => g.pick).length;
    const byDay = new Map();
    for (const g of games) {
      const k = dayKey(g.kickoff);
      if (!byDay.has(k)) byDay.set(k, []);
      byDay.get(k).push(g);
    }
    const days = [...byDay.entries()].map(([day, list]) =>
      `<div class="eyebrow day">${esc(day)}</div>${list.map((g) => gameRow(g, d)).join("")}`).join("");
    const notes = ["tiers", "value_off", "lines", "players", "news", "injuries"]
      .filter((k) => d.caveats && d.caveats[k])
      .map((k) => `<div class="note">${esc(d.caveats[k])}</div>`).join("");
    el.innerHTML = `
      <div class="event">
        <div class="eyebrow">${esc(d.season)} season</div>
        <h1>${esc(d.label)}</h1>
        <div class="mini">${games.length} games · ${predicted} with a pick · model trained on
          ${Number(d.trained_on || 0).toLocaleString()} games</div>
      </div>
      <div class="note warnbox">${esc(d.caveats && d.caveats.model)}</div>
      ${upsetCard(games)}
      <div class="mini legend">${esc(FORM_LEGEND)}</div>
      <div class="bouts">${days}</div>
      ${notes}`;
  },

  async teams() {
    const el = $("view-teams");
    let d;
    try { d = await load("teams"); } catch (e) { return fail(el, e, "the teams"); }
    const teams = d.teams || [];
    const rows = teams.map((t) => {
      const r = t.record || {};
      const mvp = ((t.key_players || {}).offense || [])[0];
      return `<tr><td class="n">${esc(t.rank)}</td>
        <td><b>${esc(t.team)}</b> <span class="mini">${esc(r.wins ?? 0)}-${esc(r.losses ?? 0)}${r.ties ? `-${esc(r.ties)}` : ""}</span></td>
        <td class="n">${num(t.elo, 0)}</td><td class="n">${signed(t.net_epa, 2)}</td>
        <td>${mvp ? `${esc(mvp.name)} <span class="mini">${esc(mvp.position)}</span>` : "—"}</td></tr>`;
    }).join("");
    const options = teams.map((t) => `<option value="${esc(t.team)}">${esc(t.team)} · ${esc(t.name)}</option>`).join("");
    el.innerHTML = `
      <div class="event"><div class="eyebrow">${esc(d.season)} season</div><h1>Teams</h1>
        <div class="mini">Ranked by Elo, after every game played so far.</div></div>
      <div class="card"><h2>Compare</h2>
        <div class="pickers">
          <div class="field b"><label for="cmp-a">Team</label><select id="cmp-a">${options}</select></div>
          <div class="field r"><label for="cmp-b">Team</label><select id="cmp-b">${options}</select></div>
        </div>
        <div id="cmp-out"></div>
      </div>
      <div class="card"><h2>Power ratings</h2>
        <div class="scroll"><table>
          <thead><tr><th class="n">#</th><th>Team</th><th class="n">Elo</th><th class="n">Net EPA</th><th>Most valuable</th></tr></thead>
          <tbody>${rows}</tbody></table></div>
        <div class="mini">Net EPA is expected points added per play on offence minus the same allowed on
          defence, weighted toward recent games. Most valuable is the player whose plays added the most
          expected points this season.</div>
      </div>`;
    const a = $("cmp-a"), b = $("cmp-b");
    if (teams.length > 1) { a.value = teams[0].team; b.value = teams[1].team; }
    const render = () => renderCompare(teams, a.value, b.value, d.elo_scale || {});
    a.addEventListener("change", render);
    b.addEventListener("change", render);
    render();
  },

  async record() {
    const el = $("view-record");
    let d;
    try { d = await load("performance"); } catch (e) { return fail(el, e, "the record"); }
    el.innerHTML = renderRecord(d);
  },

  method() {
    $("view-method").innerHTML = METHOD_HTML;
  },
};

// ---------------------------------------------------------------- teams

// Elo alone, on a neutral field: the transparent number anyone can check.
// It is NOT the model's pick, which also reads quarterbacks, efficiency,
// rest and the market.
function eloChance(a, b) { return 1 / (1 + Math.pow(10, -(a - b) / 400)); }

const TAPE = [
  ["Record", (t) => { const r = t.record || {}; return `${r.wins ?? 0}-${r.losses ?? 0}${r.ties ? `-${r.ties}` : ""}`; }, null, (v) => v],
  ["Elo", (t) => t.elo, "high", (v) => num(v, 0)],
  ["Net EPA/play", (t) => t.net_epa, "high", (v) => signed(v, 3)],
  ["Offence EPA/play", (t) => t.off_epa, "high", (v) => signed(v, 3)],
  ["Defence EPA/play allowed", (t) => t.def_epa, "low", (v) => signed(v, 3)],
  ["Points scored", (t) => t.points_for, "high", (v) => num(v, 1)],
  ["Points allowed", (t) => t.points_against, "low", (v) => num(v, 1)],
  ["QB EPA/dropback", (t) => t.qb_rating, "high", (v) => signed(v, 3)],
];

function renderCompare(teams, codeA, codeB, scale) {
  const out = $("cmp-out");
  const A = teams.find((t) => t.team === codeA), B = teams.find((t) => t.team === codeB);
  if (!A || !B) { out.innerHTML = ""; return; }
  if (A.team === B.team) { out.innerHTML = `<div class="mini">Pick two different teams.</div>`; return; }
  const rows = TAPE.map(([lab, get, better, fmt]) => {
    const va = get(A), vb = get(B);
    let wa = "", wb = "";
    if (better && va != null && vb != null && va !== vb) {
      const aWins = better === "high" ? va > vb : va < vb;
      wa = aWins ? " win" : ""; wb = aWins ? "" : " win";
    }
    return `<div class="l${wa}">${esc(fmt(va))}</div><div class="lab">${esc(lab)}</div><div class="r2${wb}">${esc(fmt(vb))}</div>`;
  }).join("");
  const p = A.elo != null && B.elo != null ? eloChance(A.elo, B.elo) : null;
  const pts = p == null ? null : (A.elo - B.elo) * (scale.points_per_elo || 0.04);
  out.innerHTML = `
    <div class="tape"><div class="l"><b>${esc(A.team)}</b></div><div class="lab"></div><div class="r2"><b>${esc(B.team)}</b></div>${rows}</div>
    <div class="note">On a neutral field, Elo alone gives <b>${esc(A.team)} ${pct(p)}</b>
      (${pts == null ? "—" : pts >= 0 ? `${esc(A.team)} by ${pts.toFixed(1)}` : `${esc(B.team)} by ${(-pts).toFixed(1)}`}).
      Home field is worth about ${num((scale.hfa || 0) * (scale.points_per_elo || 0.04), 1)} points more.
      This is the Elo rating only - not the model's pick, which also reads quarterbacks, efficiency, rest and the market.</div>
    <div class="two">${keyPlayers(A.team, A.key_players)}${keyPlayers(B.team, B.key_players)}</div>`;
}

// ---------------------------------------------------------------- record

function roiCell(r) {
  if (!r || r.roi == null) return `<td class="n">—</td>`;
  return `<td class="n ${r.roi >= 0 ? "gain" : "loss"}">${signed(r.roi * 100, 1)}% <span class="mini">${esc(r.bets)}</span></td>`;
}

function seasonChart(rows) {
  if (!rows || rows.length < 2) return "";
  const W = 640, H = 200, L = 34, R = 10, T = 12, B = 26;
  const lo = 0.5, hi = 0.8;
  const x = (i) => L + (i * (W - L - R)) / (rows.length - 1);
  const y = (v) => T + (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (H - T - B);
  const path = (key) => rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join("");
  const grid = [0.5, 0.6, 0.7, 0.8].map((v) =>
    `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line-soft)"/>
     <text x="${L - 6}" y="${y(v) + 3}" font-size="10" text-anchor="end" fill="var(--ink-3)">${Math.round(v * 100)}%</text>`).join("");
  const labels = rows.map((r, i) => (i % 3 === 0 || i === rows.length - 1)
    ? `<text x="${x(i)}" y="${H - 8}" font-size="10" text-anchor="middle" fill="var(--ink-3)">${esc(String(r.season))}</text>` : "").join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by season, model blend against the market favourite">
    ${grid}${labels}
    <path d="${path("market_accuracy")}" fill="none" stroke="var(--ink-3)" stroke-width="2" stroke-dasharray="4 3"/>
    <path d="${path("accuracy")}" fill="none" stroke="var(--red)" stroke-width="2.2"/>
  </svg>
  <div class="legend"><span><i style="background:var(--red)"></i>Pick (blend)</span>
    <span><i style="background:var(--ink-3)"></i>Market favourite</span></div>`;
}

function renderRecord(d) {
  const b = d.backtest || {};
  const per = b.periods || { dev: [2006, 2015], confirm: [2016, 2025] };
  const dev = `${per.dev[0]}-${per.dev[1]}`, conf = `${per.confirm[0]}-${per.confirm[1]}`;
  const t = d.track_record || {};
  const live = t.graded
    ? `${pct(t.accuracy, 1)}` : "—";
  const tiles = `
    <div class="tiles">
      <div class="tile"><div class="k">Live picks</div><div class="v">${esc(live)}</div>
        <div class="s">${esc(t.correct ?? 0)} of ${esc(t.graded ?? 0)} right</div></div>
      <div class="tile"><div class="k">Pending</div><div class="v">${esc(t.pending ?? 0)}</div>
        <div class="s">logged before kickoff</div></div>
      <div class="tile"><div class="k">Model side ATS</div><div class="v">${esc((t.ats || {}).correct ?? 0)}-${esc(((t.ats || {}).picks ?? 0) - ((t.ats || {}).correct ?? 0))}</div>
        <div class="s">against the logged spread</div></div>
      <div class="tile"><div class="k">Totals</div><div class="v">${esc((t.totals || {}).correct ?? 0)}-${esc(((t.totals || {}).picks ?? 0) - ((t.totals || {}).correct ?? 0))}</div>
        <div class="s">model side of the total</div></div>
    </div>`;
  const liveNote = t.graded ? "" : `<div class="note">No graded picks yet: the live record starts with the first games
    picked before kickoff, and grows every week. The tables below are the walk-forward backtest.</div>`;

  const W = b.winner || {};
  const wrow = (key, label) => {
    const r = W[key] || {};
    const a = r[dev] || {}, c = r[conf] || {};
    return `<tr><td>${esc(label)}</td><td class="n">${pct(a.accuracy, 1)}</td><td class="n">${pct(c.accuracy, 1)}</td>
      <td class="n">${num(c.log_loss, 4)}</td></tr>`;
  };
  const mkt = (W.blend || {})[conf] || {}, mktDev = (W.blend || {})[dev] || {};
  const winners = `<div class="card"><h2>Picking winners</h2>
    <div class="scroll"><table><thead><tr><th></th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th><th class="n">Log loss</th></tr></thead>
    <tbody>${wrow("elo", "Elo alone")}${wrow("model", "Model alone")}${wrow("blend", "Pick (blend)")}
    <tr><td>Market favourite</td><td class="n">${pct(mktDev.market_accuracy, 1)}</td><td class="n">${pct(mkt.market_accuracy, 1)}</td>
      <td class="n">${num(mkt.market_log_loss, 4)}</td></tr></tbody></table></div>
    <div class="mini">Every season predicted by models fitted only on the seasons before it.
      ${esc(dev)} is where choices were made; ${esc(conf)} was left untouched to judge them. Lower log loss is better.</div>
    ${seasonChart(b.by_season)}</div>`;

  const tiers = (b.tiers || {})[conf] || {};
  const tierRows = ["STRONG", "SOLID", "LEAN", "TOSS-UP"].map((k) => {
    const r = tiers[k] || {};
    return `<tr><td><b>${esc(k)}</b></td><td class="n">${esc(r.picks ?? "—")}</td><td class="n">${pct(r.hit_rate, 1)}</td><td class="n">${pct(r.mean_prob, 1)}</td></tr>`;
  }).join("");
  const tierCard = `<div class="card"><h2>Confidence bands</h2>
    <div class="scroll"><table><thead><tr><th>Band</th><th class="n">Picks</th><th class="n">Won</th><th class="n">Said</th></tr></thead>
    <tbody>${tierRows}</tbody></table></div>
    <div class="mini">${esc(conf)}. "Said" is the average probability the pick was given; "won" is how often it won.
      Close together means the percentages can be taken at face value.</div></div>`;

  const v = b.value || {};
  const vrows = Object.entries(v.edges || {}).map(([edge, r]) =>
    `<tr><td class="n">${pct(Number(edge))}</td>${roiCell(r[dev])}${roiCell(r[conf])}</tr>`).join("");
  const valueCard = `<div class="card"><h2>Betting the disagreements</h2>
    <div class="scroll"><table><thead><tr><th class="n">Edge</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${vrows}</tbody></table></div>
    <div class="mini">One unit at the closing moneyline on every side the model alone rated at least this far above
      the no-vig market (return per bet, bets beside). ${v.enabled_at == null
        ? "It lost money in the later period at every edge, so no game is ever labelled a bet."
        : `Profitable in both periods from a ${esc(pct(v.enabled_at))} edge.`}</div></div>`;

  const lineRows = (rep) => Object.entries((rep || {})[conf] || {}).map(([edge, r]) => {
    const a = ((rep || {})[dev] || {})[edge] || {};
    return `<tr><td class="n">${esc(edge)}+</td><td class="n">${pct(a.hit_rate, 1)} <span class="mini">${esc(a.bets ?? "")}</span></td>
      <td class="n">${pct(r.hit_rate, 1)} <span class="mini">${esc(r.bets ?? "")}</span></td></tr>`;
  }).join("");
  const linesCard = `<div class="card"><h2>Spread and total</h2>
    <div class="eyebrow">Model side against the closing spread</div>
    <div class="scroll"><table><thead><tr><th class="n">Gap (pts)</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${lineRows(b.spread)}</tbody></table></div>
    <div class="eyebrow">Model side against the closing total</div>
    <div class="scroll"><table><thead><tr><th class="n">Gap (pts)</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${lineRows(b.total)}</tbody></table></div>
    <div class="mini">Pushes left out. At -110 on both sides it takes 52.4% to break even.</div></div>`;

  const cal = (b.calibration || []).map((r) =>
    `<tr><td class="n">${pct(r.lo)}-${pct(r.hi)}</td><td class="n">${esc(r.games)}</td><td class="n">${pct(r.predicted, 1)}</td><td class="n">${pct(r.observed, 1)}</td></tr>`).join("");
  const calCard = cal ? `<div class="card"><h2>Calibration</h2>
    <div class="scroll"><table><thead><tr><th class="n">Home win %</th><th class="n">Games</th><th class="n">Said</th><th class="n">Happened</th></tr></thead>
    <tbody>${cal}</tbody></table></div><div class="mini">${esc(conf)}, the pick's probability for the home side.</div></div>` : "";

  const weeks = (t.by_week || []).slice(-8).reverse().map((w) =>
    `<tr><td>${esc(w.season)} week ${esc(w.week)}</td><td class="n">${esc(w.correct)}-${esc(w.picks - w.correct)}</td><td class="n">${pct(w.correct / w.picks)}</td></tr>`).join("");
  const weekCard = weeks ? `<div class="card"><h2>Live, week by week</h2><div class="scroll"><table>
    <thead><tr><th>Week</th><th class="n">Record</th><th class="n">Hit</th></tr></thead><tbody>${weeks}</tbody></table></div></div>` : "";

  return `<div class="event"><div class="eyebrow">The model's own record</div><h1>Record</h1>
      <div class="mini">Live picks are graded against final scores; the backtest replays every season since ${esc(per.dev[0])}.</div></div>
    ${tiles}${liveNote}${weekCard}${winners}${tierCard}${valueCard}${linesCard}${calCard}`;
}

// ---------------------------------------------------------------- method

const METHOD_HTML = `
  <div class="event"><div class="eyebrow">How it works</div><h1>Method</h1></div>
  <div class="card"><h2>Three numbers, one pick</h2>
    <ul class="plain">
      <li><b>Elo</b> - every team's rating moves after each game by how surprising the result was,
        more for a blowout, and is pulled part-way back to average every offseason.</li>
      <li><b>The model</b> - a logistic regression on Elo, offensive and defensive efficiency
        (expected points added per play, weighted toward recent games), the starting quarterback's own
        efficiency per dropback, rest, bye weeks, short weeks, travel across time zones, division games
        and neutral sites.</li>
      <li><b>The pick</b> - the model blended with the betting market's no-vig price, with the weights
        learned from how the two did on earlier seasons. The market is very good in the NFL, so the pick
        leans on it heavily.</li>
    </ul></div>
  <div class="card"><h2>Every number is point-in-time</h2>
    <ul class="plain">
      <li>Each game's features are read before kickoff and the games are replayed in kickoff order, so no
        prediction ever sees a result that was not final.</li>
      <li>Each season is predicted by models fitted only on the seasons before it - the same procedure for
        the backtest and for this week.</li>
      <li>A pick is logged before kickoff and frozen when the game starts; the live record grades only those.</li>
    </ul></div>
  <div class="card"><h2>What it does not do</h2>
    <ul class="plain">
      <li>Injuries other than the quarterback are shown, not applied. Nothing soft moves a number until it has
        been measured to.</li>
      <li>Weather is left out: the schedule records it after the game, and a forecast was never stored.</li>
      <li>No game is labelled a bet. Backing the model against the closing price lost money in 2016-2025.</li>
    </ul></div>
  <div class="note">Data: nflverse (schedules and closing lines, team and quarterback statistics, injury reports).
    Prices: the-odds-api when a key is set, otherwise the schedule's posted lines. For entertainment and
    analysis - not betting advice.</div>`;

// ---------------------------------------------------------------- start

for (const t of TABS) $(`tab-${t}`).addEventListener("click", () => show(t));
let first = "week";
try { const saved = localStorage.getItem("yournfl.tab"); if (TABS.includes(saved)) first = saved; } catch (e) { /* ignore */ }
show(first);

if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  window.addEventListener("load", () => { navigator.serviceWorker.register("sw.js").catch(() => {}); });
}
