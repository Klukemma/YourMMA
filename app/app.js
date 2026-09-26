// YourMMA on a phone.
//
// Everything here is read from data/*.json, which the engine writes. The one
// thing computed in the browser is the head-to-head matchup on the Compare
// tab, and that runs through matchup.js - a checked port of engine/matchup.py,
// not a second opinion.
//
// The rule the whole project runs on applies to the rendering too: a missing
// number is shown as missing. Never a zero, never a dash that could be read as
// "even", and never a probability the engine did not produce.

import { makeMatchup, makeForm } from "./matchup.js";

const $ = (id) => document.getElementById(id);
const pct = (x, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const signed = (x, d = 1) =>
  x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
const num = (x, d = 2) => (x == null ? "—" : x.toFixed(d));
const clock = (s) =>
  s == null ? "—" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Two ways in. Served from a directory, this fetches data/<name>.json. Built
// as the standalone file, the data is already on the page and there is nothing
// to fetch - which matters because a phone opening a file:// page is not
// allowed to fetch its own neighbours, so the served build simply cannot work
// offline and the standalone one cannot use fetch at all.
async function load(name) {
  const embedded = window.__YOURMMA_DATA__;
  if (embedded && embedded[name]) return embedded[name];
  const res = await fetch(`data/${name}.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name}.json — ${res.status}`);
  return res.json();
}

function fail(el, error, what) {
  el.innerHTML =
    `<div class="note err"><b>Could not load ${esc(what)}.</b><br>` +
    `${esc(error.message)}<br>` +
    `This page reads files the engine writes into <code>app/data/</code>. ` +
    `If they are missing, run <code>python3 engine/build_app_data.py</code>.</div>`;
}

/* ------------------------------------------------------------------ tabs */
const TABS = ["card", "compare", "record", "method"];
const started = {};
function show(name) {
  for (const t of TABS) {
    $(`view-${t}`).hidden = t !== name;
    $(`tab-${t}`).setAttribute("aria-selected", String(t === name));
  }
  try { localStorage.setItem("yourmma.tab", name); } catch (e) { /* private mode */ }
  if (!started[name]) { started[name] = true; BOOT[name](); }
}
for (const t of TABS) $(`tab-${t}`).addEventListener("click", () => show(t));

/* ------------------------------------------------------------------ card */
const TAG_TONE = {
  "STRONG BET": "good", "GOOD BET": "good", LEAN: "flat",
  RISKY: "warn", AVOID: "warn", LOCK: "good", STRONG: "good", VALUE: "warn",
};

function boutRow(f) {
  const p = f.win_prob;
  const redPicked = f.pick === f.red;
  // The bar is the corners, which is what the sport actually calls them.
  const redShare = p == null ? null : (redPicked ? p : 1 - p);
  const sim = f.simulation;

  const bar = redShare == null
    ? `<div class="corner-bar"><i class="r" style="flex:1"></i></div>`
    : `<div class="corner-bar">
         <i class="r" style="flex:${redShare}"></i>
         <i class="b" style="flex:${1 - redShare}"></i>
       </div>`;

  const tags = [];
  if (f.recommendation)
    tags.push(`<span class="tag ${TAG_TONE[f.recommendation] || "flat"}">${esc(f.recommendation)}</span>`);
  if (f.parlay_tier)
    tags.push(`<span class="tag ${TAG_TONE[f.parlay_tier] || "flat"}">Parlay ${esc(f.parlay_tier)}</span>`);
  if (f.rounds_scheduled === 5)
    tags.push(`<span class="tag flat">5 rounds</span>`);
  if (f.odds != null)
    tags.push(`<span class="tag flat num">${f.odds > 0 ? "+" : ""}${f.odds}</span>`);

  let detail = "";
  if (sim) {
    const ko = sim.ko ?? 0, sb = sim.sub ?? 0, dc = sim.decision ?? 0;
    const seg = (cls, v, label) =>
      v > 0.07 ? `<i class="${cls}" style="flex:${v}">${label} ${Math.round(v * 100)}</i>`
               : `<i class="${cls}" style="flex:${v}"></i>`;
    const peak = Math.max(...(sim.finish_by_round || [0]), 0.0001);
    const rounds = (sim.finish_by_round || [])
      .map((v, i) => `<div><b style="height:${Math.max(3, (v / peak) * 34)}px"></b>
         <span>R${i + 1}</span></div>`).join("");
    const assumed = (sim.assumed || []).length
      ? `<div class="mini">Assumed from the league average: ${esc(sim.assumed.join(", "))}.
         Those rates were not measured for these fighters.</div>` : "";
    detail = `
      <div class="detail">
        <div>
          <div class="eyebrow">Simulated ${(sim.simulations || 0).toLocaleString()} times</div>
          <dl class="kv">
            <dt>${esc(f.red)}</dt><dd class="num">${pct(sim.red_win, 1)}</dd>
            <dt>${esc(f.blue)}</dt><dd class="num">${pct(sim.blue_win, 1)}</dd>
            <dt>Draw</dt><dd class="num">${pct(sim.draw, 1)}</dd>
            <dt>Lasts</dt><dd class="num">${clock(sim.mean_seconds)} on average</dd>
          </dl>
        </div>
        <div>
          <div class="eyebrow">How it ends</div>
          <div class="method" role="img"
               aria-label="KO ${pct(ko)}, submission ${pct(sb)}, decision ${pct(dc)}">
            ${seg("ko", ko, "KO")}${seg("sb", sb, "SUB")}${seg("dc", dc, "DEC")}
          </div>
          <div class="mini" style="margin-top:6px">
            KO ${pct(ko)} · Submission ${pct(sb)} · Decision ${pct(dc)}
          </div>
        </div>
        ${rounds ? `<div><div class="eyebrow">Finish by round</div>
          <div class="rounds">${rounds}</div></div>` : ""}
        ${assumed}
      </div>`;
  } else {
    detail = `<div class="detail"><div class="mini">
      No simulation: too much of these fighters' record is unmeasured, so the
      only honest answer is nothing.</div></div>`;
  }

  return `<details class="bout">
    <summary>
      <div class="names">
        <span class="r">${esc(f.red)}</span>
        <span class="vs">vs</span>
        <span class="b">${esc(f.blue)}</span>
      </div>
      ${bar}
      <div class="readout">
        <span class="pick">${esc(f.pick ?? "—")}</span>
        <span class="pct num">${pct(p, 1)}</span>
        ${tags.join("")}
      </div>
    </summary>
    ${detail}
  </details>`;
}

function parlayCard(parlays, caveat) {
  if (!parlays?.length) return "";
  const blocks = parlays.map((p) => `
    <div>
      <div class="eyebrow">${p.legs.length} legs
        ${p.fair_odds != null ? `· breaks even at +${p.fair_odds}` : ""}</div>
      <ul class="plain">
        ${p.legs.map((l) => `<li><b>${esc(l.pick)}</b> — ${esc(l.fight)}
           <span class="num">${pct(l.prob)}</span></li>`).join("")}
      </ul>
      <div class="mini num">Combined ${pct(p.combined_prob, 1)}</div>
    </div>`).join("");
  return `<div class="card">
    <h2 style="font-size:19px">Parlays</h2>${blocks}
    <div class="note warnbox">${esc(caveat)}</div>
  </div>`;
}

const BOOT = {};
BOOT.card = async () => {
  const el = $("view-card");
  let d;
  try { d = await load("card"); }
  catch (e) { return fail(el, e, "the card"); }

  $("stamp").textContent = `data to ${d.dataset_end ?? "—"}`;
  const skipped = d.skipped?.length
    ? `<div class="note"><b>Not predicted — ${d.skipped.length} of
        ${d.fights.length + d.skipped.length} fights.</b>
        ${d.skipped.map((s) => `<br>${esc(s.fight)} — ${esc(s.reason)}`).join("")}
       </div>` : "";

  el.innerHTML = `
    <div class="event">
      <div class="eyebrow">${esc(d.event.date ?? "")}</div>
      <h1>${esc(d.event.name ?? "Fight card")}</h1>
      <div class="mini">${d.fights.length} predicted${
        d.trained_on ? ` · model trained on ${d.trained_on.toLocaleString()} fights` : ""}</div>
    </div>
    <div class="note warnbox">${esc(d.caveats.model)}</div>
    <div class="bouts">${d.fights.map(boutRow).join("")}</div>
    ${skipped}
    ${parlayCard(d.parlays, d.caveats.parlay)}
    <div class="note">${esc(d.caveats.simulation)}</div>`;
};

/* --------------------------------------------------------------- compare */
let FIGHTERS = null, NAMES = null, M = null, FORM = null;

function fighterById(id) {
  const row = FIGHTERS.index.get(id);
  if (!row) return null;
  const out = {};
  FIGHTERS.columns.forEach((c, i) => { out[c] = row[i]; });
  return out;
}

const int = (x) => (x == null ? "—" : String(Math.round(x)));
const TAPE = [
  ["Record", (s) => (s.cd_wins == null ? null
    : `${Math.round(s.cd_wins)}–${Math.round(s.cd_losses)}`), "high"],
  ["UFC bouts", (s) => s.cd_bouts, "high", int],
  ["Win rate", (s) => s.cd_win_rate, "high", pct],
  ["Height", (s) => s.height_cm, "high", (v) => (v == null ? "—" : `${v.toFixed(0)} cm`)],
  ["Reach", (s) => s.reach_cm, "high", (v) => (v == null ? "—" : `${v.toFixed(0)} cm`)],
  ["Strikes landed /min", (s) => s.cd_slpm, "high"],
  ["Striking accuracy", (s) => s.cd_str_acc, "high", pct],
  ["Strikes absorbed /min", (s) => s.cd_sapm, "low"],
  ["Striking defence", (s) => s.cd_str_def, "high", pct],
  ["Takedowns /15min", (s) => s.cd_td_per15, "high"],
  ["Takedown defence", (s) => s.cd_td_def, "high", pct],
  ["Submission att /15min", (s) => s.cd_sub_per15, "high"],
  ["Control share", (s) => s.cd_ctrl_share, "high", pct],
  ["Knockdowns /15min", (s) => s.cd_kd_per15, "high"],
];

// Every gauge is drawn on the same scale: three standard deviations of that
// advantage across the league. Without one shared scale a small edge in a
// narrow statistic would look like a large one.
const GAUGES = [
  ["Striking", "strikingAdvantage", "STRIKE_ADV_SD", (v) => `${signed(v, 2)} strikes/min`],
  ["Control", "controlAdvantage", "CTRL_ADV_SD", (v) => `${signed(v, 1)} sec/min`],
  ["Takedowns", "takedownAdvantage", "TD_ADV_SD", (v) => `${signed(v * 15, 2)} per 15min`],
  ["Submissions", "submissionAdvantage", "SUB_ADV_SD", (v) => `${signed(v * 15, 2)} per 15min`],
];

function renderCompare(redId, blueId) {
  const out = $("compare-out");
  if (!redId || !blueId) {
    out.innerHTML = `<div class="note">Pick two fighters to compare.</div>`;
    return;
  }
  if (redId === blueId) {
    out.innerHTML = `<div class="note">Those are the same fighter.</div>`;
    return;
  }
  const a = fighterById(redId), b = fighterById(blueId);
  const fa = FORM(a), fb = FORM(b);
  const C = FIGHTERS.constants;

  const tape = TAPE.map(([label, get, better, fmt]) => {
    const va = get(a), vb = get(b);
    const f = fmt || ((v) => (v == null ? "—" : typeof v === "number" ? num(v) : v));
    let aw = "", bw = "";
    if (typeof va === "number" && typeof vb === "number" && va !== vb) {
      const aBetter = better === "high" ? va > vb : va < vb;
      aw = aBetter ? "win" : ""; bw = aBetter ? "" : "win";
    }
    return `<div class="l num ${aw}">${f(va)}</div>
            <div class="lab">${label}</div>
            <div class="r2 num ${bw}">${f(vb)}</div>`;
  }).join("");

  const gauges = GAUGES.map(([label, fn, sd, fmt]) => {
    const v = M[fn](fa, fb);
    if (v == null || !Number.isFinite(v)) {
      return `<div class="adv-row"><div class="lab">${label}</div>
        <div class="gauge"><div class="mid"></div></div>
        <div class="advval">not measured</div></div>`;
    }
    const scale = C[sd] * 3;
    const frac = Math.max(-1, Math.min(1, v / scale));
    const half = Math.abs(frac) * 50;
    const side = frac >= 0
      ? `left:50%;width:${half}%;background:var(--red)`
      : `right:50%;width:${half}%;background:var(--blue)`;
    return `<div class="adv-row"><div class="lab">${label}</div>
      <div class="gauge" role="img" aria-label="${label}: ${fmt(v)}">
        <div class="mid"></div><i style="${side}"></i>
      </div>
      <div class="advval num">${fmt(v)}</div></div>`;
  }).join("");

  const size = M.sizeAdvantage(fa, fb);
  const height = M.heightAdvantage(fa, fb);
  const reach = M.reachAdvantage(fa, fb);

  out.innerHTML = `
    <div class="card">
      <div class="names" style="justify-content:space-between">
        <span class="r">${esc(NAMES[redId])}</span>
        <span class="b">${esc(NAMES[blueId])}</span>
      </div>
      <div class="tape">${tape}</div>
    </div>
    <div class="card">
      <h2 style="font-size:19px">Matchup advantage</h2>
      <div class="mini">Each fighter's offence against the other's defence,
        not a difference of two averages. Bars run
        <span style="color:var(--red)">${esc(NAMES[redId])}</span> right,
        <span style="color:var(--blue)">${esc(NAMES[blueId])}</span> left,
        on a shared scale of three league standard deviations.</div>
      <div class="adv">${gauges}</div>
      <div class="mini">Size ${Number.isFinite(size) ? `${signed(size, 2)} SD` : "not measured"}
        · height ${Number.isFinite(height) ? `${signed(height, 0)} cm` : "—"}
        · reach ${Number.isFinite(reach) ? `${signed(reach, 0)} cm` : "—"}</div>
    </div>
    <div class="note">These are the same formulas the engine uses, run here on
      each fighter's record as it stands today. They describe the matchup; they
      are not the model's win probability, which needs the full pipeline and
      appears on the Card tab.</div>`;
}

function attachPicker(inputId, listId, onPick) {
  const input = $(inputId);
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    const list = $(listId);
    if (q.length < 2) { list.innerHTML = ""; onPick(null); return; }
    const hits = NAMES.order
      .filter(([, n]) => n.toLowerCase().includes(q)).slice(0, 8);
    list.innerHTML = hits
      .map(([id, n]) => `<option value="${esc(n)}" data-id="${esc(id)}"></option>`)
      .join("");
    const exact = hits.find(([, n]) => n.toLowerCase() === q);
    onPick(exact ? exact[0] : null);
  });
}

BOOT.compare = async () => {
  const el = $("view-compare");
  el.innerHTML = `<div class="loading">Loading fighters…</div>`;
  try {
    const [f, n, c] = await Promise.all([
      load("fighters"), load("names"), load("constants")]);
    FIGHTERS = {
      columns: f.columns,
      constants: c.matchup,
      index: new Map(f.ids.map((id, i) => [id, f.values[i]])),
    };
    NAMES = n.names;
    NAMES.order = Object.entries(n.names)
      .sort((a, b) => a[1].localeCompare(b[1]));
    M = makeMatchup(c.matchup);
    FORM = makeForm(c.matchup);
  } catch (e) { return fail(el, e, "the fighter database"); }

  el.innerHTML = `
    <div>
      <div class="eyebrow">Head to head</div>
      <h1 style="font-size:26px">Compare any two fighters</h1>
      <p class="lede">${FIGHTERS.index.size.toLocaleString()} fighters, each
        described only by the bouts they had actually fought.</p>
    </div>
    <div class="card pickers">
      <div class="field r">
        <label for="pick-red">Red corner</label>
        <input id="pick-red" list="list-red" autocomplete="off"
               placeholder="Start typing a name">
        <datalist id="list-red"></datalist>
      </div>
      <div class="field b">
        <label for="pick-blue">Blue corner</label>
        <input id="pick-blue" list="list-blue" autocomplete="off"
               placeholder="Start typing a name">
        <datalist id="list-blue"></datalist>
      </div>
    </div>
    <div id="compare-out"></div>`;

  let redId = null, blueId = null;
  attachPicker("pick-red", "list-red", (id) => { redId = id; renderCompare(redId, blueId); });
  attachPicker("pick-blue", "list-blue", (id) => { blueId = id; renderCompare(redId, blueId); });
  renderCompare(null, null);
};

/* ---------------------------------------------------------------- record */
function lineChart(rows, key, opts) {
  const W = 600, H = 190, L = 34, R = 10, T = 14, B = 26;
  const xs = rows.map((r) => r.year);
  const ys = rows.map((r) => r[key]);
  const lo = Math.min(opts.floor, ...ys), hi = Math.max(...ys);
  const pad = (hi - lo) * 0.12 || 0.02;
  const y0 = lo - pad, y1 = hi + pad;
  const X = (i) => L + (i / (rows.length - 1)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const path = ys.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const ticks = [y0 + (y1 - y0) * 0.1, (y0 + y1) / 2, y1 - (y1 - y0) * 0.1];
  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="${opts.label}">
    ${ticks.map((t) => `<line x1="${L}" x2="${W - R}" y1="${Y(t).toFixed(1)}"
        y2="${Y(t).toFixed(1)}" stroke="var(--line-soft)" stroke-width="1"/>
      <text x="${L - 6}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end"
        font-size="10" font-family="IBM Plex Mono,monospace"
        fill="var(--ink-3)">${t.toFixed(2)}</text>`).join("")}
    ${opts.rule != null ? `<line x1="${L}" x2="${W - R}" y1="${Y(opts.rule).toFixed(1)}"
        y2="${Y(opts.rule).toFixed(1)}" stroke="var(--ink-3)" stroke-width="1"
        stroke-dasharray="3 3"/>` : ""}
    <path d="${path}" fill="none" stroke="var(--blue)" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>
    ${ys.map((v, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}"
        r="${i === ys.length - 1 ? 4.5 : 3}" fill="var(--blue)"
        stroke="var(--surface)" stroke-width="2"><title>${xs[i]}: ${v.toFixed(3)}</title></circle>`).join("")}
    ${xs.map((x, i) => (i % 2 === 0 || i === xs.length - 1
      ? `<text x="${X(i).toFixed(1)}" y="${H - 8}" text-anchor="middle"
          font-size="10" font-family="IBM Plex Mono,monospace"
          fill="var(--ink-3)">${String(x).slice(2)}</text>` : "")).join("")}
  </svg>`;
}

function roiChart(rows) {
  const W = 600, H = 190, L = 40, R = 10, T = 14, B = 26;
  const vals = rows.map((r) => r.roi);
  const hi = Math.max(0.02, ...vals), lo = Math.min(-0.02, ...vals);
  const X = (i) => L + (i / rows.length) * (W - L - R);
  const bw = ((W - L - R) / rows.length) - 3;
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const zero = Y(0);
  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Flat-stake return by year, model strategy">
    ${[hi, 0, lo].map((t) => `<text x="${L - 6}" y="${(Y(t) + 3.5).toFixed(1)}"
        text-anchor="end" font-size="10" font-family="IBM Plex Mono,monospace"
        fill="var(--ink-3)">${(t * 100).toFixed(0)}%</text>`).join("")}
    ${rows.map((r, i) => {
      const y = Y(r.roi), h = Math.max(1.5, Math.abs(y - zero));
      const up = r.roi >= 0;
      return `<rect x="${X(i).toFixed(1)}" y="${(up ? y : zero).toFixed(1)}"
        width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="2"
        fill="${up ? "var(--gain)" : "var(--loss)"}">
        <title>${r.label}: ${(r.roi * 100).toFixed(1)}% over ${r.bets} bets</title></rect>`;
    }).join("")}
    <line x1="${L}" x2="${W - R}" y1="${zero.toFixed(1)}" y2="${zero.toFixed(1)}"
      stroke="var(--ink-3)" stroke-width="1"/>
    ${rows.map((r, i) => (i % 2 === 0 || i === rows.length - 1
      ? `<text x="${(X(i) + bw / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle"
          font-size="10" font-family="IBM Plex Mono,monospace"
          fill="var(--ink-3)">${String(r.label).slice(2)}</text>` : "")).join("")}
  </svg>`;
}

BOOT.record = async () => {
  const el = $("view-record");
  let d;
  try { d = await load("performance"); }
  catch (e) { return fail(el, e, "the record"); }

  const t = d.track_record;
  const meta = d.strategy_meta || {};
  const strategies = (d.strategies || []).map((s) => `
    <tr><td><b>${esc(s.name)}</b><br><span class="mini">${esc(s.description)}</span></td>
      <td class="n">${s.bets.toLocaleString()}</td>
      <td class="n">${pct(s.hit_rate, 1)}</td>
      <td class="n" style="color:${s.roi >= 0 ? "var(--gain)" : "var(--loss)"}">
        ${signed(s.roi * 100, 1)}%</td></tr>`).join("");

  const auc = (d.auc_by_year || []).filter((r) => r.auc != null);
  const roi = (d.roi_by_year || []).filter((r) => r.roi != null);

  el.innerHTML = `
    <div>
      <div class="eyebrow">Measured, not claimed</div>
      <h1 style="font-size:26px">Track record</h1>
    </div>
    <div class="tiles">
      <div class="tile"><div class="k">Live accuracy</div>
        <div class="v">${pct(t.accuracy, 1)}</div>
        <div class="s">over ${t.unique_bouts ?? "—"} settled bouts</div></div>
      <div class="tile"><div class="k">Pending</div>
        <div class="v">${t.pending ?? "—"}</div>
        <div class="s">awaiting results</div></div>
      <div class="tile"><div class="k">Predictions</div>
        <div class="v">${(t.predictions ?? 0).toLocaleString()}</div>
        <div class="s">${t.graded ?? "—"} graded; a bout predicted twice
          counts once above</div></div>
      <div class="tile"><div class="k">Bouts in data</div>
        <div class="v">${(d.bouts || 0).toLocaleString()}</div>
        <div class="s">to ${esc(d.dataset_end ?? "—")}</div></div>
    </div>

    <div class="card">
      <h2 style="font-size:19px">The three strategies</h2>
      ${strategies ? `
        <div class="mini">Confirm period only — ${meta.confirm_from ?? "2020"}
          onward. Flat stake, one unit a bet.${
            meta.flag_quality != null
              ? ` The flag that drives FADE and COMBINED scores
                  ${meta.flag_quality.toFixed(3)}, where 0.5 is a coin toss.`
              : ""}</div>
        <div class="scroll"><table>
          <thead><tr><th>Strategy</th><th class="n">Bets</th><th class="n">Hit</th>
            <th class="n">Return</th></tr></thead>
          <tbody>${strategies}</tbody>
        </table></div>
        <div class="note warnbox">${esc(d.notes.which_number_counts)}</div>`
      : `<div class="note warnbox"><b>Not measured yet.</b> These numbers are
          read from a backtest the engine has to run; they are deliberately not
          carried in the app as constants, because a hand-copied result goes
          stale silently. Run the <code>historical_backtest</code> experiment
          to fill this in.</div>`}
    </div>

    ${roi.length ? `<div class="card">
      <h2 style="font-size:19px">Return by year</h2>
      <div class="mini">Model strategy, flat stakes. 2025 is absent because the
        odds source has no prices for it.</div>
      ${roiChart(roi)}
      <div class="legend">
        <span><i style="background:var(--gain)"></i>profit</span>
        <span><i style="background:var(--loss)"></i>loss</span>
      </div>
    </div>` : ""}

    ${auc.length ? `<div class="card">
      <h2 style="font-size:19px">Ranking power by year</h2>
      <div class="mini">AUC — the chance the winner is ranked above the loser.
        0.50 is a coin toss, marked by the dashed line.</div>
      ${lineChart(auc, "auc", { floor: 0.5, rule: 0.5, label: "Model AUC by year" })}
    </div>` : ""}

    <div class="note">${esc(d.notes.leak)}</div>`;
};

/* ---------------------------------------------------------------- method */
BOOT.method = () => {
  $("view-method").innerHTML = `
    <div>
      <div class="eyebrow">What this is</div>
      <h1 style="font-size:26px">How it works, and what it is worth</h1>
    </div>
    <div class="card">
      <h2 style="font-size:18px">Two layers</h2>
      <p class="lede">A statistical model ranks who wins — that is the
        percentage on the Card tab. A Monte Carlo simulation then runs each
        matchup 20,000 times from both fighters' measured rates, which is where
        the method, round and duration come from.</p>
      <p class="lede">The simulation is the weaker winner-picker of the two
        (AUC 0.559 against the model's 0.66), so it is shown beside the model
        and never instead of it. When they disagree they are using different
        information, and that is worth seeing rather than averaging away.</p>
    </div>
    <div class="card">
      <h2 style="font-size:18px">Every number is point-in-time</h2>
      <p class="lede">A fighter is described only by bouts that had already
        happened when the fight took place. That sounds obvious and it was not
        true until recently: the dataset's <code>wins</code> column is a
        fighter's <em>lifetime</em> record as of the day the data was
        downloaded, pasted onto every fight of their career. A model reading it
        could see how each career turned out.</p>
      <p class="lede">It inflated historical accuracy to 78% and a backtest to
        +16.2%. Both were fiction. The honest figures are the ones in this app.</p>
    </div>
    <div class="card">
      <h2 style="font-size:18px">What it does not do</h2>
      <ul class="plain">
        <li>It does not beat the market. The best strategy returned +0.3% over
          2,339 bets since 2020, and the confidence interval spans zero.</li>
        <li>It refuses fights it cannot describe rather than guessing. A name
          it does not recognise gets "no data", not a coin flip.</li>
        <li>A blank where a number should be means it was never measured — not
          that it is zero.</li>
      </ul>
    </div>
    <div class="note">Built from UFC results. Nothing here is betting advice,
      and the measured edge is not distinguishable from zero.</div>`;
};

/* ------------------------------------------------------------------ boot */
let initial = "card";
try { initial = localStorage.getItem("yourmma.tab") || "card"; } catch (e) { /* ignore */ }
if (!TABS.includes(initial)) initial = "card";
show(initial);
