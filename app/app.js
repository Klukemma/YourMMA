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

// desk.js:begin - copied from shared/desk.js by scripts/sync_desk.py; edit that file
// The Desk app chrome, shared by YourMMA and YourNFL: install, the app menu,
// theme, the update notice and the icons. This file is the source;
// scripts/sync_desk.py copies it into both apps' app.js, and a test fails if
// either copy drifts.
//
// No module syntax and one name (DESK): the MMA app runs as a module, the
// NFL app and both standalone files as classic scripts, and the MMA
// standalone concatenates this with matchup.js.
const DESK = (() => {
  const ICONS = {
    list: '<path d="M8 6h12M8 12h12M8 18h12"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>',
    compare: '<path d="M7 4v16M17 4v16"/><path d="M3 8h8M13 16h8"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    calendar: '<rect x="3" y="5" width="18" height="16" rx="3"/><path d="M3 10h18M8 3v4M16 3v4"/>',
    shield: '<path d="M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z"/>',
    download: '<path d="M12 4v11M7 10l5 5 5-5"/><path d="M5 20h14"/>',
    more: '<circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    chevron: '<path d="M6 9l6 6 6-6"/>',
    share: '<path d="M12 3v12M8 7l4-4 4 4"/><path d="M6 11H5a1 1 0 00-1 1v8a1 1 0 001 1h14a1 1 0 001-1v-8a1 1 0 00-1-1h-1"/>',
    plus: '<rect x="4" y="4" width="16" height="16" rx="4"/><path d="M12 8v8M8 12h8"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z"/>',
    auto: '<circle cx="12" cy="12" r="9"/><path d="M12 3v18" /><path d="M12 3a9 9 0 010 18z" fill="currentColor"/>',
    refresh: '<path d="M20 11a8 8 0 10-2.3 5.7"/><path d="M20 4v7h-7"/>',
    swap: '<path d="M7 4L3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2h2"/>',
    close: '<path d="M6 6l12 12M18 6L6 18"/>',
    alert: '<path d="M12 3l9.5 17h-19z"/><path d="M12 10v4.5M12 17.5v.3"/>',
  };
  const icon = (name, extra = "") =>
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
      stroke-linejoin="round" aria-hidden="true" ${extra}>${ICONS[name] || ""}</svg>`;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };

  // How long ago, for a stamp: "just now", "3h ago", "2d ago".
  function ago(iso) {
    const t = new Date(iso).getTime();
    if (!Number.isFinite(t)) return "";
    const m = (Date.now() - t) / 6e4;
    if (m < 0) return "just now";
    if (m < 2) return "just now";
    if (m < 60) return `${Math.round(m)} min ago`;
    if (m < 60 * 24) return `${Math.round(m / 60)}h ago`;
    return `${Math.round(m / 1440)}d ago`;
  }

  let cfg = null;

  /* ------------------------------------------------------------ theme */
  const THEME_BG = { light: "#f3f4f7", dark: "#0b0c10" };
  function applyTheme(mode) {
    const root = document.documentElement;
    if (mode === "light" || mode === "dark") root.setAttribute("data-theme", mode);
    else root.removeAttribute("data-theme");
    // The status bar follows the chosen theme, not only the system's.
    for (const m of document.querySelectorAll('meta[name="theme-color"]')) {
      const scheme = /dark/.test(m.getAttribute("media") || "") ? "dark" : "light";
      m.setAttribute("content", THEME_BG[mode === "light" || mode === "dark" ? mode : scheme]);
    }
  }
  const theme = () => store.get(`${cfg.key}.theme`) || "auto";

  /* ------------------------------------------------------------ sheet */
  let openSheet = null;
  function sheet(html) {
    closeSheet();
    const back = document.createElement("div");
    back.className = "sheet-back";
    back.innerHTML = `<div class="sheet" role="dialog" aria-modal="true"><div class="grab"></div>${html}</div>`;
    back.addEventListener("click", (e) => { if (e.target === back || e.target.closest("[data-close]")) closeSheet(); });
    document.body.appendChild(back);
    const prev = document.activeElement;
    openSheet = { back, prev };
    const first = back.querySelector("button,a,[tabindex]");
    if (first) first.focus({ preventScroll: true });
    return back.querySelector(".sheet");
  }
  function closeSheet() {
    if (!openSheet) return;
    const { back, prev } = openSheet;
    openSheet = null;
    back.remove();
    if (prev && prev.focus) prev.focus({ preventScroll: true });
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });

  /* ------------------------------------------------------------ toast */
  let toastTimer = null;
  function toast(text, action, onAction, ms = 5000) {
    document.querySelectorAll(".toast").forEach((t) => t.remove());
    const t = document.createElement("div");
    t.className = "toast";
    t.setAttribute("role", "status");
    t.innerHTML = `<span>${esc(text)}</span>${action ? `<button type="button">${esc(action)}</button>` : ""}`;
    if (action) t.querySelector("button").addEventListener("click", () => { t.remove(); onAction && onAction(); });
    document.body.appendChild(t);
    clearTimeout(toastTimer);
    if (ms) toastTimer = setTimeout(() => t.remove(), ms);
  }

  /* ---------------------------------------------------------- install */
  // A phone installs the app from the browser's own menu. Chrome tells the
  // page it can (beforeinstallprompt) and the page can then offer a button;
  // Safari never does, so on an iPhone the button opens the three steps
  // instead. Inside another app's browser (a chat or social app) neither
  // works, and the honest answer is to open the page in the real browser.
  let deferred = null;
  const ua = navigator.userAgent || "";
  const isIOS = /iphone|ipad|ipod/i.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const isAndroid = /android/i.test(ua);
  const inAppBrowser = /FBAN|FBAV|Instagram|Line\/|Snapchat|Twitter|TikTok|GSA\/|; wv\)|WhatsApp|Telegram|Discord/i.test(ua);
  const standalone = () =>
    window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
  const served = () => location.protocol.startsWith("http");
  const canOfferInstall = () => served() && !standalone();

  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e;
    refreshInstallUI();
  });
  window.addEventListener("appinstalled", () => {
    deferred = null;
    refreshInstallUI();
    toast(`${cfg ? cfg.name : "The app"} is installed. Open it from your home screen.`);
  });

  async function install() {
    if (deferred) {
      const e = deferred;
      deferred = null;
      e.prompt();
      try { await e.userChoice; } catch (err) { /* dismissed */ }
      refreshInstallUI();
      return;
    }
    installHelp();
  }

  function copyLink() {
    const url = location.href.split("#")[0];
    const done = () => toast("Link copied. Paste it into Safari or Chrome.");
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(done, () => toast(url, null, null, 9000));
    else toast(url, null, null, 9000);
  }

  function installHelp() {
    const name = esc(cfg.name);
    let body;
    if (inAppBrowser) {
      body = `<h2>Open it in your browser</h2>
        <p class="mini">This page is open inside another app, which cannot install apps.
          Open it in ${isIOS ? "Safari" : "Chrome"}, then install from there.</p>
        <ol class="steps">
          <li><span>Tap the ${icon("more", 'class="inl"')} or ${icon("share", 'class="inl"')} menu of this screen</span></li>
          <li><span>Choose <b>Open in ${isIOS ? "Safari" : "browser"}</b></span></li>
          <li><span>Tap <b>Install</b> in ${name}'s menu</span></li>
        </ol>
        <div class="actions"><button class="btn ghost" type="button" data-copy>${icon("copy")} Copy link</button>
          <button class="btn" type="button" data-close>Done</button></div>`;
    } else if (isIOS) {
      body = `<h2>Install ${name}</h2>
        <p class="mini">It opens full screen from your home screen and works with no signal.</p>
        <ol class="steps">
          <li><span>Tap ${icon("share", 'class="inl"')} <b>Share</b> in Safari's toolbar</span></li>
          <li><span>Scroll down and tap ${icon("plus", 'class="inl"')} <b>Add to Home Screen</b></span></li>
          <li><span>Tap <b>Add</b></span></li>
        </ol>
        <p class="mini">Opened from another app's link? Tap ${icon("globe", 'class="inl"')} to open the page in
          Safari first - only Safari itself can add it. In Chrome on iPhone the Share button is in the
          address bar. Already installed? Open ${name} from your home screen.</p>
        <div class="actions"><button class="btn" type="button" data-close>Got it</button></div>`;
    } else {
      body = `<h2>Install ${name}</h2>
        <p class="mini">It opens full screen from your home screen and works with no signal.</p>
        <ol class="steps">
          <li><span>Open the browser menu ${icon("more", 'class="inl" style="transform:rotate(90deg)"')}</span></li>
          <li><span>Tap <b>Install app</b> or <b>Add to Home screen</b></span></li>
          <li><span>Tap <b>Install</b></span></li>
        </ol>
        <p class="mini">${isAndroid ? "Use Chrome, Edge or Samsung Internet." : "In Chrome or Edge on a computer, the install button is in the address bar."}
          If the menu has no install option, the app may already be installed.</p>
        <div class="actions"><button class="btn" type="button" data-close>Got it</button></div>`;
    }
    const s = sheet(body);
    const c = s.querySelector("[data-copy]");
    if (c) c.addEventListener("click", copyLink);
  }

  // The banner a tab can show above its content, until installed or waved off.
  function banner() {
    if (!canOfferInstall() || store.get(`${cfg.key}.install-dismissed`)) return "";
    return `<div class="banner" data-install-banner>
      <img src="${esc(cfg.icon)}" alt="">
      <div class="txt"><b>Install ${esc(cfg.name)}</b><span>Full screen, works offline</span></div>
      <button class="btn small" type="button" data-install>${icon("download")} Install</button>
      <button class="iconbtn" type="button" data-dismiss aria-label="Not now">${icon("close")}</button>
    </div>`;
  }
  function refreshInstallUI() {
    document.querySelectorAll("[data-install-banner]").forEach((b) => {
      if (!canOfferInstall()) b.remove();
    });
  }
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-install]")) { e.preventDefault(); install(); }
    const d = e.target.closest("[data-dismiss]");
    if (d) {
      store.set(`${cfg.key}.install-dismissed`, "1");
      const b = d.closest("[data-install-banner]");
      if (b) b.remove();
      toast("You can install it any time from the menu.");
    }
  });

  /* ------------------------------------------------------------- menu */
  function menu() {
    const t = theme();
    const opt = (mode, label, ic) =>
      `<button type="button" data-theme-set="${mode}" aria-pressed="${t === mode}">${icon(ic, 'style="width:16px;height:16px;display:inline;vertical-align:-3px;margin-right:5px"')}${label}</button>`;
    const rows = [];
    if (canOfferInstall()) rows.push(`<button class="row" type="button" data-install style="width:100%;border:0;border-bottom:1px solid var(--line-soft);background:none;text-align:left;cursor:pointer">
      <span class="ic">${icon("download")}</span><span class="t">Install ${esc(cfg.name)}<small>Home screen, full screen, offline</small></span></button>`);
    if (cfg.sibling && served()) rows.push(`<a class="row" href="${esc(cfg.sibling.href)}" style="text-decoration:none;color:inherit">
      <span class="ic">${icon("swap")}</span><span class="t">Open ${esc(cfg.sibling.name)}<small>${esc(cfg.sibling.blurb || "")}</small></span></a>`);
    rows.push(`<button class="row" type="button" data-reload style="width:100%;border:0;border-bottom:1px solid var(--line-soft);background:none;text-align:left;cursor:pointer">
      <span class="ic">${icon("refresh")}</span><span class="t">Refresh<small>${esc(cfg.stamp ? cfg.stamp() : "Fetch the latest picks")}</small></span></button>`);
    const s = sheet(`<h2>${esc(cfg.name)}</h2>
      <div class="eyebrow" style="margin:14px 0 8px">Appearance</div>
      <div class="seg" role="group" aria-label="Theme">${opt("auto", "System", "auto")}${opt("light", "Light", "sun")}${opt("dark", "Dark", "moon")}</div>
      <div style="margin-top:8px">${rows.join("")}</div>
      <p class="mini" style="margin-top:10px">For entertainment and analysis. Not betting advice.</p>`);
    s.addEventListener("click", (e) => {
      const b = e.target.closest("[data-theme-set]");
      if (b) {
        const mode = b.getAttribute("data-theme-set");
        store.set(`${cfg.key}.theme`, mode);
        applyTheme(mode);
        s.querySelectorAll("[data-theme-set]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      }
      if (e.target.closest("[data-reload]")) { closeSheet(); reload(); }
    });
  }

  function reload() {
    if (cfg.onReload) cfg.onReload();
    else location.reload();
  }

  /* --------------------------------------------------- service worker */
  // The worker is what makes the app installable and what makes it open with
  // no signal. It is skipped on file:// because a file:// page has no origin
  // to scope one to - the standalone build carries its data on the page.
  // When a new version takes over, say so and offer the reload, rather than
  // leaving the old screen up until the next launch.
  function serviceWorker() {
    if (!("serviceWorker" in navigator) || !location.protocol.startsWith("http")) return;
    const hadController = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (hadController) toast("A new version is ready.", "Reload", () => location.reload(), 0);
    });
    const go = () => navigator.serviceWorker.register("sw.js").catch(() => {
      // No worker means no offline copy. The app still works online, so this
      // is not worth a message on the page.
    });
    if (document.readyState === "complete") go();
    else window.addEventListener("load", go);
  }

  /* ------------------------------------------------------------- init */
  // cfg: { key, name, icon, sibling: {name, href, blurb}, stamp(), onReload() }
  function init(options) {
    cfg = options;
    applyTheme(theme());
    const m = document.getElementById("menu-btn");
    if (m) m.addEventListener("click", menu);
    // The sport switch links to the other app on the same site; a saved
    // file has no neighbour to link to.
    const sw = document.getElementById("sport-switch");
    if (sw && !served()) sw.hidden = true;
    if (!served()) document.querySelectorAll("[data-served-only]").forEach((x) => { x.hidden = true; });
    serviceWorker();
  }

  // Wire a bottom tab bar: tabs are <button id="tab-NAME">, views
  // <section id="view-NAME">. boot[NAME]() runs the first time a tab opens.
  function tabs(names, boot, storageKey, fallback) {
    const started = {};
    function show(name, scroll) {
      for (const t of names) {
        document.getElementById(`view-${t}`).hidden = t !== name;
        document.getElementById(`tab-${t}`).setAttribute("aria-selected", String(t === name));
      }
      store.set(storageKey, name);
      if (scroll) window.scrollTo({ top: 0 });
      if (!started[name]) { started[name] = true; boot[name](); }
    }
    for (const t of names) {
      document.getElementById(`tab-${t}`).addEventListener("click", () => {
        const already = document.getElementById(`tab-${t}`).getAttribute("aria-selected") === "true";
        // Tapping the open tab again goes back to its top, as native apps do.
        if (already) window.scrollTo({ top: 0, behavior: "smooth" });
        else show(t, true);
      });
    }
    // A home-screen shortcut opens a tab directly (manifest "shortcuts").
    let first = new URLSearchParams(location.search).get("tab") || store.get(storageKey);
    if (!names.includes(first)) first = fallback;
    show(first, false);
    return { show, restart(name) { started[name] = false; } };
  }

  const skeleton = (n = 4) =>
    `<div class="skeleton" aria-label="Loading"><i class="short"></i>${"<i></i>".repeat(n)}</div>`;

  return { icon, esc, ago, sheet, closeSheet, toast, install, banner, init, tabs, skeleton, store };
})();
// desk.js:end

const $ = (id) => document.getElementById(id);
const pct = (x, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const signed = (x, d = 1) =>
  x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
const num = (x, d = 2) => (x == null ? "—" : x.toFixed(d));
// Round the whole duration first: rounding only the seconds printed 599.6 s
// as "9:60".
const clock = (s) => {
  if (s == null) return "—";
  const t = Math.round(s);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const odds = (o) => (o == null ? "—" : o > 0 ? `+${o}` : `${o}`);
const icon = DESK.icon;
// A card date ("2026-10-24") as the poster prints it. Read as UTC so a phone
// west of Greenwich does not move the fight to the day before.
const longDate = (d) => {
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? (d || "") : t.toLocaleDateString(undefined,
    { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
};
const shortDate = (d) => {
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? (d || "") : t.toLocaleDateString(undefined, { day: "numeric", month: "short", timeZone: "UTC" });
};
const surname = (n) => {
  const parts = String(n || "").split(" ");
  const last = parts[parts.length - 1];
  return /^(jr\.?|sr\.?|ii|iii|iv)$/i.test(last) && parts.length > 2 ? parts.slice(-2).join(" ") : last;
};

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

// A caveat folded to its first two lines, the rest one tap away.
const fold = (text, extra = "") => !text ? "" : `<details class="fold">
    <summary><span class="ic">${icon("info")}</span><span class="clip">${esc(text)}</span><span class="more">More</span></summary>
    ${extra ? `<div class="fold-body">${extra}</div>` : ""}</details>`;

/* ------------------------------------------------------------------ card */
const TAG_TONE = {
  "STRONG BET": "good", "GOOD BET": "good", LEAN: "flat",
  RISKY: "warn", AVOID: "warn", LOCK: "good", STRONG: "good", VALUE: "warn",
};

/* --------------------------------------------------- the second layer */
// A separate model, fitted only on bets that had already settled, guessing
// whether each pick is wrong. Above DOUBT_FLAG it is saying the pick is more
// likely wrong than right; below DOUBT_CLEAR it is as close to a vote of
// confidence as this project is willing to print.
const DOUBT_FLAG = 0.5;
const DOUBT_CLEAR = 0.4;
const doubtTone = (p) => (p >= DOUBT_FLAG ? "warn" : p <= DOUBT_CLEAR ? "good" : "flat");

function doubtCard(fights, caveat) {
  const scored = fights.filter((f) => f.p_fail != null);
  const unscored = fights.length - scored.length;
  if (!scored.length) {
    return `<div class="note"><b>No second-layer check on this card.</b>
      Most of what it reads is the gap between this model and the market, so
      it cannot run on fights nobody has priced.</div>`;
  }
  const ranked = [...scored].sort((a, b) => b.p_fail - a.p_fail);
  const flagged = ranked.filter((f) => f.p_fail >= DOUBT_FLAG);

  const row = (f) => `<li>
      <span><b>${esc(f.pick)}</b> <span class="mini">over ${esc(f.pick === f.red ? f.blue : f.red)}</span></span>
      <span class="num ${doubtTone(f.p_fail)}">${Math.round(f.p_fail * 100)}%</span>
      <span class="meter" aria-hidden="true"><i style="width:${Math.round(f.p_fail * 100)}%"></i></span>
    </li>`;

  return `<div class="card">
    <h2>Most likely to be wrong</h2>
    <div class="mini">A second model's chance that each pick loses, most doubtful first.</div>
    <ul class="rank">${ranked.slice(0, 5).map(row).join("")}</ul>
    ${ranked.length > 5 ? `<details class="fold"><summary><span class="clip">The other ${ranked.length - 5}</span><span class="more">Show</span></summary>
      <div class="fold-body" style="padding-left:14px"><ul class="rank">${ranked.slice(5).map(row).join("")}</ul></div></details>` : ""}
    <div class="mini">${flagged.length
      ? `<b>${flagged.length} flagged</b> — rated more likely wrong than right.`
      : `<b>Nothing flagged.</b> No pick on this card is rated more likely wrong than right.`}${
      unscored ? ` ${unscored} not scored — no price to read.` : ""}</div>
    ${caveat ? fold(caveat) : ""}
  </div>`;
}

// A labelled slice of a KO/SUB/DEC bar. Below 7% there is no room for the
// label and printing it anyway overflows the slice into its neighbour.
const segment = (cls, v, label) =>
  !v ? "" : v > 0.07
    ? `<i class="${cls}" style="flex:${v}">${label} ${Math.round(v * 100)}</i>`
    : `<i class="${cls}" style="flex:${v}"></i>`;

// Fight-week news the scout gathered before the bell (engine/fight_intel.py).
// SHOWN, NEVER APPLIED: every soft factor the engine has measured - short
// notice, a recent withdrawal, a known injury, a missed weight - is inside
// the noise, so none of it moves the number above it. It is here so a reader
// knows what the model cannot see.
const KIND_LABEL = {
  injury: "Injury", short_notice: "Short notice", hard_weight_cut: "Hard weight cut",
  missed_weight: "Missed weight", new_camp: "New camp", layoff_return: "Back from a layoff",
  personal: "Personal", opponent_switch: "Opponent changed", motivation: "Motivation",
  confidence: "Confidence", other: "Note",
};
const sourceHost = (u) => {
  try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return "source"; }
};
function newsBlock(news) {
  if (!news.length) return "";
  const items = news.map((o) => {
    const safe = /^https?:\/\//.test(o.source || "");
    const link = safe
      ? ` <a href="${esc(o.source)}" target="_blank" rel="noopener noreferrer">${esc(sourceHost(o.source))}</a>`
      : "";
    return `<li><span><b>${esc(o.fighter)}</b> · ${esc(KIND_LABEL[o.kind] || o.kind)}${
      o.confidence && o.confidence !== "reported" ? ` (${esc(o.confidence)})` : ""}${
      o.replaced_opponent ? ` · originally ${esc(o.replaced_opponent)}` : ""}
      <div class="mini">${esc(o.note)}${link}</div></span></li>`;
  }).join("");
  return `<div class="panel news">
      <div class="eyebrow">Fight-week news · shown, not applied</div>
      <ul class="list">${items}</ul>
    </div>`;
}

// A fighter's pro record and last five results beside the name, newest on
// the left: a green circle is a win, red a loss, grey a draw or no contest,
// and the two letters say how it ended - K1 knockout in round 1, S2
// submission in round 2, DC decision, DQ, NC. From Sherdog (the whole pro
// career), or the UFC archive alone when Sherdog has nothing (marked UFC).
// A record matched by name only - a debutant with no UFC bout to check it
// against - is marked with an asterisk. Not a model input.
const RESULT_WORD = { W: "Win", L: "Loss", D: "Draw", NC: "No contest" };
const FORM_LEGEND = `Under each name: pro record, then the last five fights, newest
  first - green a win, red (ringed) a loss, grey a draw or no contest. K1 is a
  knockout in round 1, S2 a submission in round 2, DC a decision, DQ a
  disqualification, NC a no contest. * means the Sherdog record was matched by
  name only, so it could be a namesake's; UFC means UFC bouts only. Open a fight
  for each opponent and date. Not a model input.`;
const fightTip = (b) => `${RESULT_WORD[b.result] || b.result} · ${b.method || b.code}${
  b.opponent ? ` · vs ${b.opponent}` : ""}${b.date ? ` · ${b.date}` : ""}${b.event ? ` · ${b.event}` : ""}`;
function formLine(form) {
  // An absent key means the card predates records: show nothing. An explicit
  // null means none was found: say so.
  if (form === undefined) return "";
  if (!form || !form.record) return `<span class="rec none">no record found</span>`;
  const r = form.record;
  const text = `${r.wins}-${r.losses}${r.draws ? `-${r.draws}` : ""}${r.nc ? ` (${r.nc} NC)` : ""}`;
  const marks = `${form.source === "ufc" ? ` <i class="src" title="UFC bouts only - no pro record found">UFC</i>` : ""}${
    form.verified === false ? `<i class="src" title="Sherdog record matched by name only">*</i>` : ""}`;
  const dots = (form.last5 || []).map((b) =>
    `<span class="res ${esc(String(b.result).toLowerCase())}" role="img"
       title="${esc(fightTip(b))}" aria-label="${esc(fightTip(b))}">${esc(b.code)}</span>`).join("");
  return `<span class="rec num">${esc(text)}${marks}</span><span class="last5">${dots}</span>`;
}
// The same five fights as text, in the opened bout: a phone shows no
// tooltips, so this is where the opponent, date and event can be read.
function formDetail(name, form) {
  if (!form || !form.last5 || !form.last5.length) return "";
  return `<div class="panel"><div class="eyebrow">${esc(name)} · last ${form.last5.length}</div>
    <ul class="list">${form.last5.map((b) =>
      `<li><span class="res ${esc(String(b.result).toLowerCase())}" aria-hidden="true">${esc(b.code)}</span>
        <span>${esc(RESULT_WORD[b.result] || b.result)} · ${esc(b.method || b.code)}${
          b.opponent ? ` · vs ${esc(b.opponent)}` : ""}<span class="mini"><br>${esc(b.date || "")}${
          b.event ? ` · ${esc(b.event)}` : ""}</span></span></li>`).join("")}</ul></div>`;
}

// One corner: its colour, the name over the record and last five, and that
// corner's own chance on the right.
function side(corner, name, record, share, picked) {
  return `<div class="side ${corner}${picked ? " picked" : ""}">
      <span class="mark"></span>
      <div class="who"><span class="nm">${esc(name)}</span></div>
      <span class="p">${share == null ? "—" : `${Math.round(share * 100)}<small>%</small>`}</span>
      <div class="sub">${record}</div>
    </div>`;
}

function boutRow(f, i) {
  const p = f.win_prob;
  const news = Array.isArray(f.intel) ? f.intel : [];
  const redPicked = f.pick === f.red;
  // The bar is the corners, which is what the sport actually calls them.
  const redShare = p == null ? null : (redPicked ? p : 1 - p);
  const sim = f.simulation;

  const bar = redShare == null
    ? `<div class="probbar none"><i style="flex:1"></i></div>`
    : `<div class="probbar" role="img" aria-label="${esc(f.red)} ${pct(redShare)}, ${esc(f.blue)} ${pct(1 - redShare)}">
         <i class="r" style="flex:${redShare}"></i>
         <i class="b" style="flex:${1 - redShare}"></i>
       </div>`;

  const badges = [];
  // By the bout's place on the card, not its place in this list: a refused
  // main event would otherwise hand its badge to the next fight.
  const place = f.number ?? i + 1;
  if (place === 1) badges.push(`<span class="badge accent">Main event</span>`);
  else if (place === 2) badges.push(`<span class="badge">Co-main</span>`);
  if (f.title_fight) badges.push(`<span class="badge gold">Title</span>`);
  if (f.rounds_scheduled === 5) badges.push(`<span class="badge">5 rounds</span>`);

  const tags = [];
  if (f.recommendation)
    tags.push(`<span class="tag ${TAG_TONE[f.recommendation] || "flat"}">${esc(f.recommendation)}</span>`);
  if (f.parlay_tier)
    tags.push(`<span class="tag ${TAG_TONE[f.parlay_tier] || "flat"}">Parlay ${esc(f.parlay_tier)}</span>`);
  // The call worth making, rather than the three-way word that reads
  // "Decision" on fights the model thinks are more likely to end early.
  {
    const mpTag = f.method_probs;
    const fin = f.p_finish != null ? f.p_finish
      : mpTag ? mpTag["KO/TKO"] + mpTag.Submission : null;
    if (fin != null)
      tags.push(`<span class="tag flat num">${
        fin >= 0.5 ? `Finish ${Math.round(fin * 100)}` :
                     `Distance ${Math.round((1 - fin) * 100)}`}%</span>`);
  }
  if (f.odds != null)
    tags.push(`<span class="tag flat num" title="The pick's price">${odds(f.odds)}</span>`);
  if (news.length)
    tags.push(`<span class="tag warn" title="Fight-week news - shown, not applied">News ${news.length}</span>`);
  // The second layer. Shown on every scored pick, not only the flagged ones:
  // "the model thinks it is right about this" is information too. A pick it
  // could not score gets no tag at all rather than a reassuring one.
  if (f.p_fail != null)
    tags.push(`<span class="tag ${doubtTone(f.p_fail)} num"
      title="A second model's estimate that this pick is wrong">${Math.round(f.p_fail * 100)}% wrong</span>`);

  // The model's own method view, as a distribution. Its top class alone is
  // right 47.2% of the time where always saying "Decision" is right 49.4%,
  // so the single word is the least defensible thing it produces; the shape
  // is what it is worth. Shown beside the simulator's, because the two are
  // different machinery on the same fight and where they disagree that is
  // worth seeing rather than hiding behind whichever one the summary picked.
  const mp = f.method_probs;
  // FINISH OR DISTANCE FIRST, then how the finish arrives.
  //
  // The three-way top class is right 47.2% of the time where saying
  // "Decision" every time is right 49.4%, and it lands on Decision 54% of the
  // time against a real 49.4% - not because the model leans that way (it puts
  // 66% of its mass on finishes) but because a decision is one bucket and a
  // finish is two. KO .27 / SUB .24 / DEC .49 is more likely than not to end
  // early and prints "Decision".
  //
  // As a binary the same numbers are 57.7% accurate, AUC 0.606. So the binary
  // leads and the KO/SUB split is reported underneath it, as a split of the
  // finish rather than as a third competitor to it.
  const pFin = f.p_finish != null ? f.p_finish
    : mp ? mp["KO/TKO"] + mp.Submission : null;
  const koShare = mp && (mp["KO/TKO"] + mp.Submission) > 0
    ? mp["KO/TKO"] / (mp["KO/TKO"] + mp.Submission) : null;

  const methodModel = pFin == null ? "" : `
    <div class="panel">
      <div class="eyebrow">Model: finish or distance</div>
      <div class="method" role="img"
           aria-label="finish ${pct(pFin)}, goes to decision ${pct(1 - pFin)}">
        ${segment("ko", pFin, "FINISH")}${segment("dc", 1 - pFin, "DISTANCE")}
      </div>
      <div class="mini">
        <b>${pFin >= 0.5 ? "Finish" : "Goes the distance"} ${pct(Math.max(pFin, 1 - pFin))}</b>${
          koShare == null ? "" : ` · if it ends early, KO ${pct(koShare)} / submission ${pct(1 - koShare)}`}
      </div>
      ${f.p_finish_raw != null && Math.abs(f.p_finish_raw - pFin) > 0.02
        ? `<div class="mini">Calibrated from ${pct(f.p_finish_raw)}: the model
             ranks fights well and prices them high, and this is the
             correction measured on fights it had not seen.</div>` : ""}
    </div>`;

  // Where the pick comes from: the model alone, the market, and the blend
  // the card shows. Only what the engine wrote.
  const other = redPicked ? f.blue : f.red;
  const pickPanel = `<div class="panel">
      <div class="eyebrow">The pick</div>
      <dl class="kv">
        <dt>Pick</dt><dd><b>${esc(f.pick ?? "—")}</b> <span class="num">${pct(p, 1)}</span> <span class="mini">over ${esc(other)}</span></dd>
        ${f.win_prob_model != null ? `<dt>Model alone</dt><dd><span class="num">${pct(f.win_prob_model, 1)}</span>${
          f.win_prob_model < 0.5 ? ` <span class="mini">the model alone would pick ${esc(other)}; the market carries the pick</span>` : ""}</dd>` : ""}
        <dt>Market</dt><dd>${f.market_prob == null ? `<span class="mini">no price</span>`
          : `<span class="num">${pct(f.market_prob, 1)}</span> <span class="mini">no-vig, at ${esc(odds(f.odds))}</span>`}</dd>
        ${f.p_fail != null ? `<dt>Doubt</dt><dd><span class="num">${pct(f.p_fail)}</span> <span class="mini">second layer's chance it loses</span></dd>` : ""}
      </dl>
    </div>`;

  let detail = "";
  if (sim) {
    const ko = sim.ko ?? 0, sb = sim.sub ?? 0, dc = sim.decision ?? 0;
    const peak = Math.max(...(sim.finish_by_round || [0]), 0.0001);
    const rounds = (sim.finish_by_round || [])
      .map((v, i) => `<div><b style="height:${Math.max(3, (v / peak) * 36)}px"></b>
         <span>R${i + 1}</span></div>`).join("");
    const assumed = (sim.assumed || []).length
      ? `<div class="mini">Assumed from the league average: ${esc(sim.assumed.join(", "))}.
         Those rates were not measured for these fighters.</div>` : "";
    detail = `
      <div class="detail">
        ${newsBlock(news)}
        ${pickPanel}
        ${methodModel}
        <div class="panel">
          <div class="eyebrow">Simulated ${(sim.simulations || 0).toLocaleString()} times</div>
          <div class="stat3">
            <div><div class="k">${esc(surname(f.red))}</div><div class="v">${pct(sim.red_win, 1)}</div></div>
            <div><div class="k">${esc(surname(f.blue))}</div><div class="v">${pct(sim.blue_win, 1)}</div></div>
            <div><div class="k">Lasts</div><div class="v">${clock(sim.mean_seconds)}</div></div>
          </div>
          <div class="method" role="img"
               aria-label="KO ${pct(ko)}, submission ${pct(sb)}, decision ${pct(dc)}">
            ${segment("ko", ko, "KO")}${segment("sb", sb, "SUB")}${segment("dc", dc, "DEC")}
          </div>
          <div class="mini">KO ${pct(ko)} · Submission ${pct(sb)} · Decision ${pct(dc)} · Draw ${pct(sim.draw, 1)}</div>
          ${rounds ? `<div class="eyebrow" style="margin-top:4px">Finish by round</div>
            <div class="rounds">${rounds}</div>` : ""}
          ${assumed}
        </div>
        <div class="two">${formDetail(f.red, f.red_form)}${formDetail(f.blue, f.blue_form)}</div>
      </div>`;
  } else {
    detail = `<div class="detail">${newsBlock(news)}${pickPanel}${methodModel}
      <div class="note">No simulation: too much of these fighters' record is unmeasured, so the
      only honest answer is nothing.</div>
      <div class="two">${formDetail(f.red, f.red_form)}${formDetail(f.blue, f.blue_form)}</div></div>`;
  }

  return `<details class="match">
    <summary>
      <div class="match-top">${badges.join("")}<span class="chev">${icon("chevron")}</span></div>
      <div class="sides">
        ${side("r", f.red, formLine(f.red_form), redShare, redPicked && f.pick != null)}
        ${side("b", f.blue, formLine(f.blue_form), redShare == null ? null : 1 - redShare, !redPicked && f.pick != null)}
      </div>
      ${bar}
      <div class="pickline">
        <span class="pick">${icon("check")}${esc(surname(f.pick) || "No pick")}</span>
        ${tags.join("")}
      </div>
    </summary>
    ${detail}
  </details>`;
}

function parlayCard(parlays, caveat) {
  if (!parlays?.length) return "";
  const blocks = parlays.map((p) => `
    <div class="panel">
      <div class="eyebrow">${p.legs.length} legs
        ${p.fair_odds != null ? `· breaks even at ${odds(p.fair_odds)}` : ""}</div>
      <ul class="list">
        ${p.legs.map((l) => `<li><span><b>${esc(l.pick)}</b> <span class="mini">${esc(l.fight)}</span></span>
           <span class="num end">${pct(l.prob)}</span></li>`).join("")}
      </ul>
      <div class="mini num">Combined ${pct(p.combined_prob, 1)}</div>
    </div>`).join("");
  return `<div class="card">
    <h2>Parlays</h2>${blocks}
    ${fold(caveat)}
  </div>`;
}

const BOOT = {};
let CARD = null;
BOOT.card = async () => {
  const el = $("view-card");
  el.innerHTML = DESK.skeleton(4);
  let d;
  try { d = await load("card"); }
  catch (e) { return fail(el, e, "the card"); }
  CARD = d;
  stamp();

  const priced = d.fights.filter((f) => f.odds != null).length;
  // A refused bout keeps both fighters' records: for a UFC debutant the pro
  // record is all there is to go on, and the model has nothing to add.
  const skipRow = (s) => s.red && s.blue
    ? `<div class="panel">
         <div class="sides">
           ${side("r", s.red, formLine(s.red_form), null, false)}
           ${side("b", s.blue, formLine(s.blue_form), null, false)}
         </div>
         <div class="mini">${esc(s.reason)}</div>
       </div>`
    : `<div class="mini">${esc(s.fight)} — ${esc(s.reason)}</div>`;
  const skipped = d.skipped?.length
    ? `<div class="card"><h2>Not predicted</h2>
        <div class="mini">${d.skipped.length} of ${d.fights.length + d.skipped.length} fights: the model
          refuses a fighter it cannot measure rather than guessing.</div>
        ${d.skipped.map(skipRow).join("")}
       </div>` : "";

  const about = [d.caveats.intel, d.caveats.simulation].filter(Boolean)
    .map((t) => `<p style="margin:0">${esc(t)}</p>`).join("");

  el.innerHTML = `
    ${DESK.banner()}
    <div class="hero">
      <div class="eyebrow">${esc(longDate(d.event.date))}</div>
      <h1>${esc(d.event.name ?? "Fight card")}</h1>
      <div class="meta">
        <span class="chip">${d.fights.length} picks</span>
        ${d.fights.length ? `<span class="chip">${priced} priced</span>` : ""}
        ${d.skipped?.length ? `<span class="chip warn">${d.skipped.length} not predicted</span>` : ""}
        ${d.dataset_end ? `<span class="chip">results to ${esc(shortDate(d.dataset_end))}</span>` : ""}
      </div>
    </div>
    ${fold(d.caveats.model)}
    <div class="matches">${d.fights.map(boutRow).join("")}</div>
    ${skipped}
    ${doubtCard(d.fights, d.caveats.flag)}
    ${parlayCard(d.parlays, d.caveats.parlay)}
    <details class="fold"><summary><span class="ic">${icon("info")}</span><span class="clip"><b>Reading the card.</b>
      ${esc(FORM_LEGEND)}</span><span class="more">More</span></summary>
      <div class="fold-body">${about}</div></details>`;
};

// The header's second line: when the card was made and how far the data runs.
function stamp() {
  const s = $("stamp");
  if (!s || !CARD) return;
  const made = CARD.generated ? DESK.ago(CARD.generated) : "";
  s.textContent = made ? `Updated ${made}` : "Fight Desk";
}

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
// A height or reach outside the band matchup.js trusts is a scrape's units
// bug (381 cm), shown as unmeasured - as the size line under it already does.
const inBand = (v, what) => {
  const C = FIGHTERS.constants;
  return v == null || v < C[`${what}_MIN_CM`] || v > C[`${what}_MAX_CM`] ? null : v;
};
const TAPE = [
  ["Record", (s) => (s.cd_wins == null ? null
    : `${Math.round(s.cd_wins)}–${Math.round(s.cd_losses)}`), "high"],
  ["UFC bouts", (s) => s.cd_bouts, "high", int],
  ["Win rate", (s) => s.cd_win_rate, "high", pct],
  ["Height", (s) => inBand(s.height_cm, "HEIGHT"), "high", (v) => (v == null ? "—" : `${v.toFixed(0)} cm`)],
  ["Reach", (s) => inBand(s.reach_cm, "REACH"), "high", (v) => (v == null ? "—" : `${v.toFixed(0)} cm`)],
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
    out.innerHTML = `<div class="empty">Pick a fighter for each corner to see the tale of the tape
      and who has the edge where.${CARD && CARD.fights.length ? `<br><br>Or start from this weekend's card:` : ""}</div>
      ${cardShortcuts()}`;
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
    const sideCss = frac >= 0
      ? `left:50%;width:${half}%;background:var(--red)`
      : `right:50%;width:${half}%;background:var(--blue)`;
    return `<div class="adv-row"><div class="lab">${label}</div>
      <div class="gauge" role="img" aria-label="${label}: ${fmt(v)}">
        <div class="mid"></div><i style="${sideCss}"></i>
      </div>
      <div class="advval num">${fmt(v)}</div></div>`;
  }).join("");

  const size = M.sizeAdvantage(fa, fb);
  const height = M.heightAdvantage(fa, fb);
  const reach = M.reachAdvantage(fa, fb);

  out.innerHTML = `
    <div class="card">
      <div class="tape">
        <div class="l head">${esc(NAMES[redId])}</div><div class="lab"></div><div class="r2 head">${esc(NAMES[blueId])}</div>
        ${tape}
      </div>
    </div>
    <div class="card">
      <h2>Matchup advantage</h2>
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

// This weekend's bouts as one-tap pairs, for the fighters the archive knows.
function cardShortcuts() {
  if (!CARD || !NAMES) return "";
  const idOf = (n) => (NAMES.order.find(([, x]) => x === n) || [])[0];
  const pairs = CARD.fights.map((f) => [idOf(f.red), idOf(f.blue), f]).filter(([a, b]) => a && b).slice(0, 8);
  if (!pairs.length) return "";
  return `<div class="chips" style="justify-content:center">${pairs.map(([a, b, f]) =>
    `<button class="chip" type="button" data-pair="${esc(a)}|${esc(b)}">${esc(surname(f.red))} v ${esc(surname(f.blue))}</button>`).join("")}</div>`;
}

// A search box with its suggestions as buttons underneath. A datalist was
// used before; phones show it badly or not at all.
// "Lone’er" typed on an iPhone, "Abdul Rakhman", "alex volkanovski": the
// phone's curly apostrophe, a space for a hyphen, a short first name.
const foldName = (t) => String(t || "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
  .toLowerCase().replace(/[’‘'.]/g, "").replace(/[-\s]+/g, " ").trim();
function attachPicker(inputId, listId, onPick) {
  const input = $(inputId);
  const list = $(listId);
  const close = () => { list.innerHTML = ""; list.hidden = true; };
  input.addEventListener("input", () => {
    const q = foldName(input.value);
    if (q.length < 2) { close(); onPick(null); return; }
    const words = q.split(" ");
    const hits = NAMES.order.filter(([, , f]) => words.every((w) => f.includes(w))).slice(0, 8);
    const exact = hits.find(([, , f]) => f === q);
    onPick(exact ? exact[0] : null);
    if (exact && hits.length === 1) { close(); return; }
    list.hidden = !hits.length;
    list.innerHTML = hits.map(([id, n]) => {
      const s = fighterById(id);
      const rec = s && s.cd_wins != null ? `${Math.round(s.cd_wins)}–${Math.round(s.cd_losses)} UFC` : "";
      return `<button type="button" data-id="${esc(id)}"><span>${esc(n)}</span><span class="mini">${esc(rec)}</span></button>`;
    }).join("");
  });
  list.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    const id = b.getAttribute("data-id");
    input.value = NAMES[id];
    close();
    onPick(id);
  });
}

BOOT.compare = async () => {
  const el = $("view-compare");
  el.innerHTML = DESK.skeleton(2);
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
      .map(([id, name]) => [id, name, foldName(name)])
      .sort((a, b) => a[1].localeCompare(b[1]));
    M = makeMatchup(c.matchup);
    FORM = makeForm(c.matchup);
  } catch (e) { return fail(el, e, "the fighter database"); }
  if (!CARD) { try { CARD = await load("card"); } catch (e) { /* shortcuts only */ } }

  el.innerHTML = `
    <div class="hero">
      <div class="eyebrow">Head to head</div>
      <h1>Compare any two fighters</h1>
      <p class="lede">${FIGHTERS.index.size.toLocaleString()} fighters, each
        described only by the bouts they had actually fought.</p>
    </div>
    <div class="card pickers">
      <div class="field r">
        <label for="pick-red">Red corner</label>
        <input id="pick-red" autocomplete="off" autocorrect="off" autocapitalize="words" spellcheck="false"
               enterkeyhint="search" placeholder="Search a fighter">
        <div class="suggest" id="list-red" hidden></div>
      </div>
      <div class="field b">
        <label for="pick-blue">Blue corner</label>
        <input id="pick-blue" autocomplete="off" autocorrect="off" autocapitalize="words" spellcheck="false"
               enterkeyhint="search" placeholder="Search a fighter">
        <div class="suggest" id="list-blue" hidden></div>
      </div>
    </div>
    <div id="compare-out"></div>`;

  let redId = null, blueId = null;
  attachPicker("pick-red", "list-red", (id) => { redId = id; renderCompare(redId, blueId); });
  attachPicker("pick-blue", "list-blue", (id) => { blueId = id; renderCompare(redId, blueId); });
  $("compare-out").addEventListener("click", (e) => {
    const b = e.target.closest("[data-pair]");
    if (!b) return;
    [redId, blueId] = b.getAttribute("data-pair").split("|");
    $("pick-red").value = NAMES[redId];
    $("pick-blue").value = NAMES[blueId];
    renderCompare(redId, blueId);
  });
  renderCompare(null, null);
};

/* ---------------------------------------------------------------- record */
function lineChart(rows, key, opts) {
  const W = 600, H = 210, L = 58, R = 12, T = 16, B = 38;
  const xs = rows.map((r) => r.year);
  const ys = rows.map((r) => r[key]);
  const lo = Math.min(opts.floor, ...ys), hi = Math.max(...ys);
  const pad = (hi - lo) * 0.12 || 0.02;
  const y0 = lo - pad, y1 = hi + pad;
  const X = (i) => L + (i / (rows.length - 1)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const path = ys.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const area = `${path}L${X(ys.length - 1).toFixed(1)},${(H - B).toFixed(1)}L${X(0).toFixed(1)},${(H - B).toFixed(1)}Z`;
  const ticks = [y0 + (y1 - y0) * 0.1, (y0 + y1) / 2, y1 - (y1 - y0) * 0.1];
  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="${opts.label}">
    <defs><linearGradient id="g-${key}" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="var(--blue)" stop-opacity=".28"/><stop offset="1" stop-color="var(--blue)" stop-opacity="0"/></linearGradient></defs>
    ${ticks.map((t) => `<line x1="${L}" x2="${W - R}" y1="${Y(t).toFixed(1)}"
        y2="${Y(t).toFixed(1)}" stroke="var(--line-soft)" stroke-width="1"/>
      <text x="${L - 6}" y="${(Y(t) + 6).toFixed(1)}" text-anchor="end"
        font-size="17" font-family="IBM Plex Mono,monospace"
        fill="var(--ink-3)">${t.toFixed(2)}</text>`).join("")}
    ${opts.rule != null ? `<line x1="${L}" x2="${W - R}" y1="${Y(opts.rule).toFixed(1)}"
        y2="${Y(opts.rule).toFixed(1)}" stroke="var(--ink-3)" stroke-width="1"
        stroke-dasharray="3 3"/>` : ""}
    <path d="${area}" fill="url(#g-${key})"/>
    <path d="${path}" fill="none" stroke="var(--blue)" stroke-width="2.4"
      stroke-linejoin="round" stroke-linecap="round"/>
    ${ys.map((v, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}"
        r="${i === ys.length - 1 ? 4.5 : 3}" fill="var(--blue)"
        stroke="var(--surface)" stroke-width="2"><title>${xs[i]}: ${v.toFixed(3)}</title></circle>`).join("")}
    ${xs.map((x, i) => (i % 2 === 0 || i === xs.length - 1
      ? `<text x="${X(i).toFixed(1)}" y="${H - 8}" text-anchor="middle"
          font-size="17" font-family="IBM Plex Mono,monospace"
          fill="var(--ink-3)">${String(x).slice(2)}</text>` : "")).join("")}
  </svg>`;
}

function roiChart(rows) {
  const W = 600, H = 210, L = 58, R = 12, T = 16, B = 38;
  const vals = rows.map((r) => r.roi);
  const hi = Math.max(0.02, ...vals), lo = Math.min(-0.02, ...vals);
  const X = (i) => L + (i / rows.length) * (W - L - R);
  const bw = ((W - L - R) / rows.length) - 3;
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const zero = Y(0);
  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Flat-stake return by year, model strategy">
    ${[hi, 0, lo].map((t) => `<text x="${L - 6}" y="${(Y(t) + 6).toFixed(1)}"
        text-anchor="end" font-size="17" font-family="IBM Plex Mono,monospace"
        fill="var(--ink-3)">${(t * 100).toFixed(0)}%</text>`).join("")}
    ${rows.map((r, i) => {
      const y = Y(r.roi), h = Math.max(1.5, Math.abs(y - zero));
      const up = r.roi >= 0;
      return `<rect x="${X(i).toFixed(1)}" y="${(up ? y : zero).toFixed(1)}"
        width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="3"
        fill="${up ? "var(--gain)" : "var(--loss)"}">
        <title>${r.label}: ${(r.roi * 100).toFixed(1)}% over ${r.bets} bets</title></rect>`;
    }).join("")}
    <line x1="${L}" x2="${W - R}" y1="${zero.toFixed(1)}" y2="${zero.toFixed(1)}"
      stroke="var(--ink-3)" stroke-width="1"/>
    ${rows.map((r, i) => (i % 2 === 0 || i === rows.length - 1
      ? `<text x="${(X(i) + bw / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle"
          font-size="17" font-family="IBM Plex Mono,monospace"
          fill="var(--ink-3)">${String(r.label).slice(2)}</text>` : "")).join("")}
  </svg>`;
}

// A filled ring for one share: how much of the circle the hit rate covers.
function ring(share, colour = "var(--gain)") {
  if (share == null) return "";
  const r = 32, c = 2 * Math.PI * r, on = Math.max(0, Math.min(1, share)) * c;
  return `<svg class="ring" viewBox="0 0 76 76" aria-hidden="true">
    <circle cx="38" cy="38" r="${r}" fill="none" stroke="var(--sunk)" stroke-width="8"/>
    <circle cx="38" cy="38" r="${r}" fill="none" stroke="${colour}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${on.toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 38 38)"/></svg>`;
}

BOOT.record = async () => {
  const el = $("view-record");
  el.innerHTML = DESK.skeleton(2);
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
        ${signed(s.roi * 100, 1)}%${Array.isArray(s.interval) && s.interval[0] != null
          ? `<br><span class="mini">${signed(s.interval[0] * 100, 1)} to ${signed(s.interval[1] * 100, 1)}</span>` : ""}</td></tr>`).join("");

  const auc = (d.auc_by_year || []).filter((r) => r.auc != null);
  const roi = (d.roi_by_year || []).filter((r) => r.roi != null);
  const events = (t.by_event || []).slice(0, 8);

  el.innerHTML = `
    <div class="hero">
      <div class="eyebrow">Measured, not claimed</div>
      <h1>Track record</h1>
    </div>
    <div class="tiles">
      <div class="tile hero-tile">${ring(t.accuracy)}
        <div><div class="k">Live accuracy</div>
        <div class="v">${pct(t.accuracy, 1)}</div>
        <div class="s">over ${t.unique_bouts ?? "—"} settled bouts, each picked before the fight</div></div></div>
      <div class="tile"><div class="k">Upcoming</div>
        <div class="v">${t.upcoming ?? "—"}</div>
        <div class="s">picks still to be fought</div></div>
      <div class="tile"><div class="k">Predictions logged</div>
        <div class="v">${t.predictions == null ? "—" : t.predictions.toLocaleString()}</div>
        <div class="s">${t.graded ?? "—"} graded, some bouts more than once; accuracy counts each bout once</div></div>
      <div class="tile"><div class="k">Bouts in data</div>
        <div class="v">${(d.bouts || 0).toLocaleString()}</div>
        <div class="s">results to ${esc(d.dataset_end ?? "—")}</div></div>
      <div class="tile"><div class="k">Updated</div>
        <div class="v" style="font-size:20px">${esc(d.generated ? DESK.ago(d.generated) : "—")}</div>
        <div class="s">when this record was built</div></div>
    </div>

    ${events.length ? `<div class="card">
      <h2>Recent cards</h2>
      <ul class="rank">${events.map((e) => `<li><span><b>${esc(e.event)}</b><br><span class="mini">${esc(e.date)}</span></span>
        <span class="num ${e.correct / e.graded >= 0.5 ? "good" : "warn"}">${e.correct}/${e.graded}</span></li>`).join("")}</ul>
      <div class="mini">Picks logged before each card, graded against the result.</div>
    </div>` : ""}

    <div class="card">
      <h2>The three strategies</h2>
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
        ${fold(d.notes.which_number_counts)}`
      : `<div class="note warnbox"><b>Not measured yet.</b> These numbers are
          read from a backtest the engine has to run; they are deliberately not
          carried in the app as constants, because a hand-copied result goes
          stale silently. Run the <code>historical_backtest</code> experiment
          to fill this in.</div>`}
    </div>

    ${roi.length ? `<div class="card">
      <h2>Return by year</h2>
      <div class="mini">Model strategy, flat stakes. 2025 is absent because the
        odds source has no prices for it.</div>
      ${roiChart(roi)}
      <div class="legend">
        <span><i style="background:var(--gain)"></i>profit</span>
        <span><i style="background:var(--loss)"></i>loss</span>
      </div>
    </div>` : ""}

    ${auc.length ? `<div class="card">
      <h2>Ranking power by year</h2>
      <div class="mini">AUC — the chance the winner is ranked above the loser.
        0.50 is a coin toss, marked by the dashed line.</div>
      ${lineChart(auc, "auc", { floor: 0.5, rule: 0.5, label: "Model AUC by year" })}
    </div>` : ""}

    ${fold(d.notes.leak)}`;
};

/* ---------------------------------------------------------------- method */
BOOT.method = () => {
  $("view-method").innerHTML = `
    <div class="hero">
      <div class="eyebrow">What this is</div>
      <h1>How it works, and what it is worth</h1>
    </div>
    <div class="card">
      <h2>Two layers</h2>
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
      <h2>Every number is point-in-time</h2>
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
      <h2>Where the results come from</h2>
      <p class="lede">UFC fight statistics, through a public dataset of
        UFCStats. When that dataset falls behind, the latest results are read
        from each event's Wikipedia page and added as provisional bouts:
        winner, method and round, but no strike counts. They are replaced the
        day the full statistics arrive.</p>
    </div>
    <div class="card">
      <h2>What it does not do</h2>
      <ul class="plain">
        <li>It does not beat the market. No strategy's return since 2020 is
          distinguishable from zero: the measured figures, each with its range,
          are on the Record tab.</li>
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
DESK.init({
  key: "yourmma",
  name: "YourMMA",
  icon: "icons/icon-192.png",
  sibling: { name: "YourNFL", href: "../nfl/app/", blurb: "The NFL pick desk" },
  stamp: () => (CARD && CARD.generated ? `Card updated ${DESK.ago(CARD.generated)}` : "Fetch the latest card"),
  onReload: () => location.reload(),
});
DESK.tabs(["card", "compare", "record", "method"], BOOT, "yourmma.tab", "card");
setInterval(stamp, 60000);
