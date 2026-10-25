// YourNFL Pick Desk: the week's NFL games, the model's picks blended with
// the betting market, each team's form, and the model's own measured record.
//
// Data comes from data/*.json (served build) or from window.__YOURNFL_DATA__
// (the standalone file, nfl/engine/nfl_standalone.py). No module imports:
// the standalone build inlines this file as a classic script.

// desk.js:begin - copied from shared/desk.js by scripts/sync_desk.py; edit that file
// The Desk app chrome, shared by YourMMA, YourNFL and YourSoccer: install,
// the app menu, theme, the update notice, the icons and the small pieces
// every match card is built from. This file is the source;
// scripts/sync_desk.py copies it into every app's app.js, and a test fails
// if any copy drifts.
//
// No module syntax and one name (DESK): the MMA app runs as a module, the
// other apps and every standalone file as classic scripts, and the MMA
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

  const calm = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // A side's whole-number chance - unless another side would print the same
  // number, when it gets a decimal, so a pick is never "50% v 50%".
  const share = (s, ...others) => {
    const a = Math.round(s * 100);
    return others.some((o) => o != null && Math.round(o * 100) === a) ? (s * 100).toFixed(1) : String(a);
  };

  // The Close button at the foot of an opened match.
  const closer = (label) =>
    `<button class="detail-close" type="button" data-collapse aria-label="Close ${esc(label)}">${icon("chevron")}Close</button>`;

  // A secondary section of an opened match, folded to its title. The caller
  // escapes the title; an empty body draws nothing.
  const sub = (title, body) => !body ? "" :
    `<details class="fold sub"><summary><span class="clip"><b>${title}</b></span><span class="more">Show</span></summary><div class="fold-body">${body}</div></details>`;

  // A record card folded to its title and what it found; the table and its
  // caveats are one tap away, never gone.
  const cardFold = (title, take, body) =>
    `<div class="card"><details class="cardfold"><summary><div class="cf-head"><h2>${title}</h2>${icon("chevron")}</div>${take ? `<div class="take">${take}</div>` : ""}</summary><div class="cardfold-body">${body}</div></details></div>`;

  // An opened match whose summary sits low on the screen scrolls up to it,
  // so the tap visibly does something. toggle does not bubble, hence capture.
  document.addEventListener("toggle", (e) => {
    const d = e.target;
    if (!d.matches || !d.matches("details.match") || !d.open) return;
    const bar = document.querySelector(".tabbar");
    const limit = (bar ? bar.getBoundingClientRect().top : window.innerHeight) - 160;
    if (d.querySelector("summary").getBoundingClientRect().bottom > limit)
      d.scrollIntoView({ block: "start", behavior: calm() ? "auto" : "smooth" });
  }, true);
  document.addEventListener("click", (e) => {
    const c = e.target.closest("[data-collapse]");
    if (!c) return;
    const d = c.closest("details.match");
    if (!d) return;
    d.open = false;
    const head = document.querySelector(".appbar");
    if (d.getBoundingClientRect().top < (head ? head.getBoundingClientRect().bottom : 0)) d.scrollIntoView({ block: "start" });
  });
  // iOS applies :active (the press feedback) only with a touchstart listener.
  document.addEventListener("touchstart", () => {}, { passive: true });

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
  // A labelled modal with a close button: the page behind is inert (focus
  // cannot walk out onto it) and held still, and a swipe down dismisses it.
  let openSheet = null;
  function sheet(html) {
    closeSheet();
    const back = document.createElement("div");
    back.className = "sheet-back";
    back.innerHTML = `<div class="sheet" role="dialog" aria-modal="true"><div class="grab" aria-hidden="true"></div><button class="iconbtn sheet-x" type="button" data-close aria-label="Close">${icon("close")}</button>${html}</div>`;
    back.addEventListener("click", (e) => { if (e.target === back || e.target.closest("[data-close]")) closeSheet(); });
    const sh = back.querySelector(".sheet");
    const h = sh.querySelector("h2");
    if (h) { h.id = "sheet-title"; sh.setAttribute("aria-labelledby", "sheet-title"); }
    const prev = document.activeElement;
    const behind = [...document.body.children];
    document.body.appendChild(back);
    behind.forEach((el) => { el.inert = true; });
    openSheet = { back, prev, behind };
    let y0 = null;
    sh.addEventListener("touchstart", (e) => { y0 = sh.scrollTop <= 0 ? e.touches[0].clientY : null; }, { passive: true });
    sh.addEventListener("touchmove", (e) => { if (y0 != null) sh.style.transform = `translateY(${Math.max(0, e.touches[0].clientY - y0)}px)`; }, { passive: true });
    sh.addEventListener("touchend", (e) => {
      if (y0 == null) return;
      const dy = e.changedTouches[0].clientY - y0;
      y0 = null;
      if (dy > 90) closeSheet();
      else sh.style.transform = "";
    });
    const first = back.querySelector("button,a,[tabindex]");
    if (first) first.focus({ preventScroll: true });
    return sh;
  }
  function closeSheet() {
    if (!openSheet) return;
    const { back, prev, behind } = openSheet;
    openSheet = null;
    behind.forEach((el) => { el.inert = false; });
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

  // The banner a tab can show above its content, from the second visit
  // (the menu offers install from the first), until installed or waved off.
  function banner() {
    if (!canOfferInstall() || store.get(`${cfg.key}.install-dismissed`)
        || (+store.get(`${cfg.key}.visits`) || 0) < 2) return "";
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
    if (canOfferInstall()) rows.push(`<button class="row" type="button" data-install>
      <span class="ic">${icon("download")}</span><span class="t">Install ${esc(cfg.name)}<small>Home screen, full screen, offline</small></span></button>`);
    if (served()) for (const sib of cfg.siblings || (cfg.sibling ? [cfg.sibling] : [])) rows.push(`<a class="row" href="${esc(sib.href)}">
      <span class="ic">${icon("swap")}</span><span class="t">Open ${esc(sib.name)}<small>${esc(sib.blurb || "")}</small></span></a>`);
    rows.push(`<button class="row" type="button" data-reload>
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
  // cfg: { key, name, icon, siblings: [{name, href, blurb}], stamp(), onReload() }
  function init(options) {
    cfg = options;
    store.set(`${cfg.key}.visits`, String((+store.get(`${cfg.key}.visits`) || 0) + 1));
    applyTheme(theme());
    const m = document.getElementById("menu-btn");
    if (m) m.addEventListener("click", menu);
    // The sport switch links to the other apps on the same site; a saved
    // file has no neighbour to link to.
    const sw = document.getElementById("sport-switch");
    if (sw && !served()) sw.hidden = true;
    if (!served()) document.querySelectorAll("[data-served-only]").forEach((x) => { x.hidden = true; });
    serviceWorker();
  }

  // Wire a bottom tab bar: tabs are <button id="tab-NAME">, views
  // <section id="view-NAME">. boot[NAME]() runs the first time a tab opens.
  // Each tab keeps its scroll position, as a native tab bar does.
  function tabs(names, boot, storageKey, fallback) {
    const started = {}, pos = {};
    let current = null;
    for (const t of names) {
      const v = document.getElementById(`view-${t}`);
      v.setAttribute("role", "tabpanel");
      v.setAttribute("aria-labelledby", `tab-${t}`);
    }
    function show(name, scroll) {
      if (current) pos[current] = window.scrollY;
      for (const t of names) {
        document.getElementById(`view-${t}`).hidden = t !== name;
        const tab = document.getElementById(`tab-${t}`);
        tab.setAttribute("aria-selected", String(t === name));
        tab.tabIndex = t === name ? 0 : -1;
      }
      current = name;
      store.set(storageKey, name);
      if (scroll) window.scrollTo({ top: pos[name] || 0 });
      if (!started[name]) { started[name] = true; boot[name](); }
    }
    for (const t of names) {
      document.getElementById(`tab-${t}`).addEventListener("click", () => {
        const already = document.getElementById(`tab-${t}`).getAttribute("aria-selected") === "true";
        // Tapping the open tab again goes back to its top, as native apps do.
        if (already) { pos[t] = 0; window.scrollTo({ top: 0, behavior: calm() ? "auto" : "smooth" }); }
        else show(t, true);
      });
    }
    const list = document.querySelector('[role="tablist"]');
    if (list) list.addEventListener("keydown", (e) => {
      const i = names.findIndex((t) => document.getElementById(`tab-${t}`) === document.activeElement);
      if (i < 0) return;
      const n = names.length;
      const j = { ArrowRight: (i + 1) % n, ArrowLeft: (i + n - 1) % n, Home: 0, End: n - 1 }[e.key];
      if (j == null) return;
      e.preventDefault();
      document.getElementById(`tab-${names[j]}`).focus();
      show(names[j], true);
    });
    // A home-screen shortcut opens a tab directly (manifest "shortcuts").
    let first = new URLSearchParams(location.search).get("tab") || store.get(storageKey);
    if (!names.includes(first)) first = fallback;
    show(first, false);
    return { show, restart(name) { started[name] = false; } };
  }

  const skeleton = (n = 4) =>
    `<div class="skeleton" aria-label="Loading"><i class="short"></i>${"<i></i>".repeat(n)}</div>`;

  return { icon, esc, ago, sheet, closeSheet, toast, install, banner, init, tabs, skeleton, store,
    share, closer, sub, cardFold, calm };
})();
// desk.js:end

const $ = (id) => document.getElementById(id);
const pct = (x, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
// Rounded before the sign is taken: -0.004 is "0.00", not "−0.00".
const signed = (x, d = 1) => {
  if (x == null) return "—";
  const r = Number(Number(x).toFixed(d));
  return r === 0 ? (0).toFixed(d) : `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(d)}`;
};
const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// American odds as a bookmaker prints them.
const odds = (o) => (o == null ? "—" : o > 0 ? `+${Math.round(o)}` : o < 0 ? `−${Math.abs(Math.round(o))}` : "0");
// A spread from one side's point of view: "KC −7.5", "BUF +3", "PK".
const line = (team, pts) =>
  pts == null ? "—" : pts === 0 ? `${team} PK` : `${team} ${pts > 0 ? "+" : "−"}${Math.abs(pts)}`;
const icon = DESK.icon;
const shortDate = (d) => {
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? (d || "") : t.toLocaleDateString(undefined, { day: "numeric", month: "short", timeZone: "UTC" });
};

// Each club's colours, for the badge behind its code: [background, text].
const TEAM_COLOURS = {
  ARI: ["#97233F", "#fff"], ATL: ["#A71930", "#fff"], BAL: ["#241773", "#fff"], BUF: ["#00338D", "#fff"],
  CAR: ["#0085CA", "#000"], CHI: ["#0B162A", "#F26522"], CIN: ["#FB4F14", "#000"], CLE: ["#311D00", "#FF3C00"],
  DAL: ["#003594", "#fff"], DEN: ["#FB4F14", "#002244"], DET: ["#0076B6", "#fff"], GB: ["#203731", "#FFB612"],
  HOU: ["#03202F", "#fff"], IND: ["#002C5F", "#fff"], JAX: ["#006778", "#fff"], KC: ["#E31837", "#fff"],
  LA: ["#003594", "#FFD100"], LAC: ["#0072B0", "#fff"], LV: ["#000000", "#A5ACAF"], MIA: ["#007780", "#fff"],
  MIN: ["#4F2683", "#FFC62F"], NE: ["#002244", "#fff"], NO: ["#D3BC8D", "#101820"], NYG: ["#0B2265", "#fff"],
  NYJ: ["#125740", "#fff"], PHI: ["#004C54", "#fff"], PIT: ["#FFB612", "#101820"], SEA: ["#002244", "#69BE28"],
  SF: ["#AA0000", "#fff"], TB: ["#D50A0A", "#fff"], TEN: ["#0C2340", "#4B92DB"], WAS: ["#5A1414", "#FFB612"],
};
const teamChip = (code) => {
  const [bg, fg] = TEAM_COLOURS[code] || ["var(--ink-3)", "#fff"];
  return `<span class="teamchip" style="background:${bg};color:${fg}">${esc(code)}</span>`;
};

async function load(name) {
  const embedded = window.__YOURNFL_DATA__;
  if (embedded && embedded[name]) return embedded[name];
  const res = await fetch(`data/${name}.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name}.json — ${res.status}`);
  return res.json();
}

// A caveat folded to its first two lines, the rest one tap away.
const fold = (text, extra = "") => !text ? "" : `<details class="fold">
    <summary><span class="ic">${icon("info")}</span><span class="clip">${esc(text)}</span><span class="more">More</span></summary>
    ${extra ? `<div class="fold-body">${extra}</div>` : ""}</details>`;

// A failed load says so in words and offers to try again; the developer's
// hint is one tap away.
function fail(el, error, what, retry) {
  el.innerHTML = `<div class="note err"><b>Could not load ${esc(what)}.</b> Check the connection and try again.${
    retry ? `<div style="margin-top:10px"><button class="btn small ghost" type="button" data-retry>${icon("refresh")} Try again</button></div>` : ""}</div>
    ${fold(`${error.message}. The runner writes these files; to build them locally, run python nfl/engine/nfl_predict.py.`)}`;
  const b = el.querySelector("[data-retry]");
  if (b) b.addEventListener("click", () => retry());
}

// ---------------------------------------------------------------- form

// A team's record and last five games beside its name, newest on the left:
// a filled teal circle a win, an orange ring a loss, grey a tie, and the number inside is the
// margin. This season's regular-season record; the last five reach back into
// last season when this one is young. Not a model input.
const RESULT_WORD = { W: "Win", L: "Loss", T: "Tie" };
const RESULT_CLASS = { W: "w", L: "l", T: "t" };
const FORM_LEGEND = `Under each team: this season's record, then its last five games,
  newest first - a filled teal circle a win, an orange ring a loss, grey a tie. The number is the
  margin: a teal 7 is a seven-point win. Open a game for opponents and scores.`;
// What a screen reader hears instead of the dots: "won by 7, lost by 3".
const saidFive = (last5) => (last5 || []).length
  ? `<span class="sr">, last ${last5.length}: ${esc(last5.map((b) => b.result === "T" ? "tied" : `${b.result === "W" ? "won" : "lost"} by ${b.code}`).join(", "))}</span>` : "";

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
  if (!form || !form.record) return `<span class="rec none">no games yet</span>`;
  const r = form.record;
  const text = `${r.wins}-${r.losses}${r.ties ? `-${r.ties}` : ""}`;
  const dots = (form.last5 || []).map((b) => {
    const cls = RESULT_CLASS[b.result] || "t";
    return `<span class="res ${cls}" title="${esc(gameTip(b))}" aria-hidden="true">${esc(b.code)}</span>`;
  }).join("");
  return `<span class="rec num">${esc(text)}</span><span class="last5">${dots}</span>${saidFive(form.last5)}`;
}

function formDetail(code, form) {
  if (!form || !form.last5 || !form.last5.length) return "";
  const items = form.last5.map((b) => {
    const cls = RESULT_CLASS[b.result] || "t";
    const where = b.home || b.neutral ? "vs" : "at";
    return `<li><span class="res ${cls}" aria-hidden="true">${esc(b.code)}</span>
      <span><b>${esc(RESULT_WORD[b.result] || b.result)} ${esc(b.score)}</b> ${where} ${esc(b.opponent)}
      <span class="mini"><br>${esc(weekName(b))} ${esc(b.season)} · ${esc(b.date)}</span></span></li>`;
  }).join("");
  const po = form.playoffs ? ` · playoffs ${form.playoffs.wins}-${form.playoffs.losses}` : "";
  return `<div class="panel"><div class="eyebrow">${esc(code)} · last 5${esc(po)}</div><ul class="list">${items}</ul></div>`;
}

// ---------------------------------------------------------------- week

const TIER_TONE = { STRONG: "good", SOLID: "good", LEAN: "flat", "TOSS-UP": "flat" };
const STATUS_WORD = { final: "Final", started: "Kicked off", upcoming: "" };
// Where a game stands by the phone's own clock. The data says "upcoming"
// until the next run, hours after kickoff on a busy Sunday; past its kickoff
// a game is no longer upcoming, whatever the file says. Nothing here reads a
// result - "final" still comes only from the data.
const phase = (g, now = Date.now()) => g.status === "final" ? "final"
  : Date.parse(g.kickoff) <= now ? "started" : "upcoming";
const localStamp = (iso) => {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? String(iso || "") : t.toLocaleString(undefined,
    { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
};

// Kickoff is stored in UTC; the phone shows its own time zone.
const kickTime = (iso) => new Date(iso).toLocaleString(undefined,
  { weekday: "short", hour: "numeric", minute: "2-digit" });
const dayKey = (iso) => new Date(iso).toLocaleDateString(undefined,
  { weekday: "long", month: "short", day: "numeric" });

// One team: club badge, nickname over record and last five, and on the right
// its chance - or its score once the game is under way.
function teamRow(g, side) {
  const code = g[side];
  const full = g[`${side}_name`] || "";
  const name = full.split(" ").slice(-1)[0];
  const p = g.home_prob == null ? null : side === "home" ? g.home_prob : 1 - g.home_prob;
  const picked = g.pick && g.pick === code;
  const scored = g.score && g.score[side] != null;
  const lost = g.status === "final" && g.score && g.score[side] < g.score[side === "home" ? "away" : "home"];
  const right = scored
    ? `<span class="score num">${esc(g.score[side])}</span>`
    : `<span class="p">${p == null ? "—" : `${DESK.share(p, 1 - p)}<small>%</small>`}</span>`;
  const lead = side === "home" && !g.neutral ? `<span class="at">home</span>` : "";
  return `<div class="side ${side === "home" ? "r" : "b"}${picked ? " picked" : ""}${lost ? " lost" : ""}">
      <span class="mark"></span>
      <div class="who">${teamChip(code)}<span class="nm" title="${esc(full)}">${esc(name)}</span>${lead}</div>
      ${right}
      <div class="sub">${formLine(g[`${side}_form`])}</div>
    </div>`;
}

function marginText(g, m) {
  if (m == null) return "—";
  if (Math.abs(m) < 0.05) return "even";
  return m > 0 ? `${esc(g.home)} wins by ${Math.abs(m).toFixed(1)}` : `${esc(g.away)} wins by ${Math.abs(m).toFixed(1)}`;
}

function resultTag(g) {
  if (g.status !== "final" || !g.pick || !g.score) return "";
  const margin = g.score.home - g.score.away;
  if (margin === 0) return `<span class="tag flat">Tie</span>`;
  const won = (margin > 0 ? g.home : g.away) === g.pick;
  return won ? `<span class="tag good">${icon("check", 'style="width:13px;height:13px"')} Right</span>` : `<span class="tag bad">Wrong</span>`;
}

function qbText(qb) {
  if (!qb) return "—";
  const src = qb.replaced ? ` (starts for ${qb.replaced}, ruled out)`
    : qb.source === "last starter" ? " (assumed: last starter)" : qb.source === "projected" ? " (projected)" : "";
  const first = qb.starts === 0 ? " · first start" : qb.change ? " · new starter" : "";
  const hurt = qb.injury_status ? ` · <b>${esc(qb.injury_status)} on the injury report</b>` : "";
  return `<b>${esc(qb.name || qb.id)}</b>${esc(src)}<div class="mini">${signed(qb.rating, 2)}&nbsp;EPA/dropback${esc(first)}</div>${hurt}`;
}

function injuryList(code, list, total) {
  if (!list || !list.length) return `<div class="panel"><div class="eyebrow">${esc(code)} injuries</div><div class="mini">None listed out, doubtful or questionable.</div></div>`;
  const items = list.map((i) => `<li><span><b>${esc(i.name)}</b> <span class="mini">${esc(i.position)}${i.injury ? ` · ${esc(i.injury)}` : ""}</span></span>
    <span class="end st ${esc(String(i.status).toLowerCase())}">${esc(i.status)}</span></li>`).join("");
  const more = total > list.length ? `<div class="mini">+${total - list.length} more on the report</div>` : "";
  return `<div class="panel"><div class="eyebrow">${esc(code)} injuries</div><ul class="list">${items}</ul>${more}</div>`;
}

// ---------------------------------------------------------------- players

// Key players, touchdown chances and news - shown, not applied: none of it
// moves the pick except the starting quarterback.
const AVAIL_WORD = { out: "Out", doubtful: "Doubtful", questionable: "Questionable", active: "" };

// "Most valuable player" reads as the league award; this is the team's own.
function alertLine(a) {
  return `<div class="alert ${esc(a.status)}"><b>${esc(a.team)} ${esc(a.name)}</b>
    <span class="mini">${esc(a.position)} · ${esc(a.role === "most valuable player" ? "team's top player" : a.role)}</span> - ${esc(a.reason || AVAIL_WORD[a.status] || a.status)}</div>`;
}
// Numbers keep their units on the same line ("185&nbsp;rush"). Applied after
// esc, so the entity is not escaped.
const nb = (h) => h.replace(/(\d) /g, "$1&nbsp;").replace(/ yds\b/g, "&nbsp;yds");

function statusBadge(p) {
  return p.status && p.status !== "active"
    ? ` <span class="st ${esc(p.status)}">${esc(AVAIL_WORD[p.status] || p.status)}</span>` : "";
}

// Projected yards, if he plays: the middle of what his usage points to.
function projText(pr) {
  if (!pr) return "";
  const parts = [];
  if (pr.pass_yds != null) parts.push(`${Math.round(pr.pass_yds)} pass`);
  if (pr.rush_yds != null && pr.rush_yds >= 5) parts.push(`${Math.round(pr.rush_yds)} rush`);
  if (pr.rec_yds != null && pr.rec_yds >= 5) parts.push(`${Math.round(pr.rec_yds)} rec`);
  return parts.length ? `proj. ${parts.join(" · ")} yds` : "";
}

function leadersList(code, leaders) {
  if (!leaders || !Object.keys(leaders).length) return "";
  const li = ["passing", "rushing", "receiving"].filter((k) => leaders[k]).map((k) => {
    const l = leaders[k];
    const result = l.played === false ? ` <span class="st questionable">Did not play</span>`
      : l.actual != null
      ? ` <span class="st ${l.over ? "scored" : "questionable"}">${Math.round(l.actual)} (${l.over ? "over" : "under"})</span>` : "";
    return `<li><span class="num lead">${Math.round(l.yards)}</span>
      <span><b>${esc(l.name)}</b> <span class="mini">${esc(l.position)}</span>${statusBadge(l)}${result}<div class="mini">${esc(k)} yards</div></span></li>`;
  }).join("");
  return `<div class="panel"><div class="eyebrow">${esc(code)} · projected leaders</div><ul class="list">${li}</ul></div>`;
}

function keyPlayers(code, kp) {
  if (!kp) return "";
  const off = (kp.offense || []).map((p) => `<li><span class="rk num">${esc(p.rank)}</span>
    <span><b>${esc(p.name)}</b> <span class="mini">${esc(p.position)}</span>${statusBadge(p)}
    <div class="mini">${signed(p.value, 1)}&nbsp;EPA · ${nb(esc(p.line))}</div>${p.proj ? `<div class="mini">${nb(esc(projText(p.proj)))}</div>` : ""}</span></li>`).join("");
  const dfn = (kp.defense || []).map((p) => `<li><span class="rk num">D</span>
    <span><b>${esc(p.name)}</b> <span class="mini">${esc(p.position)}</span>${statusBadge(p)}
    <div class="mini">${nb(esc(p.line))}</div></span></li>`).join("");
  if (!off && !dfn) return "";
  return `<div class="panel"><div class="eyebrow">${esc(code)} · most valuable</div><ul class="list">${off}${dfn}</ul></div>`;
}

function tdList(code, list) {
  if (!list || !list.length) return "";
  const items = list.map((p) => {
    const result = p.played === false ? ` <span class="st questionable">Did not play</span>`
      : p.scored === true ? ` <span class="st scored">Scored</span>`
      : p.scored === false ? ` <span class="st questionable">No TD</span>` : "";
    const season = p.games_season ? `<div class="mini">${esc(p.tds_season)} TD in ${esc(p.games_season)} games this season</div>` : "";
    return `<li><span class="num lead">${pct(p.prob)}</span>
    <span><b>${esc(p.name)}</b> <span class="mini">${esc(p.position)}</span>${statusBadge(p)}${result}${season}</span></li>`;
  }).join("");
  return `<div class="panel"><div class="eyebrow">${esc(code)} · touchdown</div><ul class="list">${items}</ul></div>`;
}

function newsList(items) {
  if (!items || !items.length) return "";
  const li = items.map((n) => {
    const safe = /^https?:\/\//.test(n.link || "");
    const title = safe ? `<a href="${esc(n.link)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>` : esc(n.title);
    const who = (n.players || []).length ? ` · ${esc(n.players.join(", "))}` : "";
    return `<li><span>${title}<div class="mini">${esc(n.source)} · ${esc(DESK.ago(n.published))}${who}</div></span></li>`;
  }).join("");
  return `<div class="panel news"><div class="eyebrow">Player news · shown, not applied</div><ul class="list">${li}</ul></div>`;
}

function gameRow(g, d) {
  const p = g.home_prob;
  const bar = p == null
    ? `<div class="probbar none"><i style="flex:1"></i></div>`
    : `<div class="probbar${g.pick ? " has-pick" : ""}" role="img" aria-label="${esc(g.away)} ${pct(1 - p)}, ${esc(g.home)} ${pct(p)}"><i class="b${g.pick === g.away ? " pk" : ""}" style="flex:${1 - p}"></i><i class="r${g.pick === g.home ? " pk" : ""}" style="flex:${p}"></i></div>`;
  const ph = phase(g);
  const statusWord = STATUS_WORD[ph] || "";
  const status = statusWord
    ? `<span class="badge ${ph === "final" ? "" : "live"}">${esc(statusWord)}</span>` : "";
  const extra = [g.neutral ? "Neutral site" : "", g.div_game ? "Division" : ""].filter(Boolean)
    .map((x) => `<span class="badge">${esc(x)}</span>`).join("");

  // Chips for the band and for trouble only. News and the injury report sit
  // in the opened game: every game has both, so a chip said nothing.
  const tags = [];
  if (g.tier) tags.push(`<span class="tag ${TIER_TONE[g.tier] || "flat"}" title="Picks in this band won ${pct(g.tier_hit_rate, 1)} of 2016-2025 games">${esc(g.tier)}</span>`);
  const m = g.market || {};
  const pickOdds = g.pick === g.home ? m.home_ml : g.pick === g.away ? m.away_ml : null;
  for (const side of ["away", "home"]) {
    const qb = g[`${side}_qb`];
    if (qb && qb.injury_status) tags.push(`<span class="tag warn">${esc(g[side])} QB ${esc(qb.injury_status)}</span>`);
    if (qb && qb.replaced) tags.push(`<span class="tag warn">${esc(g[side])} backup QB</span>`);
  }
  // A top player out gets a chip only when the alert lines above do not
  // already say so.
  const shown = (g.alerts || []).slice(0, 2);
  const mvpOut = (g.alerts || []).filter((a) => a.role === "most valuable player" && a.status !== "questionable" && !shown.includes(a));
  for (const a of mvpOut) tags.push(`<span class="tag bad">${esc(a.team)} top player ${esc(AVAIL_WORD[a.status] || a.status)}</span>`);
  const injured = (g.home_injuries_total ?? (g.home_injuries || []).length)
    + (g.away_injuries_total ?? (g.away_injuries || []).length);
  tags.push(resultTag(g));

  const pickLine = g.pick
    ? `<span class="pick">${icon("check")}<span class="sr">Pick: </span>${esc(g.pick)}${
        pickOdds != null ? `<span class="odds num">${esc(odds(pickOdds))}</span>` : ""}</span>${tags.join("")}`
    : `<span class="mini">${esc(g.no_pick_reason || "No pick.")}</span>${tags.join("")}`;

  const sp = g.spread_pick, tp = g.total_pick;
  const kv = [
    ["Pick", g.pick ? `<b>${esc(g.pick)}</b> <span class="num">${pct(g.pick_prob, 1)}</span> <span class="mini">${g.market_prob == null ? "model alone - no market price" : "model and market blended"}</span>` : "—"],
    ["Band", g.tier ? `${esc(g.tier)} <span class="mini">picks in this band won ${pct(g.tier_hit_rate, 1)} of 2016-2025 games</span>` : "—"],
    ["Model alone", g.model_prob == null ? "—" : `${esc(g.away)} ${pct(1 - g.model_prob, 1)} · ${esc(g.home)} ${pct(g.model_prob, 1)}`],
    ["Market", g.market_prob == null ? "—" : `${esc(g.away)} ${pct(1 - g.market_prob, 1)} · ${esc(g.home)} ${pct(g.market_prob, 1)} <span class="mini">bookmaker's cut removed, ${esc(m.source === "the-odds-api" ? `${m.books ?? "?"} books` : "schedule line")}</span>`],
    ["Moneyline", m.home_ml == null ? "—" : `<span class="nowrap">${esc(g.away)} ${esc(odds(m.away_ml))}</span> · <span class="nowrap">${esc(g.home)} ${esc(odds(m.home_ml))}</span>`
      + (m.best_away_ml != null && m.best_home_ml != null ? ` <span class="mini">best <span class="nowrap">${esc(g.away)} ${esc(odds(m.best_away_ml))}${
        m.best_away_book ? ` (${esc(m.best_away_book)})` : ""}</span> · <span class="nowrap">${esc(g.home)} ${esc(odds(m.best_home_ml))}${
        m.best_home_book ? ` (${esc(m.best_home_book)})` : ""}</span></span>` : "")],
    ["Spread", (g.market_spread == null ? (g.model_spread == null ? "—" : `model: ${marginText(g, g.model_spread)}`)
      : `${esc(line(g.home, -g.market_spread))} <span class="mini">market</span> · model: ${marginText(g, g.model_spread)}`)
      + (m.opening_spread != null && m.opening_spread !== g.market_spread ? ` <span class="mini nowrap">first priced ${esc(line(g.home, -m.opening_spread))}</span>` : "")
      + (sp ? `<div class="mini">Model's side: ${esc(line(sp.team, sp.line))}, ${num(sp.edge, 1)}&nbsp;pts off the line</div>` : "")],
    ["Total", (g.market_total == null ? (g.model_total == null ? "—" : `model ${num(g.model_total, 1)}`)
      : `${num(g.market_total, 1)} <span class="mini">market</span> · model ${num(g.model_total, 1)}`)
      + (m.opening_total != null && m.opening_total !== g.market_total ? ` <span class="mini nowrap">first priced ${num(m.opening_total, 1)}</span>` : "")
      + (tp ? `<div class="mini">Model's side: ${esc(String(tp.side).toLowerCase())}</div>` : "")],
    ["Rest", `${esc(g.away)} ${g.away_rest == null ? "—" : Math.round(g.away_rest) + " days"} · ${esc(g.home)} ${g.home_rest == null ? "—" : Math.round(g.home_rest) + " days"}`],
    ["Elo", `${esc(g.away)} ${num(g.elo && g.elo.away, 0)} · ${esc(g.home)} ${num(g.elo && g.elo.home, 0)}`],
    ["Venue", `${esc(g.stadium || "—")}${g.roof ? ` · ${esc(g.roof)}` : ""}`],
  ];
  if (m.as_of) kv.push(["Prices", esc(localStamp(m.as_of))]);
  if (g.logged_at) kv.push(["Logged", `${esc(localStamp(g.logged_at))} <span class="mini">before kickoff</span>`]);

  const qbAlert = ["away", "home"].some((s) => g[`${s}_qb`] && g[`${s}_qb`].injury_status)
    ? `<div class="note warnbox">${esc(d.caveats.qb)}</div>` : "";

  // The pick first: who, how sure, the prices. How the number is built is
  // one tap away; the reference sections below are folded to their titles.
  const LEAD = ["Pick", "Band", "Moneyline", "Spread", "Total"];
  const rows = (l) => l.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
  const pickPanel = `<div class="panel"><div class="eyebrow">The pick</div>
      <dl class="kv">${rows(kv.filter(([k]) => LEAD.includes(k)))}</dl>
      ${(sp || tp) && d.caveats && d.caveats.lines ? `<div class="mini">${esc(d.caveats.lines)}</div>` : ""}
      <details class="fold"><summary><span class="clip">How the number is built: model alone, market, Elo, rest, venue, when priced</span><span class="more">Show</span></summary>
        <div class="fold-body" style="padding-left:14px"><dl class="kv">${rows(kv.filter(([k]) => !LEAD.includes(k)))}</dl></div></details>
    </div>`;
  const kp = keyPlayers(g.away, g.away_players) + keyPlayers(g.home, g.home_players);
  const forms = formDetail(g.away, g.away_form) + formDetail(g.home, g.home_form);

  return `<details class="match" data-status="${esc(ph)}">
    <summary>
      <div class="match-top"><span class="when">${esc(kickTime(g.kickoff))}</span>${status}${extra}<span class="chev">${icon("chevron")}</span></div>
      <div class="sides">${teamRow(g, "away")}${teamRow(g, "home")}</div>
      ${bar}
      <div class="pickline">${pickLine}</div>
      ${(g.alerts || []).slice(0, 2).map(alertLine).join("")}
      ${(g.alerts || []).length > 2 ? `<div class="mini">+${g.alerts.length - 2} more in the detail</div>` : ""}
    </summary>
    <div class="detail">
      ${pickPanel}
      <div class="panel"><div class="eyebrow">Quarterbacks</div>
        <dl class="kv"><dt>${esc(g.away)}</dt><dd>${qbText(g.away_qb)}</dd><dt>${esc(g.home)}</dt><dd>${qbText(g.home_qb)}</dd></dl></div>
      ${qbAlert}
      ${(g.alerts || []).length > 2 ? g.alerts.slice(2).map(alertLine).join("") : ""}
      ${DESK.sub(`Injury report · ${injured ? `${injured} listed` : "none listed"} · shown, not applied`,
        `<div class="two">${injuryList(g.away, g.away_injuries, g.away_injuries_total)}${injuryList(g.home, g.home_injuries, g.home_injuries_total)}</div>`)}
      ${DESK.sub(`Player news · ${(g.news || []).length} · shown, not applied`, newsList(g.news))}
      ${DESK.sub("Most valuable players", kp && `<div class="two">${kp}</div>`)}
      ${DESK.sub("Touchdown chances", (g.away_td || g.home_td) && `<div class="two">${tdList(g.away, g.away_td)}${tdList(g.home, g.home_td)}</div>
        <div class="mini">${esc((d.caveats && d.caveats.td) || "")}</div>`)}
      ${DESK.sub("Projected yards", (g.away_leaders || g.home_leaders) && `<div class="two">${leadersList(g.away, g.away_leaders)}${leadersList(g.home, g.home_leaders)}</div>
        <div class="mini">${esc((d.caveats && d.caveats.props) || "")}</div>`)}
      ${DESK.sub("Last 5 games", forms && `<div class="two">${forms}</div>`)}
      ${DESK.closer(`${g.away} at ${g.home}`)}
    </div>
  </details>`;
}

function upsetCard(games) {
  const open = games.filter((g) => phase(g) === "upcoming" && g.pick && g.pick_prob != null)
    .sort((a, b) => a.pick_prob - b.pick_prob).slice(0, 6);
  if (!open.length) return "";
  const items = open.map((g) => {
    const opp = g.pick === g.home ? g.away : g.home;
    const wrong = 1 - g.pick_prob;
    return `<li><span><b>${esc(g.pick)}</b> <span class="mini">over ${esc(opp)}</span></span>
      <span class="num ${wrong >= 0.4 ? "warn" : "flat"}">${pct(wrong)}</span>
      <span class="meter ${wrong >= 0.4 ? "" : "flat"}" aria-hidden="true"><i style="width:${Math.round(wrong * 100)}%"></i></span></li>`;
  }).join("");
  return `<div class="card"><h2>Upset watch</h2>
    <div class="mini">Chance each pick is wrong, most doubtful first.</div>
    <ul class="rank">${items}</ul>
    ${fold("The probabilities are calibrated: picks the model gave about 60% won about 59% of 2016-2025 games.")}</div>`;
}

let WEEK = null;
// The header's second line: when the picks were made and how far the data runs.
function stamp() {
  const s = $("stamp");
  if (!s || !WEEK) return;
  const made = WEEK.generated ? DESK.ago(WEEK.generated) : "";
  s.textContent = made ? `Updated ${made}` : "Pick Desk";
}

const BOOT = {
  async week() {
    const el = $("view-week");
    el.innerHTML = DESK.skeleton(4);
    let d;
    try { d = await load("week"); } catch (e) { return fail(el, e, "the week", () => BOOT.week()); }
    WEEK = d;
    stamp();
    const games = d.games || [];
    const predicted = games.filter((g) => g.pick).length;
    const count = (s) => games.filter((g) => phase(g) === s).length;
    const finals = count("final"), live = count("started"), upcoming = count("upcoming");
    const byDay = new Map();
    for (const g of games) {
      const k = dayKey(g.kickoff);
      if (!byDay.has(k)) byDay.set(k, []);
      byDay.get(k).push(g);
    }
    const days = [...byDay.entries()].map(([day, list]) =>
      `<div class="day" data-day>${`<div class="eyebrow sticky-head">${esc(day)}</div>`}
        <div class="matches">${list.map((g) => gameRow(g, d)).join("")}</div></div>`).join("");
    // A filter only when the week has games in more than one state.
    const states = [["all", "All", games.length], ["upcoming", "Upcoming", upcoming],
      ["started", "Kicked off", live], ["final", "Final", finals]].filter(([k, , n]) => k === "all" || n);
    const seg = states.length > 2
      ? `<div class="seg" role="group" aria-label="Show games">${states.map(([k, label, n]) =>
          `<button type="button" data-filter="${k}" aria-pressed="${k === "all"}">${label} <span class="num">${n}</span></button>`).join("")}</div>` : "";
    const notes = ["tiers", "value_off", "lines", "players", "news", "injuries"]
      .filter((k) => d.caveats && d.caveats[k])
      .map((k) => `<p style="margin:0">${esc(d.caveats[k])}</p>`).join("");
    el.innerHTML = `
      ${DESK.banner()}
      <div class="hero">
        <div class="eyebrow">${esc(d.season)} season</div>
        <h1>${esc(d.label)}</h1>
        <div class="facts">${[`${games.length} games`, `${predicted} picks`, finals ? `${finals} final` : "",
          d.data_through ? `results to ${esc(shortDate(d.data_through))}` : "",
          (d.byes || []).length ? `Bye: ${d.byes.map(esc).join(", ")}` : ""].filter(Boolean).join(" · ")}</div>
      </div>
      ${fold(d.caveats && d.caveats.model)}
      ${seg}
      ${games.length ? `<div class="days">${days}</div>` : `<div class="empty">No games this week.</div>`}
      ${upsetCard(games)}
      <details class="fold"><summary><span class="ic">${icon("info")}</span><span class="clip"><b>Reading the week.</b>
        ${esc(FORM_LEGEND)}</span><span class="more">More</span></summary>
        <div class="fold-body">${legend(d.tiers)}${notes}</div></details>`;
    el.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => {
      const k = b.getAttribute("data-filter");
      el.querySelectorAll("[data-filter]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      el.querySelectorAll("details.match").forEach((m) => { m.hidden = k !== "all" && m.getAttribute("data-status") !== k; });
      el.querySelectorAll("[data-day]").forEach((day) => {
        day.hidden = ![...day.querySelectorAll("details.match")].some((m) => !m.hidden);
      });
    }));
  },

  async teams() {
    const el = $("view-teams");
    el.innerHTML = DESK.skeleton(3);
    let d;
    try { d = await load("teams"); } catch (e) { return fail(el, e, "the teams", () => BOOT.teams()); }
    const teams = d.teams || [];
    // The most valuable player sits under the team chip, so every row is the
    // same height and the table fits a phone.
    const rows = teams.map((t) => {
      const r = t.record || {};
      const mvp = ((t.key_players || {}).offense || [])[0];
      return `<tr><td class="n">${esc(t.rank)}</td>
        <td><span style="display:inline-flex;align-items:center;gap:7px">${teamChip(t.team)}<span class="mini num nowrap">${esc(r.wins ?? 0)}-${esc(r.losses ?? 0)}${r.ties ? `-${esc(r.ties)}` : ""}</span></span>
          <div class="mini mvp">${mvp ? `${esc(mvp.name)} ${esc(mvp.position)}` : "—"}</div></td>
        <td class="n">${num(t.elo, 0)}</td><td class="n">${signed(t.net_epa, 2)}</td></tr>`;
    }).join("");
    // A native picker shows about 114px of text on a phone: the nickname
    // fits, "SF · San Francisco 49ers" does not. Grouped by division.
    const nick = (t) => String(t.name || t.team).split(" ").slice(-1)[0];
    const opt = (t) => `<option value="${esc(t.team)}">${esc(nick(t))}</option>`;
    const divs = [...new Set(teams.map((t) => t.division).filter(Boolean))].sort();
    const options = divs.map((dv) => `<optgroup label="${esc(dv)}">${teams.filter((t) => t.division === dv)
      .sort((x, y) => nick(x).localeCompare(nick(y))).map(opt).join("")}</optgroup>`).join("")
      + teams.filter((t) => !t.division).map(opt).join("");
    el.innerHTML = `
      <div class="hero"><div class="eyebrow">${esc(d.season)} season</div><h1>Teams</h1>
        <p class="lede">Ranked by Elo, after every game played so far.</p></div>
      <div class="card"><h2>Compare</h2>
        <div class="pickers pair">
          <div class="field b"><label for="cmp-a">First team</label><select id="cmp-a">${options}</select></div>
          <div class="field r"><label for="cmp-b">Second team</label><select id="cmp-b">${options}</select></div>
        </div>
        <div id="cmp-out"></div>
      </div>
      <div class="card"><h2>Power ratings</h2>
        <div class="scroll"><table>
          <thead><tr><th class="n">#</th><th>Team · most valuable</th><th class="n">Elo</th><th class="n">Net EPA</th></tr></thead>
          <tbody>${rows}</tbody></table></div>
        ${fold("Net EPA is expected points added per play on offence minus the same allowed on defence, weighted toward recent games. The name under each team is its most valuable player: the one whose plays added the most expected points this season.")}
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
    el.innerHTML = DESK.skeleton(2);
    let d;
    try { d = await load("performance"); } catch (e) { return fail(el, e, "the record", () => BOOT.record()); }
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
  ["Quarterback", (t) => (t.qb ? { name: t.qb, starts: t.qb_starts } : null), null,
    (v) => (v ? `${esc(v.name)}${v.starts != null ? `<div class="mini">${esc(v.starts)} starts</div>` : ""}` : "—"), true],
  ["QB EPA/dropback", (t) => t.qb_rating, "high", (v) => signed(v, 3)],
];

// A team's last five as circles alone, under its badge: the record has its
// own row, and both will not fit half a phone's width.
function lastFive(t, align) {
  const dots = (t.last5 || []).map((b) =>
    `<span class="res ${RESULT_CLASS[b.result] || "t"}" title="${esc(gameTip(b))}" aria-hidden="true">${esc(b.code)}</span>`).join("");
  return dots ? `<div class="last5" style="margin-top:6px;justify-content:${align}">${dots}</div>${saidFive(t.last5)}` : "";
}

function renderCompare(teams, codeA, codeB, scale) {
  const out = $("cmp-out");
  const A = teams.find((t) => t.team === codeA), B = teams.find((t) => t.team === codeB);
  if (!A || !B) { out.innerHTML = ""; return; }
  if (A.team === B.team) { out.innerHTML = `<div class="mini">Pick two different teams.</div>`; return; }
  // The better value is tinted in its side's colour: the first team is the
  // blue field on the left.
  const rows = TAPE.map(([lab, get, better, fmt, html]) => {
    const va = get(A), vb = get(B);
    let wa = "", wb = "";
    if (better && va != null && vb != null && va !== vb) {
      const aWins = better === "high" ? va > vb : va < vb;
      wa = aWins ? " win b" : ""; wb = aWins ? "" : " win r";
    }
    const n = html ? "" : " num";
    const show = (v) => (html ? fmt(v) : esc(fmt(v)));
    return `<div class="l${n}${wa}">${show(va)}</div><div class="lab">${esc(lab)}</div><div class="r2${n}${wb}">${show(vb)}</div>`;
  }).join("");
  const p = A.elo != null && B.elo != null ? eloChance(A.elo, B.elo) : null;
  const pts = p == null ? null : (A.elo - B.elo) * (scale.points_per_elo || 0.04);
  out.innerHTML = `
    ${p == null ? `<div class="probbar none"><i style="flex:1"></i></div>`
      : `<div class="probbar" style="margin:4px 0 2px" role="img" aria-label="Elo alone: ${esc(A.team)} ${pct(p)}, ${esc(B.team)} ${pct(1 - p)}"><i class="b" style="flex:${p}"></i><i class="r" style="flex:${1 - p}"></i></div>`}
    <div class="tape-head"><div class="l">${teamChip(A.team)}<div class="mini">${esc(A.division || "")}</div>${lastFive(A, "flex-end")}</div>
      <div class="r2">${teamChip(B.team)}<div class="mini">${esc(B.division || "")}</div>${lastFive(B, "flex-start")}</div></div>
    <div class="tape">${rows}</div>
    <div class="note">On a neutral field, Elo alone gives <b>${esc(A.team)} ${pct(p)}</b>
      (${pts == null ? "—" : pts >= 0 ? `${esc(A.team)} by ${pts.toFixed(1)}` : `${esc(B.team)} by ${(-pts).toFixed(1)}`}).
      Home field is worth about ${num((scale.hfa || 0) * (scale.points_per_elo || 0.04), 1)} points more.
      This is the Elo rating only - not the model's pick, which also reads quarterbacks, efficiency, rest and the market.</div>
    ${DESK.sub("Most valuable players", (keyPlayers(A.team, A.key_players) + keyPlayers(B.team, B.key_players))
      && `<div class="two">${keyPlayers(A.team, A.key_players)}${keyPlayers(B.team, B.key_players)}</div>`)}`;
}

// ---------------------------------------------------------------- record

function roiCell(r) {
  if (!r || r.roi == null) return `<td class="n">—</td>`;
  return `<td class="n ${r.roi >= 0 ? "gain" : "loss"}">${signed(r.roi * 100, 1)}%<span class="mini">${esc(r.bets)} bets</span></td>`;
}

function seasonChart(rows) {
  if (!rows || rows.length < 2) return "";
  const W = 640, H = 220, L = 64, R = 12, T = 14, B = 38;
  const lo = 0.5, hi = 0.8;
  const x = (i) => L + (i * (W - L - R)) / (rows.length - 1);
  const y = (v) => T + (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (H - T - B);
  const path = (key) => rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join("");
  const grid = [0.5, 0.6, 0.7, 0.8].map((v) =>
    `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line-soft)"/>
     <text x="${L - 6}" y="${y(v) + 6}" font-size="17" text-anchor="end" fill="var(--ink-3)">${Math.round(v * 100)}%</text>`).join("");
  // The last label ends at the edge rather than centring past it.
  const last = rows.length - 1;
  const labels = rows.map((r, i) => (i === last || (i % 3 === 0 && last - i >= 2))
    ? `<text x="${x(i)}" y="${H - 8}" font-size="17" text-anchor="${i === last ? "end" : i === 0 ? "start" : "middle"}" fill="var(--ink-3)">${esc(String(r.season))}</text>` : "").join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by season, model blend against the market favourite">
    ${grid}${labels}
    <path d="${path("market_accuracy")}" fill="none" stroke="var(--ink-3)" stroke-width="3" stroke-dasharray="4 3"/>
    <path d="${path("accuracy")}" fill="none" stroke="var(--blue)" stroke-width="3.6" stroke-linejoin="round"/>
  </svg>
  <div class="legend"><span><i style="background:var(--blue)"></i>Pick (blend)</span>
    <span><i style="background:var(--ink-3)"></i>Market favourite</span></div>`;
}

// A filled ring for one share: how much of the circle the hit rate covers.
function ring(share, colour = "var(--gain)") {
  if (share == null) return "";
  const r = 32, c = 2 * Math.PI * r, on = Math.max(0, Math.min(1, share)) * c;
  return `<svg class="ring" viewBox="0 0 76 76" aria-hidden="true">
    <circle cx="38" cy="38" r="${r}" fill="none" stroke="var(--line)" stroke-width="8"/>
    <circle cx="38" cy="38" r="${r}" fill="none" stroke="${colour}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${on.toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 38 38)"/></svg>`;
}

function renderRecord(d) {
  const b = d.backtest || {};
  const per = b.periods || { dev: [2006, 2015], confirm: [2016, 2025] };
  const dev = `${per.dev[0]}-${per.dev[1]}`, conf = `${per.confirm[0]}-${per.confirm[1]}`;
  const t = d.track_record || {};
  const W = b.winner || {};
  // The measured answer leads: how often the pick was right over a decade
  // of games it had not seen, beside the betting favourite. The live record
  // is a small sample and says so.
  const bl = (W.blend || {})[conf] || {};
  const tiles = `
    <div class="tiles">
      <div class="tile hero-tile">${ring(bl.accuracy)}
        <div><div class="k">Picks right, ${esc(conf)}</div><div class="v">${pct(bl.accuracy, 1)}</div>
        <div class="s">of ${bl.games == null ? "—" : bl.games.toLocaleString()} games, each season predicted only from the ones before · betting favourite ${pct(bl.market_accuracy, 1)}</div></div></div>
      <div class="tile"><div class="k">Live so far</div><div class="v">${t.graded ? `${esc(t.correct)}-${esc(t.graded - t.correct)}` : "—"}</div>
        <div class="s">${t.graded ? `${pct(t.accuracy, 1)} · ${esc(t.pending ?? 0)} still to play · logged before kickoff, a small sample` : "no graded picks yet"}</div></div>
      <div class="tile"><div class="k">Against the spread</div><div class="v">${esc((t.ats || {}).correct ?? 0)}-${esc(((t.ats || {}).picks ?? 0) - ((t.ats || {}).correct ?? 0))}</div>
        <div class="s">model's side, live</div></div>
      <div class="tile"><div class="k">Over / under</div><div class="v">${esc((t.totals || {}).correct ?? 0)}-${esc(((t.totals || {}).picks ?? 0) - ((t.totals || {}).correct ?? 0))}</div>
        <div class="s">model's side of the total, live</div></div>
      <div class="tile"><div class="k">Touchdowns</div><div class="v">${esc((t.td || {}).scored ?? 0)}/${esc((t.td || {}).picks ?? 0)}</div>
        <div class="s">touchdown picks that scored</div></div>
    </div>`;
  const liveNote = t.graded ? "" : `<div class="note">No graded picks yet: the live record starts with the first games
    picked before kickoff, and grows every week. The tables below are the walk-forward backtest.</div>`;

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
    ${seasonChart(b.by_season)}
    ${fold(`Every season predicted by models fitted only on the seasons before it. ${dev} is where choices were made; ${conf} was left untouched to judge them. Lower log loss is better; 0.693 is a coin toss.${
      t.log_loss != null ? ` Live so far: ${num(t.log_loss, 3)} over ${t.graded} games.` : ""}`)}</div>`;

  const tiers = (b.tiers || {})[conf] || {};
  const tierRows = ["STRONG", "SOLID", "LEAN", "TOSS-UP"].map((k) => {
    const r = tiers[k] || {};
    return `<tr><td><span class="tag ${TIER_TONE[k] || "flat"}">${esc(k)}</span></td><td class="n">${esc(r.picks ?? "—")}</td><td class="n">${pct(r.hit_rate, 1)}</td><td class="n">${pct(r.mean_prob, 1)}</td></tr>`;
  }).join("");
  const tierCard = `<div class="card"><h2>Confidence bands</h2>
    <div class="scroll"><table><thead><tr><th>Band</th><th class="n">Picks</th><th class="n">Won</th><th class="n">Said</th></tr></thead>
    <tbody>${tierRows}</tbody></table></div>
    <div class="mini">${esc(conf)}. "Said" is the average probability the pick was given; "won" is how often it won.
      Close together means the percentages can be taken at face value.</div></div>`;

  const v = b.value || {};
  const vrows = Object.entries(v.edges || {}).map(([edge, r]) =>
    `<tr><td class="n">${pct(Number(edge))}</td>${roiCell(r[dev])}${roiCell(r[conf])}</tr>`).join("");
  // The specialist cards fold to a title and what each found; the table is
  // one tap away. A takeaway whose inputs are missing is left out.
  const valueCard = DESK.cardFold("Betting the disagreements", v.enabled_at == null
      ? "It lost money in the later period at every edge, so no game is ever labelled a bet."
      : `Profitable in both periods from a ${esc(pct(v.enabled_at))} edge.`,
    `<div class="scroll"><table><thead><tr><th class="n">Edge</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${vrows}</tbody></table></div>
    <div class="mini">One unit at the closing moneyline on every side the model alone rated at least this far above
      the market with the bookmaker's cut removed (return per bet, bets beside).</div>`);

  const lineRows = (rep) => Object.entries(((rep || {})[conf]) || {}).sort(([a], [b2]) => parseFloat(a) - parseFloat(b2)).map(([edge, r]) => {
    const a = ((rep || {})[dev] || {})[edge] || {};
    return `<tr><td class="n">${esc(edge)}+</td><td class="n">${pct(a.hit_rate, 1)}<span class="mini">${esc(a.bets ?? "")} bets</span></td>
      <td class="n">${pct(r.hit_rate, 1)}<span class="mini">${esc(r.bets ?? "")} bets</span></td></tr>`;
  }).join("");
  const s0 = (((b.spread || {})[conf]) || {})["0"], t0 = (((b.total || {})[conf]) || {})["0"];
  const linesCard = DESK.cardFold("Spread and total", s0 && t0
      ? `Taking the model's side won ${pct(s0.hit_rate, 1)} against the spread and ${pct(t0.hit_rate, 1)} on totals in ${esc(conf)}; at -110 it takes 52.4%.` : "",
    `<div class="eyebrow">Model side against the closing spread</div>
    <div class="scroll"><table><thead><tr><th class="n">Gap (pts)</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${lineRows(b.spread)}</tbody></table></div>
    <div class="eyebrow">Model side against the closing total</div>
    <div class="scroll"><table><thead><tr><th class="n">Gap (pts)</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${lineRows(b.total)}</tbody></table></div>
    <div class="mini">Pushes left out. At -110 on both sides it takes 52.4% to break even.</div>`);

  const cal = (b.calibration || []).map((r) =>
    `<tr><td class="n">${pct(r.lo)}-${pct(r.hi)}</td><td class="n">${esc(r.games)}</td><td class="n">${pct(r.predicted, 1)}</td><td class="n">${pct(r.observed, 1)}</td></tr>`).join("");
  const td = d.td || {};
  const tdPeriods = Object.keys(td).filter((k) => /^20/.test(k));
  const tdRows = tdPeriods.map((k) => {
    const r = td[k] || {}, top = ((r.top_pick || {})[td.model]) || {}, t3 = r.top3 || {};
    return `<tr><td class="nowrap">${esc(k.replace(/^(\d{4})-\d\d(\d\d)$/, "$1–$2"))}</td><td class="n">${pct(top.hit_rate, 1)}<span class="mini">said ${pct(top.mean_prob, 1)}</span></td>
      <td class="n">${pct(t3.any_hit_rate ?? t3.any_scored, 1)}</td><td class="n">${num((r.log_loss || {})[td.model], 3)}<span class="mini">base ${num((r.log_loss || {}).base, 3)}</span></td></tr>`;
  }).join("");
  const tdCal = ((td[tdPeriods[tdPeriods.length - 1]] || {}).calibration || []).map((b) =>
    `<tr><td class="n">${pct(b.lo)}-${pct(b.hi)}</td><td class="n">${esc(b.rows)}</td><td class="n">${pct(b.predicted, 1)}</td><td class="n">${pct(b.observed, 1)}</td></tr>`).join("");
  const lp = tdPeriods[tdPeriods.length - 1];
  const tdTop = (((td[lp] || {}).top_pick) || {})[td.model] || {};
  const tdCard = tdRows ? DESK.cardFold("Touchdown scorers", tdTop.hit_rate != null
      ? `Each team's most likely scorer scored ${pct(tdTop.hit_rate, 1)} of the time in ${esc(lp)} (it said ${pct(tdTop.mean_prob, 1)}).` : "",
    `<div class="scroll"><table><thead><tr><th>Period</th><th class="n">Top pick</th><th class="n">Any of top 3</th><th class="n">Log loss</th></tr></thead>
    <tbody>${tdRows}</tbody></table></div>
    <div class="mini">"Top pick" is each team's most likely scorer in each game; "any of top 3" is how often at least one of
      the three shown scored. Lower log loss is better; "base" is guessing each position's average rate.</div>
    ${tdCal ? `<div class="scroll"><table><thead><tr><th class="n">Said</th><th class="n">Players</th><th class="n">Avg said</th><th class="n">Scored</th></tr></thead><tbody>${tdCal}</tbody></table></div>` : ""}
    ${(t.td || {}).picks ? `<div class="mini">Live: ${esc(t.td.scored)} of ${esc(t.td.picks)} touchdown picks scored.</div>` : ""}`) : "";
  const props = d.props || {};
  const propPeriods = Object.keys(props).filter((k) => /^20/.test(k));
  const STAT_WORD = { pass: "Passing", rush: "Rushing", rec: "Receiving" };
  const propRows = propPeriods.map((k) => `<tr class="grp"><td colspan="4">${esc(k)}</td></tr>` + ["pass", "rush", "rec"].map((stat) => {
    const r = (props[k] || {})[stat] || {}, mae = r.mae || {}, over = r.over || {}, m = (props.model || {})[stat];
    return `<tr><td>${esc(STAT_WORD[stat])}</td>
      <td class="n">${num(mae[m], 1)}</td><td class="n">${num(mae.mean, 1)}</td><td class="n">${pct(over[m], 1)}</td></tr>`;
  }).join("")).join("");
  const pl = propPeriods[propPeriods.length - 1];
  const miss = (stat) => { const r = ((props[pl] || {})[stat]) || {}; return (r.mae || {})[(props.model || {})[stat]]; };
  const [mp, mr, mc] = ["pass", "rush", "rec"].map(miss);
  const propCard = propRows ? DESK.cardFold("Yardage projections", mp != null && mr != null && mc != null
      ? `Off by ${num(mp, 0)} passing, ${num(mr, 0)} rushing and ${num(mc, 0)} receiving yards on average in ${esc(pl)}.` : "",
    `<div class="scroll"><table><thead><tr><th>Stat</th><th class="n">Off by</th><th class="n">Own avg off by</th><th class="n">Went over</th></tr></thead>
    <tbody>${propRows}</tbody></table></div>
    <div class="mini">"Off by" is the average miss in yards; "own avg" is the miss from simply using the player's
      recent average. A middle projection should be beaten about half the time.</div>
    ${(t.yards || {}).picks ? `<div class="mini">Live: ${esc(t.yards.over)} of ${esc(t.yards.picks)} projected leaders went over.</div>` : ""}`) : "";
  const calCard = cal ? DESK.cardFold("Calibration", `${esc(conf)}: the chance the pick gave the home side, against how often the home side won.`,
    `<div class="scroll"><table><thead><tr><th class="n">Home win %</th><th class="n">Games</th><th class="n">Said</th><th class="n">Happened</th></tr></thead>
    <tbody>${cal}</tbody></table></div>`) : "";

  const weeks = (t.by_week || []).slice(-8).reverse().map((w) =>
    `<li><span><b>${esc(w.season)} week ${esc(w.week)}</b></span>
      <span class="num ${w.correct / w.picks >= 0.5 ? "good" : "warn"}">${esc(w.correct)}-${esc(w.picks - w.correct)}</span></li>`).join("");
  const weekCard = weeks ? `<div class="card"><h2>Live, week by week</h2><ul class="rank">${weeks}</ul>
    <div class="mini">Picks logged before each kickoff, graded against the final score.</div></div>` : "";

  return `<div class="hero"><div class="eyebrow">Measured, not claimed</div><h1>Track record</h1>
      <p class="lede">Live picks are graded against final scores; the backtest replays every season since ${esc(per.dev[0])}.</p></div>
    ${tiles}${liveNote}${winners}${tierCard}${weekCard}${valueCard}${linesCard}${tdCard}${propCard}${calCard}`;
}

// ---------------------------------------------------------------- method

// One legend for every chip and term, on the Method tab and in the week's
// "Reading the week" fold. Band rates come from the data when it has them,
// never hand-copied.
function legend(tiers) {
  const rate = (k) => pct(((tiers || {})[k] || {}).hit_rate);
  return `<dl class="gloss">
    <dt><b class="num">59%</b></dt><dd>Each team's chance: the model blended with the market, or the model alone when there is no price.</dd>
    <dt><span class="pickline"><span class="pick">${icon("check")}CHI<span class="odds num">−157</span></span></span></dt>
      <dd>The pick and its price. −150 means stake 150 to win 100; +120 means stake 100 to win 120.</dd>
    <dt><span class="tag good">SOLID</span></dt><dd>How sure the pick is${tiers
      ? `; picks in each band won STRONG ${rate("STRONG")}, SOLID ${rate("SOLID")}, LEAN ${rate("LEAN")}, TOSS-UP ${rate("TOSS-UP")} of backtest games`
      : "; Record shows how often each band has won"}.</dd>
    <dt><span class="tag warn">NE QB Questionable</span></dt><dd>The starting quarterback is uncertain or replaced.</dd>
    <dt><span class="tag bad">top player Out</span></dt><dd>A key player is out: shown, not applied.</dd>
    <dt><span class="last5"><span class="res w">7</span><span class="res l">3</span></span></dt><dd>The last five games, newest first; the number is the margin.</dd>
    <dt><b>Elo</b></dt><dd>A rating that moves after every game by how surprising the result was.</dd>
    <dt><b>EPA</b></dt><dd>Expected points added per play: what a play was worth above an average one.</dd>
    <dt><b>Against the spread</b></dt><dd>Whether the side covered the bookmaker's handicap.</dd>
    <dt><b>Log loss</b></dt><dd>How far the chances were from what happened; lower is better, 0.693 is a coin toss.</dd>
  </dl>`;
}

const METHOD_HTML = `
  <div class="hero"><div class="eyebrow">What this is</div><h1>How it works</h1></div>
  <div class="card"><h2>Reading a pick</h2>${legend(null)}</div>
  <div class="card"><h2>Three numbers, one pick</h2>
    <ul class="plain">
      <li><b>Elo</b> - every team's rating moves after each game by how surprising the result was,
        more for a blowout, and is pulled part-way back to average every offseason.</li>
      <li><b>The model</b> - a logistic regression on Elo, offensive and defensive efficiency
        (expected points added per play, weighted toward recent games), the starting quarterback's own
        efficiency per dropback, rest, bye weeks, short weeks, travel across time zones, division games
        and neutral sites.</li>
      <li><b>The pick</b> - the model blended with the betting market's no-vig moneyline and its spread,
        with the weights learned from how they did on earlier seasons. The market is very good in the NFL,
        so the pick leans on it heavily.</li>
    </ul></div>
  <div class="card"><h2>Players</h2>
    <ul class="plain">
      <li><b>Most valuable</b> - expected points added this season on offence; sacks, takeaways and passes
        defended on defence. A key player out or doubtful is flagged at the top of the game.</li>
      <li><b>Touchdowns</b> - each player's chance of a rushing or receiving touchdown if he plays, from his
        recent carries, targets and snaps and his team's expected points.</li>
      <li><b>Yards</b> - the middle of what his recent usage and the game point to, if he plays.</li>
      <li>Touchdowns and yards are measured on past seasons - see Record - and logged before kickoff,
        then graded; a player who did not play is left out, since both were "if he plays". The
        most-valuable list describes this season so far.</li>
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

DESK.init({
  key: "yournfl",
  name: "YourNFL",
  icon: "icons/icon-192.png",
  siblings: [{ name: "YourMMA", href: "../../app/", blurb: "The MMA fight desk" },
    { name: "YourSoccer", href: "../../soccer/app/", blurb: "Europe's top five leagues" }],
  stamp: () => (WEEK && WEEK.generated ? `Picks updated ${DESK.ago(WEEK.generated)}` : "Fetch the latest picks"),
  onReload: () => location.reload(),
});
DESK.tabs(["week", "teams", "record", "method"], BOOT, "yournfl.tab", "week");
setInterval(stamp, 60000);
