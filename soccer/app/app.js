// YourSoccer: the matches still to play in every league the engine carries
// (soccer_leagues.in_app(): the top five, the other European divisions and
// the rest of the world), each club's form, the league tables and the
// model's own measured record. The engine (soccer/engine) writes
// data/*.json; this only draws it.
//
// A soccer match has three results, so every match card carries three
// chances - home, draw, away - and the pick is the likeliest of the three.
//
// One league picker (Matches, Tables and Record share it) chooses what is
// shown, and opens on the top five, so the app looks as it did before the
// other leagues came in until the reader picks more.

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
const esc = DESK.esc;
const icon = DESK.icon;
// A decimal price as the American odds the other two apps print: 2.50 is
// +150, 1.50 is −200.
const american = (dec) => {
  if (dec == null || !(dec > 1)) return null;
  return dec >= 2 ? Math.round((dec - 1) * 100) : Math.round(-100 / (dec - 1));
};
const odds = (dec) => {
  const o = american(dec);
  return o == null ? "—" : o > 0 ? `+${o}` : `−${Math.abs(o)}`;
};
const shortDate = (d) => {
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? (d || "") : t.toLocaleDateString(undefined, { day: "numeric", month: "short", timeZone: "UTC" });
};
const localStamp = (iso) => {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? String(iso || "") : t.toLocaleString(undefined,
    { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
};
// Kickoff is stored in UTC; the phone shows its own time zone.
const kickTime = (iso) => new Date(iso).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
const weekday = (iso) => new Date(iso).toLocaleDateString(undefined, { weekday: "short" });
const dayKey = (iso) => new Date(iso).toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
const NO_DATE = "Date to be confirmed";

async function load(name) {
  const embedded = window.__YOURSOCCER_DATA__;
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
    ${fold(`${error.message}. The runner writes these files; to build them locally, run python soccer/engine/soccer_predict.py.`)}`;
  const b = el.querySelector("[data-retry]");
  if (b) b.addEventListener("click", () => retry());
}

// A club's badge: its colours behind its three-letter code.
const teamChip = (club) => club
  ? `<span class="teamchip" style="background:${esc(club.bg)};color:${esc(club.fg)}">${esc(club.code)}</span>` : "";

// ---------------------------------------------------------------- form

// A club's last five league matches beside its name, newest on the left: a
// filled teal circle a win, an orange ring a loss, grey a draw, and the
// score from the club's side inside. In either tier, so a promoted club's
// run-in shows in August. Not a model input.
const RESULT_WORD = { W: "won", D: "drew", L: "lost" };
const FORM_LEGEND = `Under each club: its league position, then its last five league matches,
  newest first - a filled teal circle a win, an orange ring a loss, grey a draw, with the score
  from the club's side. Open a match for opponents and dates.`;
const formTip = (b) => `${RESULT_WORD[b.result] || b.result} ${b.gf}-${b.ga} ${b.home ? "v" : "at"} ${b.opponent} · ${b.date}${
  b.provisional ? " · provisional" : ""}`;
// A result from the free feeds that football-data.co.uk has not posted yet:
// a dashed edge on its pill, and said in words.
const PROV_TIP = "Provisional: from the free feeds (ESPN, fixturedownload.com); football-data.co.uk has not posted it yet";
// Inline, like the app's other one-off touches: the shared sheet is the three
// apps' and has no provisional state.
const PROV_EDGE = ' style="outline:1.5px dashed var(--gold-ink);outline-offset:-3px"';
function formLine(form, position) {
  const pos = position ? `<span class="rec num">${esc(ordinal(position))}</span>` : "";
  if (!form || !form.length) return pos || `<span class="rec none">no matches yet</span>`;
  const dots = form.map((b) =>
    `<span class="res ${esc(String(b.result).toLowerCase())}"${b.provisional ? PROV_EDGE : ""} title="${esc(formTip(b))}" aria-hidden="true">${esc(b.gf)}-${esc(b.ga)}</span>`).join("");
  const said = `<span class="sr">, last ${form.length}: ${esc(form.map((b) => `${RESULT_WORD[b.result] || b.result} ${b.gf}-${b.ga}${b.provisional ? " (provisional)" : ""}`).join(", "))}</span>`;
  return `${pos}<span class="last5 score5">${dots}</span>${said}`;
}
function formDetail(club, form) {
  if (!form || !form.length) return "";
  const items = form.map((b) => `<li><span class="res ${esc(String(b.result).toLowerCase())}"${b.provisional ? PROV_EDGE : ""} aria-hidden="true">${esc(b.gf)}-${esc(b.ga)}</span>
      <span><b>${esc(b.result === "W" ? "Won" : b.result === "L" ? "Lost" : "Drew")} ${esc(b.gf)}-${esc(b.ga)}</b> ${b.home ? "v" : "at"} ${esc(b.opponent_name || b.opponent)}
      <span class="mini"><br>${esc(shortDate(b.date))}${b.league_name ? ` · ${esc(b.league_name)}` : ""}${
        b.provisional ? ` · <span title="${esc(PROV_TIP)}">provisional</span>` : ""}</span></span></li>`).join("");
  return `<div class="panel"><div class="eyebrow">${esc(club.name)} · last ${form.length}</div><ul class="list">${items}</ul></div>`;
}
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const ordinal = (n) => {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
};

// ---------------------------------------------------------------- leagues

// WHY ONE PICKER, AND WHY IT OPENS ON THE TOP FIVE. The engine carries every
// league soccer_leagues.in_app() lists - 37 - and about seven times the five's
// matches in a week. Drawn all at once that is hundreds of cards on a phone,
// and a reader who came for the Premier League would scroll past the
// Brazilian Serie A to find it. So the app shows one choice at a time: "All
// leagues", a group, or one league. Matches, Tables and Record show the same
// choice, remembered on this device (DESK.store, which survives a private
// window by forgetting), and the default is the top five: the app looks as
// it did until the reader picks more.
//
// WHY EVERY NUMBER IS READ PER LEAGUE. Each league was measured on its own
// seasons and decided on its own (soccer_leagues.SHIPPED): its pick is the
// market recalibrated or the raw price, it ships only the goals numbers that
// beat its own base rates, its bands were measured over its group, and it
// carries its own caveats. A card is read against its own league's
// measurement, never against the top five's pooled one, and a number a
// league did not earn is shown as missing, with the reason.
//
// Files written before every league came in have no groups: they are the
// five, one group, and everything below falls back to how it read them.
const LEAGUES_KEY = "yoursoccer.leagues";
const DEFAULT_CHOICE = "g:top";
const GROUP_ORDER = ["top", "europe", "extra"];
const GROUP_LABEL = { top: "Top five", europe: "Other European", extra: "Rest of the world" };
const GROUP_PROSE = { top: "the top five", europe: "the other European divisions", extra: "the rest of the world" };

// Every league the files name, in the engine's order (league_order:
// soccer_data's), with what matches.json says about each, merged over what
// a tab's own file says (tables.json's leagues[], performance.json's
// by_league) so a tab still works when matches.json could not be loaded.
// A league named only because a pick of it is still on screen comes last.
function world(md, rows) {
  const meta = {};
  for (const r of rows || []) {
    if (r && r.code) meta[r.code] = { code: r.code, name: r.name, label: r.label, country: r.country,
      short: r.short, group: r.group, season_label: r.season_label };
  }
  for (const [c, v] of Object.entries((md && md.leagues) || {})) meta[c] = { ...(meta[c] || {}), ...v, code: c };
  const order = [];
  const add = (c) => { if (c && meta[c] && !order.includes(c)) order.push(c); };
  ((md && md.league_order) || []).forEach(add);
  (rows || []).forEach((r) => add(r && r.code));
  Object.keys(meta).forEach(add);
  const given = (md && md.groups) || null;
  const group = {};
  for (const c of order) {
    let g = meta[c].group;
    if (!g && given) g = Object.keys(given).find((k) => (given[k].codes || []).includes(c));
    // No groups in the file: it is the five, as before.
    group[c] = g || (given ? "other" : "top");
  }
  const groups = {};
  for (const g of [...GROUP_ORDER, ...Object.keys(given || {}), "other"]) {
    const codes = order.filter((c) => group[c] === g);
    if (!groups[g] && codes.length) groups[g] = { codes, name: GROUP_LABEL[g] || (given && given[g] && given[g].name) || "Other leagues" };
  }
  return { meta, order, group, groups, grouped: !!given };
}
const leagueLabel = (c, w) => { const x = w.meta[c] || {}; return x.label || x.name || c; };

// The choice: "all", "g:<group>" or a league's code. Kept in memory as well
// as on the device, so a phone that refuses storage still keeps it while the
// page is open.
let CHOICE = null;
const validChoice = (v, w) => !!v && (v === "all" || (v.startsWith("g:") ? !!w.groups[v.slice(2)] : !!w.meta[v]));
function choiceOf(w) {
  const v = CHOICE != null ? CHOICE : DESK.store.get(LEAGUES_KEY);
  if (validChoice(v, w)) return v;
  return w.groups.top ? DEFAULT_CHOICE : "all";
}
const kindOf = (v) => v === "all" ? "all" : v.startsWith("g:") ? "group" : "league";
function chosen(v, w) {
  if (v === "all") return w.order.slice();
  if (v.startsWith("g:")) return ((w.groups[v.slice(2)] || {}).codes || []).slice();
  return [v];
}
function choiceName(v, w, prose = false) {
  if (v === "all") return prose ? "every league" : "All leagues";
  if (v.startsWith("g:")) {
    const g = v.slice(2);
    return prose ? (GROUP_PROSE[g] || (w.groups[g] || {}).name || g) : ((w.groups[g] || {}).name || g);
  }
  return leagueLabel(v, w);
}

// The picker: a native select, so a phone draws its own list (a wheel on an
// iPhone) at any width, with "All leagues" and the groups first and then
// every league under its country. counts: {code: n} where the tab has
// something to count (the Matches tab's matches).
function picker(w, choice, tab, counts) {
  if (w.order.length < 2) return "";
  const id = `league-pick-${tab}`;
  const n = (codes) => counts ? ` (${codes.reduce((s, c) => s + (counts[c] || 0), 0)})` : "";
  const opt = (v, label) => `<option value="${esc(v)}"${v === choice ? " selected" : ""}>${esc(label)}</option>`;
  const byCountry = new Map();
  for (const c of w.order) {
    const k = w.meta[c].country || "Other";
    if (!byCountry.has(k)) byCountry.set(k, []);
    byCountry.get(k).push(c);
  }
  const groups = [opt("all", `All leagues${n(w.order)}`),
    ...Object.entries(w.groups).map(([g, x]) => opt(`g:${g}`, `${x.name}${n(x.codes)}`))].join("");
  const countries = [...byCountry].map(([k, codes]) => `<optgroup label="${esc(k)}">${
    codes.map((c) => opt(c, `${w.meta[c].name || c}${n([c])}`)).join("")}</optgroup>`).join("");
  return `<div class="field" data-picker style="margin:14px 0 10px"><label for="${id}">Leagues</label>
    <select id="${id}" data-league-pick><optgroup label="Several leagues">${groups}</optgroup>${countries}</select></div>`;
}

// Within a small group (the top five), the leagues side by side as well, as
// the Matches tab has always had them: one tap to one league and back.
// present: the leagues the tab has something for.
function quick(w, choice, present) {
  const g = choice === "all" ? null : choice.startsWith("g:") ? choice.slice(2) : w.group[choice];
  const codes = g && w.groups[g] ? w.groups[g].codes.filter((c) => present.has(c)) : [];
  if (codes.length < 2 || codes.length > 5) return "";
  return `<div class="seg" role="group" aria-label="Show a league">${[[`g:${g}`, "All"], ...codes.map((c) => [c, w.meta[c].short || c])]
    .map(([k, label]) => `<button type="button" data-quick="${esc(k)}" aria-pressed="${k === choice}">${esc(label)}</button>`).join("")}</div>`;
}

// What the shown seasons are called: each league on its own calendar ("2026
// season" for a calendar-year league, "2026-27 season" for August-May).
function seasonOf(codes, w, fallback) {
  const labels = [...new Set(codes.map((c) => (w.meta[c] || {}).season_label).filter(Boolean))];
  if (!labels.length) return fallback || "";
  if (labels.length === 1) return labels[0];
  return `${labels.map((l) => String(l).replace(/ season$/, "")).join(" and ")} seasons`;
}
// The last result the ratings include, per league: one date, or the range
// when the leagues shown differ (never one league's date for all of them).
function throughOf(codes, w, fallback) {
  const ds = [...new Set(codes.map((c) => (w.meta[c] || {}).model_through).filter(Boolean))].sort();
  if (!ds.length) return fallback ? `ratings include results through ${esc(shortDate(fallback))}` : "";
  if (ds.length === 1) return `ratings include results through ${esc(shortDate(ds[0]))}`;
  return `ratings include each league's results through ${esc(shortDate(ds[0]))}-${esc(shortDate(ds[ds.length - 1]))}`;
}

// What a reader must know about a league: its pick and its record on its own
// seasons (lead), then every caveat soccer_leagues holds for it - never
// filtered - then what the free feeds cover for it, beside them.
const leagueLines = (x) => [x.lead, ...(x.caveats || []), x.coverage_caveat].filter(Boolean);
const paras = (lines) => lines.map((t) => `<p style="margin:0 0 6px">${esc(t)}</p>`).join("");
function leagueFold(c, w) {
  const lines = leagueLines(w.meta[c] || {});
  if (!lines.length) return "";
  return `<details class="fold" data-league-notes="${esc(c)}"><summary><span class="ic">${icon("info")}</span><span class="clip"><b>${
    esc(leagueLabel(c, w))}.</b> ${esc(lines[0])}</span><span class="more">More</span></summary>${
    lines.length > 1 ? `<div class="fold-body">${paras(lines.slice(1))}</div>` : ""}</details>`;
}
function leagueNotes(codes, w) {
  const folds = codes.map((c) => leagueFold(c, w)).filter(Boolean);
  return folds.length ? `<div class="eyebrow" style="margin:16px 0 6px">${codes.length === 1 ? "About this league" : "About these leagues"}</div>${folds.join("")}` : "";
}

// A tab redraws from the data it last loaded when the choice changes: the
// one on screen at once, the others when next shown.
const VIEWS = {};
const STALE = new Set();
function setChoice(v) {
  CHOICE = v;
  DESK.store.set(LEAGUES_KEY, v);
  const a = document.activeElement;
  const focus = a && a.matches && (a.matches("[data-league-pick]") ? `#${a.id}`
    : a.matches("[data-quick]") ? `[data-quick="${a.getAttribute("data-quick")}"]` : null);
  for (const [name, draw] of Object.entries(VIEWS)) {
    if ($(`view-${name}`).hidden) STALE.add(name);
    else draw();
  }
  if (focus) {
    const x = document.querySelector(`section.view:not([hidden]) ${focus}`);
    if (x) x.focus({ preventScroll: true });
  }
}
document.addEventListener("change", (e) => {
  const s = e.target.closest && e.target.closest("[data-league-pick]");
  if (s) setChoice(s.value);
});
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-quick]");
  if (b) setChoice(b.getAttribute("data-quick"));
});

// ---------------------------------------------------------------- matches

const PICK_WORD = { H: "home win", D: "draw", A: "away win" };
// Where a match stands by the phone's own clock: past kickoff it is no longer
// upcoming, whatever the file says. Nothing here reads a result - "final"
// comes only from the data.
// Postponed, cancelled and abandoned come from the free feeds (the engine
// never picks them); "tbc" is a match with no date or time from any source.
const OFF = { postponed: "Postponed", cancelled: "Cancelled", abandoned: "Abandoned", tbc: "" };
const phase = (m, now = Date.now()) => m.status === "final" ? "final"
  : m.status in OFF ? m.status
  : Date.parse(m.kickoff) <= now ? "started" : "upcoming";
const STATUS_WORD = { final: "Full time", started: "Kicked off", upcoming: "", ...OFF };
// Where the top line's time comes from: a time the feeds have not confirmed
// is not printed as if it were.
// The sources list the match on different days (by the phone's clock).
const daysDiffer = (m) => new Set(Object.values(m.kickoff_sources || {}).filter(Boolean).map(dayKey)).size > 1;
const whenText = (m, ph) => m.status === "tbc" ? `${m.round ? `Round ${m.round} · ` : ""}date and time to be confirmed`
  : ph === "upcoming" && m.time_confirmed === false ? `${weekday(m.kickoff)} · ${daysDiffer(m) ? "Date" : "Time"} to be confirmed`
  : kickTime(m.kickoff);

// One outcome row: a club (or the draw), its chance on the right.
function outcome(kind, m) {
  const club = kind === "H" ? m.home : kind === "A" ? m.away : null;
  const p = { H: m.p_h, D: m.p_d, A: m.p_a }[kind];
  const others = ["H", "D", "A"].filter((k) => k !== kind).map((k) => ({ H: m.p_h, D: m.p_d, A: m.p_a }[k]));
  const picked = m.pick === kind;
  const scored = m.score && m.score.home != null;
  const lost = m.status === "final" && m.score && (
    kind === "H" ? m.score.home < m.score.away : kind === "A" ? m.score.away < m.score.home : m.score.home !== m.score.away);
  const right = kind !== "D" && scored
    ? `<span class="score num">${esc(kind === "H" ? m.score.home : m.score.away)}</span>`
    : `<span class="p">${p == null ? "—" : `${DESK.share(p, ...others)}<small>%</small>`}</span>`;
  if (kind === "D") {
    return `<div class="side d draw${picked ? " picked" : ""}${lost ? " lost" : ""}">
        <span class="mark"></span>
        <div class="who"><span class="nm">Draw</span></div>
        ${scored ? "" : right}
      </div>`;
  }
  const side = kind === "H" ? "home" : "away";
  return `<div class="side ${kind === "H" ? "r" : "b"}${picked ? " picked" : ""}${lost ? " lost" : ""}">
      <span class="mark"></span>
      <div class="who">${teamChip(club)}<span class="nm" title="${esc(club.name)}">${esc(club.name)}</span>${kind === "H" ? `<span class="at">home</span>` : ""}</div>
      ${right}
      <div class="sub">${formLine(m[`${side}_form`], m[`${side}_position`])}</div>
    </div>`;
}

function resultTag(m) {
  if (m.status !== "final" || !m.pick || !m.score) return "";
  return m.correct ? `<span class="tag good">${icon("check", 'style="width:13px;height:13px"')} Right</span>` : `<span class="tag bad">Wrong</span>`;
}

const pickName = (m, k) => k === "H" ? m.home.name : k === "A" ? m.away.name : "Draw";
const priceOf = (m, k, set = "avg") => ((m.prices || {})[set] || {})[{ H: "h", D: "d", A: "a" }[k]];

// ---------------------------------------------------------------- goals and scores

// The numbers beside the pick (engine: soccer_scores): both teams to score,
// over 1.5 and 3.5 goals and the likeliest scores, read off one score matrix -
// the market's where the home/draw/away and the over/under are both priced,
// the model's own where not, said so - and how sure the pick is (its band),
// with the hit rate measured for that band. Every rate is read from
// matches.json's scores_measured, never typed in here.
const BAND_WORD = { STRONG: "Strong", SOLID: "Solid", LEAN: "Lean", "TOSS-UP": "Toss-up" };
// The measured table's periods: the first chose, the last confirmed.
// scores_measured is the top five's at its top level (as the app has always
// read it) and, since every league came in, each group's under the group's
// key (soccer_scores.measure over the group's leagues, on its own periods).
// With a league (code, w), its group's - and none for a group with no
// measure, rather than the top five's numbers on another league's card.
function measured(d, code, w) {
  let sm = d && d.scores_measured;
  if (sm && code && w && w.grouped) {
    const g = w.group[code];
    sm = sm[g] && sm[g].periods ? sm[g] : g === "top" ? sm : null;
  }
  const keys = Object.keys((sm && sm.periods) || {}).sort();
  return keys.length ? { sm, dev: keys[0], conf: keys[keys.length - 1] } : null;
}
// A band's range from {BAND: floor} (matches.json's band_floors).
function floorRange(floors, band) {
  const all = Object.values(floors || {}).filter((v) => v != null).sort((a, b) => a - b);
  const f = (floors || {})[band];
  if (f == null) return "";
  const next = all.find((x) => x > f);
  return f === 0 ? `under ${pct(next)}` : next == null ? `${pct(f)} or more` : `${Math.round(f * 100)}-${pct(next)}`;
}
// A group's band hit rate (matches.json's bands: soccer_leagues.BANDS, rule
// d - measured per group on its confirmation period): {hit_rate, picks}.
const groupBand = (d, g, basis, band) => ((((d && d.bands) || {})[g] || {})[basis] || {})[band] || null;
// "2016-2025" -> "2016-25", for a narrow table head.
const shortPeriod = (key) => String(key || "").replace(/^(\d{4})-\d{2}(\d{2})$/, "$1-$2");
// A band's range of the pick's chance, from the floors the engine wrote.
function bandRange(ms, band) {
  const t = ms && (ms.sm.bands.pick || {})[ms.conf];
  if (!t || !t[band]) return "";
  const floors = Object.entries(t).filter(([k, v]) => k !== "all" && v && v.floor != null).map(([, v]) => v.floor).sort((a, b) => a - b);
  const f = t[band].floor, next = floors.find((x) => x > f);
  return f === 0 ? `under ${pct(next)}` : next == null ? `${pct(f)} or more` : `${Math.round(f * 100)}-${pct(next)}`;
}
function bandRate(ms, basis, band, key) {
  const b = ms && ((((ms.sm.bands || {})[basis] || {})[key || ms.conf]) || {})[band];
  return b && b.hit_rate != null ? b.hit_rate : null;
}
// The band, with how often picks that sure won: since every league came in,
// the rate measured over the card's own league group (bands[group]), never
// the top five's on another group's card; before, the top five's measure.
function bandText(m, d, w) {
  if (!m.band) return "—";
  const basis = m.band_basis === "model" ? "model" : "pick";
  const who = `${basis === "model" ? "the model's own picks" : "picks"} this sure`;
  if (d.bands && w) {
    const g = w.group[m.league];
    const r = groupBand(d, g, basis, m.band);
    const range = floorRange(d.band_floors, m.band);
    const period = ((d.bands || {})[g] || {}).period;
    return `<b>${esc(BAND_WORD[m.band] || m.band)}</b> <span class="mini">${who}${range ? ` (${esc(range)})` : ""}${
      r && r.hit_rate != null ? ` won ${pct(r.hit_rate, 1)} of ${esc(period || "")} matches in ${esc(GROUP_PROSE[g] || g)}`
        : " - not measured for this league's group, so no hit rate is shown"}</span>`;
  }
  const ms = measured(d);
  return `<b>${esc(BAND_WORD[m.band] || m.band)}</b> <span class="mini">${who}${
    ms ? ` (${esc(bandRange(ms, m.band))}) won ${pct(bandRate(ms, basis, m.band), 1)} of ${esc(ms.conf)} matches` : " - hit rate not loaded"}</span>`;
}
// How far below what happened both teams to score has run, in points - for
// the source the card's numbers come from (the market's matrix, or the
// model's own), each measured on its own, over the card's league group.
const sourceKey = (m) => m && m.score_source === "model" ? "model" : "shown";
function bttsShort(d, m, w) {
  const ms = measured(d, m && m.league, w);
  const b = ms && (ms.sm.periods[ms.conf] || {}).btts;
  const s = b && b[sourceKey(m)];
  return b && b.rate != null && s && s.mean_p != null ? Math.round((b.rate - s.mean_p) * 100) : null;
}
const REASON = {
  no_over_under: "only the home/draw/away price is posted; the market's score chances need the over/under price too",
  fit: "the prices fit no score matrix closely enough",
  no_model: "no numbers for this match",
};
function scoreSource(m, d) {
  const s = m.scores || {};
  const ms = measured(d);
  if (m.score_source === "market") return s.flagged
    ? `the market <span class="mini">- but no score chances fit these prices closely: the nearest misses the ${esc(s.miss_on || "")} by ${
      esc((s.miss * 100).toFixed(1))} points (more than ${esc(ms ? Math.round(ms.sm.fit_max_miss * 100) : "—")}), so read them with care. On such matches, measured, the
      model's own score chances fit the prices worse and scored worse (Method)</span>`
    : `the market <span class="mini">- the score chances that best reproduce the pick and the over/under price${
    s.miss != null ? `, within ${s.miss < 0.0005 ? "less than 0.1" : esc((s.miss * 100).toFixed(1))} points (on the ${esc(s.miss_on || "")})` : ""}</span>`;
  const why = s.reason === "no_price" ? (m.price_status === "not_yet" ? "no price yet" : "no price was posted")
    : s.reason === "fit" ? `the prices fit no score matrix within ${esc(ms ? Math.round(ms.sm.fit_max_miss * 100) : "—")} points (the closest missed the ${
      esc(s.miss_on || "")} by ${esc(s.miss != null ? (s.miss * 100).toFixed(1) : "—")})`
    : REASON[s.reason] || "";
  return `the model alone <span class="mini">- ${why}. Measured too, a little less accurate than the market's (Method)</span>`;
}
// A number a league did not earn (soccer_leagues.withheld: it did not beat
// the league's own base rates in both of its periods, from the source the
// card reads it from) is shown as missing, and says why - never filled in
// from the other source or from another league.
const WITHHELD_WHY = (src) => `not shown: in this league ${src === "both" ? "neither the market's number nor the model's own"
  : src === "model" ? "the model's own number" : src === "market" ? "the market's number" : "it"
  } did not beat the league's own rates in both of its periods`;
const withheldCell = (src) => `— <span class="mini">${esc(WITHHELD_WHY(src))}</span>`;
function goalsPanel(m, d, w) {
  const ms = measured(d, m.league, w);
  const s = m.scores;
  const lw = (((w && w.meta[m.league]) || {}).withheld) || {};
  const src = m.score_source === "model" ? "model" : "market";
  const held = new Set((s && s.withheld) || []);
  // Over 2.5 is the over/under market's where it is posted, else the
  // model's own; the entry nulls the one its league withholds and lists it
  // (scores.withheld). Which source that was is the league's to say.
  const o25mk = (lw.market || []).includes("o25"), o25md = (lw.model || []).includes("o25");
  const o25src = o25mk && o25md ? "both" : o25mk ? "market" : o25md ? "model" : null;
  const o25held = m.p_o25 == null && (held.has("o25") || (o25mk && o25md));
  const modelO25held = m.model_o25 == null && o25md;
  const rows = [["Over 2.5", m.p_o25 == null ? (o25held ? withheldCell(o25src) : "—") : `<span class="num">${pct(m.p_o25, 1)}</span> <span class="mini">${
    m.market_o25 != null ? "the over/under market" : "model alone - no over/under price"}${
    m.market_o25 != null && m.model_o25 != null ? ` · model ${pct(m.model_o25, 1)}` : ""}${
    m.market_o25 != null && modelO25held ? " · the model's own not shown here" : ""}</span>`]];
  if (s) {
    const isHeld = (k, v) => v == null && (held.has(k) || (lw[src] || []).includes(k));
    const low = isHeld("btts", m.p_btts) ? null : bttsShort(d, m, w);
    const both = m.score && m.score.home != null ? m.score.home > 0 && m.score.away > 0 : null;
    rows.push(["Over 1.5", isHeld("o15", m.p_o15) ? withheldCell(src) : `<span class="num">${pct(m.p_o15, 1)}</span>`]);
    rows.push(["Over 3.5", isHeld("o35", m.p_o35) ? withheldCell(src) : `<span class="num">${pct(m.p_o35, 1)}</span>`]);
    rows.push(["Both score", isHeld("btts", m.p_btts) ? withheldCell(src) : `<span class="num">${pct(m.p_btts, 1)}</span> <span class="mini">${
      low != null && low > 0 ? `runs about ${low} ${low === 1 ? "point" : "points"} low` : ""}${
      m.btts_correct != null ? `${low > 0 ? " · " : ""}${both ? "both scored" : "not both"} - ${m.btts_correct ? "right" : "wrong"}` : ""}</span>`]);
    const top1 = ms && ((ms.sm.periods[ms.conf] || {}).exact || {})[sourceKey(m)];
    rows.push(["Likeliest", isHeld("exact", s.top) ? withheldCell(src) : `${(s.top || []).map((x) => `<span class="nowrap">${esc(x.score)} <span class="num">${pct(x.p)}</span></span>`).join(" · ")}
      <span class="mini"><br>home-away${top1 && top1.top1_hit ? ` · long shots: the likeliest happened about 1 match in ${Math.round(1 / top1.top1_hit)} (${pct(top1.top1_hit, 1)}, ${esc(ms.conf)})` : ""}</span>`]);
    // The scores' own home/draw/away, beside the pick's - never in its place.
    // The model's alone, with no price, is the pick itself.
    if (s.matrix && !(m.score_source === "model" && !m.market)) rows.push(["Results",
      `<span class="nowrap">${esc(m.home.code)} ${pct(s.matrix.p_h, 1)}</span> · <span class="nowrap">draw ${pct(s.matrix.p_d, 1)}</span> · <span class="nowrap">${
        esc(m.away.code)} ${pct(s.matrix.p_a, 1)}</span> <span class="mini">home/draw/away by ${m.score_source === "model" ? "the model's own score chances" : "these score chances"}; the pick's stays as above</span>`]);
    rows.push(["From", scoreSource(m, d)]);
  } else if (m.pick) {
    rows.push(["Both score", `— <span class="mini">${m.status === "upcoming" || !m.logged_at ? "no score numbers for this match"
      : "not logged with this pick - it was logged before these numbers were"}</span>`]);
  }
  return `<div class="panel"><div class="eyebrow">Goals and scores</div><dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl></div>`;
}

// What the pick is, for this match: its league's own (soccer_leagues: the
// market recalibrated, or the raw price where recalibrating scored worse on
// its development seasons) - from the match's basis where the file has it,
// else its league's, else the file's one basis (before every league came in).
function pickHow(m, league, d) {
  const basis = m.basis === "recalibrated" || m.basis === "raw" ? m.basis : league.pick;
  if (basis === "raw") return "the market's chances, with the bookmakers' cut removed (not recalibrated in this league)";
  if (basis === "recalibrated") return "the market's chances, recalibrated";
  return d.basis === "market" ? "the market's chances, recalibrated" : "model and market blended";
}
// No pick on a match with no price, in a league where the model alone did
// not beat its own base rates (soccer_leagues rule b): it waits for the price.
const WAITS = "No pick until the price is posted";
const waitsForPrice = (m, league) => !m.pick && !m.market && (m.basis === "unpriced" || league.model_alone === false);

// ctx: { w: world(d), named: whether more than one league is on screen, so
// each card names its own }.
function matchRow(m, d, ctx = {}) {
  const w = ctx.w || world(d);
  const named = ctx.named !== false;
  const ph = phase(m);
  const bar = m.p_h == null
    ? `<div class="probbar none"><i style="flex:1"></i></div>`
    : `<div class="probbar${m.pick ? " has-pick" : ""}" role="img" aria-label="${esc(m.home.name)} ${pct(m.p_h)}, draw ${pct(m.p_d)}, ${esc(m.away.name)} ${pct(m.p_a)}">
        <i class="r${m.pick === "H" ? " pk" : ""}" style="flex:${m.p_h}"></i><i class="d${m.pick === "D" ? " pk" : ""}" style="flex:${m.p_d}"></i><i class="b${m.pick === "A" ? " pk" : ""}" style="flex:${m.p_a}"></i></div>`;
  // A provisional result says so in place of "Full time": the score on the
  // card already says the match is over, and two badges do not fit a phone.
  const statusWord = ph === "final" && m.provisional ? "Provisional" : STATUS_WORD[ph] || "";
  const status = statusWord ? `<span class="badge ${ph === "final" && !m.provisional ? "" : ph === "started" ? "live" : "gold"}"${
    m.provisional ? ` title="${esc(PROV_TIP)}"` : ""}>${esc(statusWord)}</span>` : "";
  const league = w.meta[m.league] || {};
  const notYet = m.price_status === "not_yet";
  const waits = waitsForPrice(m, league);
  const waitWhy = waits ? (m.no_pick_reason && m.no_pick_reason.startsWith(WAITS) ? m.no_pick_reason
    : `${WAITS}: in this league the model alone did not beat the league's own base rates in both of its periods.`) : "";

  const tags = [];
  if (m.pick && !m.market) tags.push(notYet
    ? `<span class="tag warn" title="football-data.co.uk has not posted this match with its prices yet">No price yet</span>`
    : `<span class="tag warn" title="No price was posted for this match">Model only</span>`);
  tags.push(resultTag(m));
  const price = m.pick ? priceOf(m, m.pick) : null;
  const pickLine = m.pick
    ? `<span class="pick">${icon("check")}<span class="sr">Pick: </span>${esc(m.pick === "D" ? "Draw" : pickName(m, m.pick))}${
        price != null ? `<span class="odds num">${odds(price)}</span>` : ""}</span>${tags.join("")}`
    : `<span class="mini">${esc(waits ? WAITS : (m.no_pick_reason || "No pick."))}</span>${tags.join("")}`;
  const facts = [];
  if (m.p_o25 != null) facts.push(m.p_o25 >= 0.5 ? `Over 2.5 goals ${pct(m.p_o25)}` : `Under 2.5 goals ${pct(1 - m.p_o25)}`);

  // The pick first: who, how sure, the prices. How the number is built is
  // one tap away.
  const three = (x) => x ? `${esc(m.home.code)} ${pct(x.p_h, 1)} · draw ${pct(x.p_d, 1)} · ${esc(m.away.code)} ${pct(x.p_a, 1)}` : "—";
  const lead = [
    ["Pick", m.pick ? `<b>${esc(pickName(m, m.pick))}${m.pick === "D" ? "" : " to win"}</b> <span class="num">${pct(m.pick_prob, 1)}</span>
      <span class="mini">${!m.market ? (notYet ? "No price yet - the model alone" : "model alone - no market price")
        : esc(pickHow(m, league, d))}</span>` : waits ? `— <span class="mini">${esc(waitWhy)}</span>` : "—"],
    ["Band", bandText(m, d, w)],
    ["Price", price == null ? "—" : `<span class="num">${odds(price)}</span> <span class="mini">average${
      priceOf(m, m.pick, "best") != null ? ` · best ${odds(priceOf(m, m.pick, "best"))}` : ""}</span>`],
  ];
  const built = [
    ["Model alone", three(m.model)],
    ["Market", m.market ? `${three(m.market)} <span class="mini">the prices with the bookmakers' cut removed</span>` : "—"],
    ["Prices", m.prices && m.prices.avg ? `<span class="nowrap">${esc(m.home.code)} ${odds(m.prices.avg.h)}</span> · <span class="nowrap">draw ${odds(m.prices.avg.d)}</span> · <span class="nowrap">${esc(m.away.code)} ${odds(m.prices.avg.a)}</span>` : "—"],
  ];
  if (m.prices_at) built.push(["Prices fetched", esc(localStamp(m.prices_at))]);
  if (m.logged_at) built.push(["Logged", `${esc(localStamp(m.logged_at))} <span class="mini">before kickoff</span>`]);
  // When the match is, and where that comes from: said wherever it is not
  // simply football-data.co.uk's confirmed kickoff.
  if (m.time_confirmed === false && m.kickoff && ph === "upcoming") lead.push(["Kickoff",
    `${esc(localStamp(m.kickoff))} <span class="mini">listed, not confirmed by both feeds${m.locked ? "" : ` - the pick closes ${
      m.freeze_at ? esc(localStamp(m.freeze_at)) : "at midnight UK time on match day"}`}</span>`]);
  // Every source's time where they differ: the shown one is football-data.co.uk's
  // (or ESPN's for a match only the feeds list); the pick closes at the earliest.
  if (m.kickoff_sources) lead.push(["Sources", `<span class="mini">${Object.entries(m.kickoff_sources).filter(([, t]) => t).map(([k, t]) =>
    `<span class="nowrap">${esc(k === "feeds" ? "the feeds" : k)} ${esc(kickTime(t))}</span>`).join(" · ")} - no pick stays open past the earliest</span>`]);
  if (m.locked && ph === "upcoming") lead.push(["Locked", `<span class="mini">the pick closed ${m.freeze_at ? esc(localStamp(m.freeze_at)) : ""}${
    m.pick ? " - shown as it was logged" : ""}</span>`]);
  if (ph === "final" && m.provisional) lead.push(["Result", `<span class="num">${esc(m.score.home)}-${esc(m.score.away)}</span> <span class="mini">provisional, from ${
    esc(m.result_source || "the free feeds")}; football-data.co.uk's own result confirms or corrects it</span>`]);
  if (m.disputed && ph !== "final") lead.push(["Result", `<span class="mini">withheld: the feeds disagree on the score (${
    Object.entries(m.disputed).filter(([, v]) => v).map(([k, v]) => `${esc(k)} ${esc(v)}`).join(", ")}); football-data.co.uk's will settle it</span>`]);
  (m.corrections || []).forEach((c) => lead.push(["Corrected", `<span class="mini">${c.now === "restored"
    ? `was void on ${esc((c.was && c.was.by) || "the feeds")}'s kickoff; ${esc(c.by || "")}'s shows it was logged in time - graded`
    : `${c.was && c.was.home != null ? `was ${esc(c.was.home)}-${esc(c.was.away)}` : "was pending"}${
      c.now ? `, now ${esc(c.now.home)}-${esc(c.now.away)}` : ", now pending"} (${esc(c.by || "")})`}</span>`]));
  if (m.priced_kept) lead.push(["Kept", `<span class="mini">the pick priced when football-data.co.uk listed the match; its
    fixtures file no longer does, so the time is the feeds' - the pick stays the market's</span>`]);
  // A pick logged before the feeds said the match was off or undated: kept,
  // not graded, and what becomes of it.
  if (m.held && m.pick) lead.push(["Held", `<span class="mini">${ph === "tbc"
    ? `logged for ${esc(localStamp(m.logged_for || m.kickoff))}; the feeds no longer give a date`
    : `${esc((STATUS_WORD[ph] || ph).toLowerCase())} after it was logged${m.moved_to ? ` - now listed for ${esc(localStamp(m.moved_to))}` : ""}`}.
    Kept in the log, not graded: a pick on the new date replaces it, and with no result within three weeks it is void.</span>`]);
  if (m.fixture_source && m.fixture_source !== "football-data.co.uk") built.push(["Fixture from",
    `${esc(m.fixture_source)} <span class="mini">free feeds; football-data.co.uk had not posted it</span>`]);
  const rows = (l) => l.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
  const forms = formDetail(m.home, m.home_form) + formDetail(m.away, m.away_form);
  // The league's own caveats on every card, folded, never dropped.
  const notes = leagueLines(league);
  const about = notes.length ? DESK.sub(esc([leagueLabel(m.league, w), league.country].filter(Boolean).join(" · ")), paras(notes)) : "";
  // With more than one league on screen each card names its own (the short
  // label, the full name for a screen reader); with one, the picker does.
  const badge = named ? `<span class="badge" title="${esc(leagueLabel(m.league, w))}"><span aria-hidden="true">${esc(league.short || m.league)}</span><span class="sr">${
    esc(leagueLabel(m.league, w))}</span></span>` : "";

  return `<details class="match" data-status="${esc(ph)}" data-league="${esc(m.league)}">
    <summary>
      <div class="match-top"><span class="when">${esc(whenText(m, ph))}</span>${badge}${status}<span class="chev">${icon("chevron")}</span></div>
      <div class="sides">${outcome("H", m)}${outcome("D", m)}${outcome("A", m)}</div>
      ${bar}
      <div class="pickline">${pickLine}</div>
      ${facts.length ? `<div class="factline num">${facts.join(" · ")}</div>` : ""}
    </summary>
    <div class="detail">
      <div class="panel"><div class="eyebrow">The pick</div>
        <dl class="kv">${rows(lead)}</dl>
        <details class="fold"><summary><span class="clip">How the number is built: model alone, market, prices, when priced</span><span class="more">Show</span></summary>
          <div class="fold-body" style="padding-left:14px"><dl class="kv">${rows(built)}</dl></div></details>
      </div>
      ${m.pick || m.p_o25 != null || m.scores ? goalsPanel(m, d, w) : ""}
      ${DESK.sub("Last 5 matches", forms && `<div class="two">${forms}</div>`)}
      ${about}
      ${DESK.closer(`${m.home.name} v ${m.away.name}`)}
    </div>
  </details>`;
}

// One legend for every chip and term, on the Method tab and in the
// matches' "Reading the matches" fold. It defines words; it never copies a
// measured number, which would go stale silently.
const LEGEND = `<dl class="gloss">
  <dt><b class="num">48%</b></dt><dd>The chance of each result - home win, draw, away win. Record says how each is built and how well it has done.</dd>
  <dt><span class="pickline"><span class="pick">${icon("check")}Arsenal<span class="odds num">−150</span></span></span></dt>
    <dd>The pick, the likeliest of the three results, and its price. −150 means stake 150 to win 100; +240 means stake 100 to win 240.</dd>
  <dt><span class="factline">Over 2.5 goals 58%</span></dt><dd>The chance of three or more goals in the match.</dd>
  <dt><b>Band</b></dt><dd>How sure the pick is, by its chance: <b>Strong</b> 60% or more, <b>Solid</b> 50-60%, <b>Lean</b>
    40-50%, <b>Toss-up</b> under 40%. Each is shown with how often picks that sure won when they were tested (Method).</dd>
  <dt><b>Both score</b></dt><dd>The chance both clubs score. It, over 1.5 and 3.5 goals and the likeliest scores come from one set
    of score chances: the one that fits the pick and the over/under price, or the model's own where those are not both posted.</dd>
  <dt><span class="tag warn">Model only</span></dt><dd>No price was posted, so the chances are the model's alone.</dd>
  <dt><b>No pick until the price is posted</b></dt><dd>In a league where the model alone did not beat the league's own
    rates in both of its measured periods, a match with no price has no pick: it waits for the market's.</dd>
  <dt><b>— not shown</b></dt><dd>A number this league did not earn: it did not beat the league's own rates in both of
    its measured periods, so it is missing rather than guessed. Each league's caveats say which.</dd>
  <dt><span class="tag warn">No price yet</span></dt><dd>A match from the free fixture feeds that football-data.co.uk has not posted
    with its prices yet (it does on Tuesday and Friday afternoons): the model's chances alone until then.</dd>
  <dt><span class="badge gold">Provisional</span></dt><dd>A result from the free feeds that football-data.co.uk has not posted yet.
    A pick graded on it is graded again on football-data.co.uk's result; a dashed edge marks it in a club's last five.</dd>
  <dt><b>Time to be confirmed</b></dt><dd>The feeds do not agree on the kickoff time, or only one lists it: the pick closes at
    midnight UK time on match day - or earlier, at any source's earlier time. <b>Date to be confirmed</b>: they list
    different days. A kickoff listed a week or more out can still move by days, even earlier, after both feeds agree on it:
    every run reads them again, and a pick logged after the real kickoff is void.</dd>
  <dt><span class="badge gold">Postponed</span></dt><dd>As the feeds report it. Not picked; a pick logged before it was
    postponed is kept, not graded, until a pick on the new date replaces it.</dd>
  <dt><span class="last5 score5"><span class="res w">2-1</span><span class="res l">0-1</span></span></dt><dd>The last five league matches, newest first, with the score from the club's side.</dd>
  <dt><b>Market</b></dt><dd>The bookmakers' prices turned into chances, with their cut removed.</dd>
  <dt><b>Walk-forward</b></dt><dd>Judged only on matches after the ones the model learned from.</dd>
  <dt><b>Log loss</b></dt><dd>How far the chances were from what happened; lower is better. Three even chances score 1.099.</dd>
</dl>`;

let MATCHES = null;
// The header's second line: when the picks were made.
function stamp() {
  const s = $("stamp");
  if (!s || !MATCHES) return;
  const made = MATCHES.generated ? DESK.ago(MATCHES.generated) : "";
  s.textContent = made ? `Updated ${made}` : "Pick Desk";
}

// The Matches tab for the picker's choice. Only the chosen leagues' cards
// are drawn (not drawn and hidden): every league at once is hundreds of
// cards, which a phone should build only when the reader asks for them.
function renderMatches(el, d) {
  const w = world(d);
  const choice = choiceOf(w);
  const kind = kindOf(choice);
  const codes = chosen(choice, w);
  const inChoice = (code) => kind === "all" || codes.includes(code);
  // A match with no date yet sorts last, under a heading of its own.
  const when = (m) => (m.kickoff ? Date.parse(m.kickoff) : Infinity);
  const every = (d.matches || []).slice().sort((a, b) => when(a) - when(b) || (a.league || "").localeCompare(b.league || ""));
  const counts = {};
  for (const m of every) counts[m.league] = (counts[m.league] || 0) + 1;
  const matches = every.filter((m) => inChoice(m.league));
  const shownLeagues = [...new Set(matches.map((m) => m.league))];
  const ctx = { w, named: kind !== "league" };
  const picks = matches.filter((m) => m.pick).length;
  const alone = matches.filter((m) => m.pick && !m.market && phase(m) === "upcoming").length;
  const byDay = new Map();
  for (const m of matches) {
    const k = m.kickoff ? dayKey(m.kickoff) : NO_DATE;
    if (!byDay.has(k)) byDay.set(k, []);
    byDay.get(k).push(m);
  }
  const days = [...byDay.entries()].map(([day, list]) =>
    `<div class="day" data-day><div class="eyebrow sticky-head">${esc(day)}</div>
      <div class="matches">${list.map((m) => matchRow(m, d, ctx)).join("")}</div></div>`).join("");
  // The picker, and within the top five its leagues side by side, by their
  // country's short name, in the engine's order, as before.
  const present = new Set(Object.keys(counts));
  const pick = `${picker(w, choice, "matches", counts)}${quick(w, choice, present)}`;
  const notes = Object.values(d.caveats || {}).filter(Boolean).map((t) => `<p style="margin:0">${esc(t)}</p>`).join("");
  // The latest round's results: facts, with no pick (none was logged).
  // football-data.co.uk's, and the free feeds' it has not posted yet - those
  // in a list of their own, called provisional.
  const results = (d.results || []).filter((r) => inChoice(r.league))
    .sort((a, b) => (b.date || "").localeCompare(a.date || "") || (a.league || "").localeCompare(b.league || ""));
  const resultRow = (r) => `<li data-league="${esc(r.league)}">
      ${ctx.named ? `<span class="mini nowrap" style="min-width:34px" title="${esc(leagueLabel(r.league, w))}">${esc((w.meta[r.league] || {}).short || r.league)}</span>` : ""}
      ${teamChip(r.home)}<span class="num" style="margin:0 2px;font-weight:600">${esc(r.hg)}–${esc(r.ag)}</span>${teamChip(r.away)}${
      r.provisional ? `<span class="sr"> (provisional)</span>` : ""}
      <span class="mini" style="margin-left:auto">${esc(shortDate(r.date))}</span></li>`;
  const prov = results.filter((r) => r.provisional), confirmed = results.filter((r) => !r.provisional);
  const resultsCard = results.length ? `<div class="card"><h2>Latest results</h2>
    ${prov.length ? `<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px 8px;margin-top:2px"><span class="badge gold" title="${esc(PROV_TIP)}">Provisional</span><span class="mini">${esc(plural(prov.length, "result", "results"))} from the free feeds</span></div>
      <ul class="rank" data-results>${prov.map(resultRow).join("")}</ul>` : ""}
    ${confirmed.length ? `${prov.length ? `<div style="display:flex;align-items:center;gap:8px;margin-top:12px"><span class="badge">Confirmed</span><span class="mini">football-data.co.uk</span></div>` : ""}
      <ul class="rank" data-results>${confirmed.map(resultRow).join("")}</ul>` : ""}
    <div class="mini">Full-time results from the last round. Picks are logged only before kickoff, so these have none.${
      prov.length ? " Provisional ones are ESPN's score, checked against fixturedownload.com where it has one, until football-data.co.uk posts its own; they are in the tables, but not in the model's ratings." : ""}</div></div>` : "";
  // The provisional results the file counts are every league's; one league
  // or group counts its own.
  const provCount = kind === "all" || !w.grouped ? d.provisional_results
    : (d.matches || []).filter((m) => inChoice(m.league) && m.status === "final" && m.provisional).length
      + (d.results || []).filter((r) => inChoice(r.league) && r.provisional).length;
  // The ratings' date and the season: the leagues on screen, each its own.
  const dated = shownLeagues.length ? shownLeagues : codes;
  const empty = kind === "all" || (kind === "group" && choice === DEFAULT_CHOICE && !w.grouped) ? "No league matches"
    : `No matches in ${esc(choiceName(choice, w, true))}`;
  el.innerHTML = `
    ${DESK.banner()}
    <div class="hero">
      <div class="eyebrow">${esc(seasonOf(dated, w, d.season_label) || "This week")}</div>
      <h1>Matches</h1>
      <div class="facts">${[plural(matches.length, "match", "matches"),
        `${plural(picks, "pick", "picks")}${alone ? `, ${alone} the model alone` : ""}`,
        throughOf(dated, w, d.model_through || d.data_through),
        provCount ? `<span class="warn">${esc(plural(provCount, "provisional result", "provisional results"))}</span>` : "",
        d.prices_at ? `prices fetched ${esc(localStamp(d.prices_at))}` : ""].filter(Boolean).join(" · ")}</div>
    </div>
    ${fold(d.caveats && d.caveats.model)}
    ${pick}
    ${matches.length ? `<div class="days">${days}</div>` : `<div class="empty">${empty} in the next ten days. The fixtures come from the free feeds as soon as they list them, and their prices on Tuesday and Friday afternoons (UK time).</div>`}
    ${leagueNotes(kind === "league" ? codes : shownLeagues, w)}
    ${resultsCard}
    <details class="fold"><summary><span class="ic">${icon("info")}</span><span class="clip"><b>Reading the matches.</b>
      ${esc(FORM_LEGEND)}</span><span class="more">More</span></summary>
      <div class="fold-body">${LEGEND}${notes}</div></details>`;
}

const BOOT = {
  async matches() {
    const el = $("view-matches");
    el.innerHTML = DESK.skeleton(4);
    let d;
    try { d = await load("matches"); } catch (e) { return fail(el, e, "the matches", () => BOOT.matches()); }
    MATCHES = d;
    stamp();
    VIEWS.matches = () => renderMatches(el, d);
    VIEWS.matches();
  },

  async tables() {
    const el = $("view-tables");
    el.innerHTML = DESK.skeleton(3);
    let d;
    try { d = await load("tables"); } catch (e) { return fail(el, e, "the tables", () => BOOT.tables()); }
    // The picker's leagues and groups come with the matches; without them
    // the tables' own list serves.
    const md = MATCHES || await load("matches").catch(() => null);
    VIEWS.tables = () => renderTables(el, d, md);
    VIEWS.tables();
  },

  async record() {
    const el = $("view-record");
    el.innerHTML = DESK.skeleton(2);
    let d;
    try { d = await load("performance"); } catch (e) { return fail(el, e, "the record", () => BOOT.record()); }
    // The leagues, groups and the bands' hit rates come with the matches.
    const md = MATCHES || await load("matches").catch(() => null);
    VIEWS.record = () => { el.innerHTML = renderRecord(d, md); };
    VIEWS.record();
  },

  async method() {
    const el = $("view-method");
    el.innerHTML = methodHtml(MATCHES);
    // The measured table comes with the matches; the tab can open first.
    if (!MATCHES) {
      let d = null;
      try { d = await load("matches"); } catch (e) { d = null; }
      el.innerHTML = methodHtml(d, !d);
    }
  },
};

// A tab marked stale while hidden redraws the moment it is shown again.
const redraw = new MutationObserver(() => {
  for (const name of [...STALE]) {
    if (!$(`view-${name}`).hidden && VIEWS[name]) { STALE.delete(name); VIEWS[name](); }
  }
});
for (const name of ["matches", "tables", "record"]) redraw.observe($(`view-${name}`), { attributes: true, attributeFilter: ["hidden"] });

// ---------------------------------------------------------------- tables

// The Tables tab for the picker's choice: the chosen league's table, or the
// group's tables one after another (every group's, under its name, for all
// leagues). Each table says what it is not where its league's format is not
// one double round robin (table_note) and carries its league's caveats.
function renderTables(el, d, md) {
  const leagues = d.leagues || [];
  const w = world(md, leagues);
  const choice = choiceOf(w);
  const kind = kindOf(choice);
  const byCode = new Map(leagues.map((l) => [l.code, l]));
  const codes = (kind === "all" ? [...w.order, ...leagues.map((l) => l.code)] : chosen(choice, w))
    .filter((c, i, a) => byCode.has(c) && a.indexOf(c) === i);
  const one = codes.length === 1;
  // The table fits a phone: position, club with its last five under it,
  // played, goal difference, points. The model's own rating of each club
  // is a list of its own below, ranked by that rating - folded where
  // several tables follow one another.
  const table = (l) => {
      const rows = (l.rows || []).map((r) => `<tr>
          <td class="n">${esc(r.position)}</td>
          <td><span style="display:inline-flex;align-items:center;gap:7px">${teamChip(r.club)}<span>${esc(r.club.name)}</span></span>
            ${r.form && r.form.length ? `<div style="margin-top:5px">${formLine(r.form)}</div>` : ""}</td>
          <td class="n">${esc(r.played)}${r.provisional ? `<sup title="${esc(plural(r.provisional, "provisional result", "provisional results"))}">*</sup>` : ""}</td><td class="n">${signed(r.gd, 0)}</td><td class="n"><b>${esc(r.points)}</b></td></tr>`).join("");
      const rated = (l.rows || []).filter((r) => r.rating != null).sort((a, b) => b.rating - a.rating);
      const top = rated.length ? rated[0].rating : 0, low = rated.length ? rated[rated.length - 1].rating : 0;
      const through = l.model_through || d.model_through;
      const name = l.label || l.name;
      const ratingsBody = `<div class="mini">${esc((d.caveats && d.caveats.rating_short) || "How strong the model rates each club right now, strongest first.")}${
          through ? ` Ratings include results through ${esc(shortDate(through))}${l.provisional ? " - not the provisional ones" : ""}.` : ""}</div>
        <ul class="rank">${rated.map((r, i) => `<li><span class="num" style="margin:0;min-width:22px;font-size:13px;color:var(--ink-3)">${i + 1}</span>
          ${teamChip(r.club)}<span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(r.club.name)}">${esc(r.club.name)}</span>
          <span class="num">${signed(r.rating, 2)}</span>
          <span class="meter good" aria-hidden="true"><i style="width:${top > low ? Math.round(8 + 92 * (r.rating - low) / (top - low)) : 50}%"></i></span></li>`).join("")}</ul>
        ${fold(d.caveats && d.caveats.rating)}`;
      const ratings = !rated.length ? "" : one ? `<div class="card"><h2>The model's ratings</h2>${ratingsBody}</div>`
        : DESK.cardFold(`The model's ratings: ${esc(name)}`, `${esc(rated[0].club.name)} rated strongest`, ratingsBody);
      const provNote = l.provisional ? `<div class="mini" style="margin-top:8px">* ${esc(plural(l.provisional, "result", "results"))} through ${
        esc(shortDate(l.results_through))} ${l.provisional === 1 ? "is" : "are"} provisional - from the free feeds, which football-data.co.uk has
        not confirmed yet - and counted here; a dashed edge marks ${l.provisional === 1 ? "it" : "them"} in each club's last five.</div>` : "";
      // Several leagues: each says its country and season, on its own calendar.
      const sub = !one ? [l.country || (w.meta[l.code] || {}).country, l.season_label].filter(Boolean).join(" · ") : "";
      const caveats = leagueLines({ ...(w.meta[l.code] || {}), caveats: l.caveats || (w.meta[l.code] || {}).caveats });
      return `<div class="card" data-table="${esc(l.code)}"><h2>${esc(name)}</h2>${sub ? `<div class="mini">${esc(sub)}</div>` : ""}
        <table class="tight">
          <thead><tr><th class="n">#</th><th>Club · last five</th><th class="n">P</th><th class="n">GD</th><th class="n">Pts</th></tr></thead>
          <tbody>${rows}</tbody></table>${provNote}
          ${l.table_note ? fold(l.table_note) : ""}
          ${caveats.length ? DESK.sub(`About the ${esc(name)}`, paras(caveats)) : ""}</div>${ratings}`;
  };
  // All leagues: each group under its name.
  const body = !codes.length
    ? `<div class="empty">No table yet for ${esc(choiceName(choice, w, true))}.</div>`
    : kind === "all" && Object.keys(w.groups).length > 1
      ? Object.entries(w.groups).map(([g, x]) => {
        const mine = codes.filter((c) => w.group[c] === g);
        return mine.length ? `<div class="eyebrow" style="margin:16px 0 6px">${esc(x.name)}</div>${mine.map((c) => table(byCode.get(c))).join("")}` : "";
      }).join("")
      : codes.map((c) => table(byCode.get(c))).join("");
  el.innerHTML = `
    <div class="hero"><div class="eyebrow">${esc(seasonOf(codes, w, d.season_label) || d.season_label || "")}</div><h1>Tables</h1>
      <p class="lede">${one ? "The league as it stands" : "Each league as it stands"}, and how strong the model rates each club.</p></div>
    ${picker(w, choice, "tables")}${quick(w, choice, new Set(byCode.keys()))}
    <div id="table-out">${body}</div>`;
}

// ---------------------------------------------------------------- record

// A filled ring for one share: how much of the circle the hit rate covers.
function ring(share, colour = "var(--gain)") {
  if (share == null) return "";
  const r = 32, c = 2 * Math.PI * r, on = Math.max(0, Math.min(1, share)) * c;
  return `<svg class="ring" viewBox="0 0 76 76" aria-hidden="true">
    <circle cx="38" cy="38" r="${r}" fill="none" stroke="var(--line)" stroke-width="8"/>
    <circle cx="38" cy="38" r="${r}" fill="none" stroke="${colour}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${on.toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 38 38)"/></svg>`;
}

// The live tiles for one tally (soccer_log.record's counts): pooled, a
// group's or a league's own - never one read off another. t null: the file
// keeps no such tally, so every number is missing.
function liveTiles(t, what) {
  if (!t) return `<div class="tile wide"><div class="k">Live so far</div><div class="v">—</div>
      <div class="s">not counted per ${esc(what)} in this file</div></div>`;
  return `<div class="tile wide"><div class="k">Live so far</div><div class="v">${t.settled ? `${esc(t.correct)}-${esc(t.settled - t.correct)}` : "—"}</div>
        <div class="s">${t.settled ? `${pct(t.correct / t.settled, 1)}${t.provisional ? ` · <span title="${esc(PROV_TIP)}">${esc(t.provisional)} provisional</span>` : ""} · ${esc(t.pending ?? 0)} still to play · logged before kickoff` : `no settled picks yet · ${esc(t.pending ?? 0)} logged`}</div></div>
      <div class="tile"><div class="k">Over / under 2.5</div><div class="v">${t.ou_settled ? `${esc(t.ou_correct)}-${esc(t.ou_settled - t.ou_correct)}` : "—"}</div>
        <div class="s">live, the side the chance favoured</div></div>
      <div class="tile"><div class="k">Both score</div><div class="v">${t.btts_settled ? `${esc(t.btts_correct)}-${esc(t.btts_settled - t.btts_correct)}` : "—"}</div>
        <div class="s">${t.btts_settled ? "live, the side the chance favoured" : "logged with each pick; none settled yet"}</div></div>`;
}

// How often picks in each band won, for the groups given: matches.json's
// bands (soccer_leagues.BANDS, measured per group on its confirmation
// period), or, in a file from before every league, the top five's measure.
// There is no league's own band rate: a league reads its group's.
function bandsCard(md, groups, w, league) {
  if (!md) return "";
  const rows = (cell) => Object.keys(BAND_WORD).map((band) => `<tr><td>${esc(BAND_WORD[band])}<span class="mini" style="display:block">${
    esc(md.bands ? floorRange(md.band_floors, band) : bandRange(measured(md), band))}</span></td>${cell(band)}</tr>`).join("");
  const rate = (r) => `<td class="n">${pct(r && r.hit_rate, 1)}${r && r.picks != null ? `<span class="mini">${esc(r.picks.toLocaleString())} picks</span>` : ""}</td>`;
  let tables;
  if (md.bands) {
    tables = groups.filter((g) => md.bands[g]).map((g) => `${groups.length > 1 || league ? `<div class="eyebrow" style="margin:12px 0 4px">${
      esc((w.groups[g] || {}).name || g)} · ${esc(md.bands[g].period || "")}</div>` : ""}
      <div class="scroll"><table><thead><tr><th></th><th class="n">Picks won</th><th class="n">Model's own</th></tr></thead><tbody>${
      rows((band) => rate(groupBand(md, g, "pick", band)) + rate(groupBand(md, g, "model", band)))}</tbody></table></div>`).join("");
  } else {
    const ms = measured(md);
    if (!ms || !groups.includes("top")) return "";
    tables = `<div class="scroll"><table><thead><tr><th></th><th class="n">Picks won</th><th class="n">Model's own</th></tr></thead><tbody>${
      rows((band) => ["pick", "model"].map((basis) => `<td class="n">${pct(bandRate(ms, basis, band), 1)}</td>`).join(""))}</tbody></table></div>`;
  }
  if (!tables) return "";
  const period = md.bands ? [...new Set(groups.map((g) => (md.bands[g] || {}).period).filter(Boolean))].join(", ") : (measured(md) || {}).conf;
  return DESK.cardFold("How sure the pick is", `${esc(period || "")}: how often picks in each band won${
    league ? ` - measured over its group, ${esc(GROUP_PROSE[w.group[league]] || w.group[league])}, not over the league alone` : ""}.`,
    `${tables}<div class="mini">The band is set by the pick's chance. "Picks won": the pick as shown, the market's where priced.
      "Model's own": the model's own picks at the same edges - the rate a pick made by the model alone (no price yet) is shown with.</div>`);
}

// The Record tab for the picker's choice: every league pooled, a group or one
// league, each read off its own measurement - soccer_backtest.record's rows
// are soccer_leagues.SHIPPED to the digit, each league on its own periods.
// The top five's tables (the over/under, betting the disagreements,
// calibration) were measured on the top five only: shown, labelled, only
// where the choice includes the top five, and said to be missing elsewhere.
function renderRecord(d, md) {
  const b = d.backtest || {};
  const w = world(md, b.by_league);
  const choice = choiceOf(w);
  const kind = kindOf(choice);
  const g = kind === "group" ? choice.slice(2) : kind === "league" ? w.group[choice] : null;
  // Written since every league came in: groups and the pooled record.
  const split = !!(b.by_group || b.pooled);
  const per = b.periods || { dev: [2006, 2015], confirm: [2016, 2025] };
  const dev = `${per.dev[0]}-${per.dev[1]}`, conf = `${per.confirm[0]}-${per.confirm[1]}`;
  const T = d.track_record || {};
  // The five's tables, as the tab has always shown them: the top five, or
  // everything in a file from before every league (which is the five).
  const topScope = kind === "group" ? g === "top" : kind === "all" ? !split : false;
  const t = kind === "all" ? T
    : kind === "group" ? (T.by_group ? T.by_group[g] || {} : (!split && g === "top" ? T : null))
    : (T.by_league ? T.by_league[choice] || {} : null);
  const W = b.result || {};
  const scoped = split ? " · top five" : "";
  let hero, winners;
  if (topScope || (kind === "all" && !b.pooled)) {
    const bl = (W.blend || {})[conf] || {};
    hero = `<div class="tile hero-tile">${ring(bl.accuracy)}
        <div><div class="k">Picks right, ${esc(conf)}</div><div class="v">${pct(bl.accuracy, 1)}</div>
        <div class="s">of ${bl.matches == null ? "—" : bl.matches.toLocaleString()} matches, each picked only from the ones before · betting favourite ${pct(bl.market_accuracy, 1)}</div></div></div>`;
    const wrow = (key, label) => {
      const r = W[key] || {};
      const a = r[dev] || {}, c = r[conf] || {};
      return `<tr><td>${esc(label)}</td><td class="n">${pct(a.accuracy, 1)}</td><td class="n">${pct(c.accuracy, 1)}</td><td class="n">${num(c.log_loss, 4)}</td></tr>`;
    };
    const mk = (W.blend || {})[conf] || {}, mkDev = (W.blend || {})[dev] || {};
    winners = `<div class="card"><h2>Picking results${scoped}</h2>
    <div class="scroll"><table><thead><tr><th></th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th><th class="n">Log loss</th></tr></thead>
    <tbody>${wrow("model", "Model alone")}${wrow("blend", d.basis === "market" ? (split ? "Pick (each league's market)" : "Pick (market, recalibrated)") : "Pick (blend)")}
    <tr><td>Betting favourite</td><td class="n">${pct(mkDev.market_accuracy, 1)}</td><td class="n">${pct(mk.market_accuracy, 1)}</td><td class="n">${num(mk.market_log_loss, 4)}</td></tr></tbody></table></div>
    ${fold(`Every match predicted only from matches on earlier days. ${dev} is where choices were made; ${conf} was left untouched to judge them. Three results, so log loss for three even chances is 1.099; lower is better.`)}</div>`;
  } else {
    // A group's, a league's or every league's own record: {periods?, dev,
    // confirm}; a league row from before every league carries only the
    // confirmation period's numbers, at its top level.
    const src = kind === "all" ? b.pooled : kind === "group" ? (b.by_group || {})[g] : (b.by_league || []).find((r) => r.code === choice);
    const C = (src && (src.confirm || (src.accuracy !== undefined ? src : null))) || {};
    const D = (src && src.dev) || {};
    const span = (p) => p ? `${p[0]}-${p[1]}` : null;
    const pl = src && src.periods ? { dev: span(src.periods.dev), conf: span(src.periods.confirm) }
      : kind === "all" ? { dev: "Earlier", conf: "Later" } : { dev, conf };
    const x = w.meta[choice] || {};
    const pickLabel = kind === "league" ? (x.pick === "raw" ? "Pick (market, cut removed)" : x.pick === "recalibrated" ? "Pick (market, recalibrated)" : "Pick")
      : "Pick (each league's market)";
    hero = `<div class="tile hero-tile">${ring(C.accuracy)}
        <div><div class="k">Picks right, ${esc(kind === "all" ? "every league" : pl.conf)}</div><div class="v">${pct(C.accuracy, 1)}</div>
        <div class="s">${src ? `of ${C.matches == null ? "—" : C.matches.toLocaleString()} matches${kind === "all" ? ", each league on its own later seasons" : ""}, each picked only from the ones before · betting favourite ${pct(C.market_accuracy, 1)}`
          : `not measured for ${esc(choiceName(choice, w, true))} in this file`}</div></div></div>`;
    const r3 = (label, acc, ll) => `<tr><td>${esc(label)}</td><td class="n">${pct(D[acc], 1)}</td><td class="n">${pct(C[acc], 1)}</td><td class="n">${num(C[ll], 4)}</td></tr>`;
    winners = `<div class="card"><h2>Picking results</h2>
    <div class="scroll"><table><thead><tr><th></th><th class="n">${esc(pl.dev)}</th><th class="n">${esc(pl.conf)}</th><th class="n">Log loss</th></tr></thead>
    <tbody>${r3("Model alone", "model_accuracy", "model_log_loss")}${r3(pickLabel, "accuracy", "log_loss")}${r3("Betting favourite", "market_accuracy", "market_log_loss")}
    <tr><td>${esc(kind === "league" ? "The league's own rates" : "The leagues' own rates")}</td><td class="n">—</td><td class="n">—</td><td class="n">${num(C.base_log_loss, 4)}</td></tr></tbody></table></div>
    ${fold(`Every match predicted only from matches on earlier days, ${kind === "all" ? "each league on its own seasons: the earlier ones are where choices were made, the later ones were left untouched to judge them"
      : `${pl.dev} is where choices were made; ${pl.conf} was left untouched to judge them`}. Each league is measured against the price its history has (closing prices where it has no pre-match ones, as its caveats say). "Own rates": the league's home, draw and away frequencies from earlier seasons. Three results, so log loss for three even chances is 1.099; lower is better.`)}</div>`;
  }
  const tiles = `<div class="tiles">${hero}${liveTiles(t, kind === "league" ? "league" : "group")}</div>`;

  // Each league's row on its own periods; for every league, under its group.
  const inScope = (r) => kind === "all" || (kind === "group" && (w.group[r.code] || r.group || "top") === g);
  const lrow = (r) => `<tr><td>${esc(r.label || r.name)}${r.periods && split ? `<span class="mini" style="display:block">${esc(`${r.periods.confirm[0]}-${r.periods.confirm[1]}`)}</span>` : ""}</td><td class="n">${esc(r.matches)}</td>
      <td class="n">${pct(r.accuracy, 1)}</td><td class="n">${pct(r.market_accuracy, 1)}</td><td class="n">${num(r.log_loss, 3)}<span class="mini">market ${num(r.market_log_loss, 3)}</span></td></tr>`;
  const mine = kind === "league" ? [] : (b.by_league || []).filter(inScope);
  const lgroups = kind === "all" && split ? Object.keys(w.groups) : [null];
  const leagues = lgroups.map((gg) => {
    const rs = gg ? mine.filter((r) => (w.group[r.code] || r.group) === gg) : mine;
    return rs.length ? `${gg ? `<tr class="grp"><td colspan="5">${esc(w.groups[gg].name)}</td></tr>` : ""}${rs.map(lrow).join("")}` : "";
  }).join("");
  const leagueCard = leagues ? DESK.cardFold("By league", split ? "Each league on its own later seasons: its picks against its betting favourite."
    : `${esc(conf)}: each league's picks against its betting favourite.`,
    `<div class="scroll"><table><thead><tr><th>League</th><th class="n">Matches</th><th class="n">Pick</th><th class="n">Favourite</th><th class="n">Log loss</th></tr></thead>
    <tbody>${leagues}</tbody></table></div>`) : "";

  const withTop = kind === "all" || g === "top";
  const ou = withTop ? b.over_2_5 || {} : {};
  const ouCard = ou[conf] ? DESK.cardFold(`Over / under 2.5 goals${scoped}`,
    `${esc(conf)}: the model's log loss ${num(ou[conf].log_loss, 4)} against the market's ${num(ou[conf].market_log_loss, 4)} (lower is better) - so the chance shown is the market's.`,
    `<div class="scroll"><table><thead><tr><th>Log loss</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead><tbody>
      <tr><td>Model alone</td><td class="n">${num((ou[dev] || {}).log_loss, 4)}</td><td class="n">${num(ou[conf].log_loss, 4)}</td></tr>
      <tr><td>Market (shown)</td><td class="n">${num((ou[dev] || {}).market_log_loss, 4)}</td><td class="n">${num(ou[conf].market_log_loss, 4)}</td></tr>
    </tbody></table></div>`) : "";

  const v = withTop ? b.value || {} : {};
  const vrows = Object.entries(v.edges || {}).map(([edge, r]) => {
    const cell = (x) => !x || x.roi == null ? `<td class="n">—</td>`
      : `<td class="n ${x.roi >= 0 ? "gain" : "loss"}">${signed(x.roi * 100, 1)}%<span class="mini">${esc(x.bets)} bets</span></td>`;
    return `<tr><td class="n">${pct(Number(edge))}</td>${cell(r[dev])}${cell(r[conf])}</tr>`;
  }).join("");
  const valueCard = vrows ? DESK.cardFold(`Betting the disagreements${scoped}`, v.enabled_at == null
      ? "It did not make money in both periods at any edge, so no match is ever labelled a bet."
      : `Profitable in both periods from a ${esc(pct(v.enabled_at))} edge.`,
    `<div class="scroll"><table><thead><tr><th class="n">Edge</th><th class="n">${esc(dev)}</th><th class="n">${esc(conf)}</th></tr></thead>
    <tbody>${vrows}</tbody></table></div>
    <div class="mini">One unit at the best posted price on every result the model alone rated at least this far above the
      market with the bookmakers' cut removed (return per bet, bets beside).</div>`) : "";

  const cal = (withTop ? b.calibration || [] : []).map((r) =>
    `<tr><td class="n">${pct(r.lo)}-${pct(r.hi)}</td><td class="n">${esc(r.picks)}</td><td class="n">${pct(r.predicted, 1)}</td><td class="n">${pct(r.observed, 1)}</td></tr>`).join("");
  const calCard = cal ? DESK.cardFold(`Calibration${scoped}`, `${esc(conf)}: the pick's chance, against how often the pick happened.`,
    `<div class="scroll"><table><thead><tr><th class="n">Pick's chance</th><th class="n">Picks</th><th class="n">Said</th><th class="n">Happened</th></tr></thead>
    <tbody>${cal}</tbody></table></div>`) : "";
  // Measured on the top five only: elsewhere they are missing, and say so.
  const topOnly = withTop || !split ? "" : `<div class="note">The over/under, betting-the-disagreements and calibration tables
    were measured on the top five only. They are not ${esc(choiceName(choice, w, true))}'s numbers, so they are not shown here.</div>`;

  // Week by week is every league pooled: shown with every league, or with
  // a file from before every league (the five).
  const tw = kind === "all" || (!split && g === "top") ? T : {};
  const weeks = (tw.by_week || []).slice(-8).reverse().map((wk) =>
    `<li><span><b>${esc(wk.label)}</b>${wk.provisional ? ` <span class="mini">${esc(wk.provisional)} provisional</span>` : ""}</span><span class="num ${wk.correct / wk.settled >= 0.5 ? "good" : "warn"}">${esc(wk.correct)}/${esc(wk.settled)}</span></li>`).join("");
  const weekCard = weeks ? `<div class="card"><h2>Live, week by week${split ? " · every league" : ""}</h2><ul class="rank">${weeks}</ul>
    <div class="mini">Picks logged before each kickoff, graded against the full-time result. A provisional grade uses the free
      feeds' result until football-data.co.uk posts its own, which confirms or corrects it${
      tw.corrected ? ` (${esc(plural(tw.corrected, "pick", "picks"))} corrected so far)` : ""}.</div></div>` : "";
  const liveNote = t && t.settled ? "" : `<div class="note">No settled picks yet${kind === "all" ? "" : ` in ${esc(choiceName(choice, w, true))}`}: the live record starts with the first matches
    picked before kickoff, and grows every weekend. The tables below are the walk-forward backtest.</div>`;
  const bandGroups = kind === "all" ? Object.keys(w.groups) : g ? [g] : [];
  const bands = split || (md && md.bands) ? bandsCard(md, bandGroups, w, kind === "league" ? choice : null) : "";
  // The first season replayed: each league's own development period.
  const mineRows = kind === "league" ? (b.by_league || []).filter((r) => r.code === choice) : (b.by_league || []).filter(inScope);
  const firsts = mineRows.map((r) => (r.periods && r.periods.dev ? r.periods.dev[0] : null)).filter((x) => x != null);
  const first = topScope || !firsts.length ? per.dev[0] : Math.min(...firsts);

  return `<div class="hero"><div class="eyebrow">Measured, not claimed</div><h1>Track record</h1>
      <p class="lede">Live picks are graded against full-time results; the backtest replays every season since ${esc(first)}.</p></div>
    ${picker(w, choice, "record")}${quick(w, choice, new Set(w.order))}
    ${tiles}${liveNote}${winners}${weekCard}${leagueCard}${bands}${ouCard}${valueCard}${calCard}${topOnly}`;
}

// ---------------------------------------------------------------- method

// The goals and scores numbers and the bands, as this run's walk-forward
// measured them: log loss against each league's own earlier rates and the
// model's own score chances, how often the likeliest score happened, and the
// bands' hit rates - the market's picks and the model's own, same edges.
function scoresCard(d, failed) {
  const ms = measured(d);
  if (!ms) return `<div class="card"><h2>Goals and scores, measured</h2><div class="mini">${failed
    ? "Could not load the measured table - open Matches, then come back." : "Loading the measured table…"}</div></div>`;
  const P = ms.sm.periods, c = P[ms.conf] || {}, v = P[ms.dev] || {};
  const cell = (a, b, d0 = 4) => `<td class="n">${num(a, d0)}<span class="mini">${num(b, d0)}</span></td>`;
  const ll = (t, label, d0 = 4) => {
    const x = (r, k) => (((r || {})[t] || {})[k] || {}).log_loss;
    return `<tr><td>${esc(label)}</td>${cell(x(c, "shown"), x(v, "shown"), d0)}${cell(x(c, "model"), x(v, "model"), d0)}${cell(x(c, "base"), x(v, "base"), d0)}</tr>`;
  };
  const ex = (r, k) => ((r.exact || {})[k] || {});
  const hit = (k, label) => `<tr><td>${esc(label)}</td>${["shown", "model", "base"].map((s) =>
    `<td class="n">${pct(ex(c, s)[k], 1)}<span class="mini">${pct(ex(v, s)[k], 1)}</span></td>`).join("")}</tr>`;
  const bands = Object.keys(BAND_WORD).map((b) => `<tr><td>${esc(BAND_WORD[b])}<span class="mini" style="display:block">${esc(bandRange(ms, b))}</span></td>${
    ["pick", "model"].map((basis) => `<td class="n">${pct(bandRate(ms, basis, b), 1)}<span class="mini">${pct(bandRate(ms, basis, b, ms.dev), 1)}</span></td>`).join("")}</tr>`).join("");
  const head = (cols) => `<thead><tr><th></th>${cols.map((h) => `<th class="n">${h}</th>`).join("")}</tr></thead>`;
  const m = c.model || {};
  const fb = [m.no_price ? plural(m.no_price, "match with no price", "matches with no price") : "",
    m.no_over_under ? plural(m.no_over_under, "with only the home/draw/away price", "with only the home/draw/away price") : "",
    m.fit ? `${m.fit} whose prices fit no score matrix within ${Math.round(ms.sm.fit_max_miss * 100)} points` : ""].filter(Boolean);
  const btts = c.btts || {};
  const side = btts.side || {}, mx = c.matrix_1x2 || {};
  return `<div class="card"><h2>Goals and scores, measured</h2>
    <div class="mini">Every top-league match ${esc(ms.dev)} and ${esc(ms.conf)}, each from matches before it. Top number
      ${esc(ms.conf)}, the one under it ${esc(ms.dev)}. <b>Shown</b>: the numbers on the cards; <b>Model</b>: the model's own score
      chances; <b>Leagues</b>: each league's own rates from earlier seasons.${d && d.groups ? ` These are the top five's, pooled: every
      other league shows only the numbers that beat its own rates in both of its periods (Every league, below).` : ""}</div>
    <div class="eyebrow" style="margin:12px 0 4px">Log loss - lower is better</div>
    <div class="scroll"><table>${head(["Shown", "Model", "Leagues"])}<tbody>
      ${ll("btts", "Both score")}${ll("o15", "Over 1.5")}${ll("o35", "Over 3.5")}${ll("exact", "Exact score", 3)}</tbody></table></div>
    <div class="eyebrow" style="margin:12px 0 4px">How often it happened</div>
    <div class="scroll"><table>${head(["Shown", "Model", "Leagues"])}<tbody>
      ${hit("top1_hit", "The likeliest score")}${hit("top3_hit", "One of the three likeliest")}</tbody></table></div>
    <div class="eyebrow" style="margin:12px 0 4px">How sure the pick is</div>
    <div class="scroll"><table>${head(["Picks won", "Model's own"])}<tbody>${bands}</tbody></table></div>
    <div class="mini">The band is set by the pick's chance; the edges were the quartiles of ${esc(ms.dev)}, rounded. "Picks won": the
      pick as shown, the market's where priced. "Model's own": the model's own picks at the same edges - the rate a pick made by the
      model alone (no price yet) is shown with.</div>
    ${fold(`Caveats: both teams to score runs low (${ms.conf}: ${pct((btts.shown || {}).mean_p, 1)} said, ${pct(btts.rate, 1)} happened,
      low in ${btts.deciles_low ?? "—"} of ${btts.deciles ?? "—"} deciles; the model's own ${pct((btts.model || {}).mean_p, 1)}); exact scores are long shots; with no price the numbers are the model's own.`,
      `${d && d.caveats && d.caveats.scores ? `<p style="margin:0">${esc(d.caveats.scores)}</p>` : ""}
      <p style="margin:8px 0 0">In ${esc(ms.conf)} the cards' numbers would have been the market's for ${(c.market ?? 0).toLocaleString()} of
      ${(c.matches ?? 0).toLocaleString()} matches${fb.length ? `, the model's own for ${esc(fb.join(", "))}` : ""}${
        c.flagged ? `; ${esc(c.flagged.toLocaleString())} of the market's were flagged - prices no score matrix fits within ${
          Math.round(ms.sm.fit_max_miss * 100)} points, where the model's own fit them worse still` : ""}.</p>
      ${side.shown != null ? `<p style="margin:8px 0 0">Both score, the side its chance favoured (the Record tab's call): right in
        ${pct(side.shown, 1)} of ${esc(ms.conf)} matches, against ${pct(side.always_yes, 1)} for always "yes" and ${pct(side.base, 1)}
        for the leagues' own rates.</p>` : ""}
      ${mx.log_loss != null ? `<p style="margin:8px 0 0">The "Results" row - home/draw/away read off the market's score chances - is a
        by-product, never the pick: its log loss ${num(mx.log_loss, 4)} against the pick's ${num(mx.pick_log_loss, 4)} (${esc(ms.conf)},
        ${esc((mx.matches ?? 0).toLocaleString())} matches).</p>` : ""}`)}
  </div>`;
}

// Every league the app carries, by group, with what was decided for it and
// why (soccer_leagues: its pick, whether a match with no price gets the
// model's own pick, the numbers it ships from each source and those it
// withholds, the price and periods it was measured on, the feeds that cover
// it) and every caveat it carries - and the measured leagues the app leaves
// out (not_in_app), with theirs. Nothing here is typed in: it is the file's.
const NUMBER_WORD = { btts: "both score", o15: "over 1.5", o25: "over 2.5", o35: "over 3.5", exact: "likeliest scores" };
function leaguesCard(d) {
  const w = world(d);
  if (!w.grouped) return "";
  const nums = (xs) => (xs && xs.length ? xs.map((k) => NUMBER_WORD[k] || k).join(", ") : "none");
  const span = (p) => (p ? `${p[0]}-${p[1]}` : "—");
  const one = (c) => {
    const x = w.meta[c] || {};
    const kv = [
      ["Pick", x.pick === "raw" ? "the market's price with the bookmakers' cut removed" : x.pick === "recalibrated" ? "the market's price, recalibrated" : "—"],
      ["No price yet", x.model_alone === false ? WAITS.toLowerCase() : x.model_alone ? "the model's own pick, labelled" : "—"],
      ["Measured on", `${esc(x.price_measured || "—")} prices · ${esc(span((x.periods || {}).dev))} chose, ${esc(span((x.periods || {}).confirm))} judged`],
      ["Shown", `from the market: ${esc(nums((x.numbers || {}).market))}; from the model alone: ${esc(nums((x.numbers || {}).model))}`],
      ["Not shown", `from the market: ${esc(nums((x.withheld || {}).market))}; from the model alone: ${esc(nums((x.withheld || {}).model))}`],
      ["Feeds", esc((x.feeds || []).length ? x.feeds.join(", ") : "none - fixtures only once football-data.co.uk posts them")],
    ];
    const notes = leagueLines(x);
    return DESK.sub(esc([leagueLabel(c, w), x.country].filter(Boolean).join(" · ")),
      `<dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>${notes.length ? `<div style="margin-top:8px">${paras(notes)}</div>` : ""}${
        x.table_note ? `<p class="mini" style="margin:0">Table: ${esc(x.table_note)}</p>` : ""}`);
  };
  const out = Object.entries(d.not_in_app || {});
  return `<div class="card"><h2>Every league</h2>
    <div class="mini">Each league was measured on its own seasons and decided by rules set before any number was seen:
      its pick, the numbers it shows and the band rates of its group. A number a league did not earn is missing, not guessed.</div>
    ${Object.entries(w.groups).map(([g, x]) => `<div class="eyebrow" style="margin:14px 0 4px">${esc(x.name)}</div>${x.codes.map(one).join("")}`).join("")}
    ${out.length ? `<div class="eyebrow" style="margin:14px 0 4px">Not in the app</div>${out.map(([c, x]) =>
      DESK.sub(esc([x.name || c, x.country].filter(Boolean).join(" · ")), paras(x.caveats || []))).join("")}` : ""}
  </div>`;
}

function methodHtml(d = MATCHES, failed = false) {
  // The pick is the market, recalibrated, unless the data says otherwise -
  // the Method tab can open before Matches has loaded.
  const market = !(d && d.basis === "blend");
  const many = !!(d && d.groups);
  const count = many ? world(d).order.length : 5;
  return `
  <div class="hero"><div class="eyebrow">What this is</div><h1>How it works</h1></div>
  <div class="card"><h2>Reading a pick</h2>${LEGEND}</div>
  <div class="card"><h2>Three results, one pick</h2>
    <ul class="plain">
      <li><b>The model</b> - rates every club's attack and defence from its goals, shots on target and
        shots${many ? " (from its goals alone where its league's history has no shots, as that league says)" : ""}, weighted toward recent matches, and turns two clubs' ratings into the chance of every score.
        Promoted clubs carry their rating up from the second tier with a measured step. Of three models
        built and measured, it came closest to the market.</li>
      <li><b>The pick</b> - ${market && many
        ? `each league's betting market: its chances recalibrated on earlier seasons, or the raw price with the
          bookmakers' cut removed where recalibrating scored worse on the league's own earlier seasons - each
          league says which (Every league, below). Blended with the market, the model added nothing the market
          did not already know. The model's own chances are shown in every opened match, labelled "Model alone".`
        : market
        ? `the betting market's chances, recalibrated on earlier seasons. Blended with the market, the model
          added nothing the market did not already know, so the blend gives it no weight - the market's
          prices are the better forecast, and the app says so. The model's own chances are shown in every
          opened match, labelled "Model alone".`
        : `the model blended with the betting market's prices, with the weight learned from how the two did
          on earlier seasons. The market is very good at soccer, so the pick leans on it heavily.`}</li>
      <li><b>Goals</b> - the chance of three or more goals: the over/under market's, with the bookmakers'
        cut removed; the model's own chance is beside it.</li>
      <li><b>Goals and scores</b> - both teams to score, over 1.5 and 3.5 goals and the three likeliest scores are read off
        one set of chances for every score: the set that best reproduces the pick's home/draw/away chances and the
        over/under price. Where those two are not both posted - or no set of score chances fits them closely - they are the
        model's own, and the match says so. The pick's own chances stay as shown; the scores' own are beside them.</li>
      <li><b>The band</b> - Strong, Solid, Lean or Toss-up, by the pick's chance, shown with how often picks that sure won${
        many ? " in its league's group" : ""}.
        A pick made by the model alone is shown with the rate of the model's own picks.</li>
    </ul></div>
  ${scoresCard(d, failed)}
  ${leaguesCard(d)}
  <div class="card"><h2>Every number is point-in-time</h2>
    <ul class="plain">
      <li>A match is predicted only from matches played on earlier days - the same rule for the backtest and for this weekend.</li>
      <li>Promoted clubs carry their rating up from the second tier, so a club is not a stranger in August.</li>
      <li>A pick is logged before kickoff and frozen when the match starts; the live record grades only those. Where the
        kickoff time is not confirmed, the pick closes at midnight UK time on match day instead.</li>
    </ul></div>
  <div class="card"><h2>Between football-data.co.uk's postings</h2>
    <ul class="plain">
      <li><b>Fixtures</b> - football-data.co.uk posts the weekend's matches with their prices on Friday afternoon and the
        midweek's on Tuesday. Before that, the next rounds come from two free, unofficial feeds, ESPN's scoreboard and
        fixturedownload.com, and are labelled "No price yet": their chances are the model's alone. When the prices land,
        the pick becomes the market's, as long as the match has not kicked off.</li>
      <li><b>Results</b> - the feeds have the score the evening a match is played; football-data.co.uk days later. Until it
        does, the result is provisional and marked so: in the results, the tables and the live record. Its own result
        confirms the grade or corrects it.</li>
      <li><b>The ratings</b> are built from football-data.co.uk's results only, which carry the shots the model reads; the
        app says the date they run through.</li>
    </ul></div>
  <div class="card"><h2>What it does not do</h2>
    <ul class="plain">
      <li>It does not read team news - injuries, suspensions, rotation. The market does, which is one reason the pick leans on it.</li>
      <li>Cup and European matches are not predicted; only the ${count === 5 ? "five" : esc(count)} leagues${many ? " in the picker" : ""}.</li>
      <li>No match is labelled a bet unless betting the model against the price made money in both periods (see Record).</li>
    </ul></div>
  <div class="note">Data: football-data.co.uk (results, match statistics and bookmakers' prices for ${many
    ? "every league in the picker - match statistics only where a league's history has them"
    : "the Premier League, La Liga, Serie A, the Bundesliga, Ligue 1 and their second tiers"}); ESPN's public scoreboard and
    fixturedownload.com (fixtures, kickoff times and provisional scores only - never a price). For entertainment and
    analysis - not betting advice.</div>`;
}

// ---------------------------------------------------------------- start

DESK.init({
  key: "yoursoccer",
  name: "YourSoccer",
  icon: "icons/icon-192.png",
  siblings: [{ name: "YourMMA", href: "../../app/", blurb: "The MMA fight desk" },
    { name: "YourNFL", href: "../../nfl/app/", blurb: "The NFL pick desk" },
    { name: "YourMulti", href: "../../parlays/app/", blurb: "Same-day parlays, 3-5x" }],
  stamp: () => (MATCHES && MATCHES.generated ? `Picks updated ${DESK.ago(MATCHES.generated)}` : "Fetch the latest picks"),
  onReload: () => location.reload(),
});
DESK.tabs(["matches", "tables", "record", "method"], BOOT, "yoursoccer.tab", "matches");
setInterval(stamp, 60000);
