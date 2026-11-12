// YourMulti: two same-day parlays a day from the three apps' picks - UFC,
// NFL and top-five soccer - the likeliest ticket paying 3-5x and the
// underdog ticket (one underdog drawn by lot, plus favourites to 3-5x),
// every leg played that day (US Eastern), for today and the next days with
// tickets, each with its measured record. The engine
// (parlays/engine/build_sameday.py) writes data/sameday.json; this only
// draws it, and still reads an older file with one ticket a day.

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
const esc = DESK.esc;
const icon = DESK.icon;
// A finite number, or null: a missing number stays missing, never 0.
const num = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);
const pct = (x, d = 0) => (num(x) == null ? "—" : `${(x * 100).toFixed(d)}%`);
const chance = (x) => (num(x) == null ? "—" : x < 0.1 ? pct(x, 1) : pct(x));
const mult = (m) => (m == null ? "—" : `x${Number(m).toFixed(2)}`);
const signed = (x, d = 0) => {
  if (num(x) == null) return "—";
  const r = Number(Number(x).toFixed(d));
  return r === 0 ? (0).toFixed(d) : `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(d)}`;
};
// A return as signed whole percent points, "—" when missing (never 0).
const pc = (x) => (num(x) == null ? "—" : `${signed(x * 100)}%`);
const odds = (a) => (a == null ? "—" : a > 0 ? `+${a}` : `−${Math.abs(a)}`);
// Kickoffs are stored in UTC; the phone shows its own time zone.
const kickTime = (iso) => {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? "" : t.toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
};
// A ticket's day is an Eastern calendar date (YYYY-MM-DD), named as such.
const dayName = (d) => {
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? (d || "") : t.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric", timeZone: "UTC" });
};
const fold = (text, extra = "") => !text ? "" : `<details class="fold">
    <summary><span class="ic">${icon("info")}</span><span class="clip">${esc(text)}</span><span class="more">More</span></summary>
    ${extra ? `<div class="fold-body">${extra}</div>` : ""}</details>`;

// The first of these keys a record carries as a number. The underdog's
// measured block is written by the backtest, separately; read defensively,
// and a number it does not carry is shown as missing.
const first = (s, keys) => {
  if (!s || typeof s !== "object") return null;
  for (const k of keys) if (num(s[k]) != null) return s[k];
  return null;
};
// The price-implied return of the same tickets: their legs' no-vig chances
// multiplied, times the payout, less 1 - what the prices expected.
const implied = (s) => first(s, ["implied_roi", "price_implied_roi", "roi_implied", "market_implied_roi", "ev_market", "ev_prop"]);

async function load(name) {
  const res = await fetch(`data/${name}.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name}.json — ${res.status}`);
  return res.json();
}

function fail(el, error, what, retry) {
  el.innerHTML = `<div class="note err"><b>Could not load ${esc(what)}.</b> Check the connection and try again.${
    retry ? `<div style="margin-top:10px"><button class="btn small ghost" type="button" data-retry>${icon("refresh")} Try again</button></div>` : ""}</div>
    ${fold(`${error.message}. The runner writes this file; to build it locally, run python parlays/engine/build_sameday.py.`)}`;
  const b = el.querySelector("[data-retry]");
  if (b) b.addEventListener("click", () => retry());
}

let DATA = null;
let DATA_PROMISE = null;
const data = () => (DATA_PROMISE = DATA_PROMISE || load("sameday").then((d) => { DATA = d; stamp(); return d; })
  .catch((e) => { DATA_PROMISE = null; throw e; }));

function stamp() {
  const s = $("stamp");
  if (!s || !DATA) return;
  s.textContent = DATA.generated ? `Updated ${DESK.ago(DATA.generated)}` : "Same-day parlays";
}

// The days in the shape the engine writes now; an older sameday.json (one
// ticket a day, no underdog) still reads, with no underdog ticket shown.
const daysOf = (d) => (Array.isArray(d.days) ? d.days
  : (d.tickets || []).map((t, i) => ({ day: t.day, today: i === 0, likeliest: t, underdog: null, cannot_both_win: false })));
// The measured blocks; an older file has the likeliest's at the top.
const measuredOf = (d) => {
  const m = d.measured;
  if (!m) return { lk: null, dog: null, dogWhy: null };
  return { lk: m.likeliest || m, dog: m.underdog || null, dogWhy: m.underdog_missing_reason || null };
};

// What a ticket is now: still buildable, locked at its first start, or settled.
const STATUS = {
  open: ["Open", "Rebuilt at the latest prices until its first leg starts."],
  locked: ["Locked", "Its first leg has started: this ticket stays as it is until it settles."],
  pending: ["Locked", "Its first leg has started: this ticket stays as it is until it settles."],
  won: ["Won", ""], lost: ["Lost", ""], void: ["Void", "Every leg was void; the stake comes back."],
};
const OUTCOME = { won: "won", lost: "lost", void: "void" };
// The lottery, in one plain line (sameday.py's DOG and LOTTERY).
const LOTTERY = "One underdog drawn by lot - likelier ones a little more often - plus favourites to 3-5x. The draw is fixed for the day.";
const NAME = { likeliest: "Likeliest ticket", underdog: "Underdog ticket" };

const legRow = (l) => `<li><span><span class="tag">${esc(l.sport)}</span>${l.dog ? ` <span class="tag accent">Underdog</span>` : ""}
    <b>${esc(l.pick)}</b> <span class="mini">${esc(l.event)}</span><br>
    <span class="mini">${esc(kickTime(l.kickoff))}${l.outcome && OUTCOME[l.outcome] ? ` · ${OUTCOME[l.outcome]}` : ""} · ${esc(l.reason)}${
      l.dog && l.price_source ? ` · priced at the ${esc(l.price_source)}` : ""}</span></span>
    <span class="end num">${odds(l.american)}<br><span class="mini">${pct(l.p)}</span></span></li>`;

function ticketPanel(t, kind, today) {
  if (!t) return "";
  const name = NAME[kind];
  if (!t.legs || !t.legs.length) {
    return `<div class="panel"><div class="eyebrow">${name} · <span class="tag flat">None</span></div>
      <div class="mini">${esc(t.missing_reason || "No ticket for this day.")}</div></div>`;
  }
  const [word, line] = STATUS[t.status] || [t.status, ""];
  const settled = t.status === "won" || t.status === "lost" || t.status === "void";
  const dog = kind === "underdog";
  return `<div class="panel">
      <div class="eyebrow">${name} · ${esc((t.sports || []).join(" + "))} · ${t.legs.length} leg${t.legs.length === 1 ? "" : "s"} ·
        <span class="tag${t.status === "lost" ? " warn" : ""}">${esc(word)}</span></div>
      ${dog ? `<div class="mini">${esc(LOTTERY)}</div>` : ""}
      <div class="stat3">
        <div><div class="k">Pays</div><div class="v">${mult(t.multiplier)}</div></div>
        <div title="The chance every leg lands: each pick's chance, multiplied"><div class="k">Chance</div><div class="v">${chance(t.p)}</div></div>
        <div title="The same from the market's prices, the bookmaker's cut removed"><div class="k">Market</div><div class="v">${chance(t.p_market)}</div></div>
      </div>
      ${dog && t.lands_1_in ? `<div class="mini num">Lands about 1 in ${esc(String(t.lands_1_in))} by the prices.</div>` : ""}
      <ul class="list">${t.legs.map(legRow).join("")}</ul>
      ${dog && t.only_one ? `<div class="mini">Only one underdog fit ${today ? "today" : "on this day"}.</div>` : ""}
      ${dog && t.price_note ? `<div class="alert doubtful">${esc(t.price_note)}</div>` : ""}
      ${settled && t.profit != null ? `<div class="mini num">Paid ${mult(t.paid)} · ${signed(t.profit, 2)} units on 1</div>` : ""}
      ${line ? `<div class="mini">${esc(line)}</div>` : ""}
    </div>`;
}

const takeOf = (x) => ["likeliest", "underdog"].filter((k) => x[k])
  .map((k) => `${k === "likeliest" ? "Likeliest" : "Underdog"} ${x[k].legs && x[k].legs.length ? mult(x[k].multiplier) : "none"}`).join(" · ");

function dayCard(x, open) {
  const title = x.today ? `Today · ${dayName(x.day)}` : dayName(x.day);
  const body = `${ticketPanel(x.likeliest, "likeliest", x.today)}${ticketPanel(x.underdog, "underdog", x.today)}
    ${x.cannot_both_win ? `<div class="alert doubtful"><b>These two cannot both win.</b> The underdog is the other side of a leg on the likeliest ticket.</div>` : ""}`;
  if (open) return `<div class="card"><h2>${esc(title)}</h2>${body}</div>`;
  return DESK.cardFold(esc(title), esc(takeOf(x)), body);
}

function recordLine(s, label) {
  if (!s || !s.tickets) return `${label}: not measured`;
  const iv = num(s.roi_low) != null && num(s.roi_high) != null ? ` (${pc(s.roi_low)} to ${pc(s.roi_high)})` : "";
  return `${label}: ${s.tickets.toLocaleString()} tickets, ${(s.wins || 0).toLocaleString()} won (${pct(s.hit_rate)}), ${
    pc(s.roi)} a ticket${iv}; by the prices ${pc(implied(s))}`;
}

const yearsLabel = (p) => (Array.isArray(p) && p.length === 2 ? `${String(p[0]).slice(0, 4)}-${String(p[1]).slice(0, 4)}` : "");
const PERIODS = [["last_3_years", "Last 3 years"], ["first", "First years"], ["all", "All"]];
const notMeasured = (why) => `<div><span class="tag warn">Not measured yet</span></div>${why ? `<div class="mini">${esc(why)}</div>` : ""}`;

// The underdog's live-salt record: the headline, each period on its own line.
function dogLines(dog, keys = PERIODS.map(([k]) => k)) {
  const live = (dog && dog.live) || {};
  const per = (dog && dog.periods) || {};
  return PERIODS.filter(([k]) => live[k] && keys.includes(k)).map(([k, label]) =>
    `<div class="mini num">${esc(recordLine(live[k], `${label}${per[k] ? `, ${yearsLabel(per[k])}` : ""}`))}</div>`).join("");
}

// One period over the 1,000 simulation draws (sameday_backtest's salt
// stats): the mean with its day-bootstrap interval, the price-implied
// return, and the draws' 5th-95th percentile - which is NOT a confidence
// interval: the games and results are the same, only the lot changes.
function saltLine(s, label) {
  if (!s || typeof s !== "object" || !s.tickets_per_salt) return "";
  const n = num(s.salts) || 1000;
  const sp = (s.spread && s.spread.roi) || {};
  const rk = s.live_rank || {};
  const bits = [`Mean over ${n.toLocaleString()} draws ${pc(s.roi)} a ticket${
    num(s.roi_low) != null && num(s.roi_high) != null ? ` (${pc(s.roi_low)} to ${pc(s.roi_high)})` : ""}; by the prices ${pc(implied(s))}.`,
    `5th to 95th draw - how much the draw alone moves it, same games: ${pc(sp.p5)} to ${pc(sp.p95)}${num(sp.p50) == null ? "" : `, middle ${pc(sp.p50)}`}.`];
  if (num(rk.roi_below) != null && num(rk.of)) bits.push(`The live draw did better than ${rk.roi_below.toLocaleString()} of ${rk.of.toLocaleString()}.`);
  return `<li><span><b>${esc(label)}</b>${bits.map((x) => `<br><span class="mini num">${esc(x)}</span>`).join("")}</span></li>`;
}

// A paired difference, underdog minus likeliest (or lottery minus
// market-only), with its interval.
const diffLine = (s, what) => !s || num(s.roi_diff) == null ? "not measured" : `${(s.days || 0).toLocaleString()} days: ${what} ${
  pc(s.roi_diff)} a ticket${num(s.roi_diff_low) != null && num(s.roi_diff_high) != null ? ` (${pc(s.roi_diff_low)} to ${pc(s.roi_diff_high)})` : ""}`;

// The report-only rows beside the lottery: none of them can replace it in v1.
const REPORT = [
  ["gap", "Largest model gap first", "The underdog the model rates furthest above the market. In soccer that gap is the recalibration curve, not a model opinion - never a sign of value.",
    (r) => [recordLine(r.all, "All")]],
  ["likeliest_dog", "Likeliest underdog first", "The underdog with the highest chance, no lottery.",
    (r) => [recordLine(r.all, "All")]],
  ["market_only", "Market-only lottery", "The same draw with the market's chances alone: what the ticket would be with no model.",
    (r) => [recordLine(r.live && r.live.all, "All"), diffLine(r.paired_with_lottery && r.paired_with_lottery.all, "the lottery minus this, over the draws,")]],
  ["paired", "Beside the likeliest ticket", "The two on the days both exist, paired by day.",
    (r) => [diffLine(r.all, "the underdog minus the likeliest")]],
  ["both", "Both tickets held", "1 unit on each, on the days both exist.",
    (r) => { const b = r.all; return [!b || !b.days ? "not measured" : `${b.days.toLocaleString()} days: ${pc(b.roi)} a unit${
      num(b.roi_low) != null && num(b.roi_high) != null ? ` (${pc(b.roi_low)} to ${pc(b.roi_high)})` : ""}; both won on ${
      (b.both_won || 0).toLocaleString()}, and on ${(b.cannot_both_win || 0).toLocaleString()} they could not both win`]; }],
];

const reportRow = ([key, label, note, lines], r) => `<li><span><b>${esc(label)}</b><br><span class="mini">${esc(note)}</span>${
  (r && typeof r === "object" ? lines(r) : ["not measured"]).map((l) => `<br><span class="mini num">${esc(l)}</span>`).join("")}</span></li>`;

// The underdog's descriptive splits, titled.
const DOG_SPLITS = {
  by_sport: "Underdog ticket by the underdog's sport", by_price: "Underdog ticket by the underdog's price",
  by_year: "Underdog ticket by year", by_legs: "Underdog ticket by legs",
  dogs_by_price: "Every market underdog as a single bet, by price",
};

const BOOT = {
  async today() {
    const el = $("view-today");
    el.innerHTML = DESK.skeleton(3);
    let d;
    try { d = await data(); } catch (e) { return fail(el, e, "today's tickets", () => BOOT.today()); }
    const { lk, dog, dogWhy } = measuredOf(d);
    const days = daysOf(d);
    // Today and the next day with a ticket open; later days folded to their payouts.
    const next = days.find((x) => !x.today && ["likeliest", "underdog"].some((k) => x[k] && x[k].legs && x[k].legs.length));
    const measured = `
      <div class="mini"><b>Likeliest ticket</b></div>
      ${lk ? `<div class="mini num">${esc(recordLine(lk.rule.last_3_years, "Last 3 years"))}</div>
         <div class="mini num">${esc(recordLine(lk.rule.all, `All, ${yearsLabel(lk.periods && lk.periods.all)}`))}</div>`
        : notMeasured()}
      <div class="mini"><b>Underdog ticket</b>, at the live draw</div>
      ${dog ? dogLines(dog, ["last_3_years", "all"]) : notMeasured(lk ? dogWhy : null)}`;
    el.innerHTML = `
      ${DESK.banner()}
      <div class="note"><b>The market is hard to beat.</b> Two tickets a day from the day's picks: the likeliest one paying
        3-5x, and an underdog ticket for a bigger payout. Neither is a bet expected to profit: every leg pays the bookmaker's margin.</div>
      ${days.map((x) => dayCard(x, x.today || x === next)).join("") || `<div class="card"><div class="mini">No ticket yet.</div></div>`}
      <div class="card"><h2>What tickets like these returned</h2>${measured}
        ${fold((d.caveats || [])[0], (d.caveats || []).slice(1).map((c) => `<p style="margin:0">${esc(c)}</p>`).join(""))}</div>`;
  },

  async record() {
    const el = $("view-record");
    el.innerHTML = DESK.skeleton(3);
    let d;
    try { d = await data(); } catch (e) { return fail(el, e, "the record", () => BOOT.record()); }
    const recs = d.records || { likeliest: d.record || {} };
    const liveBlock = (kind) => {
      const r = recs[kind] || {};
      const by = implied(r);
      return `<div class="panel"><div class="eyebrow">${NAME[kind]}</div>${r.settled
        ? `<div class="stat3">
            <div><div class="k">Settled</div><div class="v">${r.settled}</div></div>
            <div><div class="k">Won</div><div class="v">${r.won}</div></div>
            <div><div class="k">Units</div><div class="v">${signed(r.units, 2)}</div></div></div>
           <div class="mini num">${pc(r.roi)} a ticket; by the prices ${pc(by)}</div>
           <ul class="list">${(r.last || []).slice().reverse().map((t) => `<li><span>${esc(dayName(t.day))}<br>
             <span class="mini">${esc((t.sports || []).join(" + "))}</span></span>
             <span class="end num">${mult(t.multiplier)}<br><span class="mini">${esc(t.status)}</span></span></li>`).join("")}</ul>`
        : `<div class="mini">No ticket has settled yet${r.pending ? ` (${r.pending} waiting on results)` : ""}.</div>`}</div>`;
    };
    const { lk, dog, dogWhy } = measuredOf(d);
    const rows = lk ? [["The rule", lk.rule], ["Market only (no model)", lk.market_only]].map(([name, s]) => `
        <li><span><b>${esc(name)}</b><br><span class="mini">${esc(recordLine(s.all, yearsLabel(lk.periods.all)))}</span><br>
          <span class="mini">${esc(recordLine(s.last_3_years, "last 3 years"))}</span></span></li>`).join("") : "";
    const mm = lk && lk.multipliers && lk.multipliers.all;
    const salts = (dog && dog.salts) || {};
    const per = (dog && dog.periods) || {};
    const saltRows = PERIODS.filter(([k]) => salts[k])
      .map(([k, label]) => saltLine(salts[k], `${label}${per[k] ? `, ${yearsLabel(per[k])}` : ""}`)).join("");
    el.innerHTML = `
      <div class="note"><b>The market is hard to beat.</b> Each ticket is logged before its first leg starts and graded
        from each sport's own results; the two kinds are kept apart, each on its own days.</div>
      <div class="card"><h2>Live record</h2>${liveBlock("likeliest")}${d.records ? liveBlock("underdog") : ""}</div>
      <div class="card"><h2>Backtest: the likeliest ticket</h2>
        ${lk ? `<div class="mini">One ticket a day, point in time, at the prices on file; the market-only
          control picks by the market's chances alone and says whether the models add anything.</div>
          <ul class="list">${rows}</ul>
          <div class="mini num">Typical payout ${mult(mm && mm.median)} (from ${mult(mm && mm.min)} to ${mult(mm && mm.max)})</div>`
          : notMeasured()}
      </div>
      <div class="card"><h2>Backtest: the underdog ticket</h2>
        ${dog ? `<div class="mini">The headline is the live draw - the one this page makes - on every day since August 2016,
          point in time. It ships beside the likeliest ticket whatever this shows.</div>
          <div class="eyebrow">At the live draw</div>
          ${dogLines(dog)}
          ${saltRows ? `<div class="eyebrow">The same games, other draws</div>
            <div class="mini">A spread over draws is not a confidence interval: the games and results are the same, only the lot changes.</div>
            <ul class="list">${saltRows}</ul>` : ""}
          <div class="eyebrow">Report only</div>
          <div class="mini">For comparison. None of these can replace the lottery in this version of the rule.</div>
          <ul class="list">${REPORT.map((r) => reportRow(r, (dog.report_only || {})[r[0]])).join("")}</ul>
          ${dog.missing ? `<div class="mini">Missing: ${esc(dog.missing)}.</div>` : ""}`
          : notMeasured(dogWhy || "The backtest has not been run.")}
      </div>`;
  },

  async splits() {
    const el = $("view-splits");
    el.innerHTML = DESK.skeleton(3);
    let d;
    try { d = await data(); } catch (e) { return fail(el, e, "the splits", () => BOOT.splits()); }
    const { lk, dog } = measuredOf(d);
    const sp = lk && lk.splits;
    if (!sp) { el.innerHTML = `<div class="card">${notMeasured()}</div>`; return; }
    const block = (title, rows, note) => !rows || typeof rows !== "object" ? "" : `<div class="card"><h2>${esc(title)}</h2>${note ? `<div class="mini">${esc(note)}</div>` : ""}
      <ul class="list">${Object.entries(rows).filter(([, s]) => s && s.tickets).map(([k, s]) => `<li><span><b>${esc(k)}</b><br>
        <span class="mini">${s.tickets} tickets, ${s.wins} won (${pct(s.hit_rate)}${num(s.expected_hit_rate) != null ? `; the picks said ${pct(s.expected_hit_rate)}` : ""})${
          implied(s) != null ? `; by the prices ${pc(implied(s))}` : ""}</span></span>
        <span class="end num">${pc(s.roi)}<br><span class="mini">${pc(s.roi_low)} to ${pc(s.roi_high)}</span></span></li>`).join("")}</ul></div>`;
    const ds = dog && dog.splits;
    const dogBlocks = ds && typeof ds === "object" ? Object.entries(ds).filter(([, rows]) => rows && typeof rows === "object")
      .map(([k, rows]) => block(DOG_SPLITS[k] || `Underdog ticket ${k.replace(/_/g, " ")}`, rows,
        `Descriptive: none of these can move the price band, the floor, the weight or the sports in this version${
          k === "dogs_by_price" ? "; single bets on every underdog, in and out of the band, not tickets" : ""}.`)).join("") : "";
    el.innerHTML = `<div class="note">Splits of the same backtests, 2016 onward. Small groups swing widely:
        an interval that spans zero is no evidence of anything.</div>
      ${block("By sports on the ticket", sp.by_mix)}
      ${block("By weekday", sp.by_weekday, "The day in US Eastern time.")}
      ${block("By year", sp.by_year)}
      ${dogBlocks}`;
  },

  method() {
    const r = (DATA && DATA.rule) || { band: [3, 5], max_legs: 6, pool: 12, min_p: 0.5 };
    const g = r.dog || { price: [2, 4], floor: 0.2, sim_salts: 1000, version: 1 };
    const { dog } = DATA ? measuredOf(DATA) : { dog: null };
    $("view-method").innerHTML = `<div class="card"><h2>The likeliest ticket</h2>
      <ul class="plain">
        <li><b>One ticket a day</b>, every leg played that day - the US Eastern calendar day of each game's start.</li>
        <li><b>Legs</b> are the three apps' own picks: the UFC fight desk's pick, the NFL pick, the soccer pick
          (Europe's top five leagues), each at least a ${pct(r.min_p)} chance, each with a price on file. One leg per event.</li>
        <li><b>The ticket</b> is, among the day's ${r.pool} likeliest picks, the combination of 1 to ${r.max_legs} legs
          paying ${r.band[0]}-${r.band[1]}x with the highest chance that every leg lands.</li>
        <li><b>Locked at its first start.</b> Until its first leg starts the ticket is rebuilt at the latest prices;
          from then on it stays as it is, and it is graded as it was shown.</li>
        <li><b>Fixed before it was measured.</b> The rule was written down, then backtested on every day since August 2016,
          beside a market-only control.</li>
      </ul></div>
      <div class="card"><h2>The underdog ticket</h2>
      <ul class="plain">
        <li><b>Why.</b> The likeliest ticket is favourites by construction. This one takes an underdog - for variety and a
          bigger payout, never as the better bet: a draw cannot create an edge.</li>
        <li><b>The underdog</b> of each event is the side the market rates under 50%, its bookmaker's cut removed - in soccer
          the less likely team, never the draw. It counts if it is priced ${Number(g.price[0]).toFixed(2)}-${Number(g.price[1]).toFixed(2)}
          (+100 to +300) and its chance is at least ${pct(g.floor)}. It may be the app's own pick.</li>
        <li><b>The lot.</b> Each underdog gets a fixed lottery number from the day and the event; the draw favours likelier
          underdogs a little (a 45% one comes up about 1.5 times as often as a 30% one), so it is close to an even draw.
          The same events give the same draw on every run, whatever else is on the day.</li>
        <li><b>The fill.</b> With the underdog fixed, the likeliest favourites from other events that bring the ticket to
          ${r.band[0]}-${r.band[1]}x, 2 to ${r.max_legs} legs in all. If an underdog cannot be filled, the next one drawn is
          tried; if none can, there is no underdog ticket that day.</li>
        <li><b>Priced at a typical book.</b> The underdog leg is priced at the median across books where the files carry it
          (UFC), the consensus line (NFL) or the average price (soccer). A UFC underdog priced at the best of books is
          flagged: one book may pay less.</li>
        <li><b>Shipped whatever it shows.</b> Version ${esc(String(g.version))} was fixed before any of its results; it is measured at
          the live draw and over ${(g.sim_salts || 1000).toLocaleString()} other draws, beside report-only comparisons, and
          none of them can replace it. A change would be a new version, judged only on tickets logged after it.</li>
        <li><b>Locked, logged and graded on its own</b>, like the likeliest ticket: at its own first start, with its own record.</li>
      </ul>
      ${dog ? "" : `<div class="mini">${esc(DATA && measuredOf(DATA).dogWhy || "Not measured yet.")}</div>`}</div>
      <div class="card"><h2>What they do not do</h2>
      <ul class="plain">
        <li>They do not promise a profit: the market is hard to beat, and a parlay pays the bookmaker's margin on every leg.
          By the prices, both tickets lose about 8-9% of the stake on average.</li>
        <li>They do not read news the three apps do not read. Check every leg and price before betting.</li>
        <li>No ticket on a day where nothing fits: a day with one NFL game and no soccer often has none. The page then shows
          the next days with tickets, up to ${r.window_days || 7} days ahead.</li>
      </ul></div>
      <div class="note">For entertainment and analysis - not betting advice.</div>`;
  },
};

// ---------------------------------------------------------------- start

DESK.init({
  key: "yourparlays",
  name: "YourMulti",
  icon: "icons/icon-192.png",
  siblings: [{ name: "YourMMA", href: "../../app/", blurb: "The MMA fight desk" },
    { name: "YourNFL", href: "../../nfl/app/", blurb: "The NFL pick desk" },
    { name: "YourSoccer", href: "../../soccer/app/", blurb: "The soccer pick desk" }],
  stamp: () => (DATA && DATA.generated ? `Updated ${DESK.ago(DATA.generated)}` : "Fetch the latest tickets"),
  onReload: () => location.reload(),
});
DESK.tabs(["today", "record", "splits", "method"], BOOT, "yourparlays.tab", "today");
setInterval(stamp, 60000);
