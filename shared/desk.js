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
