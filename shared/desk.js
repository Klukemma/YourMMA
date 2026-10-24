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
