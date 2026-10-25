// Service worker: makes the installed app work with no signal, and is the
// reason a phone offers to install it at all - Chrome will not show the
// install prompt for a site that has no fetch handler, however complete its
// manifest is.
//
// Two strategies, because the app is two things:
//
//   the page and its code (HTML, JS)  network first, with a short wait.
//       Ask the network; if it has not answered within SHELL_WAIT the cached
//       copy is served and the network's answer, when it comes, refreshes the
//       cache for next time. Offline it opens from the cache at once. This
//       was stale-while-revalidate until a redesign shipped and an installed
//       app kept showing the old screens for a whole launch: an update
//       should land the first time the app is opened with a signal.
//
//   the icons and manifest  stale-while-revalidate: they rarely change, and
//       a launch late is fine for them.
//
//   the data (data/*.json)  network-first. The card, the odds and the record
//       change whenever the workflow runs, and showing yesterday's card as
//       though it were today's is exactly the kind of quiet lie this project
//       is built not to tell. Fall back to cache only when the network fails,
//       and the page's own stamp then shows how old it is.
//
// Bump CACHE when the shape of what is cached changes. The install step does
// not fail on a missing extra: a 404 on one icon should not cost the user an
// offline app.

const CACHE = "yourmma-v4";
// How long the page waits for the network before opening from the cache.
const SHELL_WAIT = 2500;

// "./" is the start_url. The rest is everything the shell needs to render
// before it asks for any data.
const SHELL = [
  "./",
  "index.html",
  "app.js",
  "matchup.js",
  "manifest.json",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/maskable-192.png",
  "icons/maskable-512.png",
  "icons/apple-touch-icon.png",
  // The data too, so an installed app opened with no signal has something to
  // show on its first launch. Served network-first all the same: these are
  // only the fallback.
  "data/card.json",
  "data/performance.json",
  "data/names.json",
  "data/constants.json",
  "data/fighters.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // One at a time, tolerating failures: cache.addAll() rejects the whole
    // install if any single request 404s.
    await Promise.all(SHELL.map((url) =>
      cache.add(new Request(url, { cache: "reload" })).catch(() => {})));
    self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter((n) => n.startsWith("yourmma-") && n !== CACHE)
      .map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

// Network first. With `wait`, a slow network is not waited on: after that
// many milliseconds the cached copy is served, and the network's answer
// still refreshes the cache when it lands (event.waitUntil keeps the worker
// alive for it).
async function networkFirst(request, cache, event, wait = 0) {
  const fetching = fetch(request).then((fresh) => {
    if (fresh && fresh.ok) cache.put(request, fresh.clone());
    return fresh;
  });
  fetching.catch(() => {});
  if (event) event.waitUntil(fetching.catch(() => {}));
  try {
    if (!wait) return await fetching;
    const fallback = new Promise((resolve) => setTimeout(() => resolve(cache.match(request)), wait));
    return (await Promise.race([fetching, fallback])) || await fetching;
  } catch (err) {
    const cached = await cache.match(request);
    if (cached) return cached;
    throw err;
  }
}

async function staleWhileRevalidate(request, cache) {
  const cached = await cache.match(request);
  const fetching = fetch(request).then((fresh) => {
    if (fresh && fresh.ok) cache.put(request, fresh.clone());
    return fresh;
  }).catch(() => cached);
  return cached || fetching;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  // Fonts and anything else off-origin go straight to the network. Offline
  // the page falls back to the system stack it already declares, which is
  // what that fallback is for.
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    if (url.pathname.includes("/data/")) return networkFirst(request, cache, event);
    if (request.mode === "navigate") {
      // A navigation to any URL in scope is the one page there is.
      const response = await networkFirst(request, cache, event, SHELL_WAIT).catch(() => null);
      return response || (await cache.match("index.html")) || fetch(request);
    }
    if (/\.(html|js)$/.test(url.pathname)) return networkFirst(request, cache, event, SHELL_WAIT);
    return staleWhileRevalidate(request, cache);
  })());
});
