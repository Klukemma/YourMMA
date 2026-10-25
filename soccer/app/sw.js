// YourSoccer's service worker: the MMA and NFL apps', with its own cache name.
//
// THE PREFIX IS NOT COSMETIC. All three apps live on one origin, and each
// worker deletes the caches it no longer uses on activate - by prefix. The
// MMA worker clears every "yourmma-" cache and the NFL worker every
// "yournfl-" one, so this one uses neither, and it clears only "yoursoccer-"
// caches. A worker's scope is its own directory, so none controls another.
//
// Bump CACHE when the shape of what is cached changes.
const CACHE = "yoursoccer-v1";
// How long the page waits for the network before opening from the cache.
const SHELL_WAIT = 2500;

const SHELL = [
  "./",
  "index.html",
  "app.js",
  "manifest.json",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/maskable-192.png",
  "icons/maskable-512.png",
  "icons/apple-touch-icon.png",
  // The data too, so an installed app opened with no signal has something to
  // show on its first launch. Served network-first all the same: these are
  // only the fallback.
  "data/matches.json",
  "data/tables.json",
  "data/performance.json",
];

// One file at a time: cache.addAll() rejects the whole install on one 404.
self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await Promise.all(SHELL.map((url) =>
      cache.add(new Request(url, { cache: "reload" })).catch(() => {})));
    self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter((n) => n.startsWith("yoursoccer-") && n !== CACHE)
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
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    // The picks: network first, so last weekend's are never shown as this
    // weekend's while a connection exists. yoursoccer.html carries its data
    // inside the page, and app.js prefers that to data/*.json - served stale,
    // it would show old picks.
    if (url.pathname.includes("/data/") || url.pathname.endsWith("/yoursoccer.html")) return networkFirst(request, cache, event);
    if (request.mode === "navigate") {
      // A navigation to any URL in scope is the one page there is.
      const response = await networkFirst(request, cache, event, SHELL_WAIT).catch(() => null);
      return response || (await cache.match("index.html")) || fetch(request);
    }
    if (/\.(html|js)$/.test(url.pathname)) return networkFirst(request, cache, event, SHELL_WAIT);
    return staleWhileRevalidate(request, cache);
  })());
});
