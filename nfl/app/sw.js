// YourNFL's service worker: the MMA app's, with its own cache name.
//
// THE PREFIX IS NOT COSMETIC. Both apps live on one origin, and each worker
// deletes the caches it no longer uses on activate - by prefix. The MMA
// worker clears every "yourmma-" cache, so this one must never use that
// prefix, and it clears only "yournfl-" caches so it never touches the MMA
// app's. A worker's scope is its own directory, so neither controls the other.
//
// Bump CACHE when the shape of what is cached changes.
const CACHE = "yournfl-v1";

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
      .filter((n) => n.startsWith("yournfl-") && n !== CACHE)
      .map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

async function networkFirst(request, cache) {
  try {
    const fresh = await fetch(request);
    if (fresh && fresh.ok) cache.put(request, fresh.clone());
    return fresh;
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
    // The week's picks: network first, so last week's slate is never shown
    // as this week's while a connection exists.
    if (url.pathname.includes("/data/")) return networkFirst(request, cache);
    if (request.mode === "navigate") {
      const response = await staleWhileRevalidate(request, cache);
      return response || cache.match("index.html") || fetch(request);
    }
    return staleWhileRevalidate(request, cache);
  })());
});
