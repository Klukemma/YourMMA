// Service worker: makes the installed app work with no signal, and is the
// reason a phone offers to install it at all - Chrome will not show the
// install prompt for a site that has no fetch handler, however complete its
// manifest is.
//
// Two strategies, because the app is two things:
//
//   the shell (HTML, JS, icons)  stale-while-revalidate. Serve the cached
//       copy at once so the app opens instantly and offline, then fetch a
//       fresh one in the background for next launch. The cost is that a push
//       lands one launch late, which is the right trade for a page you open
//       on a phone on a bus.
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

const CACHE = "yourmma-v1";

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
  // Fonts and anything else off-origin go straight to the network. Offline
  // the page falls back to the system stack it already declares, which is
  // what that fallback is for.
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    if (url.pathname.includes("/data/")) return networkFirst(request, cache);
    if (request.mode === "navigate") {
      // A navigation to any URL in scope is the one page there is.
      const response = await staleWhileRevalidate(request, cache);
      return response || cache.match("index.html") || fetch(request);
    }
    return staleWhileRevalidate(request, cache);
  })());
});
