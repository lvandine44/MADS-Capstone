const CACHE_NAME = "food-app-cache-v1";
const urlsToCache = [
    "/form", // Main page
    "/static/style.css", // CSS file
    "/static/icons/balance.png" // Icon file
];

// Install event: Cache all essential files
self.addEventListener("install", event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log("Caching essential files...");
            return cache.addAll(urlsToCache);
        })
    );
});

// Activate event: Remove old caches
self.addEventListener("activate", event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cache => {
                    if (cache !== CACHE_NAME) {
                        console.log("Deleting old cache:", cache);
                        return caches.delete(cache);
                    }
                })
            );
        })
    );
});

// Fetch event: Serve cached content or fetch from network
self.addEventListener("fetch", event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            // Serve cached response if available, otherwise fetch from network
            return response || fetch(event.request).catch(() => {
                // Fallback for offline requests
                if (event.request.mode === "navigate") {
                    return new Response(
                        "<h1>Offline</h1><p>It seems you are offline. Please check your internet connection.</p>", 
                        { headers: { "Content-Type": "text/html" } }
                    );
                }
            });
        })
    );
});
