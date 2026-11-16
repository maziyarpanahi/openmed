/*
 * Installable offline profile for the OpenMed privacy playground.
 *
 * Contract (openmed.offline-assets.v1):
 * - The worker is registered only after an explicit "Prepare offline use"
 *   action. It precaches exactly the static files embedded below.
 * - install: fetch offline-assets.json and require its SHA-256 to match the
 *   embedded digest, then fetch every asset (no HTTP cache, same origin, no
 *   redirects), check its byte length and SHA-256, store it in a cache named
 *   for this generation, write a completion marker and verify the stored copy.
 *   Any failure deletes the partial cache and rejects install, so the previous
 *   generation keeps serving unchanged.
 * - activate: reached only after the new generation verified; deletes the
 *   other generations of this scope.
 * - fetch: only same-origin, query-free GET requests whose URL exactly matches
 *   the embedded table are answered, from this generation only, after a digest
 *   check. POST, query-bearing, range, cross-origin and every other request is
 *   not intercepted. Nothing is ever written to a cache from a page request,
 *   so text, results, requests and documents with user data are never stored.
 * - repair: a missing, evicted, incomplete or corrupted generation answers the
 *   start URL with a self-contained repair document instead of the app.
 *
 * Messages and errors carry fixed codes, counts and the generation digest
 * only; they never contain URLs, request data or page content.
 */
"use strict";

// BEGIN GENERATED OFFLINE PROFILE (python scripts/web/build_offline_manifest.py)
const OFFLINE_PROFILE = {
  "schema": "openmed.offline-assets.v1",
  "profile": "openmed-privacy-playground",
  "start_url": "./",
  "service_worker_sha256": "33dfc834bf7246df66565832c3583548417a8d984ec128d0036c03c14fb6fc0b",
  "generation": "daf11ad7a73878034a34cca1349776db22c64c733f4e5ab8bafb6f6a5730b46f",
  "manifest_sha256": "ce78973a42058536555acdd9f161673310e4c5d30f3684ae362c78c57fff9784",
  "total_bytes": 401360,
  "assets": [
    {
      "url": "./",
      "kind": "shell",
      "content_type": "text/html; charset=utf-8",
      "bytes": 11170,
      "sha256": "7918ba79ee6388107d73653ada4b053ea1552b8bbf6099fda893026ee63c0cd6"
    },
    {
      "url": "app.js",
      "kind": "shell",
      "content_type": "text/javascript; charset=utf-8",
      "bytes": 16179,
      "sha256": "9867b7fe16a39ca8a38073fd60ee63a5b4e507bab9a1a4f905567fd7e2bdd9c7"
    },
    {
      "url": "offline.js",
      "kind": "shell",
      "content_type": "text/javascript; charset=utf-8",
      "bytes": 14999,
      "sha256": "03fff7b9b37af24bbb2330a011f509adb530d06b16ecc4e6f70556fd9e7d0b9a"
    },
    {
      "url": "manifest.webmanifest",
      "kind": "shell",
      "content_type": "application/manifest+json",
      "bytes": 453,
      "sha256": "8e5178832e8eb95dd1fbd82220a48d117d317b1ccc3bd72cda49e604839b4ccb"
    },
    {
      "url": "../../assets/openmed-favicon.svg",
      "kind": "shell",
      "content_type": "image/svg+xml",
      "bytes": 568,
      "sha256": "f9c09e9ae482539a628dd4c239f99cfb7aa4b85606945cf1634daf206b4bc56a"
    },
    {
      "url": "../../assets/openmed-mark.svg",
      "kind": "shell",
      "content_type": "image/svg+xml",
      "bytes": 568,
      "sha256": "f9c09e9ae482539a628dd4c239f99cfb7aa4b85606945cf1634daf206b4bc56a"
    },
    {
      "url": "../../stylesheets/openmed-system.css",
      "kind": "shell",
      "content_type": "text/css; charset=utf-8",
      "bytes": 11502,
      "sha256": "62f1a98e9a06a23af47b98b8735597b5898c6db1235ccce293dd9abfc8d8f581"
    },
    {
      "url": "../../stylesheets/openmed-standalone.css",
      "kind": "shell",
      "content_type": "text/css; charset=utf-8",
      "bytes": 10006,
      "sha256": "65fd835707333a605d307782dfeceedee03de439c3174365cbb9f2664b7e6510"
    },
    {
      "url": "../../javascripts/openmed-standalone.js",
      "kind": "shell",
      "content_type": "text/javascript; charset=utf-8",
      "bytes": 1199,
      "sha256": "119ac9f4bc2206c739e2397d23c5f7f897e7168c14f19285792047cbb26f0003"
    },
    {
      "url": "../../assets/fonts/IBMPlexSans-Regular.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 54904,
      "sha256": "a40989b89e2c80751ad436b61f8ad07a3a3130fb0bb1400c85df1068524ed4ab"
    },
    {
      "url": "../../assets/fonts/IBMPlexSans-Medium.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 58440,
      "sha256": "8616ac647815fe069afa64ea256669cabcba80b780c13e02b5c6a6e6bccdcb41"
    },
    {
      "url": "../../assets/fonts/IBMPlexSans-SemiBold.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 58920,
      "sha256": "ee711b15202d9f05d8b51eab4f73b2f1832d9d0504082d5aeaeca55108c5ec52"
    },
    {
      "url": "../../assets/fonts/IBMPlexSans-Bold.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 54920,
      "sha256": "36b74f1eb2da82595b5037e8d67389a0ab6bdcab7358bfb8b7aca148cdcf27b3"
    },
    {
      "url": "../../assets/fonts/IBMPlexMono-Regular.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 34868,
      "sha256": "7dc604a2ed157d42ca477198c9cde29cfe96476065af30294f887534dd51c257"
    },
    {
      "url": "../../assets/fonts/IBMPlexMono-Medium.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 35868,
      "sha256": "403c826c3b9ef757e9d657c704f279e87f06824dfd5e18e9a86beb1b40d1dbf7"
    },
    {
      "url": "../../assets/fonts/IBMPlexMono-SemiBold.woff2",
      "kind": "shell",
      "content_type": "font/woff2",
      "bytes": 36796,
      "sha256": "f9646f0df18f79398785b3859eae690f731aa0815a015f8efe6fafd3115b7a99"
    }
  ]
};
// END GENERATED OFFLINE PROFILE

const SCHEMA = "openmed.offline-assets.v1";
const MESSAGE_SOURCE = "openmed-offline";
const MAX_MANIFEST_BYTES = 1024 * 1024;
const SCOPE_URL = new URL(self.registration.scope);
const CACHE_PREFIX = `openmed-offline:v1:${SCOPE_URL.pathname}:`;
const GENERATION = OFFLINE_PROFILE ? OFFLINE_PROFILE.generation : "";
const CACHE_NAME = `${CACHE_PREFIX}${GENERATION}`;
const MANIFEST_URL = new URL("offline-assets.json", self.location.href).href;
const MARKER_URL = new URL("__openmed_offline__/complete", SCOPE_URL).href;
const ASSETS = Object.freeze(
  (OFFLINE_PROFILE ? OFFLINE_PROFILE.assets : []).map((asset) =>
    Object.freeze({ ...asset, key: new URL(asset.url, self.location.href).href }),
  ),
);
const ASSET_BY_KEY = new Map(ASSETS.map((asset) => [asset.key, asset]));
const START_KEY = OFFLINE_PROFILE
  ? new URL(OFFLINE_PROFILE.start_url, self.location.href).href
  : "";
const TOTAL_BYTES = ASSETS.reduce((total, asset) => total + asset.bytes, 0);
const CODES = new Set([
  "asset-digest-mismatch",
  "asset-fetch-failed",
  "asset-missing",
  "asset-size-mismatch",
  "cache-write-failed",
  "generation-mismatch",
  "insufficient-storage",
  "manifest-invalid",
  "manifest-unavailable",
  "offline-copy-evicted",
  "offline-copy-incomplete",
  "other-tabs-open",
  "profile-invalid",
  "profile-unbuilt",
  "quota-exceeded",
  "unexpected-error",
  "unknown-request",
  "unregister-failed",
  "unsupported-browser",
]);
const REPAIR_TEXT = {
  "asset-digest-mismatch":
    "A stored file no longer matches its published SHA-256 digest.",
  "asset-missing": "A stored file is missing from the offline copy.",
  "asset-size-mismatch": "A stored file has an unexpected size.",
  "offline-copy-evicted":
    "The browser removed the offline copy, for example to free storage space.",
  "offline-copy-incomplete": "The offline copy was not completely prepared.",
  "profile-invalid": "The offline profile failed its integrity check.",
  "profile-unbuilt": "This build has no versioned offline profile.",
  "unexpected-error": "The offline copy could not be verified.",
};
const RESULT_TEXT = {
  "asset-digest-mismatch":
    "Repair stopped: a downloaded file did not match its published digest.",
  "asset-fetch-failed":
    "Repair needs a network connection to the site that published this copy.",
  "asset-size-mismatch":
    "Repair stopped: a downloaded file had an unexpected size.",
  "generation-mismatch":
    "A newer version is published. Remove this copy, then prepare it again.",
  "insufficient-storage": "Repair needs more browser storage than is available.",
  "manifest-unavailable":
    "Repair needs a network connection to the site that published this copy.",
  "quota-exceeded": "The browser storage quota was exceeded during repair.",
  "service-worker-unavailable": "The offline worker is not available.",
};

let repairInFlight = null;

class OfflineError extends Error {
  constructor(code) {
    super(code);
    this.name = "OfflineError";
    this.code = CODES.has(code) ? code : "unexpected-error";
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(installGeneration());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(activateGeneration());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (!OFFLINE_PROFILE || request.method !== "GET" || request.headers.has("range")) {
    return;
  }
  let url;
  try {
    url = new URL(request.url);
  } catch {
    return;
  }
  if (
    url.origin !== self.location.origin ||
    url.search ||
    url.username ||
    url.password
  ) {
    return;
  }
  url.hash = "";
  const key = url.href;
  if (request.mode === "navigate") {
    if (key === START_KEY) {
      event.respondWith(serveNavigation());
    }
    return;
  }
  const asset = ASSET_BY_KEY.get(key);
  if (asset) {
    event.respondWith(serveAsset(asset));
  }
});

self.addEventListener("message", (event) => {
  const port = event.ports && event.ports[0];
  const source = event.source;
  if (!port || !source || source.type !== "window" || !inScope(source.url)) {
    return;
  }
  const type = event.data && typeof event.data.type === "string" ? event.data.type : "";
  event.waitUntil(
    handleMessage(type).then(
      (reply) => port.postMessage(reply),
      () => port.postMessage({ ok: false, code: "unexpected-error" }),
    ),
  );
});

async function installGeneration() {
  try {
    await prepareGeneration();
  } catch (error) {
    const code = errorCode(error);
    await discardGeneration();
    await broadcast({ event: "prepare-failed", code, generation: GENERATION });
    throw new OfflineError(code);
  }
  await broadcast({
    assets: ASSETS.length,
    bytes: TOTAL_BYTES,
    event: "prepared",
    generation: GENERATION,
  });
}

async function activateGeneration() {
  // Reached only after installGeneration verified this generation, so the
  // previous generation is no longer needed by any page.
  const names = await caches.keys();
  await Promise.all(
    names
      .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
      .map((name) => caches.delete(name)),
  );
  await broadcast({ event: "activated", generation: GENERATION });
}

async function prepareGeneration() {
  await verifyEmbeddedProfile();
  await caches.delete(CACHE_NAME);
  await verifyDeployedManifest();
  await checkStorage(TOTAL_BYTES);
  const cache = await caches.open(CACHE_NAME);
  let done = 0;
  for (const asset of ASSETS) {
    const body = await fetchAsset(asset);
    await putEntry(cache, asset.key, body, asset.content_type);
    done += 1;
    await broadcast({ done, event: "progress", total: ASSETS.length });
  }
  await putEntry(
    cache,
    MARKER_URL,
    new TextEncoder().encode(GENERATION),
    "text/plain; charset=utf-8",
  );
  const result = await verifyGeneration({ digest: "all" });
  if (!result.ok) {
    throw new OfflineError(result.code);
  }
}

async function verifyEmbeddedProfile() {
  if (!OFFLINE_PROFILE) {
    throw new OfflineError("profile-unbuilt");
  }
  if (!self.caches || !self.crypto || !self.crypto.subtle) {
    throw new OfflineError("unsupported-browser");
  }
  const profile = OFFLINE_PROFILE;
  const keys = new Set();
  for (const asset of ASSETS) {
    const url = new URL(asset.key);
    if (
      url.origin !== self.location.origin ||
      url.search ||
      url.hash ||
      keys.has(asset.key) ||
      asset.key === self.location.href ||
      !Number.isSafeInteger(asset.bytes) ||
      asset.bytes < 0 ||
      !/^[0-9a-f]{64}$/.test(asset.sha256)
    ) {
      throw new OfflineError("profile-invalid");
    }
    keys.add(asset.key);
  }
  if (
    profile.schema !== SCHEMA ||
    !ASSET_BY_KEY.has(START_KEY) ||
    profile.total_bytes !== TOTAL_BYTES ||
    (await computeGeneration(profile)) !== GENERATION
  ) {
    throw new OfflineError("profile-invalid");
  }
}

async function computeGeneration(profile) {
  const lines = [
    SCHEMA,
    `profile\t${profile.profile}`,
    `start_url\t${profile.start_url}`,
    `service_worker_sha256\t${profile.service_worker_sha256}`,
    ...profile.assets.map((asset) =>
      [
        "asset",
        asset.url,
        asset.kind,
        asset.content_type,
        String(asset.bytes),
        asset.sha256,
      ].join("\t"),
    ),
  ];
  return sha256Hex(new TextEncoder().encode(`${lines.join("\n")}\n`));
}

async function verifyDeployedManifest() {
  const bytes = await fetchStatic(MANIFEST_URL, "manifest-unavailable");
  if (bytes.byteLength > MAX_MANIFEST_BYTES) {
    throw new OfflineError("manifest-invalid");
  }
  if ((await sha256Hex(bytes)) !== OFFLINE_PROFILE.manifest_sha256) {
    // The deployed manifest belongs to a different generation than this
    // worker, so the site is mid-deployment or mixed. Never assemble it.
    throw new OfflineError("generation-mismatch");
  }
  let manifest;
  try {
    manifest = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new OfflineError("manifest-invalid");
  }
  if (
    !manifest ||
    manifest.schema !== SCHEMA ||
    manifest.profile !== OFFLINE_PROFILE.profile ||
    manifest.generation !== GENERATION ||
    manifest.start_url !== OFFLINE_PROFILE.start_url ||
    manifest.service_worker_sha256 !== OFFLINE_PROFILE.service_worker_sha256 ||
    JSON.stringify(manifest.assets) !== JSON.stringify(OFFLINE_PROFILE.assets)
  ) {
    throw new OfflineError("manifest-invalid");
  }
}

async function checkStorage(requiredBytes) {
  const storage = self.navigator && self.navigator.storage;
  if (!storage || typeof storage.estimate !== "function") {
    return;
  }
  let estimate;
  try {
    estimate = await storage.estimate();
  } catch {
    return;
  }
  const quota = Number(estimate.quota);
  const usage = Number(estimate.usage) || 0;
  // The previous generation stays stored until this one verifies, so the
  // free space must hold a complete second copy.
  if (Number.isFinite(quota) && quota > 0 && quota - usage < requiredBytes) {
    throw new OfflineError("insufficient-storage");
  }
}

async function fetchAsset(asset) {
  const body = await fetchStatic(asset.key, "asset-fetch-failed");
  if (body.byteLength !== asset.bytes) {
    throw new OfflineError("asset-size-mismatch");
  }
  if ((await sha256Hex(body)) !== asset.sha256) {
    throw new OfflineError("asset-digest-mismatch");
  }
  return body;
}

async function fetchStatic(url, failureCode) {
  let response;
  try {
    response = await fetch(
      new Request(url, {
        cache: "no-store",
        credentials: "omit",
        mode: "same-origin",
        redirect: "error",
      }),
    );
  } catch {
    throw new OfflineError(failureCode);
  }
  if (
    response.status !== 200 ||
    response.type !== "basic" ||
    response.redirected
  ) {
    throw new OfflineError(failureCode);
  }
  try {
    return await response.arrayBuffer();
  } catch {
    // An interrupted body never reaches the cache.
    throw new OfflineError(failureCode);
  }
}

async function putEntry(cache, key, body, contentType) {
  try {
    await cache.put(
      key,
      new Response(body, {
        headers: { "content-type": contentType },
        status: 200,
      }),
    );
  } catch (error) {
    throw new OfflineError(isQuotaError(error) ? "quota-exceeded" : "cache-write-failed");
  }
}

// digest: "all" hashes every file (install and repair), "eager" hashes shell
// and runtime files (every navigation), and "presence" only checks that the
// complete generation is still stored (status). Every served file is hashed
// again before it is returned, whatever mode verified the generation.
async function verifyGeneration({ digest }) {
  if (!OFFLINE_PROFILE) {
    return { code: "profile-unbuilt", ok: false };
  }
  if (!(await caches.has(CACHE_NAME))) {
    return { code: "offline-copy-evicted", ok: false };
  }
  const cache = await caches.open(CACHE_NAME);
  const marker = await cache.match(MARKER_URL);
  if (!marker || (await marker.text()) !== GENERATION) {
    return { code: "offline-copy-incomplete", ok: false };
  }
  for (const asset of ASSETS) {
    const cached = await cache.match(asset.key);
    if (!cached) {
      return { code: "asset-missing", ok: false };
    }
    if (digest === "presence" || (digest === "eager" && asset.kind === "model")) {
      // Large model files are digest-checked whenever they are served.
      continue;
    }
    const failure = await bodyFailure(asset, await cached.arrayBuffer());
    if (failure) {
      return { code: failure, ok: false };
    }
  }
  return { ok: true };
}

async function bodyFailure(asset, body) {
  if (body.byteLength !== asset.bytes) {
    return "asset-size-mismatch";
  }
  if ((await sha256Hex(body)) !== asset.sha256) {
    return "asset-digest-mismatch";
  }
  return "";
}

async function readVerifiedAsset(asset) {
  if (!(await caches.has(CACHE_NAME))) {
    return { code: "offline-copy-evicted", ok: false };
  }
  const cached = await caches.match(asset.key, { cacheName: CACHE_NAME });
  if (!cached) {
    return { code: "asset-missing", ok: false };
  }
  const body = await cached.arrayBuffer();
  const failure = await bodyFailure(asset, body);
  if (failure) {
    return { code: failure, ok: false };
  }
  return {
    ok: true,
    response: new Response(body, {
      headers: {
        "cache-control": "no-store",
        "content-type": asset.content_type,
        "x-content-type-options": "nosniff",
        "x-openmed-offline-generation": GENERATION,
      },
      status: 200,
    }),
  };
}

async function serveNavigation() {
  try {
    const result = await verifyGeneration({ digest: "eager" });
    if (!result.ok) {
      return repairDocument(result.code);
    }
    const shell = await readVerifiedAsset(ASSET_BY_KEY.get(START_KEY));
    return shell.ok ? shell.response : repairDocument(shell.code);
  } catch {
    return repairDocument("unexpected-error");
  }
}

async function serveAsset(asset) {
  let result;
  try {
    result = await readVerifiedAsset(asset);
  } catch {
    result = { code: "unexpected-error", ok: false };
  }
  if (result.ok) {
    return result.response;
  }
  await broadcast({ code: result.code, event: "repair-required", generation: GENERATION });
  return new Response(null, {
    headers: {
      "cache-control": "no-store",
      "x-openmed-offline-code": result.code,
      "x-openmed-offline-state": "repair",
    },
    status: 503,
    statusText: "Offline Repair Required",
  });
}

async function handleMessage(type) {
  switch (type) {
    case "openmed-offline:status":
      return statusReply(await verifyGeneration({ digest: "presence" }));
    case "openmed-offline:repair":
      return repairGeneration();
    case "openmed-offline:remove":
      return removeProfile();
    case "openmed-offline:activate":
      return activateWaiting();
    default:
      return { code: "unknown-request", ok: false };
  }
}

function statusReply(result) {
  return {
    assets: ASSETS.length,
    bytes: TOTAL_BYTES,
    code: result.ok ? undefined : result.code,
    generation: GENERATION,
    ok: result.ok,
    state: result.ok ? "ready" : "repair",
  };
}

function repairGeneration() {
  if (!repairInFlight) {
    repairInFlight = (async () => {
      try {
        await prepareGeneration();
        await broadcast({ event: "repaired", generation: GENERATION });
        return { generation: GENERATION, ok: true };
      } catch (error) {
        await discardGeneration();
        return { code: errorCode(error), ok: false };
      } finally {
        repairInFlight = null;
      }
    })();
  }
  return repairInFlight;
}

async function removeProfile() {
  const names = await caches.keys();
  await Promise.all(
    names
      .filter((name) => name.startsWith(CACHE_PREFIX))
      .map((name) => caches.delete(name)),
  );
  const removed = await self.registration.unregister();
  await broadcast({ event: "removed" });
  return removed ? { ok: true } : { code: "unregister-failed", ok: false };
}

async function activateWaiting() {
  // Switching while another tab still runs the previous generation would
  // let that tab load files from two generations, so only a single tab may
  // switch. Otherwise the update activates once every tab is closed.
  const windows = await self.clients.matchAll({
    includeUncontrolled: true,
    type: "window",
  });
  if (windows.filter((client) => inScope(client.url)).length > 1) {
    return { code: "other-tabs-open", ok: false };
  }
  await self.skipWaiting();
  return { generation: GENERATION, ok: true };
}

async function discardGeneration() {
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    // Nothing else to clean up; the incomplete marker check still applies.
  }
}

async function broadcast(message) {
  let windows = [];
  try {
    windows = await self.clients.matchAll({
      includeUncontrolled: true,
      type: "window",
    });
  } catch {
    return;
  }
  for (const client of windows) {
    if (inScope(client.url)) {
      client.postMessage({ ...message, source: MESSAGE_SOURCE });
    }
  }
}

function inScope(url) {
  return typeof url === "string" && url.startsWith(SCOPE_URL.href);
}

function errorCode(error) {
  if (error instanceof OfflineError) {
    return error.code;
  }
  return isQuotaError(error) ? "quota-exceeded" : "unexpected-error";
}

function isQuotaError(error) {
  return Boolean(error) && (error.name === "QuotaExceededError" || error.code === 22);
}

async function sha256Hex(payload) {
  const digest = await crypto.subtle.digest("SHA-256", payload);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function repairDocument(code) {
  const safeCode = CODES.has(code) ? code : "unexpected-error";
  const nonceBytes = crypto.getRandomValues(new Uint8Array(16));
  const nonce = btoa(String.fromCharCode(...nonceBytes));
  const reason = REPAIR_TEXT[safeCode] || REPAIR_TEXT["unexpected-error"];
  const results = JSON.stringify(RESULT_TEXT).replace(/</g, "\\u003c");
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>OpenMed offline copy needs repair</title>
<style nonce="${nonce}">
body{margin:0;font:16px/1.5 system-ui,sans-serif;background:#f4f7f8;color:#0e1116}
main{max-width:40rem;margin:0 auto;padding:2rem 1rem}
h1{font-size:1.5rem;margin:0 0 1rem}
.actions{display:flex;flex-wrap:wrap;gap:.65rem;margin-top:1rem}
button{min-height:2.5rem;padding:.6rem .85rem;border:1px solid #8f2f2c;border-radius:.4rem;background:#8f2f2c;color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.secondary{background:#fff;color:#8f2f2c}
button:disabled{opacity:.65;cursor:wait}
button:focus-visible{outline:3px solid #0e1116;outline-offset:2px}
code{font-family:ui-monospace,monospace}
@media (prefers-color-scheme:dark){body{background:#0e1116;color:#f4f7f8}button.secondary{background:#0e1116;color:#f4f7f8;border-color:#f4f7f8}button:focus-visible{outline-color:#f4f7f8}}
</style>
</head>
<body data-openmed-offline-state="repair" data-openmed-offline-code="${safeCode}">
<main>
<h1>Offline copy needs repair</h1>
<p id="repair-reason">${reason}</p>
<p>The playground does not run from an unverified copy. Text you entered was never stored in the offline copy.</p>
<p>Reason code: <code id="repair-code">${safeCode}</code></p>
<div class="actions">
<button id="repair-button" type="button">Repair offline copy</button>
<button id="remove-button" class="secondary" type="button">Remove offline copy</button>
</div>
<p id="repair-status" role="status" aria-live="polite"></p>
</main>
<script nonce="${nonce}">
(() => {
  "use strict";
  const results = ${results};
  const status = document.getElementById("repair-status");
  const buttons = Array.from(document.querySelectorAll("button"));
  function setBusy(busy) {
    for (const button of buttons) button.disabled = busy;
  }
  function ask(type) {
    return new Promise((resolve) => {
      const controller = navigator.serviceWorker && navigator.serviceWorker.controller;
      if (!controller) {
        resolve({ ok: false, code: "service-worker-unavailable" });
        return;
      }
      const channel = new MessageChannel();
      channel.port1.onmessage = (event) => resolve(event.data || {});
      controller.postMessage({ type }, [channel.port2]);
    });
  }
  document.getElementById("repair-button").addEventListener("click", async () => {
    setBusy(true);
    status.textContent = "Downloading and verifying the offline copy...";
    const reply = await ask("openmed-offline:repair");
    if (reply.ok) {
      status.textContent = "Offline copy verified. Reloading...";
      location.reload();
      return;
    }
    status.dataset.code = String(reply.code || "unexpected-error");
    status.textContent = results[reply.code] || "Repair failed. Remove the offline copy and prepare it again.";
    setBusy(false);
  });
  document.getElementById("remove-button").addEventListener("click", async () => {
    setBusy(true);
    const reply = await ask("openmed-offline:remove");
    status.textContent = reply.ok
      ? "Offline copy removed. Reload while online to use the playground."
      : "The offline copy could not be removed.";
    if (reply.ok) location.reload();
    else setBusy(false);
  });
})();
</script>
</body>
</html>
`;
  return new Response(html, {
    headers: {
      "cache-control": "no-store",
      "content-security-policy":
        `default-src 'none'; script-src 'nonce-${nonce}'; ` +
        `style-src 'nonce-${nonce}'; base-uri 'none'; form-action 'none'; ` +
        "frame-ancestors 'none'",
      "content-type": "text/html; charset=utf-8",
      "referrer-policy": "no-referrer",
      "x-openmed-offline-code": safeCode,
      "x-openmed-offline-state": "repair",
    },
    status: 503,
    statusText: "Offline Repair Required",
  });
}
