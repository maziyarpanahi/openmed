// Same-origin static server for the ONNX Runtime Web token-classification lane.
//
// It serves only four allowlisted roots from one loopback origin: the harness,
// ONNX Runtime Web from the pinned local `onnxruntime-web` package, the built
// OpenMed browser runtime, and the graphs named in the generated lane fixture.
// `/lane/isolated.html` is cross-origin isolated (COOP + COEP) so threaded WASM
// can use SharedArrayBuffer; `/lane/plain.html` is not. Requests are never
// logged, and nothing outside the allowlist is readable.
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const laneDir = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = resolve(laneDir, "..", "..", "..");
const port = Number(process.env.OPENMED_ORT_LANE_PORT ?? "4174");
const fixtureDir = resolve(
  repoRoot,
  process.env.OPENMED_ORT_LANE_FIXTURE_DIR ?? "output/playwright/ort-lane/fixture",
);
const roots = {
  harness: resolve(laneDir, "harness"),
  ort: resolve(laneDir, "node_modules", "onnxruntime-web", "dist"),
  sdk: resolve(repoRoot, "js", "openmedkit-web", "dist"),
};
const FIXTURE_FILENAME = "lane-fixture.json";
const ORT_ASSET = /^ort(?:\.[a-z]+)*\.min\.mjs$|^ort-wasm-simd-threaded(?:\.[a-z]+)?\.(?:mjs|wasm)$/;
const SDK_ASSET = /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.js$/;
const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
};
const HARNESS_HTML =
  "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n" +
  "<title>OpenMed ORT Web lane</title>\n</head>\n<body>\n" +
  "<main id=\"lane\">ONNX Runtime Web token-classification lane</main>\n" +
  "</body>\n</html>\n";
const ISOLATION_HEADERS = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
};

const server = createServer((request, response) => {
  handle(request, response).catch(() => {
    if (!response.headersSent) send(response, 500, "internal error");
    else response.destroy();
  });
});
server.listen(port, "127.0.0.1");

async function handle(request, response) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    send(response, 405, "method not allowed");
    return;
  }
  const { pathname } = new URL(request.url ?? "/", "http://127.0.0.1");
  if (pathname === "/healthz") {
    send(response, 200, "ok");
    return;
  }
  if (pathname === "/lane/isolated.html" || pathname === "/lane/plain.html") {
    const isolated = pathname === "/lane/isolated.html";
    send(
      response,
      200,
      HARNESS_HTML,
      { "Content-Type": CONTENT_TYPES[".html"] },
      isolated,
    );
    return;
  }
  if (pathname === "/lane/lane.mjs") {
    await sendFile(request, response, roots.harness, "lane.mjs");
    return;
  }
  const [, scope, ...rest] = pathname.split("/");
  const name = rest.join("/");
  if (scope === "ort" && ORT_ASSET.test(name)) {
    await sendFile(request, response, roots.ort, name);
    return;
  }
  if (scope === "openmedkit" && SDK_ASSET.test(name)) {
    await sendFile(request, response, roots.sdk, name);
    return;
  }
  if (scope === "fixture" && (await fixtureAllows(name))) {
    await sendFile(request, response, fixtureDir, name);
    return;
  }
  send(response, 404, "not found");
}

async function fixtureAllows(name) {
  if (name === FIXTURE_FILENAME) return true;
  let fixture;
  try {
    fixture = JSON.parse(await readFile(resolve(fixtureDir, FIXTURE_FILENAME), "utf8"));
  } catch {
    return false;
  }
  const allowed = new Set();
  for (const graph of Object.values(fixture.graphs ?? {})) {
    allowed.add(graph.file);
    for (const item of graph.external_data ?? []) allowed.add(item.file);
  }
  return allowed.has(name);
}

async function sendFile(request, response, root, name) {
  const target = resolve(root, name);
  if (!target.startsWith(root + sep)) {
    send(response, 404, "not found");
    return;
  }
  let info;
  try {
    info = await stat(target);
  } catch {
    send(response, 404, "not found");
    return;
  }
  if (!info.isFile()) {
    send(response, 404, "not found");
    return;
  }
  response.writeHead(200, {
    ...baseHeaders(),
    "Content-Length": String(info.size),
    "Content-Type": CONTENT_TYPES[extname(target)] ?? "application/octet-stream",
  });
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  createReadStream(target).pipe(response);
}

function send(response, status, body, headers = {}, isolated = true) {
  response.writeHead(status, {
    ...baseHeaders(isolated),
    "Content-Type": "text/plain; charset=utf-8",
    ...headers,
  });
  response.end(body);
}

// Every response except the plain harness document carries the isolation
// headers: pthread worker scripts must be cross-origin isolated themselves
// before the page can share WebAssembly memory with them.
function baseHeaders(isolated = true) {
  return {
    "Cache-Control": "no-store",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    ...(isolated ? ISOLATION_HEADERS : {}),
  };
}
