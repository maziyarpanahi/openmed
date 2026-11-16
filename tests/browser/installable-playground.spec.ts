// Installable offline profile for the privacy playground.
//
// Each case stages a private copy of the playground, serves it from a local
// fault-injecting server, and drives the real service worker: offline reopen
// after explicit preparation, interrupted and corrupted updates, at-rest
// corruption, storage eviction and quota limits, and request interception
// proving that inference uploads nothing and user text is never cached.
// All text is synthetic. The pinned browser-test package lives in ./brand.
import { execFileSync, spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile, copyFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve, sep } from "node:path";

import {
  chromium,
  expect,
  test,
  type BrowserContext,
  type Page,
  type Response,
  type TestInfo,
} from "./brand/node_modules/@playwright/test/index.js";

const PLAYGROUND = "/docs/demo/privacy-playground/";
const PROFILE_DIR = "docs/demo/privacy-playground";
const CACHE_PREFIX = "openmed-offline:v1:";
const PYTHON = process.env.OPENMED_PYTHON ?? "python3";
// Service-worker startup and Cache Storage reads can be slow on loaded CI hosts.
const READY = { timeout: 20_000 };
const FIXTURE_DIR = "offline-fixture";
const FIXTURE_ASSETS = [
  { kind: "runtime", url: `${FIXTURE_DIR}/synthetic-adapter.js` },
  { kind: "runtime", url: `${FIXTURE_DIR}/synthetic-worker.js` },
  { kind: "runtime", url: `${FIXTURE_DIR}/synthetic-scorer.wasm` },
  { kind: "model", url: `${FIXTURE_DIR}/models/synthetic/rules.json` },
];
// Minimal synthetic module exporting end(start, length) = start + length.
const SYNTHETIC_WASM = Uint8Array.from([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60,
  0x02, 0x7f, 0x7f, 0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01,
  0x03, 0x65, 0x6e, 0x64, 0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x20,
  0x00, 0x20, 0x01, 0x6a, 0x0b,
]);
const CONTENT_TYPES: Record<string, string> = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".wasm": "application/wasm",
  ".webmanifest": "application/manifest+json",
  ".woff2": "font/woff2",
};

test.use({ screenshot: "off", serviceWorkers: "allow", trace: "off", video: "off" });
test.skip(
  ({ browserName }) => browserName !== "chromium",
  "Quota override and service-worker response provenance use Chromium protocols",
);

type Fault = { mode: "abort" | "corrupt"; path: string } | { mode: "stale"; path: string; root: string };
type LoggedRequest = { body: string; headers: string; method: string; url: string };
type StagedSite = { generation: string; profile: string; root: string };

class SiteServer {
  readonly log: LoggedRequest[] = [];
  fault: Fault | null = null;
  root: string;
  private port = 0;
  private server: Server | null = null;

  constructor(root: string) {
    this.root = root;
  }

  get origin(): string {
    return `http://127.0.0.1:${this.port}`;
  }

  url(path: string): string {
    return `${this.origin}${path}`;
  }

  async start(): Promise<void> {
    const server = createServer((request, response) => {
      void this.handle(request, response);
    });
    this.server = server;
    await new Promise<void>((resolveListen) =>
      server.listen(this.port, "127.0.0.1", () => resolveListen()),
    );
    this.port = (server.address() as AddressInfo).port;
  }

  async stop(): Promise<void> {
    const server = this.server;
    if (!server) {
      return;
    }
    this.server = null;
    server.closeAllConnections();
    await new Promise<void>((resolveClose) => server.close(() => resolveClose()));
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const chunks: Buffer[] = [];
    for await (const chunk of request) {
      chunks.push(chunk as Buffer);
    }
    const url = new URL(request.url ?? "/", this.origin);
    this.log.push({
      body: Buffer.concat(chunks).toString("utf8"),
      headers: JSON.stringify(request.headers),
      method: request.method ?? "",
      url: url.href,
    });
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { "cache-control": "no-store" }).end();
      return;
    }
    const fault = this.fault && this.fault.path === url.pathname ? this.fault : null;
    const root = fault?.mode === "stale" ? fault.root : this.root;
    let pathname = decodeURIComponent(url.pathname);
    if (pathname.endsWith("/")) {
      pathname += "index.html";
    }
    const file = resolve(root, `.${pathname}`);
    let body: Buffer;
    try {
      if (!file.startsWith(root + sep)) {
        throw new Error("outside");
      }
      body = await readFile(file);
    } catch {
      response.writeHead(404, { "cache-control": "no-store" }).end();
      return;
    }
    if (fault?.mode === "corrupt") {
      body = Buffer.from(body);
      body[Math.floor(body.length / 2)] ^= 0x01;
    }
    response.writeHead(200, {
      "cache-control": "no-cache",
      "content-length": String(body.length),
      "content-type": CONTENT_TYPES[extname(file)] ?? "application/octet-stream",
    });
    if (request.method === "HEAD") {
      response.end();
    } else if (fault?.mode === "abort") {
      // Send half of the declared body, then drop the connection.
      response.write(body.subarray(0, Math.floor(body.length / 2)), () => {
        response.socket?.destroy();
      });
    } else {
      response.end(body);
    }
  }
}

function repoRoot(): string {
  return resolve(dirname(test.info().file), "..", "..");
}

async function stageSite(options: { fixture?: boolean; variant?: string } = {}): Promise<StagedSite> {
  const repository = repoRoot();
  const root = await mkdtemp(join(tmpdir(), "openmed-offline-site-"));
  const sourceProfile = join(repository, PROFILE_DIR);
  const profile = join(root, PROFILE_DIR);
  const declaration = JSON.parse(
    await readFile(join(sourceProfile, "offline-assets.json"), "utf8"),
  ) as { assets: Array<{ kind: string; url: string }> };
  const files = [
    ...declaration.assets.map((asset) => (asset.url === "./" ? "index.html" : asset.url)),
    "offline-assets.json",
    "offline-sw.js",
  ];
  for (const file of files) {
    const target = resolve(profile, file);
    await mkdir(dirname(target), { recursive: true });
    await copyFile(resolve(sourceProfile, file), target);
  }

  let rebuild = false;
  if (options.fixture) {
    const fixtures = join(repository, "tests/browser/fixtures/offline-install");
    const destination = join(profile, FIXTURE_DIR);
    await mkdir(join(destination, "models/synthetic"), { recursive: true });
    for (const name of ["synthetic-adapter.js", "synthetic-worker.js", "models/synthetic/rules.json"]) {
      await copyFile(join(fixtures, name), join(destination, name));
    }
    await writeFile(join(destination, "synthetic-scorer.wasm"), SYNTHETIC_WASM);
    const manifestPath = join(profile, "offline-assets.json");
    const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
    manifest.assets.push(...FIXTURE_ASSETS);
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    rebuild = true;
  }
  if (options.variant) {
    const appPath = join(profile, "app.js");
    await writeFile(
      appPath,
      `${await readFile(appPath, "utf8")}// Synthetic generation ${options.variant}.\n`,
    );
    rebuild = true;
  }
  if (rebuild) {
    execFileSync(
      PYTHON,
      [
        join(repository, "scripts/web/build_offline_manifest.py"),
        "--root",
        profile,
        "--site-root",
        join(root, "docs"),
      ],
      { stdio: "pipe" },
    );
  }
  const built = JSON.parse(await readFile(join(profile, "offline-assets.json"), "utf8"));
  return { generation: built.generation as string, profile, root };
}

async function cleanUp(...paths: string[]): Promise<void> {
  await Promise.all(paths.map((path) => rm(path, { force: true, recursive: true })));
}

async function launchPersistent(
  userDataDir: string,
  testInfo: TestInfo,
  offline = false,
): Promise<BrowserContext> {
  return chromium.launchPersistentContext(userDataDir, {
    channel: testInfo.project.use.channel,
    headless: true,
    offline,
    serviceWorkers: "allow",
  });
}

function card(page: Page) {
  return page.locator("#offline-card");
}

async function prepareOffline(page: Page, server: SiteServer): Promise<string> {
  await page.goto(server.url(PLAYGROUND));
  await expect(card(page)).toHaveAttribute("data-offline-state", "not-prepared");
  await page.locator("#offline-prepare").click();
  await expect(card(page)).toHaveAttribute("data-offline-state", "ready-reopen", {
    timeout: 20_000,
  });
  return (await card(page).getAttribute("data-offline-generation")) ?? "";
}

async function reloadControlled(page: Page, generation: string): Promise<Response[]> {
  const responses: Response[] = [];
  const listener = (response: Response) => responses.push(response);
  page.on("response", listener);
  await page.reload();
  await expect(card(page)).toHaveAttribute("data-offline-state", "ready", READY);
  await expect(card(page)).toHaveAttribute("data-offline-generation", generation);
  page.off("response", listener);
  return responses;
}

function expectSingleGeneration(responses: Response[], generation: string): void {
  expect(responses.length).toBeGreaterThan(0);
  for (const response of responses) {
    expect(response.fromServiceWorker(), "served by the offline worker").toBe(true);
    expect(response.headers()["x-openmed-offline-generation"]).toBe(generation);
  }
}

// Keep an unpatched fetch for test probes; the playground's local-only policy
// blocks script-initiated reads in the page itself.
function keepNativeFetch(): void {
  (window as unknown as { __nativeFetch: typeof fetch }).__nativeFetch =
    window.fetch.bind(window);
}

async function servedGeneration(page: Page, path: string): Promise<string | null> {
  return page.evaluate(async (relative) => {
    const nativeFetch = (window as unknown as { __nativeFetch: typeof fetch }).__nativeFetch;
    const response = await nativeFetch(new URL(relative, location.href).href);
    return response.headers.get("x-openmed-offline-generation");
  }, path);
}

// Requests the browser itself makes to check the worker script for updates.
function isWorkerUpdateCheck(entry: LoggedRequest): boolean {
  return entry.method === "GET" && new URL(entry.url).pathname.endsWith("/offline-sw.js");
}

function pageTraffic(server: SiteServer, since: number): LoggedRequest[] {
  return server.log.slice(since).filter((entry) => !isWorkerUpdateCheck(entry));
}

// A navigation schedules the browser's own worker update check; let it finish
// before the server content changes so it cannot overlap a scripted update.
async function settleSoftUpdate(server: SiteServer, since: number): Promise<void> {
  for (let waited = 0; waited < 3_000; waited += 100) {
    if (server.log.slice(since).some(isWorkerUpdateCheck)) {
      await new Promise((resolveWait) => setTimeout(resolveWait, 300));
      return;
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }
}

async function offlineCaches(page: Page): Promise<string[]> {
  return page.evaluate(async (prefix) => {
    const names = await caches.keys();
    return names.filter((name) => name.startsWith(prefix)).sort();
  }, CACHE_PREFIX);
}

async function redactBuiltin(page: Page, marker: string): Promise<void> {
  await page.locator("#runtime-module").fill("");
  await page.locator("#model-url").fill("");
  await page
    .locator("#input-text")
    .fill(`Synthetic note uses ${marker} and demo.person@example.invalid on 2030-04-12.`);
  await page.locator("#redact-button").click();
  await expect(page.locator("#total-count")).toHaveText("3");
  await expect(page.locator("#redacted-output")).toContainText("[ID]");
  await expect(page.locator("#redacted-output")).not.toContainText(marker);
}

async function redactWithAdapter(page: Page, marker: string): Promise<void> {
  await page.locator("#runtime-module").fill(`./${FIXTURE_DIR}/synthetic-adapter.js`);
  await page.locator("#model-url").fill(`./${FIXTURE_DIR}/models/synthetic/`);
  await page.locator("#input-text").fill(`Synthetic adapter note ${marker}.`);
  await page.locator("#redact-button").click();
  await expect(page.locator("#runtime-status")).toHaveText(
    "Trusted same-origin local adapter ready.",
  );
  await expect(page.locator("#total-count")).toHaveText("1");
  await expect(page.locator("#redacted-output")).toHaveText("Synthetic adapter note [ID].");
}

test("a prepared profile reopens and runs synthetic inference with the network gone", async ({}, testInfo) => {
  test.setTimeout(90_000);
  const site = await stageSite({ fixture: true });
  const server = new SiteServer(site.root);
  const userDataDir = await mkdtemp(join(tmpdir(), "openmed-offline-profile-"));
  await server.start();
  let context = await launchPersistent(userDataDir, testInfo);
  try {
    const page = await context.newPage();
    expect(await prepareOffline(page, server)).toBe(site.generation);
    await expect(page.locator("#offline-storage")).not.toHaveText("Not requested");
    await context.close();

    // Close the browser and take the origin away entirely before reopening.
    await server.stop();
    const requestsBeforeReopen = server.log.length;
    context = await launchPersistent(userDataDir, testInfo, true);
    const reopened = await context.newPage();
    const responses: Response[] = [];
    reopened.on("response", (response) => responses.push(response));
    const consoleErrors: string[] = [];
    reopened.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });

    await reopened.goto(server.url(PLAYGROUND));
    await expect(card(reopened)).toHaveAttribute("data-offline-state", "ready", READY);
    await expect(card(reopened)).toHaveAttribute("data-offline-generation", site.generation);
    await redactBuiltin(reopened, "ID-SYNTHETIC-OFFLINE-7310");
    await redactWithAdapter(reopened, "SYNOFF-55821");

    expectSingleGeneration(responses, site.generation);
    const servedPaths = responses.map((response) => new URL(response.url()).pathname);
    expect(servedPaths).toEqual(
      expect.arrayContaining([
        `${PLAYGROUND}${FIXTURE_DIR}/synthetic-adapter.js`,
        `${PLAYGROUND}${FIXTURE_DIR}/synthetic-worker.js`,
        `${PLAYGROUND}${FIXTURE_DIR}/synthetic-scorer.wasm`,
        `${PLAYGROUND}${FIXTURE_DIR}/models/synthetic/rules.json`,
      ]),
    );
    expect(server.log.length).toBe(requestsBeforeReopen);
    expect(consoleErrors).toEqual([]);
  } finally {
    await context.close().catch(() => undefined);
    await server.stop();
    await cleanUp(site.root, userDataDir);
  }
});

test("interrupted, corrupted and mixed updates keep the verified generation", async ({ context, page }) => {
  test.setTimeout(120_000);
  const current = await stageSite();
  const next = await stageSite({ variant: "next" });
  const server = new SiteServer(current.root);
  await server.start();
  try {
    await page.addInitScript(keepNativeFetch);
    expect(await prepareOffline(page, server)).toBe(current.generation);
    let since = server.log.length;
    expectSingleGeneration(await reloadControlled(page, current.generation), current.generation);
    await settleSoftUpdate(server, since);

    const update = page.locator("#offline-update");
    const attempts: Array<[Fault, string]> = [
      [{ mode: "abort", path: `${PLAYGROUND}app.js` }, "asset-fetch-failed"],
      [{ mode: "corrupt", path: `${PLAYGROUND}app.js` }, "asset-digest-mismatch"],
      // A half-deployed site: the new worker next to the previous manifest.
      [
        { mode: "stale", path: `${PLAYGROUND}offline-assets.json`, root: current.root },
        "generation-mismatch",
      ],
    ];
    server.root = next.root;
    for (const [fault, code] of attempts) {
      server.fault = fault;
      await update.click();
      await expect(card(page)).toHaveAttribute("data-offline-state", "update-failed", {
        timeout: 20_000,
      });
      await expect(card(page)).toHaveAttribute("data-offline-code", code);
      expect(await offlineCaches(page)).toEqual([
        expect.stringMatching(new RegExp(`:${current.generation}$`)),
      ]);
      // The page keeps receiving files from the verified generation only.
      expect(await servedGeneration(page, "app.js")).toBe(current.generation);
      expect(await servedGeneration(page, "../../stylesheets/openmed-system.css")).toBe(
        current.generation,
      );
    }

    // Reopening after the failed attempts still runs the previous copy.
    server.root = current.root;
    server.fault = null;
    since = server.log.length;
    expectSingleGeneration(await reloadControlled(page, current.generation), current.generation);
    await settleSoftUpdate(server, since);
    await redactBuiltin(page, "ID-SYNTHETIC-UPDATE-5512");

    server.root = next.root;
    await update.click();
    await expect(card(page)).toHaveAttribute("data-offline-state", "update-ready", {
      timeout: 20_000,
    });
    expect(await offlineCaches(page)).toHaveLength(2);

    // A second tab keeps running the verified current generation, and the
    // switch is refused while it is open so no tab can mix generations.
    const secondTab = await context.newPage();
    const secondResponses: Response[] = [];
    secondTab.on("response", (response) => secondResponses.push(response));
    await secondTab.goto(server.url(PLAYGROUND));
    await expect(card(secondTab)).toHaveAttribute("data-offline-state", "update-ready");
    expectSingleGeneration(secondResponses, current.generation);
    await page.locator("#offline-switch").click();
    await expect(card(page)).toHaveAttribute("data-offline-code", "other-tabs-open");
    await secondTab.close();

    // The worker's client list can trail a closed tab briefly; retry the
    // explicit switch until the single remaining tab reloads into it.
    await expect(async () => {
      const reloaded = page.waitForEvent("framenavigated", { timeout: 3_000 });
      await page.locator("#offline-switch").click({ timeout: 3_000 });
      await reloaded;
    }).toPass({ timeout: 20_000 });
    await expect(card(page)).toHaveAttribute("data-offline-state", "ready", READY);
    await expect(card(page)).toHaveAttribute("data-offline-generation", next.generation);
    expectSingleGeneration(await reloadControlled(page, next.generation), next.generation);
    expect(await offlineCaches(page)).toEqual([
      expect.stringMatching(new RegExp(`:${next.generation}$`)),
    ]);
  } finally {
    await server.stop();
    await cleanUp(current.root, next.root);
  }
});

test("a corrupted asset opens the repair state instead of running the app", async ({ browser }) => {
  test.setTimeout(90_000);
  const site = await stageSite();
  const server = new SiteServer(site.root);
  await server.start();
  const context = await browser.newContext({ serviceWorkers: "allow" });
  try {
    // Corruption while preparing: nothing is kept and nothing is registered.
    const first = await context.newPage();
    await first.goto(server.url(PLAYGROUND));
    server.fault = { mode: "corrupt", path: `${PLAYGROUND}app.js` };
    await first.locator("#offline-prepare").click();
    await expect(card(first)).toHaveAttribute("data-offline-state", "failed", {
      timeout: 20_000,
    });
    await expect(card(first)).toHaveAttribute("data-offline-code", "asset-digest-mismatch");
    expect(await offlineCaches(first)).toEqual([]);
    expect(
      await first.evaluate(async () => Boolean(await navigator.serviceWorker.getRegistration())),
    ).toBe(false);

    // Corruption at rest after a verified preparation.
    server.fault = null;
    await first.locator("#offline-prepare").click();
    await expect(card(first)).toHaveAttribute("data-offline-state", "ready-reopen", {
      timeout: 20_000,
    });
    await reloadControlled(first, site.generation);
    await first.evaluate(async (prefix) => {
      const [name] = (await caches.keys()).filter((key) => key.startsWith(prefix));
      const cache = await caches.open(name);
      const key = new URL("app.js", location.href).href;
      const bytes = new Uint8Array(await (await cache.match(key))!.arrayBuffer());
      bytes[100] ^= 0x01;
      await cache.put(key, new Response(bytes, { headers: { "content-type": "text/javascript" } }));
    }, CACHE_PREFIX);

    const navigation = await first.reload();
    expect(navigation?.status()).toBe(503);
    expect(navigation?.headers()["x-openmed-offline-state"]).toBe("repair");
    await expect(first.locator("body")).toHaveAttribute("data-openmed-offline-state", "repair");
    await expect(first.locator("#repair-code")).toHaveText("asset-digest-mismatch");
    await expect(first.getByRole("heading", { name: "Offline copy needs repair" })).toBeVisible();
    await expect(first.locator("#redact-button")).toHaveCount(0);
    await expect(first.locator("#input-text")).toHaveCount(0);

    await first.locator("#repair-button").click();
    await expect(card(first)).toHaveAttribute("data-offline-state", "ready", READY);
    await redactBuiltin(first, "ID-SYNTHETIC-REPAIRED-2211");
  } finally {
    await context.close();
    await server.stop();
    await cleanUp(site.root);
  }
});

test("storage eviction and quota limits surface explicit states", async ({ browser }) => {
  test.setTimeout(90_000);
  const site = await stageSite();
  const server = new SiteServer(site.root);
  await server.start();
  const evicted = await browser.newContext({ serviceWorkers: "allow" });
  const limited = await browser.newContext({ serviceWorkers: "allow" });
  try {
    const page = await evicted.newPage();
    await prepareOffline(page, server);
    await reloadControlled(page, site.generation);
    // Simulate the browser evicting best-effort Cache Storage under pressure.
    await page.evaluate(async (prefix) => {
      for (const name of await caches.keys()) {
        if (name.startsWith(prefix)) await caches.delete(name);
      }
    }, CACHE_PREFIX);
    await evicted.setOffline(true);
    await page.reload();
    await expect(page.locator("body")).toHaveAttribute("data-openmed-offline-code", "offline-copy-evicted");
    await expect(page.locator("#redact-button")).toHaveCount(0);
    await page.locator("#repair-button").click();
    await expect(page.locator("#repair-status")).toHaveAttribute(
      "data-code",
      "manifest-unavailable",
      { timeout: 15_000 },
    );
    expect(await offlineCaches(page)).toEqual([]);

    await evicted.setOffline(false);
    await page.locator("#repair-button").click();
    await expect(card(page)).toHaveAttribute("data-offline-state", "ready", READY);

    // A quota smaller than one verified copy is refused before any download.
    const quotaPage = await limited.newPage();
    await quotaPage.goto(server.url(PLAYGROUND));
    const cdp = await limited.newCDPSession(quotaPage);
    await cdp.send("Storage.overrideQuotaForOrigin", {
      origin: server.origin,
      quotaSize: 64 * 1024,
    });
    const before = server.log.length;
    await quotaPage.locator("#offline-prepare").click();
    await expect(card(quotaPage)).toHaveAttribute("data-offline-state", "failed", {
      timeout: 30_000,
    });
    const code = await card(quotaPage).getAttribute("data-offline-code");
    expect(["insufficient-storage", "quota-exceeded"]).toContain(code);
    await expect(quotaPage.locator("#offline-status")).toContainText("storage");
    expect(await offlineCaches(quotaPage)).toEqual([]);
    const assetRequests = pageTraffic(server, before).filter(
      (entry) => !new URL(entry.url).pathname.endsWith("/offline-assets.json"),
    );
    if (code === "insufficient-storage") {
      expect(assetRequests).toEqual([]);
    }
  } finally {
    await evicted.close();
    await limited.close();
    await server.stop();
    await cleanUp(site.root);
  }
});

test("the worker bypasses uploads and queries and never stores user text", async ({ page }, testInfo) => {
  test.setTimeout(90_000);
  const site = await stageSite({ fixture: true });
  const server = new SiteServer(site.root);
  await server.start();
  const builtinMarker = "ID-SYNTHETIC-NOSTORE-4096";
  const adapterMarker = "SYNOFF-77310";
  const traceDir = await mkdtemp(join(tmpdir(), "openmed-offline-trace-"));
  try {
    await page.addInitScript(keepNativeFetch);
    await prepareOffline(page, server);
    await reloadControlled(page, site.generation);

    const trace: Array<{ method: string; resource_type: string; url: string }> = [];
    const pageBodies: string[] = [];
    page.on("request", (request) => {
      trace.push({
        method: request.method(),
        resource_type: request.resourceType(),
        url: request.url(),
      });
      pageBodies.push(`${request.url()} ${request.postData() ?? ""}`);
    });
    const serverBefore = server.log.length;
    await redactBuiltin(page, builtinMarker);
    await redactWithAdapter(page, adapterMarker);

    // Inference reached no server at all, and the reused no-egress harness
    // accepts the action trace with only the cached fixture bundle allowed.
    expect(pageTraffic(server, serverBefore)).toEqual([]);
    expect(trace.every(({ method }) => method === "GET")).toBe(true);
    expect(pageBodies.join("\n")).not.toContain(builtinMarker);
    expect(pageBodies.join("\n")).not.toContain(adapterMarker);
    const tracePath = join(traceDir, "action-trace.json");
    await writeFile(tracePath, JSON.stringify(trace));
    const egress = spawnSync(
      PYTHON,
      [
        join(repoRoot(), "scripts/web/network_egress_check.py"),
        tracePath,
        "--allow-model-asset",
        server.url(`${PLAYGROUND}${FIXTURE_DIR}/`),
      ],
      { encoding: "utf8" },
    );
    expect(egress.status, "no unexpected egress during inference").toBe(0);
    expect(egress.stdout).not.toContain(adapterMarker);

    // Negative control: POST, query-bearing and unlisted reads carrying the
    // marker pass through to the network untouched instead of being cached.
    const controlResponses: Response[] = [];
    page.on("response", (response) => controlResponses.push(response));
    await page.evaluate(async (marker) => {
      const nativeFetch = (window as unknown as { __nativeFetch: typeof fetch }).__nativeFetch;
      await nativeFetch("./synthetic-upload", { body: marker, method: "POST" }).catch(() => undefined);
      await nativeFetch(`./app.js?note=${encodeURIComponent(marker)}`).catch(() => undefined);
      await nativeFetch(`./unlisted-${marker}.json`).catch(() => undefined);
    }, builtinMarker);
    const controlLog = pageTraffic(server, serverBefore);
    expect(controlLog.map(({ method }) => method)).toEqual(["POST", "GET", "GET"]);
    expect(controlLog[0].body).toBe(builtinMarker);
    expect(controlResponses.filter((response) => response.fromServiceWorker())).toEqual([]);
    const controlTrace = join(traceDir, "control-trace.json");
    await writeFile(
      controlTrace,
      JSON.stringify(controlLog.map(({ method, url }) => ({ method, resource_type: "fetch", url }))),
    );
    const control = spawnSync(
      PYTHON,
      [join(repoRoot(), "scripts/web/network_egress_check.py"), controlTrace],
      { encoding: "utf8" },
    );
    expect(control.status, "the harness flags the deliberate upload").toBe(1);
    expect(control.stdout).not.toContain(builtinMarker);

    // Storage holds exactly the pinned assets and the completion marker.
    const stored = await page.evaluate(async (prefix) => {
      const result: Array<{ body: string; url: string }> = [];
      for (const name of await caches.keys()) {
        if (!name.startsWith(prefix)) continue;
        const cache = await caches.open(name);
        for (const request of await cache.keys()) {
          const response = await cache.match(request);
          const bytes = new Uint8Array(await response!.arrayBuffer());
          result.push({ body: new TextDecoder("latin1").decode(bytes), url: request.url });
        }
      }
      const databases = (await indexedDB.databases?.()) ?? [];
      return {
        databases: databases.length,
        local: JSON.stringify({ ...localStorage }),
        names: await caches.keys(),
        result,
        session: JSON.stringify({ ...sessionStorage }),
      };
    }, CACHE_PREFIX);
    const manifest = JSON.parse(await readFile(join(site.profile, "offline-assets.json"), "utf8"));
    const expectedUrls = [
      ...manifest.assets.map((asset: { url: string }) => new URL(asset.url, server.url(PLAYGROUND)).href),
      server.url(`${PLAYGROUND}__openmed_offline__/complete`),
    ].sort();
    expect(stored.names).toHaveLength(1);
    expect(stored.result.map(({ url }) => url).sort()).toEqual(expectedUrls);
    for (const marker of [builtinMarker, adapterMarker]) {
      expect(JSON.stringify(stored)).not.toContain(marker);
    }
    expect(stored.databases).toBe(0);
    testInfo.annotations.push({
      description: `${stored.result.length} cached entries, ${trace.length} action requests`,
      type: "offline-profile",
    });
  } finally {
    await server.stop();
    await cleanUp(site.root, traceDir);
  }
});
