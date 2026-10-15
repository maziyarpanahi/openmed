// Verifies the documented browser CSP profile (js/openmedkit-web/README.md):
// synthetic WebAssembly inference works, a cross-origin request is refused and
// recorded as a content-free violation, and dropping 'wasm-unsafe-eval' fails
// loudly instead of silently falling back.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "./brand/node_modules/@playwright/test/index.js";

test.use({ screenshot: "off", trace: "off", video: "off" });

const ORIGIN = "http://localhost:4199";
// Playwright runs from tests/browser/brand (see its config), one level below.
const DIR = join(process.cwd(), "..", "csp-profile");

const PROFILE_DIRECTIVES = [
  "default-src 'self'",
  "base-uri 'none'",
  "connect-src 'self' blob:",
  "font-src 'self'",
  "form-action 'none'",
  "img-src 'self' data:",
  "object-src 'none'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "style-src 'self'",
  "worker-src 'self' blob:",
];

function headers(csp: string, type: string) {
  return {
    "content-type": type,
    "content-security-policy": csp,
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-embedder-policy": "require-corp",
  };
}

async function serve(page: import("@playwright/test").Page, csp: string) {
  const files: Record<string, [string, string]> = {
    "/csp-profile/page.html": ["page.html", "text/html"],
    "/csp-profile/page.js": ["page.js", "text/javascript"],
    "/csp-profile/worker.js": ["worker.js", "text/javascript"],
  };
  const external: string[] = [];
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    const entry = url.origin === ORIGIN ? files[url.pathname] : undefined;
    if (!entry) {
      external.push(url.origin);
      return route.abort();
    }
    await route.fulfill({
      body: readFileSync(join(DIR, entry[0])),
      headers: headers(csp, entry[1]),
    });
  });
  return external;
}

test("documented profile: local inference works, cross-origin fetch is refused", async ({
  page,
}) => {
  const external = await serve(page, PROFILE_DIRECTIVES.join("; "));
  await page.goto(`${ORIGIN}/csp-profile/page.html`);
  expect(await page.evaluate(() => window.crossOriginIsolated)).toBe(true);

  const result = await page.evaluate(() => (window as any).__run());
  expect(result.main).toEqual({ ok: true });
  expect(result.worker).toEqual({ ok: true });
  expect(result.fetch).toEqual({ refused: true });
  expect(result.violations).toContain("connect-src");
  // The refused request never reached the network layer.
  expect(external).toEqual([]);
});

test("removing 'wasm-unsafe-eval' fails loudly, not silently", async ({ page }) => {
  const csp = PROFILE_DIRECTIVES.map((d) =>
    d.startsWith("script-src") ? "script-src 'self'" : d,
  ).join("; ");
  await serve(page, csp);
  await page.goto(`${ORIGIN}/csp-profile/page.html`);

  const result = await page.evaluate(() => (window as any).__run());
  expect(result.main.ok).toBe(false);
  expect(result.main.error).toBe("CompileError");
  expect(result.violations).toContain("script-src");
});
