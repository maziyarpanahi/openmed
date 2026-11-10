import { expect, test } from "@playwright/test";
import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PUBLIC_ORIGIN = "https://openmed.life";
const SOCIAL_IMAGE_DIMENSIONS = { height: 630, width: 1200 };
const STAGED_SITE_DIRECTORY = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../site",
);

async function stagedHtmlFiles(directory: string): Promise<string[]> {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const candidate = path.join(directory, entry.name);
      if (entry.isDirectory()) return stagedHtmlFiles(candidate);
      if (entry.isFile() && entry.name.endsWith(".html")) return [candidate];
      return [];
    }),
  );
  return files.flat();
}

function metaAttributes(tag: string): Map<string, string> {
  const attributes = new Map<string, string>();
  const attributePattern =
    /\s+([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/g;
  for (const match of tag.matchAll(attributePattern)) {
    const [, name, doubleQuoted, singleQuoted, unquoted] = match;
    attributes.set(
      name.toLowerCase(),
      doubleQuoted ?? singleQuoted ?? unquoted ?? "",
    );
  }
  return attributes;
}

function openGraphImageUrls(html: string): string[] {
  const urls: string[] = [];
  for (const [tag] of html.matchAll(/<meta\b[^>]*>/gi)) {
    const attributes = metaAttributes(tag);
    if (attributes.get("property")?.toLowerCase() !== "og:image") continue;
    const content = attributes.get("content");
    if (content) urls.push(content);
  }
  return urls;
}

test.use({
  screenshot: "off",
  trace: "off",
  video: "off",
});

test("staged OpenGraph images resolve with the public metadata contract", async ({
  baseURL,
  page,
  request,
}) => {
  const localBaseURL = new URL(baseURL ?? "http://127.0.0.1:4173");
  const htmlFiles = await stagedHtmlFiles(STAGED_SITE_DIRECTORY);
  const imageUrls = new Set<string>();

  for (const htmlFile of htmlFiles) {
    const html = await fs.readFile(htmlFile, "utf8");
    for (const imageUrl of openGraphImageUrls(html)) imageUrls.add(imageUrl);
  }

  expect(htmlFiles.length).toBeGreaterThan(0);
  expect(
    imageUrls.size,
    "staged HTML must declare at least one og:image",
  ).toBeGreaterThan(0);
  await page.setContent("<!doctype html><html><body></body></html>");

  for (const imageUrl of [...imageUrls].sort()) {
    await test.step(imageUrl, async () => {
      const publicUrl = new URL(imageUrl);
      expect(publicUrl.origin, "og:image must use the public site origin").toBe(
        PUBLIC_ORIGIN,
      );
      const localUrl = new URL(
        `${publicUrl.pathname}${publicUrl.search}`,
        localBaseURL,
      );
      const response = await request.get(localUrl.href);
      expect(response.status(), `${localUrl.href} must resolve`).toBe(200);
      expect(
        response.headers()["content-type"]?.split(";", 1)[0].toLowerCase(),
        `${localUrl.href} must be served as PNG`,
      ).toBe("image/png");

      const dimensions = await page.evaluate(async (source) => {
        const image = new Image();
        image.src = source;
        await image.decode();
        return {
          height: image.naturalHeight,
          width: image.naturalWidth,
        };
      }, localUrl.href);
      expect(dimensions, `${localUrl.href} dimensions`).toEqual(
        SOCIAL_IMAGE_DIMENSIONS,
      );
    });
  }
});

test("browser demo checks every request without retaining synthetic input", async ({
  baseURL,
  page,
}, testInfo) => {
  const marker = "SYNTHETIC-PHI-NEVER-SEND-4182";
  const markerHash = crypto.createHash("sha256").update(marker).digest("hex");
  const expectedOrigin = new URL(
    baseURL ?? "http://127.0.0.1:4173",
  ).origin;
  const audit = {
    console_errors: [] as string[],
    external_requests: [] as string[],
    page_errors: [] as string[],
    request_failures: [] as string[],
  };
  const redact = (value: string) =>
    value.split(marker).join(`[redacted-sha256:${markerHash}]`);
  page.on("console", (message) => {
    if (message.type() === "error") {
      audit.console_errors.push(redact(message.text()));
    }
  });
  page.on("pageerror", (error) => audit.page_errors.push(redact(error.message)));
  page.on("requestfailed", (request) => {
    audit.request_failures.push(
      redact(`${request.method()} ${request.url()}`),
    );
  });
  const leakSignals: string[] = [];
  const requestEvidence: Array<{
    method: string;
    origin: "external" | "same-origin";
    url_sha256: string;
  }> = [];
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    for (const [surface, value] of Object.entries({
      body: request.postData() ?? "",
      headers: JSON.stringify(request.headers()),
      url: request.url(),
    })) {
      if (value.includes(marker)) leakSignals.push(surface);
    }
    if (!["GET", "HEAD"].includes(request.method())) leakSignals.push("method");
    requestEvidence.push({
      method: request.method(),
      origin: url.origin === expectedOrigin ? "same-origin" : "external",
      url_sha256: crypto
        .createHash("sha256")
        .update(request.url())
        .digest("hex"),
    });
    if (url.origin !== expectedOrigin) {
      audit.external_requests.push(
        `${request.method()} sha256:${requestEvidence.at(-1)?.url_sha256}`,
      );
    }
    if (leakSignals.length > 0 || url.origin !== expectedOrigin) {
      await route.abort("blockedbyclient");
      return;
    }
    await route.continue();
  });

  await page.goto("/docs/demo/web/", { waitUntil: "domcontentloaded" });
  await page.locator("#runtime-module").fill("https://example.invalid/runtime.js");
  await page.locator("#repo-id").fill("./models/maple-synthetic/");
  await page.locator("#input-text").evaluate((element, value) => {
    (element as HTMLTextAreaElement).value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
  }, marker);
  await page.locator("#load-model").click();
  await page.locator("#input-text").evaluate((element) => {
    (element as HTMLTextAreaElement).value = "";
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#status")).toHaveAttribute("data-kind", "error", {
    timeout: 20_000,
  });
  await expect(page.locator("#status")).toContainText(/origin/i);

  const retained = await page.evaluate((syntheticMarker) => {
    const stores = `${JSON.stringify(localStorage)}${JSON.stringify(sessionStorage)}`;
    return stores.includes(syntheticMarker);
  }, marker);
  const cookieText = JSON.stringify(await page.context().cookies());
  expect(leakSignals, `marker ${markerHash} crossed a request boundary`).toEqual([]);
  expect(
    retained || cookieText.includes(marker),
    `marker ${markerHash} was retained in browser state`,
  ).toBe(false);
  expect(audit.console_errors).toEqual([]);
  expect(audit.external_requests).toEqual([]);
  expect(audit.page_errors).toEqual([]);
  expect(audit.request_failures).toEqual([]);

  const sanitizedEvidence = JSON.stringify(
    {
      audit,
      marker_sha256: markerHash,
      requests: requestEvidence,
      storage_clean: !retained,
    },
    null,
    2,
  );
  expect(sanitizedEvidence.includes(marker)).toBe(false);
  await testInfo.attach("privacy-boundary-audit.json", {
    body: Buffer.from(sanitizedEvidence),
    contentType: "application/json",
  });
});
