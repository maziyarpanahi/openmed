import { expect, test } from "@playwright/test";

test.use({
  screenshot: "off",
  trace: "off",
  video: "off",
});

const PLAYGROUND_PATH = "/docs/demo/privacy-playground/";

function sameOrigin(baseURL: string | undefined): string {
  return new URL(baseURL ?? "http://127.0.0.1:4173").origin;
}

test("redacts synthetic input locally without exposing source values", async ({
  baseURL,
  page,
}) => {
  const origin = sameOrigin(baseURL);
  const marker = "ID-SYNTHETIC-PRIVATE-2048";
  const requests: Array<{ method: string; origin: string; body: string }> = [];
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];

  page.on("request", (request) => {
    requests.push({
      body: request.postData() ?? "",
      method: request.method(),
      origin: new URL(request.url()).origin,
    });
  });
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto(PLAYGROUND_PATH, { waitUntil: "domcontentloaded" });
  const blockedUploadCount = await page.evaluate(async (value) => {
    try {
      await fetch("/synthetic-upload", { body: value, method: "POST" });
    } catch {
      // The local-only policy rejects the request before it reaches the server.
    }
    return document.querySelector("#upload-count")?.textContent;
  }, marker);
  expect(blockedUploadCount).toBe("1 blocked");
  await expect(page.locator("#network-status")).toContainText(
    "upload bodies blocked",
  );
  await page.locator("#input-text").fill(
    `Synthetic note uses ${marker} and demo.person@example.invalid on 2030-04-12.`,
  );
  await page.locator("#redact-button").click();

  await expect(page.locator("#status")).toContainText("redaction");
  await expect(page.locator("#total-count")).toHaveText("3");
  await expect(page.locator("#redacted-output")).toContainText("[ID]");
  await expect(page.locator("#redacted-output")).not.toContainText(marker);
  await expect(page.locator("#redacted-output")).not.toContainText(
    "demo.person@example.invalid",
  );

  const reportText = await page
    .locator(
      "#status, #runtime-status, #network-status, #redacted-output, #redaction-counts",
    )
    .allTextContents();
  expect(reportText.join(" ")).not.toContain(marker);
  expect(requests.every(({ method }) => ["GET", "HEAD"].includes(method))).toBe(
    true,
  );
  expect(
    requests.every(({ origin: requestOrigin }) => requestOrigin === origin),
  ).toBe(true);
  expect(requests.every(({ body }) => !body.includes(marker))).toBe(true);
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("uses the same-origin browser adapter contract", async ({
  baseURL,
  page,
}) => {
  const runtimePath = "/docs/demo/privacy-playground/test-runtime.js";
  const marker = "SYNTHETIC-ADAPTER-771";
  await page.route(`**${runtimePath}`, async (route) => {
    await route.fulfill({
      body: `
        export async function createOpenMedPipeline(options) {
          if (
            options.task !== "token-classification" ||
            options.backend !== "wasm" ||
            !options.modelUrl.endsWith("/models/synthetic/")
          ) {
            throw new Error("unexpected adapter options");
          }
          return async function detect(text) {
            const start = text.indexOf("${marker}");
            return [{
              entity_group: "${marker}",
              score: 1,
              start,
              end: start + "${marker}".length,
            }];
          };
        }
      `,
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });

  await page.goto(PLAYGROUND_PATH, { waitUntil: "domcontentloaded" });
  await page.locator("#runtime-module").fill("./test-runtime.js");
  await page.locator("#model-url").fill("./models/synthetic/");
  await page.locator("#input-text").fill(`Synthetic value ${marker}.`);
  await page.locator("#redact-button").click();

  await expect(page.locator("#runtime-status")).toHaveText(
    "Trusted same-origin local adapter ready.",
  );
  await expect(page.locator("#processing-status")).toHaveText(
    "Same-origin adapter in this tab",
  );
  await expect(page.locator("#total-count")).toHaveText("1");
  await expect(page.locator("#redacted-output")).toContainText("[PII]");
  await expect(page.locator("#redacted-output")).not.toContainText(marker);
  expect(new URL(baseURL ?? "http://127.0.0.1:4173").origin).toBe(
    sameOrigin(baseURL),
  );
});

test("blocks query, body, and connection egress before values leave", async ({
  page,
}) => {
  const marker = "SYNTHETIC-EGRESS-DO-NOT-SEND-883";
  const requests: Array<{ body: string; url: string }> = [];
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on("request", (request) => {
    requests.push({ body: request.postData() ?? "", url: request.url() });
  });
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto(PLAYGROUND_PATH, { waitUntil: "domcontentloaded" });
  const blocked = await page.evaluate(async (value) => {
    const attempts = [
      fetch(`/synthetic-asset?value=${encodeURIComponent(value)}`),
      fetch(`https://example.invalid/${encodeURIComponent(value)}`),
    ];
    await Promise.allSettled(attempts);
    navigator.sendBeacon("/synthetic-beacon", value);
    for (const Connection of [WebSocket, EventSource]) {
      try {
        new Connection(`https://example.invalid/${encodeURIComponent(value)}`);
      } catch {
        // The local-only constructors reject before opening a connection.
      }
    }
    const xhr = new XMLHttpRequest();
    xhr.open("GET", `/synthetic-xhr?value=${encodeURIComponent(value)}`);
    xhr.send();
    return document.querySelector("#upload-count")?.textContent;
  }, marker);

  expect(blocked).toBe("6 blocked");
  expect(requests.every(({ body, url }) => !`${url} ${body}`.includes(marker))).toBe(
    true,
  );
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("rejects unsafe adapter URLs and oversized input without echoing values", async ({
  page,
}) => {
  const marker = "SYNTHETIC-UNSAFE-ADAPTER-991";
  const requestUrls: string[] = [];
  page.on("request", (request) => requestUrls.push(request.url()));

  await page.goto(PLAYGROUND_PATH, { waitUntil: "domcontentloaded" });
  await page
    .locator("#runtime-module")
    .fill(`https://example.invalid/${marker}.js`);
  await page.locator("#model-url").fill("./models/synthetic/");
  await page.locator("#input-text").fill(`Synthetic value ${marker}.`);
  await page.locator("#redact-button").click();

  await expect(page.locator("#status")).toHaveText(
    "Local adapter and model URLs must use this page's origin.",
  );
  const publicText = await page
    .locator(
      "#status, #runtime-status, #network-status, #redacted-output, #redaction-counts",
    )
    .allTextContents();
  expect(publicText.join(" ")).not.toContain(marker);
  expect(requestUrls.every((url) => !url.includes(marker))).toBe(true);

  await page.locator("#runtime-module").fill("");
  await page.locator("#model-url").fill("");
  await page.locator("#input-text").evaluate((element) => {
    (element as HTMLTextAreaElement).value = "X".repeat(100_001);
  });
  await page.locator("#redact-button").click();
  await expect(page.locator("#status")).toHaveText(
    "Synthetic input exceeds the 100,000-character local limit.",
  );
  await expect(page.locator("#total-count")).toHaveText("0");
});

test("rejects oversized adapter output without rendering it", async ({ page }) => {
  const runtimePath = "/docs/demo/privacy-playground/oversized-runtime.js";
  await page.route(`**${runtimePath}`, async (route) => {
    await route.fulfill({
      body: `
        export async function createOpenMedPipeline() {
          return async function detect() {
            return Array.from({ length: 10001 }, (_, index) => ({
              entity_group: "ID",
              score: 1,
              start: index,
              end: index + 1,
            }));
          };
        }
      `,
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });

  await page.goto(PLAYGROUND_PATH, { waitUntil: "domcontentloaded" });
  await page.locator("#runtime-module").fill("./oversized-runtime.js");
  await page.locator("#model-url").fill("./models/synthetic/");
  await page.locator("#input-text").fill("Synthetic bounded input.");
  await page.locator("#redact-button").click();

  await expect(page.locator("#status")).toHaveText(
    "Local redaction failed. Check the supplied adapter and model bundle.",
  );
  await expect(page.locator("#redacted-output")).toHaveText(
    "No local result yet.",
  );
  await expect(page.locator("#total-count")).toHaveText("0");
});
