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
  await expect(page.locator("#network-status")).toContainText("uploads blocked");
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
          if (options.task !== "token-classification" || options.backend !== "wasm") {
            throw new Error("unexpected adapter options");
          }
          return async function detect(text) {
            const start = text.indexOf("${marker}");
            return [{
              entity_group: "ID",
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
    "Same-origin local adapter ready.",
  );
  await expect(page.locator("#processing-status")).toHaveText(
    "Same-origin adapter in this tab",
  );
  await expect(page.locator("#total-count")).toHaveText("1");
  await expect(page.locator("#redacted-output")).toContainText("[ID]");
  await expect(page.locator("#redacted-output")).not.toContainText(marker);
  expect(new URL(baseURL ?? "http://127.0.0.1:4173").origin).toBe(
    sameOrigin(baseURL),
  );
});
