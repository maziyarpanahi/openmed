import { writeFile } from "node:fs/promises";

import { chromium, expect, test } from "@playwright/test";

const RUN_REAL_MODEL = process.env.OPENMED_MAPLE_REAL_WEBGPU === "1";
const BASE_URL = process.env.OPENMED_PREVIEW_URL ?? "http://127.0.0.1:4173";
const MODEL_PATH =
  process.env.OPENMED_MAPLE_MODEL_PATH ??
  "./models/maple-preview-2bit-webgpu-coherent/";

test("real two-bit Maple WebGPU stays coherent across PII, entities, and chat", async ({
}, testInfo) => {
  test.skip(!RUN_REAL_MODEL, "Set OPENMED_MAPLE_REAL_WEBGPU=1 for the 5 GB gate");
  test.setTimeout(20 * 60_000);

  const browser = await chromium.launch({
    args: [
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--enable-features=WebGPUDeveloperFeatures",
      "--use-angle=metal",
    ],
    headless: true,
  });
  const context = await browser.newContext({
    serviceWorkers: "block",
    viewport: { height: 1100, width: 1440 },
  });
  const page = await context.newPage();
  const expectedOrigin = new URL(BASE_URL).origin;
  const externalRequests: string[] = [];
  const browserDiagnostics: string[] = [];
  page.on("request", (request) => {
    if (new URL(request.url()).origin !== expectedOrigin) {
      externalRequests.push(request.url());
    }
  });
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      browserDiagnostics.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => {
    browserDiagnostics.push(`pageerror: ${error.stack ?? error.message}`);
  });

  const timings: Record<string, number> = {};
  try {
    const url = new URL("/docs/demo/web/", BASE_URL);
    url.searchParams.set("runtime", "./maple-ort-web-adapter.mjs");
    url.searchParams.set("model", MODEL_PATH);
    await page.goto(url.href, { waitUntil: "domcontentloaded" });
    const gpu = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter({
        powerPreference: "high-performance",
      });
      return {
        available: Boolean(adapter),
        maxBufferSize: Number(adapter?.limits.maxBufferSize ?? 0),
        maxStorageBufferBindingSize: Number(
          adapter?.limits.maxStorageBufferBindingSize ?? 0,
        ),
      };
    });
    expect(gpu.available, "real WebGPU adapter is required").toBe(true);

    const loadStart = Date.now();
    await page.locator("#load-model").click();
    await expect
      .poll(() => page.locator("#model-state").textContent(), {
        timeout: 12 * 60_000,
      })
      .toMatch(/Ready|Load failed/);
    const modelState = await page.locator("#model-state").textContent();
    if (modelState !== "Ready") {
      await writeFile(
        testInfo.outputPath("maple-real-webgpu-diagnostics.txt"),
        `${browserDiagnostics.join("\n")}\n`,
      );
      throw new Error(
        `Maple load failed: ${await page.locator("#status").textContent()}\n` +
          browserDiagnostics.join("\n"),
      );
    }
    timings.loadMs = Date.now() - loadStart;
    await expect(page.locator("#runtime-details")).toContainText(
      "qmoe-2bit-ternary-rowwise",
    );

    const piiStart = Date.now();
    await page.locator("#run-task").click();
    await expect(page.locator("#status")).toContainText(
      /PII removal completed locally/i,
      { timeout: 4 * 60_000 },
    );
    timings.piiMs = Date.now() - piiStart;
    const piiCount = await page.locator("#results mark").count();
    expect(piiCount).toBeGreaterThanOrEqual(4);

    await page.locator('[data-task="entities"]').click();
    const entitiesStart = Date.now();
    await page.locator("#run-task").click();
    await expect(page.locator("#status")).toContainText(
      /Clinical entities completed locally/i,
      { timeout: 4 * 60_000 },
    );
    timings.entitiesMs = Date.now() - entitiesStart;
    await expect(page.locator(".entity-grid")).toContainText(/metformin/i);
    await expect(page.locator(".entity-grid")).toContainText(/diabetes/i);

    const chatStart = Date.now();
    await page.locator("#ask-maple").click();
    await expect(page.locator("#chat-status")).toContainText(
      /Answer completed locally/i,
      { timeout: 4 * 60_000 },
    );
    timings.chatMs = Date.now() - chatStart;
    const assistant = page.locator(
      '.chat-message[data-role="assistant"] .chat-bubble',
    );
    await expect(assistant).toContainText(/metformin/i);
    await expect(assistant).toContainText(/nausea/i);
    expect(await page.locator(".chat-evidence article").count()).toBeGreaterThan(0);
    await expect(page.locator(".chat-uncertainty")).toBeVisible();
    await expect(page.locator("#chat-metric-first-token")).not.toHaveText("—");
    expect(externalRequests).toEqual([]);

    await page.screenshot({
      fullPage: true,
      path: testInfo.outputPath("maple-real-webgpu.png"),
    });
    await writeFile(
      testInfo.outputPath("maple-real-webgpu.json"),
      JSON.stringify(
        {
          coherent: true,
          gpu,
          piiCount,
          quantization: "qmoe-2bit-ternary-rowwise",
          timings,
        },
        null,
        2,
      ) + "\n",
    );
  } finally {
    await context.close();
    await browser.close();
  }
});
