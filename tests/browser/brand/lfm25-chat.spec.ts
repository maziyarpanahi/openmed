import { expect, test } from "@playwright/test";

const DEMO_PATH = "/docs/demo/web/";
const LFM_ADAPTER_PATH = `${DEMO_PATH}lfm25-chat-web-adapter.mjs`;

test("selectable LFM2.5 chat streams locally while preserving Maple extraction", async ({
  baseURL,
  browserName,
  page,
}) => {
  test.skip(
    browserName !== "chromium",
    "The canonical WebGPU UI gate uses Chromium",
  );

  const expectedOrigin = new URL(
    baseURL ?? "http://127.0.0.1:4173",
  ).origin;
  const externalRequests: string[] = [];
  const runtimeErrors: string[] = [];
  const adapterRequests: string[] = [];

  page.on("request", (request) => {
    const url = new URL(request.url());
    if (
      ["http:", "https:"].includes(url.protocol)
      && url.origin !== expectedOrigin
    ) {
      externalRequests.push(`${request.method()} ${url.href}`);
    }
  });
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      runtimeErrors.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => runtimeErrors.push(error.message));

  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", {
      configurable: true,
      value: {
        async requestAdapter() {
          return {
            features: new Set(["shader-f16"]),
            info: { architecture: "fixture", vendor: "OpenMed" },
          };
        },
      },
    });
  });
  await page.route(`**${LFM_ADAPTER_PATH}`, async (route) => {
    adapterRequests.push(route.request().url());
    await route.fulfill({
      body: `
        const state = globalThis.__openmedLfm25Mock ??= {
          completedGenerations: 0,
          createCalls: [],
          disposeCalls: 0,
          firstDeltaYielded: false,
        };

        export async function createOpenMedLfm25ChatRuntime(options) {
          state.createCalls.push({
            contextTokens: options.contextTokens,
            device: options.device,
            modelUrl: options.modelUrl,
            networkPolicy: options.networkPolicy,
          });
          options.onProgress({
            detail: "Synthetic Q4F16 fixture",
            loaded: 1534153680,
            phase: "LFM2.5 fixture ready",
            total: 1534153680,
          });
          return {
            async *generate(messages, generation) {
              if (generation.reasoning !== false) {
                throw new Error("LFM2.5 chat must use direct-generation mode");
              }
              const output = JSON.stringify({
                answer: "The note directly links metformin to treatment of type 2 diabetes.",
                evidence: [{
                  quote: "metformin 500 mg twice daily for type 2 diabetes",
                  why: "This exact sentence states the medication, dose, and treated condition.",
                }],
                uncertainty: "The note does not establish whether metformin caused the nausea.",
                safety_note: "Human review is required.",
              });
              const split = output.indexOf(" to treatment");
              state.firstDeltaYielded = true;
              yield { delta: output.slice(0, split), index: 0 };
              await new Promise((resolve) => {
                state.completeGeneration = resolve;
              });
              state.completedGenerations += 1;
              state.completeGeneration = undefined;
              yield { delta: output.slice(split), index: 1 };
            },
            async dispose() {
              state.disposeCalls += 1;
              state.completeGeneration?.();
              state.completeGeneration = undefined;
            },
          };
        }
      `,
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });

  await page.goto(DEMO_PATH, { waitUntil: "domcontentloaded" });
  await page.locator("#try-preview").click();
  await page.locator('[data-task="entities"]').click();
  await page.locator("#run-task").click();
  await expect(page.locator("#status")).toContainText(
    /Clinical entities completed locally/i,
  );
  const medicationEntity = page.locator(".entity-card", {
    hasText: "metformin",
  });
  await expect(medicationEntity).toContainText("MEDICATION");
  await expect(
    page.locator(".entity-card", { hasText: "Amoxicillin" }),
  ).toHaveCount(0);

  await page.locator('[data-task="relations"]').click();
  await page.locator("#run-task").click();
  await expect(page.locator("#status")).toContainText(
    /Relation extraction completed locally/i,
  );
  const relation = page.locator(".relation-card", { hasText: "TREATS" });
  await expect(relation).toContainText("TREATS");
  await expect(relation).toContainText(
    "started metformin 500 mg twice daily for type 2 diabetes",
  );

  await page.locator("#chat-model").selectOption("lfm25");
  await expect(page.locator("#chat-model-loader")).toBeVisible();
  await expect(page.locator("#chat-send-label")).toHaveText("Ask LFM2.5");
  await page.locator("#load-chat-model").click();
  await expect(page.locator("#chat-model-state")).toHaveText("Ready");
  await expect(page.locator("#model-state")).toHaveText("Released");
  await expect(relation).toContainText("TREATS");

  expect(adapterRequests).toHaveLength(1);
  expect(new URL(adapterRequests[0]).origin).toBe(expectedOrigin);
  const firstLoad = await page.evaluate(() =>
    globalThis.__openmedLfm25Mock?.createCalls[0],
  );
  expect(firstLoad).toEqual({
    contextTokens: 4096,
    device: "webgpu",
    modelUrl: `${expectedOrigin}${DEMO_PATH}models/lfm2.5-2.6b-onnx-q4f16/`,
    networkPolicy: "same-origin-model-assets-only",
  });

  await page.locator("#ask-maple").click();
  await expect
    .poll(() =>
      page.evaluate(
        () => globalThis.__openmedLfm25Mock?.firstDeltaYielded ?? false,
      ),
    )
    .toBe(true);
  const assistantBody = page.locator(
    '.chat-message[data-role="assistant"] .chat-message__body',
  );
  await expect(assistantBody.locator(".chat-model-attribution")).toHaveText(
    "LFM2.5 · local WebGPU",
  );
  await expect(assistantBody.locator(".chat-cursor")).toContainText(
    "The note directly links metformin",
  );
  await expect(assistantBody.locator(".chat-cursor")).not.toContainText(
    "type 2 diabetes",
  );
  await expect(page.locator("#chat-status")).toContainText(
    "LFM2.5 is reading the note locally",
  );
  expect(
    await page.evaluate(
      () => globalThis.__openmedLfm25Mock?.completedGenerations,
    ),
  ).toBe(0);

  await expect
    .poll(() =>
      page.evaluate(
        () => typeof globalThis.__openmedLfm25Mock?.completeGeneration,
      ),
    )
    .toBe("function");
  await page.evaluate(() =>
    globalThis.__openmedLfm25Mock?.completeGeneration?.(),
  );
  await expect(page.locator("#chat-status")).toContainText(
    /Answer completed locally/i,
  );
  await expect(assistantBody.locator(".chat-bubble")).toHaveText(
    "The note directly links metformin to treatment of type 2 diabetes.",
  );
  await expect(assistantBody.locator(".chat-evidence")).toContainText(
    "metformin 500 mg twice daily for type 2 diabetes",
  );
  await expect(assistantBody.locator(".chat-uncertainty")).toContainText(
    "does not establish whether metformin caused the nausea",
  );
  await expect(page.locator("#chat-metric-first-token")).not.toHaveText("—");
  await expect(relation).toContainText("TREATS");

  await page.locator("#release-chat-model").click();
  await expect(page.locator("#chat-model-state")).toHaveText("Not loaded");
  await expect(page.locator("#ask-maple")).toBeDisabled();
  await expect(relation).toContainText("TREATS");
  await expect
    .poll(() =>
      page.evaluate(() => globalThis.__openmedLfm25Mock?.disposeCalls ?? 0),
    )
    .toBe(1);

  await page.locator("#load-chat-model").click();
  await expect(page.locator("#chat-model-state")).toHaveText("Ready");
  await page.locator("#chat-model").selectOption("maple");
  await expect
    .poll(() =>
      page.evaluate(() => globalThis.__openmedLfm25Mock?.disposeCalls ?? 0),
    )
    .toBe(2);
  await expect(page.locator("#chat-model-loader")).toBeHidden();
  await expect(page.locator("#chat-model-state")).toHaveText("Maple not loaded");
  await expect(page.locator("#chat-messages .chat-empty")).toContainText(
    "Ask Maple about the note",
  );
  await expect(
    page.locator('.chat-message[data-role="assistant"]'),
  ).toHaveCount(0);
  await expect(relation).toContainText("TREATS");

  expect(externalRequests).toEqual([]);
  expect(runtimeErrors).toEqual([]);
});

declare global {
  var __openmedLfm25Mock:
    | {
      completeGeneration?: () => void;
      completedGenerations: number;
      createCalls: Array<{
        contextTokens: number;
        device: string;
        modelUrl: string;
        networkPolicy: string;
      }>;
      disposeCalls: number;
      firstDeltaYielded: boolean;
    }
    | undefined;
}
