import { expect, test } from "@playwright/test";

const RUNTIME_PATH = "/docs/demo/web/maple-ort-web-adapter.mjs";
const ORT_MODULE_PATH = "/docs/demo/web/vendor/ort.webgpu.min.mjs";
const TOKENIZER_MODULE_PATH = "/docs/demo/web/vendor/maple-tokenizer.mjs";
const MODEL_BASE_PATH = "/docs/demo/web/models/maple-browser-fixture/";
const MANIFEST_PATH = `${MODEL_BASE_PATH}maple-bundle.json`;
const EOS_TOKEN_ID = 151_645;
const VOCAB_SIZE = 151_936;

test("Maple ORT Web adapter keeps unified decode local and disposable", async ({
  baseURL,
  page,
}) => {
  const marker = "SYNTHETIC-PHI-ORT-ADAPTER-9021";
  const consoleErrors: string[] = [];
  const externalRequests: string[] = [];
  const markerLeaks: string[] = [];
  const pageErrors: string[] = [];
  const expectedOrigin = new URL(baseURL ?? "http://127.0.0.1:4173").origin;
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin !== expectedOrigin) externalRequests.push(url.href);
    if (
      request.url().includes(marker) ||
      JSON.stringify(request.headers()).includes(marker) ||
      request.postData()?.includes(marker)
    ) {
      markerLeaks.push(request.method());
    }
  });

  await page.route(`**${ORT_MODULE_PATH}`, async (route) => {
    await route.fulfill({
      body: ortModuleFixture(),
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });
  await page.route(`**${TOKENIZER_MODULE_PATH}`, async (route) => {
    await route.fulfill({
      body: tokenizerModuleFixture(),
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });
  await page.route(`**${MANIFEST_PATH}`, async (route) => {
    await route.fulfill({
      body: JSON.stringify(mapleManifestFixture()),
      contentType: "application/json; charset=utf-8",
      status: 200,
    });
  });

  await page.goto("/docs/demo/web/", { waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle("Maple clinical studio for WebGPU");
  await expect(page.locator("#page-title")).toBeVisible();
  await page.locator("#runtime-module").fill(`.${RUNTIME_PATH.split("/web")[1]}`);
  await page
    .locator("#repo-id")
    .fill(`.${MODEL_BASE_PATH.split("/web")[1]}`);
  await page.locator("#input-text").fill(
    `${marker}: Patient Avery Morgan attended the synthetic clinic.`,
  );
  await page.locator("#load-model").click();

  await expect(page.locator("#model-state")).toHaveText("Ready");
  await expect(page.locator("#runtime-details")).toContainText(
    "ONNX Runtime Web reference adapter",
  );
  await expect(page.locator("#runtime-details")).toContainText(
    "real Maple WebGPU inference unvalidated",
  );
  await page.locator("#run-task").click();
  await expect(page.locator("#status")).toContainText(
    /PII removal completed locally/i,
  );
  await expect(page.locator("#results mark")).toHaveText("[NAME_1]");

  const evidence = await page.evaluate(() => globalThis.__mapleOrtMock);
  expect(evidence.sessionCreates).toBe(1);
  expect(evidence.graphUrl).toMatch(/\/models\/maple-browser-fixture\/model\.onnx$/);
  expect(evidence.executionProviders).toEqual(["webgpu"]);
  expect(evidence.externalData).toEqual([
    {
      data: expect.stringMatching(/\/model\.onnx\.data$/),
      path: "model.onnx.data",
    },
  ]);
  expect(evidence.runs.length).toBeGreaterThan(1);
  expect(evidence.runs[0]).toMatchObject({
    attentionMaskDims: [1, 2],
    firstPastDims: [1, 4, 0, 128],
    hasPositionIds: false,
    inputIdsDims: [1, 2],
  });
  expect(evidence.runs[1]).toMatchObject({
    attentionMaskDims: [1, 3],
    firstPastDims: [1, 4, 2, 128],
    hasPositionIds: false,
    inputIdsDims: [1, 1],
  });

  await page.evaluate(() => {
    globalThis.__mapleOrtMock.delayMs = 100;
  });
  const completedRunCount = evidence.runs.length;
  await page.locator("#run-task").click();
  await expect(page.locator("#stop-generation")).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => globalThis.__mapleOrtMock.runs.length))
    .toBeGreaterThan(completedRunCount);
  await page.locator("#stop-generation").click();
  await expect(page.locator("#status")).toContainText(/Generation stopped/i);
  await expect
    .poll(() => page.evaluate(() => globalThis.__mapleOrtMock.terminatedRuns))
    .toBeGreaterThan(0);

  await page.locator("#try-preview").click();
  await expect(page.locator("#model-state")).toHaveText("UI preview");
  await expect
    .poll(() => page.evaluate(() => globalThis.__mapleOrtMock.sessionReleases))
    .toBe(1);
  await expect
    .poll(() => page.evaluate(() => globalThis.__mapleOrtMock.tokenizerDisposes))
    .toBe(1);
  expect(consoleErrors).toEqual([]);
  expect(externalRequests).toEqual([]);
  expect(markerLeaks).toEqual([]);
  expect(pageErrors).toEqual([]);
});

function mapleManifestFixture() {
  return {
    schema_version: 1,
    source_model: "deepgrove/maple-preview",
    source_revision: "a".repeat(40),
    architecture: "MapleForCausalLM",
    quantization: "qmoe-4bit-blockwise-128",
    runtime: "onnxruntime-web",
    tokenizer_path: "tokenizer.json",
    graphs: {
      prefill_path: "model.onnx",
      decode_path: "model.onnx",
      input_ids_name: "input_ids",
      attention_mask_name: "attention_mask",
      position_ids_name: "position_ids",
      logits_name: "logits",
    },
    cache: {
      past_input_prefix: "past_key_values.",
      present_output_prefix: "present.",
    },
    generation: {
      eos_token_ids: [EOS_TOKEN_ID],
      max_context_tokens: 4096,
      max_input_tokens: 3072,
    },
    files: [
      { path: "model.onnx", size_bytes: 10, sha256: "1".repeat(64) },
      { path: "model.onnx.data", size_bytes: 20, sha256: "2".repeat(64) },
      { path: "tokenizer.json", size_bytes: 30, sha256: "3".repeat(64) },
    ],
  };
}

function tokenizerModuleFixture(): string {
  return `
    const state = globalThis.__mapleOrtMock ??= {
      delayMs: 0, runs: [], sessionCreates: 0, sessionReleases: 0,
      terminatedRuns: 0, tokenizerDisposes: 0,
    };
    export async function createOpenMedMapleTokenizer(options) {
      if (!options.tokenizerUrl.endsWith("/tokenizer.json")) {
        throw new Error("unexpected tokenizer URL");
      }
      return {
        async encodeMessages() { return [151643, 10]; },
        async decode(ids) { return String.fromCodePoint(...ids); },
        dispose() { state.tokenizerDisposes += 1; },
      };
    }
  `;
}

function ortModuleFixture(): string {
  const output = JSON.stringify({
    spans: [{ text: "Avery Morgan", type: "NAME" }],
    warnings: [],
  });
  return `
    const OUTPUT = ${JSON.stringify(output)};
    const EOS = ${EOS_TOKEN_ID};
    const VOCAB = ${VOCAB_SIZE};
    const state = globalThis.__mapleOrtMock ??= {
      delayMs: 0, runs: [], sessionCreates: 0, sessionReleases: 0,
      terminatedRuns: 0, tokenizerDisposes: 0,
    };
    export const env = { wasm: {} };
    export class Tensor {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = [...dims];
      }
      dispose() {}
    }
    const pastNames = [];
    const presentNames = [];
    for (let layer = 0; layer < 24; layer += 1) {
      for (const kind of ["key", "value"]) {
        pastNames.push(\`past_key_values.\${layer}.\${kind}\`);
        presentNames.push(\`present.\${layer}.\${kind}\`);
      }
    }
    export const InferenceSession = {
      async create(graphUrl, options) {
        state.sessionCreates += 1;
        state.graphUrl = graphUrl;
        state.executionProviders = options.executionProviders;
        state.externalData = options.externalData;
        let cursor = 0;
        return {
          inputNames: ["input_ids", "attention_mask", ...pastNames],
          outputNames: ["logits", ...presentNames],
          async run(feeds, _outputs, runOptions) {
            if (feeds.input_ids.dims[1] > 1) cursor = 0;
            const totalLength = feeds.attention_mask.dims[1];
            state.runs.push({
              attentionMaskDims: [...feeds.attention_mask.dims],
              firstPastDims: [...feeds[pastNames[0]].dims],
              hasPositionIds: "position_ids" in feeds,
              inputIdsDims: [...feeds.input_ids.dims],
            });
            if (state.delayMs) {
              await new Promise((resolve) => setTimeout(resolve, state.delayMs));
            }
            if (runOptions.terminate) state.terminatedRuns += 1;
            const token = cursor < OUTPUT.length ? OUTPUT.codePointAt(cursor++) : EOS;
            const inputLength = feeds.input_ids.dims[1];
            const logits = new Uint16Array(inputLength * VOCAB);
            logits.fill(0xfc00);
            logits[(inputLength - 1) * VOCAB + token] = 0;
            const result = {
              logits: new Tensor("float16", logits, [1, inputLength, VOCAB]),
            };
            for (const name of presentNames) {
              result[name] = new Tensor(
                "float16", new Uint16Array(0), [1, 4, totalLength, 128],
              );
            }
            return result;
          },
          release() { state.sessionReleases += 1; },
        };
      },
    };
  `;
}

declare global {
  // The route-fulfilled local modules expose sanitized lifecycle/KV evidence.
  // No prompt or note text is copied into this object.
  // eslint-disable-next-line no-var
  var __mapleOrtMock: {
    attentionMaskDims?: number[];
    delayMs: number;
    executionProviders: string[];
    externalData: Array<{ data: string; path: string }>;
    graphUrl: string;
    runs: Array<{
      attentionMaskDims: number[];
      firstPastDims: number[];
      hasPositionIds: boolean;
      inputIdsDims: number[];
    }>;
    sessionCreates: number;
    sessionReleases: number;
    terminatedRuns: number;
    tokenizerDisposes: number;
  };
}
