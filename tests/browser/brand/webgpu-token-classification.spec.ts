import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium, expect, test } from "@playwright/test";

const rootDir = fileURLToPath(new URL("../../..", import.meta.url));
const runtimePath = join(rootDir, "js", "openmedkit-web", "dist", "index.js");
const fixturePath = join(
  rootDir,
  "tests",
  "web",
  "fixtures",
  "webgpu_token_classification_golden.json",
);

test("batched classification head matches the Python reference on headless WebGPU", async ({
  baseURL,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "The WebGPU conformance gate uses Chromium");

  const browser = await chromium.launch({
    headless: true,
    args: webGpuLaunchArgs(),
  });
  const context = await browser.newContext({ serviceWorkers: "block" });
  const page = await context.newPage();
  const runtimeSource = await readFile(runtimePath, "utf8");
  const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
  const expectedOrigin = new URL(baseURL ?? "http://127.0.0.1:4173").origin;
  const externalRequests: string[] = [];
  page.on("request", (request) => {
    if (new URL(request.url()).origin !== expectedOrigin) {
      externalRequests.push(request.url());
    }
  });
  await page.route("**/openmed-webgpu-runtime.js", async (route) => {
    await route.fulfill({
      body: runtimeSource,
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });
  await page.route("**/webgpu-test-shell", async (route) => {
    await route.fulfill({
      body: "<!doctype html><title>OpenMed WebGPU runtime test</title>",
      contentType: "text/html; charset=utf-8",
      status: 200,
    });
  });

  try {
    await page.goto(`${expectedOrigin}/webgpu-test-shell`, {
      waitUntil: "domcontentloaded",
    });
    const result = await page.evaluate(async (syntheticFixture) => {
      const gpu = (
        navigator as Navigator & {
          gpu?: {
            requestAdapter(options?: Record<string, unknown>): Promise<{
              requestDevice(): Promise<unknown>;
            } | null>;
          };
        }
      ).gpu;
      if (gpu === undefined) throw new Error("headless Chromium exposes no WebGPU API");
      const adapter = await gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (adapter === null) throw new Error("headless Chromium returned no GPU adapter");
      const device = await adapter.requestDevice();
      const runtime = await import("/openmed-webgpu-runtime.js");
      const batchSize = 2;
      const hiddenStates = [
        ...syntheticFixture.hidden_states,
        ...syntheticFixture.hidden_states,
      ];
      const referenceLogits = [
        ...syntheticFixture.reference_logits,
        ...syntheticFixture.reference_logits,
      ];
      const attentionMask = [
        ...syntheticFixture.tokens.attention_mask,
        ...syntheticFixture.tokens.attention_mask,
      ];
      const head = await runtime.createWebGpuClassificationHead(device, {
        weights: syntheticFixture.classification_head.weights,
        bias: syntheticFixture.classification_head.bias,
        hiddenSize: syntheticFixture.classification_head.hidden_size,
        labelCount: syntheticFixture.classification_head.label_count,
      });
      try {
        const logits = await head.run(
          hiddenStates,
          batchSize,
          syntheticFixture.tokens.sequence_length,
        );
        const certification = runtime.certifyWebGpuReference({
          referenceLogits: {
            data: Float32Array.from(referenceLogits),
            dims: [
              batchSize,
              syntheticFixture.tokens.sequence_length,
              syntheticFixture.classification_head.label_count,
            ],
            outputName: "logits",
          },
          candidateLogits: logits,
          id2label: syntheticFixture.id2label,
          attentionMask,
          tolerance: syntheticFixture.logit_tolerance,
          maxRecallDelta: syntheticFixture.max_recall_delta,
          criticalLabels: syntheticFixture.critical_labels,
        });
        return {
          logits: [...logits.data],
          dims: [...logits.dims],
          maxAbsLogitDelta: certification.max_abs_logit_delta,
          spans: certification.candidate_token_spans,
          recall: certification.recall_gate.recall,
        };
      } finally {
        head.dispose();
        (device as { destroy?: () => void }).destroy?.();
      }
    }, fixture);

    expect(result.dims).toEqual([2, 4, 3]);
    expect(result.logits).toHaveLength(fixture.reference_logits.length * 2);
    expect(result.maxAbsLogitDelta).toBeLessThanOrEqual(
      fixture.logit_tolerance,
    );
    expect(result.spans).toEqual([
      ...fixture.reference_token_spans,
      ...fixture.reference_token_spans.map((span) => ({
        ...span,
        batch_index: 1,
      })),
    ]);
    expect(result.recall).toBe(1);
    expect(externalRequests).toEqual([]);
  } finally {
    await context.close();
    await browser.close();
  }
});

function webGpuLaunchArgs(): string[] {
  const common = [
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
  ];
  if (process.platform === "darwin") {
    return [
      ...common,
      "--enable-features=WebGPUDeveloperFeatures",
      "--use-angle=metal",
    ];
  }
  if (process.platform === "win32") {
    return [
      ...common,
      "--enable-features=WebGPUDeveloperFeatures",
      "--use-angle=d3d11",
    ];
  }
  return [
    ...common,
    "--enable-features=Vulkan,WebGPUDeveloperFeatures",
    "--use-angle=swiftshader",
    "--use-vulkan=swiftshader",
  ];
}
