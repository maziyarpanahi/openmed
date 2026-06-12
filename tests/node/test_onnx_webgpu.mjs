/**
 * T6: onnxruntime-web WebGPU EP smoke test (skip-by-default in CI).
 *
 * Requires a Chromium browser with WebGPU support (headed).  Run manually:
 *
 *   npm install          # installs playwright if not present
 *   npx playwright install chromium
 *   node test_onnx_webgpu.mjs
 *
 * Prerequisites — run the Python T5 (webgpu) test first to generate fixtures:
 *
 *   uv run pytest tests/unit/onnx/test_webgpu.py::TestFp16Conversion::test_fp16_argmax_matches_fp32 -v
 *
 * The test is skipped automatically when:
 *   - WEBGPU_SMOKE=1 is not set (CI guard)
 *   - fixture files are missing
 *   - playwright or chromium are not installed
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const FIXTURE_FP16  = join(__dirname, 'fixtures', 'tiny_model_fp16.onnx');
const FIXTURE_JSON  = join(__dirname, 'fixtures', 'tiny_expected.json');

test('T6: WebGPU EP smoke — fp16 ONNX loads in headed Chromium', async (t) => {
  if (!process.env.WEBGPU_SMOKE) {
    t.skip('Set WEBGPU_SMOKE=1 to run the headed-Chrome WebGPU test');
    return;
  }

  if (!existsSync(FIXTURE_FP16) || !existsSync(FIXTURE_JSON)) {
    t.skip(
      'Fixture not found. Run the Python T5 test first:\n' +
      '  uv run pytest tests/unit/onnx/test_webgpu.py::TestFp16Conversion::test_fp16_argmax_matches_fp32 -v'
    );
    return;
  }

  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch {
    t.skip('playwright not installed — run: npm install playwright');
    return;
  }

  const expected = JSON.parse(readFileSync(FIXTURE_JSON, 'utf8'));
  const { batch_size, seq_len, num_labels, input_ids, attention_mask, logits: expectedLogits } = expected;
  const flatExpected = expectedLogits.flat(Infinity).map(Number);

  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  const ortLogits = await page.evaluate(
    async ({ fp16Path, batchSize, seqLen, inputIds, attnMask }) => {
      const ort = window.ort;
      ort.env.wasm.numThreads = 1;

      const response = await fetch(fp16Path);
      const buffer = await response.arrayBuffer();
      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['webgpu'],
      });

      const flatIds  = inputIds.flat().map(BigInt);
      const flatMask = attnMask.flat().map(BigInt);
      const feeds = {
        input_ids:      new ort.Tensor('int64', new BigInt64Array(flatIds),  [batchSize, seqLen]),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(flatMask), [batchSize, seqLen]),
      };
      const results = await session.run(feeds);
      return Array.from(results.logits.data);
    },
    {
      fp16Path: `file://${FIXTURE_FP16}`,
      batchSize: batch_size,
      seqLen: seq_len,
      inputIds: input_ids,
      attnMask: attention_mask,
    }
  );

  await browser.close();

  assert.equal(ortLogits.length, flatExpected.length, 'logit count mismatch');

  // fp16 tolerance — same as Python test_fp16_argmax_matches_fp32
  let maxAbsDiff = 0;
  for (let i = 0; i < ortLogits.length; i++) {
    const diff = Math.abs(ortLogits[i] - flatExpected[i]);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
  }
  assert.ok(maxAbsDiff < 0.1, `max abs diff ${maxAbsDiff.toFixed(3)} >= 0.1`);

  // Per-token argmax agreement
  const numTokens = batch_size * seq_len;
  for (let tok = 0; tok < numTokens; tok++) {
    const base = tok * num_labels;
    const a = ortLogits.slice(base, base + num_labels).indexOf(
      Math.max(...ortLogits.slice(base, base + num_labels))
    );
    const b = flatExpected.slice(base, base + num_labels).indexOf(
      Math.max(...flatExpected.slice(base, base + num_labels))
    );
    assert.equal(a, b, `argmax mismatch at token ${tok}: webgpu=${a}, torch=${b}`);
  }
});
