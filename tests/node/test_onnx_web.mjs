/**
 * T4: onnxruntime-web WASM EP loads and runs the exported ONNX artifact.
 *
 * Prerequisites:
 *   Run the Python T1 pytest test first to generate fixtures/tiny_model.onnx
 *   and fixtures/tiny_expected.json:
 *
 *     uv run pytest tests/unit/onnx/test_onnx_convert.py::TestOnnxLogitParity -v
 *
 * Then run this test:
 *
 *     cd tests/node && npm install && npm test
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const FIXTURE_ONNX = process.env.FIXTURE_PATH
  ?? join(__dirname, 'fixtures', 'tiny_model.onnx');
const FIXTURE_JSON = process.env.EXPECTED_PATH
  ?? join(__dirname, 'fixtures', 'tiny_expected.json');

test('T4: onnxruntime-web WASM EP loads and runs the exported ONNX artifact', async (t) => {
  if (!existsSync(FIXTURE_ONNX) || !existsSync(FIXTURE_JSON)) {
    t.skip(
      'Fixture not found. Run the Python T1 test first:\n' +
      '  uv run pytest tests/unit/onnx/test_onnx_convert.py::TestOnnxLogitParity -v'
    );
    return;
  }

  // onnxruntime-web WASM backend — GPU-free, same code path as browser
  const ort = (await import('onnxruntime-web')).default;

  // Disable threads and SIMD for maximum compatibility in Node.js WASM mode
  ort.env.wasm.numThreads = 1;

  const session = await ort.InferenceSession.create(FIXTURE_ONNX, {
    executionProviders: ['wasm'],
  });

  const expected = JSON.parse(readFileSync(FIXTURE_JSON, 'utf8'));
  const { batch_size, seq_len, input_ids, attention_mask, logits: expectedLogits } = expected;

  // Build int64 tensors (onnxruntime-web requires BigInt64Array for int64)
  const flatIds = input_ids.flat().map(BigInt);
  const flatMask = attention_mask.flat().map(BigInt);

  const feeds = {
    input_ids: new ort.Tensor('int64', new BigInt64Array(flatIds), [batch_size, seq_len]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(flatMask), [batch_size, seq_len]),
  };

  const results = await session.run(feeds);
  const ortLogits = Array.from(results.logits.data);
  const flatExpected = expectedLogits.flat(Infinity).map(Number);

  assert.equal(
    ortLogits.length,
    flatExpected.length,
    `logit count mismatch: got ${ortLogits.length}, expected ${flatExpected.length}`,
  );

  // Max abs diff < 1e-3 (same tolerance as T1)
  let maxAbsDiff = 0;
  for (let i = 0; i < ortLogits.length; i++) {
    const diff = Math.abs(ortLogits[i] - flatExpected[i]);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
  }
  assert.ok(
    maxAbsDiff < 1e-3,
    `max abs logit diff ${maxAbsDiff.toExponential(2)} >= 1e-3 (onnxruntime-web vs torch)`,
  );

  // Per-token argmax agreement
  const numLabels = expected.num_labels;
  const numTokens = batch_size * seq_len;
  for (let tok = 0; tok < numTokens; tok++) {
    const base = tok * numLabels;
    const ortArgmax = ortLogits.slice(base, base + numLabels).indexOf(
      Math.max(...ortLogits.slice(base, base + numLabels))
    );
    const expArgmax = flatExpected.slice(base, base + numLabels).indexOf(
      Math.max(...flatExpected.slice(base, base + numLabels))
    );
    assert.equal(
      ortArgmax, expArgmax,
      `argmax mismatch at token ${tok}: ort=${ortArgmax}, expected=${expArgmax}`,
    );
  }
});
