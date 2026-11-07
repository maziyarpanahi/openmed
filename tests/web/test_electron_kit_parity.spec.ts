import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  CANONICAL_LABELS,
  KitParityError,
  parseKitParityResult,
  scalarToUtf16Offset,
  serializeKitParityResult,
  verifyKitParityResult,
} from "../../js/openmedkit-web/src/index";
import {
  OPENMED_ELECTRON_SCHEMA_VERSION,
  createElectronDeidentifyClient,
  redactTextWithSpans,
  toRendererKitParityResult,
  type ElectronDeidentifyRequest,
  type RendererOpenMedSpan,
} from "../../js/openmedkit-electron/src/index";

// Shared OpenMedKit kit-parity golden: tests/fixtures/parity/kit_parity_vectors.json,
// digest-pinned by fixtures/parity/span_parity_manifest.json.
const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "kit_parity_vectors.json");
const MANIFEST_PATH = join(rootDir, "fixtures", "parity", "span_parity_manifest.json");
const ENTITY_TYPE_PATTERN = /^[A-Z][A-Z0-9_:-]*$/;
const RENDERER_LABELS = new Set<string>(CANONICAL_LABELS);

interface VectorSpan {
  label: string;
  canonical_label: string;
  action: string;
  start: number;
  end: number;
  confidence_ppm?: number;
  replacement?: string;
}

interface Vector {
  id: string;
  text: string;
  spans: VectorSpan[];
  deidentified_text: string;
  result_json: string;
}

async function loadVectors(): Promise<Vector[]> {
  const bytes = await readFile(VECTORS_PATH);
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as {
    kit_parity: { schema: string; version: number; sha256: string };
  };
  const digest = createHash("sha256").update(bytes).digest("hex");
  assert.equal(digest, manifest.kit_parity.sha256, "digest_mismatch");
  const fixture = JSON.parse(bytes.toString("utf8")) as {
    schema: string;
    version: number;
    vectors: Vector[];
  };
  assert.equal(fixture.schema, manifest.kit_parity.schema);
  assert.equal(fixture.version, 1);
  return fixture.vectors;
}

/**
 * Renderer spans have no action or replacement and always redact to
 * `[canonical_label]`, so the Electron projection claims the vectors whose
 * every span is a `redact` to that token under a renderer-safe label.
 */
function rendererRepresentable(vector: Vector): boolean {
  return (
    vector.spans.length > 0 &&
    vector.spans.every(
      (span) =>
        span.action === "redact" &&
        span.replacement === `[${span.canonical_label}]` &&
        ENTITY_TYPE_PATTERN.test(span.label) &&
        RENDERER_LABELS.has(span.canonical_label),
    )
  );
}

function rendererSpans(vector: Vector): RendererOpenMedSpan[] {
  return vector.spans.map((span) => ({
    schema_version: 1,
    start: scalarToUtf16Offset(vector.text, span.start),
    end: scalarToUtf16Offset(vector.text, span.end),
    entity_type: span.label,
    canonical_label: span.canonical_label as RendererOpenMedSpan["canonical_label"],
    policy_label: "DIRECT_IDENTIFIER",
    score: span.confidence_ppm === undefined ? null : span.confidence_ppm / 1_000_000,
  }));
}

test("Electron renderer projection passes every renderer-representable vector", async () => {
  const vectors = await loadVectors();
  const claimed = vectors.filter(rendererRepresentable);

  assert.deepEqual(
    claimed.map((vector) => vector.id),
    ["renderer_redaction"],
  );
  for (const vector of claimed) {
    const requests: ElectronDeidentifyRequest[] = [];
    const client = createElectronDeidentifyClient({
      async invoke(_channel, request) {
        requests.push(request as ElectronDeidentifyRequest);
        return {
          schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
          requestId: (request as ElectronDeidentifyRequest).requestId,
          spans: rendererSpans(vector),
        };
      },
    });

    const response = await client.deidentify(vector.text);
    const redacted = redactTextWithSpans(vector.text, response.spans);
    const projected = toRendererKitParityResult(vector.text, response.spans);
    const serialized = serializeKitParityResult(projected);

    assert.equal(requests.length, 1);
    assert.equal(
      Buffer.compare(Buffer.from(redacted, "utf8"), Buffer.from(vector.deidentified_text, "utf8")),
      0,
      `case=${vector.id} de-identified UTF-8 bytes differ`,
    );
    assert.equal(serialized, vector.result_json, `case=${vector.id} result bytes differ`);
    assert.deepEqual(parseKitParityResult(serialized), projected);
    verifyKitParityResult(projected, vector.text);
  }
});

test("Electron projection converts UTF-16 renderer offsets to scalar offsets", async () => {
  const vector = (await loadVectors()).find((item) => item.id === "renderer_redaction")!;
  const spans = rendererSpans(vector);
  const projected = toRendererKitParityResult(vector.text, spans);

  assert.ok(spans.some((span, index) => span.start !== vector.spans[index]!.start));
  assert.deepEqual(
    projected.spans.map((span) => [span.start, span.end]),
    vector.spans.map((span) => [span.start, span.end]),
  );

  const shifted = spans.map((span, index) =>
    index === 0 ? { ...span, end: span.end + 1 } : span,
  );
  assert.notEqual(
    serializeKitParityResult(toRendererKitParityResult(vector.text, shifted)),
    vector.result_json,
  );

  const relabeled = spans.map((span, index) =>
    index === 0 ? { ...span, canonical_label: "LOCATION" as const } : span,
  );
  assert.notEqual(
    serializeKitParityResult(toRendererKitParityResult(vector.text, relabeled)),
    vector.result_json,
  );

  // A scalar offset leaked into the renderer contract splits a surrogate pair.
  const leaked = [{ ...spans[0]!, start: 1, end: spans[0]!.end }];
  assert.throws(
    () => toRendererKitParityResult(vector.text, leaked),
    (error: unknown) => {
      assert.ok(error instanceof KitParityError);
      assert.equal(error.message, "offset_not_scalar_boundary at $.spans[0].start");
      return true;
    },
  );
});
