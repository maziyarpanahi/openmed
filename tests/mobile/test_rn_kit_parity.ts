import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  KIT_PARITY_ACTIONS,
  KIT_PARITY_CANONICAL_LABELS,
  KIT_PARITY_PROFILE,
  KIT_PARITY_PROFILE_VERSION,
  OpenMedKitParityError,
  analyzeText,
  deidentify,
  extractPii,
  loadModel,
  serializeKitParityResult,
  setOpenMedKitNativeModuleForTests,
  toKitParityResult,
  type OpenMedKitNativeModule,
} from "../../js/openmedkit-react-native/src/index";

// Shared OpenMedKit kit-parity golden: tests/fixtures/parity/kit_parity_vectors.json,
// digest-pinned by fixtures/parity/span_parity_manifest.json.
const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "kit_parity_vectors.json");
const MANIFEST_PATH = join(rootDir, "fixtures", "parity", "span_parity_manifest.json");

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
  result_sha256: string;
}

interface Fixture {
  schema: string;
  version: number;
  profile: { name: string; version: number; actions: string[]; canonical_labels: string[] };
  vectors: Vector[];
}

async function loadFixture(): Promise<Fixture> {
  const bytes = await readFile(VECTORS_PATH);
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as {
    kit_parity: { schema: string; version: number; sha256: string };
  };
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    manifest.kit_parity.sha256,
    "digest_mismatch",
  );
  const fixture = JSON.parse(bytes.toString("utf8")) as Fixture;
  assert.equal(fixture.schema, manifest.kit_parity.schema);
  assert.equal(fixture.version, manifest.kit_parity.version);
  assert.equal(fixture.profile.name, KIT_PARITY_PROFILE);
  assert.equal(fixture.profile.version, KIT_PARITY_PROFILE_VERSION);
  return fixture;
}

/** A native span payload as the Swift and Kotlin bridge modules emit it. */
function nativeSpan(span: VectorSpan): Record<string, unknown> {
  return {
    schema_version: 1,
    doc_id: "kit-parity",
    start: span.start,
    end: span.end,
    text_hash: "hmac-sha256:00",
    entity_type: span.label,
    canonical_label: span.canonical_label,
    policy_label: "DIRECT_IDENTIFIER",
    regulatory_tags: [],
    score: span.confidence_ppm === undefined ? null : span.confidence_ppm / 1_000_000,
    detector: "kit-parity-fixture",
    evidence: {},
    action: span.action,
    replacement: span.replacement ?? null,
    reversible_id: null,
    section: null,
    metadata: {},
  };
}

function nativeModule(
  respond: (text: string) => { deidentifiedText: string; spans: Record<string, unknown>[] },
): OpenMedKitNativeModule {
  return {
    async loadModel(source) {
      return { ...source, platform: "ios", loaded: true };
    },
    async analyzeText(text) {
      return respond(text).spans as never;
    },
    async extractPii(text) {
      return { spans: respond(text).spans as never };
    },
    async deidentify(text) {
      const response = respond(text);
      // Native modules return spans in detection order; reverse to make the
      // bridge sort them.
      return { text, deidentifiedText: response.deidentifiedText, spans: [...response.spans].reverse() as never };
    },
  };
}

test.afterEach(() => {
  setOpenMedKitNativeModuleForTests(null);
});

test("bridge profile constants match the shared kit-parity profile", async () => {
  const fixture = await loadFixture();

  assert.deepEqual(fixture.profile.actions, [...KIT_PARITY_ACTIONS]);
  assert.deepEqual(fixture.profile.canonical_labels, [...KIT_PARITY_CANONICAL_LABELS]);
});

test("React Native deidentify results project to every vector's canonical bytes", async () => {
  const fixture = await loadFixture();
  for (const vector of fixture.vectors) {
    setOpenMedKitNativeModuleForTests(
      nativeModule(() => ({
        deidentifiedText: vector.deidentified_text,
        spans: vector.spans.map(nativeSpan),
      })),
    );
    await loadModel({ modelPath: "/app/models/kit-parity", backend: "mlx" });

    const result = await deidentify(vector.text, { hashSecret: "kit-parity" });
    const serialized = serializeKitParityResult(toKitParityResult(result));

    assert.equal(serialized, vector.result_json, `case=${vector.id} result bytes differ`);
    assert.equal(
      createHash("sha256").update(serialized).digest("hex"),
      vector.result_sha256,
    );
    assert.deepEqual(
      result.spans.map((span) => [span.start, span.end]),
      vector.spans.map((span) => [span.start, span.end]),
      `case=${vector.id} scalar offsets changed in the bridge`,
    );
  }
});

test("an altered offset, label, action, or byte fails the bridge gate", async () => {
  const fixture = await loadFixture();
  const vector = fixture.vectors.find((item) => item.id === "emoji_sequences")!;
  const project = async (mutate: (spans: Record<string, unknown>[]) => void, text = vector.deidentified_text) => {
    const spans = vector.spans.map(nativeSpan);
    mutate(spans);
    setOpenMedKitNativeModuleForTests(nativeModule(() => ({ deidentifiedText: text, spans })));
    return serializeKitParityResult(toKitParityResult(await deidentify(vector.text)));
  };

  assert.equal(await project(() => {}), vector.result_json);
  for (const mutate of [
    (spans: Record<string, unknown>[]) => {
      spans[0]!.end = (spans[0]!.end as number) + 1;
    },
    (spans: Record<string, unknown>[]) => {
      spans[0]!.canonical_label = "PERSON";
    },
    (spans: Record<string, unknown>[]) => {
      spans[1]!.action = "redact";
    },
  ]) {
    assert.notEqual(await project(mutate), vector.result_json);
  }
  assert.notEqual(await project(() => {}, `${vector.deidentified_text} `), vector.result_json);
});

test("malformed native spans fail with value-free reasons instead of coercion", async () => {
  const fixture = await loadFixture();
  const vector = fixture.vectors.find((item) => item.id === "emoji_sequences")!;
  const cases: [(span: Record<string, unknown>) => void, string][] = [
    [(span) => delete span.start, "field_missing at $.spans[0].start"],
    [(span) => (span.end = "45"), "field_type_invalid at $.spans[0].end"],
    [(span) => (span.end = 4.5), "field_type_invalid at $.spans[0].end"],
    [(span) => (span.action = "format_preserve"), "action_unknown at $.spans[0].action"],
  ];
  for (const [mutate, message] of cases) {
    const spans = vector.spans.map(nativeSpan);
    mutate(spans[0]!);
    setOpenMedKitNativeModuleForTests(
      nativeModule(() => ({ deidentifiedText: vector.deidentified_text, spans })),
    );
    for (const call of [analyzeText, extractPii]) {
      await assert.rejects(call(vector.text), (error: unknown) => {
        assert.ok(error instanceof OpenMedKitParityError);
        assert.equal(error.message, message);
        assert.equal(error.message.includes("Bo Reyes"), false);
        return true;
      });
    }
  }

  const unknownLabel = vector.spans.map(nativeSpan);
  unknownLabel[0]!.canonical_label = "kitparity";
  setOpenMedKitNativeModuleForTests(
    nativeModule(() => ({ deidentifiedText: vector.deidentified_text, spans: unknownLabel })),
  );
  const result = await deidentify(vector.text);
  assert.throws(() => toKitParityResult(result), {
    name: "OpenMedKitParityError",
    message: "label_unknown at $.spans[0].canonical_label",
  });
});
