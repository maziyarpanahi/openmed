import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  CANONICAL_LABELS,
  KIT_PARITY_ACTIONS,
  KIT_PARITY_CANONICAL_LABELS,
  KIT_PARITY_DECODER_CONTRACT_VERSIONS,
  KIT_PARITY_OFFSET_CONTRACT_VERSIONS,
  KIT_PARITY_PROFILE,
  KIT_PARITY_PROFILE_VERSION,
  KitParityError,
  acceptKitParityResult,
  parseKitParityResult,
  scalarToUtf16Offset,
  scalarToUtf8Offset,
  serializeKitParityResult,
  spansToRedactedText,
  toKitParityResult,
  utf16ToScalarOffset,
  utf8ToScalarOffset,
  verifyKitParityResult,
  type KitParityResult,
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
} from "../../js/openmedkit-web/src/index";

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
  utf8_start: number;
  utf8_end: number;
  utf16_start: number;
  utf16_end: number;
}

interface Vector {
  id: string;
  categories: string[];
  text: string;
  spans: VectorSpan[];
  deidentified_text: string;
  deidentified_sha256: string;
  result_json: string;
  result_sha256: string;
}

interface Rejection {
  id: string;
  input: string;
  reason: string;
  field_path: string;
  text?: string;
}

interface Vectors {
  schema: string;
  version: number;
  profile: {
    name: string;
    version: number;
    actions: string[];
    canonical_labels: string[];
    offset_contract_versions: number[];
    decoder_contract_versions: number[];
    privacy_sentinel: string;
  };
  vectors: Vector[];
  rejections: Rejection[];
  source_rejections: Required<Rejection>[];
  offset_rejections: {
    id: string;
    text: string;
    unit: "scalar" | "utf8" | "utf16";
    offset: number;
    reason: string;
  }[];
  compatibility: {
    id: string;
    offset_contract_version: number;
    decoder_contract_version: number;
    outcome: string;
  }[];
}

class FixtureError extends Error {
  constructor(readonly code: string) {
    super(code);
  }
}

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function loadVectors(bytes: Buffer, manifest: Record<string, any>): Vectors {
  const entry = manifest.kit_parity;
  if (entry?.schema !== "openmed.kit_parity_vectors" || entry.version !== 1) {
    throw new FixtureError("kit_vectors_unsupported");
  }
  if (sha256(bytes) !== entry.sha256) {
    throw new FixtureError("digest_mismatch");
  }
  const vectors = JSON.parse(bytes.toString("utf8")) as Vectors;
  if (
    vectors.schema !== entry.schema ||
    vectors.version !== entry.version ||
    vectors.profile.name !== KIT_PARITY_PROFILE ||
    vectors.profile.version !== KIT_PARITY_PROFILE_VERSION
  ) {
    throw new FixtureError("kit_vectors_unsupported");
  }
  return vectors;
}

let cached: Promise<Vectors> | undefined;
function vectors(): Promise<Vectors> {
  cached ??= (async () =>
    loadVectors(
      await readFile(VECTORS_PATH),
      JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as Record<string, any>,
    ))();
  return cached;
}

function webSpan(text: string, span: VectorSpan): OpenMedSpan {
  return {
    schema_version: 1,
    doc_id: "kit-parity",
    start: scalarToUtf16Offset(text, span.start),
    end: scalarToUtf16Offset(text, span.end),
    text_hash: "hmac-sha256:00",
    entity_type: span.label,
    canonical_label: span.canonical_label as OpenMedSpan["canonical_label"],
    policy_label: "DIRECT_IDENTIFIER",
    regulatory_tags: [],
    score:
      span.confidence_ppm === undefined ? null : span.confidence_ppm / 1_000_000,
    detector: "kit-parity-fixture",
    evidence: {},
    action: span.action as OpenMedSpan["action"],
    replacement: span.replacement ?? null,
    reversible_id: null,
    section: null,
    metadata: {},
  };
}

function webResult(vector: Vector): OpenMedDeidentifyResult {
  return {
    text: vector.text,
    deidentifiedText: vector.deidentified_text,
    // Reverse the order so the projection's own sort is exercised.
    spans: vector.spans.map((span) => webSpan(vector.text, span)).reverse(),
  };
}

/** Value-free drift report: case id, field path, offsets, labels, actions. */
function differences(id: string, expected: string, actual: string): string[] {
  if (expected === actual) {
    return [];
  }
  const want = JSON.parse(expected) as KitParityResult;
  const got = JSON.parse(actual) as KitParityResult;
  const out: string[] = [];
  if (want.deidentified_text !== got.deidentified_text) {
    const left = Buffer.from(want.deidentified_text, "utf8");
    const right = Buffer.from(got.deidentified_text, "utf8");
    let offset = 0;
    while (offset < Math.min(left.length, right.length) && left[offset] === right[offset]) {
      offset += 1;
    }
    out.push(`case=${id} path=$.deidentified_text first_differing_utf8_byte=${offset}`);
  }
  if (want.spans.length !== got.spans.length) {
    out.push(`case=${id} path=$.spans count expected=${want.spans.length} actual=${got.spans.length}`);
    return out;
  }
  want.spans.forEach((left, index) => {
    const right = got.spans[index] as unknown as Record<string, unknown>;
    const leftRecord = left as unknown as Record<string, unknown>;
    for (const field of [...new Set([...Object.keys(left), ...Object.keys(right)])].sort()) {
      if (leftRecord[field] === right[field]) {
        continue;
      }
      const path = `case=${id} path=$.spans[${index}].${field}`;
      out.push(
        field === "replacement"
          ? path
          : `${path} expected=${String(leftRecord[field])} actual=${String(right[field])}`,
      );
    }
  });
  return out.length > 0 ? out : [`case=${id} path=$ bytes differ`];
}

test("kit-parity vectors are digest-pinned and match the web profile constants", async () => {
  const fixture = await vectors();
  const bytes = await readFile(VECTORS_PATH);
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, "utf8")) as Record<string, any>;

  assert.deepEqual(fixture.profile.actions, [...KIT_PARITY_ACTIONS]);
  assert.deepEqual(fixture.profile.canonical_labels, [...KIT_PARITY_CANONICAL_LABELS]);
  assert.deepEqual(fixture.profile.offset_contract_versions, [
    ...KIT_PARITY_OFFSET_CONTRACT_VERSIONS,
  ]);
  assert.deepEqual(fixture.profile.decoder_contract_versions, [
    ...KIT_PARITY_DECODER_CONTRACT_VERSIONS,
  ]);
  const labels = new Set<string>(KIT_PARITY_CANONICAL_LABELS);
  assert.ok(CANONICAL_LABELS.every((label) => labels.has(label)));
  assert.ok(fixture.vectors.length >= 13);
  assert.ok(
    manifest.kit_parity.consumers.some(
      (consumer: { surface: string; test: string }) =>
        consumer.surface === "web" && consumer.test.endsWith("test_kit_parity.spec.ts"),
    ),
  );

  const tampered = Buffer.from(bytes);
  tampered[tampered.length - 2] = 0x20;
  assert.throws(() => loadVectors(tampered, manifest), { code: "digest_mismatch" });
  assert.throws(
    () => loadVectors(bytes, { kit_parity: { ...manifest.kit_parity, version: 2 } }),
    { code: "kit_vectors_unsupported" },
  );
});

test("scalar, UTF-16, and UTF-8 conversions match every derived offset", async () => {
  const fixture = await vectors();
  let sawUtf16Shift = false;
  let sawUtf8Shift = false;
  for (const vector of fixture.vectors) {
    for (const span of vector.spans) {
      for (const [scalar, utf8, utf16] of [
        [span.start, span.utf8_start, span.utf16_start],
        [span.end, span.utf8_end, span.utf16_end],
      ] as const) {
        const tag = `case=${vector.id}`;
        assert.equal(scalarToUtf16Offset(vector.text, scalar), utf16, tag);
        assert.equal(scalarToUtf8Offset(vector.text, scalar), utf8, tag);
        assert.equal(utf16ToScalarOffset(vector.text, utf16), scalar, tag);
        assert.equal(utf8ToScalarOffset(vector.text, utf8), scalar, tag);
        assert.equal(
          Buffer.byteLength(vector.text.slice(0, utf16), "utf8"),
          utf8,
          tag,
        );
      }
      sawUtf16Shift ||= span.utf16_start !== span.start;
      sawUtf8Shift ||= span.utf8_start !== span.utf16_start;
    }
  }
  assert.ok(sawUtf16Shift, "vectors lack a UTF-16 surrogate shift");
  assert.ok(sawUtf8Shift, "vectors lack a multibyte UTF-8 shift");

  for (const record of fixture.offset_rejections) {
    const convert = {
      scalar: scalarToUtf16Offset,
      utf8: utf8ToScalarOffset,
      utf16: utf16ToScalarOffset,
    }[record.unit];
    assert.throws(() => convert(record.text, record.offset), {
      name: "KitParityError",
      code: record.reason,
    });
  }
});

test("web results project to the canonical bytes of every vector", async () => {
  const fixture = await vectors();
  for (const vector of fixture.vectors) {
    const result = webResult(vector);
    const actual = serializeKitParityResult(toKitParityResult(result));

    assert.deepEqual(differences(vector.id, vector.result_json, actual), []);
    assert.equal(sha256(actual), vector.result_sha256, vector.id);
    assert.equal(sha256(vector.deidentified_text), vector.deidentified_sha256);

    // The web package's own redaction routine works on UTF-16 offsets and
    // has no keep semantics, so it receives only the replaced spans.
    const replaced = result.spans.filter((span) => span.action !== "keep");
    assert.equal(
      Buffer.compare(
        Buffer.from(spansToRedactedText(vector.text, replaced), "utf8"),
        Buffer.from(vector.deidentified_text, "utf8"),
      ),
      0,
      `case=${vector.id} de-identified UTF-8 bytes differ`,
    );
  }
});

test("canonical result JSON round-trips through the strict parser", async () => {
  const fixture = await vectors();
  for (const vector of fixture.vectors) {
    const parsed = parseKitParityResult(vector.result_json);
    const fromBytes = parseKitParityResult(new TextEncoder().encode(vector.result_json));

    assert.deepEqual(parsed, fromBytes);
    assert.equal(serializeKitParityResult(parsed), vector.result_json);
    assert.deepEqual(
      parsed.spans.map((span) => [span.label, span.canonical_label, span.action, span.start, span.end]),
      vector.spans.map((span) => [span.label, span.canonical_label, span.action, span.start, span.end]),
    );
    for (const [index, span] of parsed.spans.entries()) {
      assert.equal("confidence_ppm" in span, "confidence_ppm" in vector.spans[index]!);
      assert.equal("replacement" in span, span.action !== "keep");
    }
    verifyKitParityResult(parsed, vector.text);
  }
});

test("rejections fail with the documented value-free reason and field path", async () => {
  const fixture = await vectors();
  const sentinel = fixture.profile.privacy_sentinel;
  for (const record of fixture.rejections) {
    assert.throws(
      () => parseKitParityResult(record.input),
      (error: unknown) => {
        assert.ok(error instanceof KitParityError, record.id);
        assert.equal(error.code, record.reason, record.id);
        assert.equal(error.fieldPath, record.field_path, record.id);
        assert.equal(error.message, `${record.reason} at ${record.field_path}`);
        assert.equal(error.message.includes(sentinel), false);
        return true;
      },
    );
  }
  for (const record of fixture.source_rejections) {
    const parsed = parseKitParityResult(record.input);
    assert.throws(() => verifyKitParityResult(parsed, record.text), {
      code: record.reason,
      fieldPath: record.field_path,
    });
  }
});

test("bundle contracts are gated before any result is parsed", async () => {
  const fixture = await vectors();
  const payload = fixture.vectors.find((vector) => vector.id === "emoji_sequences")!.result_json;
  for (const record of fixture.compatibility) {
    const bundle = {
      offsetContractVersion: record.offset_contract_version,
      decoderContractVersion: record.decoder_contract_version,
    };
    if (record.outcome === "accepted") {
      assert.deepEqual(acceptKitParityResult(payload, bundle), parseKitParityResult(payload));
      continue;
    }
    for (const candidate of [payload, "not json", fixture.rejections[0]!.input]) {
      assert.throws(() => acceptKitParityResult(candidate, bundle), {
        code: record.outcome,
      });
    }
  }
});

test("an altered offset, label, action, order, or byte fails the gate", async () => {
  const fixture = await vectors();
  const vector = fixture.vectors.find((item) => item.id === "emoji_sequences")!;
  const project = (result: OpenMedDeidentifyResult) =>
    serializeKitParityResult(toKitParityResult(result));

  const shifted = webResult(vector);
  const last = shifted.spans.length - 1;
  shifted.spans[last] = { ...shifted.spans[last]!, end: shifted.spans[last]!.end + 1 };
  const firstEnd = vector.spans[0]!.end;
  assert.deepEqual(differences(vector.id, vector.result_json, project(shifted)), [
    `case=emoji_sequences path=$.spans[0].end expected=${firstEnd} actual=${firstEnd + 1}`,
  ]);

  const relabeled = webResult(vector);
  relabeled.spans[last] = { ...relabeled.spans[last]!, canonical_label: "PERSON" };
  assert.deepEqual(differences(vector.id, vector.result_json, project(relabeled)), [
    "case=emoji_sequences path=$.spans[0].canonical_label expected=OTHER actual=PERSON",
  ]);

  const reaction = webResult(vector);
  reaction.spans[1] = { ...reaction.spans[1]!, action: "redact" };
  assert.deepEqual(differences(vector.id, vector.result_json, project(reaction)), [
    "case=emoji_sequences path=$.spans[1].action expected=mask actual=redact",
  ]);

  const bytes = webResult(vector);
  bytes.deidentifiedText += " ";
  const changed = project(bytes);
  assert.match(
    differences(vector.id, vector.result_json, changed)[0]!,
    /^case=emoji_sequences path=\$\.deidentified_text first_differing_utf8_byte=\d+$/,
  );
  assert.throws(() => verifyKitParityResult(parseKitParityResult(changed), vector.text), {
    code: "deidentified_text_mismatch",
  });

  const swapped = JSON.parse(vector.result_json) as KitParityResult;
  swapped.spans.reverse();
  const reordered = serializeUnchecked(swapped);
  assert.throws(() => parseKitParityResult(reordered), {
    code: "span_order_invalid",
    fieldPath: "$.spans[1]",
  });

  const mixedUnits = webResult(vector);
  mixedUnits.spans = vector.spans.map((span) => ({
    ...webSpan(vector.text, span),
    // A surface that leaked scalar offsets into UTF-16 coordinates.
    start: span.start,
    end: span.end,
  }));
  assert.notEqual(project(mixedUnits), vector.result_json);
});

test("web projection rejects actions and offsets outside the profile", async () => {
  const fixture = await vectors();
  const vector = fixture.vectors.find((item) => item.id === "emoji_sequences")!;

  const formatPreserve = webResult(vector);
  formatPreserve.spans = [{ ...webSpan(vector.text, vector.spans[0]!), action: "format_preserve" }];
  assert.throws(() => toKitParityResult(formatPreserve), {
    code: "action_unknown",
    fieldPath: "$.spans[0].action",
  });

  const emojiText = "\u{1F600}x";
  const splitSurrogate: OpenMedDeidentifyResult = {
    text: emojiText,
    deidentifiedText: "[OTHER]",
    spans: [
      { ...webSpan(emojiText, { ...vector.spans[0]!, start: 0, end: 2 }), start: 1, end: 3 },
    ],
  };
  assert.throws(() => toKitParityResult(splitSurrogate), {
    code: "offset_not_scalar_boundary",
    fieldPath: "$.spans[0].start",
  });

  const unknownLabel = webResult(vector);
  unknownLabel.spans = [
    { ...webSpan(vector.text, vector.spans[0]!), canonical_label: "kitparity" as OpenMedSpan["canonical_label"] },
  ];
  assert.throws(() => toKitParityResult(unknownLabel), (error: unknown) => {
    assert.ok(error instanceof KitParityError);
    assert.equal(error.message, "label_unknown at $.spans[0].canonical_label");
    return true;
  });

  const noScore = webResult(vector);
  noScore.spans = [{ ...webSpan(vector.text, vector.spans[0]!), score: null }];
  noScore.deidentifiedText = vector.text.slice(0, scalarToUtf16Offset(vector.text, vector.spans[0]!.start)) +
    "[OTHER]" + vector.text.slice(scalarToUtf16Offset(vector.text, vector.spans[0]!.end));
  const projected = toKitParityResult(noScore);
  assert.equal("confidence_ppm" in projected.spans[0]!, false);
  verifyKitParityResult(projected, vector.text);
});

function serializeUnchecked(result: KitParityResult): string {
  const spans = result.spans.map((span) => {
    const ordered: Record<string, unknown> = {};
    for (const key of Object.keys(span).sort()) {
      ordered[key] = (span as unknown as Record<string, unknown>)[key];
    }
    return ordered;
  });
  return JSON.stringify({
    deidentified_text: result.deidentified_text,
    offset_unit: "unicode_scalar",
    profile: KIT_PARITY_PROFILE,
    profile_version: KIT_PARITY_PROFILE_VERSION,
    spans,
  });
}
