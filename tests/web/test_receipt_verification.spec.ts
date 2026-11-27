// Web verification of Python-signed audit reports against the shared vectors
// in tests/fixtures/parity/receipt_verification_v1.json, generated from
// openmed.core.audit.AuditReport by scripts/parity/generate_receipt_vectors.py.
import assert from "node:assert/strict";
import { createHash, createHmac } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  AUDIT_REPORT_SIGNATURE_ALGORITHM,
  AUDIT_REPORT_VERIFICATION_REASONS,
  AuditReceiptError,
  auditReportPayloads,
  canonicalAuditJson,
  verifyAuditReport,
  type AuditReportCrypto,
  type AuditReportVerificationReason,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "receipt_verification_v1.json");
const RECEIPT_SOURCE = join(rootDir, "js", "openmedkit-web", "src", "receipt.ts");

interface Candidate {
  span_order: "sorted" | "stored";
  repro_payload_b64: string;
  expected_repro_hash: string;
  signing_payload_b64: string;
  expected_hmac: string;
}

interface VectorCase {
  id: string;
  covers: string[];
  key_id: string;
  report: Record<string, unknown>;
  candidates: Candidate[];
  expected: {
    repro_hash_matches: boolean;
    verifies: boolean;
    matched_span_order: "sorted" | "stored" | null;
  };
}

interface Vectors {
  schema: string;
  version: number;
  synthetic_only: boolean;
  signature_algorithm: string;
  test_keys: { key_id: string; test_only: boolean; encoding: string; material: string }[];
  cases: VectorCase[];
}

// The verdict each vector must produce, with the reason codes the Swift and
// Kotlin verifiers share.
const EXPECTED_REASONS: Record<string, AuditReportVerificationReason> = {
  signed_baseline: "valid",
  unsigned: "signature_mismatch",
  tampered_offset: "hash_mismatch",
  tampered_label: "hash_mismatch",
  tampered_action: "hash_mismatch",
  tampered_signature_byte: "signature_mismatch",
  unknown_algorithm: "unsupported_algorithm",
  legacy_span_order: "valid",
  non_ascii_keys_and_values: "valid",
  float_edge_cases: "valid",
};

/** The exact source text of the JSON value at `path` (no re-encoding). */
function rawJson(text: string, path: readonly (string | number)[]): string {
  let index = 0;
  const whitespace = () => {
    while (index < text.length && " \t\n\r".includes(text[index]!)) index += 1;
  };
  const skip = (): void => {
    whitespace();
    const char = text[index];
    if (char === '"') {
      index += 1;
      while (text[index] !== '"') index += text[index] === "\\" ? 2 : 1;
      index += 1;
      return;
    }
    if (char === "{" || char === "[") {
      const close = char === "{" ? "}" : "]";
      index += 1;
      whitespace();
      if (text[index] === close) {
        index += 1;
        return;
      }
      for (;;) {
        if (char === "{") {
          skip();
          whitespace();
          index += 1;
        }
        skip();
        whitespace();
        if (text[index++] === close) return;
      }
    }
    while (index < text.length && !",}] \t\n\r".includes(text[index]!)) index += 1;
  };
  for (const step of path) {
    whitespace();
    const container = text[index++];
    assert.ok(container === "{" || container === "[", "path leaves the document");
    for (let position = 0; ; position += 1) {
      whitespace();
      if (container === "{") {
        const keyStart = index;
        skip();
        const key = JSON.parse(text.slice(keyStart, index)) as string;
        whitespace();
        index += 1;
        if (key === step) break;
      } else if (position === step) {
        break;
      }
      skip();
      whitespace();
      assert.equal(text[index++], ",", "path not found");
    }
  }
  whitespace();
  const start = index;
  skip();
  return text.slice(start, index);
}

let cached: Promise<{ text: string; vectors: Vectors }> | undefined;
function load() {
  cached ??= readFile(VECTORS_PATH, "utf8").then((text) => ({
    text,
    vectors: JSON.parse(text) as Vectors,
  }));
  return cached;
}

/** Each case's report exactly as the generator wrote it, float literals intact. */
async function reports(): Promise<{ item: VectorCase; raw: string }[]> {
  const { text, vectors } = await load();
  return vectors.cases.map((item, index) => ({ item, raw: rawJson(text, ["cases", index, "report"]) }));
}

async function material(item: VectorCase): Promise<string> {
  const { vectors } = await load();
  const key = vectors.test_keys.find((entry) => entry.key_id === item.key_id);
  assert.ok(key?.test_only, item.id);
  return key.material;
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

test("the vectors are the shared synthetic receipt contract", async () => {
  const { vectors } = await load();
  assert.equal(vectors.schema, "openmed.receipt_verification_vectors");
  assert.equal(vectors.version, 1);
  assert.equal(vectors.synthetic_only, true);
  assert.equal(vectors.signature_algorithm, AUDIT_REPORT_SIGNATURE_ALGORITHM);
  assert.ok(vectors.test_keys.length > 0);
  assert.ok(vectors.test_keys.every((key) => key.test_only && key.encoding === "utf-8"));
  assert.deepEqual(
    vectors.cases.map((item) => item.id).sort(),
    Object.keys(EXPECTED_REASONS).sort(),
  );
  for (const { item, raw } of await reports()) {
    assert.deepEqual(JSON.parse(raw), item.report, item.id);
  }
});

test("every vector yields byte-identical canonical payloads, hashes and HMACs", async () => {
  for (const { item, raw } of await reports()) {
    const key = await material(item);
    const payloads = auditReportPayloads(raw);
    assert.deepEqual(
      payloads.map((payload) => payload.spanOrder),
      item.candidates.map((candidate) => candidate.span_order),
      item.id,
    );
    payloads.forEach((payload, index) => {
      const candidate = item.candidates[index]!;
      const tag = `case=${item.id} order=${payload.spanOrder}`;
      assert.equal(
        Buffer.compare(Buffer.from(payload.repro), Buffer.from(candidate.repro_payload_b64, "base64")),
        0,
        `${tag} repro payload bytes differ`,
      );
      assert.equal(
        Buffer.compare(
          Buffer.from(payload.signing),
          Buffer.from(candidate.signing_payload_b64, "base64"),
        ),
        0,
        `${tag} signing payload bytes differ`,
      );
      assert.equal(`sha256:${sha256(payload.repro)}`, candidate.expected_repro_hash, tag);
      assert.equal(
        createHmac("sha256", key).update(payload.signing).digest("hex"),
        candidate.expected_hmac,
        tag,
      );
    });
    const stored = item.report.repro_hash;
    assert.equal(
      payloads.some((payload) => `sha256:${sha256(payload.repro)}` === stored),
      item.expected.repro_hash_matches,
      item.id,
    );
  }
});

test("every vector produces the expected verdict and reason code", async () => {
  for (const { item, raw } of await reports()) {
    const key = await material(item);
    const expected = {
      result: item.expected.verifies,
      reason: EXPECTED_REASONS[item.id],
      spanOrder: item.expected.matched_span_order,
    };
    assert.deepEqual(await verifyAuditReport(raw, key), expected, item.id);
    // The same report as UTF-8 bytes, as compact canonical text, and with a
    // byte key gives the same verdict.
    const bytes = new TextEncoder().encode(raw);
    assert.deepEqual(await verifyAuditReport(bytes, key), expected, item.id);
    assert.deepEqual(await verifyAuditReport(canonicalAuditJson(raw), key), expected, item.id);
    assert.deepEqual(
      await verifyAuditReport(raw, new TextEncoder().encode(key)),
      expected,
      item.id,
    );
    assert.ok(AUDIT_REPORT_VERIFICATION_REASONS.includes(expected.reason!), item.id);

    // Negative control: the wrong key never verifies.
    const wrong = await verifyAuditReport(raw, `${key}!`);
    assert.equal(wrong.result, false, item.id);
    if (item.expected.verifies) {
      assert.equal(wrong.reason, "signature_mismatch", item.id);
    }
  }
});

test("a report round-tripped through JSON.parse loses Python's float literals", async () => {
  const { item, raw } = (await reports()).find(({ item }) => item.id === "float_edge_cases")!;
  const key = await material(item);
  assert.match(raw, /1e-07/);
  assert.equal((await verifyAuditReport(raw, key)).result, true);
  // JSON.stringify prints 1e-7 and drops the ".0" of 1.0, so the hash differs.
  assert.deepEqual(await verifyAuditReport(JSON.stringify(JSON.parse(raw)), key), {
    result: false,
    reason: "hash_mismatch",
    spanOrder: null,
  });
});

test("tampering after signing fails with the specific reason code", async () => {
  const { item, raw } = (await reports()).find(({ item }) => item.id === "signed_baseline")!;
  const key = await material(item);
  const report = item.report as { signature: { value: string } };
  const variants: [string, AuditReportVerificationReason][] = [
    [raw.replace(/"document_length": (\d+)/, (_, value) => `"document_length": ${Number(value) + 1}`), "hash_mismatch"],
    [raw.replace(/^\{/, '{"appended_after_signing": 1, '), "hash_mismatch"],
    [raw.replace("HMAC-SHA256", "HMAC-SHA512"), "unsupported_algorithm"],
    [raw.replace(report.signature.value, report.signature.value.toUpperCase()), "signature_mismatch"],
    [raw.replace(report.signature.value, report.signature.value.slice(0, 63)), "signature_mismatch"],
    [raw.replace(/"signature": \{[^}]*\}/, '"signature": null'), "signature_mismatch"],
    [raw.replace(/"signature": \{[^}]*\}/, '"signature": "HMAC-SHA256"'), "malformed"],
    // A signature without a string value is malformed, as in Swift and Kotlin.
    [raw.replace(/"signature": \{[^}]*\}/, '"signature": {}'), "malformed"],
    [raw.replace(`"${report.signature.value}"`, "7"), "malformed"],
    [raw.replace(/"value": "[0-9a-f]{64}"/, '"other": "x"'), "malformed"],
  ];
  for (const [variant, reason] of variants) {
    assert.notEqual(variant, raw);
    assert.deepEqual(await verifyAuditReport(variant, key), { result: false, reason, spanOrder: null }, reason);
  }
});

test("malformed input returns malformed without echoing it", async () => {
  const sentinel = "Quorble Zyxwick MRN-PLANTED-0042";
  const { item, raw } = (await reports()).find(({ item }) => item.id === "signed_baseline")!;
  const key = await material(item);
  const inputs: (string | Uint8Array)[] = [
    `not json ${sentinel}`,
    `{"policy": "${sentinel}"`,
    `["${sentinel}"]`,
    `{"policy": "${sentinel}", "spans": [], "document_length": 1}`,
    `{"policy": "${sentinel}", "spans": "x", "document_length": 1, "repro_hash": "sha256:00"}`,
    raw.replace(/"document_length": \d+/, '"document_length": 64.0'),
    raw.replace(/"document_length": \d+/, '"document_length": 1'),
    raw.replace(/"start": 8/, '"start": -1'),
    raw.replace(/"start": 8/, '"start": 8.0'),
    raw.replace(/"confidence": 0.97/, '"confidence": NaN'),
    raw.replace(/"confidence": 0.97/, '"confidence": 1e400'),
    raw.replace(/"policy": "/, `"policy": "${sentinel}\u0001`),
    raw.replace(/"repro_hash": "[^"]*"/, '"repro_hash": ""'),
    `${raw} trailing`,
    "[".repeat(10_000),
    Uint8Array.of(0x7b, 0xff, 0x7d),
    // A byte order mark is not JSON, as text or as bytes.
    `﻿${raw}`,
    new TextEncoder().encode(`﻿${raw}`),
  ];
  for (const input of inputs) {
    const outcome = await verifyAuditReport(input, key);
    assert.deepEqual(outcome, { result: false, reason: "malformed", spanOrder: null });
    assert.ok(!JSON.stringify(outcome).includes(sentinel));
    assert.throws(() => auditReportPayloads(input), (error: unknown) => {
      assert.ok(error instanceof AuditReceiptError);
      assert.equal(error.code, "malformed");
      assert.equal(error.message, "malformed");
      return true;
    });
  }
});

test("verification uses crypto.subtle.verify, never a signature string comparison", async () => {
  const { item, raw } = (await reports()).find(({ item }) => item.id === "legacy_span_order")!;
  const key = await material(item);
  const subtle = globalThis.crypto.subtle;
  const calls: string[] = [];
  const spy: AuditReportCrypto = {
    digest: (...args) => {
      calls.push("digest");
      return subtle.digest(...args);
    },
    importKey: ((...args: Parameters<SubtleCrypto["importKey"]>) => {
      calls.push(`importKey:${JSON.stringify(args[2])}:${String(args[4])}`);
      return subtle.importKey(...(args as Parameters<SubtleCrypto["importKey"]>));
    }) as SubtleCrypto["importKey"],
    verify: (...args) => {
      calls.push(`verify:${String(args[0])}`);
      return subtle.verify(...args);
    },
  };
  assert.deepEqual(await verifyAuditReport(raw, key, { crypto: spy }), {
    result: true,
    reason: "valid",
    spanOrder: "stored",
  });
  // Sorted order fails its hash, the stored order matches and is verified.
  assert.deepEqual(calls, [
    "digest",
    "digest",
    'importKey:{"name":"HMAC","hash":"SHA-256"}:verify',
    "verify:HMAC",
  ]);

  // With a verifier that always says no, a correct signature does not pass.
  const refusing: AuditReportCrypto = { ...spy, verify: async () => false };
  assert.equal((await verifyAuditReport(raw, key, { crypto: refusing })).reason, "signature_mismatch");

  const source = await readFile(RECEIPT_SOURCE, "utf8");
  assert.doesNotMatch(source, /\bfrom\s+["']node:|require\(|import\(["']node:/);
  assert.doesNotMatch(source, /subtle\.sign|\["sign"\]|createHmac|timingSafeEqual/);
  assert.match(source, /subtle\.verify\("HMAC"/);
});

test("no key material appears in thrown errors", async () => {
  const { item, raw } = (await reports()).find(({ item }) => item.id === "signed_baseline")!;
  const key = "synthetic-key-sentinel-7b1f";
  const leaks = (error: unknown) =>
    error instanceof AuditReceiptError &&
    !error.message.includes(key) &&
    !String(error).includes(key) &&
    !JSON.stringify(error).includes(key) &&
    error.cause === undefined;

  await assert.rejects(verifyAuditReport(raw, ""), (error: unknown) => {
    assert.ok(leaks(error));
    assert.equal((error as AuditReceiptError).code, "key_invalid");
    return true;
  });
  await assert.rejects(verifyAuditReport(raw, new Uint8Array()), { code: "key_invalid" });
  await assert.rejects(
    verifyAuditReport(raw, 42 as unknown as string),
    { code: "key_invalid" },
  );

  const subtle = globalThis.crypto.subtle;
  const failing: AuditReportCrypto = {
    digest: (...args) => subtle.digest(...args),
    importKey: (async () => {
      throw new Error(`cannot import ${key}`);
    }) as SubtleCrypto["importKey"],
    verify: (...args) => subtle.verify(...args),
  };
  await assert.rejects(verifyAuditReport(raw, key, { crypto: failing }), (error: unknown) => {
    assert.ok(leaks(error));
    assert.equal((error as AuditReceiptError).code, "verifier_unavailable");
    return true;
  });
  await assert.rejects(
    verifyAuditReport(raw, key, { crypto: {} as AuditReportCrypto }),
    { code: "verifier_unavailable" },
  );

  // The caller's key bytes are left untouched; the verifier zeroes its copy.
  const bytes = new TextEncoder().encode(await material(item));
  const copy = Uint8Array.from(bytes);
  assert.equal((await verifyAuditReport(raw, bytes)).result, true);
  assert.deepEqual(bytes, copy);
});

test("the canonical encoder matches Python's _canonical_json", () => {
  // Expected bytes come from Python json.dumps(..., sort_keys=True,
  // separators=(",", ":"), ensure_ascii=True, allow_nan=False).
  const input =
    '{"b":1.0,"a":1e-07,"c":[5e-324,1e16,0.1,-0.0,1E5,10,-0,1e15,123.456,0.0001,' +
    "0.00001,2.5e-5,1e22,9999999999999998.0,12345678901234567890,0.30000000000000004]," +
    '"\\u00e9":"\\ud83d\\ude00","z\\u007f":"\\"\\\\\\n\\t\\u0001","\\uffff":1,' +
    '"\\ud83d\\ude01":2,"":null,"t":true,"f":false, "s" : "\\/x"}';
  assert.equal(
    canonicalAuditJson(input),
    '{"":null,"a":1e-07,"b":1.0,"c":[5e-324,1e+16,0.1,-0.0,100000.0,10,0,' +
      "1000000000000000.0,123.456,0.0001,1e-05,2.5e-05,1e+22,9999999999999998.0," +
      '12345678901234567890,0.30000000000000004],"f":false,"s":"/x","t":true,' +
      '"z\\u007f":"\\"\\\\\\n\\t\\u0001","\\u00e9":"\\ud83d\\ude00","\\uffff":1,' +
      '"\\ud83d\\ude01":2}',
  );
  // Code-point order puts U+FFFF before U+1F601; UTF-16 order would not.
  assert.ok("\u{1F601}" < "￿");
  assert.equal(canonicalAuditJson('"é\u{1F600}"'), '"\\u00e9\\ud83d\\ude00"');
  assert.equal(canonicalAuditJson('{"k":"\\udc00"}'), '{"k":"\\udc00"}');
  assert.equal(canonicalAuditJson('{"k":1,"k":2}'), '{"k":2}');
  for (const bad of ["NaN", "Infinity", "1e400", "01", "1.", ".5", "+1", "'x'", '"\t"', "[1,]"]) {
    assert.throws(() => canonicalAuditJson(bad), { code: "malformed" }, bad);
  }
});
