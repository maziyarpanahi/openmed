// Node tests for the React Native parity harness under
// js/openmedkit-react-native/example. They drive the harness with JavaScript
// fakes of OpenMedKitRN and of the direct probe. That proves the comparison,
// privacy and evidence logic; it is not release evidence, which only the
// simulator and emulator jobs produce from real native modules.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash, createHmac, randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import * as bridgeApi from "../../js/openmedkit-react-native/src/index";
import type {
  OpenMedKitNativeModule,
  OpenMedKitNativeModuleSource,
} from "../../js/openmedkit-react-native/src/index";
import {
  canonicalDigest,
  canonicalJson,
  replaceScalarRange,
  scalarLength,
  scalarSlice,
} from "../../js/openmedkit-react-native/example/src/parity/canonical";
import {
  CorpusError,
  extractCases,
  loadCorpora,
  parseCorpusManifest,
  type CorpusManifest,
} from "../../js/openmedkit-react-native/example/src/parity/corpus";
import {
  CORPUS_MANIFEST,
  CORPUS_REGISTRY,
} from "../../js/openmedkit-react-native/example/src/parity/corpusRegistry";
import {
  diffSpanLists,
  diffValues,
} from "../../js/openmedkit-react-native/example/src/parity/diff";
import { checkEvidenceReport } from "../../js/openmedkit-react-native/example/src/parity/evidence";
import {
  bundleDigest,
  validateFixtureManifest,
} from "../../js/openmedkit-react-native/example/src/parity/fixture";
import {
  findReportLeaks,
  findSurfaceLeaks,
  installEgressGuard,
} from "../../js/openmedkit-react-native/example/src/parity/privacy";
import {
  parseDirectResponse,
  resolveNativeProbe,
  type DirectProbe,
  type DirectRequest,
} from "../../js/openmedkit-react-native/example/src/parity/probe";
import {
  reconstructRedaction,
  runParity,
  type BridgeApi,
  type ParityReport,
} from "../../js/openmedkit-react-native/example/src/parity/runner";
import { sha256Hex } from "../../js/openmedkit-react-native/example/src/parity/sha256";
import {
  loadModelSources,
  verifyFixture,
  writeFixtureManifest,
} from "../../js/openmedkit-react-native/example/scripts/stage-model-fixture";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const exampleDir = join(rootDir, "js", "openmedkit-react-native", "example");

type Platform = "ios" | "android";

interface Entity {
  label: string;
  start: number;
  end: number;
  confidence: number;
}

interface OffsetCase {
  id: string;
  text: string;
  expected_start: number;
  expected_end: number;
}

const offsetCases = (CORPUS_REGISTRY.offset_contract as { cases: OffsetCase[] }).cases;
const offsetDetections = new Map(
  offsetCases.map((item) => [item.text, { start: item.expected_start, end: item.expected_end }]),
);
const allCorpusTexts = offsetCases.map((item) => item.text);

test.afterEach(() => {
  bridgeApi.setOpenMedKitNativeModuleForTests(null);
});

test("sha256 and canonical JSON are stable and match node:crypto", () => {
  for (const sample of ["", "abc", "患者𠀀今日复诊", "Clinician 👩‍⚕️", "x".repeat(200)]) {
    assert.equal(sha256Hex(sample), createHash("sha256").update(sample, "utf8").digest("hex"));
  }
  assert.equal(
    sha256Hex("lone \ud800 surrogate"),
    createHash("sha256").update("lone � surrogate", "utf8").digest("hex"),
  );
  assert.equal(canonicalJson({ b: 1, a: [true, null, "é"], c: undefined }), '{"a":[true,null,"é"],"b":1}');
  assert.equal(canonicalDigest({ a: 1, b: 2 }), canonicalDigest({ b: 2, a: 1 }));
  assert.equal(scalarLength("𠀀ab"), 3);
  assert.equal(scalarSlice("患者𠀀今日", 2, 3), "𠀀");
  assert.equal(replaceScalarRange("a𠀀b", 1, 2, "[X]"), "a[X]b");
});

test("the corpus manifest pins every bundled corpus by canonical digest", async () => {
  const manifest = parseCorpusManifest(CORPUS_MANIFEST);
  assert.deepEqual(
    Object.keys(CORPUS_REGISTRY).sort(),
    manifest.corpora.map((entry) => entry.id).sort(),
  );
  for (const entry of manifest.corpora) {
    const onDisk = JSON.parse(await readFile(join(rootDir, entry.path), "utf8"));
    assert.equal(canonicalDigest(onDisk), entry.canonical_sha256, entry.id);
  }
  const corpora = loadCorpora(manifest, CORPUS_REGISTRY);
  const offset = corpora.find((corpus) => corpus.id === "offset_contract");
  assert.ok(offset);
  assert.ok(offset.cases.length >= 52);
  assert.equal(new Set(offset.cases.map((item) => item.caseId)).size, offset.cases.length);
  assert.ok(offset.cases.every((item) => item.policy === manifest.defaults.policy));
  assert.ok(
    offset.cases.some((item) => item.text.length > scalarLength(item.text)),
    "the corpus must exercise supplementary-plane scalars",
  );
});

test("corpus loading fails closed on drift, gaps, and malformed cases", () => {
  const manifest = parseCorpusManifest(CORPUS_MANIFEST);
  const original = CORPUS_REGISTRY.offset_contract as { cases: OffsetCase[]; offset_unit: string };

  const tampered = structuredClone(original);
  tampered.cases[0]!.text = `${tampered.cases[0]!.text} `;
  assertCorpusError(() => loadCorpora(manifest, { offset_contract: tampered }), "corpus_digest_mismatch");
  assertCorpusError(() => loadCorpora(manifest, {}), "corpus_missing");

  const utf16 = { ...structuredClone(original), offset_unit: "utf16" };
  const retargeted = withCorpusDigest(manifest, utf16);
  assertCorpusError(() => loadCorpora(retargeted, { offset_contract: utf16 }), "corpus_offset_unit_mismatch");

  const duplicated = structuredClone(original);
  duplicated.cases[1]!.id = duplicated.cases[0]!.id;
  assertCorpusError(
    () => loadCorpora(withCorpusDigest(manifest, duplicated), { offset_contract: duplicated }),
    "corpus_case_duplicate",
  );

  const badSelector = structuredClone(CORPUS_MANIFEST) as { corpora: Array<{ select: { text: string } }> };
  badSelector.corpora[0]!.select.text = "../text";
  assertCorpusError(() => parseCorpusManifest(badSelector), "corpus_manifest_invalid");
});

test("pending kit-parity and portable-policy vectors can be dropped in by selector", async () => {
  const manifest = parseCorpusManifest(CORPUS_MANIFEST);
  const kit = manifest.pending.find((entry) => entry.id === "kit_parity_vectors");
  const portable = manifest.pending.find((entry) => entry.id === "portable_policy_vectors");
  assert.ok(kit && portable);

  const kitCases = extractCases(
    kit.id,
    kit.select,
    {
      schema: "openmed.kit_parity_vectors",
      vectors: [{ id: "ascii", text: "Ada seen today", engine: { policy: "hipaa_safe_harbor" } }],
    },
    manifest.defaults,
  );
  assert.deepEqual(kitCases.map((item) => [item.caseId, item.policy]), [["ascii", "hipaa_safe_harbor"]]);

  const portableCases = extractCases(
    portable.id,
    portable.select,
    {
      schema: "openmed.portable_policy_vectors",
      decision_cases: [{ id: "edge", profile: "strict_no_leak", text: "Seen 2026-03-02" }],
    },
    manifest.defaults,
  );
  assert.deepEqual(portableCases.map((item) => [item.caseId, item.policy]), [["edge", "strict_no_leak"]]);

  // Once the newer vector files land, their real shape must extract as well.
  for (const entry of manifest.pending) {
    const path = join(rootDir, entry.path);
    if (existsSync(path)) {
      const cases = extractCases(entry.id, entry.select, JSON.parse(await readFile(path, "utf8")), manifest.defaults);
      assert.ok(cases.length > 0, entry.id);
    }
  }
});

test("field diffs name paths and carry digests, never values", () => {
  const policy = { scoreAbsTolerance: 1e-6 };
  const base = [wireSpan({ label: "FIRST_NAME", start: 8, end: 11, confidence: 0.9 }, "Patient Ana arrived", "s3cret", "ios")];
  const clone = () => structuredClone(base);

  assert.deepEqual(diffSpanLists(base, clone(), policy, "$.spans"), []);

  const cases: Array<[(span: Record<string, unknown>) => void, string, string]> = [
    [(span) => { span.start = 7; }, "$.spans[0].start", "value"],
    [(span) => { span.canonical_label = "LAST_NAME"; }, "$.spans[0].canonical_label", "value"],
    [(span) => { span.action = "mask"; }, "$.spans[0].action", "value"],
    [(span) => { span.replacement = "[FIRST_NAME]"; }, "$.spans[0].replacement", "type"],
    [(span) => { span.score = 0.9 + 1e-3; }, "$.spans[0].score", "score_tolerance"],
    [(span) => { delete span.section; }, "$.spans[0].section", "missing_in_direct"],
  ];
  for (const [mutate, path, kind] of cases) {
    const direct = clone();
    mutate(direct[0] as Record<string, unknown>);
    const mismatches = diffSpanLists(base, direct, policy, "$.spans");
    assert.deepEqual(mismatches.map((item) => [item.path, item.kind]), [[path, kind]]);
    const encoded = JSON.stringify(mismatches);
    for (const value of ["Ana", "[FIRST_NAME]", "LAST_NAME", "FIRST_NAME", "mask"]) {
      assert.equal(encoded.includes(value), false, `${path} leaked ${value}`);
    }
  }

  const withinTolerance = clone();
  (withinTolerance[0] as Record<string, unknown>).score = 0.9 + 1e-8;
  assert.deepEqual(diffSpanLists(base, withinTolerance, policy, "$.spans"), []);

  const second = wireSpan({ label: "LAST_NAME", start: 12, end: 15, confidence: 0.8 }, "Patient Ana Lee", "s3cret", "ios");
  const reordered = diffSpanLists([base[0], second], [second, base[0]], policy, "$.spans");
  assert.deepEqual(reordered.map((item) => [item.path, item.kind]), [["$.spans", "order"]]);

  const dropped = diffSpanLists([base[0], second], [base[0]], policy, "$.spans");
  assert.deepEqual(dropped.map((item) => [item.path, item.kind]), [
    ["$.spans", "length"],
    ["$.spans[1]", "missing_in_direct"],
  ]);

  const textDiff = diffValues({ deidentifiedText: "Patient [X]" }, { deidentifiedText: "Patient Ana" }, policy);
  assert.equal(textDiff[0]?.path, "$.deidentifiedText");
  assert.equal(textDiff[0]?.bridge.length, 11);
  assert.equal(JSON.stringify(textDiff).includes("Ana"), false);
});

test("privacy scans find leaked surfaces without flagging digests", () => {
  const text = "Patient Ana arrived on 2026-03-02";
  const clean = [
    wireSpan({ label: "FIRST_NAME", start: 8, end: 11, confidence: 0.9 }, text, "k", "ios"),
    { ...wireSpan({ label: "DATE", start: 23, end: 33, confidence: 0.9 }, text, "k", "ios"), replacement: "DATE_3a5f00c0ffee1234abcd" },
  ];
  assert.deepEqual(findSurfaceLeaks(clean, text), []);

  const leaked = structuredClone(clean);
  (leaked[0] as { metadata: Record<string, string> }).metadata = { note: "saw Ana" };
  assert.deepEqual(findSurfaceLeaks(leaked, text), ["$.spans[0].metadata.note"]);

  assert.deepEqual(
    findReportLeaks({ case_id: "Ana", notes: ["fine", "has Ana inside"], digest: "sha256:" + "a".repeat(64) }, ["Ana"]),
    ["$.notes[1]"],
  );
});

test("the egress guard blocks and counts JavaScript network calls, then restores", async () => {
  const scope: Record<string, unknown> = { fetch: () => "real" };
  const guard = installEgressGuard(scope);
  await assert.rejects((scope.fetch as () => Promise<unknown>)());
  assert.throws(() => new (scope.XMLHttpRequest as new () => unknown)());
  assert.throws(() => new (scope.WebSocket as new () => unknown)());
  assert.equal(guard.attempts(), 3);
  guard.restore();
  assert.equal((scope.fetch as () => string)(), "real");
  assert.equal("WebSocket" in scope, false);
});

for (const platform of ["ios", "android"] as const) {
  test(`${platform} bridge matches the direct probe on every shared case`, async () => {
    const harness = createHarness(platform);
    const report = await runParity(harness.options("development"));

    assert.equal(report.status, "pass", JSON.stringify(report.failure_reasons));
    assert.equal(report.platform, platform);
    assert.equal(report.runtime_backend, platform === "ios" ? "coreml" : "onnx");
    assert.equal(report.bridge_source, "bridge-module");
    assert.equal(report.evidence_eligible, false, "development runs are never evidence");
    assert.equal(report.cases.total, offsetCases.length + 1);
    assert.equal(report.cases.mismatched, 0);
    for (const operation of ["analyzeText", "extractPii", "deidentify"] as const) {
      assert.equal(report.operations[operation]?.mismatched, 0);
      assert.equal(report.operations[operation]?.error_parity, 0);
    }
    assert.deepEqual(outcomes(report), {
      release_guard: "not_applicable",
      analyze_before_load_rejects: "pass",
      model_load_initial: "pass",
      model_load_reuse: "pass",
      model_replacement: "pass",
      model_replacement_reuse: "pass",
      cancellation: "not_covered",
      bridge_invalidation: "not_covered",
      background_expiration: "not_covered",
      memory_pressure: "not_covered",
      generation_isolation: "not_covered",
    });
    assert.equal(report.egress.blocked_attempts, 0);
    assert.ok(report.sentinel.bridge_results_scanned > 0);
    assertValueFree(report, harness);
    assert.equal(report.report_digest, canonicalDigest({ ...report, report_digest: undefined }));
  });
}

test("UTF-16 surface hashing in a bridge is caught on supplementary-plane cases", async () => {
  const harness = createHarness("android", { bridgeSurfaceUnits: "utf16" });
  const report = await runParity(harness.options("development"));

  assert.equal(report.status, "fail");
  assert.ok(report.failure_reasons.includes("parity_mismatch"));
  const astralCases = new Set(
    offsetCases
      .filter((item) => {
        const detection = offsetDetections.get(item.text)!;
        return scalarSlice(item.text, 0, detection.end).length !== detection.end;
      })
      .map((item) => item.id),
  );
  assert.ok(astralCases.size > 0);
  const flagged = new Set(report.mismatches.map((item) => item.case_id));
  assert.deepEqual([...flagged].sort(), [...astralCases].sort());
  assert.ok(report.mismatches.every((item) => item.path.endsWith(".text_hash") && item.kind === "value"));
  assertValueFree(report, harness);
});

test("deliberate field, offset, label, action and score drift fails with value-free paths", async () => {
  const drifts: Array<[string, (span: Record<string, unknown>) => void, string, string]> = [
    ["offset", (span) => { span.end = (span.end as number) - 1; }, ".end", "value"],
    ["label", (span) => { span.canonical_label = "OTHER"; }, ".canonical_label", "value"],
    ["action", (span) => { span.action = "redact"; }, ".action", "value"],
    ["score", (span) => { span.score = (span.score as number) + 0.01; }, ".score", "score_tolerance"],
    ["field", (span) => { span.section = "notes"; }, ".section", "type"],
  ];
  for (const [name, mutate, suffix, kind] of drifts) {
    const harness = createHarness("ios", { mutateBridgeSpan: mutate });
    const report = await runParity(harness.options("development"));
    assert.equal(report.status, "fail", name);
    assert.ok(report.mismatches.length > 0, name);
    assert.ok(
      report.mismatches.every((item) => item.path.endsWith(suffix) && item.kind === kind),
      `${name}: ${JSON.stringify(report.mismatches.slice(0, 2).map((item) => [item.path, item.kind]))}`,
    );
    assertValueFree(report, harness);
  }
});

test("a direct-side ordering difference is reported once, not per field", async () => {
  const target = offsetCases.find((item) => item.id === "joiner-combining-latin-01");
  assert.ok(target && target.expected_start >= 7);
  const harness = createHarness("ios", {
    extraEntities: (text) =>
      text === target.text ? [{ label: "LAST_NAME", start: 0, end: 7, confidence: 0.7 }] : [],
    reverseDirect: true,
  });
  const report = await runParity(harness.options("development"));
  const kinds = new Set(report.mismatches.map((item) => `${item.path}:${item.kind}`));
  assert.deepEqual([...kinds], ["$.spans:order"]);
});

test("missing native de-identified text fails closed and surfaces as an outcome mismatch", async () => {
  const harness = createHarness("ios", { dropDeidentifiedText: true });
  const report = await runParity(harness.options("development"));
  assert.equal(report.status, "fail");
  const deid = report.mismatches.filter((item) => item.operation === "deidentify");
  assert.ok(deid.length > 0);
  assert.ok(deid.every((item) => item.path === "$" && item.kind === "outcome"));
  assert.equal(report.operations.deidentify?.matched, 0);
});

test("an unverified model bundle stops the run before any comparison", async () => {
  const harness = createHarness("ios", { fixtureError: "model_fixture_digest_mismatch" });
  const report = await runParity(harness.options("development"));
  assert.equal(report.status, "fail");
  assert.deepEqual(report.failure_reasons, ["model_fixture_digest_mismatch"]);
  assert.equal(report.cases.total, 0);
  assert.equal(harness.directCalls(), 0);
});

test("a JavaScript test double can never produce a passing parity report", async () => {
  const harness = createHarness("ios", { bridgeSource: "test-double" });
  const report = await runParity(harness.options("development"));
  assert.equal(report.status, "fail");
  assert.deepEqual(report.failure_reasons, ["bridge_not_native_test_double"]);
  assert.equal(report.evidence_eligible, false);
  assert.equal(harness.directCalls(), 0);
});

test("lifecycle regressions in load reuse and pre-load calls are reported", async () => {
  const harness = createHarness("android", { alwaysReportLoaded: true, answerBeforeLoad: true });
  const report = await runParity(harness.options("development"));
  assert.equal(report.status, "fail");
  assert.ok(report.failure_reasons.includes("lifecycle_check_failed"));
  const byCheck = Object.fromEntries(report.lifecycle.map((item) => [item.check, item]));
  assert.equal(byCheck.analyze_before_load_rejects?.outcome, "fail");
  assert.equal(byCheck.model_load_reuse?.outcome, "fail");
  assert.deepEqual(byCheck.model_load_reuse?.mismatched_paths, ["$.loaded"]);
});

test("JavaScript network attempts during a run fail the report", async () => {
  const harness = createHarness("ios", { touchNetwork: true });
  const report = await runParity(harness.options("development"));
  assert.equal(report.status, "fail");
  assert.ok(report.failure_reasons.includes("network_egress_attempted"));
  assert.ok(report.egress.blocked_attempts > 0);
  assert.equal(typeof globalThis.fetch, "function", "the guard restores fetch");
});

test("release runs prove the double guard and become evidence only when everything passes", async () => {
  const harness = createHarness("ios", { releaseBridge: "guarded", buildConfiguration: "release" });
  const report = await runParity(harness.options("release"));
  assert.equal(report.status, "pass", JSON.stringify(report.failure_reasons));
  assert.equal(outcomes(report).release_guard, "pass");
  assert.equal(report.evidence_eligible, true);

  const manifest = parseCorpusManifest(CORPUS_MANIFEST);
  const corpora = loadCorpora(manifest, CORPUS_REGISTRY);
  assert.deepEqual(checkEvidenceReport(report, manifest, corpora, "ios"), { ok: true, problems: [] });
  assert.ok(checkEvidenceReport(report, manifest, corpora, "android").problems.includes("report_platform_unexpected"));

  const tampered = structuredClone(report);
  tampered.cases.matched += 1;
  assert.ok(checkEvidenceReport(tampered, manifest, corpora, "ios").problems.includes("report_digest_mismatch"));

  const leaking = structuredClone(report) as ParityReport & { note?: string };
  leaking.note = offsetCases[0]!.text;
  delete leaking.report_digest;
  leaking.report_digest = canonicalDigest(leaking);
  assert.ok(checkEvidenceReport(leaking, manifest, corpora, "ios").problems.includes("report_contains_source_text"));

  const broken = createHarness("ios", { releaseBridge: "unguarded", buildConfiguration: "release" });
  const brokenReport = await runParity(broken.options("release"));
  assert.equal(outcomes(brokenReport).release_guard, "fail");
  assert.equal(brokenReport.status, "fail");
  assert.equal(brokenReport.evidence_eligible, false);
});

test("development and debug-native reports are rejected as evidence", async () => {
  const manifest = parseCorpusManifest(CORPUS_MANIFEST);
  const corpora = loadCorpora(manifest, CORPUS_REGISTRY);
  const development = await runParity(createHarness("android").options("development"));
  const problems = checkEvidenceReport(development, manifest, corpora, "android").problems;
  assert.ok(problems.includes("report_not_release_build"));
  assert.ok(problems.includes("report_not_evidence_eligible"));
  assert.ok(problems.includes("report_lifecycle_release_guard_not_pass"));

  const debugNative = await runParity(
    createHarness("android", { releaseBridge: "guarded", buildConfiguration: "debug" }).options("release"),
  );
  assert.equal(debugNative.evidence_eligible, false);
});

test("redaction reconstruction applies replacements at Unicode scalar offsets", () => {
  const text = "患者𠀀今日复诊";
  assert.equal(
    reconstructRedaction(text, [{ start: 2, end: 3, replacement: "[NAME]" }]),
    "患者[NAME]今日复诊",
  );
  assert.equal(
    reconstructRedaction(text, [
      { start: 0, end: 2, replacement: null },
      { start: 2, end: 3, replacement: "" },
    ]),
    "患者今日复诊",
  );
  assert.equal(reconstructRedaction(text, [{ start: 0, end: 3, replacement: "" }, { start: 2, end: 4, replacement: "" }]), null);
});

test("probe responses and module resolution are validated", () => {
  assert.deepEqual(parseDirectResponse("not json"), { ok: false, error: { code: "direct_response_invalid" } });
  assert.deepEqual(parseDirectResponse('{"ok":false,"error":{"code":"Bad Code with text"}}'), {
    ok: false,
    error: { code: "unknown_error" },
  });
  assert.deepEqual(parseDirectResponse('{"ok":true,"result":{"spans":[]}}'), { ok: true, result: { spans: [] } });
  assert.equal(resolveNativeProbe({ OpenMedKitParityProbe: { runDirect() {} } }), null);
  assert.equal(resolveNativeProbe(undefined), null);
});

test("model fixture manifests are digest-bound and staged trees are re-verified", async () => {
  assert.equal(
    bundleDigest([
      { path: "model/b.json", sha256: "b".repeat(64), bytes: 2 },
      { path: "model/a.onnx", sha256: "a".repeat(64), bytes: 1 },
    ]),
    `sha256:${createHash("sha256")
      .update(`model/a.onnx\t${"a".repeat(64)}\t1\nmodel/b.json\t${"b".repeat(64)}\t2\n`)
      .digest("hex")}`,
  );

  const dir = await mkdtemp(join(tmpdir(), "openmed-rn-fixture-"));
  try {
    await mkdir(join(dir, "model", "tokenizer"), { recursive: true });
    await writeFile(join(dir, "model", "model.onnx"), "synthetic-weights");
    await writeFile(join(dir, "model", "id2label.json"), '{"0":"O"}');
    await writeFile(join(dir, "model", "tokenizer", "tokenizer.json"), "{}");
    const manifest = await writeFixtureManifest(dir, {
      platform: "android",
      backend: "onnx",
      model_path: "model",
      id2label_path: "model/id2label.json",
      tokenizer_folder_path: null,
      tokenizer_name: null,
      source: { repo: "synthetic/fixture", revision: "0".repeat(40) },
    });
    assert.equal(manifest.files.length, 3);
    assert.equal((await verifyFixture(dir)).bundle_digest, manifest.bundle_digest);

    await writeFile(join(dir, "model", "extra.bin"), "unlisted");
    await assert.rejects(verifyFixture(dir), /file set/);
    await rm(join(dir, "model", "extra.bin"));
    await writeFile(join(dir, "model", "model.onnx"), "tampered-weights!");
    await assert.rejects(verifyFixture(dir), /digest/);

    assert.throws(
      () => validateFixtureManifest({ ...manifest, files: [{ ...manifest.files[0]!, path: "model/../x" }] }),
      /fixture_manifest_file_invalid/,
    );
    assert.throws(
      () => validateFixtureManifest({ ...manifest, bundle_digest: `sha256:${"0".repeat(64)}` }),
      /fixture_bundle_digest_mismatch/,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("pinned model sources use immutable revisions and permissive licenses", async () => {
  const sources = await loadModelSources();
  const repos = [sources.android, sources.ios.tokenizer];
  if (sources.ios.coreml !== null) {
    repos.push(sources.ios.coreml);
  }
  for (const repo of repos) {
    assert.match(repo.revision, /^[0-9a-f]{40}$/);
    assert.equal((repo as { license?: string }).license, "apache-2.0");
    assert.ok(repo.files.length > 0);
    for (const file of repo.files) {
      assert.match(file.sha256, /^[0-9a-f]{64}$/);
      assert.ok(Number.isInteger(file.bytes) && file.bytes > 0);
    }
  }
  assert.ok(sources.android.files.some((file) => file.path === "model.onnx"));
  for (const name of ["tokenizer.json", "tokenizer_config.json"]) {
    assert.ok(sources.ios.tokenizer.files.some((file) => file.path === name), name);
  }
});

test("the iOS fixture fails closed while no published Core ML export is pinned", async () => {
  const sources = await loadModelSources();
  if (sources.ios.coreml !== null) {
    return;
  }
  const dir = await mkdtemp(join(tmpdir(), "openmed-rn-ios-"));
  try {
    const run = (args: string[]) =>
      spawnSync(
        process.execPath,
        [...process.execArgv, join(exampleDir, "scripts", "stage-model-fixture.ts"), ...args],
        { encoding: "utf8" },
      );
    for (const args of [
      ["download", "--platform", "ios", "--cache", dir],
      ["assemble", "--platform", "ios", "--source", dir, "--out", join(dir, "out")],
    ]) {
      const result = run(args);
      assert.equal(result.status, 1, result.stderr);
      assert.match(result.stderr, /no pinned Core ML export/);
    }
    assert.equal(existsSync(join(dir, "tokenizer")), false, "nothing is downloaded");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("native probes mirror OpenMedKit tables and call it offline at scalar offsets", async () => {
  const read = (path: string) => readFile(join(rootDir, path), "utf8");
  const swiftProbe = await read("js/openmedkit-react-native/example/probe/ios/OpenMedKitParityProbe.swift");
  const objcExport = await read("js/openmedkit-react-native/example/probe/ios/OpenMedKitParityProbe.m");
  const kotlinProbe = await read(
    "js/openmedkit-react-native/example/probe/android/src/main/java/com/openmed/openmedkit/parityprobe/OpenMedKitParityProbeModule.kt",
  );
  const swiftPolicy = await read("swift/OpenMedKit/Sources/OpenMedKit/Policy.swift");
  const kotlinPolicy = await read(
    "android/openmedkit/src/main/kotlin/com/openmed/openmedkit/policy/PolicyProfiles.kt",
  );

  for (const name of ["clinicalConceptLabels", "quasiIdentifierLabels"]) {
    assert.deepEqual(labelSet(swiftProbe, name), labelSet(swiftPolicy, name), name);
  }
  for (const name of ["CLINICAL_CONCEPT_LABELS", "QUASI_IDENTIFIER_LABELS"]) {
    assert.deepEqual(labelSet(kotlinProbe, name), labelSet(kotlinPolicy, name), name);
  }

  for (const method of [
    "describeEnvironment",
    "prepareModelFixture",
    "createEphemeralHashSecret",
    "runDirect",
    "writeReport",
  ]) {
    assert.match(objcExport, new RegExp(`RCT_EXTERN_METHOD\\(${method}\\b`), method);
    assert.match(swiftProbe, new RegExp(`@objc\\(${method}:`), method);
    assert.match(kotlinProbe, new RegExp(`@ReactMethod\\s+fun ${method}\\(`), method);
  }

  assert.ok(!/OpenMedKitRN/i.test(codeOnly(swiftProbe)), "the iOS probe must not call the bridge");
  assert.ok(!/OpenMedKitRN/i.test(codeOnly(kotlinProbe)), "the Android probe must not call the bridge");
  assert.equal(swiftProbe.match(/allowNetworkAccess: false/g)?.length, 2);
  assert.ok(!swiftProbe.includes("allowNetworkAccess: true"));
  assert.match(swiftProbe, /text\.unicodeScalars/);
  assert.match(kotlinProbe, /UnicodeOffsetContract\.substring\(text, start, end\)/);
});

// ---------------------------------------------------------------------------
// Fakes

interface HarnessConfig {
  bridgeSurfaceUnits?: "scalar" | "utf16";
  mutateBridgeSpan?: (span: Record<string, unknown>) => void;
  extraEntities?: (text: string) => Entity[];
  reverseDirect?: boolean;
  dropDeidentifiedText?: boolean;
  fixtureError?: string;
  bridgeSource?: OpenMedKitNativeModuleSource;
  alwaysReportLoaded?: boolean;
  answerBeforeLoad?: boolean;
  touchNetwork?: boolean;
  /** Use a hand-written bridge object whose double guard behaves as stated. */
  releaseBridge?: "guarded" | "unguarded";
  buildConfiguration?: "debug" | "release";
}

interface Harness {
  options(jsMode: "release" | "development"): Parameters<typeof runParity>[0];
  directCalls(): number;
  fixturePaths: string[];
  secrets: string[];
}

function createHarness(platform: Platform, config: HarnessConfig = {}): Harness {
  const fixturePaths =
    platform === "ios"
      ? ["/sandbox/Documents/openmed-parity-fixture/model/model.mlpackage", "/sandbox/Documents/openmed-parity-fixture/model/id2label.json", "/sandbox/Documents/openmed-parity-fixture/model/tokenizer"]
      : ["/storage/emulated/0/Android/data/org.openmed/files/openmed-parity-fixture/model", "/storage/emulated/0/Android/data/org.openmed/files/openmed-parity-fixture/model/id2label.json"];
  const secrets: string[] = [];
  let directCalls = 0;

  const detect = (text: string): Entity[] => {
    const hit = offsetDetections.get(text);
    const base = hit ? [{ label: "FIRST_NAME", ...hit, confidence: Math.fround(0.9876543) }] : [];
    return [...base, ...(config.extraEntities?.(text) ?? [])].sort((a, b) => a.start - b.start);
  };

  const native = createNativeBridge(platform, detect, config);
  const probe: DirectProbe = {
    async describeEnvironment() {
      return {
        platform,
        runtime_backend: platform === "ios" ? "coreml" : "onnx",
        device_class: platform === "ios" ? "simulator" : "emulator",
        os_version: platform === "ios" ? "18.2.0" : "15 (API 35)",
        build_configuration: config.buildConfiguration ?? "debug",
      };
    },
    async prepareModelFixture() {
      if (config.fixtureError) {
        throw Object.assign(new Error("model fixture verification failed"), { code: config.fixtureError });
      }
      return {
        modelPath: fixturePaths[0]!,
        backend: platform === "ios" ? "coreml" : "onnx",
        id2LabelPath: fixturePaths[1]!,
        ...(fixturePaths[2] ? { tokenizerFolderPath: fixturePaths[2] } : {}),
        bundleDigest: `sha256:${"c".repeat(64)}`,
        fileCount: 6,
      };
    },
    async createEphemeralHashSecret() {
      const secret = randomBytes(32).toString("hex");
      secrets.push(secret);
      return secret;
    },
    async runDirect(requestJson) {
      directCalls += 1;
      const request = JSON.parse(requestJson) as DirectRequest;
      if (request.model.modelPath !== fixturePaths[0]) {
        return JSON.stringify({ ok: false, error: { code: "direct_model_unverified" } });
      }
      const entities = detect(request.text);
      const ordered = config.reverseDirect ? [...entities].reverse() : entities;
      const secret = request.options.hashSecret;
      if (request.operation === "deidentify") {
        const spans = ordered.map((entity) =>
          wireSpan(entity, request.text, secret, platform, "mask", `[${entity.label}]`, request.options),
        );
        return JSON.stringify({
          ok: true,
          result: { text: request.text, deidentifiedText: redact(request.text, entities), spans },
        });
      }
      return JSON.stringify({
        ok: true,
        result: { spans: ordered.map((entity) => wireSpan(entity, request.text, secret, platform, "keep", null, request.options)) },
      });
    },
    async writeReport() {
      return true;
    },
  };

  const bridge: BridgeApi = config.releaseBridge
    ? createReleaseBridge(native, config.releaseBridge)
    : {
        ...bridgeApi,
        getOpenMedKitNativeModuleSource: () => config.bridgeSource ?? "bridge-module",
      };

  return {
    options(jsMode) {
      if (!config.releaseBridge) {
        bridgeApi.setOpenMedKitNativeModuleForTests(native);
      }
      return {
        bridge,
        probe,
        manifest: CORPUS_MANIFEST,
        registry: CORPUS_REGISTRY,
        jsMode,
      };
    },
    directCalls: () => directCalls,
    fixturePaths,
    secrets,
  };
}

/** A JavaScript stand-in for the OpenMedKitRN native module's wire output. */
function createNativeBridge(
  platform: Platform,
  detect: (text: string) => Entity[],
  config: HarnessConfig,
): OpenMedKitNativeModule {
  let loadedKey: string | null = null;
  const requireLoaded = (code: string) => {
    if (!loadedKey && !config.answerBeforeLoad) {
      throw Object.assign(new Error("OpenMedKit model is not loaded"), { code });
    }
  };
  const spansFor = (text: string, options: Record<string, unknown> | undefined, action: string) =>
    detect(text).map((entity) => {
      const span = wireSpan(
        entity,
        text,
        String(options?.hashSecret ?? ""),
        platform,
        action,
        action === "keep" ? null : `[${entity.label}]`,
        options,
        config.bridgeSurfaceUnits ?? "scalar",
      );
      config.mutateBridgeSpan?.(span);
      return span;
    });
  const touchNetwork = () => {
    if (config.touchNetwork) {
      void globalThis.fetch("https://example.invalid/").catch(() => undefined);
    }
  };

  return {
    async loadModel(source) {
      const loaded = config.alwaysReportLoaded ? true : loadedKey !== source.cacheKey;
      loadedKey = source.cacheKey;
      return { cacheKey: source.cacheKey, modelPath: source.modelPath, backend: source.backend, platform, loaded };
    },
    async analyzeText(text, options) {
      requireLoaded("openmedkit_analyze_failed");
      touchNetwork();
      return spansFor(text, options as Record<string, unknown>, "keep") as never;
    },
    async extractPii(text, options) {
      requireLoaded("openmedkit_extract_failed");
      return { spans: spansFor(text, options as Record<string, unknown>, "keep") } as never;
    },
    async deidentify(text, options) {
      requireLoaded("openmedkit_deidentify_failed");
      const spans = spansFor(text, options as Record<string, unknown>, "mask");
      return (config.dropDeidentifiedText
        ? { text, spans }
        : { text, deidentifiedText: redact(text, detect(text)), spans }) as never;
    },
  };
}

/** A bridge object for release-mode runs, with an explicit double guard. */
function createReleaseBridge(native: OpenMedKitNativeModule, guard: "guarded" | "unguarded"): BridgeApi {
  return {
    loadModel: (source) => native.loadModel({ backend: "coreml", cacheKey: "", ...source } as never),
    analyzeText: async (text, options) => bridgeNormalize(await native.analyzeText(text, options as never)),
    extractPii: async (text, options) => bridgeNormalize(await native.extractPii(text, options as never)),
    deidentify: async (text, options) => {
      const response = (await native.deidentify(text, options as never)) as unknown as {
        text: string;
        deidentifiedText: string;
        spans: never[];
      };
      return { text: response.text, deidentifiedText: response.deidentifiedText, spans: response.spans };
    },
    getOpenMedKitNativeModuleSource: () => "bridge-module",
    setOpenMedKitNativeModuleForTests(module) {
      if (module !== null && guard === "guarded") {
        throw new bridgeApi.OpenMedKitBridgeError("openmedkit_test_double_forbidden", "forbidden");
      }
    },
  };
}

function bridgeNormalize(response: unknown): never {
  return (Array.isArray(response) ? response : (response as { spans: unknown[] }).spans) as never;
}

function wireSpan(
  entity: Entity,
  text: string,
  secret: string,
  platform: Platform,
  action = "keep",
  replacement: string | null = null,
  options?: { docId?: unknown; metadata?: unknown },
  units: "scalar" | "utf16" = "scalar",
): Record<string, unknown> {
  const surface = units === "scalar" ? scalarSlice(text, entity.start, entity.end) : text.slice(entity.start, entity.end);
  return {
    schema_version: 1,
    doc_id: typeof options?.docId === "string" ? options.docId : "document",
    start: entity.start,
    end: entity.end,
    text_hash: `hmac-sha256:${createHmac("sha256", Buffer.from(secret, "utf8")).update(surface, "utf8").digest("hex")}`,
    entity_type: entity.label,
    canonical_label: entity.label,
    policy_label: "DIRECT_IDENTIFIER",
    regulatory_tags: [],
    score: entity.confidence,
    detector: `openmedkit-${platform}`,
    evidence: { bridge: "react-native", runtime: "OpenMedKit" },
    action,
    replacement,
    reversible_id: null,
    section: null,
    metadata: (options?.metadata as Record<string, unknown> | undefined) ?? {},
  };
}

function redact(text: string, entities: readonly Entity[]): string {
  let rewritten = text;
  for (const entity of [...entities].sort((a, b) => b.start - a.start)) {
    rewritten = replaceScalarRange(rewritten, entity.start, entity.end, `[${entity.label}]`);
  }
  return rewritten;
}

function outcomes(report: ParityReport): Record<string, string> {
  return Object.fromEntries(report.lifecycle.map((item) => [item.check, item.outcome]));
}

function assertValueFree(report: ParityReport, harness: Harness): void {
  const encoded = JSON.stringify(report);
  assert.deepEqual(findReportLeaks(report, allCorpusTexts), []);
  for (const sentinel of [...harness.fixturePaths, ...harness.secrets]) {
    assert.equal(encoded.includes(sentinel), false, "report must not hold paths or secrets");
  }
  for (const item of offsetCases) {
    const detection = offsetDetections.get(item.text)!;
    const surface = scalarSlice(item.text, detection.start, detection.end);
    if (scalarLength(surface) >= 2) {
      assert.equal(encoded.includes(surface), false, `report leaked a surface of ${item.id}`);
    }
  }
}

function assertCorpusError(body: () => unknown, code: string): void {
  assert.throws(body, (error: unknown) => error instanceof CorpusError && error.code === code);
}

function withCorpusDigest(manifest: CorpusManifest, json: unknown): CorpusManifest {
  const copy = structuredClone(manifest);
  copy.corpora[0]!.canonical_sha256 = canonicalDigest(json);
  return copy;
}

function codeOnly(source: string): string {
  return source
    .split("\n")
    .filter((line) => !/^\s*(\/\/|\/\*|\*)/.test(line))
    .join("\n");
}

function labelSet(source: string, name: string): string[] {
  const match = new RegExp(`${name}[^=]*=\\s*(?:setOf\\(|\\[)([\\s\\S]*?)[\\])]`).exec(source);
  assert.ok(match, `${name} not found`);
  return [...match[1]!.matchAll(/"([A-Z_]+)"/g)].map((item) => item[1]!).sort();
}
