// In-app parity run: the React Native bridge against direct native OpenMedKit.
//
// For every shared corpus case and operation the runner calls the public
// `@openmed/openmedkit-react-native` API (which crosses the real React Native
// boundary into `OpenMedKitRN`) and the example-only probe (which calls
// OpenMedKit directly), then diffs the two results field by field. The report
// it returns holds platform, runtime, bundle digest, counts, lifecycle
// outcomes, and value-free mismatch paths; it never holds source text.

import type {
  OpenMedBridgeOptions,
  OpenMedDeidentifyOptions,
  OpenMedDeidentifyResult,
  OpenMedKitNativeModule,
  OpenMedKitNativeModuleSource,
  OpenMedModelLoadResult,
  OpenMedModelSource,
  OpenMedSpan,
} from "../../../src/index";
import { canonicalDigest, replaceScalarRange, scalarLength } from "./canonical";
import {
  CorpusError,
  loadCorpora,
  parseCorpusManifest,
  type CorpusManifest,
  type LoadedCorpus,
  type ParityCase,
  type ParityOperation,
} from "./corpus";
import {
  diffSpanLists,
  diffValues,
  fingerprint,
  type DiffPolicy,
  type FieldMismatch,
  type ValueFingerprint,
} from "./diff";
import {
  findReportLeaks,
  findSurfaceLeaks,
  installEgressGuard,
  spanSurfaces,
  type EgressGuard,
} from "./privacy";
import {
  parseDirectResponse,
  parseModelFixture,
  parseProbeEnvironment,
  safeCode,
  type DirectModel,
  type DirectOptions,
  type DirectProbe,
  type DirectResponse,
  type ProbeEnvironment,
  type ProbeModelFixture,
} from "./probe";

export const PARITY_REPORT_SCHEMA = "openmed.rn.parity.report" as const;
export const PARITY_REPORT_FILE_NAME = "openmed-rn-parity-report.json";

/** The public bridge surface the harness drives. */
export interface BridgeApi {
  loadModel(source: OpenMedModelSource): Promise<OpenMedModelLoadResult>;
  analyzeText(text: string, options?: OpenMedBridgeOptions): Promise<OpenMedSpan[]>;
  extractPii(text: string, options?: OpenMedBridgeOptions): Promise<OpenMedSpan[]>;
  deidentify(
    text: string,
    options?: OpenMedDeidentifyOptions,
  ): Promise<OpenMedDeidentifyResult>;
  getOpenMedKitNativeModuleSource(): OpenMedKitNativeModuleSource;
  setOpenMedKitNativeModuleForTests(module: OpenMedKitNativeModule | null): void;
}

export type JsMode = "release" | "development";

export interface ParityRunOptions {
  bridge: BridgeApi;
  probe: DirectProbe;
  manifest: unknown;
  /** Parsed corpus JSON keyed by manifest corpus id. */
  registry: Readonly<Record<string, unknown>>;
  jsMode: JsMode;
  /** Override the egress guard; defaults to guarding `globalThis`. */
  egressGuard?: EgressGuard;
  maxReportedMismatches?: number;
}

export type LifecycleOutcome = "pass" | "fail" | "not_applicable" | "not_covered";

export interface LifecycleCheck {
  check: string;
  outcome: LifecycleOutcome;
  reason?: string;
  mismatched_paths?: string[];
}

export type RecordedOperation = ParityOperation;
export type RunPhase = "initial" | "replacement";

export interface CaseMismatch extends FieldMismatch {
  corpus_id: string;
  case_id: string;
  operation: RecordedOperation;
  phase: RunPhase;
}

export type ContractViolationKind =
  | "offset_bounds"
  | "text_hash_format"
  | "redaction_reconstruction"
  | "surface_leak";

export interface ContractViolation {
  corpus_id: string;
  case_id: string;
  operation: RecordedOperation;
  phase: RunPhase;
  path: string;
  kind: ContractViolationKind;
  bridge: ValueFingerprint;
}

export interface OperationStats {
  compared: number;
  matched: number;
  mismatched: number;
  /** Both sides failed with a typed error (counted as matched). */
  error_parity: number;
}

export interface ParityReport {
  schema: typeof PARITY_REPORT_SCHEMA;
  version: 1;
  status: "pass" | "fail";
  failure_reasons: string[];
  evidence_eligible: boolean;
  platform: ProbeEnvironment["platform"] | "unknown";
  runtime_backend: ProbeEnvironment["runtime_backend"] | "unknown";
  device_class: ProbeEnvironment["device_class"] | "unknown";
  os_version: string;
  build_configuration: ProbeEnvironment["build_configuration"] | "unknown";
  js_mode: JsMode;
  bridge_source: OpenMedKitNativeModuleSource;
  bundle: { digest: string | null; file_count: number };
  corpora: Array<{ corpus_id: string; digest: string; cases: number }>;
  operations: Partial<Record<RecordedOperation, OperationStats>>;
  cases: { total: number; matched: number; mismatched: number };
  lifecycle: LifecycleCheck[];
  egress: { blocked_attempts: number };
  sentinel: {
    bridge_results_scanned: number;
    bridge_surface_leaks: number;
    report_leaks: number;
  };
  mismatches: CaseMismatch[];
  contract_violations: ContractViolation[];
  diagnostics_truncated: boolean;
  diagnostics_redacted: boolean;
  report_digest?: string;
}

/** Lifecycle guarantees that need the runtime APIs from #3646 and #3647. */
export const PENDING_LIFECYCLE_CHECKS = [
  "cancellation",
  "bridge_invalidation",
  "background_expiration",
  "memory_pressure",
  "generation_isolation",
] as const;

const LIFECYCLE_PROBE_TEXT = "OpenMedKit parity lifecycle probe";
const PARITY_METADATA = { parity: "rn-bridge" } as const;
const HMAC_TEXT_HASH = /^hmac-sha256:[0-9a-f]{64}$/;
const NATIVE_SOURCES: ReadonlySet<OpenMedKitNativeModuleSource> = new Set([
  "turbo-module",
  "bridge-module",
]);

/** Run every parity case and return a value-free report. */
export async function runParity(options: ParityRunOptions): Promise<ParityReport> {
  const egress = options.egressGuard ?? installEgressGuard();
  const state = new RunState(options);
  try {
    await state.run();
  } catch (error) {
    state.fail(
      error instanceof CorpusError ? error.code : safeCode(codeOf(error), "harness_error"),
    );
  } finally {
    state.report.egress.blocked_attempts = egress.attempts();
    egress.restore();
  }
  return state.finalize();
}

class RunState {
  readonly report: ParityReport;
  private readonly options: ParityRunOptions;
  private readonly maxMismatches: number;
  private readonly sentinels = new Set<string>();
  private readonly mismatchedCases = new Set<string>();
  private readonly comparedCases = new Set<string>();
  private manifest: CorpusManifest | null = null;
  private diffPolicy: DiffPolicy = { scoreAbsTolerance: 0 };
  private hashSecret = "";

  constructor(options: ParityRunOptions) {
    this.options = options;
    this.maxMismatches = options.maxReportedMismatches ?? 200;
    this.report = {
      schema: PARITY_REPORT_SCHEMA,
      version: 1,
      status: "pass",
      failure_reasons: [],
      evidence_eligible: false,
      platform: "unknown",
      runtime_backend: "unknown",
      device_class: "unknown",
      os_version: "unknown",
      build_configuration: "unknown",
      js_mode: options.jsMode,
      bridge_source: "unavailable",
      bundle: { digest: null, file_count: 0 },
      corpora: [],
      operations: {},
      cases: { total: 0, matched: 0, mismatched: 0 },
      lifecycle: [],
      egress: { blocked_attempts: 0 },
      sentinel: { bridge_results_scanned: 0, bridge_surface_leaks: 0, report_leaks: 0 },
      mismatches: [],
      contract_violations: [],
      diagnostics_truncated: false,
      diagnostics_redacted: false,
    };
  }

  fail(reason: string): void {
    this.report.status = "fail";
    if (!this.report.failure_reasons.includes(reason)) {
      this.report.failure_reasons.push(reason);
    }
  }

  async run(): Promise<void> {
    const { bridge, probe } = this.options;

    const manifest = parseCorpusManifest(this.options.manifest);
    this.manifest = manifest;
    this.diffPolicy = { scoreAbsTolerance: manifest.tolerances.score_abs };
    const corpora = loadCorpora(manifest, this.options.registry);
    this.report.corpora = corpora.map((corpus) => ({
      corpus_id: corpus.id,
      digest: corpus.digest,
      cases: corpus.cases.length,
    }));
    for (const corpus of corpora) {
      for (const parityCase of corpus.cases) {
        this.sentinels.add(parityCase.text);
      }
    }

    const environment = parseProbeEnvironment(await probe.describeEnvironment());
    this.report.platform = environment.platform;
    this.report.runtime_backend = environment.runtime_backend;
    this.report.device_class = environment.device_class;
    this.report.os_version = environment.os_version;
    this.report.build_configuration = environment.build_configuration;

    this.report.bridge_source = bridge.getOpenMedKitNativeModuleSource();
    if (!NATIVE_SOURCES.has(this.report.bridge_source)) {
      this.fail(`bridge_not_native_${this.report.bridge_source.replace("-", "_")}`);
      return;
    }

    this.checkReleaseGuard();

    let fixture: ProbeModelFixture;
    try {
      fixture = parseModelFixture(await probe.prepareModelFixture());
    } catch (error) {
      this.fail(safeCode(codeOf(error), "model_fixture_unverified"));
      return;
    }
    this.report.bundle = { digest: fixture.bundleDigest, file_count: fixture.fileCount };
    // File paths are needed to load the model but must never reach evidence.
    for (const path of [
      fixture.modelPath,
      fixture.id2LabelPath,
      fixture.tokenizerFolderPath,
    ]) {
      if (path !== undefined) {
        this.sentinels.add(path);
      }
    }
    if (fixture.backend !== environment.runtime_backend) {
      this.fail("model_fixture_backend_mismatch");
      return;
    }

    const secret = await probe.createEphemeralHashSecret();
    if (typeof secret !== "string" || !/^[0-9a-f]{64}$/.test(secret)) {
      this.fail("ephemeral_hash_secret_invalid");
      return;
    }
    this.hashSecret = secret;
    this.sentinels.add(secret);

    await this.checkNotLoadedRejects();

    const generationKey = fixture.bundleDigest.slice("sha256:".length, 23);
    const initialModel = directModel(fixture, `parity:${generationKey}:initial`);
    await this.checkLoad("model_load_initial", initialModel, environment, true);
    await this.checkLoad("model_load_reuse", initialModel, environment, false);

    for (const corpus of corpora) {
      for (const parityCase of corpus.cases) {
        await this.compareCase(parityCase, manifest, initialModel, "initial");
      }
    }

    const replacementModel = directModel(fixture, `parity:${generationKey}:replacement`);
    await this.checkLoad("model_replacement", replacementModel, environment, true);
    for (const corpus of corpora) {
      const first = corpus.cases[0];
      if (first) {
        await this.compareCase(first, manifest, replacementModel, "replacement");
      }
    }
    await this.checkLoad(
      "model_replacement_reuse",
      replacementModel,
      environment,
      false,
    );

    for (const check of PENDING_LIFECYCLE_CHECKS) {
      this.report.lifecycle.push({
        check,
        outcome: "not_covered",
        reason: "requires_lifecycle_apis_3646_3647",
      });
    }

    this.checkCoverage(corpora, manifest);
  }

  finalize(): ParityReport {
    const report = this.report;
    if (report.egress.blocked_attempts > 0) {
      this.fail("network_egress_attempted");
    }
    if (report.sentinel.bridge_surface_leaks > 0) {
      this.fail("bridge_surface_leak");
    }
    if (report.mismatches.length > 0 || report.cases.mismatched > 0) {
      this.fail("parity_mismatch");
    }
    if (report.contract_violations.length > 0) {
      this.fail("contract_violation");
    }
    if (report.lifecycle.some((check) => check.outcome === "fail")) {
      this.fail("lifecycle_check_failed");
    }

    const leaks = findReportLeaks(report, [...this.sentinels]);
    if (leaks.length > 0) {
      report.sentinel.report_leaks = leaks.length;
      report.mismatches = [];
      report.contract_violations = [];
      report.diagnostics_redacted = true;
      this.fail("report_sentinel_leak");
    }

    const releaseGuard = report.lifecycle.find((check) => check.check === "release_guard");
    report.evidence_eligible =
      report.status === "pass" &&
      report.js_mode === "release" &&
      report.build_configuration === "release" &&
      NATIVE_SOURCES.has(report.bridge_source) &&
      releaseGuard?.outcome === "pass";

    delete report.report_digest;
    report.report_digest = canonicalDigest(report);
    return report;
  }

  private checkReleaseGuard(): void {
    const { bridge, jsMode } = this.options;
    if (jsMode !== "release") {
      this.report.lifecycle.push({
        check: "release_guard",
        outcome: "not_applicable",
        reason: "development_bundle",
      });
      return;
    }
    let outcome: LifecycleOutcome = "fail";
    try {
      bridge.setOpenMedKitNativeModuleForTests(REJECTED_DOUBLE);
    } catch (error) {
      outcome = codeOf(error) === "openmedkit_test_double_forbidden" ? "pass" : "fail";
    } finally {
      bridge.setOpenMedKitNativeModuleForTests(null);
    }
    this.report.lifecycle.push({ check: "release_guard", outcome });
    if (!NATIVE_SOURCES.has(bridge.getOpenMedKitNativeModuleSource())) {
      this.fail("bridge_not_native_after_guard");
    }
  }

  private async checkNotLoadedRejects(): Promise<void> {
    let outcome: LifecycleOutcome = "fail";
    try {
      await this.options.bridge.analyzeText(LIFECYCLE_PROBE_TEXT, {
        hashSecret: this.hashSecret,
      });
    } catch {
      outcome = "pass";
    }
    this.report.lifecycle.push({ check: "analyze_before_load_rejects", outcome });
  }

  private async checkLoad(
    check: string,
    model: DirectModel,
    environment: ProbeEnvironment,
    expectLoaded: boolean,
  ): Promise<void> {
    let result: OpenMedModelLoadResult;
    try {
      result = await this.options.bridge.loadModel(bridgeModelSource(model));
    } catch (error) {
      this.report.lifecycle.push({
        check,
        outcome: "fail",
        reason: safeCode(codeOf(error), "bridge_load_rejected"),
      });
      return;
    }
    const mismatches = diffValues(
      result,
      {
        cacheKey: model.cacheKey,
        modelPath: model.modelPath,
        backend: model.backend,
        platform: environment.platform,
        loaded: expectLoaded,
      },
      this.diffPolicy,
    );
    const entry: LifecycleCheck = {
      check,
      outcome: mismatches.length === 0 ? "pass" : "fail",
    };
    if (mismatches.length > 0) {
      entry.mismatched_paths = mismatches.map((mismatch) => mismatch.path);
    }
    this.report.lifecycle.push(entry);
  }

  private async compareCase(
    parityCase: ParityCase,
    manifest: CorpusManifest,
    model: DirectModel,
    phase: RunPhase,
  ): Promise<void> {
    const caseKey = `${phase}:${parityCase.corpusId}:${parityCase.caseId}`;
    this.comparedCases.add(caseKey);
    const options: DirectOptions = {
      docId: `rn-parity:${parityCase.corpusId}:${parityCase.caseId}`,
      hashSecret: this.hashSecret,
      metadata: { ...PARITY_METADATA },
      confidenceThreshold: manifest.defaults.confidence_threshold,
      useSmartMerging: manifest.defaults.use_smart_merging,
    };

    for (const operation of manifest.defaults.operations) {
      const operationOptions =
        operation === "deidentify" ? { ...options, policy: parityCase.policy } : options;
      const [bridgeOutcome, directOutcome] = [
        await this.callBridge(operation, parityCase.text, operationOptions),
        await this.callDirect(operation, parityCase.text, operationOptions, model),
      ];
      const stats = this.stats(operation);
      stats.compared += 1;
      const mismatches = this.compareOutcomes(
        operation,
        parityCase.text,
        bridgeOutcome,
        directOutcome,
      );
      if (bridgeOutcome.ok) {
        this.checkContract(parityCase, operation, phase, bridgeOutcome.value);
      }
      if (mismatches.length === 0) {
        stats.matched += 1;
        if (!bridgeOutcome.ok && !directOutcome.ok) {
          stats.error_parity += 1;
        }
        continue;
      }
      stats.mismatched += 1;
      this.mismatchedCases.add(caseKey);
      for (const mismatch of mismatches) {
        this.recordMismatch({
          ...mismatch,
          corpus_id: parityCase.corpusId,
          case_id: parityCase.caseId,
          operation,
          phase,
        });
      }
    }

    this.report.cases.total = this.comparedCases.size;
    this.report.cases.mismatched = this.mismatchedCases.size;
    this.report.cases.matched = this.comparedCases.size - this.mismatchedCases.size;
  }

  private async callBridge(
    operation: ParityOperation,
    text: string,
    options: DirectOptions,
  ): Promise<BridgeOutcome> {
    const { bridge } = this.options;
    try {
      switch (operation) {
        case "analyzeText":
          return { ok: true, value: { spans: await bridge.analyzeText(text, options) } };
        case "extractPii":
          return { ok: true, value: { spans: await bridge.extractPii(text, options) } };
        case "deidentify": {
          const result = await bridge.deidentify(text, options);
          return {
            ok: true,
            value: {
              spans: result.spans,
              text: result.text,
              deidentifiedText: result.deidentifiedText,
            },
          };
        }
      }
    } catch (error) {
      return { ok: false, code: safeCode(codeOf(error), "bridge_rejected") };
    }
  }

  private async callDirect(
    operation: ParityOperation,
    text: string,
    options: DirectOptions,
    model: DirectModel,
  ): Promise<DirectResponse> {
    try {
      const response = await this.options.probe.runDirect(
        JSON.stringify({ operation, text, options, model }),
      );
      return typeof response === "string"
        ? parseDirectResponse(response)
        : { ok: false, error: { code: "direct_response_invalid" } };
    } catch (error) {
      return { ok: false, error: { code: safeCode(codeOf(error), "direct_rejected") } };
    }
  }

  private compareOutcomes(
    operation: ParityOperation,
    text: string,
    bridge: BridgeOutcome,
    direct: DirectResponse,
  ): FieldMismatch[] {
    if (!bridge.ok || !direct.ok) {
      if (!bridge.ok && !direct.ok) {
        return [];
      }
      return [
        {
          path: "$",
          kind: "outcome",
          bridge: fingerprint({ outcome: bridge.ok ? "ok" : "error" }),
          direct: fingerprint({ outcome: direct.ok ? "ok" : "error" }),
        },
      ];
    }

    this.report.sentinel.bridge_results_scanned += 1;
    for (const surface of spanSurfaces(bridge.value.spans, text)) {
      this.sentinels.add(surface);
    }
    for (const surface of spanSurfaces(direct.result.spans, text)) {
      this.sentinels.add(surface);
    }

    const mismatches = diffSpanLists(
      bridge.value.spans,
      direct.result.spans,
      this.diffPolicy,
      "$.spans",
    );
    if (operation === "deidentify") {
      mismatches.push(
        ...diffValues(
          { text: bridge.value.text, deidentifiedText: bridge.value.deidentifiedText },
          { text: direct.result.text, deidentifiedText: direct.result.deidentifiedText },
          this.diffPolicy,
        ),
      );
    }
    return mismatches;
  }

  private checkContract(
    parityCase: ParityCase,
    operation: ParityOperation,
    phase: RunPhase,
    value: BridgeValue,
  ): void {
    const violations: Array<Pick<ContractViolation, "path" | "kind" | "bridge">> = [];
    const textLength = scalarLength(parityCase.text);
    value.spans.forEach((span, index) => {
      const path = `$.spans[${index}]`;
      if (
        !Number.isInteger(span.start) ||
        !Number.isInteger(span.end) ||
        span.start < 0 ||
        span.end <= span.start ||
        span.end > textLength
      ) {
        violations.push({
          path,
          kind: "offset_bounds",
          bridge: fingerprint([span.start, span.end]),
        });
      }
      if (!HMAC_TEXT_HASH.test(span.text_hash)) {
        violations.push({
          path: `${path}.text_hash`,
          kind: "text_hash_format",
          bridge: fingerprint(span.text_hash),
        });
      }
    });

    const leaks = findSurfaceLeaks(value.spans, parityCase.text);
    this.report.sentinel.bridge_surface_leaks += leaks.length;
    for (const leakPath of leaks) {
      violations.push({ path: leakPath, kind: "surface_leak", bridge: fingerprint(null) });
    }

    if (operation === "deidentify" && value.deidentifiedText !== undefined) {
      const expected = reconstructRedaction(parityCase.text, value.spans);
      if (expected !== null && expected !== value.deidentifiedText) {
        violations.push({
          path: "$.deidentifiedText",
          kind: "redaction_reconstruction",
          bridge: fingerprint(value.deidentifiedText),
        });
      }
    }

    for (const violation of violations) {
      if (this.report.contract_violations.length >= this.maxMismatches) {
        this.report.diagnostics_truncated = true;
        break;
      }
      this.report.contract_violations.push({
        corpus_id: parityCase.corpusId,
        case_id: parityCase.caseId,
        operation,
        phase,
        ...violation,
      });
    }
  }

  private checkCoverage(corpora: LoadedCorpus[], manifest: CorpusManifest): void {
    for (const operation of manifest.defaults.operations) {
      const stats = this.stats(operation);
      if (stats.compared - stats.error_parity <= 0) {
        this.fail(`no_successful_comparison_${operation.toLowerCase()}`);
      }
    }
    if (corpora.length !== manifest.corpora.length) {
      this.fail("corpus_coverage_incomplete");
    }
  }

  private recordMismatch(mismatch: CaseMismatch): void {
    if (this.report.mismatches.length >= this.maxMismatches) {
      this.report.diagnostics_truncated = true;
      return;
    }
    this.report.mismatches.push(mismatch);
  }

  private stats(operation: ParityOperation): OperationStats {
    let stats = this.report.operations[operation];
    if (!stats) {
      stats = { compared: 0, matched: 0, mismatched: 0, error_parity: 0 };
      this.report.operations[operation] = stats;
    }
    return stats;
  }
}

interface BridgeValue {
  spans: OpenMedSpan[];
  text?: string;
  deidentifiedText?: string;
}

type BridgeOutcome = { ok: true; value: BridgeValue } | { ok: false; code: string };

/**
 * Re-apply span replacements at Unicode scalar offsets, last span first.
 *
 * Returns null when spans overlap, because native de-identification never
 * emits overlapping rewritten spans and the expected text is then undefined.
 */
export function reconstructRedaction(
  text: string,
  spans: readonly Pick<OpenMedSpan, "start" | "end" | "replacement">[],
): string | null {
  const ordered = [...spans].sort((left, right) => left.start - right.start);
  for (let index = 1; index < ordered.length; index += 1) {
    if (ordered[index]!.start < ordered[index - 1]!.end) {
      return null;
    }
  }
  let rewritten = text;
  for (const span of ordered.reverse()) {
    if (span.replacement !== null) {
      rewritten = replaceScalarRange(rewritten, span.start, span.end, span.replacement);
    }
  }
  return rewritten;
}

function directModel(fixture: ProbeModelFixture, cacheKey: string): DirectModel {
  const model: DirectModel = {
    cacheKey,
    modelPath: fixture.modelPath,
    backend: fixture.backend,
  };
  if (fixture.id2LabelPath !== undefined) {
    model.id2LabelPath = fixture.id2LabelPath;
  }
  if (fixture.tokenizerFolderPath !== undefined) {
    model.tokenizerFolderPath = fixture.tokenizerFolderPath;
  }
  if (fixture.tokenizerName !== undefined) {
    model.tokenizerName = fixture.tokenizerName;
  }
  return model;
}

function bridgeModelSource(model: DirectModel): OpenMedModelSource {
  return { ...model };
}

function codeOf(error: unknown): unknown {
  if (error !== null && typeof error === "object" && "code" in error) {
    return (error as { code: unknown }).code;
  }
  if (error instanceof Error && /^[a-z][a-z0-9_]{0,63}$/.test(error.message)) {
    return error.message;
  }
  return undefined;
}

const rejectNative = async (): Promise<never> => {
  throw new Error("parity_release_guard_double");
};

/** A double that must never be accepted in a release bundle. */
const REJECTED_DOUBLE: OpenMedKitNativeModule = {
  loadModel: rejectNative,
  analyzeText: rejectNative,
  extractPii: rejectNative,
  deidentify: rejectNative,
};
