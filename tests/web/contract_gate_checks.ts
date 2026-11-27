// Web case checks for the contract-gate reporter (contract_gate_reporter.ts).
//
// One checker per fixture whose `web` cell in contract_gate.json is
// `required`. A checker prepares the fixture once and returns one check per
// case group; a check throws ContractCaseFailure with a snake_case reason
// code, never a value, when the web package disagrees with the case. The
// harnesses are shared with each fixture's own web spec.
import { createHash } from "node:crypto";
import { posix } from "node:path";
import { isDeepStrictEqual } from "node:util";

import {
  DecisionRuntimeError,
  EncoderDecisionRuntime,
  KitParityError,
  PortablePolicyError,
  acceptKitParityResult,
  createOrtDecisionEncoder,
  deidentify,
  parseDecisionRequest,
  parseKitParityResult,
  scalarToUtf16Offset,
  serializeKitParityResult,
  spansToRedactedText,
  toKitParityResult,
  utf16ToScalarOffset,
  utf8ToScalarOffset,
  verifyKitParityResult,
  type KitParityResult,
  type RawTokenClassificationPipeline,
} from "../../js/openmedkit-web/src/index";
import {
  Tensor,
  WordLevelTokenizer,
  assertClose,
  configuration,
  tinySession,
  type Fixture as DecisionFixture,
} from "./decision_encoder_support";
import { webResult, type Vectors as KitParityVectors } from "./kit_parity_support";
import {
  decide,
  runLoadCase,
  type DecisionCase,
  type ErrorCase,
  type LoadCase,
  type Vectors as PortablePolicyVectors,
} from "./portable_policy_support";
import {
  loadTokenizerClass,
  makeRunner,
  verify,
  type RuntimeSupport,
  type Vectors as TokenizerVectors,
} from "./tokenizer_conformance_support";

/** A case check failed with a controlled, value-free reason code. */
export class ContractCaseFailure extends Error {
  constructor(readonly reason: string) {
    super(reason);
    this.name = "ContractCaseFailure";
  }
}

/** Fail the current case with `reason` unless `condition` holds. */
export function expectCase(condition: unknown, reason: string): asserts condition {
  if (!condition) {
    throw new ContractCaseFailure(reason);
  }
}

/** Run `action` and return what it threw, failing the case when it returns. */
export function rejectionOf(action: () => unknown): unknown {
  try {
    action();
  } catch (error) {
    return error;
  }
  throw new ContractCaseFailure("rejection_not_raised");
}

/** Run `action`, failing the case with `reason` when it throws. */
function attempt<T>(action: () => T, reason: string): T {
  try {
    return action();
  } catch {
    throw new ContractCaseFailure(reason);
  }
}

/** Read a repository-relative file. */
export type FixtureReader = (path: string) => Promise<Uint8Array>;

export interface CheckContext {
  /** Repository-relative path of the fixture being checked. */
  fixturePath: string;
  read: FixtureReader;
}

/** Checks one case group item; throws to fail the case. */
export type CaseCheck = (item: any, index: number) => void | Promise<void>;

export interface FixturePlan {
  /** One check per case-group key. A group without a check is not exercised. */
  groups: Record<string, CaseCheck>;
  /** Fixture-level reason codes, for example a stale companion asset. */
  errors?: string[];
}

/** Prepares the case checks for one fixture document. */
export type FixtureChecker = (
  document: any,
  context: CheckContext,
) => FixturePlan | Promise<FixturePlan>;

function sha256(text: string | Uint8Array): string {
  return createHash("sha256").update(text).digest("hex");
}

function sameUtf8(left: string, right: string): boolean {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8")) === 0;
}

function closeEnough(actual: unknown, expected: unknown, tolerance: number): boolean {
  try {
    assertClose(actual, expected, tolerance, "$");
    return true;
  } catch {
    return false;
  }
}

/** Reason codes the offset_contract checker writes. */
export const OFFSET_CONTRACT_REASONS = [
  "span_count_mismatch",
  "offset_not_scalar_boundary",
  "grapheme_boundary_mismatch",
  "offset_unit_mismatch",
  "redaction_mismatch",
] as const;

interface OffsetCase {
  id: string;
  text: string;
  input_start: number;
  input_end: number;
  expected_start: number;
  expected_end: number;
  replacement: string;
  expected_redacted: string;
}

const OFFSET_HASH_SECRET = new Uint8Array(32).fill(0x5a);

/**
 * offset_contract: a model prediction over the case's input span goes through
 * `deidentify` with an injected pipeline. The prediction carries the package's
 * own input unit (UTF-16, converted with `scalarToUtf16Offset`). The returned
 * span must land on the case's grapheme-snapped scalar offsets, report them in
 * the contract's unit (Unicode scalars) and redact to the expected bytes.
 */
async function checkOffsetCase(item: OffsetCase): Promise<void> {
  const text = item.text;
  const start = scalarToUtf16Offset(text, item.input_start);
  const end = scalarToUtf16Offset(text, item.input_end);
  const pipeline: RawTokenClassificationPipeline = () => [
    { entity: "B-NAME", score: 0.99, index: 1, word: text.slice(start, end), start, end },
  ];
  const result = await deidentify(text, {
    pipeline,
    hashSecret: OFFSET_HASH_SECRET,
    docId: "contract-gate",
    replacement: () => item.replacement,
  });
  expectCase(result.spans.length === 1, "span_count_mismatch");
  const span = result.spans[0]!;
  const scalarStart = attempt(() => utf16ToScalarOffset(text, span.start), "offset_not_scalar_boundary");
  const scalarEnd = attempt(() => utf16ToScalarOffset(text, span.end), "offset_not_scalar_boundary");
  expectCase(
    scalarStart === item.expected_start && scalarEnd === item.expected_end,
    "grapheme_boundary_mismatch",
  );
  expectCase(
    span.start === item.expected_start && span.end === item.expected_end,
    "offset_unit_mismatch",
  );
  expectCase(sameUtf8(result.deidentifiedText, item.expected_redacted), "redaction_mismatch");
}

const offsetContract: FixtureChecker = () => ({ groups: { cases: checkOffsetCase } });

function expectKitRejection(error: unknown, reason: string, fieldPath?: string): void {
  expectCase(error instanceof KitParityError, "rejection_type_mismatch");
  expectCase(error.code === reason, "reason_mismatch");
  if (fieldPath !== undefined) {
    expectCase(error.fieldPath === fieldPath, "field_path_mismatch");
  }
}

const kitParityVectors: FixtureChecker = (document: KitParityVectors) => {
  const sentinel = document.profile.privacy_sentinel;
  const sample =
    document.vectors.find((vector) => vector.id === "emoji_sequences") ?? document.vectors[0];
  const converters = {
    scalar: scalarToUtf16Offset,
    utf8: utf8ToScalarOffset,
    utf16: utf16ToScalarOffset,
  };
  return {
    groups: {
      vectors: (vector: KitParityVectors["vectors"][number]) => {
        const result = webResult(vector);
        const actual = attempt(
          () => serializeKitParityResult(toKitParityResult(result)),
          "projection_rejected",
        );
        expectCase(actual === vector.result_json, "canonical_bytes_mismatch");
        expectCase(sha256(actual) === vector.result_sha256, "result_digest_mismatch");
        expectCase(
          sha256(vector.deidentified_text) === vector.deidentified_sha256,
          "deidentified_digest_mismatch",
        );
        expectCase(
          sameUtf8(spansToRedactedText(vector.text, result.spans), vector.deidentified_text),
          "redaction_mismatch",
        );
        const parsed = attempt(() => parseKitParityResult(vector.result_json), "parse_rejected");
        const fromBytes = attempt(
          () => parseKitParityResult(new TextEncoder().encode(vector.result_json)),
          "parse_rejected",
        );
        expectCase(
          serializeKitParityResult(parsed) === vector.result_json &&
            isDeepStrictEqual(parsed, fromBytes),
          "round_trip_mismatch",
        );
        attempt(() => verifyKitParityResult(parsed, vector.text), "verification_failed");
      },
      rejections: (record: KitParityVectors["rejections"][number]) => {
        const error = rejectionOf(() => parseKitParityResult(record.input));
        expectKitRejection(error, record.reason, record.field_path);
        expectCase(!(error as Error).message.includes(sentinel), "message_leaks_value");
      },
      source_rejections: (record: KitParityVectors["source_rejections"][number]) => {
        const parsed = attempt(() => parseKitParityResult(record.input), "parse_rejected");
        const error = rejectionOf(() => verifyKitParityResult(parsed, record.text));
        expectKitRejection(error, record.reason, record.field_path);
      },
      offset_rejections: (record: KitParityVectors["offset_rejections"][number]) => {
        const convert = Object.hasOwn(converters, record.unit) ? converters[record.unit] : undefined;
        expectCase(convert !== undefined, "offset_unit_unknown");
        expectKitRejection(rejectionOf(() => convert(record.text, record.offset)), record.reason);
      },
      compatibility: (record: KitParityVectors["compatibility"][number]) => {
        expectCase(sample !== undefined, "case_not_exercised");
        const bundle = {
          offsetContractVersion: record.offset_contract_version,
          decoderContractVersion: record.decoder_contract_version,
        };
        if (record.outcome === "accepted") {
          const accepted: KitParityResult = attempt(
            () => acceptKitParityResult(sample.result_json, bundle),
            "compatibility_mismatch",
          );
          expectCase(
            isDeepStrictEqual(accepted, parseKitParityResult(sample.result_json)),
            "compatibility_mismatch",
          );
          return;
        }
        const candidates = [sample.result_json, "not json", document.rejections[0]?.input ?? "{}"];
        for (const candidate of candidates) {
          expectKitRejection(
            rejectionOf(() => acceptKitParityResult(candidate, bundle)),
            record.outcome,
          );
        }
      },
    },
  };
};

function expectPolicyRejection(error: unknown, code: string, fieldPath: string): void {
  expectCase(error instanceof PortablePolicyError, "rejection_type_mismatch");
  expectCase(error.code === code, "reason_mismatch");
  expectCase(error.fieldPath === fieldPath, "field_path_mismatch");
}

const portablePolicyVectors: FixtureChecker = (document: PortablePolicyVectors) => ({
  groups: {
    decision_cases: (item: DecisionCase) => {
      const outcome = attempt(() => decide(document, item), "decision_rejected");
      const actual = serializeKitParityResult(outcome.result);
      expectCase(actual === item.expected.result_json, "canonical_bytes_mismatch");
      expectCase(sha256(actual) === item.expected.result_sha256, "result_digest_mismatch");
      expectCase(
        sameUtf8(outcome.result.deidentified_text, item.expected.deidentified_text),
        "redaction_mismatch",
      );
      expectCase(isDeepStrictEqual(outcome.decisions, item.expected.decisions), "decision_mismatch");
      expectCase(outcome.policyDigest === item.expected.policy_digest, "policy_digest_mismatch");
    },
    error_cases: (item: ErrorCase) => {
      const error = rejectionOf(() => decide(document, item));
      expectPolicyRejection(error, item.expected.code, item.expected.field_path);
      expectCase(
        item.text === "" || !(error as Error).message.includes(item.text),
        "message_leaks_value",
      );
    },
    load_cases: (item: LoadCase) => {
      if (item.expected.code !== undefined) {
        expectPolicyRejection(
          rejectionOf(() => runLoadCase(item)),
          item.expected.code,
          item.expected.field_path ?? "$",
        );
        return;
      }
      const policy = attempt(() => runLoadCase(item), "unexpected_rejection");
      expectCase(policy.name === item.expected.name, "policy_name_mismatch");
      expectCase(policy.digest === item.expected.digest, "policy_digest_mismatch");
      expectCase(
        isDeepStrictEqual([...policy.requiredCapabilities], item.expected.required_capabilities),
        "capabilities_mismatch",
      );
    },
  },
});

const decisionEncoder: FixtureChecker = (document: DecisionFixture) => {
  const tokenizer = new WordLevelTokenizer(document.tokenizer);
  return {
    groups: {
      cases: async (row: DecisionFixture["cases"][number]) => {
        const settings = Object.hasOwn(document.runtimes, row.runtime)
          ? document.runtimes[row.runtime]
          : undefined;
        expectCase(settings !== undefined, "runtime_settings_missing");
        const session = tinySession(document.encoder, settings.fault);
        const runtime = new EncoderDecisionRuntime({
          encoder: createOrtDecisionEncoder({ session, Tensor }),
          tokenizer,
          identity: document.backend,
          configuration: configuration(settings),
          ...(row.policy === null ? {} : { policy: row.policy }),
          ...(row.calibration_profiles === null
            ? {}
            : { calibrationProfiles: row.calibration_profiles }),
        });
        const result = await runtime.decide(JSON.stringify(row.request));
        expectCase(
          closeEnough(result, row.expected.result, document.tolerance.scores),
          "result_mismatch",
        );
        expectCase(
          result.autonomous_action === false && result.review.required === true,
          "review_contract_mismatch",
        );
        expectCase(
          session.calls.length === row.expected.encoder_calls.length,
          "encoder_call_count_mismatch",
        );
        row.expected.encoder_calls.forEach((call, index) => {
          const recorded = session.calls[index]!;
          expectCase(
            isDeepStrictEqual(recorded.ids, call.input_ids) && recorded.type === "int64",
            "encoder_input_mismatch",
          );
          if (call.logits !== null && recorded.logits !== null) {
            expectCase(
              closeEnough(recorded.logits, call.logits, document.tolerance.logits),
              "encoder_logits_mismatch",
            );
          }
        });
      },
      invalid_requests: (row: DecisionFixture["invalid_requests"][number]) => {
        const error = rejectionOf(() => parseDecisionRequest(JSON.stringify(row.request)));
        expectCase(error instanceof DecisionRuntimeError, "rejection_type_mismatch");
        expectCase(error.kind === row.error, "reason_mismatch");
      },
    },
  };
};

const tokenizerConformance: FixtureChecker = async (
  document: TokenizerVectors,
  context: CheckContext,
) => {
  const directory = posix.dirname(context.fixturePath);
  const text = async (path: string) =>
    new TextDecoder("utf-8", { fatal: true }).decode(await context.read(path));
  const support = JSON.parse(
    await text(posix.join(directory, "tokenizer_runtime_support_v1.json")),
  ) as RuntimeSupport;
  const assets = posix.join(directory, document.tokenizer.directory);
  const tokenizerBytes = await context.read(posix.join(assets, "tokenizer.json"));
  const tokenizerConfig = JSON.parse(
    await text(posix.join(assets, "tokenizer_config.json")),
  ) as Record<string, unknown>;
  const errors: string[] = [];
  if (sha256(tokenizerBytes) !== document.tokenizer.sha256) {
    errors.push("tokenizer_asset_digest_mismatch");
  }
  const Tokenizer = await loadTokenizerClass();
  const tokenizer = new Tokenizer(
    JSON.parse(new TextDecoder().decode(tokenizerBytes)),
    tokenizerConfig,
  );
  const report = verify(document, support, makeRunner(tokenizer));
  const byCase = new Map<string, string>();
  let everyCase: string | undefined;
  for (const mismatch of report.mismatches) {
    const code = mismatch.code.toLowerCase();
    if (mismatch.caseId !== "*") {
      if (!byCase.has(mismatch.caseId)) {
        byCase.set(mismatch.caseId, code);
      }
    } else if (mismatch.code === "TOKENIZER_DECLARED_DIVERGENCE_STALE") {
      errors.push(code);
    } else {
      everyCase ??= code;
    }
  }
  return {
    groups: {
      cases: (item: TokenizerVectors["cases"][number]) => {
        const code = everyCase ?? byCase.get(item.id);
        if (code !== undefined) {
          throw new ContractCaseFailure(code);
        }
      },
    },
    errors,
  };
};

/** The web package's checker for each web-required contract-gate fixture. */
export const WEB_CONTRACT_CHECKERS: Readonly<Record<string, FixtureChecker>> = Object.freeze({
  decision_encoder_v1: decisionEncoder,
  kit_parity_vectors: kitParityVectors,
  offset_contract: offsetContract,
  portable_policy_vectors: portablePolicyVectors,
  tokenizer_conformance_v1: tokenizerConformance,
});
