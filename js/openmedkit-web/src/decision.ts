/**
 * On-device fixed-option decisions over a GLiClass-style encoder.
 *
 * The runtime validates an OpenMed decision request (schema 1.0.0, 1.1.0 or
 * 1.2.0), applies the access policy and the supplied calibration profile,
 * scores the options through an injected tokenizer and encoder (ONNX Runtime
 * Web, Transformers.js or any other local runtime), and returns a result with
 * the same fields, states, abstention codes and review flags as the Python
 * `openmed.structured.decision.decide`. It is text-only: media requests return
 * `unsupported`. It makes no network call and never logs or returns the input,
 * prompt or examples. Results are assistive review artifacts and never
 * authorize an autonomous clinical action.
 */
import type {
  OrtInferenceSession,
  OrtTensorConstructor,
  OrtWebLoaderOptions,
} from "./runtime/ort-web-loader";
import { loadOrtWebSession } from "./runtime/ort-web-loader";
import type { OrtWebBackendChoice } from "./runtime/capability";

export const DECISION_SCHEMA_VERSION = "1.2.0";
export const SUPPORTED_DECISION_SCHEMA_VERSIONS = ["1.0.0", "1.1.0", "1.2.0"] as const;
export const DECISION_COMPATIBILITY_POLICY = "same_major";
export const DECISION_POLICY_VERSION = "1.0.0";
export const DEFAULT_DECISION_CALIBRATION_ID = "openmed.synthetic.fixed_option.v1";
export const DEFAULT_DECISION_QUESTION_ID = "decision";
export const DECISION_ADVISORY =
  "Decision scores are assistive review evidence, not a diagnosis, treatment " +
  "recommendation, enrollment decision, or autonomous clinical action.";
export const GLICLASS_DEFAULT_SCORE_LEVELS = [
  "very low",
  "low",
  "moderate",
  "high",
  "very high",
] as const;
export const GLICLASS_DEFAULT_MAX_LENGTH = 2048;

const BASE_VERSION = "1.0.0";
const QUESTION_VERSION = "1.1.0";
const MEDIA_VERSION = "1.2.0";
const MAX_INPUT_CHARS = 8192;
const MAX_CONFIGURABLE_INPUT_CHARS = 131_072;
const INPUT_BYTES_PER_CHAR = 4;
const MAX_OPTIONS = 64;
const MAX_OPTION_CHARS = 256;
const MAX_OPTION_BYTES = 1024;
const MAX_TOTAL_OPTION_BYTES = 16_384;
const MAX_QUESTIONS = 32;
const MAX_INSTRUCTIONS_CHARS = 2048;
const MAX_DESCRIPTION_CHARS = 1024;
const MAX_TOTAL_QUESTION_BYTES = 65_536;
const MIN_TIMEOUT_MS = 1;
const MAX_TIMEOUT_MS = 30_000;
const MAX_MEDIA_ITEMS = 4;
const MAX_MEDIA_BYTES = 8 * 1024 * 1024;
const MAX_TOTAL_MEDIA_BYTES = 16 * 1024 * 1024;
const MEDIA_TYPES = new Set(["image/png", "image/jpeg", "video/mp4", "audio/wav"]);
const BACKEND_KINDS = ["deterministic", "encoder", "cross_encoder", "specialist"] as const;
const CONTROLLED = /^[a-z][a-z0-9_.:/-]{0,127}$/;
const QUESTION_ID = /^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$/;
const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
const LABEL_TOKEN = "<<LABEL>>";
const SEPARATOR_TOKEN = "<<SEP>>";
const EXAMPLE_TOKEN = "<<EXAMPLE>>";
const MARKERS = [LABEL_TOKEN, SEPARATOR_TOKEN, EXAMPLE_TOKEN];
const REQUEST_FIELDS = new Set([
  "mode",
  "input_text",
  "options",
  "namespace",
  "purpose",
  "calibration_id",
  "timeout_ms",
  "schema_version",
  "compatibility_policy",
  "question_id",
  "instructions",
  "option_descriptions",
  "levels",
  "questions",
  "input_format",
  "max_input_chars",
  "media",
]);
const QUESTION_FIELDS = new Set([
  "question_id",
  "mode",
  "options",
  "option_descriptions",
  "instructions",
  "levels",
]);
const PROFILE_FIELDS = [
  "calibration_id",
  "version",
  "method",
  "minimum_confidence",
  "minimum_margin",
  "multi_label_threshold",
  "in_domain_digest",
  "out_of_domain_digest",
] as const;
const MEDIA_FIELDS = ["byte_size", "data", "media_type", "sha256"];

export type DecisionMode =
  | "fixed_choice"
  | "boolean_choice"
  | "ordered_preference"
  | "scalar_score"
  | "multi_label"
  | "question_set";
export type DecisionState =
  | "success"
  | "abstained"
  | "partial"
  | "unknown"
  | "conflict"
  | "unsupported"
  | "denied"
  | "failure";
export type DecisionBackendKind = (typeof BACKEND_KINDS)[number];
export type DecisionErrorKind =
  | "invalid_request"
  | "unsupported_contract"
  | "invalid_configuration";
export type DecisionEncoderErrorReason =
  | "encoder_output_missing"
  | "encoder_output_invalid"
  | "encoder_input_invalid";

const MODES: readonly DecisionMode[] = [
  "fixed_choice",
  "boolean_choice",
  "ordered_preference",
  "scalar_score",
  "multi_label",
  "question_set",
];

/**
 * Value-free decision error. `reason` is a fixed code; no request text,
 * option, prompt, identifier or path is ever included.
 */
export class DecisionRuntimeError extends Error {
  readonly kind: DecisionErrorKind;
  readonly reason: string;

  constructor(kind: DecisionErrorKind, reason: string) {
    super(`${kind}:${reason}`);
    this.name = "DecisionRuntimeError";
    this.kind = kind;
    this.reason = reason;
  }
}

/** Value-free encoder adapter failure. */
export class DecisionEncoderError extends Error {
  readonly reason: DecisionEncoderErrorReason;

  constructor(reason: DecisionEncoderErrorReason) {
    super(reason);
    this.name = "DecisionEncoderError";
    this.reason = reason;
  }
}

/** Versioned thresholds pinned to frozen calibration slices. */
export interface DecisionCalibrationProfile {
  calibration_id: string;
  version: string;
  method: string;
  minimum_confidence: number;
  minimum_margin: number;
  multi_label_threshold: number;
  in_domain_digest: string;
  out_of_domain_digest: string;
}

/** The profile every kit and the Python runtime use when none is supplied. */
export const DEFAULT_DECISION_CALIBRATION_PROFILE: Readonly<DecisionCalibrationProfile> =
  Object.freeze({
    calibration_id: DEFAULT_DECISION_CALIBRATION_ID,
    version: "1.0.0",
    method: "deterministic_margin",
    minimum_confidence: 0.7,
    minimum_margin: 0.15,
    multi_label_threshold: 0.7,
    in_domain_digest:
      "sha256:a6a2e0596c95b0568fe8871c99774fb1edc5b1db6de692e86172df952642dd6c",
    out_of_domain_digest:
      "sha256:d20c55e3877710223c6c37f87f6a8aaf9dea1b326087b37c60a81522ac0c9289",
  });

/** Namespace, purpose and backend-family allowlist. */
export interface DecisionAccessPolicy {
  allowed_namespaces: readonly string[];
  allowed_purposes: readonly string[];
  allowed_backend_kinds?: readonly DecisionBackendKind[];
  policy_version?: string;
}

/**
 * The Python default: namespace `default`; purposes `analytics`,
 * `care_review` and `quality`; every backend family.
 */
export const DEFAULT_DECISION_ACCESS_POLICY: Readonly<DecisionAccessPolicy> = Object.freeze({
  allowed_namespaces: ["default"],
  allowed_purposes: ["analytics", "care_review", "quality"],
  allowed_backend_kinds: BACKEND_KINDS,
  policy_version: DECISION_POLICY_VERSION,
});

/** Provenance supplied by the caller for an encoder backend. */
export interface DecisionBackendIdentityInput {
  backend_id: string;
  runtime: string;
  model_id: string;
  revision: string;
  license_id: string;
  license_sha256?: string;
}

/** Value-free provenance recorded in every result. */
export interface DecisionBackendIdentity extends DecisionBackendIdentityInput {
  kind: "encoder";
  local: true;
  deterministic: false;
}

export interface DecisionAccess {
  allowed: boolean;
  code: string | null;
  namespace: string;
  purpose: string;
  policy_version: string;
}

export interface DecisionOptionScore {
  index: number;
  option: string;
  score: number;
}

export interface DecisionReview {
  required: true;
  reasons: string[];
}

interface DecisionAnswerFields {
  mode: DecisionMode;
  state: DecisionState;
  code: string | null;
  option_scores: DecisionOptionScore[];
  choice: string | null;
  choices: string[];
  ranking: string[];
  scalar_score: number | null;
  confidence: number | null;
  margin: number | null;
}

/** Calibrated outcome for one named question of a question set. */
export interface DecisionQuestionResult extends DecisionAnswerFields {
  question_id: string;
  level_scores: [];
  expected_level: null;
  warnings: string[];
  review: DecisionReview;
}

/**
 * Canonical decision result with the schema's field names. Results for a
 * 1.0.0 request omit `level_scores`, `expected_level` and `questions`.
 */
export interface DecisionResult extends DecisionAnswerFields {
  level_scores?: [];
  expected_level?: null;
  questions?: DecisionQuestionResult[];
  calibration: DecisionCalibrationProfile;
  backend: DecisionBackendIdentity;
  access: DecisionAccess;
  warnings: string[];
  review: DecisionReview;
  advisory: string;
  autonomous_action: false;
  schema_version: string;
  compatibility_policy: string;
  extensions: Record<string, never>;
}

interface DecisionQuestion {
  questionId: string;
  mode: DecisionMode;
  options: string[];
  optionDescriptions: (string | null)[] | null;
  instructions: string | null;
  levels: string[];
}

// Transient request content, kept off the public object so that it never
// appears in `toString()`, `JSON.stringify()`, inspection, results or errors.
const VALIDATED: unique symbol = Symbol("openmed.decision.validated");
const REQUEST_CONTENT = new WeakMap<
  DecisionRequest,
  { inputText: string; questions: readonly DecisionQuestion[] }
>();

/**
 * One validated decision request. The input text and question wording are
 * private: they never appear in `toString()`, `JSON.stringify()`, results or
 * errors.
 */
export class DecisionRequest {
  readonly mode: DecisionMode;
  readonly namespace: string;
  readonly purpose: string;
  readonly calibrationId: string;
  readonly timeoutMs: number;
  /** The declared (or resolved) schema version. */
  readonly schemaVersion: string;
  /** The newest supported version not newer than `schemaVersion`. */
  readonly contractVersion: string;
  readonly inputFormat: "text" | "json";
  readonly maxInputChars: number | null;
  /** Number of media items. On-device runtimes are text-only and refuse media. */
  readonly mediaCount: number;
  /** Number of scored questions (one unless `mode` is `question_set`). */
  readonly questionCount: number;

  /** Use {@link parseDecisionRequest}; requests are only built after validation. */
  constructor(token: typeof VALIDATED, fields: {
    mode: DecisionMode;
    namespace: string;
    purpose: string;
    calibrationId: string;
    timeoutMs: number;
    schemaVersion: string;
    contractVersion: string;
    inputFormat: "text" | "json";
    maxInputChars: number | null;
    mediaCount: number;
    inputText: string;
    questions: readonly DecisionQuestion[];
  }) {
    if (token !== VALIDATED) {
      throw requestError("request_not_validated");
    }
    this.mode = fields.mode;
    this.namespace = fields.namespace;
    this.purpose = fields.purpose;
    this.calibrationId = fields.calibrationId;
    this.timeoutMs = fields.timeoutMs;
    this.schemaVersion = fields.schemaVersion;
    this.contractVersion = fields.contractVersion;
    this.inputFormat = fields.inputFormat;
    this.maxInputChars = fields.maxInputChars;
    this.mediaCount = fields.mediaCount;
    this.questionCount = fields.questions.length;
    REQUEST_CONTENT.set(this, { inputText: fields.inputText, questions: fields.questions });
  }

  toString(): string {
    return (
      `DecisionRequest(mode=${this.mode}, schema=${this.contractVersion}, ` +
      `questions=${this.questionCount}, media=${this.mediaCount})`
    );
  }

  toJSON(): Record<string, unknown> {
    return {
      mode: this.mode,
      schema_version: this.contractVersion,
      question_count: this.questionCount,
      media_count: this.mediaCount,
    };
  }
}

function requestContent(request: DecisionRequest): {
  inputText: string;
  questions: readonly DecisionQuestion[];
} {
  const content = REQUEST_CONTENT.get(request);
  if (content === undefined) {
    throw requestError("request_not_validated");
  }
  return content;
}

/** Tokenizer seam; it must add the checkpoint's special tokens and never truncate or pad. */
export interface DecisionTokenizer {
  encode(text: string): readonly number[] | Promise<readonly number[]>;
  /** The id of a control token such as `<<LABEL>>`, when the vocabulary has it. */
  tokenToId?(token: string): number | null | undefined;
}

/**
 * A GLiClass-style uni-encoder: `input_ids` and `attention_mask` of shape
 * `[1, tokens]` in, the first row of `logits` (one value per `<<LABEL>>`
 * slot, in label order) out.
 */
export interface DecisionEncoder {
  logits(
    inputIds: readonly number[],
    attentionMask: readonly number[],
  ): ArrayLike<number> | Promise<ArrayLike<number>>;
}

/** One synthetic few-shot example rendered as GLiClass example tokens. */
export interface GLiClassExample {
  text: string;
  labels: readonly string[];
}

/**
 * Backend settings of a GLiClass v4 uni-encoder decision runtime. The prompt,
 * examples and score levels belong to the runtime, as in the Python
 * `GLiClassDecisionBackend`; requests that carry per-question instructions,
 * option descriptions or levels are reported as `unsupported`.
 */
export interface GLiClassDecisionConfiguration {
  prompt?: string | null;
  examples?: readonly GLiClassExample[];
  /** Ordered levels, lowest first, scored as labels for `scalar_score`. */
  scoreLevels?: readonly string[];
  /** `true` (default) for GLiClass v4: labels and prompt before the text. */
  promptFirst?: boolean;
  /** Token budget; longer inputs are refused, never truncated. */
  maxLength?: number;
  /** The `<<LABEL>>` id (`class_token_index`); falls back to `tokenizer.tokenToId`. */
  labelTokenId?: number | null;
}

export interface EncoderDecisionRuntimeOptions {
  encoder: DecisionEncoder;
  tokenizer: DecisionTokenizer;
  identity: DecisionBackendIdentityInput;
  configuration?: GLiClassDecisionConfiguration;
  policy?: DecisionAccessPolicy;
  /** Profiles available to requests by `calibration_id`; the default profile when omitted. */
  calibrationProfiles?: readonly DecisionCalibrationProfile[];
  /** Monotonic milliseconds used for the request timeout. */
  clock?: () => number;
}

interface ResolvedConfiguration {
  prompt: string | null;
  examples: { text: string; labels: string[] }[];
  scoreLevels: string[];
  promptFirst: boolean;
  maxLength: number;
  labelTokenId: number | null;
}

type DecisionScores =
  | { kind: "options"; scores: number[] }
  | { kind: "scalar"; scalar: number; confidence: number };

type Answer = DecisionAnswerFields;

class DecisionRefusal extends Error {
  constructor(
    readonly state: DecisionState,
    readonly code: string,
  ) {
    super(code);
  }
}

/** On-device fixed-option decisions over a GLiClass-style encoder. */
export class EncoderDecisionRuntime {
  readonly identity: DecisionBackendIdentity;
  readonly #encoder: DecisionEncoder;
  readonly #tokenizer: DecisionTokenizer;
  readonly #configuration: ResolvedConfiguration;
  readonly #policy: Required<DecisionAccessPolicy>;
  readonly #profiles: Map<string, DecisionCalibrationProfile>;
  readonly #clock: () => number;

  constructor(options: EncoderDecisionRuntimeOptions) {
    this.#encoder = options.encoder;
    this.#tokenizer = options.tokenizer;
    this.identity = validateIdentity(options.identity);
    this.#configuration = resolveConfiguration(options.configuration ?? {});
    this.#policy = validatePolicy(options.policy ?? DEFAULT_DECISION_ACCESS_POLICY);
    const profiles = (options.calibrationProfiles ?? [DEFAULT_DECISION_CALIBRATION_PROFILE]).map(
      validateDecisionCalibrationProfile,
    );
    this.#profiles = new Map(profiles.map((profile) => [profile.calibration_id, profile]));
    if (profiles.length === 0 || this.#profiles.size !== profiles.length) {
      throw configurationError("calibration_profiles_invalid");
    }
    this.#clock = options.clock ?? defaultClock;
  }

  /**
   * Validates and decides a request: a mapping with the schema's field names,
   * its JSON text, or a {@link DecisionRequest}. Contract violations reject
   * with a {@link DecisionRuntimeError}; abstentions, denials, refusals and
   * failures are typed result states, never fabricated selections.
   */
  async decide(request: unknown): Promise<DecisionResult> {
    const parsed = request instanceof DecisionRequest ? request : parseDecisionRequest(request);
    const access = decideAccess(this.#policy, parsed, this.identity.kind);
    const profile = this.#profiles.get(parsed.calibrationId);
    if (profile === undefined) {
      return this.#terminal(
        parsed,
        "unsupported",
        "calibration_unavailable",
        { ...DEFAULT_DECISION_CALIBRATION_PROFILE },
        access,
      );
    }
    if (!access.allowed) {
      return this.#terminal(parsed, "denied", access.code ?? "access_denied", profile, access);
    }
    if (parsed.mediaCount > 0) {
      return this.#terminal(parsed, "unsupported", "backend_unsupported", profile, access);
    }
    const started = this.#clock();
    const answers: Answer[] = [];
    for (const question of requestContent(parsed).questions) {
      if (parsed.mode === "question_set" && this.#clock() - started > parsed.timeoutMs) {
        answers.push(terminalAnswer(question.mode, "failure", "backend_timeout"));
        continue;
      }
      answers.push(await this.#score(question, parsed, profile, started));
    }
    return this.#assemble(parsed, answers, profile, access);
  }

  async #score(
    question: DecisionQuestion,
    request: DecisionRequest,
    profile: DecisionCalibrationProfile,
    started: number,
  ): Promise<Answer> {
    let output: DecisionScores;
    try {
      output = await this.#scores(question, requestContent(request).inputText);
    } catch (error) {
      if (error instanceof DecisionRefusal) {
        return terminalAnswer(question.mode, error.state, error.code);
      }
      return terminalAnswer(question.mode, "failure", "backend_failure");
    }
    if (this.#clock() - started > request.timeoutMs) {
      return terminalAnswer(question.mode, "failure", "backend_timeout");
    }
    return decideScores(output, question, profile);
  }

  async #scores(question: DecisionQuestion, text: string): Promise<DecisionScores> {
    const unsupported = new DecisionRefusal("unsupported", "backend_unsupported");
    const conflict = new DecisionRefusal("conflict", "backend_conflict");
    if (
      question.instructions !== null ||
      (question.optionDescriptions ?? []).some((item) => item !== null) ||
      question.levels.length > 0
    ) {
      throw unsupported;
    }
    const labels =
      question.mode === "scalar_score" ? this.#configuration.scoreLevels : question.options;
    if (hasMarker(text) || labels.some(hasMarker)) {
      throw unsupported;
    }
    const inputIds = Array.from(
      await this.#tokenizer.encode(formatInput(this.#configuration, text, labels)),
    );
    if (inputIds.length > this.#configuration.maxLength) {
      throw unsupported;
    }
    const labelId =
      this.#configuration.labelTokenId ?? this.#tokenizer.tokenToId?.(LABEL_TOKEN) ?? null;
    if (labelId !== null && inputIds.filter((id) => id === labelId).length !== labels.length) {
      throw conflict;
    }
    let raw: ArrayLike<number>;
    try {
      raw = await this.#encoder.logits(
        inputIds,
        inputIds.map(() => 1),
      );
    } catch (error) {
      if (
        error instanceof DecisionEncoderError &&
        error.reason !== "encoder_input_invalid"
      ) {
        throw conflict;
      }
      throw error;
    }
    if (raw.length < labels.length) {
      throw conflict;
    }
    const logits = Array.from({ length: labels.length }, (_, index) => Number(raw[index]));
    if (!logits.every(Number.isFinite)) {
      throw conflict;
    }
    const probabilities =
      question.mode === "multi_label" ? logits.map(sigmoid) : softmax(logits);
    if (question.mode === "scalar_score") {
      return {
        kind: "scalar",
        scalar: expectedLevel(probabilities),
        confidence: Math.max(...probabilities),
      };
    }
    return { kind: "options", scores: probabilities };
  }

  #terminal(
    request: DecisionRequest,
    state: DecisionState,
    code: string,
    profile: DecisionCalibrationProfile,
    access: DecisionAccess,
  ): DecisionResult {
    const answers = requestContent(request).questions.map((question) =>
      terminalAnswer(question.mode, state, code),
    );
    return this.#assemble(request, answers, profile, access);
  }

  #assemble(
    request: DecisionRequest,
    answers: Answer[],
    profile: DecisionCalibrationProfile,
    access: DecisionAccess,
  ): DecisionResult {
    const questions = requestContent(request).questions;
    let outer: Answer = answers[0] ?? terminalAnswer(request.mode, "failure", "backend_failure");
    let questionResults: DecisionQuestionResult[] = [];
    if (request.mode === "question_set") {
      questionResults = questions.map((question, index) => {
        const answer = answers[index] ?? terminalAnswer(question.mode, "failure", "backend_failure");
        return {
          question_id: question.questionId,
          ...answer,
          level_scores: [],
          expected_level: null,
          warnings: warningsFor(answer.code),
          review: reviewFor(answer.code),
        };
      });
      outer = aggregate(answers);
    }
    const result: DecisionResult = {
      ...outer,
      mode: request.mode,
      calibration: { ...profile },
      backend: { ...this.identity },
      access,
      warnings: warningsFor(outer.code),
      review: reviewFor(outer.code),
      advisory: DECISION_ADVISORY,
      autonomous_action: false,
      schema_version: request.contractVersion,
      compatibility_policy: DECISION_COMPATIBILITY_POLICY,
      extensions: {},
    };
    if (request.contractVersion !== BASE_VERSION) {
      result.level_scores = [];
      result.expected_level = null;
      result.questions = questionResults;
    }
    return result;
  }
}

/** Renders the uni-encoder input used by `gliclass` 0.1.21 and the Python reference. */
export function formatGLiClassInput(
  text: string,
  labels: readonly string[],
  configuration: GLiClassDecisionConfiguration = {},
): string {
  return formatInput(resolveConfiguration(configuration), text, labels);
}

/** Validates a plain calibration profile exactly as the Python `DecisionCalibrationProfile`. */
export function validateDecisionCalibrationProfile(value: unknown): DecisionCalibrationProfile {
  const invalid = () => configurationError("calibration_profile_invalid");
  if (!isRecord(value)) {
    throw invalid();
  }
  const keys = Object.keys(value).sort();
  if (keys.join(",") !== [...PROFILE_FIELDS].sort().join(",")) {
    throw invalid();
  }
  const profile = value as unknown as DecisionCalibrationProfile;
  const probabilities = [
    profile.minimum_confidence,
    profile.minimum_margin,
    profile.multi_label_threshold,
  ];
  if (
    !isControlled(profile.calibration_id) ||
    typeof profile.version !== "string" ||
    !VERSION.test(profile.version) ||
    !isControlled(profile.method) ||
    !probabilities.every(isProbability) ||
    profile.minimum_margin > profile.minimum_confidence ||
    typeof profile.in_domain_digest !== "string" ||
    !DIGEST.test(profile.in_domain_digest) ||
    typeof profile.out_of_domain_digest !== "string" ||
    !DIGEST.test(profile.out_of_domain_digest)
  ) {
    throw invalid();
  }
  return {
    calibration_id: profile.calibration_id,
    version: profile.version,
    method: profile.method,
    minimum_confidence: profile.minimum_confidence,
    minimum_margin: profile.minimum_margin,
    multi_label_threshold: profile.multi_label_threshold,
    in_domain_digest: profile.in_domain_digest,
    out_of_domain_digest: profile.out_of_domain_digest,
  };
}

/**
 * Parses and validates a decision request (a mapping or its JSON text), in
 * the order of the Python contract: structure, values, contract support, then
 * the timeout.
 */
export function parseDecisionRequest(payload: unknown): DecisionRequest {
  let value = payload;
  if (typeof payload === "string") {
    try {
      value = JSON.parse(payload) as unknown;
    } catch {
      throw requestError("request_json_invalid");
    }
  }
  if (!isRecord(value)) {
    throw requestError("request_not_object");
  }
  if (!Object.keys(value).every((key) => REQUEST_FIELDS.has(key))) {
    throw requestError("request_field_unsupported");
  }
  if (!("mode" in value) || !("input_text" in value)) {
    throw requestError("request_field_missing");
  }
  const mode = value.mode;
  if (typeof mode !== "string" || !MODES.includes(mode as DecisionMode)) {
    throw requestError("mode_invalid");
  }
  const decisionMode = mode as DecisionMode;
  const options = stringList("options" in value ? value.options : [], "options_invalid");
  const descriptions = optionalProseList(value.option_descriptions);
  const levels = stringList(value.levels ?? [], "levels_invalid");
  const rawQuestions = value.questions ?? [];
  if (!Array.isArray(rawQuestions)) {
    throw requestError("questions_invalid");
  }
  const inputFormat = value.input_format ?? "text";
  if (inputFormat !== "text" && inputFormat !== "json") {
    throw requestError("input_format_invalid");
  }
  const mediaCount = mediaItems("media" in value ? value.media : undefined);

  let maxInputChars: number | null = null;
  if (value.max_input_chars !== undefined && value.max_input_chars !== null) {
    if (!isInteger(value.max_input_chars, 1, MAX_CONFIGURABLE_INPUT_CHARS)) {
      throw requestError("max_input_chars_invalid");
    }
    maxInputChars = value.max_input_chars;
  }
  const inputText = value.input_text;
  if (typeof inputText !== "string") {
    throw requestError("input_invalid");
  }
  checkInput(inputText, maxInputChars ?? MAX_INPUT_CHARS, inputFormat);

  const instructions = prose(value.instructions, MAX_INSTRUCTIONS_CHARS, "instructions_invalid");
  let questionId: string | null = null;
  let questions: DecisionQuestion[];
  if (decisionMode === "question_set") {
    if (
      options.length > 0 ||
      descriptions !== null ||
      instructions !== null ||
      levels.length > 0 ||
      (value.question_id !== undefined && value.question_id !== null)
    ) {
      throw requestError("question_set_fields_invalid");
    }
    questions = validatedQuestions(rawQuestions);
  } else {
    if (rawQuestions.length > 0) {
      throw requestError("questions_require_question_set");
    }
    if (value.question_id !== undefined && value.question_id !== null) {
      if (typeof value.question_id !== "string" || !QUESTION_ID.test(value.question_id)) {
        throw requestError("question_id_invalid");
      }
      questionId = value.question_id;
    }
    const question = validatedQuestion(
      questionId ?? DEFAULT_DECISION_QUESTION_ID,
      decisionMode,
      options,
      descriptions,
      instructions,
      levels,
    );
    if (questionBytes(question) > MAX_TOTAL_QUESTION_BYTES) {
      throw requestError("questions_too_large");
    }
    questions = [question];
  }
  const namespace = controlled("namespace" in value ? value.namespace : "default", "namespace_invalid");
  const purpose = controlled("purpose" in value ? value.purpose : "care_review", "purpose_invalid");
  const calibrationId = controlled(
    "calibration_id" in value ? value.calibration_id : DEFAULT_DECISION_CALIBRATION_ID,
    "calibration_id_invalid",
  );

  const usesQuestionFields =
    decisionMode === "question_set" ||
    questionId !== null ||
    descriptions !== null ||
    instructions !== null ||
    levels.length > 0 ||
    inputFormat !== "text" ||
    maxInputChars !== null;
  const usesMedia = mediaCount > 0;
  let declared: string;
  if (value.schema_version === undefined || value.schema_version === null) {
    declared = usesMedia ? MEDIA_VERSION : usesQuestionFields ? QUESTION_VERSION : BASE_VERSION;
  } else if (typeof value.schema_version === "string" && VERSION.test(value.schema_version)) {
    declared = value.schema_version;
  } else {
    throw requestError("schema_version_invalid");
  }

  if (declared.split(".")[0] !== "1") {
    throw contractError("schema_major_unsupported");
  }
  const contract = contractVersion(declared);
  if (usesQuestionFields && contract === BASE_VERSION) {
    throw contractError("question_fields_require_1_1_0");
  }
  if (usesMedia && versionPrecedes(contract, MEDIA_VERSION)) {
    throw contractError("media_requires_1_2_0");
  }
  const policy =
    "compatibility_policy" in value ? value.compatibility_policy : DECISION_COMPATIBILITY_POLICY;
  if (policy !== DECISION_COMPATIBILITY_POLICY) {
    throw contractError("compatibility_policy_unsupported");
  }

  const timeout = "timeout_ms" in value ? value.timeout_ms : 5000;
  if (!isInteger(timeout, MIN_TIMEOUT_MS, MAX_TIMEOUT_MS)) {
    throw requestError("timeout_invalid");
  }
  return new DecisionRequest(VALIDATED, {
    mode: decisionMode,
    namespace,
    purpose,
    calibrationId,
    timeoutMs: timeout,
    schemaVersion: declared,
    contractVersion: contract,
    inputFormat,
    maxInputChars,
    mediaCount,
    inputText,
    questions,
  });
}

export interface OrtDecisionEncoderOptions {
  session: OrtInferenceSession;
  Tensor: OrtTensorConstructor;
  /** Integer type of `input_ids` and `attention_mask`; `int64` by default. */
  inputType?: "int64" | "int32";
  outputName?: string;
}

/**
 * Runs an ONNX export of a GLiClass-style uni-encoder through an ONNX Runtime
 * Web (or onnxruntime-node) session. Inputs are `[1, tokens]` integer tensors;
 * the output must be a `[1, labels]` `logits` tensor.
 */
export function createOrtDecisionEncoder(options: OrtDecisionEncoderOptions): DecisionEncoder {
  const { session, Tensor } = options;
  const inputType = options.inputType ?? "int64";
  const outputName = options.outputName ?? "logits";
  const tensor = (values: readonly number[]) =>
    new Tensor(
      inputType,
      inputType === "int64"
        ? BigInt64Array.from(values, (item) => BigInt(item))
        : Int32Array.from(values),
      [1, values.length],
    );
  return {
    async logits(inputIds, attentionMask) {
      if (
        inputIds.length === 0 ||
        inputIds.length !== attentionMask.length ||
        !inputIds.every((item) => Number.isInteger(item) && item >= 0)
      ) {
        throw new DecisionEncoderError("encoder_input_invalid");
      }
      const outputs = await session.run({
        input_ids: tensor(inputIds),
        attention_mask: tensor(attentionMask),
      });
      const output = outputs[outputName] as { data?: unknown; dims?: readonly number[] } | undefined;
      if (output === undefined || output === null) {
        throw new DecisionEncoderError("encoder_output_missing");
      }
      const dims = output.dims;
      const data = output.data as ArrayLike<number | bigint> | undefined;
      const columns = dims?.[1];
      if (
        dims === undefined ||
        dims.length !== 2 ||
        (dims[0] ?? 0) < 1 ||
        columns === undefined ||
        columns < 1 ||
        data === undefined ||
        typeof data.length !== "number" ||
        data.length < columns
      ) {
        throw new DecisionEncoderError("encoder_output_invalid");
      }
      return Array.from({ length: columns }, (_, index) => Number(data[index]));
    },
  };
}

/**
 * Loads a local GLiClass ONNX export with the package's offline ONNX Runtime
 * Web loader and returns a decision encoder for it.
 */
export async function loadOrtWebDecisionEncoder(
  options: OrtWebLoaderOptions & { inputType?: "int64" | "int32"; outputName?: string },
): Promise<DecisionEncoder & { backend: OrtWebBackendChoice }> {
  const loaded = await loadOrtWebSession(options);
  const Tensor = loaded.runtime?.Tensor;
  if (Tensor === undefined) {
    throw configurationError("encoder_tensor_unavailable");
  }
  const encoderOptions: OrtDecisionEncoderOptions = { session: loaded.session, Tensor };
  if (options.inputType !== undefined) {
    encoderOptions.inputType = options.inputType;
  }
  if (options.outputName !== undefined) {
    encoderOptions.outputName = options.outputName;
  }
  const encoder = createOrtDecisionEncoder(encoderOptions);
  return { logits: encoder.logits, backend: loaded.backend };
}

// MARK: request validation helpers

function requestError(reason: string): DecisionRuntimeError {
  return new DecisionRuntimeError("invalid_request", reason);
}

function contractError(reason: string): DecisionRuntimeError {
  return new DecisionRuntimeError("unsupported_contract", reason);
}

function configurationError(reason: string): DecisionRuntimeError {
  return new DecisionRuntimeError("invalid_configuration", reason);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isInteger(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= minimum && value <= maximum;
}

function isProbability(value: unknown): boolean {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isControlled(value: unknown): value is string {
  return typeof value === "string" && CONTROLLED.test(value);
}

function codePoints(value: string): number {
  let count = 0;
  for (const _ of value) {
    count += 1;
  }
  return count;
}

function utf8Size(value: string): number {
  return new TextEncoder().encode(value).length;
}

function isBlank(value: string): boolean {
  return value.trim().length === 0;
}

function hasControl(value: string, allowed = ""): boolean {
  for (const char of value) {
    const code = char.codePointAt(0) ?? 0;
    if ((code < 32 && !allowed.includes(char)) || code === 127) {
      return true;
    }
  }
  return false;
}

function isBoundedLine(value: string, maximum: number): boolean {
  return (
    !isBlank(value) &&
    codePoints(value) <= maximum &&
    !LONE_SURROGATE.test(value) &&
    !hasControl(value)
  );
}

/** NFKC, case folding (upper- then lower-casing) and whitespace collapsing. */
function normalizedOption(value: string): string {
  return value
    .normalize("NFKC")
    .toUpperCase()
    .toLowerCase()
    .split(/\s+/u)
    .filter((part) => part.length > 0)
    .join(" ");
}

function controlled(value: unknown, reason: string): string {
  if (!isControlled(value)) {
    throw requestError(reason);
  }
  return value;
}

function stringList(value: unknown, reason: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw requestError(reason);
  }
  return value as string[];
}

function optionalProseList(value: unknown): (string | null)[] | null {
  if (value === undefined || value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    throw requestError("option_descriptions_invalid");
  }
  return value.map((item) => prose(item, MAX_DESCRIPTION_CHARS, "option_descriptions_invalid"));
}

function prose(value: unknown, maximum: number, reason: string): string | null {
  if (value === undefined || value === null) {
    return null;
  }
  if (
    typeof value !== "string" ||
    isBlank(value) ||
    codePoints(value) > maximum ||
    LONE_SURROGATE.test(value) ||
    hasControl(value, "\n\r\t") ||
    utf8Size(value) > maximum * INPUT_BYTES_PER_CHAR
  ) {
    throw requestError(reason);
  }
  return value;
}

function mediaItems(value: unknown): number {
  if (value === undefined) {
    return 0;
  }
  if (!Array.isArray(value) || value.length > MAX_MEDIA_ITEMS) {
    throw requestError("media_invalid");
  }
  let total = 0;
  for (const item of value) {
    if (
      !isRecord(item) ||
      Object.keys(item).sort().join(",") !== MEDIA_FIELDS.join(",") ||
      !isInteger(item.byte_size, 1, MAX_MEDIA_BYTES) ||
      typeof item.data !== "string" ||
      codePoints(item.data) !== 4 * Math.floor((item.byte_size + 2) / 3) ||
      typeof item.media_type !== "string" ||
      !MEDIA_TYPES.has(item.media_type) ||
      typeof item.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(item.sha256)
    ) {
      throw requestError("media_invalid");
    }
    total += item.byte_size;
  }
  if (total > MAX_TOTAL_MEDIA_BYTES) {
    throw requestError("media_invalid");
  }
  // Media is refused on device and is never decoded here.
  return value.length;
}

function checkInput(text: string, maximum: number, format: "text" | "json"): void {
  if (
    isBlank(text) ||
    text.includes("\u0000") ||
    LONE_SURROGATE.test(text) ||
    codePoints(text) > maximum ||
    utf8Size(text) > maximum * INPUT_BYTES_PER_CHAR
  ) {
    throw requestError("input_invalid");
  }
  if (format === "json") {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text) as unknown;
    } catch {
      throw requestError("input_invalid");
    }
    if (typeof parsed !== "object" || parsed === null) {
      throw requestError("input_invalid");
    }
  }
}

function option(value: string, reason: string): string {
  const stripped = value.trim();
  if (!isBoundedLine(stripped, MAX_OPTION_CHARS) || utf8Size(stripped) > MAX_OPTION_BYTES) {
    throw requestError(reason);
  }
  return stripped;
}

function distinct(values: string[], reason: string): void {
  const normalized = values.map(normalizedOption);
  if (
    new Set(normalized).size !== normalized.length ||
    values.reduce((total, item) => total + utf8Size(item), 0) > MAX_TOTAL_OPTION_BYTES
  ) {
    throw requestError(reason);
  }
}

function validatedQuestion(
  questionId: string,
  mode: DecisionMode,
  rawOptions: string[],
  descriptions: (string | null)[] | null,
  instructions: string | null,
  rawLevels: string[],
): DecisionQuestion {
  const options = rawOptions.map((item) => option(item, "options_invalid"));
  const countValid =
    mode === "scalar_score"
      ? options.length === 0
      : mode === "boolean_choice"
        ? options.length === 2
        : options.length >= 2 && options.length <= MAX_OPTIONS;
  if (!countValid) {
    throw requestError("options_invalid");
  }
  if (options.length > 0) {
    distinct(options, "options_invalid");
  }
  if (descriptions !== null && (options.length === 0 || descriptions.length !== options.length)) {
    throw requestError("option_descriptions_invalid");
  }
  let levels: string[] = [];
  if (rawLevels.length > 0) {
    if (mode !== "scalar_score") {
      throw requestError("levels_invalid");
    }
    levels = rawLevels.map((item) => option(item, "levels_invalid"));
    if (levels.length < 2 || levels.length > MAX_OPTIONS) {
      throw requestError("levels_invalid");
    }
    distinct(levels, "levels_invalid");
  }
  return {
    questionId,
    mode,
    options,
    optionDescriptions: descriptions,
    instructions,
    levels,
  };
}

function validatedQuestions(values: unknown[]): DecisionQuestion[] {
  const questions = values.map((value) => {
    if (!isRecord(value) || !Object.keys(value).every((key) => QUESTION_FIELDS.has(key))) {
      throw requestError("question_invalid");
    }
    if (typeof value.question_id !== "string" || !QUESTION_ID.test(value.question_id)) {
      throw requestError("question_id_invalid");
    }
    const mode = value.mode;
    if (typeof mode !== "string" || !MODES.includes(mode as DecisionMode) || mode === "question_set") {
      throw requestError("question_mode_invalid");
    }
    return validatedQuestion(
      value.question_id,
      mode as DecisionMode,
      stringList(value.options ?? [], "options_invalid"),
      optionalProseList(value.option_descriptions),
      prose(value.instructions, MAX_INSTRUCTIONS_CHARS, "instructions_invalid"),
      stringList(value.levels ?? [], "levels_invalid"),
    );
  });
  if (
    questions.length < 1 ||
    questions.length > MAX_QUESTIONS ||
    new Set(questions.map((item) => item.questionId)).size !== questions.length ||
    questions.reduce((total, item) => total + questionBytes(item), 0) > MAX_TOTAL_QUESTION_BYTES
  ) {
    throw requestError("questions_invalid");
  }
  return questions;
}

function questionBytes(question: DecisionQuestion): number {
  const parts = [
    ...question.options,
    ...(question.optionDescriptions ?? []).filter((item): item is string => item !== null),
    ...question.levels,
  ];
  return (
    parts.reduce((total, item) => total + utf8Size(item), 0) +
    (question.instructions === null ? 0 : utf8Size(question.instructions))
  );
}

function contractVersion(declared: string): string {
  let selected = BASE_VERSION;
  for (const version of SUPPORTED_DECISION_SCHEMA_VERSIONS) {
    if (!versionPrecedes(declared, version)) {
      selected = version;
    }
  }
  return selected;
}

function versionPrecedes(left: string, right: string): boolean {
  const lhs = left.split(".").map((part) => Number(part));
  const rhs = right.split(".").map((part) => Number(part));
  for (let index = 0; index < Math.min(lhs.length, rhs.length); index += 1) {
    const a = lhs[index] ?? 0;
    const b = rhs[index] ?? 0;
    if (a !== b) {
      return a < b;
    }
  }
  return lhs.length < rhs.length;
}

// MARK: runtime helpers

function defaultClock(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function validateIdentity(input: DecisionBackendIdentityInput): DecisionBackendIdentity {
  const invalid = () => configurationError("backend_identity_invalid");
  const fields = [input.backend_id, input.runtime, input.model_id, input.revision];
  if (
    !fields.every((item) => typeof item === "string" && isBoundedLine(item, 256)) ||
    typeof input.license_id !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9 .()+:/_-]{0,127}$/.test(input.license_id) ||
    input.license_id !== input.license_id.trim()
  ) {
    throw invalid();
  }
  if (
    input.license_sha256 !== undefined &&
    (!/^[0-9a-f]{64}$/.test(input.license_sha256) || !/^[0-9a-f]{40,64}$/.test(input.revision))
  ) {
    throw invalid();
  }
  return {
    backend_id: input.backend_id,
    kind: "encoder",
    runtime: input.runtime,
    model_id: input.model_id,
    revision: input.revision,
    license_id: input.license_id,
    ...(input.license_sha256 === undefined ? {} : { license_sha256: input.license_sha256 }),
    local: true,
    deterministic: false,
  };
}

function validatePolicy(policy: DecisionAccessPolicy): Required<DecisionAccessPolicy> {
  const kinds = policy.allowed_backend_kinds ?? BACKEND_KINDS;
  const version = policy.policy_version ?? DECISION_POLICY_VERSION;
  if (
    policy.allowed_namespaces.length === 0 ||
    policy.allowed_purposes.length === 0 ||
    kinds.length === 0 ||
    ![...policy.allowed_namespaces, ...policy.allowed_purposes].every(isControlled) ||
    !kinds.every((kind) => (BACKEND_KINDS as readonly string[]).includes(kind)) ||
    !VERSION.test(version)
  ) {
    throw configurationError("access_policy_invalid");
  }
  return {
    allowed_namespaces: [...policy.allowed_namespaces],
    allowed_purposes: [...policy.allowed_purposes],
    allowed_backend_kinds: [...kinds],
    policy_version: version,
  };
}

function decideAccess(
  policy: Required<DecisionAccessPolicy>,
  request: DecisionRequest,
  kind: DecisionBackendKind,
): DecisionAccess {
  let code: string | null = null;
  if (!policy.allowed_namespaces.includes(request.namespace)) {
    code = "namespace_denied";
  } else if (!policy.allowed_purposes.includes(request.purpose)) {
    code = "purpose_denied";
  } else if (!policy.allowed_backend_kinds.includes(kind)) {
    code = "backend_denied";
  }
  return {
    allowed: code === null,
    code,
    namespace: request.namespace,
    purpose: request.purpose,
    policy_version: policy.policy_version,
  };
}

function isValidGLiClassText(
  value: unknown,
  maximumChars: number,
  maximumBytes: number,
  multiline: boolean,
): value is string {
  return (
    typeof value === "string" &&
    !isBlank(value) &&
    codePoints(value) <= maximumChars &&
    !LONE_SURROGATE.test(value) &&
    !hasControl(value, multiline ? "\n\r\t" : "") &&
    utf8Size(value) <= maximumBytes &&
    !hasMarker(value)
  );
}

function gliclassLabel(value: unknown, reason: string): string {
  const stripped = typeof value === "string" ? value.trim() : value;
  if (!isValidGLiClassText(stripped, 256, 1024, false)) {
    throw configurationError(reason);
  }
  return stripped;
}

function resolveConfiguration(configuration: GLiClassDecisionConfiguration): ResolvedConfiguration {
  const reason = "gliclass_configuration_invalid";
  const prompt = configuration.prompt ?? null;
  if (prompt !== null && !isValidGLiClassText(prompt, 4096, 16_384, true)) {
    throw configurationError(reason);
  }
  const examples = (configuration.examples ?? []).map((example) => {
    if (
      !isValidGLiClassText(example.text, 2048, 8192, true) ||
      example.labels.length < 1 ||
      example.labels.length > MAX_OPTIONS
    ) {
      throw configurationError("gliclass_example_invalid");
    }
    return {
      text: example.text,
      labels: example.labels.map((label) => gliclassLabel(label, "gliclass_example_invalid")),
    };
  });
  const scoreLevels = (configuration.scoreLevels ?? GLICLASS_DEFAULT_SCORE_LEVELS).map((level) =>
    gliclassLabel(level, reason),
  );
  const maxLength = configuration.maxLength ?? GLICLASS_DEFAULT_MAX_LENGTH;
  const labelTokenId = configuration.labelTokenId ?? null;
  if (
    examples.length > 8 ||
    scoreLevels.length < 2 ||
    scoreLevels.length > 16 ||
    new Set(scoreLevels.map((level) => level.toUpperCase().toLowerCase())).size !==
      scoreLevels.length ||
    !Number.isInteger(maxLength) ||
    maxLength < 1 ||
    (labelTokenId !== null && (!Number.isInteger(labelTokenId) || labelTokenId < 0))
  ) {
    throw configurationError(reason);
  }
  return {
    prompt,
    examples,
    scoreLevels,
    promptFirst: configuration.promptFirst ?? true,
    maxLength,
    labelTokenId,
  };
}

function formatInput(
  configuration: ResolvedConfiguration,
  text: string,
  labels: readonly string[],
): string {
  let head = labels.map((label) => LABEL_TOKEN + label).join("") + SEPARATOR_TOKEN;
  if (configuration.prompt) {
    head += configuration.prompt;
  }
  let rendered = "";
  if (configuration.examples.length > 0) {
    rendered =
      configuration.examples
        .map((example) => `${EXAMPLE_TOKEN}${example.text} \nLabels:\n ${example.labels.join(", ")}`)
        .join("") + SEPARATOR_TOKEN;
  }
  return configuration.promptFirst ? head + text + rendered : text + head + rendered;
}

function hasMarker(value: string): boolean {
  return MARKERS.some((marker) => value.includes(marker));
}

function unit(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function softmax(logits: number[]): number[] {
  const peak = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - peak));
  let total = 0;
  for (const value of exps) {
    total += value;
  }
  return exps.map((value) => unit(value / total));
}

function sigmoid(value: number): number {
  if (value >= 0) {
    return unit(1 / (1 + Math.exp(-value)));
  }
  const exp = Math.exp(value);
  return unit(exp / (1 + exp));
}

function expectedLevel(probabilities: number[]): number {
  let expected = 0;
  probabilities.forEach((value, index) => {
    expected += index * value;
  });
  return unit(expected / (probabilities.length - 1));
}

function terminalAnswer(mode: DecisionMode, state: DecisionState, code: string): Answer {
  return {
    mode,
    state,
    code,
    option_scores: [],
    choice: null,
    choices: [],
    ranking: [],
    scalar_score: null,
    confidence: null,
    margin: null,
  };
}

/** The Python `_decision_from_scores` policy for encoder outputs. */
function decideScores(
  output: DecisionScores,
  question: DecisionQuestion,
  profile: DecisionCalibrationProfile,
): Answer {
  if (output.kind === "scalar") {
    const success = output.confidence >= profile.minimum_confidence;
    return {
      ...terminalAnswer(question.mode, success ? "success" : "abstained", "low_confidence"),
      code: success ? null : "low_confidence",
      scalar_score: success ? output.scalar : null,
      confidence: output.confidence,
    };
  }
  const scores = output.scores;
  const options = question.options;
  const score = (index: number) => scores[index] ?? 0;
  const ranked = scores
    .map((_, index) => index)
    .sort((left, right) => score(right) - score(left) || left - right);
  const confidence = score(ranked[0] ?? 0);
  const margin = confidence - score(ranked[1] ?? 0);
  const ranking = ranked.map((index) => options[index] ?? "");
  const answer: Answer = {
    mode: question.mode,
    state: "abstained",
    code: null,
    option_scores: scores.map((value, index) => ({ index, option: options[index] ?? "", score: value })),
    choice: null,
    choices: [],
    ranking,
    scalar_score: null,
    confidence,
    margin,
  };
  if (question.mode === "multi_label") {
    const selected = options.filter((_, index) => score(index) >= profile.multi_label_threshold);
    answer.state = selected.length > 0 ? "success" : "abstained";
    answer.code = selected.length > 0 ? null : "no_label_above_threshold";
    answer.choice = selected[0] ?? null;
    answer.choices = selected;
    return answer;
  }
  const clearsConfidence = confidence >= profile.minimum_confidence;
  const clearsMargin = margin >= profile.minimum_margin;
  answer.state = clearsConfidence && clearsMargin ? "success" : "abstained";
  answer.code = !clearsConfidence ? "low_confidence" : !clearsMargin ? "ambiguous_scores" : null;
  if (answer.state === "success") {
    answer.choice = ranking[0] ?? null;
    answer.choices = answer.choice === null ? [] : [answer.choice];
  }
  return answer;
}

/** Question-set state: success, the shared state and code, `question_set_<state>`, or partial. */
function aggregate(answers: Answer[]): Answer {
  const states = new Set(answers.map((answer) => answer.state));
  const outer = terminalAnswer("question_set", "success", "");
  if (states.size === 1 && states.has("success")) {
    return { ...outer, code: null };
  }
  const outcomes = new Set(answers.map((answer) => `${answer.state}|${answer.code ?? ""}`));
  const first = answers[0];
  if (outcomes.size === 1 && first !== undefined) {
    return { ...outer, state: first.state, code: first.code };
  }
  if (states.size === 1 && first !== undefined) {
    return { ...outer, state: first.state, code: `question_set_${first.state}` };
  }
  return { ...outer, state: "partial", code: "question_set_partial" };
}

function warningsFor(code: string | null): string[] {
  return code === null || code === "human_review_required"
    ? ["human_review_required"]
    : ["human_review_required", code];
}

function reviewFor(code: string | null): DecisionReview {
  return {
    required: true,
    reasons: code === null ? ["clinical_use_requires_review"] : ["clinical_use_requires_review", code],
  };
}
