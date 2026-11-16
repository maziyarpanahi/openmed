/**
 * `openmed/speech`: the streaming speech data contract shared with
 * `openmed.multimodal.streaming.speech.providers` (Python) and OpenMedKit's
 * `StreamingSpeech` types (Swift).
 *
 * Times are stream-relative integer milliseconds, token indices are zero-based,
 * confidences are integer parts per million and text lengths count Unicode
 * code points (Swift `unicodeScalars`), never UTF-16 code units. This module
 * validates data only: it captures no audio, runs no recognizer and makes no
 * network request.
 *
 * Text is protected. It is readable through the `text` accessor but is kept
 * out of `JSON.stringify`, `String()`, Node's `util.inspect` and every error
 * message. Decoder confidence is uncalibrated and is not clinical certainty;
 * a finalized segment never finalizes a clinical note or releases PHI.
 */

/** Stable contract error codes, matching Swift's `StreamingSpeechError`. */
export const SPEECH_CONTRACT_ERROR_CODES = Object.freeze([
  "speech_hypothesis_invalid",
  "speech_artifact_unverified",
] as const);

export type SpeechContractErrorCode = (typeof SPEECH_CONTRACT_ERROR_CODES)[number];

/** Contract bounds shared with Python and Swift. */
export const SPEECH_CONTRACT_LIMITS = Object.freeze({
  /** Largest token index. */
  maxTokenIndex: 4096,
  /** Most token timings in one result. */
  maxTokenTimes: 4096,
  /** Most Unicode code points in one text. */
  maxTextScalars: 8192,
  /** Largest stream-relative millisecond (24 hours). */
  maxStreamMilliseconds: 86_400_000,
  /** Largest confidence, in parts per million. */
  maxConfidencePPM: 1_000_000,
  /** Largest partial revision or segment index (2^31 - 1). */
  maxSequence: 2_147_483_647,
  /** Most UTF-8 bytes in a language tag. */
  maxLanguageBytes: 63,
});

/** Whether word timings are present and bounded by the source audio. */
export const SPEECH_ALIGNMENT_STATUSES = Object.freeze([
  "unavailable",
  "aligned",
  "outsideWindow",
  "unverifiedTokenization",
] as const);

export type SpeechAlignmentStatus = (typeof SPEECH_ALIGNMENT_STATUSES)[number];

/**
 * A refused speech value. The message names the code and the offending field
 * path only, never transcript text or the rejected value.
 */
export class SpeechContractError extends Error {
  readonly code: SpeechContractErrorCode;
  readonly field: string;

  constructor(code: SpeechContractErrorCode, field: string) {
    super(`${code} at ${field}`);
    this.name = "SpeechContractError";
    this.code = code;
    this.field = field;
  }
}

const HYPOTHESIS: SpeechContractErrorCode = "speech_hypothesis_invalid";
const ARTIFACT: SpeechContractErrorCode = "speech_artifact_unverified";
const INSPECT = Symbol.for("nodejs.util.inspect.custom");
const PROVIDER_ID = /^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$/;
const PROVIDER_REVISION = /^[A-Za-z0-9](?:[A-Za-z0-9_.+-]{0,62}[A-Za-z0-9])?$/;
const MODEL_FINGERPRINT = /^sha256:[0-9a-f]{64}$/;
const LANGUAGE = /^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/;
const UTF8 = new TextEncoder();

function integer(
  value: unknown,
  minimum: number,
  maximum: number,
  field: string,
  code: SpeechContractErrorCode = HYPOTHESIS,
): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new SpeechContractError(code, field);
  }
  return value;
}

function protectedText(value: unknown, field: string): string {
  if (typeof value !== "string" || scalarCount(value) > SPEECH_CONTRACT_LIMITS.maxTextScalars) {
    throw new SpeechContractError(HYPOTHESIS, field);
  }
  return value;
}

function optionalConfidence(value: unknown, field: string): number | null {
  return value === undefined || value === null
    ? null
    : integer(value, 0, SPEECH_CONTRACT_LIMITS.maxConfidencePPM, field);
}

// Node's util.inspect (console.log) shows the redacted string, not fields.
function redactInspection(type: { prototype: { toString(): string } }): void {
  Object.defineProperty(type.prototype, INSPECT, {
    value(this: { toString(): string }) {
      return this.toString();
    },
  });
}

/** Number of Unicode code points in `text` (Swift `unicodeScalars.count`). */
export function scalarCount(text: string): number {
  let count = 0;
  for (const _ of text) {
    count += 1;
  }
  return count;
}

/** Construction input for {@link SpeechTokenTiming}. */
export interface SpeechTokenTimingInit {
  readonly tokenIndex: number;
  readonly startMilliseconds: number;
  readonly endMilliseconds: number;
  readonly confidencePPM: number;
  readonly text: string;
}

/**
 * Timing for one recognized token: `[startMilliseconds, endMilliseconds]` in
 * stream-relative milliseconds, with a decoder confidence in ppm.
 */
export class SpeechTokenTiming {
  readonly tokenIndex: number;
  readonly startMilliseconds: number;
  readonly endMilliseconds: number;
  readonly confidencePPM: number;
  /** Token length in Unicode code points. */
  readonly textLength: number;
  readonly #text: string;

  /**
   * @throws SpeechContractError `speech_hypothesis_invalid` when the index is
   *   outside 0-4096, a time is negative, after 24 hours or reversed, the
   *   confidence is outside 0-1,000,000 ppm or the text exceeds 8,192 code
   *   points.
   */
  constructor(init: SpeechTokenTimingInit) {
    const limits = SPEECH_CONTRACT_LIMITS;
    this.tokenIndex = integer(init?.tokenIndex, 0, limits.maxTokenIndex, "tokenIndex");
    this.startMilliseconds = integer(
      init.startMilliseconds,
      0,
      limits.maxStreamMilliseconds,
      "startMilliseconds",
    );
    this.endMilliseconds = integer(
      init.endMilliseconds,
      this.startMilliseconds,
      limits.maxStreamMilliseconds,
      "endMilliseconds",
    );
    this.confidencePPM = integer(
      init.confidencePPM,
      0,
      limits.maxConfidencePPM,
      "confidencePPM",
    );
    this.#text = protectedText(init.text, "text");
    this.textLength = scalarCount(this.#text);
    Object.freeze(this);
  }

  /** Protected token text. Route it only to a local review or privacy step. */
  get text(): string {
    return this.#text;
  }

  /** Return a copy moved later in the stream by `milliseconds`. */
  shifted(milliseconds: number): SpeechTokenTiming {
    const offset = integer(
      milliseconds,
      0,
      SPEECH_CONTRACT_LIMITS.maxStreamMilliseconds - this.endMilliseconds,
      "shift",
    );
    return new SpeechTokenTiming({
      tokenIndex: this.tokenIndex,
      startMilliseconds: this.startMilliseconds + offset,
      endMilliseconds: this.endMilliseconds + offset,
      confidencePPM: this.confidencePPM,
      text: this.#text,
    });
  }

  /** Value-free summary, matching Python's `TokenTime.to_dict()`. */
  toJSON() {
    return {
      token_index: this.tokenIndex,
      start_ms: this.startMilliseconds,
      end_ms: this.endMilliseconds,
      confidence_ppm: this.confidencePPM,
      token_length: this.textLength,
    };
  }

  toString(): string {
    return "SpeechTokenTiming(text=<private>)";
  }

  static {
    redactInspection(this);
  }
}

function tokenList(value: unknown, field: string): readonly SpeechTokenTiming[] {
  const tokens = value ?? [];
  if (!Array.isArray(tokens) || tokens.length > SPEECH_CONTRACT_LIMITS.maxTokenTimes) {
    throw new SpeechContractError(HYPOTHESIS, field);
  }
  tokens.forEach((token: unknown, index: number) => {
    if (!(token instanceof SpeechTokenTiming)) {
      throw new SpeechContractError(HYPOTHESIS, `${field}[${index}]`);
    }
    const previous = tokens[index - 1] as SpeechTokenTiming | undefined;
    if (
      previous !== undefined &&
      (previous.tokenIndex >= token.tokenIndex ||
        previous.startMilliseconds > token.startMilliseconds ||
        previous.endMilliseconds > token.endMilliseconds)
    ) {
      throw new SpeechContractError(HYPOTHESIS, `${field}[${index}]`);
    }
  });
  return Object.freeze([...(tokens as SpeechTokenTiming[])]);
}

function alignment(
  value: unknown,
  tokens: readonly SpeechTokenTiming[],
): SpeechAlignmentStatus {
  const status =
    value === undefined || value === null
      ? tokens.length === 0
        ? "unavailable"
        : "aligned"
      : value;
  if (
    typeof status !== "string" ||
    !(SPEECH_ALIGNMENT_STATUSES as readonly string[]).includes(status) ||
    (status === "aligned") !== (tokens.length > 0)
  ) {
    throw new SpeechContractError(HYPOTHESIS, "alignmentStatus");
  }
  return status as SpeechAlignmentStatus;
}

function withinInterval(
  tokens: readonly SpeechTokenTiming[],
  start: number,
  end: number,
  field: string,
): void {
  tokens.forEach((token, index) => {
    if (token.startMilliseconds < start || token.endMilliseconds > end) {
      throw new SpeechContractError(HYPOTHESIS, `${field}[${index}]`);
    }
  });
}

/** Construction input for {@link SpeechRecognition}. */
export interface SpeechRecognitionInit {
  readonly text: string;
  readonly tokenTimes?: readonly SpeechTokenTiming[];
  readonly decoderConfidencePPM?: number | null;
  /** Defaults to `aligned` with token timings and `unavailable` without. */
  readonly alignmentStatus?: SpeechAlignmentStatus;
}

/**
 * One protected decoder result. Token times are relative to the decoded
 * window; token indices strictly increase and start and end times never
 * decrease. Only an `aligned` result carries token timings.
 */
export class SpeechRecognition {
  readonly tokenTimes: readonly SpeechTokenTiming[];
  readonly decoderConfidencePPM: number | null;
  readonly alignmentStatus: SpeechAlignmentStatus;
  /** Text length in Unicode code points. */
  readonly textLength: number;
  readonly #text: string;

  /** @throws SpeechContractError `speech_hypothesis_invalid`. */
  constructor(init: SpeechRecognitionInit) {
    this.#text = protectedText(init?.text, "text");
    this.textLength = scalarCount(this.#text);
    this.tokenTimes = tokenList(init.tokenTimes, "tokenTimes");
    this.decoderConfidencePPM = optionalConfidence(
      init.decoderConfidencePPM,
      "decoderConfidencePPM",
    );
    this.alignmentStatus = alignment(init.alignmentStatus, this.tokenTimes);
    Object.freeze(this);
  }

  /** Protected recognized text. */
  get text(): string {
    return this.#text;
  }

  /** Value-free summary. */
  toJSON() {
    return {
      text_length: this.textLength,
      token_count: this.tokenTimes.length,
      decoder_confidence_ppm: this.decoderConfidencePPM,
      alignment_status: this.alignmentStatus,
    };
  }

  toString(): string {
    return "SpeechRecognition(text=<private>)";
  }

  static {
    redactInspection(this);
  }
}

/** Construction input for {@link SpeechPartialHypothesis}. */
export interface SpeechPartialHypothesisInit {
  readonly revision: number;
  readonly text: string;
  readonly tokenTimes?: readonly SpeechTokenTiming[];
  readonly windowStartMilliseconds: number;
  readonly windowEndMilliseconds: number;
  readonly decoderConfidencePPM?: number | null;
  readonly alignmentStatus?: SpeechAlignmentStatus;
}

/**
 * A protected, revisable partial hypothesis. The window interval locates the
 * decoded input; stream-relative token timings must fall inside it.
 */
export class SpeechPartialHypothesis {
  readonly revision: number;
  readonly isFinal = false;
  readonly tokenTimes: readonly SpeechTokenTiming[];
  readonly windowStartMilliseconds: number;
  readonly windowEndMilliseconds: number;
  readonly decoderConfidencePPM: number | null;
  readonly alignmentStatus: SpeechAlignmentStatus;
  readonly textLength: number;
  readonly #text: string;

  /** @throws SpeechContractError `speech_hypothesis_invalid`. */
  constructor(init: SpeechPartialHypothesisInit) {
    const limits = SPEECH_CONTRACT_LIMITS;
    this.revision = integer(init?.revision, 0, limits.maxSequence, "revision");
    this.#text = protectedText(init.text, "text");
    this.textLength = scalarCount(this.#text);
    this.tokenTimes = tokenList(init.tokenTimes, "tokenTimes");
    this.windowStartMilliseconds = integer(
      init.windowStartMilliseconds,
      0,
      limits.maxStreamMilliseconds,
      "windowStartMilliseconds",
    );
    this.windowEndMilliseconds = integer(
      init.windowEndMilliseconds,
      this.windowStartMilliseconds,
      limits.maxStreamMilliseconds,
      "windowEndMilliseconds",
    );
    withinInterval(
      this.tokenTimes,
      this.windowStartMilliseconds,
      this.windowEndMilliseconds,
      "tokenTimes",
    );
    this.decoderConfidencePPM = optionalConfidence(
      init.decoderConfidencePPM,
      "decoderConfidencePPM",
    );
    this.alignmentStatus = alignment(init.alignmentStatus, this.tokenTimes);
    Object.freeze(this);
  }

  get text(): string {
    return this.#text;
  }

  /** Value-free summary, matching Python's `PartialHypothesis.to_dict()`. */
  toJSON() {
    return {
      revision: this.revision,
      text_length: this.textLength,
      is_final: this.isFinal,
      token_count: this.tokenTimes.length,
      window_start_ms: this.windowStartMilliseconds,
      window_end_ms: this.windowEndMilliseconds,
      alignment_status: this.alignmentStatus,
    };
  }

  toString(): string {
    return "SpeechPartialHypothesis(text=<private>)";
  }

  static {
    redactInspection(this);
  }
}

/** Construction input for {@link SpeechFinalizedSegment}. */
export interface SpeechFinalizedSegmentInit {
  readonly segmentIndex: number;
  readonly text: string;
  readonly startMilliseconds: number;
  readonly endMilliseconds: number;
  readonly tokenTimes?: readonly SpeechTokenTiming[];
  readonly decoderConfidencePPM?: number | null;
  readonly alignmentStatus?: SpeechAlignmentStatus;
}

/**
 * A finalized acoustic segment `[startMilliseconds, endMilliseconds]`. Its
 * stream-relative token timings must fall inside it. It never finalizes a
 * clinical note or releases PHI.
 */
export class SpeechFinalizedSegment {
  readonly allowsAutomaticClinicalFinalization = false;
  readonly segmentIndex: number;
  readonly startMilliseconds: number;
  readonly endMilliseconds: number;
  readonly tokenTimes: readonly SpeechTokenTiming[];
  readonly decoderConfidencePPM: number | null;
  readonly alignmentStatus: SpeechAlignmentStatus;
  readonly textLength: number;
  readonly #text: string;

  /** @throws SpeechContractError `speech_hypothesis_invalid`. */
  constructor(init: SpeechFinalizedSegmentInit) {
    const limits = SPEECH_CONTRACT_LIMITS;
    this.segmentIndex = integer(init?.segmentIndex, 0, limits.maxSequence, "segmentIndex");
    this.#text = protectedText(init.text, "text");
    this.textLength = scalarCount(this.#text);
    this.startMilliseconds = integer(
      init.startMilliseconds,
      0,
      limits.maxStreamMilliseconds,
      "startMilliseconds",
    );
    this.endMilliseconds = integer(
      init.endMilliseconds,
      this.startMilliseconds,
      limits.maxStreamMilliseconds,
      "endMilliseconds",
    );
    this.tokenTimes = tokenList(init.tokenTimes, "tokenTimes");
    withinInterval(this.tokenTimes, this.startMilliseconds, this.endMilliseconds, "tokenTimes");
    this.decoderConfidencePPM = optionalConfidence(
      init.decoderConfidencePPM,
      "decoderConfidencePPM",
    );
    this.alignmentStatus = alignment(init.alignmentStatus, this.tokenTimes);
    Object.freeze(this);
  }

  get text(): string {
    return this.#text;
  }

  /** Segment duration in milliseconds. */
  get durationMilliseconds(): number {
    return this.endMilliseconds - this.startMilliseconds;
  }

  /** Value-free summary, matching Python's `FinalizedSegment.to_dict()`. */
  toJSON() {
    return {
      segment_index: this.segmentIndex,
      text_length: this.textLength,
      start_ms: this.startMilliseconds,
      end_ms: this.endMilliseconds,
      duration_ms: this.durationMilliseconds,
      token_count: this.tokenTimes.length,
      alignment_status: this.alignmentStatus,
    };
  }

  toString(): string {
    return "SpeechFinalizedSegment(text=<private>)";
  }

  static {
    redactInspection(this);
  }
}

/** Construction input for {@link SpeechEngineProvenance}. */
export interface SpeechEngineProvenanceInit {
  readonly providerId: string;
  readonly providerRevision: string;
  readonly modelFingerprint: string;
  readonly language: string;
}

/**
 * Pinned speech engine identity. It describes provenance, not source custody
 * or authorization.
 */
export class SpeechEngineProvenance {
  /** Lowercase identifier, 1-64 characters of `[a-z0-9_.-]`. */
  readonly providerId: string;
  /** Adapter revision, 1-64 characters of `[A-Za-z0-9_.+-]`. */
  readonly providerRevision: string;
  /** `sha256:` followed by 64 lowercase hex digits. */
  readonly modelFingerprint: string;
  /** Lowercase language tag such as `en` or `en-US`, at most 63 bytes. */
  readonly language: string;

  /**
   * @throws SpeechContractError `speech_artifact_unverified` naming the first
   *   field that does not match the shared grammar.
   */
  constructor(init: SpeechEngineProvenanceInit) {
    const matches = (value: unknown, pattern: RegExp, field: string): string => {
      if (typeof value !== "string" || !pattern.test(value)) {
        throw new SpeechContractError(ARTIFACT, field);
      }
      return value;
    };
    this.providerId = matches(init?.providerId, PROVIDER_ID, "providerId");
    this.providerRevision = matches(init.providerRevision, PROVIDER_REVISION, "providerRevision");
    this.modelFingerprint = matches(init.modelFingerprint, MODEL_FINGERPRINT, "modelFingerprint");
    const language = matches(init.language, LANGUAGE, "language");
    if (UTF8.encode(language).length > SPEECH_CONTRACT_LIMITS.maxLanguageBytes) {
      throw new SpeechContractError(ARTIFACT, "language");
    }
    this.language = language;
    Object.freeze(this);
  }
}
