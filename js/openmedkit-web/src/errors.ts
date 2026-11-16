// Coded errors for the web package, keyed by the portable OpenMed error-code
// registry (openmed/core/error_codes.json; docs/api/errors.md).
import {
  OPENMED_ERROR_CODES,
  OPENMED_ERROR_CODE_RETRYABLE,
  type OpenMedErrorCode,
} from "./error-codes";

const PORTABLE_CODES: ReadonlySet<string> = new Set(Object.values(OPENMED_ERROR_CODES));

export interface OpenMedErrorOptions {
  /**
   * A finer, value-free reason code, kept by errors that had one before the
   * registry existed (for example `profile_unknown`). Defaults to `errorCode`.
   */
  code?: string;
  /** The underlying error. It is never part of the message. */
  cause?: unknown;
}

/**
 * Base class of the web package's coded errors.
 *
 * `errorCode` is always the portable registry code that Python, Swift and
 * Kotlin report for the same failure. `code` is the most specific stable code:
 * the registry code itself, or the reason code an older error class already
 * exposed (`PortablePolicyError`, `OpenMedOfflineError`), which keeps working.
 * Messages are value-free: they never include input text, secrets, paths or
 * URLs. The `name` stays `"Error"` for errors that used to be plain `Error`s,
 * so `String(error)` output is unchanged; match on `instanceof OpenMedError`,
 * `errorCode` or `code` instead of the message.
 */
export class OpenMedError extends Error {
  /** Portable registry code, for example `configuration_error`. */
  readonly errorCode: OpenMedErrorCode;
  /** Most specific stable code; equals `errorCode` unless a reason is kept. */
  readonly code: string;

  constructor(errorCode: OpenMedErrorCode, message: string, options: OpenMedErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.errorCode = PORTABLE_CODES.has(errorCode) ? errorCode : "internal_error";
    this.code = options.code ?? this.errorCode;
  }

  /** Whether retrying the unchanged request may succeed, per the registry. */
  get retryable(): boolean {
    return OPENMED_ERROR_CODE_RETRYABLE[this.errorCode];
  }
}

/** Return the portable registry code of an OpenMed error, or `null`. */
export function openMedErrorCode(error: unknown): OpenMedErrorCode | null {
  return error instanceof OpenMedError ? error.errorCode : null;
}
