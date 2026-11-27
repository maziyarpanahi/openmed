package com.openmed.openmedkit

/** What a caller may do after an [OpenMedRuntimeException]. */
enum class OpenMedRetryGuidance {
    /** Retry immediately; the runtime is usable and the failure was transient. */
    RETRY_NOW,

    /** Retry after the host process returns to the foreground. */
    RETRY_AFTER_FOREGROUND,

    /** Retry after a back-off; see [OpenMedRuntimeException.retryAfterMillis]. */
    RETRY_AFTER_BACKOFF,

    /** Do not retry on this runtime; replace the model or create a new runtime. */
    RECREATE_RUNTIME,

    /** Load a model first, then retry. */
    LOAD_MODEL,
}

/**
 * Stable, value-free reasons an OpenMedKit lifecycle operation did not complete.
 *
 * [code] is a stable wire identifier shared by bridges. Neither the code nor
 * the exception message ever contains source text, span surfaces, file paths,
 * secrets, or identity-bearing values.
 */
enum class OpenMedRuntimeFailureReason(
    val code: String,
    val retryGuidance: OpenMedRetryGuidance,
) {
    /** The runtime or facade was closed before or during the operation. */
    RUNTIME_CLOSED("runtime_closed", OpenMedRetryGuidance.RECREATE_RUNTIME),

    /** The model was replaced while the operation ran; its result was discarded. */
    STALE_GENERATION("stale_generation", OpenMedRetryGuidance.RETRY_NOW),

    /** The host is in the background and the policy refuses background work. */
    ADMISSION_PAUSED_BACKGROUND(
        "admission_paused_background",
        OpenMedRetryGuidance.RETRY_AFTER_FOREGROUND,
    ),

    /** Memory pressure stopped admission; active work was preserved. */
    ADMISSION_PAUSED_MEMORY_PRESSURE(
        "admission_paused_memory_pressure",
        OpenMedRetryGuidance.RETRY_AFTER_BACKOFF,
    ),

    /** Opening or re-verifying the session failed. */
    SESSION_LOAD_FAILED("session_load_failed", OpenMedRetryGuidance.RETRY_AFTER_BACKOFF),

    /** Consecutive load failures reached the policy bound. */
    SESSION_RELOAD_LIMIT("session_reload_limit", OpenMedRetryGuidance.RECREATE_RUNTIME),

    /** A bridge host received work before any model was loaded. */
    MODEL_NOT_LOADED("model_not_loaded", OpenMedRetryGuidance.LOAD_MODEL),
}

/**
 * Typed lifecycle failure raised by [OpenMedRuntime] and [OpenMedKit].
 *
 * Cancellation is never reported with this type: a cancelled caller receives
 * a normal coroutine `CancellationException`. The message holds only the
 * stable reason code. Underlying load failures are not attached as a cause,
 * because their messages may contain artifact paths; [causeKind] carries the
 * failing type's simple class name instead.
 *
 * @property reason Stable reason with retry guidance.
 * @property retryAfterMillis Suggested back-off for [OpenMedRetryGuidance.RETRY_AFTER_BACKOFF].
 * @property causeKind Simple class name of an underlying load failure, if any.
 */
class OpenMedRuntimeException(
    val reason: OpenMedRuntimeFailureReason,
    val retryAfterMillis: Long? = null,
    val causeKind: String? = null,
) : RuntimeException("OpenMedKit runtime: ${reason.code}") {
    /** What the caller may do next. */
    val retryGuidance: OpenMedRetryGuidance
        get() = reason.retryGuidance

    /** Stable wire code for bridges, equal to [OpenMedRuntimeFailureReason.code]. */
    val code: String
        get() = reason.code
}
