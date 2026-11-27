package com.openmed.openmedkit.speech

/**
 * Controlled speech-contract failures, matching Swift `StreamingSpeechError` raw
 * values for the data contract.
 */
enum class SpeechContractError(val code: String) {
    /** A token timing or recognition result breaks the shared bounds or order. */
    INVALID_HYPOTHESIS("speech_hypothesis_invalid"),

    /** Engine provenance does not match the pinned identity grammar. */
    UNVERIFIED_ARTIFACT("speech_artifact_unverified"),
}

/**
 * A value-free speech-contract failure.
 *
 * [error] is the Swift `StreamingSpeechError` code and [category] the Python
 * `SpeechProviderError.category`, such as `token_start_ms_out_of_range`. The
 * message holds only those two codes: never transcript or token text, paths,
 * identifiers or engine errors.
 */
class SpeechContractException internal constructor(
    val error: SpeechContractError,
    val category: String,
) : IllegalArgumentException("${error.code}: $category")

/** Whether word alignment is available and bounded by the actual source audio. */
enum class SpeechAlignmentStatus(val code: String) {
    UNAVAILABLE("unavailable"),
    ALIGNED("aligned"),
    OUTSIDE_WINDOW("outsideWindow"),
    UNVERIFIED_TOKENIZATION("unverifiedTokenization"),
}

private const val MAX_TOKEN_INDEX = 4096
private const val MAX_TOKEN_TIMES = 4096
private const val MAX_STREAM_MILLISECONDS = 86_400_000L
private const val MAX_CONFIDENCE_PPM = 1_000_000
private const val MAX_TEXT_SCALARS = 8192
private val PROVIDER_ID = Regex("[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?")
private val PROVIDER_REVISION = Regex("[A-Za-z0-9](?:[A-Za-z0-9_.+-]{0,62}[A-Za-z0-9])?")
private val MODEL_FINGERPRINT = Regex("sha256:[0-9a-f]{64}")
private val LANGUAGE = Regex("[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")

private fun ensure(condition: Boolean, error: SpeechContractError, category: String) {
    if (!condition) throw SpeechContractException(error, category)
}

private fun hypothesis(condition: Boolean, category: String) =
    ensure(condition, SpeechContractError.INVALID_HYPOTHESIS, category)

/** Unicode scalar count, as Python `len` and Swift `unicodeScalars.count`. */
private fun scalars(text: String): Int = text.codePointCount(0, text.length)

/**
 * Shared speech-contract timing for one token.
 *
 * Times are milliseconds relative to the recognized window and confidence is in
 * parts per million. These are decoder estimates, not clinical confidence. The
 * token text is protected and never appears in [toString] or in errors.
 *
 * @throws SpeechContractException with [SpeechContractError.INVALID_HYPOTHESIS]
 *   when the index, either time, the time order, the confidence or the text
 *   length is out of range.
 */
class SpeechTokenTiming(
    val tokenIndex: Int,
    val startMilliseconds: Long,
    val endMilliseconds: Long,
    val confidencePpm: Int,
    val text: String,
) {
    init {
        hypothesis(tokenIndex in 0..MAX_TOKEN_INDEX, "token_index_out_of_range")
        hypothesis(startMilliseconds in 0..MAX_STREAM_MILLISECONDS, "token_start_ms_out_of_range")
        hypothesis(endMilliseconds in 0..MAX_STREAM_MILLISECONDS, "token_end_ms_out_of_range")
        hypothesis(endMilliseconds >= startMilliseconds, "token_time_range_invalid")
        hypothesis(confidencePpm in 0..MAX_CONFIDENCE_PPM, "token_confidence_ppm_out_of_range")
        hypothesis(scalars(text) <= MAX_TEXT_SCALARS, "token_text_invalid")
    }

    override fun toString(): String = "SpeechTokenTiming(text=<private>)"
}

/**
 * One protected decoder result. Token times are relative to the supplied window.
 *
 * Tokens must be in strictly increasing index order with non-decreasing start
 * and end times. [alignmentStatus] defaults to [SpeechAlignmentStatus.ALIGNED]
 * when there are tokens and [SpeechAlignmentStatus.UNAVAILABLE] otherwise, and
 * only an aligned result may carry tokens. The text never appears in [toString]
 * or in errors.
 *
 * @throws SpeechContractException with [SpeechContractError.INVALID_HYPOTHESIS].
 */
class SpeechRecognition(
    val text: String,
    tokenTimes: List<SpeechTokenTiming> = emptyList(),
    /** Optional whole-result decoder estimate, in parts per million. */
    val decoderConfidencePpm: Int? = null,
    alignmentStatus: SpeechAlignmentStatus? = null,
) {
    /** Token timings, in token order. */
    val tokenTimes: List<SpeechTokenTiming> = tokenTimes.toList()

    /** Whether the token timings are aligned to the source audio. */
    val alignmentStatus: SpeechAlignmentStatus = alignmentStatus
        ?: if (this.tokenTimes.isEmpty()) SpeechAlignmentStatus.UNAVAILABLE else SpeechAlignmentStatus.ALIGNED

    init {
        hypothesis(scalars(text) <= MAX_TEXT_SCALARS, "hypothesis_text_invalid")
        hypothesis(this.tokenTimes.size <= MAX_TOKEN_TIMES, "token_times_out_of_range")
        hypothesis(
            decoderConfidencePpm == null || decoderConfidencePpm in 0..MAX_CONFIDENCE_PPM,
            "decoder_confidence_ppm_out_of_range",
        )
        this.tokenTimes.zipWithNext { previous, token ->
            hypothesis(
                previous.tokenIndex < token.tokenIndex &&
                    previous.startMilliseconds <= token.startMilliseconds &&
                    previous.endMilliseconds <= token.endMilliseconds,
                "token_times_unsorted",
            )
        }
        hypothesis(
            (this.alignmentStatus == SpeechAlignmentStatus.ALIGNED) == this.tokenTimes.isNotEmpty(),
            "alignment_status_invalid",
        )
    }

    override fun toString(): String = "SpeechRecognition(text=<private>)"
}

/**
 * Pinned identity of one local speech engine.
 *
 * It describes provenance, not source custody or authorization, and carries no
 * audio, path or network endpoint.
 *
 * @param providerId Lowercase bounded identifier.
 * @param providerRevision Bounded revision label of the adapter build.
 * @param modelFingerprint `sha256:` and 64 lowercase hex digits of the weights.
 * @param language Lowercase BCP 47-style tag of at most 63 characters.
 * @throws SpeechContractException with [SpeechContractError.UNVERIFIED_ARTIFACT].
 */
class SpeechEngineProvenance(
    val providerId: String,
    val providerRevision: String,
    val modelFingerprint: String,
    val language: String,
) {
    init {
        val error = SpeechContractError.UNVERIFIED_ARTIFACT
        ensure(PROVIDER_ID.matches(providerId), error, "provider_id_invalid")
        ensure(PROVIDER_REVISION.matches(providerRevision), error, "provider_revision_invalid")
        ensure(MODEL_FINGERPRINT.matches(modelFingerprint), error, "provider_model_fingerprint_invalid")
        ensure(language.length <= 63 && LANGUAGE.matches(language), error, "provider_language_invalid")
    }

    override fun toString(): String =
        "SpeechEngineProvenance(providerId=$providerId, providerRevision=$providerRevision, language=$language)"
}
