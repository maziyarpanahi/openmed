package com.openmed.openmedkit.deid

/**
 * Android-side OpenMed span contract used by the de-identification engine.
 *
 * The span intentionally stores offsets, labels, hashes, and replacements only.
 * It never stores the raw identifier text.
 */
data class OpenMedSpan(
    val start: Int,
    val end: Int,
    val canonicalLabel: String,
    val score: Double? = null,
    val textHash: String? = null,
    val action: DeidentifyMethod? = null,
    val replacement: String? = null,
) {
    init {
        require(start >= 0) { "start must be non-negative" }
        require(end >= start) { "end must be greater than or equal to start" }
        require(canonicalLabel.isNotBlank()) { "canonicalLabel must be non-blank" }
        if (score != null) {
            require(score in 0.0..1.0) { "score must be between 0.0 and 1.0" }
        }
    }
}

/**
 * One applied rewrite action.
 *
 * [span] keeps original document offsets. [outputStart] and [outputEnd] point
 * at the replacement in the redacted text.
 */
data class DeidentifyAction(
    val span: OpenMedSpan,
    val method: DeidentifyMethod,
    val replacement: String,
    val outputStart: Int,
    val outputEnd: Int,
)

/**
 * Result of applying de-identification actions to a document.
 */
data class DeidentifyResult(
    val redactedText: String,
    val actions: List<DeidentifyAction>,
) {
    val appliedSpans: List<OpenMedSpan>
        get() = actions.map { it.span }
}
