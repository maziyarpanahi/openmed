package org.openmed.r8smoke

/** Outcome of one smoke check. */
enum class SmokeStatus {
    PASS,
    FAIL,
    SKIPPED,
}

/** A half-open span reported by offset only. */
data class SmokeSpan(val label: String, val start: Int, val end: Int) {
    init {
        require(label.matches(SmokeCheck.LABEL_PATTERN)) { "label is not value-free" }
        require(start in 0..end) { "span offsets must satisfy 0 <= start <= end" }
    }
}

/**
 * One value-free smoke check.
 *
 * A check carries an identifier, a status, an optional count, labelled offsets
 * and a reason code. It has no field for free text, so no input, model output
 * or exception message can reach the report.
 */
class SmokeCheck(
    val id: String,
    val status: SmokeStatus,
    val count: Int? = null,
    val spans: List<SmokeSpan> = emptyList(),
    val reason: String? = null,
) {
    init {
        require(id.matches(ID_PATTERN)) { "check id is not value-free" }
        require(count == null || count >= 0) { "count must be non-negative" }
        require(reason == null || reason.matches(REASON_PATTERN)) {
            "reason is not a value-free code"
        }
    }

    /** Return the report line for this check. */
    fun format(): String = buildString {
        append(LINE_PREFIX)
        append(" check=").append(id)
        append(" status=").append(status.name)
        count?.let { append(" count=").append(it) }
        if (spans.isNotEmpty()) {
            append(" spans=")
            append(spans.joinToString(",") { "${it.label}:${it.start}-${it.end}" })
        }
        reason?.let { append(" reason=").append(it) }
    }

    companion object {
        internal const val LINE_PREFIX = "openmed_r8_smoke"
        internal val ID_PATTERN = Regex("[a-z][a-z0-9_]{0,63}")
        internal val LABEL_PATTERN = Regex("[A-Za-z0-9_.:-]{1,64}")
        internal val REASON_PATTERN = Regex("[A-Za-z0-9_.:$-]{1,128}")

        /**
         * Return a failed check for [error], keeping only its class name.
         *
         * Exception messages can carry paths or text, so they are dropped.
         * Under R8 the class name is the obfuscated one; `mapping.txt`
         * translates it back.
         */
        fun failure(id: String, error: Throwable): SmokeCheck = SmokeCheck(
            id = id,
            status = SmokeStatus.FAIL,
            reason = "error:" + error.javaClass.name.filter {
                it.isLetterOrDigit() || it in "_.$"
            }.take(96).ifEmpty { "unknown" },
        )
    }
}

/** Value-free result of one smoke run in one build flavor. */
class SmokeReport(val flavor: String, val checks: List<SmokeCheck>) {
    init {
        require(flavor.matches(SmokeCheck.ID_PATTERN)) { "flavor is not value-free" }
    }

    /** True when at least one check ran and none failed. */
    val passed: Boolean
        get() = checks.isNotEmpty() && checks.none { it.status == SmokeStatus.FAIL }

    /** Return one line per check followed by the result line. */
    fun lines(): List<String> = checks.map(SmokeCheck::format) + resultLine()

    /** Return the summary line the emulator lane matches on. */
    fun resultLine(): String {
        val failed = checks.count { it.status == SmokeStatus.FAIL }
        val result = if (passed) "PASS" else "FAIL"
        return "${SmokeCheck.LINE_PREFIX} flavor=$flavor result=$result " +
            "checks=${checks.size} failed=$failed"
    }

    companion object {
        /** Logcat tag of every report line. */
        const val LOG_TAG = "OpenMedR8Smoke"
    }
}
