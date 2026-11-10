package com.openmed.openmedkit

/**
 * A half-open span measured in Unicode scalar (code point) offsets.
 */
data class ScalarSpan(
    val start: Int,
    val end: Int,
) {
    init {
        require(start >= 0) { "start must be non-negative" }
        require(end >= start) { "end must be greater than or equal to start" }
    }
}

/**
 * A half-open span measured in Kotlin/JVM UTF-16 code-unit indices.
 */
data class Utf16Span(
    val start: Int,
    val end: Int,
) {
    init {
        require(start >= 0) { "start must be non-negative" }
        require(end >= start) { "end must be greater than or equal to start" }
    }
}

/**
 * Conversion helpers for OpenMed's cross-runtime Unicode scalar offset contract.
 *
 * Public entity coordinates never use Kotlin's native UTF-16 indices. Callers
 * convert only at the point where a JVM substring or replacement API requires
 * UTF-16.
 */
object UnicodeOffsetContract {
    /** Return the number of Unicode scalar values in [text]. */
    @JvmStatic
    fun scalarLength(text: String): Int = text.codePointCount(0, text.length)

    /**
     * Convert a Unicode scalar offset to a Kotlin UTF-16 index.
     */
    @JvmStatic
    fun scalarToUtf16Index(
        text: String,
        scalarOffset: Int,
    ): Int {
        val scalarLength = scalarLength(text)
        require(scalarOffset in 0..scalarLength) {
            "scalarOffset must be between 0 and $scalarLength"
        }
        return text.offsetByCodePoints(0, scalarOffset)
    }

    /**
     * Convert a Kotlin UTF-16 index to a Unicode scalar offset.
     *
     * An index between a surrogate pair is rejected because it cannot
     * represent a valid OpenMed entity boundary.
     */
    @JvmStatic
    fun utf16ToScalarOffset(
        text: String,
        utf16Index: Int,
    ): Int {
        require(utf16Index in 0..text.length) {
            "utf16Index must be between 0 and ${text.length}"
        }
        require(isCodePointBoundary(text, utf16Index)) {
            "utf16Index must not split a surrogate pair"
        }
        return text.codePointCount(0, utf16Index)
    }

    /** Return the native UTF-16 range for a scalar span. */
    @JvmStatic
    fun utf16Span(
        text: String,
        start: Int,
        end: Int,
    ): Utf16Span {
        require(end >= start) { "end must be greater than or equal to start" }
        return Utf16Span(
            start = scalarToUtf16Index(text, start),
            end = scalarToUtf16Index(text, end),
        )
    }

    /** Slice [text] using Unicode scalar offsets. */
    @JvmStatic
    fun substring(
        text: String,
        start: Int,
        end: Int,
    ): String {
        val range = utf16Span(text, start, end)
        return text.substring(range.start, range.end)
    }

    /** Replace a Unicode scalar span without exposing UTF-16 indices. */
    @JvmStatic
    fun replaceScalarSpan(
        text: String,
        start: Int,
        end: Int,
        replacement: String,
    ): String {
        val range = utf16Span(text, start, end)
        return text.replaceRange(range.start, range.end, replacement)
    }

    /** Return whether [utf16Index] is outside the middle of a surrogate pair. */
    @JvmStatic
    fun isCodePointBoundary(
        text: String,
        utf16Index: Int,
    ): Boolean {
        if (utf16Index !in 0..text.length) {
            return false
        }
        if (utf16Index == 0 || utf16Index == text.length) {
            return true
        }
        return !(
            Character.isHighSurrogate(text[utf16Index - 1]) &&
                Character.isLowSurrogate(text[utf16Index])
            )
    }
}
