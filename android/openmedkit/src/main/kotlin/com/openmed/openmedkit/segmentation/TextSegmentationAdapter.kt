package com.openmed.openmedkit.segmentation

import android.icu.text.BreakIterator
import com.openmed.openmedkit.ScalarSpan
import com.openmed.openmedkit.UnicodeOffsetContract
import java.util.Locale

/**
 * Platform boundary-iterator seam for local text segmentation.
 *
 * Adapter boundaries use UTF-16 because both Android ICU and ICU4J expose
 * native JVM indices. [IcuTextSegmenter] converts every public result to the
 * shared Unicode scalar offset contract.
 */
interface TextSegmentationAdapter {
    fun graphemeBoundaries(
        text: String,
        locale: Locale = Locale.ROOT,
    ): IntArray

    fun wordBoundaries(
        text: String,
        locale: Locale,
    ): IntArray
}

/**
 * On-device boundary adapter backed by the ICU runtime included with Android.
 */
class AndroidIcuTextSegmentationAdapter : TextSegmentationAdapter {
    override fun graphemeBoundaries(
        text: String,
        locale: Locale,
    ): IntArray = collectBoundaries(
        BreakIterator.getCharacterInstance(locale),
        text,
    )

    override fun wordBoundaries(
        text: String,
        locale: Locale,
    ): IntArray = collectBoundaries(
        BreakIterator.getWordInstance(locale),
        text,
    )

    private fun collectBoundaries(
        iterator: BreakIterator,
        text: String,
    ): IntArray {
        iterator.setText(text)
        val boundaries = mutableListOf<Int>()
        var boundary = iterator.first()
        while (boundary != BreakIterator.DONE) {
            boundaries += boundary
            boundary = iterator.next()
        }
        return normalizedBoundaries(boundaries, text.length)
    }
}

/**
 * One ICU-produced text segment with Unicode scalar offsets.
 */
data class ScalarTextSegment(
    val text: String,
    val start: Int,
    val end: Int,
) {
    init {
        require(start >= 0) { "start must be non-negative" }
        require(end > start) { "end must be greater than start" }
    }
}

/**
 * ICU word and grapheme fallback for no-whitespace Android text paths.
 *
 * No application dictionary is bundled. Android provides the production ICU
 * data, while JVM tests inject an ICU4J implementation through
 * [TextSegmentationAdapter].
 */
class IcuTextSegmenter(
    private val adapter: TextSegmentationAdapter = AndroidIcuTextSegmentationAdapter(),
) {
    /**
     * Snap a scalar span outward to complete extended grapheme clusters.
     *
     * Empty spans remain empty and move to the preceding cluster boundary.
     */
    fun snapScalarSpan(
        text: String,
        start: Int,
        end: Int,
        locale: Locale = localeForText(text),
    ): ScalarSpan {
        val scalarLength = UnicodeOffsetContract.scalarLength(text)
        val safeStart = start.coerceIn(0, scalarLength)
        val safeEnd = end.coerceIn(safeStart, scalarLength)
        val utf16Start = UnicodeOffsetContract.scalarToUtf16Index(text, safeStart)
        val utf16End = UnicodeOffsetContract.scalarToUtf16Index(text, safeEnd)
        val boundaries = checkedBoundaries(
            adapter.graphemeBoundaries(text, locale),
            text,
        )

        val snappedUtf16Start = boundaries.last { it <= utf16Start }
        if (safeStart == safeEnd) {
            val snapped = UnicodeOffsetContract.utf16ToScalarOffset(
                text,
                snappedUtf16Start,
            )
            return ScalarSpan(snapped, snapped)
        }

        val snappedUtf16End = boundaries.first { it >= utf16End }
        return ScalarSpan(
            start = UnicodeOffsetContract.utf16ToScalarOffset(text, snappedUtf16Start),
            end = UnicodeOffsetContract.utf16ToScalarOffset(text, snappedUtf16End),
        )
    }

    /** Return whether [scalarOffset] is an extended-grapheme boundary. */
    fun isGraphemeBoundary(
        text: String,
        scalarOffset: Int,
        locale: Locale = localeForText(text),
    ): Boolean {
        if (scalarOffset !in 0..UnicodeOffsetContract.scalarLength(text)) {
            return false
        }
        val utf16Index = UnicodeOffsetContract.scalarToUtf16Index(text, scalarOffset)
        return checkedBoundaries(
            adapter.graphemeBoundaries(text, locale),
            text,
        ).binarySearch(utf16Index) >= 0
    }

    /** Return every extended grapheme cluster with scalar offsets. */
    fun graphemeSegments(
        text: String,
        locale: Locale = localeForText(text),
    ): List<ScalarTextSegment> = segmentsFromBoundaries(
        text,
        checkedBoundaries(adapter.graphemeBoundaries(text, locale), text),
    )

    /** Return word-like ICU segments, excluding whitespace and punctuation. */
    fun wordSegments(
        text: String,
        locale: Locale = localeForText(text),
    ): List<ScalarTextSegment> = segmentsFromBoundaries(
        text,
        checkedBoundaries(adapter.wordBoundaries(text, locale), text),
    ).filter { isWordLike(it.text) }

    /**
     * Segment zh/hi/ta paths without an application-bundled dictionary.
     *
     * If a platform word iterator cannot split no-whitespace text, complete
     * grapheme clusters become the deterministic last-resort units.
     */
    fun fallbackWordSegments(
        text: String,
        locale: Locale = localeForText(text),
    ): List<ScalarTextSegment> {
        if (text.isEmpty()) {
            return emptyList()
        }
        val words = wordSegments(text, locale)
        val hasNoWhitespace = text.codePoints().noneMatch { Character.isWhitespace(it) }
        if (!hasNoWhitespace || words.size > 1 || UnicodeOffsetContract.scalarLength(text) <= 1) {
            return words
        }
        return graphemeSegments(text, locale).filter { isWordLike(it.text) }
    }

    private fun segmentsFromBoundaries(
        text: String,
        boundaries: IntArray,
    ): List<ScalarTextSegment> {
        if (text.isEmpty()) {
            return emptyList()
        }
        return boundaries.toList().zipWithNext().mapNotNull { (utf16Start, utf16End) ->
            if (utf16End <= utf16Start) {
                return@mapNotNull null
            }
            ScalarTextSegment(
                text = text.substring(utf16Start, utf16End),
                start = UnicodeOffsetContract.utf16ToScalarOffset(text, utf16Start),
                end = UnicodeOffsetContract.utf16ToScalarOffset(text, utf16End),
            )
        }
    }

    private fun checkedBoundaries(
        boundaries: IntArray,
        text: String,
    ): IntArray {
        val normalized = normalizedBoundaries(boundaries.asIterable(), text.length)
            .filterNot { isIndicConjunctBoundary(text, it) }
            .toIntArray()
        require(normalized.all { UnicodeOffsetContract.isCodePointBoundary(text, it) }) {
            "text segmentation adapter returned a boundary inside a surrogate pair"
        }
        return normalized
    }

    companion object {
        private val HINDI = Locale.forLanguageTag("hi")
        private val TAMIL = Locale.forLanguageTag("ta")

        /** Select the ICU locale for the supported no-whitespace script paths. */
        @JvmStatic
        fun localeForText(text: String): Locale {
            for (codePoint in text.codePoints().toArray()) {
                when (Character.UnicodeScript.of(codePoint)) {
                    Character.UnicodeScript.HAN -> return Locale.SIMPLIFIED_CHINESE
                    Character.UnicodeScript.DEVANAGARI -> return HINDI
                    Character.UnicodeScript.TAMIL -> return TAMIL
                    else -> Unit
                }
            }
            return Locale.ROOT
        }

        /** Return whether text should use the ICU zh/hi/ta fallback path. */
        @JvmStatic
        fun requiresFallback(text: String): Boolean {
            for (codePoint in text.codePoints().toArray()) {
                if (
                    Character.UnicodeScript.of(codePoint) in setOf(
                        Character.UnicodeScript.HAN,
                        Character.UnicodeScript.DEVANAGARI,
                        Character.UnicodeScript.TAMIL,
                    )
                ) {
                    return true
                }
            }
            return false
        }
    }
}

private fun normalizedBoundaries(
    boundaries: Iterable<Int>,
    textLength: Int,
): IntArray {
    return (boundaries + listOf(0, textLength))
        .filter { it in 0..textLength }
        .distinct()
        .sorted()
        .toIntArray()
}

private fun isWordLike(text: String): Boolean {
    return text.codePoints().anyMatch { codePoint ->
        when (Character.getType(codePoint)) {
            Character.UPPERCASE_LETTER.toInt(),
            Character.LOWERCASE_LETTER.toInt(),
            Character.TITLECASE_LETTER.toInt(),
            Character.MODIFIER_LETTER.toInt(),
            Character.OTHER_LETTER.toInt(),
            Character.NON_SPACING_MARK.toInt(),
            Character.COMBINING_SPACING_MARK.toInt(),
            Character.ENCLOSING_MARK.toInt(),
            Character.DECIMAL_DIGIT_NUMBER.toInt(),
            Character.LETTER_NUMBER.toInt(),
            Character.OTHER_NUMBER.toInt(),
            -> true

            else -> false
        }
    }
}

/**
 * Suppress the platform-version-dependent break before an Indic consonant
 * following virama plus an optional joiner. This is the narrow tailoring
 * required by OpenMed's shared grapheme contract for the nine fixture scripts.
 */
private fun isIndicConjunctBoundary(
    text: String,
    utf16Index: Int,
): Boolean {
    if (utf16Index <= 0 || utf16Index >= text.length) {
        return false
    }
    val following = text.codePointAt(utf16Index)
    if (!Character.isLetter(following)) {
        return false
    }

    var cursor = utf16Index
    var preceding = text.codePointBefore(cursor)
    if (preceding == ZERO_WIDTH_JOINER || preceding == ZERO_WIDTH_NON_JOINER) {
        cursor -= Character.charCount(preceding)
        if (cursor <= 0) {
            return false
        }
        preceding = text.codePointBefore(cursor)
    }
    return preceding in INDIC_VIRAMAS
}

private const val ZERO_WIDTH_NON_JOINER = 0x200C
private const val ZERO_WIDTH_JOINER = 0x200D

private val INDIC_VIRAMAS = setOf(
    0x094D, // Devanagari
    0x09CD, // Bengali
    0x0A4D, // Gurmukhi
    0x0ACD, // Gujarati
    0x0B4D, // Oriya
    0x0BCD, // Tamil
    0x0C4D, // Telugu
    0x0CCD, // Kannada
    0x0D4D, // Malayalam
)
