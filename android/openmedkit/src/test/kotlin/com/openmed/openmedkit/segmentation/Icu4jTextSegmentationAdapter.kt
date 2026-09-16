package com.openmed.openmedkit.segmentation

import com.ibm.icu.text.BreakIterator
import java.util.Locale

/**
 * JVM test adapter backed by the same ICU data model used by Android ICU.
 */
internal class Icu4jTextSegmentationAdapter : TextSegmentationAdapter {
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
        return (boundaries + listOf(0, text.length))
            .distinct()
            .sorted()
            .toIntArray()
    }
}
