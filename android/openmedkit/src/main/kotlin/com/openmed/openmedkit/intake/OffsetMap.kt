package com.openmed.openmedkit.intake

import com.openmed.openmedkit.ocr.OcrBoundingBox

/**
 * Source of a document-level character offset.
 */
enum class OffsetSource {
    OCR_TEXT,
    PAGE_SEPARATOR,
}

/**
 * Per-character source metadata for a [DocumentIntakeResult].
 */
data class OffsetMapEntry(
    val documentOffset: Int,
    val page: Int,
    val pageOffset: Int,
    val tokenStart: Int? = null,
    val tokenEnd: Int? = null,
    val bbox: OcrBoundingBox? = null,
    val source: OffsetSource = OffsetSource.OCR_TEXT,
) {
    init {
        require(documentOffset >= 0) { "documentOffset must be non-negative" }
        require(page >= 0) { "page must be non-negative" }
        require(pageOffset >= 0) { "pageOffset must be non-negative" }
        require((tokenStart == null) == (tokenEnd == null)) {
            "tokenStart and tokenEnd must either both be set or both be null"
        }
        if (tokenStart != null && tokenEnd != null) {
            require(tokenStart >= 0) { "tokenStart must be non-negative" }
            require(tokenEnd >= tokenStart) {
                "tokenEnd must be greater than or equal to tokenStart"
            }
        }
    }
}

/**
 * Continuous map from document text offsets back to page-local OCR positions.
 */
data class OffsetMap(
    val entries: List<OffsetMapEntry>,
) {
    val size: Int
        get() = entries.size

    init {
        entries.forEachIndexed { index, entry ->
            require(entry.documentOffset == index) {
                "offset map entries must be continuous and ordered"
            }
        }
    }

    operator fun get(offset: Int): OffsetMapEntry = entries[offset]

    fun entriesForSpan(start: Int, end: Int): List<OffsetMapEntry> {
        require(start >= 0) { "start must be non-negative" }
        require(end >= start) { "end must be greater than or equal to start" }
        require(end <= entries.size) { "end must not exceed text length" }
        return entries.subList(start, end)
    }

    fun pagesForSpan(start: Int, end: Int): Set<Int> =
        entriesForSpan(start, end)
            .map { it.page }
            .toSet()
}
