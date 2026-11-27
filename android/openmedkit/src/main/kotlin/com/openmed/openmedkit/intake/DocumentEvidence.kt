package com.openmed.openmedkit.intake

import com.openmed.openmedkit.ScalarSpan
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.ocr.OcrResult
import java.security.MessageDigest

/**
 * Document-level intake result shared with Python and Swift OpenMedKit.
 */
enum class DocumentIntakeOutcome(val code: String) {
    COMPLETE("complete"),
    PARTIAL("partial"),
    REFUSED("refused"),
}

/**
 * Closed, value-free intake reasons, in report order.
 */
enum class DocumentIntakeReason(val code: String) {
    UNSUPPORTED_DOCUMENT("unsupported_document"),
    DOCUMENT_LOCKED("document_locked"),
    PERMISSION_DENIED("permission_denied"),
    SIZE_LIMIT("size_limit"),
    EMPTY_DOCUMENT("empty_document"),
    PAGE_LIMIT("page_limit"),
    PAGE_MISSING("page_missing"),
    PERMISSION_LOST("permission_lost"),
    RECOGNITION_FAILED("recognition_failed"),
    RESOURCE_LIMIT("resource_limit"),
    AMBIGUOUS_TRANSFORM("ambiguous_transform"),
    REGION_OUT_OF_BOUNDS("region_out_of_bounds"),
    INVALID_RECOGNITION("invalid_recognition"),
    ;

    companion object {
        /** Returns the reason with [code], or null for an unknown code. */
        fun fromCode(code: String): DocumentIntakeReason? = values().firstOrNull { it.code == code }
    }
}

/**
 * Whether a page contributed text and evidence, or was held with a reason.
 */
enum class DocumentPageStatus(val code: String) {
    MAPPED("mapped"),
    HELD("held"),
}

/**
 * Coordinate unit of an original page: pixels for images, PDF points for PDFs.
 */
enum class DocumentSpatialUnit(val code: String) {
    PIXEL("pixel"),
    POINT("point"),
}

/**
 * Kind of visual evidence, matching OpenMedKit's `MultimodalVisualKind`.
 */
enum class DocumentVisualKind(val code: String) {
    PAGE("page_geometry"),
    PIXELS("pixels"),
    OCR("ocr_span"),
}

/**
 * Value-free intake failure carrying only reason codes.
 *
 * Sources throw it for document- or page-level failures, and
 * [DocumentIntake.extract] throws it when an intake is not complete.
 */
class DocumentIntakeException(
    val reasons: List<DocumentIntakeReason>,
    val outcome: DocumentIntakeOutcome = DocumentIntakeOutcome.REFUSED,
) : IllegalStateException(reasons.joinToString(",") { it.code }) {
    constructor(reason: DocumentIntakeReason) : this(listOf(reason))

    init {
        require(reasons.isNotEmpty()) { "at least one reason is required" }
    }

    /** The first reason in report order. */
    val reason: DocumentIntakeReason
        get() = reasons.first()
}

/**
 * Half-open, top-left-origin rectangle in pixels or PDF points.
 */
data class DocumentBox(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double,
)

/**
 * Original page extent plus the crop and clockwise rotation handed to OCR.
 *
 * The OCR image is the crop, turned upright by [rotationDegrees], at any render
 * scale. Inconsistent declarations hold the page as
 * [DocumentIntakeReason.AMBIGUOUS_TRANSFORM] when it is mapped.
 *
 * @param width Original page width, top-left origin, in [unit].
 * @param height Original page height.
 * @param unit Pixels for images, PDF points for PDF pages.
 * @param rotationDegrees Clockwise quarter turn that made the crop upright.
 * @param crop Original-page region handed to OCR; null means the whole page.
 */
data class DocumentPageGeometry(
    val width: Double,
    val height: Double,
    val unit: DocumentSpatialUnit = DocumentSpatialUnit.PIXEL,
    val rotationDegrees: Int = 0,
    val crop: DocumentBox? = null,
)

/**
 * One OCR region: native UTF-16 indices into the page text and an OCR-image box.
 *
 * A null [box] cannot be mapped and holds the page as
 * [DocumentIntakeReason.INVALID_RECOGNITION].
 */
data class DocumentRecognizedRegion(
    val start: Int,
    val end: Int,
    val box: DocumentBox?,
)

/**
 * OCR output for one upright page image. [toString] never includes the text.
 *
 * @param width Upright OCR image width in pixels.
 * @param height Upright OCR image height in pixels.
 */
class DocumentRecognition(
    val width: Int,
    val height: Int,
    val text: String,
    val regions: List<DocumentRecognizedRegion>,
) {
    override fun toString(): String =
        "DocumentRecognition(width=$width, height=$height, regions=${regions.size})"

    companion object {
        /**
         * Adapts [OcrAdapter][com.openmed.openmedkit.ocr.OcrAdapter] output whose
         * token boxes use the upright image's pixel space.
         */
        fun fromOcrResult(width: Int, height: Int, result: OcrResult): DocumentRecognition =
            DocumentRecognition(
                width = width,
                height = height,
                text = result.text,
                regions = result.tokens.map { token ->
                    DocumentRecognizedRegion(
                        start = token.start,
                        end = token.end,
                        box = token.bbox?.let {
                            DocumentBox(
                                left = it.left.toDouble(),
                                top = it.top.toDouble(),
                                right = it.right.toDouble(),
                                bottom = it.bottom.toDouble(),
                            )
                        },
                    )
                },
            )
    }
}

/**
 * What a native source produced for one page, in page order.
 */
sealed class DocumentPageInput {
    /** A rendered and recognized page. */
    class Rendered(
        val geometry: DocumentPageGeometry,
        val recognition: DocumentRecognition,
    ) : DocumentPageInput()

    /**
     * A page that could not be rendered or recognized.
     *
     * [DocumentIntakeReason.PERMISSION_LOST] ends the document, and
     * [DocumentIntakeReason.PAGE_LIMIT] marks a source that stopped at the limit.
     */
    data class Unavailable(val reason: DocumentIntakeReason) : DocumentPageInput()
}

/**
 * Bounds applied before any page contributes text.
 */
data class DocumentIntakeLimits(
    val maxPages: Int = 2048,
    val maxRegionsPerPage: Int = 4096,
    val maxPageScalars: Int = 65536,
    val maxDocumentBytes: Long = 64L * 1024 * 1024,
) {
    init {
        require(maxPages >= 1) { "maxPages must be positive" }
        require(maxRegionsPerPage >= 1) { "maxRegionsPerPage must be positive" }
        require(maxPageScalars >= 1) { "maxPageScalars must be positive" }
        require(maxDocumentBytes >= 1) { "maxDocumentBytes must be positive" }
    }
}

/**
 * Visual evidence record equivalent to OpenMedKit's `MultimodalVisual`.
 *
 * Source coordinates stay separate from the display geometry derived from
 * [rotationDegrees]. Spans are Unicode scalar offsets into [DocumentEvidence.text].
 */
data class DocumentVisualRecord(
    val nodeRef: String,
    val sourceDigest: String,
    val kind: DocumentVisualKind,
    val page: Int,
    val width: Double,
    val height: Double,
    val unit: DocumentSpatialUnit,
    val sourceBox: DocumentBox?,
    val span: ScalarSpan?,
    val rotationDegrees: Int,
    val displayWidth: Double,
    val displayHeight: Double,
    val displayBox: DocumentBox?,
)

/**
 * Page record: mapped with a text span and evidence, or held with a reason.
 */
data class DocumentPageRecord(
    val page: Int,
    val status: DocumentPageStatus,
    val reason: DocumentIntakeReason? = null,
    val textSpan: ScalarSpan? = null,
    val evidence: DocumentVisualRecord? = null,
    val cropEvidence: DocumentVisualRecord? = null,
)

/**
 * Recognized region mapped back to the original page.
 */
data class DocumentRegionRecord(
    val page: Int,
    val span: ScalarSpan,
    val sourceBox: DocumentBox,
    val evidence: DocumentVisualRecord,
) {
    val nodeRef: String
        get() = evidence.nodeRef
}

/**
 * Bounded intake evidence. Every region span indexes [text]; held pages add no text.
 *
 * [toString] reports codes and counts only.
 */
class DocumentEvidence(
    val sourceDigest: String?,
    val outcome: DocumentIntakeOutcome,
    val reasons: List<DocumentIntakeReason>,
    val text: String,
    val pages: List<DocumentPageRecord>,
    val regions: List<DocumentRegionRecord>,
) {
    /** Returns regions whose scalar span overlaps the scalar span `[start, end)`. */
    fun regionsForSpan(start: Int, end: Int): List<DocumentRegionRecord> {
        require(start in 0..end) { "span must be ordered and non-negative" }
        return regions.filter { it.span.start < end && start < it.span.end }
    }

    /** Returns regions overlapping a native UTF-16 span of [text]. */
    fun regionsForUtf16Span(start: Int, end: Int): List<DocumentRegionRecord> =
        regionsForSpan(
            UnicodeOffsetContract.utf16ToScalarOffset(text, start),
            UnicodeOffsetContract.utf16ToScalarOffset(text, end),
        )

    override fun toString(): String =
        "DocumentEvidence(outcome=${outcome.code}, " +
            "reasons=${reasons.map { it.code }}, pages=${pages.size}, regions=${regions.size})"
}

/**
 * Reference mapping shared with Python and Swift through
 * `tests/fixtures/parity/document_intake_v1.json`.
 *
 * OCR-image boxes are mapped back by undoing the render scale, the quarter-turn
 * rotation and the crop, in that order. The arithmetic order is part of the
 * contract, so all three implementations produce identical binary64 values.
 */
object DocumentEvidenceMapper {
    const val SCHEMA_VERSION: String = "openmed.multimodal.document-intake.v1"
    const val PAGE_SEPARATOR: String = "\n\n"

    /** Overshoot past the OCR image edge, as a fraction of its size, that is clamped. */
    const val BOX_TOLERANCE: Double = 0.01

    private const val MAX_COORDINATE = 2_147_483_647.0
    private val QUARTER_TURNS = setOf(0, 90, 180, 270)
    private val DIGEST = Regex("[0-9a-f]{64}")

    /** Opaque 128-bit reference for `"page"`, `"crop"` or `"region:<index>"`. */
    fun nodeRef(sourceDigest: String, page: Int, label: String): String {
        val identity = "$SCHEMA_VERSION|$sourceDigest|$page|$label"
        val digest = MessageDigest.getInstance("SHA-256").digest(identity.toByteArray(Charsets.US_ASCII))
        return digest.take(16).joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    /** Evidence for a document that could not be opened. */
    fun refused(reason: DocumentIntakeReason): DocumentEvidence =
        DocumentEvidence(
            sourceDigest = null,
            outcome = DocumentIntakeOutcome.REFUSED,
            reasons = listOf(reason),
            text = "",
            pages = emptyList(),
            regions = emptyList(),
        )

    /**
     * Maps one OCR-image box to original-page coordinates.
     *
     * @throws DocumentIntakeException with the hold reason when the box or
     *   geometry cannot be mapped unambiguously.
     */
    fun mapBox(
        box: DocumentBox,
        geometry: DocumentPageGeometry,
        imageWidth: Int,
        imageHeight: Int,
    ): DocumentBox {
        val crop = checkedCrop(geometry, imageWidth, imageHeight)
        val cropWidth = crop.right - crop.left
        val cropHeight = crop.bottom - crop.top
        val quarter = geometry.rotationDegrees == 90 || geometry.rotationDegrees == 270
        val uprightWidth = if (quarter) cropHeight else cropWidth
        val uprightHeight = if (quarter) cropWidth else cropHeight
        val width = imageWidth.toDouble()
        val height = imageHeight.toDouble()
        if (!listOf(box.left, box.top, box.right, box.bottom).all { it.isFinite() }) {
            hold(DocumentIntakeReason.REGION_OUT_OF_BOUNDS)
        }
        val toleranceX = width * BOX_TOLERANCE
        val toleranceY = height * BOX_TOLERANCE
        if (box.left < -toleranceX || box.top < -toleranceY ||
            box.right > width + toleranceX || box.bottom > height + toleranceY
        ) {
            hold(DocumentIntakeReason.REGION_OUT_OF_BOUNDS)
        }
        val x0 = clamp(box.left, 0.0, width)
        val x1 = clamp(box.right, 0.0, width)
        val y0 = clamp(box.top, 0.0, height)
        val y1 = clamp(box.bottom, 0.0, height)
        if (!(x0 < x1 && y0 < y1)) {
            hold(DocumentIntakeReason.REGION_OUT_OF_BOUNDS)
        }
        val u0 = clamp(x0 * uprightWidth / width, 0.0, uprightWidth)
        val u1 = clamp(x1 * uprightWidth / width, 0.0, uprightWidth)
        val v0 = clamp(y0 * uprightHeight / height, 0.0, uprightHeight)
        val v1 = clamp(y1 * uprightHeight / height, 0.0, uprightHeight)
        val local = when (geometry.rotationDegrees) {
            0 -> doubleArrayOf(u0, v0, u1, v1)
            90 -> doubleArrayOf(v0, cropHeight - u1, v1, cropHeight - u0)
            180 -> doubleArrayOf(cropWidth - u1, cropHeight - v1, cropWidth - u0, cropHeight - v0)
            else -> doubleArrayOf(cropWidth - v1, u0, cropWidth - v0, u1)
        }
        val mapped = DocumentBox(
            left = clamp(crop.left + local[0], crop.left, crop.right) + 0.0,
            top = clamp(crop.top + local[1], crop.top, crop.bottom) + 0.0,
            right = clamp(crop.left + local[2], crop.left, crop.right) + 0.0,
            bottom = clamp(crop.top + local[3], crop.top, crop.bottom) + 0.0,
        )
        if (!(mapped.left < mapped.right && mapped.top < mapped.bottom)) {
            hold(DocumentIntakeReason.AMBIGUOUS_TRANSFORM)
        }
        return mapped
    }

    /**
     * Maps native page inputs to page, crop and region evidence.
     *
     * The page index is the position in [pages]. [DocumentIntakeOutcome.PARTIAL]
     * means at least one page was mapped and a reason applies;
     * [DocumentIntakeOutcome.REFUSED] means no page was mapped.
     */
    fun map(
        sourceDigest: String,
        pages: List<DocumentPageInput>,
        limits: DocumentIntakeLimits = DocumentIntakeLimits(),
        pageSeparator: String = PAGE_SEPARATOR,
    ): DocumentEvidence {
        require(DIGEST.matches(sourceDigest)) { "sourceDigest must be a lowercase SHA-256" }
        if (pages.isEmpty()) {
            return refused(DocumentIntakeReason.EMPTY_DOCUMENT)
        }
        val reasons = mutableSetOf<DocumentIntakeReason>()
        if (pages.size > limits.maxPages) {
            reasons += DocumentIntakeReason.PAGE_LIMIT
        }
        val text = StringBuilder()
        var length = 0
        val records = mutableListOf<DocumentPageRecord>()
        val regions = mutableListOf<DocumentRegionRecord>()
        for ((index, item) in pages.take(limits.maxPages).withIndex()) {
            val reason = when (item) {
                is DocumentPageInput.Unavailable -> {
                    if (item.reason == DocumentIntakeReason.PAGE_LIMIT) {
                        reasons += DocumentIntakeReason.PAGE_LIMIT
                        break
                    }
                    pageReason(item.reason)
                }
                is DocumentPageInput.Rendered -> {
                    val separator = if (records.any { it.status == DocumentPageStatus.MAPPED }) {
                        pageSeparator
                    } else {
                        ""
                    }
                    val offset = length + UnicodeOffsetContract.scalarLength(separator)
                    try {
                        val (page, mapped) = mapPage(sourceDigest, index, item, limits, offset)
                        text.append(separator).append(item.recognition.text)
                        length = offset + UnicodeOffsetContract.scalarLength(item.recognition.text)
                        records += page
                        regions += mapped
                        continue
                    } catch (hold: DocumentIntakeException) {
                        hold.reason
                    }
                }
            }
            reasons += reason
            records += DocumentPageRecord(page = index, status = DocumentPageStatus.HELD, reason = reason)
            if (reason == DocumentIntakeReason.PERMISSION_LOST) {
                break
            }
        }
        val outcome = when {
            records.none { it.status == DocumentPageStatus.MAPPED } -> DocumentIntakeOutcome.REFUSED
            reasons.isEmpty() -> DocumentIntakeOutcome.COMPLETE
            else -> DocumentIntakeOutcome.PARTIAL
        }
        return DocumentEvidence(
            sourceDigest = sourceDigest,
            outcome = outcome,
            reasons = reasons.sortedBy { it.ordinal },
            text = text.toString(),
            pages = records.toList(),
            regions = regions.toList(),
        )
    }

    /** Normalizes a document-level failure, for a source that could not be opened. */
    internal fun documentReason(reason: DocumentIntakeReason): DocumentIntakeReason =
        when (reason) {
            DocumentIntakeReason.PERMISSION_DENIED,
            DocumentIntakeReason.PERMISSION_LOST,
            -> DocumentIntakeReason.PERMISSION_DENIED
            DocumentIntakeReason.DOCUMENT_LOCKED -> DocumentIntakeReason.DOCUMENT_LOCKED
            DocumentIntakeReason.SIZE_LIMIT -> DocumentIntakeReason.SIZE_LIMIT
            else -> DocumentIntakeReason.UNSUPPORTED_DOCUMENT
        }

    /** Normalizes a page-level failure, after the document was opened. */
    internal fun pageReason(reason: DocumentIntakeReason): DocumentIntakeReason =
        when (reason) {
            DocumentIntakeReason.PERMISSION_LOST,
            DocumentIntakeReason.PERMISSION_DENIED,
            -> DocumentIntakeReason.PERMISSION_LOST
            DocumentIntakeReason.AMBIGUOUS_TRANSFORM -> DocumentIntakeReason.AMBIGUOUS_TRANSFORM
            DocumentIntakeReason.RESOURCE_LIMIT,
            DocumentIntakeReason.SIZE_LIMIT,
            -> DocumentIntakeReason.RESOURCE_LIMIT
            DocumentIntakeReason.RECOGNITION_FAILED -> DocumentIntakeReason.RECOGNITION_FAILED
            else -> DocumentIntakeReason.PAGE_MISSING
        }

    private fun hold(reason: DocumentIntakeReason): Nothing = throw DocumentIntakeException(reason)

    private fun clamp(value: Double, low: Double, high: Double): Double =
        when {
            value < low -> low
            value > high -> high
            else -> value
        }

    private fun bounded(value: Double): Boolean =
        value.isFinite() && value >= 0.0 && value <= MAX_COORDINATE

    private fun checkedCrop(geometry: DocumentPageGeometry, width: Int, height: Int): DocumentBox {
        if (!bounded(geometry.width) || !bounded(geometry.height) ||
            geometry.width <= 0.0 || geometry.height <= 0.0 ||
            geometry.rotationDegrees !in QUARTER_TURNS || width < 1 || height < 1
        ) {
            hold(DocumentIntakeReason.AMBIGUOUS_TRANSFORM)
        }
        val crop = geometry.crop ?: DocumentBox(0.0, 0.0, geometry.width, geometry.height)
        if (!listOf(crop.left, crop.top, crop.right, crop.bottom).all(::bounded) ||
            !(crop.left < crop.right && crop.right <= geometry.width) ||
            !(crop.top < crop.bottom && crop.bottom <= geometry.height)
        ) {
            hold(DocumentIntakeReason.AMBIGUOUS_TRANSFORM)
        }
        val cropWidth = crop.right - crop.left
        val cropHeight = crop.bottom - crop.top
        val quarter = geometry.rotationDegrees == 90 || geometry.rotationDegrees == 270
        val uprightWidth = if (quarter) cropHeight else cropWidth
        val uprightHeight = if (quarter) cropWidth else cropHeight
        // Rendering rounds each OCR dimension to a whole pixel, so the image may
        // differ from the upright crop's aspect by at most half a pixel per side.
        val skew = Math.abs(width.toDouble() * uprightHeight - height.toDouble() * uprightWidth)
        if (!(skew <= (uprightWidth + uprightHeight) * 0.5)) {
            hold(DocumentIntakeReason.AMBIGUOUS_TRANSFORM)
        }
        return crop
    }

    private fun scalarOffset(text: String, utf16Index: Int): Int {
        if (utf16Index !in 0..text.length || !isBoundary(text, utf16Index)) {
            hold(DocumentIntakeReason.INVALID_RECOGNITION)
        }
        return text.codePointCount(0, utf16Index)
    }

    private fun isBoundary(text: String, index: Int): Boolean =
        index == 0 || index == text.length ||
            !(Character.isHighSurrogate(text[index - 1]) && Character.isLowSurrogate(text[index]))

    private fun display(
        box: DocumentBox,
        geometry: DocumentPageGeometry,
    ): DocumentBox {
        val width = geometry.width
        val height = geometry.height
        val result = when (geometry.rotationDegrees) {
            0 -> DocumentBox(box.left, box.top, box.right, box.bottom)
            90 -> DocumentBox(height - box.bottom, box.left, height - box.top, box.right)
            180 -> DocumentBox(width - box.right, height - box.bottom, width - box.left, height - box.top)
            else -> DocumentBox(box.top, width - box.right, box.bottom, width - box.left)
        }
        if (!(result.left < result.right && result.top < result.bottom)) {
            hold(DocumentIntakeReason.AMBIGUOUS_TRANSFORM)
        }
        return DocumentBox(result.left + 0.0, result.top + 0.0, result.right + 0.0, result.bottom + 0.0)
    }

    private fun visual(
        digest: String,
        index: Int,
        label: String,
        kind: DocumentVisualKind,
        geometry: DocumentPageGeometry,
        box: DocumentBox?,
        span: ScalarSpan?,
    ): DocumentVisualRecord {
        val quarter = geometry.rotationDegrees == 90 || geometry.rotationDegrees == 270
        return DocumentVisualRecord(
            nodeRef = nodeRef(digest, index, label),
            sourceDigest = digest,
            kind = kind,
            page = index,
            width = geometry.width,
            height = geometry.height,
            unit = geometry.unit,
            sourceBox = box,
            span = span,
            rotationDegrees = geometry.rotationDegrees,
            displayWidth = if (quarter) geometry.height else geometry.width,
            displayHeight = if (quarter) geometry.width else geometry.height,
            displayBox = box?.let { display(it, geometry) },
        )
    }

    private fun mapPage(
        digest: String,
        index: Int,
        input: DocumentPageInput.Rendered,
        limits: DocumentIntakeLimits,
        offset: Int,
    ): Pair<DocumentPageRecord, List<DocumentRegionRecord>> {
        val geometry = input.geometry
        val recognition = input.recognition
        val crop = checkedCrop(geometry, recognition.width, recognition.height)
        val text = recognition.text
        val length = UnicodeOffsetContract.scalarLength(text)
        if (recognition.regions.size > limits.maxRegionsPerPage || length > limits.maxPageScalars) {
            hold(DocumentIntakeReason.RESOURCE_LIMIT)
        }
        val mapped = recognition.regions.map { region ->
            val start = scalarOffset(text, region.start)
            val end = scalarOffset(text, region.end)
            if (!(start < end)) {
                hold(DocumentIntakeReason.INVALID_RECOGNITION)
            }
            val box = region.box ?: hold(DocumentIntakeReason.INVALID_RECOGNITION)
            ScalarSpan(offset + start, offset + end) to
                mapBox(box, geometry, recognition.width, recognition.height)
        }
        val page = visual(digest, index, "page", DocumentVisualKind.PAGE, geometry, null, null)
        val cropEvidence = geometry.crop?.let {
            visual(digest, index, "crop", DocumentVisualKind.PIXELS, geometry, crop, null)
        }
        val regions = mapped.mapIndexed { ordinal, (span, box) ->
            DocumentRegionRecord(
                page = index,
                span = span,
                sourceBox = box,
                evidence = visual(digest, index, "region:$ordinal", DocumentVisualKind.OCR, geometry, box, span),
            )
        }
        val record = DocumentPageRecord(
            page = index,
            status = DocumentPageStatus.MAPPED,
            textSpan = ScalarSpan(offset, offset + length),
            evidence = page,
            cropEvidence = cropEvidence,
        )
        return record to regions
    }
}
