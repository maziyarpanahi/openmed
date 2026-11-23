package com.openmed.openmedkit.export

import com.openmed.openmedkit.DeidentifiedSpanAction
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.intake.DocumentBox
import com.openmed.openmedkit.intake.DocumentIntakeOutcome
import com.openmed.openmedkit.intake.DocumentIntakeReport
import com.openmed.openmedkit.intake.DocumentPageGeometry
import com.openmed.openmedkit.intake.DocumentPageRecord
import com.openmed.openmedkit.intake.DocumentPageStatus
import com.openmed.openmedkit.policy.PolicyAction
import java.security.MessageDigest
import java.security.SecureRandom
import java.text.Normalizer
import java.util.Locale

/**
 * Closed, value-free export failures shared with Swift OpenMedKit.
 */
enum class DocumentRedactionExportReason(val code: String) {
    /** The intake report is not complete, so some page was never examined. */
    INCOMPLETE_REPORT("incomplete_report"),

    /** A mask span or box lies outside the report text or its page. */
    MASK_OUT_OF_RANGE("mask_out_of_range"),

    /** A mask span covers text that no recognized region places on a page. */
    MASK_UNPLACED("mask_unplaced"),

    /** The source no longer has the digest, page count or geometry of the report. */
    SOURCE_CHANGED("source_changed"),

    /** A page could not be rendered into pixels. */
    RENDER_FAILED("render_failed"),

    /** The sanitized artifact exceeds the byte or pixel limit. */
    RESOURCE_LIMIT("resource_limit"),

    /** Private export storage could not be created, written or read. */
    STORAGE_UNAVAILABLE("storage_unavailable"),

    /** Re-extraction of the artifact did not complete on every page. */
    VERIFICATION_INCOMPLETE("verification_incomplete"),

    /** The artifact does not have one page per source page. */
    PAGE_COUNT_MISMATCH("page_count_mismatch"),

    /** A painted mask is not opaque in the re-read artifact. */
    MASK_NOT_OPAQUE("mask_not_opaque"),

    /** The artifact carries a text layer or font resources. */
    HIDDEN_TEXT("hidden_text"),

    /** The artifact carries annotations, attachments, forms or actions. */
    ATTACHMENTS_PRESENT("attachments_present"),

    /** The artifact carries document metadata. */
    METADATA_PRESENT("metadata_present"),

    /** The artifact is not exactly the image-only PDF profile. */
    PROFILE_VIOLATION("profile_violation"),

    /** Text extracted from the artifact still contains a masked identifier. */
    RESIDUAL_TEXT("residual_text"),

    /** The source, masks or policy changed since the artifact was exported. */
    STALE_REQUEST("stale_request"),

    /** The artifact was completed, invalidated or never exported here. */
    ARTIFACT_UNAVAILABLE("artifact_unavailable"),

    /** The stored artifact bytes no longer match the verified digest. */
    ARTIFACT_CHANGED("artifact_changed"),
}

/**
 * Value-free export failure; the message is the reason code only.
 */
class DocumentRedactionExportException(
    val reason: DocumentRedactionExportReason,
) : IllegalStateException(reason.code)

/**
 * Verification checks every exported artifact passed, in receipt order.
 */
enum class DocumentRedactionCheck(val code: String) {
    PROFILE("profile"),
    HIDDEN_TEXT("hidden_text"),
    ATTACHMENTS("attachments"),
    METADATA("metadata"),
    PAGE_COUNT("page_count"),
    VISIBLE_PIXELS("visible_pixels"),
    RESIDUAL_TEXT("residual_text"),
}

/**
 * A region to remove from the exported pages.
 */
sealed class DocumentRedactionMask {
    /**
     * Unicode scalar offsets `[start, end)` into the report text. Every
     * recognized region the span overlaps is painted over, and every
     * non-whitespace scalar in it must belong to a recognized region.
     */
    data class Span(val start: Int, val end: Int) : DocumentRedactionMask()

    /**
     * A box on a mapped page, in original-page coordinates with a top-left
     * origin and the page unit, for content OCR cannot see, such as a signature.
     */
    data class Box(val page: Int, val box: DocumentBox) : DocumentRedactionMask()

    companion object {
        /** Converts a native UTF-16 span of [text] into a scalar span mask. */
        fun fromUtf16(text: String, start: Int, end: Int): Span =
            Span(
                UnicodeOffsetContract.utf16ToScalarOffset(text, start),
                UnicodeOffsetContract.utf16ToScalarOffset(text, end),
            )

        /** Masks every policy de-identification action that was not kept. */
        fun fromActions(actions: List<DeidentifiedSpanAction>): List<DocumentRedactionMask> =
            actions.filter { it.action != PolicyAction.KEEP }.map { Span(it.start, it.end) }
    }
}

/**
 * Export settings bound into every artifact; changing them invalidates it.
 *
 * @param policyId Caller policy identity, such as a de-identification profile
 *   name. Only its digest enters the binding; it never appears in a receipt.
 * @param padding OCR-image pixels added on every side of a painted mask.
 * @param minimumResidualTokenScalars Masked tokens shorter than this many
 *   scalars are not searched for in the re-extracted text.
 */
class DocumentRedactionPolicy(
    val policyId: String = DEFAULT_POLICY_ID,
    padding: Double = 2.0,
    minimumResidualTokenScalars: Int = 2,
) {
    val padding: Double = if (padding.isFinite()) maxOf(0.0, padding) else 0.0
    val minimumResidualTokenScalars: Int = maxOf(1, minimumResidualTokenScalars)

    override fun equals(other: Any?): Boolean =
        other is DocumentRedactionPolicy && other.policyId == policyId &&
            other.padding == padding && other.minimumResidualTokenScalars == minimumResidualTokenScalars

    override fun hashCode(): Int = listOf(policyId, padding, minimumResidualTokenScalars).hashCode()

    override fun toString(): String =
        "DocumentRedactionPolicy(padding=$padding, minimumResidualTokenScalars=$minimumResidualTokenScalars)"

    companion object {
        const val DEFAULT_POLICY_ID: String = "openmed.redaction-export.default"
    }
}

/**
 * A verified redaction result: the intake report, the applied masks and the policy.
 *
 * [binding] is the SHA-256 that ties an artifact to exactly this source, text,
 * mask set and policy. Mask order and duplicates do not change it, and Swift
 * OpenMedKit computes the same value.
 */
class DocumentRedactionRequest(
    val report: DocumentIntakeReport,
    val masks: List<DocumentRedactionMask>,
    val policy: DocumentRedactionPolicy = DocumentRedactionPolicy(),
) {
    val binding: String = DocumentRedactionExportContract.binding(report, masks, policy)

    override fun toString(): String = "DocumentRedactionRequest(masks=${masks.size}, binding=$binding)"
}

/**
 * Value-free evidence of one verified export.
 *
 * @param paintedRegionCount Rectangles painted across all pages after masks
 *   were mapped to pixels.
 */
data class DocumentRedactionReceipt(
    val schemaVersion: String,
    val profile: String,
    val binding: String,
    val sourceDigest: String,
    val outputDigest: String,
    val byteCount: Long,
    val pageCount: Int,
    val maskCount: Int,
    val paintedRegionCount: Int,
    val checks: List<DocumentRedactionCheck>,
)

/**
 * A verified artifact in private storage. It is shared only through an
 * explicit exporter action and exposes no file location itself.
 *
 * @param id Random 128-bit reference, also the artifact file name.
 */
data class DocumentRedactionArtifact(
    val id: String,
    val receipt: DocumentRedactionReceipt,
)

/** One mapped page and the original-page boxes to paint on it. */
internal data class DocumentRedactionPagePlan(
    val page: Int,
    val geometry: DocumentPageGeometry,
    val boxes: List<DocumentBox>,
)

/**
 * Mask placement, pixel mapping, binding and residual search shared with Swift.
 */
object DocumentRedactionExportContract {
    const val SCHEMA_VERSION: String = "openmed.redaction-export.v1"
    const val PROFILE: String = "openmed.redaction-export.image-pdf.v1"

    private val random = SecureRandom()

    /**
     * SHA-256 over the source digest, text digest, page count, policy and the
     * sorted, de-duplicated masks.
     */
    fun binding(
        report: DocumentIntakeReport,
        masks: List<DocumentRedactionMask>,
        policy: DocumentRedactionPolicy,
    ): String {
        val lines = masks.map { mask ->
            when (mask) {
                is DocumentRedactionMask.Span -> "span:${mask.start}:${mask.end}"
                is DocumentRedactionMask.Box -> "box:${mask.page}:" +
                    listOf(mask.box.left, mask.box.top, mask.box.right, mask.box.bottom)
                        .joinToString(":", transform = ::bits)
            }
        }.toSortedSet()
        val evidence = report.evidence
        val header = listOf(
            SCHEMA_VERSION,
            "source:${evidence.sourceDigest ?: ""}",
            "text:${sha256(evidence.text.toByteArray(Charsets.UTF_8))}",
            "pages:${evidence.pages.size}",
            "policy:${sha256(policy.policyId.toByteArray(Charsets.UTF_8))}:" +
                "${bits(policy.padding)}:${policy.minimumResidualTokenScalars}",
        )
        return sha256((header + lines).joinToString("\n").toByteArray(Charsets.UTF_8))
    }

    /**
     * Maps an original-page box into OCR-image pixels: the crop, then the
     * quarter-turn rotation, then the render scale. This is the forward
     * transform that `DocumentEvidenceMapper.mapBox` inverts. Returns null when
     * the box lies outside the crop, which is not part of the exported image.
     */
    fun imageBox(
        box: DocumentBox,
        geometry: DocumentPageGeometry,
        imageWidth: Int,
        imageHeight: Int,
    ): DocumentBox? {
        val crop = geometry.crop ?: DocumentBox(0.0, 0.0, geometry.width, geometry.height)
        val left = maxOf(box.left, crop.left) - crop.left
        val top = maxOf(box.top, crop.top) - crop.top
        val right = minOf(box.right, crop.right) - crop.left
        val bottom = minOf(box.bottom, crop.bottom) - crop.top
        if (!(left < right && top < bottom) || imageWidth <= 0 || imageHeight <= 0) {
            return null
        }
        val cropWidth = crop.right - crop.left
        val cropHeight = crop.bottom - crop.top
        val quarter = geometry.rotationDegrees == 90 || geometry.rotationDegrees == 270
        val upright = when (geometry.rotationDegrees) {
            0 -> doubleArrayOf(left, top, right, bottom)
            90 -> doubleArrayOf(cropHeight - bottom, left, cropHeight - top, right)
            180 -> doubleArrayOf(cropWidth - right, cropHeight - bottom, cropWidth - left, cropHeight - top)
            270 -> doubleArrayOf(top, cropWidth - right, bottom, cropWidth - left)
            else -> return null
        }
        val scaleX = imageWidth.toDouble() / (if (quarter) cropHeight else cropWidth)
        val scaleY = imageHeight.toDouble() / (if (quarter) cropWidth else cropHeight)
        return DocumentBox(upright[0] * scaleX, upright[1] * scaleY, upright[2] * scaleX, upright[3] * scaleY)
    }

    /**
     * Whole-pixel rectangle covering [box] plus [padding], rounded outward and
     * clamped to the image, as `[left, top, right, bottom]`.
     */
    fun pixelRect(box: DocumentBox, padding: Double, imageWidth: Int, imageHeight: Int): IntArray? {
        val values = doubleArrayOf(box.left - padding, box.top - padding, box.right + padding, box.bottom + padding)
        if (!values.all { it.isFinite() }) {
            return null
        }
        val width = imageWidth.toDouble()
        val height = imageHeight.toDouble()
        val left = Math.floor(values[0]).coerceIn(0.0, width).toInt()
        val top = Math.floor(values[1]).coerceIn(0.0, height).toInt()
        val right = Math.ceil(values[2]).coerceIn(0.0, width).toInt()
        val bottom = Math.ceil(values[3]).coerceIn(0.0, height).toInt()
        return if (left < right && top < bottom) intArrayOf(left, top, right, bottom) else null
    }

    /** Normalized search tokens: NFKC, lowercased, maximal runs of letters, marks and numbers. */
    fun tokens(text: String): List<String> {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(Locale.ROOT)
        val result = mutableListOf<String>()
        val current = StringBuilder()
        normalized.codePoints().forEach { codePoint ->
            if (isTokenScalar(codePoint)) {
                current.appendCodePoint(codePoint)
            } else if (current.isNotEmpty()) {
                result += current.toString()
                current.setLength(0)
            }
        }
        if (current.isNotEmpty()) {
            result += current.toString()
        }
        return result
    }

    /**
     * True when a masked token of at least [minimumScalars] scalars is an
     * extracted token, or, for scripts written without spaces, occurs inside one.
     */
    fun containsResidual(masked: List<String>, extracted: String, minimumScalars: Int): Boolean {
        val found = tokens(extracted)
        val exact = found.toSet()
        for (text in masked) {
            for (token in tokens(text)) {
                if (UnicodeOffsetContract.scalarLength(token) < maxOf(1, minimumScalars)) {
                    continue
                }
                if (token in exact) {
                    return true
                }
                if (token.codePoints().anyMatch(::isUnspacedScript) && found.any { it.contains(token) }) {
                    return true
                }
            }
        }
        return false
    }

    internal fun plan(request: DocumentRedactionRequest): List<DocumentRedactionPagePlan> {
        val evidence = request.report.evidence
        if (evidence.outcome != DocumentIntakeOutcome.COMPLETE || evidence.reasons.isNotEmpty() ||
            evidence.pages.isEmpty() || evidence.sourceDigest == null ||
            evidence.pages.withIndex().any { (index, page) ->
                page.page != index || page.status != DocumentPageStatus.MAPPED
            }
        ) {
            fail(DocumentRedactionExportReason.INCOMPLETE_REPORT)
        }
        val geometries = evidence.pages.map(::geometry)
        val boxes = geometries.map { mutableListOf<DocumentBox>() }
        fun add(page: Int, box: DocumentBox) {
            if (box !in boxes[page]) {
                boxes[page] += box
            }
        }
        val scalars = evidence.text.codePoints().toArray()
        for (mask in request.masks) {
            when (mask) {
                is DocumentRedactionMask.Span -> {
                    if (!(mask.start >= 0 && mask.start < mask.end && mask.end <= scalars.size)) {
                        fail(DocumentRedactionExportReason.MASK_OUT_OF_RANGE)
                    }
                    val regions = evidence.regionsForSpan(mask.start, mask.end)
                    for (index in mask.start until mask.end) {
                        if (!isWhitespace(scalars[index]) &&
                            regions.none { index >= it.span.start && index < it.span.end }
                        ) {
                            fail(DocumentRedactionExportReason.MASK_UNPLACED)
                        }
                    }
                    regions.forEach { add(it.page, it.sourceBox) }
                }
                is DocumentRedactionMask.Box -> {
                    val box = mask.box
                    val geometry = geometries.getOrNull(mask.page)
                    if (geometry == null ||
                        !listOf(box.left, box.top, box.right, box.bottom).all { it.isFinite() } ||
                        box.left < 0.0 || box.top < 0.0 || !(box.left < box.right) || !(box.top < box.bottom) ||
                        box.right > geometry.width || box.bottom > geometry.height
                    ) {
                        fail(DocumentRedactionExportReason.MASK_OUT_OF_RANGE)
                    }
                    add(mask.page, box)
                }
            }
        }
        return geometries.mapIndexed { index, geometry ->
            DocumentRedactionPagePlan(page = index, geometry = geometry, boxes = boxes[index].toList())
        }
    }

    /** Masked text searched for in the re-extracted artifact. */
    internal fun maskedTexts(request: DocumentRedactionRequest): List<String> {
        val text = request.report.evidence.text
        return request.masks.filterIsInstance<DocumentRedactionMask.Span>().mapNotNull { mask ->
            runCatching { UnicodeOffsetContract.substring(text, mask.start, mask.end) }.getOrNull()
        }
    }

    internal fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }

    internal fun randomReference(): String {
        val bytes = ByteArray(16)
        random.nextBytes(bytes)
        return bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    internal fun fail(reason: DocumentRedactionExportReason): Nothing =
        throw DocumentRedactionExportException(reason)

    private fun geometry(page: DocumentPageRecord): DocumentPageGeometry {
        val visual = page.evidence ?: fail(DocumentRedactionExportReason.INCOMPLETE_REPORT)
        val crop = page.cropEvidence?.let {
            it.sourceBox ?: fail(DocumentRedactionExportReason.INCOMPLETE_REPORT)
        }
        return DocumentPageGeometry(
            width = visual.width,
            height = visual.height,
            unit = visual.unit,
            rotationDegrees = visual.rotationDegrees,
            crop = crop,
        )
    }

    private fun bits(value: Double): String =
        java.lang.Long.toHexString(java.lang.Double.doubleToRawLongBits(value + 0.0)).padStart(16, '0')

    /** The Unicode `White_Space` property. */
    private fun isWhitespace(codePoint: Int): Boolean =
        codePoint in 0x09..0x0D || codePoint == 0x20 || codePoint == 0x85 || codePoint == 0xA0 ||
            codePoint == 0x1680 || codePoint in 0x2000..0x200A || codePoint == 0x2028 ||
            codePoint == 0x2029 || codePoint == 0x202F || codePoint == 0x205F || codePoint == 0x3000

    private fun isTokenScalar(codePoint: Int): Boolean =
        when (Character.getType(codePoint).toByte()) {
            Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
            Character.MODIFIER_LETTER, Character.OTHER_LETTER, Character.NON_SPACING_MARK,
            Character.COMBINING_SPACING_MARK, Character.ENCLOSING_MARK, Character.DECIMAL_DIGIT_NUMBER,
            Character.LETTER_NUMBER, Character.OTHER_NUMBER,
            -> true
            else -> false
        }

    private fun isUnspacedScript(codePoint: Int): Boolean =
        Character.isIdeographic(codePoint) || codePoint in 0x3040..0x30FF || codePoint in 0x0E00..0x0EFF ||
            codePoint in 0x1000..0x109F || codePoint in 0x1780..0x17FF
}
