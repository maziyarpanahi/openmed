package com.openmed.openmedkit.intake

import android.content.Context
import android.net.Uri
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrImage
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
import java.io.IOException
import java.security.MessageDigest
import java.util.Locale
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/**
 * Supported source document kinds.
 */
enum class DocumentInputKind {
    PDF,
    IMAGE,
}

/**
 * Character range for one source page inside document-level text.
 */
data class DocumentPage(
    val page: Int,
    val start: Int,
    val end: Int,
    val width: Int? = null,
    val height: Int? = null,
) {
    init {
        require(page >= 0) { "page must be non-negative" }
        require(start >= 0) { "start must be non-negative" }
        require(end >= start) { "end must be greater than or equal to start" }
        if (width != null) {
            require(width > 0) { "width must be positive when set" }
        }
        if (height != null) {
            require(height > 0) { "height must be positive when set" }
        }
    }
}

/**
 * Text extracted from a PDF or image URI plus span-placement metadata.
 *
 * [offsetMap], [pages] and [tokens] use native UTF-16 indices. [evidence] holds
 * the shared page, crop and region records in Unicode scalar offsets, with
 * boxes on the original page; its text equals [text].
 */
data class DocumentIntakeResult(
    val text: String,
    val offsetMap: OffsetMap,
    val pages: List<DocumentPage>,
    val tokens: List<OcrToken>,
    val inputKind: DocumentInputKind,
    val metadata: Map<String, String> = emptyMap(),
    val evidence: DocumentEvidence? = null,
) {
    init {
        require(offsetMap.size == text.length) {
            "offset map must have one entry per text character"
        }
        tokens.forEach { token ->
            require(token.end <= text.length) { "token end offset exceeds text length" }
            require(text.substring(token.start, token.end) == token.text) {
                "token offsets must index back into the extracted text"
            }
        }
        if (evidence != null) {
            require(evidence.text == text) { "evidence text must equal the extracted text" }
        }
    }
}

/**
 * Bounded intake outcome with value-free reasons.
 *
 * [result] holds only mapped pages and is null when the intake was refused.
 */
class DocumentIntakeReport(
    val evidence: DocumentEvidence,
    val result: DocumentIntakeResult?,
) {
    val outcome: DocumentIntakeOutcome
        get() = evidence.outcome

    val reasons: List<DocumentIntakeReason>
        get() = evidence.reasons

    override fun toString(): String = "DocumentIntakeReport($evidence)"
}

/**
 * On-device document intake for Android OpenMedKit.
 *
 * The core artifact ships no OCR engine, so callers pass an [OcrAdapter]: the
 * ML Kit adapter from the optional `openmedkit-mlkit` artifact, another
 * on-device engine, or `FakeOcrAdapter` in tests.
 *
 * Every page is rendered or decoded in memory, recognized, and recycled before
 * the next one. The source is read once to compute its digest and enforce
 * [DocumentIntakeLimits.maxDocumentBytes]; nothing is written to storage.
 *
 * @param context Context used to resolve document URIs.
 * @param ocrAdapter On-device OCR engine that receives each page as an [OcrImage].
 * @param pdfPageSource Renders PDF pages to bitmaps.
 * @param pageSeparator Text inserted between pages in the document-level text.
 * @param imageSource Decodes image URIs to bitmaps plus EXIF rotation.
 * @param limits Page, region, text and byte bounds.
 */
class DocumentIntake(
    private val context: Context,
    private val ocrAdapter: OcrAdapter,
    private val pdfPageSource: PdfPageSource = PdfPageRenderer(),
    private val pageSeparator: String = DocumentEvidenceMapper.PAGE_SEPARATOR,
    private val imageSource: DocumentImageSource = BitmapDocumentImageSource(),
    private val limits: DocumentIntakeLimits = DocumentIntakeLimits(),
) {
    /**
     * Extracts a complete document.
     *
     * @throws DocumentIntakeException with value-free reasons when the intake is
     *   partial or refused; call [intake] to inspect a partial result.
     */
    suspend fun extract(uri: Uri): DocumentIntakeResult {
        val report = intake(uri)
        if (report.outcome != DocumentIntakeOutcome.COMPLETE) {
            throw DocumentIntakeException(report.reasons, report.outcome)
        }
        return requireNotNull(report.result)
    }

    /**
     * Runs a bounded intake with explicit complete, partial and refused outcomes.
     *
     * Missing pages, lost permission, OCR failures and ambiguous coordinate
     * transforms become value-free reasons; held pages contribute no text.
     * Only coroutine cancellation is thrown, after the open document and the
     * current page bitmap are released.
     */
    suspend fun intake(uri: Uri): DocumentIntakeReport {
        currentCoroutineContext().ensureActive()
        val kind = detectInputKind(uri) ?: return refused(DocumentIntakeReason.UNSUPPORTED_DOCUMENT)
        val digest = try {
            sourceDigest(uri)
        } catch (error: DocumentIntakeException) {
            return refused(DocumentEvidenceMapper.documentReason(error.reason))
        } catch (_: SecurityException) {
            return refused(DocumentIntakeReason.PERMISSION_DENIED)
        } catch (_: IOException) {
            return refused(DocumentIntakeReason.PERMISSION_DENIED)
        }
        val inputs = mutableListOf<DocumentPageInput>()
        val recognized = mutableMapOf<Int, RecognizedPage>()
        val refusal = when (kind) {
            DocumentInputKind.PDF -> collectPdf(uri, inputs, recognized)
            DocumentInputKind.IMAGE -> collectImage(uri, inputs, recognized)
        }
        if (refusal != null) {
            return refused(refusal)
        }
        val evidence = DocumentEvidenceMapper.map(digest, inputs, limits, pageSeparator)
        if (evidence.outcome == DocumentIntakeOutcome.REFUSED) {
            return DocumentIntakeReport(evidence = evidence, result = null)
        }
        val builder = ResultBuilder(pageSeparator = pageSeparator)
        evidence.pages
            .filter { it.status == DocumentPageStatus.MAPPED }
            .forEach { page ->
                val source = recognized.getValue(page.page)
                builder.appendPage(source.page, source.result, source.width, source.height)
            }
        return DocumentIntakeReport(
            evidence = evidence,
            result = builder.build(inputKind = kind, evidence = evidence),
        )
    }

    private suspend fun collectPdf(
        uri: Uri,
        inputs: MutableList<DocumentPageInput>,
        recognized: MutableMap<Int, RecognizedPage>,
    ): DocumentIntakeReason? {
        try {
            pdfPageSource.renderPages(context, uri) { event ->
                currentCoroutineContext().ensureActive()
                val page = when (event) {
                    is PdfPageEvent.Rendered -> event.page.page
                    is PdfPageEvent.Unavailable -> event.page
                }
                while (inputs.size < page && inputs.size < limits.maxPages) {
                    inputs += DocumentPageInput.Unavailable(DocumentIntakeReason.PAGE_MISSING)
                }
                if (inputs.size >= limits.maxPages) {
                    inputs += DocumentPageInput.Unavailable(DocumentIntakeReason.PAGE_LIMIT)
                    throw StopPages()
                }
                if (page < inputs.size) {
                    // A repeated or out-of-order page index cannot be placed.
                    inputs += DocumentPageInput.Unavailable(DocumentIntakeReason.PAGE_MISSING)
                    return@renderPages
                }
                when (event) {
                    is PdfPageEvent.Rendered -> {
                        val bitmap = event.page.bitmap
                        val image = OcrImage.fromBitmap(bitmap)
                        val geometry = event.page.geometry ?: DocumentPageGeometry(
                            width = bitmap.width.toDouble(),
                            height = bitmap.height.toDouble(),
                        )
                        inputs += recognizePage(image, geometry, inputs.size, page, recognized)
                    }
                    is PdfPageEvent.Unavailable -> {
                        val reason = DocumentEvidenceMapper.pageReason(event.reason)
                        inputs += DocumentPageInput.Unavailable(reason)
                        if (reason == DocumentIntakeReason.PERMISSION_LOST) {
                            throw StopPages()
                        }
                    }
                }
            }
        } catch (_: StopPages) {
            return null
        } catch (error: CancellationException) {
            throw error
        } catch (error: DocumentIntakeException) {
            return stopped(inputs, error.reason)
        } catch (_: SecurityException) {
            return stopped(inputs, DocumentIntakeReason.PERMISSION_DENIED)
        } catch (_: IOException) {
            return stopped(inputs, DocumentIntakeReason.PAGE_MISSING)
        } catch (_: RuntimeException) {
            return stopped(inputs, DocumentIntakeReason.PAGE_MISSING)
        }
        return null
    }

    private suspend fun collectImage(
        uri: Uri,
        inputs: MutableList<DocumentPageInput>,
        recognized: MutableMap<Int, RecognizedPage>,
    ): DocumentIntakeReason? {
        val loaded = try {
            imageSource.loadPage(context, uri)
        } catch (error: CancellationException) {
            throw error
        } catch (error: DocumentIntakeException) {
            return DocumentEvidenceMapper.documentReason(error.reason)
        } catch (_: SecurityException) {
            return DocumentIntakeReason.PERMISSION_DENIED
        } catch (_: IOException) {
            return DocumentIntakeReason.PERMISSION_DENIED
        } catch (_: RuntimeException) {
            return DocumentIntakeReason.UNSUPPORTED_DOCUMENT
        }
        try {
            inputs += recognizePage(loaded.image, loaded.geometry, 0, 0, recognized)
        } finally {
            loaded.image.bitmap.recycle()
        }
        return null
    }

    private suspend fun recognizePage(
        image: OcrImage.BitmapImage,
        geometry: DocumentPageGeometry,
        index: Int,
        sourcePage: Int,
        recognized: MutableMap<Int, RecognizedPage>,
    ): DocumentPageInput {
        val width = image.uprightWidth
        val height = image.uprightHeight
        val result = try {
            ocrAdapter.recognize(image)
        } catch (error: CancellationException) {
            throw error
        } catch (_: Exception) {
            currentCoroutineContext().ensureActive()
            return DocumentPageInput.Unavailable(DocumentIntakeReason.RECOGNITION_FAILED)
        }
        currentCoroutineContext().ensureActive()
        recognized[index] = RecognizedPage(sourcePage, result, width, height)
        return DocumentPageInput.Rendered(
            geometry = geometry,
            recognition = DocumentRecognition.fromOcrResult(width, height, result),
        )
    }

    /**
     * Converts a source failure into a document refusal, or into a hold for the
     * page being read when earlier pages were already read. Later pages are
     * never requested after a source stops.
     */
    private fun stopped(
        inputs: MutableList<DocumentPageInput>,
        reason: DocumentIntakeReason,
    ): DocumentIntakeReason? {
        if (inputs.isEmpty()) {
            return DocumentEvidenceMapper.documentReason(reason)
        }
        inputs += DocumentPageInput.Unavailable(DocumentEvidenceMapper.pageReason(reason))
        return null
    }

    private fun refused(reason: DocumentIntakeReason): DocumentIntakeReport =
        DocumentIntakeReport(evidence = DocumentEvidenceMapper.refused(reason), result = null)

    private fun sourceDigest(uri: Uri): String {
        val stream = context.contentResolver.openInputStream(uri)
            ?: throw DocumentIntakeException(DocumentIntakeReason.PERMISSION_DENIED)
        val digest = MessageDigest.getInstance("SHA-256")
        stream.use { input ->
            val buffer = ByteArray(DIGEST_BUFFER_BYTES)
            var total = 0L
            while (true) {
                val read = input.read(buffer)
                if (read < 0) {
                    break
                }
                total += read
                if (total > limits.maxDocumentBytes) {
                    throw DocumentIntakeException(DocumentIntakeReason.SIZE_LIMIT)
                }
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun detectInputKind(uri: Uri): DocumentInputKind? {
        val mimeType = context.contentResolver.getType(uri)?.lowercase(Locale.US)
        val extension = uri.path
            ?.substringAfterLast('.', missingDelimiterValue = "")
            ?.lowercase(Locale.US)

        return when {
            mimeType == "application/pdf" || extension == "pdf" -> DocumentInputKind.PDF
            mimeType?.startsWith("image/") == true || extension in IMAGE_EXTENSIONS ->
                DocumentInputKind.IMAGE
            else -> null
        }
    }
}

private const val DIGEST_BUFFER_BYTES = 64 * 1024

private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "webp", "heic", "heif")

/** OCR output kept for the legacy UTF-16 result of one mapped page. */
private class RecognizedPage(
    val page: Int,
    val result: OcrResult,
    val width: Int,
    val height: Int,
)

/** Ends page iteration early; never escapes [DocumentIntake]. */
private class StopPages : RuntimeException() {
    override fun fillInStackTrace(): Throwable = this
}

private class ResultBuilder(
    private val pageSeparator: String,
) {
    private val text = StringBuilder()
    private val entries = mutableListOf<OffsetMapEntry>()
    private val pages = mutableListOf<DocumentPage>()
    private val tokens = mutableListOf<OcrToken>()

    fun appendPage(
        page: Int,
        result: OcrResult,
        width: Int? = null,
        height: Int? = null,
    ) {
        appendSeparatorIfNeeded()

        val pageStart = text.length
        val tokenByPageOffset = result.tokenByPageOffset(documentStart = pageStart)
        result.text.forEachIndexed { pageOffset, char ->
            val documentOffset = text.length
            val token = tokenByPageOffset[pageOffset]
            entries += OffsetMapEntry(
                documentOffset = documentOffset,
                page = page,
                pageOffset = pageOffset,
                tokenStart = token?.start,
                tokenEnd = token?.end,
                bbox = token?.bbox,
            )
            text.append(char)
        }

        tokens += result.tokens.map { token ->
            token.copy(
                start = pageStart + token.start,
                end = pageStart + token.end,
                page = page,
            )
        }
        pages += DocumentPage(
            page = page,
            start = pageStart,
            end = text.length,
            width = width,
            height = height,
        )
    }

    fun build(inputKind: DocumentInputKind, evidence: DocumentEvidence): DocumentIntakeResult =
        DocumentIntakeResult(
            text = text.toString(),
            offsetMap = OffsetMap(entries.toList()),
            pages = pages.toList(),
            tokens = tokens.toList(),
            inputKind = inputKind,
            metadata = mapOf(
                "input_kind" to inputKind.name.lowercase(Locale.US),
                "page_count" to pages.size.toString(),
                "outcome" to evidence.outcome.code,
            ),
            evidence = evidence,
        )

    private fun appendSeparatorIfNeeded() {
        if (pages.isEmpty() || pageSeparator.isEmpty()) {
            return
        }

        val previousPage = pages.last()
        pageSeparator.forEachIndexed { separatorOffset, char ->
            val documentOffset = text.length
            entries += OffsetMapEntry(
                documentOffset = documentOffset,
                page = previousPage.page,
                pageOffset = (previousPage.end - previousPage.start) + separatorOffset,
                source = OffsetSource.PAGE_SEPARATOR,
            )
            text.append(char)
        }
    }
}

private fun OcrResult.tokenByPageOffset(documentStart: Int): Map<Int, OcrToken> {
    val tokenByPageOffset = mutableMapOf<Int, OcrToken>()
    tokens.forEach { token ->
        val documentToken = token.copy(
            start = documentStart + token.start,
            end = documentStart + token.end,
        )
        for (offset in token.start until token.end) {
            tokenByPageOffset[offset] = documentToken
        }
    }
    return tokenByPageOffset
}
