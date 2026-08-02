package com.openmed.openmedkit.intake

import android.content.Context
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.openmed.openmedkit.ocr.MlKitOcrAdapter
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
import java.util.Locale

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
 */
data class DocumentIntakeResult(
    val text: String,
    val offsetMap: OffsetMap,
    val pages: List<DocumentPage>,
    val tokens: List<OcrToken>,
    val inputKind: DocumentInputKind,
    val metadata: Map<String, String> = emptyMap(),
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
    }
}

/**
 * On-device document intake for Android OpenMedKit.
 */
class DocumentIntake(
    private val context: Context,
    private val ocrAdapter: OcrAdapter = MlKitOcrAdapter(),
    private val pdfPageSource: PdfPageSource = PdfPageRenderer(),
    private val pageSeparator: String = "\n\n",
) {
    suspend fun extract(uri: Uri): DocumentIntakeResult =
        when (detectInputKind(uri)) {
            DocumentInputKind.PDF -> extractPdf(uri)
            DocumentInputKind.IMAGE -> extractImage(uri)
        }

    private suspend fun extractPdf(uri: Uri): DocumentIntakeResult {
        val builder = ResultBuilder(pageSeparator = pageSeparator)
        pdfPageSource.renderEach(context, uri) { renderedPage ->
            val image = InputImage.fromBitmap(renderedPage.bitmap, 0)
            val result = ocrAdapter.recognize(image)
            builder.appendPage(
                page = renderedPage.page,
                result = result,
                width = renderedPage.bitmap.width,
                height = renderedPage.bitmap.height,
            )
        }
        return builder.build(inputKind = DocumentInputKind.PDF)
    }

    private suspend fun extractImage(uri: Uri): DocumentIntakeResult {
        val image = InputImage.fromFilePath(context, uri)
        val result = ocrAdapter.recognize(image)
        return ResultBuilder(pageSeparator = pageSeparator)
            .apply {
                appendPage(
                    page = 0,
                    result = result,
                    width = image.width,
                    height = image.height,
                )
            }
            .build(inputKind = DocumentInputKind.IMAGE)
    }

    private fun detectInputKind(uri: Uri): DocumentInputKind {
        val mimeType = context.contentResolver.getType(uri)?.lowercase(Locale.US)
        val extension = uri.path
            ?.substringAfterLast('.', missingDelimiterValue = "")
            ?.lowercase(Locale.US)

        return when {
            mimeType == "application/pdf" || extension == "pdf" -> DocumentInputKind.PDF
            mimeType?.startsWith("image/") == true || extension in IMAGE_EXTENSIONS ->
                DocumentInputKind.IMAGE
            else -> error("Unsupported document input type")
        }
    }
}

private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "webp", "heic", "heif")

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

    fun build(inputKind: DocumentInputKind): DocumentIntakeResult =
        DocumentIntakeResult(
            text = text.toString(),
            offsetMap = OffsetMap(entries.toList()),
            pages = pages.toList(),
            tokens = tokens.toList(),
            inputKind = inputKind,
            metadata = mapOf(
                "input_kind" to inputKind.name.lowercase(Locale.US),
                "page_count" to pages.size.toString(),
            ),
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
