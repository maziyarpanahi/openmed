package com.openmed.openmedkit.intake

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import java.io.IOException
import kotlin.math.roundToInt

/**
 * A rendered PDF page ready for OCR.
 *
 * @param page Zero-based page index.
 * @param bitmap Upright rendering of the page, or of its declared crop.
 * @param geometry Original page geometry for evidence coordinates. When null,
 *   the bitmap itself is treated as the original page, in pixels.
 */
data class RenderedPdfPage(
    val page: Int,
    val bitmap: Bitmap,
    val geometry: DocumentPageGeometry? = null,
)

/**
 * One declared PDF page: rendered, or unavailable with a value-free reason.
 */
sealed class PdfPageEvent {
    /** A rendered page; the source recycles the bitmap after the callback. */
    data class Rendered(val page: RenderedPdfPage) : PdfPageEvent()

    /**
     * A declared page that could not be read, for example
     * [DocumentIntakeReason.PAGE_MISSING] or [DocumentIntakeReason.PERMISSION_LOST].
     */
    data class Unavailable(
        val page: Int,
        val reason: DocumentIntakeReason,
    ) : PdfPageEvent()
}

/**
 * Source of rendered PDF page bitmaps.
 */
fun interface PdfPageSource {
    suspend fun renderEach(
        context: Context,
        uri: Uri,
        block: suspend (RenderedPdfPage) -> Unit,
    )

    /**
     * Renders every declared page in order, reporting unreadable pages instead
     * of skipping them.
     *
     * Throw [DocumentIntakeException] for document-level failures such as
     * [DocumentIntakeReason.PERMISSION_DENIED]. The default adapts [renderEach].
     */
    suspend fun renderPages(
        context: Context,
        uri: Uri,
        block: suspend (PdfPageEvent) -> Unit,
    ) {
        renderEach(context, uri) { page -> block(PdfPageEvent.Rendered(page)) }
    }
}

/**
 * Renders PDF pages on-device with [PdfRenderer].
 *
 * [PdfRenderer] exposes only the displayed page size, so evidence from this
 * source is in displayed page points with no separate crop or rotation. A
 * [PdfPageSource] that reads the media box, crop box and `/Rotate` can declare
 * them through [RenderedPdfPage.geometry]. Bitmaps stay in memory and are
 * recycled after each callback; nothing is written to storage.
 */
class PdfPageRenderer(
    private val scale: Float = 2.0f,
) : PdfPageSource {
    init {
        require(scale > 0.0f) { "scale must be positive" }
    }

    override suspend fun renderEach(
        context: Context,
        uri: Uri,
        block: suspend (RenderedPdfPage) -> Unit,
    ) {
        renderPages(context, uri) { event ->
            when (event) {
                is PdfPageEvent.Rendered -> block(event.page)
                is PdfPageEvent.Unavailable -> throw DocumentIntakeException(event.reason)
            }
        }
    }

    override suspend fun renderPages(
        context: Context,
        uri: Uri,
        block: suspend (PdfPageEvent) -> Unit,
    ) {
        val descriptor = try {
            context.contentResolver.openFileDescriptor(uri, "r")
        } catch (_: SecurityException) {
            null
        } catch (_: IOException) {
            null
        } ?: throw DocumentIntakeException(DocumentIntakeReason.PERMISSION_DENIED)
        descriptor.use { fileDescriptor ->
            val renderer = try {
                PdfRenderer(fileDescriptor)
            } catch (_: SecurityException) {
                throw DocumentIntakeException(DocumentIntakeReason.DOCUMENT_LOCKED)
            } catch (_: IOException) {
                throw DocumentIntakeException(DocumentIntakeReason.UNSUPPORTED_DOCUMENT)
            }
            try {
                for (pageIndex in 0 until renderer.pageCount) {
                    val rendered = try {
                        render(renderer, pageIndex)
                    } catch (_: IllegalStateException) {
                        null
                    } catch (_: IllegalArgumentException) {
                        null
                    }
                    if (rendered == null) {
                        block(PdfPageEvent.Unavailable(pageIndex, DocumentIntakeReason.PAGE_MISSING))
                        continue
                    }
                    try {
                        block(PdfPageEvent.Rendered(rendered))
                    } finally {
                        rendered.bitmap.recycle()
                    }
                }
            } finally {
                renderer.close()
            }
        }
    }

    private fun render(renderer: PdfRenderer, pageIndex: Int): RenderedPdfPage {
        val page = renderer.openPage(pageIndex)
        try {
            val bitmap = Bitmap.createBitmap(
                (page.width * scale).roundToInt().coerceAtLeast(1),
                (page.height * scale).roundToInt().coerceAtLeast(1),
                Bitmap.Config.ARGB_8888,
            )
            try {
                bitmap.eraseColor(Color.WHITE)
                page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
            } catch (error: RuntimeException) {
                bitmap.recycle()
                throw error
            }
            return RenderedPdfPage(
                page = pageIndex,
                bitmap = bitmap,
                geometry = DocumentPageGeometry(
                    width = page.width.toDouble(),
                    height = page.height.toDouble(),
                    unit = DocumentSpatialUnit.POINT,
                ),
            )
        } finally {
            page.close()
        }
    }
}
