package com.openmed.openmedkit.intake

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import kotlin.math.roundToInt

/**
 * A rendered PDF page ready for OCR.
 */
data class RenderedPdfPage(
    val page: Int,
    val bitmap: Bitmap,
)

/**
 * Source of rendered PDF page bitmaps.
 */
fun interface PdfPageSource {
    suspend fun renderEach(
        context: Context,
        uri: Uri,
        block: suspend (RenderedPdfPage) -> Unit,
    )
}

/**
 * Renders PDF pages on-device with [PdfRenderer].
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
        val descriptor = requireNotNull(context.contentResolver.openFileDescriptor(uri, "r")) {
            "Unable to open PDF document"
        }
        descriptor.use { fileDescriptor ->
            val renderer = PdfRenderer(fileDescriptor)
            try {
                for (pageIndex in 0 until renderer.pageCount) {
                    val page = renderer.openPage(pageIndex)
                    try {
                        val bitmap = Bitmap.createBitmap(
                            (page.width * scale).roundToInt().coerceAtLeast(1),
                            (page.height * scale).roundToInt().coerceAtLeast(1),
                            Bitmap.Config.ARGB_8888,
                        )
                        bitmap.eraseColor(Color.WHITE)
                        page.render(
                            bitmap,
                            null,
                            null,
                            PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY,
                        )
                        try {
                            block(RenderedPdfPage(page = pageIndex, bitmap = bitmap))
                        } finally {
                            bitmap.recycle()
                        }
                    } finally {
                        page.close()
                    }
                }
            } finally {
                renderer.close()
            }
        }
    }
}
