package com.openmed.openmedkit.export

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.openmed.openmedkit.intake.DocumentIntakeException
import com.openmed.openmedkit.intake.DocumentIntakeReason
import com.openmed.openmedkit.intake.DocumentPageGeometry
import com.openmed.openmedkit.intake.DocumentSpatialUnit
import com.openmed.openmedkit.intake.PdfPageEvent
import com.openmed.openmedkit.intake.PdfPageSource
import com.openmed.openmedkit.intake.RenderedPdfPage
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.Locale
import java.util.zip.DataFormatException
import java.util.zip.Deflater
import java.util.zip.Inflater

/**
 * Writer and strict reader for the image-only PDF export profile shared with
 * Swift OpenMedKit.
 *
 * Every page is one Flate-compressed 8-bit RGB image drawn over the whole media
 * box by a fixed content stream. There is no text, font, annotation,
 * attachment, form, action, outline, document information or XMP metadata. The
 * reader accepts exactly the bytes this writer produces and nothing else.
 */
internal object DocumentRedactionPdf {
    class Page(
        val pixelWidth: Int,
        val pixelHeight: Int,
        val mediaWidth: String,
        val mediaHeight: String,
        val stream: ByteArray,
    ) {
        override fun equals(other: Any?): Boolean =
            other is Page && other.pixelWidth == pixelWidth && other.pixelHeight == pixelHeight &&
                other.mediaWidth == mediaWidth && other.mediaHeight == mediaHeight &&
                other.stream.contentEquals(stream)

        override fun hashCode(): Int = listOf(pixelWidth, pixelHeight, mediaWidth, mediaHeight).hashCode()

        override fun toString(): String = "Page(pixelWidth=$pixelWidth, pixelHeight=$pixelHeight)"
    }

    private val HEADER = "%PDF-1.4\n%".toByteArray(Charsets.US_ASCII) +
        byteArrayOf(0xE2.toByte(), 0xE3.toByte(), 0xCF.toByte(), 0xD3.toByte(), 0x0A)

    /** Largest PDF page side, in default user-space units. */
    private const val MAX_MEDIA_SIDE = 14_400.0

    /**
     * Compresses one upright page of opaque ARGB pixels. [mediaWidth] and
     * [mediaHeight] are the physical page size in points.
     */
    fun page(argb: IntArray, pixelWidth: Int, pixelHeight: Int, mediaWidth: Double, mediaHeight: Double): Page {
        if (pixelWidth <= 0 || pixelHeight <= 0 || argb.size.toLong() != pixelWidth.toLong() * pixelHeight ||
            !mediaWidth.isFinite() || !mediaHeight.isFinite() || mediaWidth <= 0.0 || mediaHeight <= 0.0
        ) {
            DocumentRedactionExportContract.fail(DocumentRedactionExportReason.RENDER_FAILED)
        }
        val rgb = ByteArray(argb.size * 3)
        argb.forEachIndexed { index, pixel ->
            rgb[index * 3] = (pixel shr 16).toByte()
            rgb[index * 3 + 1] = (pixel shr 8).toByte()
            rgb[index * 3 + 2] = pixel.toByte()
        }
        val factor = minOf(1.0, MAX_MEDIA_SIDE / maxOf(mediaWidth, mediaHeight))
        return Page(pixelWidth, pixelHeight, number(mediaWidth * factor), number(mediaHeight * factor), zlib(rgb))
    }

    fun write(pages: List<Page>): ByteArray {
        val output = ByteArrayOutputStream()
        val offsets = mutableListOf(0)
        output.write(HEADER)
        fun objectAt(body: String, stream: ByteArray? = null) {
            offsets += output.size()
            output.write("${offsets.size - 1} 0 obj\n$body\n".toByteArray(Charsets.US_ASCII))
            if (stream != null) {
                output.write("stream\n".toByteArray(Charsets.US_ASCII))
                output.write(stream)
                output.write("\nendstream\n".toByteArray(Charsets.US_ASCII))
            }
            output.write("endobj\n".toByteArray(Charsets.US_ASCII))
        }
        objectAt("<< /Type /Catalog /Pages 2 0 R >>")
        val kids = pages.indices.joinToString(" ") { "${3 + 3 * it} 0 R" }
        objectAt("<< /Type /Pages /Kids [$kids] /Count ${pages.size} >>")
        pages.forEachIndexed { index, page ->
            val base = 3 + 3 * index
            val content = contentStream(page).toByteArray(Charsets.US_ASCII)
            objectAt(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${page.mediaWidth} ${page.mediaHeight}] " +
                    "/Resources << /XObject << /Im0 ${base + 1} 0 R >> >> /Contents ${base + 2} 0 R >>",
            )
            objectAt(
                "<< /Type /XObject /Subtype /Image /Width ${page.pixelWidth} /Height ${page.pixelHeight} " +
                    "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length ${page.stream.size} >>",
                page.stream,
            )
            objectAt("<< /Length ${content.size} >>", content)
        }
        val xref = output.size()
        val table = StringBuilder("xref\n0 ${offsets.size}\n0000000000 65535 f \n")
        offsets.drop(1).forEach { table.append(String.format(Locale.ROOT, "%010d 00000 n \n", it)) }
        table.append("trailer\n<< /Size ${offsets.size} /Root 1 0 R >>\nstartxref\n$xref\n%%EOF\n")
        output.write(table.toString().toByteArray(Charsets.US_ASCII))
        return output.toByteArray()
    }

    /**
     * Re-reads an artifact, accepting only the exact profile layout.
     *
     * @throws DocumentRedactionExportException with
     *   [DocumentRedactionExportReason.HIDDEN_TEXT],
     *   [DocumentRedactionExportReason.ATTACHMENTS_PRESENT] or
     *   [DocumentRedactionExportReason.METADATA_PRESENT] when a rejected file
     *   carries those structures, otherwise
     *   [DocumentRedactionExportReason.PROFILE_VIOLATION].
     */
    fun inspect(data: ByteArray, maxPixels: Long): List<Page> =
        try {
            read(data, maxPixels)
        } catch (_: Malformed) {
            throw DocumentRedactionExportException(classify(data))
        }

    /** Inflates one page into opaque ARGB pixels. */
    fun pixels(page: Page): IntArray {
        val count = page.pixelWidth * page.pixelHeight
        val rgb = ByteArray(count * 3)
        val inflater = Inflater()
        try {
            inflater.setInput(page.stream)
            var filled = 0
            while (filled < rgb.size) {
                val read = inflater.inflate(rgb, filled, rgb.size - filled)
                if (read == 0 && (inflater.finished() || inflater.needsInput() || inflater.needsDictionary())) {
                    break
                }
                filled += read
            }
            val spare = ByteArray(1)
            if (filled != rgb.size || inflater.inflate(spare) != 0 || !inflater.finished() || inflater.remaining != 0) {
                DocumentRedactionExportContract.fail(DocumentRedactionExportReason.PROFILE_VIOLATION)
            }
        } catch (_: DataFormatException) {
            DocumentRedactionExportContract.fail(DocumentRedactionExportReason.PROFILE_VIOLATION)
        } finally {
            inflater.end()
        }
        return IntArray(count) { index ->
            (0xFF shl 24) or ((rgb[index * 3].toInt() and 0xFF) shl 16) or
                ((rgb[index * 3 + 1].toInt() and 0xFF) shl 8) or (rgb[index * 3 + 2].toInt() and 0xFF)
        }
    }

    /** A positive decimal with at most four fractional digits and no exponent. */
    fun number(value: Double): String =
        String.format(Locale.ROOT, "%.4f", value).trimEnd('0').trimEnd('.')

    private fun zlib(data: ByteArray): ByteArray {
        val deflater = Deflater(Deflater.DEFAULT_COMPRESSION)
        try {
            deflater.setInput(data)
            deflater.finish()
            val output = ByteArrayOutputStream(maxOf(64, data.size / 8))
            val buffer = ByteArray(64 * 1024)
            while (!deflater.finished()) {
                output.write(buffer, 0, deflater.deflate(buffer))
            }
            return output.toByteArray()
        } finally {
            deflater.end()
        }
    }

    private fun contentStream(page: Page): String =
        "q ${page.mediaWidth} 0 0 ${page.mediaHeight} 0 0 cm /Im0 Do Q"

    private class Malformed : Exception() {
        override fun fillInStackTrace(): Throwable = this
    }

    private fun read(data: ByteArray, maxPixels: Long): List<Page> {
        val reader = Reader(data)
        val offsets = mutableListOf(0)
        reader.expect(HEADER)
        offsets += reader.position
        reader.expect("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        offsets += reader.position
        reader.expect("2 0 obj\n<< /Type /Pages /Kids [")
        val kids = mutableListOf<Int>()
        do {
            if (kids.isNotEmpty()) {
                reader.expect(" ")
            }
            kids += reader.integer()
            reader.expect(" 0 R")
        } while (reader.peek() == ' '.code)
        reader.expect("] /Count ")
        val count = reader.integer()
        reader.expect(" >>\nendobj\n")
        if (count < 1 || kids != (0 until count).map { 3 + 3 * it }) {
            throw Malformed()
        }
        val pages = mutableListOf<Page>()
        for (index in 0 until count) {
            val base = 3 + 3 * index
            offsets += reader.position
            reader.expect("$base 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ")
            val mediaWidth = reader.number()
            reader.expect(" ")
            val mediaHeight = reader.number()
            reader.expect(
                "] /Resources << /XObject << /Im0 ${base + 1} 0 R >> >> /Contents ${base + 2} 0 R >>\nendobj\n",
            )
            offsets += reader.position
            reader.expect("${base + 1} 0 obj\n<< /Type /XObject /Subtype /Image /Width ")
            val width = reader.integer()
            reader.expect(" /Height ")
            val height = reader.integer()
            reader.expect(" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length ")
            val length = reader.integer()
            val stream = reader.stream(length)
            if (width < 1 || height < 1 || width.toLong() * height > maxPixels) {
                throw Malformed()
            }
            val page = Page(width, height, mediaWidth, mediaHeight, stream)
            offsets += reader.position
            reader.expect("${base + 2} 0 obj\n<< /Length ")
            val contentLength = reader.integer()
            if (!reader.stream(contentLength).contentEquals(contentStream(page).toByteArray(Charsets.US_ASCII))) {
                throw Malformed()
            }
            pages += page
        }
        val xref = reader.position
        reader.expect("xref\n0 ${offsets.size}\n0000000000 65535 f \n")
        offsets.drop(1).forEach { reader.expect(String.format(Locale.ROOT, "%010d 00000 n \n", it)) }
        reader.expect("trailer\n<< /Size ${offsets.size} /Root 1 0 R >>\nstartxref\n$xref\n%%EOF\n")
        if (!reader.atEnd) {
            throw Malformed()
        }
        return pages
    }

    private fun classify(data: ByteArray): DocumentRedactionExportReason {
        val text = String(data, Charsets.ISO_8859_1)
        val markers = listOf(
            DocumentRedactionExportReason.HIDDEN_TEXT to listOf("/Font", "/ToUnicode", "/ActualText"),
            DocumentRedactionExportReason.ATTACHMENTS_PRESENT to listOf(
                "/Annots", "/EmbeddedFile", "/FileAttachment", "/AcroForm", "/JavaScript", "/OpenAction",
                "/AA ", "/Launch", "/URI",
            ),
            DocumentRedactionExportReason.METADATA_PRESENT to listOf(
                "/Info", "/Metadata", "/Title", "/Author", "/Producer", "/Creator", "/CreationDate", "/ModDate",
            ),
        )
        return markers.firstOrNull { (_, needles) -> needles.any(text::contains) }?.first
            ?: DocumentRedactionExportReason.PROFILE_VIOLATION
    }

    private class Reader(private val bytes: ByteArray) {
        var position = 0
            private set

        val atEnd: Boolean
            get() = position == bytes.size

        fun peek(): Int? = if (position < bytes.size) bytes[position].toInt() and 0xFF else null

        fun expect(text: String) = expect(text.toByteArray(Charsets.US_ASCII))

        fun expect(expected: ByteArray) {
            if (position + expected.size > bytes.size) {
                throw Malformed()
            }
            for (offset in expected.indices) {
                if (bytes[position + offset] != expected[offset]) {
                    throw Malformed()
                }
            }
            position += expected.size
        }

        /** A non-negative decimal integer of at most nine digits, without leading zeros. */
        fun integer(): Int {
            val start = position
            while (peek()?.let { it in DIGITS } == true && position - start < 9) {
                position += 1
            }
            if (position == start || (bytes[start] == '0'.code.toByte() && position != start + 1) ||
                peek()?.let { it in DIGITS } != false
            ) {
                throw Malformed()
            }
            return String(bytes, start, position - start, Charsets.US_ASCII).toInt()
        }

        /** A positive decimal with at most four fractional digits, as written. */
        fun number(): String {
            val start = position
            integer()
            if (peek() == '.'.code) {
                position += 1
                val fraction = position
                while (peek()?.let { it in DIGITS } == true && position - fraction < 4) {
                    position += 1
                }
                if (position == fraction || bytes[position - 1] == '0'.code.toByte()) {
                    throw Malformed()
                }
            }
            val text = String(bytes, start, position - start, Charsets.US_ASCII)
            if (!(text.toDouble() > 0.0)) {
                throw Malformed()
            }
            return text
        }

        fun stream(length: Int): ByteArray {
            expect(" >>\nstream\n")
            if (length > bytes.size - position) {
                throw Malformed()
            }
            val body = bytes.copyOfRange(position, position + length)
            position += length
            expect("\nendstream\nendobj\n")
            return body
        }

        private companion object {
            val DIGITS = '0'.code..'9'.code
        }
    }
}

/**
 * Reads exported artifacts back for verification, accepting only the
 * image-only export profile.
 *
 * Pages are decoded into transient bitmaps in pixel geometry and recycled
 * after each callback. Pass `PdfPageRenderer()` as the exporter's verification
 * source instead to have the platform renderer read the artifact independently.
 */
class DocumentRedactionPdfPageSource(
    private val maxPixels: Long = DEFAULT_MAX_PIXELS,
) : PdfPageSource {
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
        val bytes = try {
            context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
        } catch (_: SecurityException) {
            null
        } catch (_: IOException) {
            null
        } ?: throw DocumentIntakeException(DocumentIntakeReason.PERMISSION_DENIED)
        val pages = try {
            DocumentRedactionPdf.inspect(bytes, maxPixels)
        } catch (_: DocumentRedactionExportException) {
            throw DocumentIntakeException(DocumentIntakeReason.UNSUPPORTED_DOCUMENT)
        }
        pages.forEachIndexed { index, page ->
            val pixels = try {
                DocumentRedactionPdf.pixels(page)
            } catch (_: DocumentRedactionExportException) {
                null
            }
            if (pixels == null) {
                block(PdfPageEvent.Unavailable(index, DocumentIntakeReason.PAGE_MISSING))
                return@forEachIndexed
            }
            val bitmap = Bitmap.createBitmap(pixels, page.pixelWidth, page.pixelHeight, Bitmap.Config.ARGB_8888)
            try {
                block(
                    PdfPageEvent.Rendered(
                        RenderedPdfPage(
                            page = index,
                            bitmap = bitmap,
                            geometry = DocumentPageGeometry(
                                width = page.pixelWidth.toDouble(),
                                height = page.pixelHeight.toDouble(),
                                unit = DocumentSpatialUnit.PIXEL,
                            ),
                        ),
                    ),
                )
            } finally {
                bitmap.recycle()
            }
        }
    }

    private companion object {
        const val DEFAULT_MAX_PIXELS = 40_000_000L
    }
}
