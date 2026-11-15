package com.openmed.openmedkit.intake

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrBoundingBox
import com.openmed.openmedkit.ocr.OcrImage
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
import com.openmed.openmedkit.ocr.SyntheticImages
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class DocumentIntakeTest {
    private lateinit var context: Context

    @Before
    fun setUp() {
        context = RuntimeEnvironment.getApplication()
    }

    @Test
    fun extractsPdfPagesInOrderWithContinuousOffsets() = runBlocking {
        val samplePdf = resourceFile("sample_multipage.pdf")
        val adapter = QueueOcrAdapter(
            listOf(
                ocrResult(
                    text = "Page one\nAlpha",
                    tokens = listOf(
                        token("Page", 0, 4, page = 0),
                        token("one", 5, 8, page = 0),
                        token("Alpha", 9, 14, page = 0),
                    ),
                ),
                ocrResult(
                    text = "Page two\nBeta",
                    tokens = listOf(
                        token("Page", 0, 4, page = 0),
                        token("two", 5, 8, page = 0),
                        token("Beta", 9, 13, page = 0),
                    ),
                ),
            ),
        )
        val pageSource = FixturePdfPageSource()
        val intake = DocumentIntake(
            context = context,
            ocrAdapter = adapter,
            pdfPageSource = pageSource,
        )

        val result = intake.extract(Uri.fromFile(samplePdf))

        assertEquals(DocumentInputKind.PDF, result.inputKind)
        assertEquals("Page one\nAlpha\n\nPage two\nBeta", result.text)
        assertEquals(2, result.pages.size)
        assertEquals(0, result.pages[0].start)
        assertEquals(14, result.pages[0].end)
        assertEquals(16, result.pages[1].start)
        assertEquals(result.text.length, result.pages[1].end)
        assertEquals(result.text.length, result.offsetMap.size)
        assertEquals(listOf(0, 1), pageSource.pagesRendered)
        assertEquals(Uri.fromFile(samplePdf), pageSource.uri)

        val betaStart = result.text.indexOf("Beta")
        assertEquals(setOf(1), result.offsetMap.pagesForSpan(betaStart, betaStart + 4))
        assertEquals(1, result.offsetMap[betaStart].page)
        assertEquals(9, result.offsetMap[betaStart].pageOffset)
        assertEquals(betaStart, result.offsetMap[betaStart].tokenStart)
        assertEquals(betaStart + 4, result.offsetMap[betaStart].tokenEnd)
        assertEquals(OffsetSource.PAGE_SEPARATOR, result.offsetMap[14].source)
        assertEquals(0, result.offsetMap[14].page)

        result.tokens.forEach { extractedToken ->
            assertEquals(
                extractedToken.text,
                result.text.substring(extractedToken.start, extractedToken.end),
            )
        }
        assertEquals(listOf(0, 0, 0, 1, 1, 1), result.tokens.map { it.page })
        assertFalse(pageSource.bitmaps.any { !it.isRecycled })
        assertEquals(
            pageSource.bitmaps,
            adapter.calls.map { (it as OcrImage.BitmapImage).bitmap },
        )
        assertTrue(adapter.calls.all { it.rotationDegrees == 0 })
        assertEquals(listOf(16, 16), result.pages.map { it.width })
    }

    @Test
    fun extractsImageInputAsSinglePageDocument() = runBlocking {
        val sampleImage = resourceFile("sample_document_image.png")
        val adapter = QueueOcrAdapter(
            listOf(
                ocrResult(
                    text = "Synthetic image",
                    tokens = listOf(
                        token("Synthetic", 0, 9, page = 0),
                        token("image", 10, 15, page = 0),
                    ),
                ),
            ),
        )
        val intake = DocumentIntake(context = context, ocrAdapter = adapter)

        val result = intake.extract(Uri.fromFile(sampleImage))

        assertEquals(DocumentInputKind.IMAGE, result.inputKind)
        assertEquals("Synthetic image", result.text)
        assertEquals(1, adapter.calls.size)
        assertEquals(1, result.pages.size)
        assertEquals(0, result.pages.single().start)
        assertEquals(result.text.length, result.pages.single().end)
        assertEquals(setOf(0), result.offsetMap.pagesForSpan(0, result.text.length))
        assertTrue(result.offsetMap.entries.all { it.source == OffsetSource.OCR_TEXT })
        val image = adapter.calls.single() as OcrImage.BitmapImage
        assertEquals(0, image.rotationDegrees)
        assertTrue("decoded image bitmap is recycled after OCR", image.bitmap.isRecycled)
        assertTrue(requireNotNull(result.pages.single().width) > 0)
        assertTrue(requireNotNull(result.pages.single().height) > 0)
    }

    @Test
    fun appliesExifRotationAndReportsUprightPageSize() = runBlocking {
        val rotatedJpeg = File(context.cacheDir, "synthetic_rotated.jpg").apply {
            writeBytes(jpegWithExifOrientation(width = 40, height = 20, orientation = 6))
        }
        val adapter = QueueOcrAdapter(
            listOf(
                ocrResult(
                    text = "Rotated",
                    tokens = listOf(token("Rotated", 0, 7, page = 0)),
                ),
            ),
        )

        try {
            val result = DocumentIntake(context = context, ocrAdapter = adapter)
                .extract(Uri.fromFile(rotatedJpeg))

            val image = adapter.calls.single() as OcrImage.BitmapImage
            assertEquals(90, image.rotationDegrees)
            assertEquals(40 to 20, image.width to image.height)
            assertEquals(20, result.pages.single().width)
            assertEquals(40, result.pages.single().height)
        } finally {
            rotatedJpeg.delete()
        }
    }

    @Test
    fun usesInjectedImageSourceAndRecyclesItsBitmap() = runBlocking {
        val bitmap = Bitmap.createBitmap(24, 12, Bitmap.Config.ARGB_8888)
        val adapter = QueueOcrAdapter(
            listOf(ocrResult(text = "", tokens = emptyList())),
        )
        var loadedUri: Uri? = null
        val intake = DocumentIntake(
            context = context,
            ocrAdapter = adapter,
            imageSource = DocumentImageSource { _, uri ->
                loadedUri = uri
                OcrImage.fromBitmap(bitmap, rotationDegrees = 270)
            },
        )
        val uri = Uri.parse("content://synthetic.documents/page.png")

        val result = intake.extract(uri)

        assertEquals(uri, loadedUri)
        assertEquals(270, adapter.calls.single().rotationDegrees)
        assertEquals(12, result.pages.single().width)
        assertEquals(24, result.pages.single().height)
        assertTrue(bitmap.isRecycled)
    }

    @Test
    fun exifRotationIgnoresUnreadableMetadata() {
        assertEquals(0, exifRotationDegrees("not an image".byteInputStream()))
        val expected = listOf(1 to 0, 3 to 180, 6 to 90, 8 to 270, 5 to 270, 7 to 90, 4 to 180)
        expected.forEach { (orientation, degrees) ->
            val jpeg = jpegWithExifOrientation(width = 2, height = 2, orientation = orientation)
            assertEquals(degrees, exifRotationDegrees(jpeg.inputStream()))
        }
    }

    private fun resourceFile(name: String): File =
        File(requireNotNull(javaClass.classLoader?.getResource(name)).toURI())

    /**
     * Writes a blank synthetic JPEG with a big-endian EXIF APP1 segment that
     * carries only the orientation tag. A zero-filled comment segment pads the
     * file past the 5000-byte signature window API 26 `ExifInterface` reads
     * before it parses any marker.
     */
    private fun jpegWithExifOrientation(width: Int, height: Int, orientation: Int): ByteArray {
        val jpeg = SyntheticImages.encode(width = width, height = height, format = "jpg")
        val tiff = byteArrayOf(
            0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08,
            0x00, 0x01,
            0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
            0x00, orientation.toByte(), 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        )
        val exifHeader = "Exif".toByteArray() + byteArrayOf(0x00, 0x00)
        val segmentLength = 2 + exifHeader.size + tiff.size
        val app1 = byteArrayOf(
            0xFF.toByte(),
            0xE1.toByte(),
            (segmentLength shr 8).toByte(),
            (segmentLength and 0xFF).toByte(),
        ) + exifHeader + tiff
        val paddingLength = 2 + EXIF_SIGNATURE_WINDOW_BYTES
        val comment = byteArrayOf(
            0xFF.toByte(),
            0xFE.toByte(),
            (paddingLength shr 8).toByte(),
            (paddingLength and 0xFF).toByte(),
        ) + ByteArray(EXIF_SIGNATURE_WINDOW_BYTES)
        return jpeg.copyOfRange(0, 2) + app1 + comment + jpeg.copyOfRange(2, jpeg.size)
    }

    private fun ocrResult(text: String, tokens: List<OcrToken>): OcrResult =
        OcrResult(text = text, words = tokens, metadata = mapOf("engine" to "fixture"))

    private fun token(text: String, start: Int, end: Int, page: Int): OcrToken =
        OcrToken(
            text = text,
            start = start,
            end = end,
            confidence = 0.99f,
            bbox = OcrBoundingBox(start, page * 100, end, page * 100 + 10),
            page = page,
        )

    private companion object {
        const val EXIF_SIGNATURE_WINDOW_BYTES = 5000
    }

    private class QueueOcrAdapter(
        private val results: List<OcrResult>,
    ) : OcrAdapter {
        val calls: List<OcrImage>
            get() = _calls.toList()

        private val _calls = mutableListOf<OcrImage>()
        private var index = 0

        override suspend fun recognize(image: OcrImage): OcrResult {
            _calls += image
            return results[index++]
        }
    }

    private class FixturePdfPageSource : PdfPageSource {
        val pagesRendered = mutableListOf<Int>()
        val bitmaps = mutableListOf<Bitmap>()
        var uri: Uri? = null

        override suspend fun renderEach(
            context: Context,
            uri: Uri,
            block: suspend (RenderedPdfPage) -> Unit,
        ) {
            this.uri = uri
            for (page in 0..1) {
                val bitmap = Bitmap.createBitmap(16, 16, Bitmap.Config.ARGB_8888)
                bitmaps += bitmap
                pagesRendered += page
                try {
                    block(RenderedPdfPage(page = page, bitmap = bitmap))
                } finally {
                    bitmap.recycle()
                }
            }
        }
    }
}
