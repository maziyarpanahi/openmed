package com.openmed.openmedkit.intake

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.openmed.openmedkit.initializeMlKitForTests
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrBoundingBox
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
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
        initializeMlKitForTests(context)
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
    }

    private fun resourceFile(name: String): File =
        File(requireNotNull(javaClass.classLoader?.getResource(name)).toURI())

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

    private class QueueOcrAdapter(
        private val results: List<OcrResult>,
    ) : OcrAdapter {
        val calls: List<InputImage>
            get() = _calls.toList()

        private val _calls = mutableListOf<InputImage>()
        private var index = 0

        override suspend fun recognize(image: InputImage): OcrResult {
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
