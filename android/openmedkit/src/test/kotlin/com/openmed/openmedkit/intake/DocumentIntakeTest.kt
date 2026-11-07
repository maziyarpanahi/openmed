package com.openmed.openmedkit.intake

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrBoundingBox
import com.openmed.openmedkit.ocr.OcrImage
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
import com.openmed.openmedkit.ocr.FakeOcrAdapter
import com.openmed.openmedkit.ocr.SyntheticImages
import java.io.File
import java.security.MessageDigest
import kotlin.test.assertFailsWith
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
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
                        token("Synthetic", 0, 9, page = 0, bbox = OcrBoundingBox(0, 0, 1, 2)),
                        token("image", 10, 15, page = 0, bbox = OcrBoundingBox(1, 0, 2, 2)),
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
            // Evidence stays in stored pixels; the upright OCR box maps back.
            val evidence = requireNotNull(result.evidence)
            val page = requireNotNull(evidence.pages.single().evidence)
            assertEquals(listOf(40.0, 20.0), listOf(page.width, page.height))
            assertEquals(90, page.rotationDegrees)
            assertEquals(DocumentBox(0.0, 13.0, 10.0, 20.0), evidence.regions.single().sourceBox)
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
        val uri = Uri.fromFile(resourceFile("sample_document_image.png"))

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

    @Test
    fun extractAttachesEvidenceThatMapsSpansBackToPages() = runBlocking {
        val samplePdf = resourceFile("sample_multipage.pdf")
        val adapter = QueueOcrAdapter(
            listOf(
                ocrResult(text = "Page one", tokens = listOf(token("Page", 0, 4, page = 0))),
                ocrResult(text = "Beta", tokens = listOf(token("Beta", 0, 4, page = 0))),
            ),
        )
        val result = DocumentIntake(context, adapter, pdfPageSource = FixturePdfPageSource())
            .extract(Uri.fromFile(samplePdf))

        val evidence = requireNotNull(result.evidence)
        assertEquals(DocumentIntakeOutcome.COMPLETE, evidence.outcome)
        assertEquals(result.text, evidence.text)
        assertEquals(sha256(samplePdf.readBytes()), evidence.sourceDigest)
        val betaStart = result.text.indexOf("Beta")
        val region = evidence.regionsForUtf16Span(betaStart, betaStart + 4).single()
        assertEquals(1, region.page)
        assertEquals(DocumentBox(0.0, 0.0, 4.0, 10.0), region.sourceBox)
        assertEquals(DocumentSpatialUnit.PIXEL, region.evidence.unit)
        assertEquals("complete", result.metadata["outcome"])
    }

    @Test
    fun croppedRotatedImageEvidenceUsesOriginalPixels() = runBlocking {
        // A 30x10 cut from (10, 20)-(40, 30) of a 100x50 original, turned 90 degrees for OCR.
        val bitmap = Bitmap.createBitmap(30, 10, Bitmap.Config.ARGB_8888)
        val source = object : DocumentImageSource {
            override suspend fun load(context: Context, uri: Uri) =
                OcrImage.fromBitmap(bitmap, rotationDegrees = 90)

            override suspend fun loadPage(context: Context, uri: Uri) =
                DocumentImagePage(
                    image = load(context, uri),
                    geometry = DocumentPageGeometry(
                        width = 100.0,
                        height = 50.0,
                        rotationDegrees = 90,
                        crop = DocumentBox(10.0, 20.0, 40.0, 30.0),
                    ),
                )
        }
        val adapter = QueueOcrAdapter(
            listOf(
                ocrResult(
                    text = "Word",
                    tokens = listOf(token("Word", 0, 4, page = 0, bbox = OcrBoundingBox(2, 4, 8, 10))),
                ),
            ),
        )

        val report = DocumentIntake(context, adapter, imageSource = source)
            .intake(Uri.fromFile(resourceFile("sample_document_image.png")))

        assertEquals(DocumentIntakeOutcome.COMPLETE, report.outcome)
        val region = report.evidence.regions.single()
        assertEquals(DocumentBox(14.0, 22.0, 20.0, 28.0), region.sourceBox)
        assertEquals(DocumentBox(22.0, 14.0, 28.0, 20.0), region.evidence.displayBox)
        assertEquals(
            DocumentBox(10.0, 20.0, 40.0, 30.0),
            report.evidence.pages.single().cropEvidence?.sourceBox,
        )
        assertTrue(bitmap.isRecycled)
    }

    @Test
    fun missingPagesAndPermissionLossAreExplicitPartialOutcomes() = runBlocking {
        val uri = Uri.fromFile(resourceFile("sample_multipage.pdf"))
        val source = ScriptedPdfPageSource(
            listOf(
                Step.Page(0),
                Step.Missing(1),
                Step.Page(2),
                Step.Fail(SecurityException("synthetic-private-path")),
                Step.Page(4),
            ),
        )
        val adapter = QueueOcrAdapter(
            listOf(
                ocrResult(text = "One", tokens = listOf(token("One", 0, 3, page = 0))),
                ocrResult(text = "Three", tokens = listOf(token("Three", 0, 5, page = 0))),
            ),
        )
        val intake = DocumentIntake(context, adapter, pdfPageSource = source)

        val report = intake.intake(uri)

        assertEquals(DocumentIntakeOutcome.PARTIAL, report.outcome)
        assertEquals(
            listOf(DocumentIntakeReason.PAGE_MISSING, DocumentIntakeReason.PERMISSION_LOST),
            report.reasons,
        )
        assertEquals(
            listOf(
                DocumentPageStatus.MAPPED,
                DocumentPageStatus.HELD,
                DocumentPageStatus.MAPPED,
                DocumentPageStatus.HELD,
            ),
            report.evidence.pages.map { it.status },
        )
        assertEquals("One\n\nThree", report.result?.text)
        assertEquals(listOf(0, 2), report.result?.pages?.map { it.page })
        assertEquals(listOf(0, 2), source.requested)
        assertTrue(source.closed)
        assertTrue(source.bitmaps.all { it.isRecycled })
        assertFalse(report.toString().contains("synthetic-private-path"))

        val error = assertFailsWith<DocumentIntakeException> {
            DocumentIntake(context, FakeOcrAdapter(), pdfPageSource = ScriptedPdfPageSource(source.script))
                .extract(uri)
        }
        assertEquals(DocumentIntakeOutcome.PARTIAL, error.outcome)
        assertEquals("page_missing,permission_lost", error.message)
    }

    @Test
    fun documentLevelFailuresAreRefusedWithValueFreeReasons() = runBlocking {
        val uri = Uri.fromFile(resourceFile("sample_multipage.pdf"))
        fun intake(vararg steps: Step) =
            DocumentIntake(
                context = context,
                ocrAdapter = FakeOcrAdapter(),
                pdfPageSource = ScriptedPdfPageSource(steps.toList()),
            )

        val denied = intake(Step.Fail(SecurityException("synthetic-private-path"))).intake(uri)
        assertEquals(DocumentIntakeOutcome.REFUSED, denied.outcome)
        assertEquals(listOf(DocumentIntakeReason.PERMISSION_DENIED), denied.reasons)
        assertNull(denied.result)
        assertFalse(denied.toString().contains("synthetic-private-path"))
        val locked = intake(Step.Fail(DocumentIntakeException(DocumentIntakeReason.DOCUMENT_LOCKED)))
        assertEquals(listOf(DocumentIntakeReason.DOCUMENT_LOCKED), locked.intake(uri).reasons)
        assertEquals(listOf(DocumentIntakeReason.EMPTY_DOCUMENT), intake().intake(uri).reasons)

        val tooLarge = ScriptedPdfPageSource(listOf(Step.Page(0)))
        val limited = DocumentIntake(
            context = context,
            ocrAdapter = FakeOcrAdapter(),
            pdfPageSource = tooLarge,
            limits = DocumentIntakeLimits(maxDocumentBytes = 8),
        ).intake(uri)
        assertEquals(listOf(DocumentIntakeReason.SIZE_LIMIT), limited.reasons)
        assertTrue(tooLarge.requested.isEmpty())

        val missing = Uri.fromFile(File(context.cacheDir, "missing-synthetic.pdf"))
        assertEquals(
            listOf(DocumentIntakeReason.PERMISSION_DENIED),
            intake(Step.Page(0)).intake(missing).reasons,
        )
        val text = Uri.fromFile(File(context.cacheDir, "synthetic.txt"))
        val error = assertFailsWith<DocumentIntakeException> { intake(Step.Page(0)).extract(text) }
        assertEquals("unsupported_document", error.message)
        assertEquals(DocumentIntakeReason.UNSUPPORTED_DOCUMENT, error.reason)
    }

    @Test
    fun pageLimitAndRecognitionFailuresHoldPages() = runBlocking {
        val uri = Uri.fromFile(resourceFile("sample_multipage.pdf"))
        val source = ScriptedPdfPageSource((0..3).map { Step.Page(it) })
        var calls = 0
        val adapter = OcrAdapter {
            if (calls++ == 1) {
                throw IllegalStateException("synthetic-engine-detail")
            }
            ocrResult(text = "Ok", tokens = listOf(token("Ok", 0, 2, page = 0)))
        }

        val report = DocumentIntake(
            context = context,
            ocrAdapter = adapter,
            pdfPageSource = source,
            limits = DocumentIntakeLimits(maxPages = 2),
        ).intake(uri)

        assertEquals(DocumentIntakeOutcome.PARTIAL, report.outcome)
        assertEquals(
            listOf(DocumentIntakeReason.PAGE_LIMIT, DocumentIntakeReason.RECOGNITION_FAILED),
            report.reasons,
        )
        assertEquals(listOf(0, 1, 2), source.requested)
        assertEquals("Ok", report.result?.text)
        assertTrue(source.closed)
        assertTrue(source.bitmaps.all { it.isRecycled })
        assertFalse(report.toString().contains("synthetic-engine-detail"))
    }

    @Test
    fun cancellationReleasesHandlesAndBitmapsWithoutWritingStorage() = runBlocking {
        val storage = listOfNotNull(
            context.cacheDir,
            context.filesDir,
            context.noBackupFilesDir,
            context.externalCacheDir,
        )
        fun snapshot() = storage.map { dir -> dir.listFiles()?.map { it.name }?.sorted() }
        val before = snapshot()
        val uri = Uri.fromFile(resourceFile("sample_multipage.pdf"))
        val source = ScriptedPdfPageSource((0..2).map { Step.Page(it) })
        var calls = 0
        val adapter = OcrAdapter {
            if (calls++ == 1) {
                currentCoroutineContext().cancel()
                awaitCancellation()
            }
            ocrResult(text = "Ok", tokens = listOf(token("Ok", 0, 2, page = 0)))
        }

        val job = async { DocumentIntake(context, adapter, pdfPageSource = source).intake(uri) }

        assertFailsWith<CancellationException> { job.await() }
        assertEquals(listOf(0, 1), source.requested)
        assertTrue(source.closed)
        assertEquals(2, source.bitmaps.size)
        assertTrue(source.bitmaps.all { it.isRecycled })
        assertEquals(before, snapshot())
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

    private fun token(
        text: String,
        start: Int,
        end: Int,
        page: Int,
        bbox: OcrBoundingBox? = OcrBoundingBox(start, page * 100, end, page * 100 + 10),
    ): OcrToken =
        OcrToken(
            text = text,
            start = start,
            end = end,
            confidence = 0.99f,
            bbox = bbox,
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

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private sealed class Step {
        data class Page(val index: Int) : Step()

        data class Missing(val index: Int) : Step()

        data class Fail(val error: Exception) : Step()
    }

    /** Emits scripted page events, tracking renders, bitmaps and the open handle. */
    private class ScriptedPdfPageSource(val script: List<Step>) : PdfPageSource {
        val requested = mutableListOf<Int>()
        val bitmaps = mutableListOf<Bitmap>()
        var closed = false

        override suspend fun renderEach(
            context: Context,
            uri: Uri,
            block: suspend (RenderedPdfPage) -> Unit,
        ) {
            throw UnsupportedOperationException("renderPages is used")
        }

        override suspend fun renderPages(
            context: Context,
            uri: Uri,
            block: suspend (PdfPageEvent) -> Unit,
        ) {
            try {
                for (step in script) {
                    when (step) {
                        is Step.Page -> {
                            requested += step.index
                            val bitmap = Bitmap.createBitmap(16, 16, Bitmap.Config.ARGB_8888)
                            bitmaps += bitmap
                            try {
                                block(PdfPageEvent.Rendered(RenderedPdfPage(step.index, bitmap)))
                            } finally {
                                bitmap.recycle()
                            }
                        }
                        is Step.Missing -> block(
                            PdfPageEvent.Unavailable(step.index, DocumentIntakeReason.PAGE_MISSING),
                        )
                        is Step.Fail -> throw step.error
                    }
                }
            } finally {
                closed = true
            }
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
