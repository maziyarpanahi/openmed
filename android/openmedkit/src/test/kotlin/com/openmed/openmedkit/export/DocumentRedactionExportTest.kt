package com.openmed.openmedkit.export

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.content.pm.ProviderInfo
import android.graphics.Bitmap
import android.net.Uri
import android.provider.OpenableColumns
import com.openmed.openmedkit.export.DocumentRedactionExportContract as Contract
import com.openmed.openmedkit.export.DocumentRedactionExportReason as Reason
import com.openmed.openmedkit.export.DocumentRedactionMask as Mask
import com.openmed.openmedkit.intake.DocumentBox
import com.openmed.openmedkit.intake.DocumentEvidenceMapper
import com.openmed.openmedkit.intake.DocumentImagePage
import com.openmed.openmedkit.intake.DocumentImageSource
import com.openmed.openmedkit.intake.DocumentIntake
import com.openmed.openmedkit.intake.DocumentIntakeLimits
import com.openmed.openmedkit.intake.DocumentIntakeOutcome
import com.openmed.openmedkit.intake.DocumentIntakeReason
import com.openmed.openmedkit.intake.DocumentIntakeReport
import com.openmed.openmedkit.intake.DocumentPageGeometry
import com.openmed.openmedkit.intake.DocumentPageInput
import com.openmed.openmedkit.intake.DocumentRecognition
import com.openmed.openmedkit.intake.DocumentRecognizedRegion
import com.openmed.openmedkit.intake.DocumentSpatialUnit
import com.openmed.openmedkit.intake.PdfPageEvent
import com.openmed.openmedkit.intake.PdfPageSource
import com.openmed.openmedkit.intake.RenderedPdfPage
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrBoundingBox
import com.openmed.openmedkit.ocr.OcrImage
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
import java.io.File
import java.io.FileNotFoundException
import kotlin.test.assertFailsWith
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

private val PALETTE = intArrayOf(
    0xFFFF0000.toInt(),
    0xFF009900.toInt(),
    0xFF0000FF.toInt(),
    0xFF00FFFF.toInt(),
    0xFFFF00FF.toInt(),
)
private val WORDS = listOf("Jane", "Roe", "Visit", "Plan", "Jane")
private const val WHITE = 0xFFFFFFFF.toInt()
private const val AUTHORITY = "org.openmed.openmedkit.test.openmed.redaction-exports"

/** A colored word block at upright pixel coordinates `[left, top, right, bottom)`. */
private class Block(val color: Int, val left: Int, val top: Int, val right: Int, val bottom: Int)

private fun synthetic(width: Int, height: Int, blocks: List<Block>): Bitmap {
    val pixels = IntArray(width * height) { WHITE }
    blocks.forEach { block ->
        for (y in block.top until block.bottom) {
            for (x in block.left until block.right) {
                pixels[y * width + x] = PALETTE[block.color]
            }
        }
    }
    return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
        setPixels(pixels, 0, width, 0, 0, width, height)
    }
}

/**
 * Recognizes each palette-colored block as one word, in palette order, with
 * token boxes in upright pixels as the [OcrAdapter] contract requires.
 */
private class BlockOcrAdapter(
    private val behavior: Behavior = Behavior.READ,
) : OcrAdapter {
    enum class Behavior { READ, FAIL, CANCEL }

    var calls = 0
        private set

    override suspend fun recognize(image: OcrImage): OcrResult {
        calls += 1
        when (behavior) {
            Behavior.FAIL -> throw IllegalStateException("synthetic-private-diagnostic")
            Behavior.CANCEL -> {
                currentCoroutineContext().cancel()
                awaitCancellation()
            }
            Behavior.READ -> Unit
        }
        val bitmap = (image as OcrImage.BitmapImage).bitmap
        val stored = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(stored, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val quarter = image.rotationDegrees == 90 || image.rotationDegrees == 270
        val width = if (quarter) bitmap.height else bitmap.width
        val height = if (quarter) bitmap.width else bitmap.height
        val boxes = mutableMapOf<Int, IntArray>()
        for (y in 0 until height) {
            for (x in 0 until width) {
                // Upright (x, y) read back from stored pixels by the inverse turn.
                val (sx, sy) = when (image.rotationDegrees) {
                    90 -> y to bitmap.height - 1 - x
                    180 -> bitmap.width - 1 - x to bitmap.height - 1 - y
                    270 -> bitmap.width - 1 - y to x
                    else -> x to y
                }
                val color = PALETTE.indexOf(stored[sy * bitmap.width + sx])
                if (color < 0) {
                    continue
                }
                val box = boxes.getOrPut(color) { intArrayOf(x, y, x + 1, y + 1) }
                box[0] = minOf(box[0], x)
                box[1] = minOf(box[1], y)
                box[2] = maxOf(box[2], x + 1)
                box[3] = maxOf(box[3], y + 1)
            }
        }
        val text = StringBuilder()
        val tokens = boxes.keys.sorted().map { color ->
            if (text.isNotEmpty()) {
                text.append(' ')
            }
            val start = text.length
            text.append(WORDS[color])
            val box = boxes.getValue(color)
            OcrToken(
                text = WORDS[color],
                start = start,
                end = text.length,
                confidence = 1.0f,
                bbox = OcrBoundingBox(box[0], box[1], box[2], box[3]),
            )
        }
        return OcrResult(text = text.toString(), words = tokens)
    }
}

private class SyntheticPage(
    val width: Int,
    val height: Int,
    val blocks: List<Block>,
    val geometry: DocumentPageGeometry,
)

/** Renders upright synthetic page bitmaps, as a platform renderer would. */
private class BlockPdfSource(private val pages: List<SyntheticPage>) : PdfPageSource {
    val bitmaps = mutableListOf<Bitmap>()

    override suspend fun renderEach(context: Context, uri: Uri, block: suspend (RenderedPdfPage) -> Unit) {
        throw UnsupportedOperationException("renderPages is used")
    }

    override suspend fun renderPages(context: Context, uri: Uri, block: suspend (PdfPageEvent) -> Unit) {
        pages.forEachIndexed { index, page ->
            val bitmap = synthetic(page.width, page.height, page.blocks)
            bitmaps += bitmap
            try {
                block(PdfPageEvent.Rendered(RenderedPdfPage(index, bitmap, page.geometry)))
            } finally {
                bitmap.recycle()
            }
        }
    }
}

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class DocumentRedactionExportTest {
    private lateinit var context: Context
    private val exporters = mutableListOf<DocumentRedactionExporter>()

    @Before
    fun setUp() {
        context = RuntimeEnvironment.getApplication()
        Robolectric.buildContentProvider(DocumentRedactionExportProvider::class.java).create(
            ProviderInfo().apply {
                authority = AUTHORITY
                exported = false
                grantUriPermissions = true
            },
        )
    }

    @After
    fun tearDown() {
        exporters.forEach { it.close() }
        DocumentRedactionExporter.exportRoot(context).deleteRecursively()
    }

    private fun exporter(
        ocr: OcrAdapter = BlockOcrAdapter(),
        pdf: PdfPageSource = twoPageSource(),
        image: DocumentImageSource? = null,
    ): DocumentRedactionExporter =
        if (image == null) {
            DocumentRedactionExporter(context, ocr, pdfPageSource = pdf)
        } else {
            DocumentRedactionExporter(context, ocr, pdfPageSource = pdf, imageSource = image)
        }.also { exporters += it }

    private fun sourceFile(name: String, bytes: String): Uri =
        Uri.fromFile(File(context.cacheDir, name).apply { writeText(bytes) })

    private fun storedFiles(): List<String> =
        DocumentRedactionExporter.exportRoot(context).walkTopDown().filter { it.isFile }.map { it.name }.toList()

    /** "Jane Roe" on an unrotated page, then "Visit Plan" on a rotated, cropped page. */
    private fun twoPageSource(extra: List<Block> = emptyList()): BlockPdfSource =
        BlockPdfSource(
            listOf(
                SyntheticPage(
                    400,
                    200,
                    listOf(Block(0, 40, 40, 120, 80), Block(1, 160, 40, 220, 80)) + extra,
                    DocumentPageGeometry(200.0, 100.0, DocumentSpatialUnit.POINT),
                ),
                SyntheticPage(
                    160,
                    360,
                    listOf(Block(2, 40, 60, 80, 140), Block(3, 40, 200, 80, 280)),
                    DocumentPageGeometry(
                        200.0,
                        100.0,
                        DocumentSpatialUnit.POINT,
                        rotationDegrees = 90,
                        crop = DocumentBox(10.0, 10.0, 190.0, 90.0),
                    ),
                ),
            ),
        )

    private suspend fun intake(source: PdfPageSource, uri: Uri, limits: DocumentIntakeLimits = DocumentIntakeLimits()) =
        DocumentIntake(context, BlockOcrAdapter(), pdfPageSource = source, limits = limits).intake(uri)

    private inline fun reason(block: () -> Unit): Reason {
        val error = assertFailsWith<DocumentRedactionExportException> { block() }
        // Messages are the value-free code.
        assertEquals(error.reason.code, error.message)
        return error.reason
    }

    private fun mappedReport(text: String, regions: List<Pair<IntArray, DocumentBox>>): DocumentIntakeReport {
        val recognition = DocumentRecognition(
            width = 200,
            height = 100,
            text = text,
            regions = regions.map { (span, box) -> DocumentRecognizedRegion(span[0], span[1], box) },
        )
        val evidence = DocumentEvidenceMapper.map(
            "a".repeat(64),
            listOf(DocumentPageInput.Rendered(DocumentPageGeometry(200.0, 100.0), recognition)),
        )
        return DocumentIntakeReport(evidence = evidence, result = null)
    }

    @Test
    fun bindingIsOrderFreeAndMatchesSwift() {
        val regions = listOf(
            intArrayOf(0, 4) to DocumentBox(10.0, 10.0, 50.0, 30.0),
            intArrayOf(5, 8) to DocumentBox(60.0, 10.0, 90.0, 30.0),
        )
        val report = mappedReport("Jane Roe", regions)
        val masks = listOf(Mask.Span(0, 4), Mask.Box(0, DocumentBox(1.0, 2.0, 3.0, 4.5)))
        val request = DocumentRedactionRequest(report, masks)
        // The same vector is asserted by Swift's DocumentRedactionContractTests.
        assertEquals("9e1f9f4ea6a29afb6aeaddae7371b52ea919ae6e800b19c320d8b5a688eb2d4d", request.binding)
        assertEquals(request.binding, DocumentRedactionRequest(report, masks.reversed() + masks).binding)
        assertNotEquals(request.binding, DocumentRedactionRequest(report, masks.take(1)).binding)
        val padded = DocumentRedactionPolicy(padding = 3.0)
        assertNotEquals(request.binding, DocumentRedactionRequest(report, masks, padded).binding)
        assertNotEquals(
            request.binding,
            DocumentRedactionRequest(report, masks, DocumentRedactionPolicy(policyId = "hipaa_safe_harbor")).binding,
        )
        assertNotEquals(request.binding, DocumentRedactionRequest(mappedReport("Jane Rob", regions), masks).binding)
        assertFalse(request.toString().contains("Jane"))
        assertEquals(Mask.Span(2, 4), Mask.fromUtf16("😀 Roe Jane", 3, 5))
    }

    @Test
    fun planPlacesSpansThroughRegionsAndRejectsUnplacedOutOfRangeAndIncompleteReports() {
        // "Roe" has no recognized region, so a mask over it cannot be placed.
        val report = mappedReport("Jane Roe", listOf(intArrayOf(0, 4) to DocumentBox(10.0, 10.0, 50.0, 30.0)))
        val plan = Contract.plan(
            DocumentRedactionRequest(
                report,
                listOf(Mask.Span(0, 5), Mask.Box(0, DocumentBox(0.0, 0.0, 5.0, 5.0))),
            ),
        )
        assertEquals(
            listOf(listOf(DocumentBox(10.0, 10.0, 50.0, 30.0), DocumentBox(0.0, 0.0, 5.0, 5.0))),
            plan.map { it.boxes },
        )
        listOf(
            Mask.Span(5, 8) to Reason.MASK_UNPLACED,
            Mask.Span(0, 9) to Reason.MASK_OUT_OF_RANGE,
            Mask.Span(3, 3) to Reason.MASK_OUT_OF_RANGE,
            Mask.Box(1, DocumentBox(0.0, 0.0, 5.0, 5.0)) to Reason.MASK_OUT_OF_RANGE,
            Mask.Box(0, DocumentBox(0.0, 0.0, 201.0, 5.0)) to Reason.MASK_OUT_OF_RANGE,
            Mask.Box(0, DocumentBox(5.0, 0.0, 5.0, 5.0)) to Reason.MASK_OUT_OF_RANGE,
            Mask.Box(0, DocumentBox(Double.NaN, 0.0, 5.0, 5.0)) to Reason.MASK_OUT_OF_RANGE,
        ).forEach { (mask, expected) ->
            val actual = reason { Contract.plan(DocumentRedactionRequest(report, listOf(mask))) }
            assertEquals(mask.toString(), expected, actual)
        }
        val partial = DocumentIntakeReport(
            evidence = DocumentEvidenceMapper.map(
                "a".repeat(64),
                listOf(DocumentPageInput.Unavailable(DocumentIntakeReason.PAGE_MISSING)),
            ),
            result = null,
        )
        assertEquals(
            Reason.INCOMPLETE_REPORT,
            reason { Contract.plan(DocumentRedactionRequest(partial, emptyList())) },
        )
    }

    @Test
    fun forwardTransformReproducesSharedIntakeVectors() {
        val corpus = requireNotNull(javaClass.classLoader?.getResourceAsStream("document_intake_v1.json"))
            .bufferedReader(Charsets.UTF_8).use { JSONObject(it.readText()) }
        var checked = 0
        corpus.getJSONArray("cases").objects().forEach { case ->
            val exact = case.getDouble("tolerance") == 0.0
            case.getJSONArray("pages").objects().forEach pages@{ page ->
                if (page.isNull("geometry")) {
                    return@pages
                }
                val geometryJson = page.getJSONObject("geometry")
                val ocr = page.getJSONObject("ocr")
                val crop = if (geometryJson.isNull("crop")) null else geometryJson.getJSONArray("crop").box()
                val geometry = DocumentPageGeometry(
                    geometryJson.getDouble("width"),
                    geometryJson.getDouble("height"),
                    rotationDegrees = geometryJson.getInt("rotation"),
                    crop = crop,
                )
                val visible = crop ?: DocumentBox(0.0, 0.0, geometry.width, geometry.height)
                val width = ocr.getInt("width")
                val height = ocr.getInt("height")
                ocr.getJSONArray("regions").objects().forEach regions@{ region ->
                    if (region.isNull("truth")) {
                        return@regions
                    }
                    val truth = region.getJSONArray("truth").box()
                    if (truth.left < visible.left || truth.top < visible.top || truth.right > visible.right ||
                        truth.bottom > visible.bottom
                    ) {
                        return@regions
                    }
                    val mapped = Contract.imageBox(truth, geometry, width, height) ?: return@regions
                    var values = listOf(mapped.left, mapped.top, mapped.right, mapped.bottom)
                    if (!exact) {
                        values = listOf(
                            Math.floor(values[0]),
                            Math.floor(values[1]),
                            Math.ceil(values[2]),
                            Math.ceil(values[3]),
                        )
                    }
                    val expected = region.getJSONArray("box").let { box -> (0 until 4).map { box.getDouble(it) } }
                    val clamped = listOf(
                        expected[0].coerceIn(0.0, width.toDouble()),
                        expected[1].coerceIn(0.0, height.toDouble()),
                        expected[2].coerceIn(0.0, width.toDouble()),
                        expected[3].coerceIn(0.0, height.toDouble()),
                    )
                    assertTrue(case.getString("id"), values == expected || values == clamped)
                    checked += 1
                }
            }
        }
        assertTrue(checked > 20)
        assertArrayEquals(
            intArrayOf(8, 1, 21, 9),
            Contract.pixelRect(DocumentBox(10.2, 3.5, 20.1, 7.0), 2.0, 21, 30),
        )
        assertNull(Contract.pixelRect(DocumentBox(30.0, 3.0, 40.0, 7.0), 2.0, 21, 30))
    }

    @Test
    fun residualSearchNormalizesTokensAndScriptsWithoutSpaces() {
        val contract = Contract
        assertEquals(listOf("jane", "o", "neil", "रवि", "2026"), contract.tokens("Ｊａｎｅ  O'Neil, रवि 2026"))
        assertTrue(contract.containsResidual(listOf("Jane Roe"), "Seen by JANE today", 2))
        assertFalse(contract.containsResidual(listOf("Jane Roe"), "Seen by Janet today", 2))
        assertFalse(contract.containsResidual(listOf("J. Roe"), "J was seen", 2))
        assertTrue(contract.containsResidual(listOf("王小明"), "患者王小明来访", 2))
        assertFalse(contract.containsResidual(listOf("王小明"), "患者王小来访", 2))
        assertTrue(contract.containsResidual(listOf("रवि"), "नाम रवि", 2))
    }

    @Test
    fun profileRoundTripsAndRejectsForeignStructuresWithTheirCategory() {
        val red = IntArray(12) { PALETTE[0] }
        val pages = listOf(
            DocumentRedactionPdf.page(red, 4, 3, 2.0, 1.5),
            DocumentRedactionPdf.page(red, 3, 4, 612.25, 792.0),
        )
        val data = DocumentRedactionPdf.write(pages)
        assertEquals(pages, DocumentRedactionPdf.inspect(data, 100))
        assertEquals(listOf("2", "1.5", "612.25", "792"), pages.flatMap { listOf(it.mediaWidth, it.mediaHeight) })
        assertArrayEquals(red, DocumentRedactionPdf.pixels(pages[0]))
        assertEquals("14400", DocumentRedactionPdf.number(14_399.99999))
        fun inspected(bytes: ByteArray, maxPixels: Long = 100) =
            reason { DocumentRedactionPdf.inspect(bytes, maxPixels) }
        assertEquals(Reason.PROFILE_VIOLATION, inspected(data, maxPixels = 11))
        assertEquals(Reason.PROFILE_VIOLATION, inspected(data.copyOf().also { it[20] = 'X'.code.toByte() }))
        assertEquals(Reason.PROFILE_VIOLATION, inspected(data + "\n".toByteArray()))
        assertEquals(Reason.PROFILE_VIOLATION, inspected(data.copyOf(data.size - 7)))
        val text = String(data, Charsets.ISO_8859_1)
        fun foreign(from: String, to: String): ByteArray {
            assertTrue(from in text)
            return text.replaceFirst(from, to).toByteArray(Charsets.ISO_8859_1)
        }
        assertEquals(
            Reason.HIDDEN_TEXT,
            inspected(foreign("/XObject << /Im0 4 0 R >>", "/XObject << /Im0 4 0 R >> /Font << /F1 9 0 R >>")),
        )
        assertEquals(
            Reason.ATTACHMENTS_PRESENT,
            inspected(foreign("/Contents 5 0 R", "/Contents 5 0 R /Annots [9 0 R]")),
        )
        assertEquals(
            Reason.METADATA_PRESENT,
            inspected(foreign("/Root 1 0 R", "/Root 1 0 R /Info 9 0 R")),
        )
        val corrupt = DocumentRedactionPdf.Page(4, 3, "2", "1.5", byteArrayOf(0x78, 0x9C.toByte(), 1, 2, 3))
        assertEquals(Reason.PROFILE_VIOLATION, reason { DocumentRedactionPdf.pixels(corrupt) })
    }

    @Test
    fun exportsAVerifiedImageOnlyPdfAndSharesItOnlyThroughExplicitActions() = runBlocking {
        val source = twoPageSource()
        val uri = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes with Jane Roe metadata")
        val report = intake(source, uri)
        assertEquals(DocumentIntakeOutcome.COMPLETE, report.outcome)
        assertEquals("Jane Roe\n\nVisit Plan", report.evidence.text)

        val request = DocumentRedactionRequest(
            report,
            listOf(Mask.Span(0, 8), Mask.Span(10, 15)),
        )
        val exporter = exporter(pdf = source)
        val artifact = exporter.export(request, uri)
        val receipt = artifact.receipt
        assertEquals("openmed.redaction-export.image-pdf.v1", receipt.profile)
        assertEquals(request.binding, receipt.binding)
        assertEquals(report.evidence.sourceDigest, receipt.sourceDigest)
        assertEquals(listOf(2, 2, 3), listOf(receipt.pageCount, receipt.maskCount, receipt.paintedRegionCount))
        assertEquals(DocumentRedactionCheck.values().toList(), receipt.checks)
        assertFalse(artifact.toString().contains("Jane"))
        assertEquals(1, exporter.artifactCount)
        assertTrue(source.bitmaps.all { it.isRecycled })

        val shared = exporter.contentUri(artifact, request, AUTHORITY)
        assertEquals("content", shared.scheme)
        assertEquals(AUTHORITY, shared.authority)
        assertEquals("${artifact.id}.pdf", shared.lastPathSegment)
        val bytes = requireNotNull(context.contentResolver.openInputStream(shared)).use { it.readBytes() }
        assertEquals(receipt.outputDigest, Contract.sha256(bytes))
        assertEquals(receipt.byteCount, bytes.size.toLong())
        assertEquals("application/pdf", context.contentResolver.getType(shared))
        val text = String(bytes, Charsets.ISO_8859_1)
        listOf("Jane", "synthetic", "/Font", "/Annots", "/Info", "/Metadata").forEach { assertFalse(it, it in text) }
        assertEquals(
            listOf(listOf(400, 200), listOf(160, 360)),
            DocumentRedactionPdf.inspect(bytes, 100_000).map { listOf(it.pixelWidth, it.pixelHeight) },
        )
        val copy = File(context.cacheDir, "reread.pdf").apply { writeBytes(bytes) }
        val reread = DocumentIntake(context, BlockOcrAdapter(), pdfPageSource = DocumentRedactionPdfPageSource())
            .intake(Uri.fromFile(copy))
        assertEquals("\n\nPlan", reread.evidence.text)

        val chooser = exporter.shareIntent(artifact, request, AUTHORITY, "Share")
        assertEquals(Intent.ACTION_CHOOSER, chooser.action)
        @Suppress("DEPRECATION")
        val send = requireNotNull(chooser.getParcelableExtra<Intent>(Intent.EXTRA_INTENT))
        assertEquals(Intent.ACTION_SEND, send.action)
        assertEquals("application/pdf", send.type)
        @Suppress("DEPRECATION")
        assertEquals(shared, send.getParcelableExtra<Uri>(Intent.EXTRA_STREAM))
        assertEquals(shared, send.clipData?.getItemAt(0)?.uri)
        assertEquals(Intent.FLAG_GRANT_READ_URI_PERMISSION, send.flags and GRANT_FLAGS)

        val saved = File(context.cacheDir, "user-picked.pdf")
        exporter.saveTo(artifact, request, Uri.fromFile(saved))
        assertArrayEquals(bytes, saved.readBytes())

        exporter.complete(artifact)
        assertEquals(0, exporter.artifactCount)
        assertFailsWith<FileNotFoundException> { context.contentResolver.openInputStream(shared) }
        assertEquals(Reason.ARTIFACT_UNAVAILABLE, reason { exporter.contentUri(artifact, request, AUTHORITY) })
        assertEquals(emptyList<String>(), storedFiles())
    }

    @Test
    fun rotatedCroppedImageExportsOnlyTheExaminedUprightCrop() = runBlocking {
        // A stored 100x60 crop of a 120x80 original, turned upright by 90 degrees.
        val geometry = DocumentPageGeometry(
            120.0,
            80.0,
            DocumentSpatialUnit.PIXEL,
            rotationDegrees = 90,
            crop = DocumentBox(10.0, 10.0, 110.0, 70.0),
        )
        val images = object : DocumentImageSource {
            val bitmaps = mutableListOf<Bitmap>()

            override suspend fun load(context: Context, uri: Uri): OcrImage.BitmapImage =
                throw UnsupportedOperationException("loadPage is used")

            override suspend fun loadPage(context: Context, uri: Uri): DocumentImagePage {
                val bitmap = synthetic(100, 60, listOf(Block(0, 20, 10, 40, 30), Block(1, 60, 10, 80, 30)))
                bitmaps += bitmap
                return DocumentImagePage(OcrImage.fromBitmap(bitmap, rotationDegrees = 90), geometry)
            }
        }
        val uri = sourceFile("synthetic-photo.png", "synthetic png bytes")
        val report = DocumentIntake(context, BlockOcrAdapter(), imageSource = images).intake(uri)
        assertEquals("Jane Roe", report.evidence.text)
        assertEquals(DocumentBox(30.0, 20.0, 50.0, 40.0), report.evidence.regions.first().sourceBox)

        val request = DocumentRedactionRequest(report, listOf(Mask.Span(0, 4)))
        val exporter = exporter(image = images)
        val artifact = exporter.export(request, uri)
        val shared = exporter.contentUri(artifact, request, AUTHORITY)
        val bytes = requireNotNull(context.contentResolver.openInputStream(shared)).use { it.readBytes() }
        val page = DocumentRedactionPdf.inspect(bytes, 100_000).single()
        assertEquals(listOf(60, 100), listOf(page.pixelWidth, page.pixelHeight))
        assertEquals(listOf("60", "100"), listOf(page.mediaWidth, page.mediaHeight))
        val pixels = DocumentRedactionPdf.pixels(page)
        // Upright, "Jane" sits at x 30..50, y 20..40 and is now black; "Roe" survives below it.
        assertEquals(0xFF000000.toInt(), pixels[30 * 60 + 40])
        assertEquals(PALETTE[1], pixels[70 * 60 + 40])
        assertEquals(1, artifact.receipt.paintedRegionCount)
        assertTrue(images.bitmaps.all { it.isRecycled })
    }

    @Test
    fun residualIdentifierElsewhereRefusesTheExportAndLeavesNoFile() = runBlocking {
        // Negative control: "Jane" also appears as a second, unmasked block.
        val source = twoPageSource(extra = listOf(Block(4, 40, 120, 120, 160)))
        val uri = sourceFile("synthetic-residual.pdf", "synthetic residual bytes")
        val report = intake(source, uri)
        assertEquals("Jane Roe Jane\n\nVisit Plan", report.evidence.text)
        val exporter = exporter(pdf = source)
        val once = DocumentRedactionRequest(report, listOf(Mask.Span(0, 4)))
        assertEquals(Reason.RESIDUAL_TEXT, reason { exporter.export(once, uri) })
        assertEquals(emptyList<String>(), storedFiles())
        val both = DocumentRedactionRequest(report, listOf(Mask.Span(0, 4), Mask.Span(9, 13)))
        assertEquals(2, exporter.export(both, uri).receipt.paintedRegionCount)
    }

    @Test
    fun unverifiableOrChangedSourcesNeverProduceArtifacts() = runBlocking {
        val source = twoPageSource()
        val uri = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes")
        val report = intake(source, uri)
        val request = DocumentRedactionRequest(report, listOf(Mask.Span(0, 4)))

        val failing = exporter(ocr = BlockOcrAdapter(BlockOcrAdapter.Behavior.FAIL), pdf = source)
        assertEquals(Reason.VERIFICATION_INCOMPLETE, reason { failing.export(request, uri) })
        val exporter = exporter(pdf = source)
        val changed = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes, edited")
        assertEquals(Reason.SOURCE_CHANGED, reason { exporter.export(request, changed) })
        val missing = Uri.fromFile(File(context.cacheDir, "missing.pdf"))
        assertEquals(Reason.SOURCE_CHANGED, reason { exporter.export(request, missing) })
        val restored = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes")
        val onePage = exporter(pdf = BlockPdfSource(listOf(twoPageSourcePages().first())))
        assertEquals(Reason.SOURCE_CHANGED, reason { onePage.export(request, restored) })
        val partial = intake(source, uri, DocumentIntakeLimits(maxPages = 1))
        assertEquals(DocumentIntakeOutcome.PARTIAL, partial.outcome)
        assertEquals(
            Reason.INCOMPLETE_REPORT,
            reason { exporter.export(DocumentRedactionRequest(partial, emptyList()), uri) },
        )
        assertEquals(listOf(0, 0, 0), listOf(failing, exporter, onePage).map { it.artifactCount })
        assertEquals(emptyList<String>(), storedFiles())
    }

    private fun twoPageSourcePages(): List<SyntheticPage> =
        listOf(
            SyntheticPage(
                400,
                200,
                listOf(Block(0, 40, 40, 120, 80), Block(1, 160, 40, 220, 80)),
                DocumentPageGeometry(200.0, 100.0, DocumentSpatialUnit.POINT),
            ),
        )

    @Test
    fun cancelledExportRemovesPartialOutput() = runBlocking {
        val source = twoPageSource()
        val uri = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes")
        val report = intake(source, uri)
        val exporter = exporter(ocr = BlockOcrAdapter(BlockOcrAdapter.Behavior.CANCEL), pdf = source)
        val job = async { exporter.export(DocumentRedactionRequest(report, listOf(Mask.Span(0, 4))), uri) }
        assertFailsWith<CancellationException> { job.await() }
        assertEquals(0, exporter.artifactCount)
        assertEquals(emptyList<String>(), storedFiles())
    }

    @Test
    fun changedMasksPolicyOrBytesInvalidateTheArtifact() = runBlocking {
        val source = twoPageSource()
        val uri = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes")
        val report = intake(source, uri)
        val request = DocumentRedactionRequest(report, listOf(Mask.Span(0, 4)))
        val widened = DocumentRedactionRequest(report, listOf(Mask.Span(0, 8)))
        val exporter = exporter(pdf = source)

        val first = exporter.export(request, uri)
        assertEquals(Reason.STALE_REQUEST, reason { exporter.contentUri(first, widened, AUTHORITY) })
        assertEquals(Reason.ARTIFACT_UNAVAILABLE, reason { exporter.contentUri(first, request, AUTHORITY) })

        val second = exporter.export(request, uri)
        val third = exporter.export(widened, uri)
        exporter.invalidate(keeping = widened)
        assertEquals(Reason.ARTIFACT_UNAVAILABLE, reason { exporter.contentUri(second, request, AUTHORITY) })
        val shared = exporter.contentUri(third, widened, AUTHORITY)
        val stricter = DocumentRedactionRequest(report, widened.masks, DocumentRedactionPolicy(padding = 4.0))
        assertEquals(Reason.STALE_REQUEST, reason { exporter.shareIntent(third, stricter, AUTHORITY) })
        assertFailsWith<FileNotFoundException> { context.contentResolver.openInputStream(shared) }

        val fourth = exporter.export(widened, uri)
        val file = DocumentRedactionExporter.exportRoot(context).walkTopDown().single { it.name == "${fourth.id}.pdf" }
        file.writeBytes(file.readBytes().also { it[it.size - 2] = (it[it.size - 2].toInt() xor 1).toByte() })
        assertEquals(Reason.ARTIFACT_CHANGED, reason { exporter.contentUri(fourth, widened, AUTHORITY) })
        assertFalse(file.exists())

        exporter.export(request, uri)
        exporter.invalidateAll()
        assertEquals(0, exporter.artifactCount)
        assertEquals(emptyList<String>(), storedFiles())
    }

    @Test
    fun restartSweepRemovesLeftoversButKeepsLiveSessions() = runBlocking {
        val root = DocumentRedactionExporter.exportRoot(context)
        val leftover = File(root, "0123456789abcdef0123456789abcdef").apply { mkdirs() }
        File(leftover, "pending-1.pdf").writeText("synthetic")
        File(root, "stray.pdf").writeText("synthetic")
        val source = twoPageSource()
        val uri = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes")
        val report = intake(source, uri)
        val request = DocumentRedactionRequest(report, listOf(Mask.Span(0, 4)))

        val live = DocumentRedactionExporter(context, BlockOcrAdapter(), pdfPageSource = source)
        assertFalse(leftover.exists())
        assertFalse(File(root, "stray.pdf").exists())
        val artifact = live.export(request, uri)
        val second = exporter(pdf = source)
        val shared = live.contentUri(artifact, request, AUTHORITY)
        assertTrue(requireNotNull(context.contentResolver.openInputStream(shared)).use { it.readBytes() }.isNotEmpty())
        assertEquals(Reason.ARTIFACT_UNAVAILABLE, reason { second.contentUri(artifact, request, AUTHORITY) })
        live.close()
        assertFailsWith<FileNotFoundException> { context.contentResolver.openInputStream(shared) }
        assertEquals(emptyList<String>(), storedFiles())
    }

    @Test
    fun providerServesOnlyVerifiedArtifactsReadOnly() = runBlocking {
        val source = twoPageSource()
        val uri = sourceFile("synthetic-two-page.pdf", "synthetic pdf bytes")
        val request = DocumentRedactionRequest(intake(source, uri), listOf(Mask.Span(0, 4)))
        val exporter = exporter(pdf = source)
        val artifact = exporter.export(request, uri)
        val shared = exporter.contentUri(artifact, request, AUTHORITY)
        val resolver = context.contentResolver
        requireNotNull(resolver.query(shared, null, null, null, null)).use { cursor ->
            assertTrue(cursor.moveToFirst())
            val name = cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME))
            assertEquals("${artifact.id}.pdf", name)
            assertEquals(artifact.receipt.byteCount, cursor.getLong(cursor.getColumnIndexOrThrow(OpenableColumns.SIZE)))
        }
        assertFailsWith<SecurityException> { resolver.openFileDescriptor(shared, "w") }
        val session = shared.pathSegments[0]
        listOf(
            "content://$AUTHORITY/$session/pending-${artifact.id}.pdf",
            "content://$AUTHORITY/$session/../${artifact.id}.pdf",
            "content://$AUTHORITY/${artifact.id}.pdf",
            "content://$AUTHORITY/$session/${"0".repeat(32)}.pdf",
        ).map(Uri::parse).forEach { foreign ->
            assertFailsWith<FileNotFoundException>(foreign.toString()) { resolver.openInputStream(foreign) }
            assertNull(resolver.getType(foreign))
        }
        assertFailsWith<UnsupportedOperationException> { resolver.delete(shared, null, null) }
        listOf(ProviderInfo().apply { authority = "x.exported"; exported = true; grantUriPermissions = true },
            ProviderInfo().apply { authority = "x.nogrant"; exported = false; grantUriPermissions = false })
            .forEach { info ->
                assertFailsWith<SecurityException> {
                    Robolectric.buildContentProvider(DocumentRedactionExportProvider::class.java).create(info)
                }
            }
        assertEquals(1, ClipData.newRawUri(null, shared).itemCount)
    }

    private companion object {
        const val GRANT_FLAGS = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
            Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION or Intent.FLAG_GRANT_PREFIX_URI_PERMISSION
    }
}

private fun JSONArray.objects(): List<JSONObject> = (0 until length()).map(::getJSONObject)

private fun JSONArray.box(): DocumentBox = DocumentBox(getDouble(0), getDouble(1), getDouble(2), getDouble(3))
