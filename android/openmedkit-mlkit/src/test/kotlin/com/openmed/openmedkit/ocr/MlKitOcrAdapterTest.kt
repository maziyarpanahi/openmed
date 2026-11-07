package com.openmed.openmedkit.ocr

import android.graphics.Bitmap
import android.graphics.Point
import android.graphics.Rect
import android.net.Uri
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.openmed.openmedkit.intake.DocumentBox
import com.openmed.openmedkit.intake.DocumentInputKind
import com.openmed.openmedkit.intake.DocumentIntake
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class MlKitOcrAdapterTest {
    @Before
    fun setUp() {
        initializeMlKitForTests(RuntimeEnvironment.getApplication())
    }

    @Test
    fun reproducesCoreOffsetsOnSyntheticTwoLineFixture() {
        val recognized = syntheticTwoLineLayout()

        val result = recognized.toMlKitText().toOcrResult()

        assertEquals("Patient Jordan\nMRN-123", result.text)
        assertEquals(
            listOf(0 to 7, 8 to 14, 15 to 22),
            result.tokens.map { it.start to it.end },
        )
        assertEquals(buildOcrResult(recognized).words, result.words)
        assertEquals(MLKIT_ENGINE_NAME, result.metadata["engine"])
    }

    @Test
    fun reproducesCoreOrderingAcrossBlocksLinesAndTokens() {
        val recognized = listOf(
            RecognizedOcrBlock(
                bbox = OcrBoundingBox(0, 100, 200, 130),
                lines = listOf(
                    RecognizedOcrLine(
                        bbox = OcrBoundingBox(0, 100, 200, 130),
                        tokens = listOf(
                            RecognizedOcrToken("second", 0.88f, OcrBoundingBox(0, 100, 60, 130)),
                        ),
                    ),
                ),
            ),
            RecognizedOcrBlock(
                bbox = OcrBoundingBox(0, 0, 200, 30),
                lines = listOf(
                    RecognizedOcrLine(
                        bbox = OcrBoundingBox(0, 0, 200, 30),
                        tokens = listOf(
                            RecognizedOcrToken("block", 0.92f, OcrBoundingBox(60, 0, 110, 30)),
                            RecognizedOcrToken("first", 0.93f, OcrBoundingBox(0, 0, 50, 30)),
                        ),
                    ),
                ),
            ),
        )

        val result = recognized.toMlKitText().toOcrResult()

        assertEquals("first block\n\nsecond", result.text)
        assertEquals(
            listOf(0 to 5, 6 to 11, 13 to 19),
            result.tokens.map { it.start to it.end },
        )
        assertEquals(buildOcrResult(recognized).words, result.words)
    }

    @Test
    fun clampsOutOfRangeEngineConfidence() {
        val text = listOf(
            RecognizedOcrBlock(
                bbox = OcrBoundingBox(0, 0, 10, 10),
                lines = listOf(
                    RecognizedOcrLine(
                        bbox = OcrBoundingBox(0, 0, 10, 10),
                        tokens = listOf(
                            RecognizedOcrToken("A", 1.5f, OcrBoundingBox(0, 0, 5, 10)),
                            RecognizedOcrToken("B", -0.5f, OcrBoundingBox(5, 0, 10, 10)),
                        ),
                    ),
                ),
            ),
        ).toMlKitText()

        assertEquals(listOf(1.0f, 0.0f), text.toOcrResult().tokens.map { it.confidence })
    }

    @Test
    fun passesNeutralBitmapAndRotationToMlKit() = runBlocking {
        val captured = mutableListOf<InputImage>()
        val adapter = MlKitOcrAdapter(
            process = { image ->
                captured += image
                emptyList<RecognizedOcrBlock>().toMlKitText()
            },
            release = {},
        )
        val bitmap = Bitmap.createBitmap(40, 20, Bitmap.Config.ARGB_8888)

        val result = adapter.recognize(OcrImage.fromBitmap(bitmap, rotationDegrees = 90))

        val image = captured.single()
        assertEquals(90, image.rotationDegrees)
        assertEquals(40, image.width)
        assertEquals(20, image.height)
        assertEquals(InputImage.IMAGE_FORMAT_BITMAP, image.format)
        assertSame(bitmap, image.bitmapInternal)
        assertFalse("caller-owned bitmaps stay usable", bitmap.isRecycled)
        assertEquals("", result.text)
    }

    @Test
    fun decodesEncodedImageAndRecyclesTheTemporaryBitmap() = runBlocking {
        val captured = mutableListOf<InputImage>()
        val adapter = MlKitOcrAdapter(
            process = { image ->
                captured += image
                emptyList<RecognizedOcrBlock>().toMlKitText()
            },
            release = {},
        )

        adapter.recognize(OcrImage.fromEncodedBytes(syntheticPng(12, 8), rotationDegrees = 180))

        val image = captured.single()
        assertEquals(180, image.rotationDegrees)
        assertEquals(12, image.width)
        assertEquals(8, image.height)
        assertTrue("decoded bitmap is recycled after OCR", image.bitmapInternal!!.isRecycled)
    }

    @Test
    fun closeReleasesTheRecognizer() {
        var released = 0
        val adapter = MlKitOcrAdapter(
            process = { emptyList<RecognizedOcrBlock>().toMlKitText() },
            release = { released += 1 },
        )

        adapter.close()

        assertEquals(1, released)
    }

    @Test
    fun awaitResolvesCompletedTasksWithoutTheMainLooper() = runBlocking {
        assertEquals("done", Tasks.forResult("done").await())

        val thrown = runCatching {
            Tasks.forException<String>(IllegalStateException("synthetic failure")).await()
        }.exceptionOrNull()
        assertTrue(thrown is IllegalStateException)
        assertEquals("synthetic failure", thrown?.message)

        val canceled = runCatching { Tasks.forCanceled<String>().await() }.exceptionOrNull()
        assertTrue(canceled is CancellationException)
    }

    @Test
    fun plugsIntoCoreDocumentIntakeForImageDocuments() = runBlocking {
        val context = RuntimeEnvironment.getApplication()
        // Sized to hold the synthetic layout: intake now validates every box.
        val document = File(context.cacheDir, "synthetic_intake.png").apply {
            writeBytes(syntheticPng(200, 40))
        }
        val captured = mutableListOf<InputImage>()
        val adapter = MlKitOcrAdapter(
            process = { image ->
                captured += image
                syntheticTwoLineLayout().toMlKitText()
            },
            release = {},
        )

        try {
            val result = DocumentIntake(context = context, ocrAdapter = adapter)
                .extract(Uri.fromFile(document))

            assertEquals(DocumentInputKind.IMAGE, result.inputKind)
            assertEquals("Patient Jordan\nMRN-123", result.text)
            assertEquals(200, result.pages.single().width)
            assertEquals(40, result.pages.single().height)
            assertEquals(0, captured.single().rotationDegrees)
            assertEquals(result.text.length, result.offsetMap.size)
            val jordan = requireNotNull(result.evidence).regionsForUtf16Span(8, 14).single()
            assertEquals(DocumentBox(70.0, 0.0, 130.0, 20.0), jordan.sourceBox)
        } finally {
            document.delete()
        }
    }

    private fun syntheticPng(width: Int, height: Int): ByteArray =
        SyntheticImages.encode(width = width, height = height, format = "png")
}

/** The synthetic layout shared with the core `OcrOffsetMapTest`. */
private fun syntheticTwoLineLayout(): List<RecognizedOcrBlock> =
    listOf(
        RecognizedOcrBlock(
            bbox = OcrBoundingBox(0, 0, 200, 40),
            lines = listOf(
                RecognizedOcrLine(
                    bbox = OcrBoundingBox(0, 0, 200, 20),
                    tokens = listOf(
                        RecognizedOcrToken("Patient", 0.96f, OcrBoundingBox(0, 0, 60, 20)),
                        RecognizedOcrToken("Jordan", 0.94f, OcrBoundingBox(70, 0, 130, 20)),
                    ),
                ),
                RecognizedOcrLine(
                    bbox = OcrBoundingBox(0, 22, 120, 40),
                    tokens = listOf(
                        RecognizedOcrToken("MRN-123", 0.90f, OcrBoundingBox(0, 22, 80, 40)),
                    ),
                ),
            ),
        ),
    )

/**
 * Builds ML Kit [Text] through its public constructors so the adapter's mapping
 * is exercised against the same synthetic layout as the core offset tests.
 */
private fun List<RecognizedOcrBlock>.toMlKitText(): Text {
    val blocks = map { block ->
        val lines = block.lines.map { line ->
            val elements = line.tokens.map { token ->
                Text.Element(
                    token.text,
                    token.bbox.toRect(),
                    emptyList<Point>(),
                    "und",
                    null,
                    token.confidence,
                    0.0f,
                    emptyList<Text.Symbol>(),
                )
            }
            Text.Line(
                elements.joinToString(" ") { it.text },
                line.bbox.toRect(),
                emptyList<Point>(),
                "und",
                null,
                elements,
                1.0f,
                0.0f,
            )
        }
        Text.TextBlock(
            lines.joinToString("\n") { it.text },
            block.bbox.toRect(),
            emptyList<Point>(),
            "und",
            null,
            lines,
        )
    }
    return Text(blocks.joinToString("\n\n") { it.text }, blocks)
}

private fun OcrBoundingBox?.toRect(): Rect =
    requireNotNull(this) { "fixture boxes are required" }.let {
        Rect(it.left, it.top, it.right, it.bottom)
    }
