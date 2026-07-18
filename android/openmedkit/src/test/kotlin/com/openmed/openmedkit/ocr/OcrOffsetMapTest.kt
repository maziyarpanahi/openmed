package com.openmed.openmedkit.ocr

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.openmed.openmedkit.initializeMlKitForTests
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class OcrOffsetMapTest {
    @Test
    fun buildsOffsetsThatIndexBackIntoRecognizedText() {
        val result = buildOcrResult(
            listOf(
                RecognizedOcrBlock(
                    bbox = OcrBoundingBox(0, 0, 200, 40),
                    lines = listOf(
                        RecognizedOcrLine(
                            bbox = OcrBoundingBox(0, 0, 200, 20),
                            tokens = listOf(
                                RecognizedOcrToken(
                                    text = "Patient",
                                    confidence = 0.96f,
                                    bbox = OcrBoundingBox(0, 0, 60, 20),
                                ),
                                RecognizedOcrToken(
                                    text = "Jordan",
                                    confidence = 0.94f,
                                    bbox = OcrBoundingBox(70, 0, 130, 20),
                                ),
                            ),
                        ),
                        RecognizedOcrLine(
                            bbox = OcrBoundingBox(0, 22, 120, 40),
                            tokens = listOf(
                                RecognizedOcrToken(
                                    text = "MRN-123",
                                    confidence = 0.90f,
                                    bbox = OcrBoundingBox(0, 22, 80, 40),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        assertEquals("Patient Jordan\nMRN-123", result.text)
        assertEquals(listOf("Patient", "Jordan", "MRN-123"), result.tokens.map { it.text })
        result.tokens.forEach { token ->
            assertEquals(token.text, result.text.substring(token.start, token.end))
            assertTrue(token.confidence in 0.0f..1.0f)
        }
    }

    @Test
    fun ordersMultipleBlocksLinesAndTokensByPagePosition() {
        val result = buildOcrResult(
            listOf(
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
            ),
        )

        assertEquals("first block\n\nsecond", result.text)
        assertEquals(
            listOf(0 to 5, 6 to 11, 13 to 19),
            result.tokens.map { it.start to it.end },
        )
    }

    @Test
    fun fakeAdapterReturnsEmptyResultWithoutRunningRecognizer() = runBlocking {
        val adapter = FakeOcrAdapter()
        initializeMlKitForTests(RuntimeEnvironment.getApplication())
        val image = InputImage.fromBitmap(Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888), 0)

        val result = adapter.recognize(image)

        assertEquals("", result.text)
        assertTrue(result.tokens.isEmpty())
        assertEquals("fake", result.metadata["engine"])
        assertEquals(1, adapter.calls.size)
    }
}
