package org.openmed.scan

import android.graphics.Bitmap
import com.google.mlkit.common.MlKit
import com.google.mlkit.vision.common.InputImage
import com.openmed.openmedkit.ocr.FakeOcrAdapter
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class ScanPipelineTest {
    @Test
    fun sampleOcrTextFeedsDeIdentificationAndRedactsExpectedSpans() = runTest {
        MlKit.initialize(RuntimeEnvironment.getApplication())
        val adapter = FakeOcrAdapter(SampleClinicalDocument.ocrResult())
        val pipeline = ScanPipeline(
            ocrAdapter = adapter,
            deidentifier = OnDeviceDeidentifier(),
        )
        val image = InputImage.fromBitmap(Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888), 0)

        val result = pipeline.run(image)

        assertEquals(1, adapter.calls.size)
        assertEquals("bundled-synthetic-sample", result.ocrEngine)
        assertTrue(result.ocrTokenCount > 100)
        assertTrue(result.deidentified.spans.size >= 18)
        assertTrue(result.deidentified.spans.any { it.label == "NAME" })
        assertTrue(result.deidentified.spans.any { it.label == "MRN" })
        assertTrue(result.deidentified.spans.any { it.label == "SSN" })
        assertTrue(result.deidentified.spans.any { it.label == "EMAIL" })
        assertFalse(result.deidentified.redactedText.contains("Jordan"))
        assertFalse(result.deidentified.redactedText.contains("SRMC-7741920"))
        assertFalse(result.deidentified.redactedText.contains("900-21-7755"))
        assertFalse(result.deidentified.redactedText.contains("jordan.whitfield@samplemail.test"))
        assertTrue(result.deidentified.redactedText.contains("[NAME]"))
        assertTrue(result.deidentified.redactedText.contains("[MRN]"))
    }
}
