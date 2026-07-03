package com.openmed.openmedkit.ocr

import com.google.mlkit.vision.common.InputImage

/**
 * In-memory OCR adapter for deterministic JVM tests and demos.
 */
class FakeOcrAdapter(
    private val result: OcrResult = OcrResult.empty(metadata = mapOf("engine" to "fake")),
) : OcrAdapter {
    val calls: List<InputImage>
        get() = _calls.toList()

    private val _calls = mutableListOf<InputImage>()

    override suspend fun recognize(image: InputImage): OcrResult {
        _calls += image
        return result
    }
}
