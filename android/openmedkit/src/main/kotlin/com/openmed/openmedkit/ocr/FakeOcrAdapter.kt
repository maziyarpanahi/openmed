package com.openmed.openmedkit.ocr

/**
 * In-memory OCR adapter for deterministic JVM tests and demos.
 */
class FakeOcrAdapter(
    private val result: OcrResult = OcrResult.empty(metadata = mapOf("engine" to "fake")),
) : OcrAdapter {
    val calls: List<OcrImage>
        get() = _calls.toList()

    private val _calls = mutableListOf<OcrImage>()

    override suspend fun recognize(image: OcrImage): OcrResult {
        _calls += image
        return result
    }
}
