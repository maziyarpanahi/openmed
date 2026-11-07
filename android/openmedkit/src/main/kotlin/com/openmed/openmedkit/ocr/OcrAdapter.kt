package com.openmed.openmedkit.ocr

/**
 * On-device OCR adapter contract for Android OpenMedKit.
 *
 * The contract uses the platform-neutral [OcrImage] so the core artifact carries
 * no OCR engine. The ML Kit implementation, `MlKitOcrAdapter`, ships in the
 * optional `openmedkit-mlkit` artifact; tests and demos can use
 * [FakeOcrAdapter].
 */
fun interface OcrAdapter {
    /**
     * Recognizes text in [image] on device.
     *
     * @param image Image plus the clockwise rotation that makes it upright.
     * @return Recognized text with token offsets that index back into it. Token
     *   boxes use the upright image's pixel space, top-left origin, after
     *   [OcrImage.rotationDegrees] is applied; `DocumentIntake` maps them back to
     *   the original page.
     */
    suspend fun recognize(image: OcrImage): OcrResult
}
