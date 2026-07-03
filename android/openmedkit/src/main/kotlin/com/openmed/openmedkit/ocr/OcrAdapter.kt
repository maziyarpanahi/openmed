package com.openmed.openmedkit.ocr

import com.google.mlkit.vision.common.InputImage

/**
 * On-device OCR adapter contract for Android OpenMedKit.
 */
fun interface OcrAdapter {
    suspend fun recognize(image: InputImage): OcrResult
}
