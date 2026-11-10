package com.openmed.openmedkit.ocr

import android.graphics.Rect
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.Closeable
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * ML Kit Text Recognition v2 adapter using the bundled Latin script model.
 */
class MlKitOcrAdapter(
    private val recognizer: TextRecognizer =
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS),
) : OcrAdapter, Closeable {
    override suspend fun recognize(image: InputImage): OcrResult =
        recognizer.process(image)
            .await()
            .toOcrResult()

    override fun close() {
        recognizer.close()
    }
}

private fun Text.toOcrResult(): OcrResult =
    buildOcrResult(
        blocks = textBlocks.map { block ->
            RecognizedOcrBlock(
                bbox = block.boundingBox.toOcrBoundingBox(),
                lines = block.lines.map { line ->
                    RecognizedOcrLine(
                        bbox = line.boundingBox.toOcrBoundingBox(),
                        tokens = line.elements.map { element ->
                            RecognizedOcrToken(
                                text = element.text,
                                confidence = element.confidence.coerceConfidence(),
                                bbox = element.boundingBox.toOcrBoundingBox(),
                            )
                        },
                    )
                },
            )
        },
        metadata = mapOf("engine" to "mlkit-text-recognition-v2-latin"),
    )

private fun Rect?.toOcrBoundingBox(): OcrBoundingBox? =
    this?.let { rect ->
        OcrBoundingBox(
            left = rect.left,
            top = rect.top,
            right = rect.right,
            bottom = rect.bottom,
        )
    }

private fun Float?.coerceConfidence(): Float =
    (this ?: 0.0f).coerceIn(0.0f, 1.0f)

private suspend fun <T> Task<T>.await(): T =
    suspendCancellableCoroutine { continuation ->
        addOnSuccessListener { result ->
            continuation.resume(result)
        }
        addOnFailureListener { error ->
            continuation.resumeWithException(error)
        }
        addOnCanceledListener {
            continuation.cancel()
        }
    }
