package com.openmed.openmedkit.ocr

import android.graphics.Rect
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.Closeable
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * ML Kit Text Recognition v2 adapter using the bundled Latin script model.
 *
 * This class ships in the optional `openmedkit-mlkit` artifact. Adding that
 * artifact brings Google's ML Kit terms, its auto-initializing components, and
 * the datatransport usage-metrics stack (including the `INTERNET` and
 * `ACCESS_NETWORK_STATE` permissions) into the host application. The core
 * `openmedkit` artifact carries none of them.
 *
 * @param recognizer ML Kit recognizer; defaults to the bundled Latin model.
 */
class MlKitOcrAdapter internal constructor(
    private val process: suspend (InputImage) -> Text,
    private val release: () -> Unit,
) : OcrAdapter, Closeable {
    constructor(
        recognizer: TextRecognizer =
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS),
    ) : this(
        process = { image -> recognizer.process(image).await() },
        release = recognizer::close,
    )

    /**
     * Converts the neutral [image] to an ML Kit [InputImage] and recognizes it.
     *
     * Encoded images are decoded with the platform decoder, and the temporary
     * bitmap is recycled once recognition finishes.
     */
    override suspend fun recognize(image: OcrImage): OcrResult =
        when (image) {
            is OcrImage.BitmapImage ->
                recognize(InputImage.fromBitmap(image.bitmap, image.rotationDegrees))
            is OcrImage.EncodedImage -> {
                val bitmap = image.decodeBitmap()
                try {
                    recognize(InputImage.fromBitmap(bitmap, image.rotationDegrees))
                } finally {
                    bitmap.recycle()
                }
            }
        }

    /**
     * Recognizes an ML Kit [InputImage] directly, for example a CameraX frame
     * wrapped with `InputImage.fromMediaImage`.
     */
    suspend fun recognize(image: InputImage): OcrResult =
        process(image).toOcrResult()

    override fun close() {
        release()
    }
}

internal fun Text.toOcrResult(): OcrResult =
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
        metadata = mapOf("engine" to MLKIT_ENGINE_NAME),
    )

internal const val MLKIT_ENGINE_NAME = "mlkit-text-recognition-v2-latin"

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

/**
 * Awaits a Play services [Task] without hopping through the main looper, so a
 * caller that blocks the main thread cannot deadlock on its own result.
 */
internal suspend fun <T> Task<T>.await(): T =
    suspendCancellableCoroutine { continuation ->
        addOnSuccessListener(DirectExecutor) { result ->
            continuation.resume(result)
        }
        addOnFailureListener(DirectExecutor) { error ->
            continuation.resumeWithException(error)
        }
        addOnCanceledListener(DirectExecutor) {
            continuation.cancel()
        }
    }

private object DirectExecutor : Executor {
    override fun execute(command: Runnable) {
        command.run()
    }
}
