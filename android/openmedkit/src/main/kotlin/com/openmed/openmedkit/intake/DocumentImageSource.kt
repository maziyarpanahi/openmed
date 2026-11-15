package com.openmed.openmedkit.intake

import android.content.Context
import android.graphics.BitmapFactory
import android.media.ExifInterface
import android.net.Uri
import com.openmed.openmedkit.ocr.OcrImage
import java.io.InputStream

/**
 * An image handed to OCR plus the original geometry its evidence maps back to.
 *
 * @param image Stored bitmap plus the clockwise rotation that makes it upright.
 * @param geometry Original image geometry. The default declares the whole
 *   stored bitmap; a source that crops a larger original declares the original
 *   size and the crop instead.
 */
data class DocumentImagePage(
    val image: OcrImage.BitmapImage,
    val geometry: DocumentPageGeometry = image.documentGeometry(),
)

/**
 * Geometry for an uncropped stored bitmap: pixels, with the image rotation.
 */
fun OcrImage.BitmapImage.documentGeometry(): DocumentPageGeometry =
    DocumentPageGeometry(
        width = width.toDouble(),
        height = height.toDouble(),
        unit = DocumentSpatialUnit.PIXEL,
        rotationDegrees = rotationDegrees,
    )

/**
 * Source of decoded image documents for [DocumentIntake].
 *
 * [DocumentIntake] owns the returned bitmap and recycles it after OCR.
 */
fun interface DocumentImageSource {
    /**
     * Decodes [uri] into a bitmap plus the clockwise rotation that makes it upright.
     */
    suspend fun load(context: Context, uri: Uri): OcrImage.BitmapImage

    /**
     * Decodes [uri] with the original geometry used for evidence coordinates.
     *
     * The default wraps [load] and declares the whole stored bitmap. Throw
     * [DocumentIntakeException] for value-free document-level failures.
     */
    suspend fun loadPage(context: Context, uri: Uri): DocumentImagePage =
        DocumentImagePage(load(context, uri))
}

/**
 * Decodes image URIs with the platform [BitmapFactory] and EXIF orientation.
 *
 * This keeps image intake inside the Android framework, so core OpenMedKit
 * needs no OCR-engine image type. Errors never include the URI or file name.
 */
class BitmapDocumentImageSource : DocumentImageSource {
    override suspend fun load(context: Context, uri: Uri): OcrImage.BitmapImage {
        val resolver = context.contentResolver
        val rotationDegrees = resolver.openInputStream(uri)?.use(::exifRotationDegrees) ?: 0
        val bitmap = resolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
            ?: throw IllegalArgumentException("Unable to decode image document")
        return OcrImage.fromBitmap(bitmap = bitmap, rotationDegrees = rotationDegrees)
    }
}

/**
 * Maps the EXIF orientation tag to a clockwise rotation, ignoring mirroring.
 *
 * Unreadable or missing EXIF data is treated as upright.
 */
internal fun exifRotationDegrees(stream: InputStream): Int {
    val orientation = try {
        ExifInterface(stream).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
    } catch (_: Exception) {
        ExifInterface.ORIENTATION_NORMAL
    }
    return when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90,
        ExifInterface.ORIENTATION_TRANSVERSE,
        -> 90
        ExifInterface.ORIENTATION_ROTATE_180,
        ExifInterface.ORIENTATION_FLIP_VERTICAL,
        -> 180
        ExifInterface.ORIENTATION_ROTATE_270,
        ExifInterface.ORIENTATION_TRANSPOSE,
        -> 270
        else -> 0
    }
}
