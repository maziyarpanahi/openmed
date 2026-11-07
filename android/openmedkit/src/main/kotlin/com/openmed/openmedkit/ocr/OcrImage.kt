package com.openmed.openmedkit.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory

/**
 * Platform-neutral image handed to an [OcrAdapter].
 *
 * OpenMedKit core does not depend on any OCR engine, so adapters receive either
 * a decoded [Bitmap] or encoded image bytes plus the clockwise rotation, in
 * degrees, that makes the content upright. The rotation follows the same
 * convention as CameraX `ImageInfo.rotationDegrees` and EXIF orientation, and
 * must be 0, 90, 180, or 270.
 *
 * Instances never render pixel data or bytes in [toString], so they are safe to
 * pass through diagnostics that only record the image kind and dimensions.
 */
sealed class OcrImage(
    val rotationDegrees: Int,
) {
    init {
        require(rotationDegrees in SUPPORTED_ROTATIONS) {
            "rotationDegrees must be one of 0, 90, 180, or 270"
        }
    }

    /**
     * A decoded bitmap. The caller owns the bitmap and its recycling.
     */
    class BitmapImage(
        val bitmap: Bitmap,
        rotationDegrees: Int = 0,
    ) : OcrImage(rotationDegrees) {
        init {
            require(!bitmap.isRecycled) { "bitmap must not be recycled" }
        }

        /** Width of the stored pixels before rotation. */
        val width: Int
            get() = bitmap.width

        /** Height of the stored pixels before rotation. */
        val height: Int
            get() = bitmap.height

        /** Width after applying [rotationDegrees]. */
        val uprightWidth: Int
            get() = if (isQuarterTurn) height else width

        /** Height after applying [rotationDegrees]. */
        val uprightHeight: Int
            get() = if (isQuarterTurn) width else height

        override fun toString(): String =
            "OcrImage.BitmapImage(width=$width, height=$height, " +
                "rotationDegrees=$rotationDegrees)"
    }

    /**
     * Encoded JPEG, PNG, WebP, or another format the platform decoder supports.
     *
     * The bytes are copied on construction and on [bytes] so callers cannot
     * mutate the image after handing it to an adapter.
     */
    class EncodedImage(
        bytes: ByteArray,
        rotationDegrees: Int = 0,
    ) : OcrImage(rotationDegrees) {
        private val data: ByteArray = bytes.copyOf()

        init {
            require(data.isNotEmpty()) { "encoded image bytes must not be empty" }
        }

        /** Number of encoded bytes. */
        val byteCount: Int
            get() = data.size

        /** Returns a copy of the encoded bytes. */
        fun bytes(): ByteArray = data.copyOf()

        /**
         * Decodes the bytes with the platform [BitmapFactory].
         *
         * The caller owns the returned bitmap and should recycle it after OCR.
         *
         * @throws IllegalArgumentException when the bytes are not a decodable image.
         */
        fun decodeBitmap(): Bitmap =
            BitmapFactory.decodeByteArray(data, 0, data.size)
                ?: throw IllegalArgumentException("Unable to decode OCR image bytes")

        override fun toString(): String =
            "OcrImage.EncodedImage(byteCount=$byteCount, rotationDegrees=$rotationDegrees)"
    }

    /** True when [rotationDegrees] swaps the image width and height. */
    val isQuarterTurn: Boolean
        get() = rotationDegrees == 90 || rotationDegrees == 270

    companion object {
        /** Rotations every adapter must accept. */
        val SUPPORTED_ROTATIONS: Set<Int> = setOf(0, 90, 180, 270)

        /**
         * Wraps a decoded bitmap.
         *
         * @param bitmap Caller-owned bitmap; it must not be recycled before OCR finishes.
         * @param rotationDegrees Clockwise rotation that makes the content upright.
         */
        fun fromBitmap(bitmap: Bitmap, rotationDegrees: Int = 0): BitmapImage =
            BitmapImage(bitmap = bitmap, rotationDegrees = rotationDegrees)

        /**
         * Wraps encoded image bytes, for example a CameraX JPEG capture.
         *
         * @param bytes Encoded image bytes; they are copied.
         * @param rotationDegrees Clockwise rotation that makes the content upright.
         */
        fun fromEncodedBytes(bytes: ByteArray, rotationDegrees: Int = 0): EncodedImage =
            EncodedImage(bytes = bytes, rotationDegrees = rotationDegrees)
    }
}
