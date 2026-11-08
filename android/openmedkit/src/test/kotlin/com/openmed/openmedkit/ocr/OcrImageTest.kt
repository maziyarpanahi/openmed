package com.openmed.openmedkit.ocr

import android.graphics.Bitmap
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class OcrImageTest {
    @Test
    fun acceptsOnlyQuarterTurnRotations() {
        val bitmap = Bitmap.createBitmap(4, 2, Bitmap.Config.ARGB_8888)

        OcrImage.SUPPORTED_ROTATIONS.forEach { rotation ->
            assertEquals(rotation, OcrImage.fromBitmap(bitmap, rotation).rotationDegrees)
        }
        listOf(-90, 45, 360).forEach { rotation ->
            assertThrows(IllegalArgumentException::class.java) {
                OcrImage.fromBitmap(bitmap, rotation)
            }
            assertThrows(IllegalArgumentException::class.java) {
                OcrImage.fromEncodedBytes(byteArrayOf(1), rotation)
            }
        }
    }

    @Test
    fun reportsUprightDimensionsForQuarterTurns() {
        val bitmap = Bitmap.createBitmap(40, 20, Bitmap.Config.ARGB_8888)

        val upright = OcrImage.fromBitmap(bitmap, rotationDegrees = 180)
        val sideways = OcrImage.fromBitmap(bitmap, rotationDegrees = 270)

        assertFalse(upright.isQuarterTurn)
        assertEquals(40 to 20, upright.uprightWidth to upright.uprightHeight)
        assertTrue(sideways.isQuarterTurn)
        assertEquals(40 to 20, sideways.width to sideways.height)
        assertEquals(20 to 40, sideways.uprightWidth to sideways.uprightHeight)
    }

    @Test
    fun rejectsRecycledBitmapsAndEmptyBytes() {
        val recycled = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).also { it.recycle() }

        assertThrows(IllegalArgumentException::class.java) { OcrImage.fromBitmap(recycled) }
        assertThrows(IllegalArgumentException::class.java) {
            OcrImage.fromEncodedBytes(ByteArray(0))
        }
    }

    @Test
    fun encodedBytesAreCopiedAndNeverRendered() {
        val source = "SYNTHETIC-MRN-000".toByteArray()
        val image = OcrImage.fromEncodedBytes(source, rotationDegrees = 90)

        source.fill(0)
        val copy = image.bytes()
        copy.fill(0)

        assertArrayEquals("SYNTHETIC-MRN-000".toByteArray(), image.bytes())
        assertEquals(17, image.byteCount)
        assertFalse(image.toString().contains("SYNTHETIC"))
        assertEquals(
            "OcrImage.EncodedImage(byteCount=17, rotationDegrees=90)",
            image.toString(),
        )
    }

    @Test
    fun decodesEncodedBytesWithThePlatformDecoder() {
        val png = SyntheticImages.encode(width = 12, height = 8, format = "png")

        val bitmap = OcrImage.fromEncodedBytes(png).decodeBitmap()

        assertEquals(12, bitmap.width)
        assertEquals(8, bitmap.height)
    }

    @Test
    fun bitmapImageDescriptionCarriesOnlyGeometry() {
        val image = OcrImage.fromBitmap(Bitmap.createBitmap(3, 5, Bitmap.Config.ARGB_8888), 90)

        assertEquals(
            "OcrImage.BitmapImage(width=3, height=5, rotationDegrees=90)",
            image.toString(),
        )
    }
}
