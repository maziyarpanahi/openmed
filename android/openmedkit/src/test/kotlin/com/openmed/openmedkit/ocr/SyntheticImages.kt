package com.openmed.openmedkit.ocr

import java.io.ByteArrayOutputStream
import java.io.OutputStream

/**
 * Encodes blank synthetic images with the host JDK's ImageIO.
 *
 * Android unit tests compile against `android.jar`, which hides `java.awt` and
 * `javax.imageio`, but Robolectric runs on the host JDK where both exist. The
 * images contain no pixels beyond a solid fill, so they carry no document data.
 */
internal object SyntheticImages {
    private const val TYPE_INT_RGB = 1

    fun encode(width: Int, height: Int, format: String): ByteArray {
        val bufferedImageClass = Class.forName("java.awt.image.BufferedImage")
        val image = bufferedImageClass
            .getConstructor(Int::class.java, Int::class.java, Int::class.java)
            .newInstance(width, height, TYPE_INT_RGB)
        val output = ByteArrayOutputStream()
        val written = Class.forName("javax.imageio.ImageIO")
            .getMethod(
                "write",
                Class.forName("java.awt.image.RenderedImage"),
                String::class.java,
                OutputStream::class.java,
            )
            .invoke(null, image, format, output) as Boolean
        check(written) { "no ImageIO writer for $format" }
        return output.toByteArray()
    }
}
