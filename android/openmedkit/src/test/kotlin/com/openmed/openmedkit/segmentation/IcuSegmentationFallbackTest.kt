package com.openmed.openmedkit.segmentation

import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.defaultTokenOffsets
import java.util.Locale
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class IcuSegmentationFallbackTest {
    private val androidSegmenter = IcuTextSegmenter(
        AndroidIcuTextSegmentationAdapter(),
    )
    private val jvmSegmenter = IcuTextSegmenter(
        Icu4jTextSegmentationAdapter(),
    )

    @Test
    fun defaultTokenOffsetsUseIcuFallbackWithoutSplittingSupplementaryScalars() {
        val text = "患者𠀀张伟因高血压入院"
        val offsets = defaultTokenOffsets(text)

        assertTrue(offsets.size > 1)
        for (offset in offsets) {
            val utf16End = offset.last + 1
            assertTrue(
                UnicodeOffsetContract.isCodePointBoundary(text, offset.first),
                "$text start=${offset.first}",
            )
            assertTrue(
                UnicodeOffsetContract.isCodePointBoundary(text, utf16End),
                "$text end=$utf16End",
            )
            val scalarStart = UnicodeOffsetContract.utf16ToScalarOffset(
                text,
                offset.first,
            )
            val scalarEnd = UnicodeOffsetContract.utf16ToScalarOffset(
                text,
                utf16End,
            )
            assertTrue(androidSegmenter.isGraphemeBoundary(text, scalarStart))
            assertTrue(androidSegmenter.isGraphemeBoundary(text, scalarEnd))
        }
    }

    @Test
    fun zhFixtureCorpusHasNoMidClusterFallbackBoundaries() {
        val fixture = loadFixture("zh_segmentation_gold.json")
        val names = fixture.getJSONArray("names").strings()
        val conditions = fixture.getJSONArray("conditions").strings()
        val templates = fixture.getJSONArray("gold_tokens").strings()
        var sentenceCount = 0

        for (name in names) {
            for (condition in conditions) {
                val text = templates.joinToString(separator = "") {
                    it.replace("{name}", name).replace("{condition}", condition)
                }
                assertValidSegments(
                    text,
                    androidSegmenter.fallbackWordSegments(
                        text,
                        Locale.SIMPLIFIED_CHINESE,
                    ),
                )
                sentenceCount += 1
            }
        }

        assertEquals(200, sentenceCount)
    }

    @Test
    fun hindiAndTamilFallbacksEmitWholeGraphemeSegments() {
        val fixtures = listOf(
            "रोगी राहुल शर्मा" to Locale.forLanguageTag("hi"),
            "நோயாளி தமிழ் பெயர்" to Locale.forLanguageTag("ta"),
        )

        for ((text, locale) in fixtures) {
            val androidSegments = androidSegmenter.fallbackWordSegments(text, locale)
            val jvmSegments = jvmSegmenter.fallbackWordSegments(text, locale)

            assertValidSegments(text, androidSegments)
            assertValidSegments(text, jvmSegments)
            assertEquals(
                androidSegments.map { it.start to it.end },
                jvmSegments.map { it.start to it.end },
                "$text Android/ICU4J fallback parity",
            )
        }
    }

    private fun assertValidSegments(
        text: String,
        segments: List<ScalarTextSegment>,
    ) {
        assertTrue(segments.isNotEmpty(), text)
        assertEquals(segments.sortedBy { it.start }, segments, text)

        val covered = mutableSetOf<Int>()
        for (segment in segments) {
            assertEquals(
                UnicodeOffsetContract.substring(text, segment.start, segment.end),
                segment.text,
                "$text ${segment.start}:${segment.end}",
            )
            assertTrue(
                androidSegmenter.isGraphemeBoundary(text, segment.start),
                "$text start=${segment.start}",
            )
            assertTrue(
                androidSegmenter.isGraphemeBoundary(text, segment.end),
                "$text end=${segment.end}",
            )
            assertTrue(
                jvmSegmenter.isGraphemeBoundary(text, segment.start),
                "$text ICU4J start=${segment.start}",
            )
            assertTrue(
                jvmSegmenter.isGraphemeBoundary(text, segment.end),
                "$text ICU4J end=${segment.end}",
            )
            covered += segment.start until segment.end
        }

        val wordLikeOffsets = (0 until UnicodeOffsetContract.scalarLength(text)).filter {
            UnicodeOffsetContract.substring(text, it, it + 1)
                .codePoints()
                .anyMatch { codePoint -> Character.isLetterOrDigit(codePoint) }
        }
        assertTrue(covered.containsAll(wordLikeOffsets), text)
    }

    private fun loadFixture(name: String): JSONObject {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) {
            "$name was not synced into Android test resources"
        }
        return stream.bufferedReader(Charsets.UTF_8).use {
            JSONObject(it.readText())
        }
    }

    private fun org.json.JSONArray.strings(): List<String> =
        (0 until length()).map(::getString)
}
