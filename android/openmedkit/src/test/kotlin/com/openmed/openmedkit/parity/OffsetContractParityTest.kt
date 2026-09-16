package com.openmed.openmedkit.parity

import com.openmed.openmedkit.DeidentifyEngine
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.TokenClassificationDecoder
import com.openmed.openmedkit.TokenClassificationPrediction
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.policy.PolicyAction
import com.openmed.openmedkit.policy.PolicyProfile
import com.openmed.openmedkit.segmentation.AndroidIcuTextSegmentationAdapter
import com.openmed.openmedkit.segmentation.Icu4jTextSegmentationAdapter
import com.openmed.openmedkit.segmentation.IcuTextSegmenter
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OffsetContractParityTest {
    private val androidSegmenter = IcuTextSegmenter(
        AndroidIcuTextSegmentationAdapter(),
    )
    private val jvmSegmenter = IcuTextSegmenter(
        Icu4jTextSegmentationAdapter(),
    )

    @Test
    fun scalarAndUtf16ConversionsRejectSurrogatePairSplits() {
        val text = "A𠀀B"

        assertEquals(3, UnicodeOffsetContract.scalarLength(text))
        assertEquals(0, UnicodeOffsetContract.scalarToUtf16Index(text, 0))
        assertEquals(1, UnicodeOffsetContract.scalarToUtf16Index(text, 1))
        assertEquals(3, UnicodeOffsetContract.scalarToUtf16Index(text, 2))
        assertEquals(4, UnicodeOffsetContract.scalarToUtf16Index(text, 3))
        assertEquals(2, UnicodeOffsetContract.utf16ToScalarOffset(text, 3))
        assertFailsWith<IllegalArgumentException> {
            UnicodeOffsetContract.utf16ToScalarOffset(text, 2)
        }
    }

    @Test
    fun sharedFixtureMatchesAndroidAndJvmOffsetContracts() {
        val fixture = loadFixture()
        val cases = fixture.getJSONArray("cases")

        assertEquals(1, fixture.getInt("version"))
        assertEquals("unicode_scalar", fixture.getString("offset_unit"))
        assertTrue(cases.length() >= 52)

        val supplementaryCases = cases.objects().filter {
            it.getString("id").startsWith("cjk-extension-b-")
        }
        assertTrue(supplementaryCases.size >= 5)
        assertTrue(
            supplementaryCases.all {
                val text = it.getString("text")
                text.length > UnicodeOffsetContract.scalarLength(text)
            },
        )

        for (fixtureCase in cases.objects()) {
            assertCaseParity(fixtureCase)
        }
    }

    private fun assertCaseParity(fixtureCase: JSONObject) {
        val id = fixtureCase.getString("id")
        val text = fixtureCase.getString("text")
        val inputStart = fixtureCase.getInt("input_start")
        val inputEnd = fixtureCase.getInt("input_end")
        val expectedStart = fixtureCase.getInt("expected_start")
        val expectedEnd = fixtureCase.getInt("expected_end")
        val replacement = fixtureCase.getString("replacement")
        val expectedRedacted = fixtureCase.getString("expected_redacted")

        val androidSpan = androidSegmenter.snapScalarSpan(text, inputStart, inputEnd)
        val jvmSpan = jvmSegmenter.snapScalarSpan(text, inputStart, inputEnd)

        assertEquals(expectedStart, androidSpan.start, "$id Android start")
        assertEquals(expectedEnd, androidSpan.end, "$id Android end")
        assertEquals(androidSpan, jvmSpan, "$id Android/ICU4J parity")
        assertTrue(
            androidSegmenter.isGraphemeBoundary(text, androidSpan.start),
            "$id Android start boundary",
        )
        assertTrue(
            androidSegmenter.isGraphemeBoundary(text, androidSpan.end),
            "$id Android end boundary",
        )
        assertTrue(
            jvmSegmenter.isGraphemeBoundary(text, jvmSpan.start),
            "$id ICU4J start boundary",
        )
        assertTrue(
            jvmSegmenter.isGraphemeBoundary(text, jvmSpan.end),
            "$id ICU4J end boundary",
        )

        val utf16Span = UnicodeOffsetContract.utf16Span(
            text,
            androidSpan.start,
            androidSpan.end,
        )
        assertEquals(
            androidSpan.start,
            UnicodeOffsetContract.utf16ToScalarOffset(text, utf16Span.start),
            "$id scalar/UTF-16 start round trip",
        )
        assertEquals(
            androidSpan.end,
            UnicodeOffsetContract.utf16ToScalarOffset(text, utf16Span.end),
            "$id scalar/UTF-16 end round trip",
        )

        val entity = EntityPrediction(
            label = "NAME",
            text = "partial",
            confidence = 0.99f,
            start = inputStart,
            end = inputEnd,
        ).snappedToGraphemeBoundaries(text, androidSegmenter)
        assertEquals(expectedStart, entity.start, "$id entity start")
        assertEquals(expectedEnd, entity.end, "$id entity end")
        assertEquals(
            UnicodeOffsetContract.substring(text, expectedStart, expectedEnd),
            entity.text,
            "$id entity text",
        )

        val decoded = TokenClassificationDecoder(androidSegmenter).decode(
            predictions = listOf(
                TokenClassificationPrediction(
                    label = "NAME",
                    text = "partial",
                    confidence = 0.99f,
                    start = inputStart,
                    end = inputEnd,
                ),
            ),
            sourceText = text,
        ).single()
        assertEquals(expectedStart, decoded.start, "$id decoded start")
        assertEquals(expectedEnd, decoded.end, "$id decoded end")

        val redacted = UnicodeOffsetContract.replaceScalarSpan(
            text,
            decoded.start,
            decoded.end,
            replacement,
        )
        assertEquals(expectedRedacted, redacted, "$id redacted text")
        assertContentEquals(
            expectedRedacted.toByteArray(Charsets.UTF_8),
            redacted.toByteArray(Charsets.UTF_8),
            "$id redacted UTF-8",
        )

        val policyResult = DeidentifyEngine(androidSegmenter).deidentify(
            text = text,
            entities = listOf(
                EntityPrediction(
                    label = "PERSON",
                    text = "partial",
                    confidence = 0.99f,
                    start = inputStart,
                    end = inputEnd,
                ),
            ),
            policy = maskingPolicy,
        )
        assertEquals(expectedStart, policyResult.actions.single().start, "$id policy start")
        assertEquals(expectedEnd, policyResult.actions.single().end, "$id policy end")
        assertEquals(
            UnicodeOffsetContract.replaceScalarSpan(
                text,
                expectedStart,
                expectedEnd,
                "[PERSON]",
            ),
            policyResult.redactedText,
            "$id policy redaction",
        )
    }

    private fun loadFixture(): JSONObject {
        val stream = requireNotNull(
            javaClass.classLoader?.getResourceAsStream("offset_contract.json"),
        ) {
            "shared offset_contract.json was not synced into Android test resources"
        }
        return stream.bufferedReader(Charsets.UTF_8).use {
            JSONObject(it.readText())
        }
    }

    private fun JSONArray.objects(): List<JSONObject> =
        (0 until length()).map(::getJSONObject)

    private val maskingPolicy = PolicyProfile(
        schemaVersion = 1,
        name = "offset_contract_test",
        posture = "test",
        thresholdProfile = "test",
        defaultAction = PolicyAction.MASK,
        defaultActionBias = "test",
        arbitrationMode = "test",
        strictNoLeak = false,
        safetySweepMandatory = false,
        keepMapping = false,
        reversibleId = false,
        forcedCascadeTiers = emptyList(),
        policyLabelActions = emptyMap(),
        actions = mapOf("PERSON" to PolicyAction.MASK),
    )
}
