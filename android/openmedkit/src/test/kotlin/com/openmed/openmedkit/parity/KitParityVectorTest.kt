package com.openmed.openmedkit.parity

import com.openmed.openmedkit.DeidentifiedSpanAction
import com.openmed.openmedkit.DeidentifyEngine
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.KitParity
import com.openmed.openmedkit.KitParityException
import com.openmed.openmedkit.KitParityResult
import com.openmed.openmedkit.PolicyDeidentificationResult
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.policy.PolicyAction
import com.openmed.openmedkit.policy.PolicyProfiles
import com.openmed.openmedkit.toKitParityResult
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.security.MessageDigest
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Runs the repository kit-parity golden in
 * `tests/fixtures/parity/kit_parity_vectors.json` (digest-pinned by
 * `fixtures/parity/span_parity_manifest.json`) through the Kotlin kit-parity
 * profile, the `PolicyDeidentificationResult` projection, and the policy
 * `DeidentifyEngine`. Robolectric supplies Android ICU for grapheme snapping.
 *
 * Failure messages carry case identifiers, field paths, offsets, labels, and
 * actions only.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class KitParityVectorTest {
    @Test
    fun vectorsDigestAndProfileMatchKotlinConstants() {
        val manifest = readJson(MANIFEST_FILE)
        val bytes = readBytes(VECTORS_FILE)
        val fixture = loadVectors(bytes, manifest)
        val profile = fixture.getJSONObject("profile")

        assertEquals(KitParity.ACTIONS, profile.getJSONArray("actions").strings())
        assertEquals(KitParity.CANONICAL_LABELS, profile.getJSONArray("canonical_labels").strings())
        assertEquals(
            KitParity.OFFSET_CONTRACT_VERSIONS,
            profile.getJSONArray("offset_contract_versions").ints().toSet(),
        )
        assertEquals(
            KitParity.DECODER_CONTRACT_VERSIONS,
            profile.getJSONArray("decoder_contract_versions").ints().toSet(),
        )
        assertTrue(fixture.getJSONArray("vectors").length() >= 13)
        assertTrue(
            manifest.getJSONObject("kit_parity").getJSONArray("consumers").objects().any {
                it.getString("surface") == "kotlin" &&
                    it.getString("test").endsWith("KitParityVectorTest.kt") &&
                    it.getString("gate").contains("KitParityVectorTest")
            },
        )

        val tampered = bytes.copyOf().also { it[it.size - 2] = 0x20 }
        assertEquals("digest_mismatch", assertFailsWith<FixtureException> { loadVectors(tampered, manifest) }.code)
        val future = JSONObject(manifest.toString())
        future.getJSONObject("kit_parity").put("version", 2)
        assertEquals("kit_vectors_unsupported", assertFailsWith<FixtureException> { loadVectors(bytes, future) }.code)
    }

    @Test
    fun policyVocabularyMatchesTheProfile() {
        for (label in KitParity.CANONICAL_LABELS) {
            assertEquals(label, PolicyProfiles.canonicalLabel(label))
        }
        for (raw in listOf("patient_name", "medical_record_number", "phone_number", "unknown_kind")) {
            assertTrue(PolicyProfiles.canonicalLabel(raw) in KitParity.CANONICAL_LABELS, raw)
        }
        assertEquals(KitParity.ACTIONS.toSet(), PolicyAction.values().map { it.wireName }.toSet())
    }

    @Test
    fun scalarUtf16AndUtf8ConversionsMatchDerivedOffsets() {
        val fixture = loadValidatedVectors()
        var sawSurrogateShift = false
        var sawMultibyteShift = false
        for (vector in fixture.getJSONArray("vectors").objects()) {
            val text = vector.getString("text")
            vector.getJSONArray("spans").objects().forEachIndexed { index, span ->
                val tag = "case=${vector.getString("id")} span=$index"
                listOf(
                    Triple(span.getInt("start"), span.getInt("utf8_start"), span.getInt("utf16_start")),
                    Triple(span.getInt("end"), span.getInt("utf8_end"), span.getInt("utf16_end")),
                ).forEach { (scalar, utf8, utf16) ->
                    assertEquals(utf16, KitParity.scalarToUtf16Offset(text, scalar), tag)
                    assertEquals(utf8, KitParity.scalarToUtf8Offset(text, scalar), tag)
                    assertEquals(scalar, KitParity.utf16ToScalarOffset(text, utf16), tag)
                    assertEquals(scalar, KitParity.utf8ToScalarOffset(text, utf8), tag)
                    assertEquals(utf16, UnicodeOffsetContract.scalarToUtf16Index(text, scalar), tag)
                    assertEquals(utf8, text.substring(0, utf16).toByteArray(Charsets.UTF_8).size, tag)
                }
                sawSurrogateShift = sawSurrogateShift || span.getInt("utf16_start") != span.getInt("start")
                sawMultibyteShift = sawMultibyteShift || span.getInt("utf8_start") != span.getInt("utf16_start")
            }
        }
        assertTrue(sawSurrogateShift, "vectors lack a UTF-16 surrogate shift")
        assertTrue(sawMultibyteShift, "vectors lack a multibyte UTF-8 shift")

        for (record in fixture.getJSONArray("offset_rejections").objects()) {
            val text = record.getString("text")
            val offset = record.getInt("offset")
            val error = assertFailsWith<KitParityException>(record.getString("id")) {
                when (record.getString("unit")) {
                    "scalar" -> KitParity.scalarToUtf16Offset(text, offset)
                    "utf8" -> KitParity.utf8ToScalarOffset(text, offset)
                    else -> KitParity.utf16ToScalarOffset(text, offset)
                }
            }
            assertEquals(record.getString("reason"), error.code, record.getString("id"))
        }
    }

    @Test
    fun policyResultsProjectToCanonicalBytes() {
        val vectors = loadValidatedVectors().getJSONArray("vectors").objects()
        var projected = 0
        for (vector in vectors) {
            val spans = vector.getJSONArray("spans").objects()
            if (spans.any { !it.has("confidence_ppm") }) {
                // Kotlin policy actions always carry a score; absent scores are
                // covered by the round-trip test.
                assertEquals("confidence_encoding", vector.getString("id"))
                continue
            }
            val actions = spans.asReversed().map { span ->
                DeidentifiedSpanAction(
                    label = span.getString("label"),
                    canonicalLabel = span.getString("canonical_label"),
                    action = PolicyAction.fromWireName(span.getString("action")),
                    start = span.getInt("start"),
                    end = span.getInt("end"),
                    confidence = span.getInt("confidence_ppm") / 1_000_000f,
                    replacement = if (span.has("replacement")) span.getString("replacement") else null,
                )
            }
            val result = PolicyDeidentificationResult(
                redactedText = vector.getString("deidentified_text"),
                policyName = "kit_parity_fixture",
                actions = actions,
            )
            val serialized = KitParity.serialize(result.toKitParityResult())

            assertEquals(
                emptyList<String>(),
                differences(vector.getString("id"), vector.getString("result_json"), serialized),
            )
            assertEquals(vector.getString("result_sha256"), sha256Hex(serialized.toByteArray(Charsets.UTF_8)))
            projected += 1
        }
        assertEquals(vectors.size - 1, projected)
    }

    @Test
    fun policyEngineDeidentificationMatchesVectors() {
        val engine = DeidentifyEngine()
        var ran = 0
        for (vector in loadValidatedVectors().getJSONArray("vectors").objects()) {
            if (!vector.has("engine")) {
                continue
            }
            val text = vector.getString("text")
            val config = vector.getJSONObject("engine")
            val entities = config.getJSONArray("detections").objects().map { detection ->
                EntityPrediction(
                    label = detection.getString("label"),
                    text = UnicodeOffsetContract.substring(text, detection.getInt("start"), detection.getInt("end")),
                    confidence = detection.getInt("confidence_ppm") / 1_000_000f,
                    start = detection.getInt("start"),
                    end = detection.getInt("end"),
                )
            }
            val result = engine.deidentify(text, entities, PolicyProfiles.load(config.getString("policy")))
            val id = vector.getString("id")

            assertEquals(
                emptyList<String>(),
                differences(id, vector.getString("result_json"), KitParity.serialize(result.toKitParityResult())),
            )
            assertTrue(
                result.redactedText.toByteArray(Charsets.UTF_8)
                    .contentEquals(vector.getString("deidentified_text").toByteArray(Charsets.UTF_8)),
                "case=$id de-identified UTF-8 bytes differ",
            )
            ran += 1
        }
        assertTrue(ran >= 10)
    }

    @Test
    fun canonicalResultJsonRoundTripsThroughTheStrictParser() {
        for (vector in loadValidatedVectors().getJSONArray("vectors").objects()) {
            val id = vector.getString("id")
            val bytes = vector.getString("result_json").toByteArray(Charsets.UTF_8)
            val parsed = KitParity.parse(bytes)
            val spans = vector.getJSONArray("spans").objects()

            assertTrue(KitParity.serialize(parsed).toByteArray(Charsets.UTF_8).contentEquals(bytes), id)
            assertEquals(
                spans.map { "${it.getString("label")}/${it.getString("canonical_label")}/${it.getString("action")}[${it.getInt("start")},${it.getInt("end")})" },
                parsed.spans.map { "${it.label}/${it.canonicalLabel}/${it.action}[${it.start},${it.end})" },
                id,
            )
            assertEquals(spans.map { it.has("confidence_ppm") }, parsed.spans.map { it.confidencePpm != null }, id)
            assertEquals(parsed.spans.map { it.action == "keep" }, parsed.spans.map { it.replacement == null }, id)
            assertEquals(
                vector.getString("deidentified_sha256"),
                sha256Hex(parsed.deidentifiedText.toByteArray(Charsets.UTF_8)),
                id,
            )
            KitParity.verify(parsed, vector.getString("text"))
        }
    }

    @Test
    fun rejectionsFailWithDocumentedValueFreeReasons() {
        val fixture = loadValidatedVectors()
        val sentinel = fixture.getJSONObject("profile").getString("privacy_sentinel")
        for (record in fixture.getJSONArray("rejections").objects()) {
            val id = record.getString("id")
            val error = assertFailsWith<KitParityException>(id) { KitParity.parse(record.getString("input")) }
            assertEquals(record.getString("reason"), error.code, id)
            assertEquals(record.getString("field_path"), error.fieldPath, id)
            assertEquals("${record.getString("reason")} at ${record.getString("field_path")}", error.message, id)
            assertFalse(error.toString().contains(sentinel), id)
        }
        for (record in fixture.getJSONArray("source_rejections").objects()) {
            val id = record.getString("id")
            val parsed = KitParity.parse(record.getString("input"))
            val error = assertFailsWith<KitParityException>(id) { KitParity.verify(parsed, record.getString("text")) }
            assertEquals(record.getString("reason"), error.code, id)
            assertEquals(record.getString("field_path"), error.fieldPath, id)
        }
    }

    @Test
    fun bundleContractsAreGatedBeforeResultsAreParsed() {
        val fixture = loadValidatedVectors()
        val payload = fixture.getJSONArray("vectors").objects()
            .single { it.getString("id") == "emoji_sequences" }
            .getString("result_json").toByteArray(Charsets.UTF_8)
        val garbage = fixture.getJSONArray("rejections").getJSONObject(0).getString("input")
        for (record in fixture.getJSONArray("compatibility").objects()) {
            val offsetVersion = record.getInt("offset_contract_version")
            val decoderVersion = record.getInt("decoder_contract_version")
            if (record.getString("outcome") == "accepted") {
                assertEquals(KitParity.parse(payload), KitParity.accept(payload, offsetVersion, decoderVersion))
                continue
            }
            for (candidate in listOf(payload, "not json".toByteArray(), garbage.toByteArray())) {
                val error = assertFailsWith<KitParityException>(record.getString("id")) {
                    KitParity.accept(candidate, offsetVersion, decoderVersion)
                }
                assertEquals(record.getString("outcome"), error.code, record.getString("id"))
            }
        }
    }

    @Test
    fun alteredOffsetLabelActionOrderOrBytesFailTheGate() {
        val vector = loadValidatedVectors().getJSONArray("vectors").objects()
            .single { it.getString("id") == "emoji_sequences" }
        val expected = vector.getString("result_json")
        val parsed = KitParity.parse(expected)
        val spans = parsed.spans
        val end = spans[0].end

        fun serialized(result: KitParityResult) = KitParity.serialize(result)

        assertEquals(
            listOf("case=emoji_sequences path=$.spans[0].end expected=$end actual=${end + 1}"),
            differences("emoji_sequences", expected, serialized(parsed.copy(spans = listOf(spans[0].copy(end = end + 1)) + spans.drop(1)))),
        )
        assertEquals(
            listOf("case=emoji_sequences path=$.spans[0].canonical_label expected=OTHER actual=PERSON"),
            differences("emoji_sequences", expected, serialized(parsed.copy(spans = listOf(spans[0].copy(canonicalLabel = "PERSON")) + spans.drop(1)))),
        )
        assertEquals(
            listOf("case=emoji_sequences path=$.spans[1].action expected=mask actual=redact"),
            differences("emoji_sequences", expected, serialized(parsed.copy(spans = listOf(spans[0], spans[1].copy(action = "redact"), spans[2])))),
        )
        val reordered = assertFailsWith<KitParityException> {
            serialized(parsed.copy(spans = listOf(spans[1], spans[0], spans[2])))
        }
        assertEquals("span_order_invalid at $.spans[1]", reordered.message)

        val changedBytes = parsed.copy(deidentifiedText = parsed.deidentifiedText + " ")
        val report = differences("emoji_sequences", expected, serialized(changedBytes))
        assertTrue(report[0].startsWith("case=emoji_sequences path=$.deidentified_text first_differing_utf8_byte="))
        assertFalse(report.joinToString().contains("Bo Reyes"))
        assertEquals(
            "deidentified_text_mismatch",
            assertFailsWith<KitParityException> { KitParity.verify(changedBytes, vector.getString("text")) }.code,
        )

        // UTF-16 coordinates passed off as scalar offsets must not verify.
        val utf16Spans = vector.getJSONArray("spans").objects().mapIndexed { index, span ->
            spans[index].copy(start = span.getInt("utf16_start"), end = span.getInt("utf16_end"))
        }
        assertNotEquals(expected, serialized(parsed.copy(spans = utf16Spans)))
        assertFailsWith<KitParityException> { KitParity.verify(parsed.copy(spans = utf16Spans), vector.getString("text")) }
    }

    // Fixture loading.

    private fun loadValidatedVectors(): JSONObject = loadVectors(readBytes(VECTORS_FILE), readJson(MANIFEST_FILE))

    private fun loadVectors(bytes: ByteArray, manifest: JSONObject): JSONObject {
        val entry = manifest.getJSONObject("kit_parity")
        val supported = entry.getString("schema") == VECTORS_SCHEMA &&
            entry.getInt("version") == 1 &&
            entry.getString("profile") == KitParity.PROFILE &&
            entry.getInt("profile_version") == KitParity.PROFILE_VERSION
        if (!supported) {
            throw FixtureException("kit_vectors_unsupported")
        }
        if (sha256Hex(bytes) != entry.getString("sha256")) {
            throw FixtureException("digest_mismatch")
        }
        val vectors = JSONObject(String(bytes, Charsets.UTF_8))
        val profile = vectors.getJSONObject("profile")
        if (vectors.getString("schema") != VECTORS_SCHEMA ||
            vectors.getInt("version") != 1 ||
            profile.getString("name") != KitParity.PROFILE ||
            profile.getInt("version") != KitParity.PROFILE_VERSION
        ) {
            throw FixtureException("kit_vectors_unsupported")
        }
        return vectors
    }

    private fun readJson(name: String): JSONObject = JSONObject(String(readBytes(name), Charsets.UTF_8))

    private fun readBytes(name: String): ByteArray {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) {
            "shared parity fixture $name was not registered as an Android test resource"
        }
        return stream.use { it.readBytes() }
    }

    private fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    /** Value-free drift report: case id, field path, offsets, labels, actions. */
    private fun differences(caseId: String, expected: String, actual: String): List<String> {
        if (expected.toByteArray(Charsets.UTF_8).contentEquals(actual.toByteArray(Charsets.UTF_8))) {
            return emptyList()
        }
        val want = KitParity.parse(expected)
        val got = KitParity.parse(actual)
        val out = mutableListOf<String>()
        val left = want.deidentifiedText.toByteArray(Charsets.UTF_8)
        val right = got.deidentifiedText.toByteArray(Charsets.UTF_8)
        if (!left.contentEquals(right)) {
            val offset = left.indices.firstOrNull { it >= right.size || left[it] != right[it] }
                ?: minOf(left.size, right.size)
            out += "case=$caseId path=$.deidentified_text first_differing_utf8_byte=$offset"
        }
        if (want.spans.size != got.spans.size) {
            return out + "case=$caseId path=$.spans count expected=${want.spans.size} actual=${got.spans.size}"
        }
        want.spans.zip(got.spans).forEachIndexed { index, (lhs, rhs) ->
            val prefix = "case=$caseId path=$.spans[$index]"
            listOf(
                Triple("action", lhs.action, rhs.action),
                Triple("canonical_label", lhs.canonicalLabel, rhs.canonicalLabel),
                Triple("confidence_ppm", lhs.confidencePpm?.toString() ?: "absent", rhs.confidencePpm?.toString() ?: "absent"),
                Triple("end", lhs.end.toString(), rhs.end.toString()),
                Triple("label", lhs.label, rhs.label),
                Triple("start", lhs.start.toString(), rhs.start.toString()),
            ).filter { it.second != it.third }.forEach { (field, wantValue, gotValue) ->
                out += "$prefix.$field expected=$wantValue actual=$gotValue"
            }
            if (lhs.replacement != rhs.replacement) {
                out += "$prefix.replacement"
            }
        }
        return out.ifEmpty { listOf("case=$caseId path=$ bytes differ") }
    }

    private fun JSONArray.objects(): List<JSONObject> = (0 until length()).map(::getJSONObject)

    private fun JSONArray.strings(): List<String> = (0 until length()).map(::getString)

    private fun JSONArray.ints(): List<Int> = (0 until length()).map(::getInt)

    private class FixtureException(val code: String) : IllegalStateException(code)

    private companion object {
        const val VECTORS_FILE = "kit_parity_vectors.json"
        const val MANIFEST_FILE = "span_parity_manifest.json"
        const val VECTORS_SCHEMA = "openmed.kit_parity_vectors"
    }
}
