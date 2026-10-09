package com.openmed.openmedkit.parity

import com.openmed.openmedkit.multimodal.AbstentionReason
import com.openmed.openmedkit.multimodal.AbstentionRecord
import com.openmed.openmedkit.multimodal.AssetManifest
import com.openmed.openmedkit.multimodal.ManifestProfile
import com.openmed.openmedkit.multimodal.MetadataField
import com.openmed.openmedkit.multimodal.Modality
import com.openmed.openmedkit.multimodal.MultimodalContractException
import com.openmed.openmedkit.multimodal.PreflightFinding
import com.openmed.openmedkit.multimodal.ProviderResultEnvelope
import java.lang.reflect.Modifier
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Pure JVM tests: no Android framework, model, transport, or network. */
class MultimodalContractParityTest {
    private val fixture: JsonObject = javaClass.classLoader!!.getResourceAsStream(
        "multimodal_contracts_v1.json",
    )!!.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
    private val vectors = fixture.getValue("cases").jsonArray.map { it.jsonObject }
    private fun parse(kind: String, payload: String): String = when (kind) {
        "asset_manifest" -> AssetManifest.fromJson(payload).toJson()
        "manifest_profile" -> ManifestProfile.fromJson(payload).toJson()
        "preflight_finding" -> PreflightFinding.fromJson(payload).toJson()
        "abstention_record" -> AbstentionRecord.fromJson(payload).toJson()
        "provider_result" -> ProviderResultEnvelope.fromJson(payload).toJson()
        else -> error("unsupported fixture kind")
    }
    private fun kind(vector: JsonObject) = vector.getValue("kind").jsonPrimitive.content
    private fun payload(vector: JsonObject) = vector.getValue("canonical_json").jsonPrimitive.content
    private fun first(kind: String): JsonObject = Json.parseToJsonElement(payload(vectors.first { kind(it) == kind })).jsonObject
    private fun rejected(kind: String, fields: Map<String, JsonElement>) {
        val error = assertFailsWith<MultimodalContractException> { parse(kind, JsonObject(fields).toString()) }
        assertEquals("multimodal contract is invalid", error.message)
        assertEquals(null, error.cause)
    }

    @Test
    fun sharedVectorsRoundTripByteIdentically() {
        assertEquals(1, fixture.getValue("version").jsonPrimitive.int)
        assertTrue(fixture.getValue("synthetic").jsonPrimitive.boolean)
        assertEquals(106, vectors.size)
        for (vector in vectors) {
            val expected = payload(vector)
            assertContentEquals(expected.toByteArray(Charsets.UTF_8), parse(kind(vector), expected).toByteArray(Charsets.UTF_8))
            val reordered = JsonObject(Json.parseToJsonElement(expected).jsonObject.entries.reversed().associate { it.toPair() })
            assertEquals(expected, parse(kind(vector), " \n" + reordered.toString() + "\n "))
        }
    }

    @Test
    fun fieldsAreClosedAndPrivacySentinelsNeverReachErrors() {
        val forbidden = listOf("path", "url", "text", "message", "transcript", "prompt", "credentials", "pixels", "waveform", "dicom_values")
        for (kind in vectors.map(::kind).toSet()) {
            val fields = first(kind)
            for (key in forbidden) rejected(kind, fields + (key to JsonPrimitive("SYNTH_PRIVATE_SENTINEL /private/a https://invalid.test")))
            val key = when (kind) {
                "asset_manifest", "manifest_profile" -> "version"
                "abstention_record", "provider_result" -> "schema_version"
                else -> null // Findings inherit the containing preflight schema, and have no version field.
            }
            if (key != null) {
                rejected(kind, fields + (key to JsonPrimitive("unknown-version")))
                rejected(kind, fields + (key to JsonPrimitive(2)))
            } else {
                rejected(kind, fields + ("schema_version" to JsonPrimitive(2)))
            }
        }
        for (value in listOf("/private/a", "C:\\private\\a", "https://invalid.test", "SYNTH PRIVATE TEXT", "~/a")) {
            rejected("asset_manifest", first("asset_manifest") + ("asset_id" to JsonPrimitive(value)))
            rejected("provider_result", first("provider_result") + ("provider_id" to JsonPrimitive(value)))
        }
        for (value in listOf("patient-001", "bearer-synth", "model-secret", "mrn-001")) {
            rejected("provider_result", first("provider_result") + ("model_id" to JsonPrimitive(value)))
        }
    }

    @Test
    fun unknownReasonsAndInvalidCombinationsFailClosed() {
        rejected("preflight_finding", first("preflight_finding") + ("reason_code" to JsonPrimitive("unsupported_reason")))
        rejected("abstention_record", first("abstention_record") + ("reason" to JsonPrimitive("unknown")))
        val provider = Json.parseToJsonElement(payload(vectors.first {
            kind(it) == "provider_result" && payload(it).contains("\"outcome\":\"abstention\"")
        })).jsonObject
        rejected("provider_result", provider + ("abstention_code" to JsonPrimitive("unknown")))
        rejected("provider_result", provider + ("abstention_code" to JsonPrimitive("provider_unavailable")))
        rejected("provider_result", provider + ("output_digest" to JsonPrimitive("a".repeat(64))))
        rejected("provider_result", first("provider_result") - "output_digest")
        rejected("provider_result", first("provider_result") + ("count_metadata" to buildJsonObject { put("unknown", 1) }))
        val allowed = mapOf(
            "preflight" to setOf("unsupported_media", "resource_limit", "provider_unavailable"),
            "decode" to setOf("malformed_media", "resource_limit", "low_quality"),
            "inference" to setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability", "provider_unavailable"),
            "post_process" to setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability"),
        )
        for ((stage, reasons) in allowed) for (reason in AbstentionReason.values()) {
            val fields = first("abstention_record") + mapOf("stage" to JsonPrimitive(stage), "reason" to JsonPrimitive(reason.code))
            if (reason.code in reasons) parse("abstention_record", JsonObject(fields).toString()) else rejected("abstention_record", fields)
        }
        for (vector in vectors.filter { kind(it) == "preflight_finding" }) {
            val fields = Json.parseToJsonElement(payload(vector)).jsonObject
            rejected("preflight_finding", fields + ("field_name" to JsonPrimitive("unknown")))
            rejected("preflight_finding", fields + ("check" to JsonPrimitive("unknown")))
            if (fields["limit"] == JsonNull) rejected("preflight_finding", fields + ("limit" to JsonPrimitive(1)))
        }
    }

    @Test
    fun malformedJsonDuplicatesAndNumericBoundsFailClosed() {
        for (kind in vectors.map(::kind).toSet()) {
            for (bad in listOf("[]", "null", "{", "{\"text\":NaN}", "{\"text\":Infinity}", "{" + "\"text\":[] ,".repeat(7000) + "\"x\":1}")) {
                assertFailsWith<MultimodalContractException> { parse(kind, bad) }
            }
            val valid = first(kind)
            val key = valid.keys.first()
            val duplicate = valid.toString().dropLast(1) + ",\"$key\":" + valid.getValue(key) + "}"
            assertFailsWith<MultimodalContractException> { parse(kind, duplicate) }
            val requiredKey = when (kind) {
                "asset_manifest" -> "asset_id"
                "manifest_profile" -> "modality"
                "preflight_finding" -> "check"
                "abstention_record" -> "stage"
                else -> "provider_id"
            }
            rejected(kind, valid - requiredKey)
        }
        val asset = first("asset_manifest")
        for (bad in listOf(JsonPrimitive(true), JsonPrimitive("1"), JsonPrimitive(0), JsonPrimitive(-1), JsonPrimitive(1.0), JsonPrimitive(2147483648L))) {
            rejected("asset_manifest", asset + ("width" to bad))
        }
        for (bad in listOf("-1", "0", "1e309", "2147483648", "true", "\"1\"")) {
            rejected("asset_manifest", asset + ("duration_seconds" to Json.parseToJsonElement(bad)))
        }
        for (bad in listOf("-1", "1e309", "86400001", "true", "\"1\"")) {
            rejected("provider_result", first("provider_result") + ("duration_ms" to Json.parseToJsonElement(bad)))
        }
        for (bad in listOf("-1", "1.0", "9223372036854775808", "true", "\"1\"")) {
            rejected("provider_result", first("provider_result") + ("count_metadata" to buildJsonObject { put("input_bytes", Json.parseToJsonElement(bad)) }))
        }
        val limitFinding = Json.parseToJsonElement(payload(vectors.first {
            kind(it) == "preflight_finding" && payload(it).contains("\"reason_code\":\"insufficient_metadata\"")
        })).jsonObject
        rejected("preflight_finding", limitFinding + ("observed" to JsonPrimitive(1)))
        rejected("preflight_finding", limitFinding + ("field_name" to JsonPrimitive("width")))
        for (bad in listOf("0", "-1", "1e309", "true", "\"1\"", "100000000000000000000000000000")) {
            rejected("preflight_finding", limitFinding + ("limit" to Json.parseToJsonElement(bad)))
        }
        val nestedDuplicate = first("provider_result").toString().replace("\"count_metadata\":{}", "\"count_metadata\":{\"input_bytes\":1,\"input_bytes\":2}")
        assertFailsWith<MultimodalContractException> { parse("provider_result", nestedDuplicate) }
    }

    @Test
    fun profileFieldsMatchPythonAndPublicTypesHaveNoContentFields() {
        for (modality in Modality.values()) {
            val profile = ManifestProfile.fromJson("{\"modality\":\"${modality.code}\",\"version\":\"1.0\"}")
            assertTrue(profile.optionalFields.isEmpty())
            assertEquals(MetadataField.values().toSet(), profile.requiredFields + profile.inapplicableFields)
            assertTrue((profile.requiredFields intersect profile.inapplicableFields).isEmpty())
        }
        val required = mapOf(
            Modality.IMAGE to setOf(MetadataField.WIDTH, MetadataField.HEIGHT),
            Modality.PDF to setOf(MetadataField.PAGES),
            Modality.DICOM to setOf(MetadataField.FRAMES, MetadataField.WIDTH, MetadataField.HEIGHT),
            Modality.AUDIO to setOf(MetadataField.DURATION_SECONDS),
        )
        for ((modality, expected) in required) {
            val profile = ManifestProfile.fromJson("{\"modality\":\"${modality.code}\",\"version\":\"1.0\"}")
            assertEquals(expected, profile.requiredFields)
        }
        val fields = mapOf(
            AssetManifest::class.java to setOf("assetId", "mediaType", "sha256", "byteSize", "metadata"),
            ManifestProfile::class.java to setOf("modality"),
            PreflightFinding::class.java to setOf("check", "reasonCode", "fieldName", "limit", "observed"),
            AbstentionRecord::class.java to setOf("stage", "reason"),
            ProviderResultEnvelope::class.java to setOf("providerId", "modelId", "inputDigest", "outputDigest", "outcome", "abstentionCode", "duration", "counts"),
        )
        val forbidden = setOf("path", "url", "text", "message", "transcript", "prompt", "credentials", "payload", "source")
        for ((type, allowed) in fields) {
            assertEquals(allowed, type.declaredFields.filterNot { Modifier.isStatic(it.modifiers) }.map { it.name }.toSet())
            assertFalse(type.methods.any { it.name.removePrefix("get").lowercase() in forbidden })
        }
    }
}
