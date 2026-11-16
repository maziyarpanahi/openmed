package com.openmed.openmedkit.parity

import com.openmed.openmedkit.multimodal.AbstentionReason
import com.openmed.openmedkit.multimodal.AbstentionRecord
import com.openmed.openmedkit.multimodal.AssetManifest
import com.openmed.openmedkit.multimodal.ManifestProfile
import com.openmed.openmedkit.multimodal.MetadataField
import com.openmed.openmedkit.multimodal.Modality
import com.openmed.openmedkit.multimodal.MultimodalAsset
import com.openmed.openmedkit.multimodal.MultimodalClockAlignment
import com.openmed.openmedkit.multimodal.MultimodalClockReason
import com.openmed.openmedkit.multimodal.MultimodalContractException
import com.openmed.openmedkit.multimodal.MultimodalContractRefusal
import com.openmed.openmedkit.multimodal.MultimodalPacket
import com.openmed.openmedkit.multimodal.MultimodalPacketException
import com.openmed.openmedkit.multimodal.MultimodalProjectionKind
import com.openmed.openmedkit.multimodal.MultimodalProvider
import com.openmed.openmedkit.multimodal.MultimodalTranscript
import com.openmed.openmedkit.multimodal.MultimodalVisual
import com.openmed.openmedkit.multimodal.PreflightFinding
import com.openmed.openmedkit.multimodal.ProviderResultEnvelope
import com.openmed.openmedkit.multimodal.ProviderResultOutcome
import java.lang.reflect.Modifier
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
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
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put
import org.junit.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
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
        assertEquals(167, vectors.size)
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
        val guarded = Json.parseToJsonElement(payload(vectors.first {
            kind(it) == "provider_result" && payload(it).contains("\"decision\":\"redact\"")
        })).jsonObject
        val guard = guarded.getValue("guard_metadata").jsonObject
        rejected("provider_result", guarded + ("guard_metadata" to JsonObject(guard + ("decision" to JsonPrimitive("withhold")))))
        rejected("provider_result", guarded + ("guard_metadata" to JsonObject(guard + ("text" to JsonPrimitive("SYNTH_PRIVATE_SENTINEL")))))
        rejected("provider_result", guarded + ("schema_version" to JsonPrimitive("openmed.multimodal.provider_result.v2")))
        val allowed = mapOf(
            "preflight" to setOf("unsupported_media", "resource_limit", "provider_unavailable", "consent_unavailable", "consent_withdrawn", "language_unsupported", "policy_refused"),
            "decode" to setOf("malformed_media", "resource_limit", "low_quality", "consent_withdrawn", "capture_interrupted"),
            "inference" to setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability", "provider_unavailable", "consent_withdrawn", "capture_interrupted", "language_unsupported", "language_uncertain", "policy_refused"),
            "post_process" to setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability", "consent_withdrawn", "capture_interrupted", "language_unsupported", "language_uncertain", "policy_refused"),
        )
        for ((stage, reasons) in allowed) for (reason in AbstentionReason.values()) {
            val fields = first("abstention_record") + mapOf(
                "stage" to JsonPrimitive(stage), "reason" to JsonPrimitive(reason.code),
                "schema_version" to JsonPrimitive(reason.schemaVersion),
            )
            if (reason.code in reasons) {
                parse("abstention_record", JsonObject(fields).toString())
                rejected("abstention_record", fields + ("schema_version" to JsonPrimitive(3 - reason.schemaVersion)))
            } else rejected("abstention_record", fields)
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
        val required = mapOf(
            Modality.IMAGE to setOf(MetadataField.WIDTH, MetadataField.HEIGHT),
            Modality.PDF to setOf(MetadataField.PAGES),
            Modality.DICOM to setOf(MetadataField.FRAMES, MetadataField.WIDTH, MetadataField.HEIGHT),
            Modality.AUDIO to setOf(MetadataField.DURATION_SECONDS),
            Modality.VIDEO to setOf(MetadataField.DURATION_SECONDS, MetadataField.FRAMES, MetadataField.HEIGHT, MetadataField.WIDTH),
            Modality.WAVEFORM to setOf(MetadataField.DURATION_SECONDS, MetadataField.CHANNEL_COUNT, MetadataField.SAMPLE_RATE_HZ, MetadataField.SAMPLES_PER_CHANNEL),
            Modality.DICOM_NON_PIXEL to emptySet(),
        )
        val waveformFields = setOf(MetadataField.CHANNEL_COUNT, MetadataField.SAMPLE_RATE_HZ, MetadataField.SAMPLES_PER_CHANNEL)
        assertEquals(Modality.values().toSet(), required.keys)
        for ((modality, expected) in required) {
            val waveform = modality in setOf(Modality.WAVEFORM, Modality.DICOM_NON_PIXEL)
            val version = if (waveform) "1.1" else "1.0"
            val fields = mapOf("modality" to JsonPrimitive(modality.code), "version" to JsonPrimitive(version))
            val profile = ManifestProfile.fromJson(JsonObject(fields).toString())
            assertEquals(expected, profile.requiredFields)
            assertTrue(profile.optionalFields.isEmpty())
            val classified = if (waveform) MetadataField.values().toSet() else MetadataField.values().toSet() - waveformFields
            assertEquals(classified - expected, profile.inapplicableFields)
            rejected("manifest_profile", fields + ("version" to JsonPrimitive(if (waveform) "1.0" else "1.1")))
        }
        val waveformAsset = first("asset_manifest") + mapOf(
            "media_type" to JsonPrimitive("application/edf"),
            "channel_count" to JsonPrimitive(2),
            "sample_rate_hz" to JsonPrimitive(125.5),
            "samples_per_channel" to JsonPrimitive(251),
        )
        rejected("asset_manifest", waveformAsset + ("dicom_subprofile" to JsonPrimitive("waveform")))
        rejected("asset_manifest", waveformAsset + ("channel_count" to JsonPrimitive(1.5)))
        rejected("asset_manifest", waveformAsset + ("sample_rate_hz" to JsonPrimitive(0)))
        rejected("asset_manifest", waveformAsset + ("samples_per_channel" to JsonPrimitive("251")))
        rejected("asset_manifest", waveformAsset + mapOf(
            "media_type" to JsonPrimitive("application/dicom"),
            "dicom_subprofile" to JsonPrimitive("SYNTH_PRIVATE_SENTINEL"),
        ))
        val fields = mapOf(
            AssetManifest::class.java to setOf("assetId", "mediaType", "sha256", "byteSize", "metadata"),
            ManifestProfile::class.java to setOf("modality"),
            PreflightFinding::class.java to setOf("check", "reasonCode", "fieldName", "limit", "observed"),
            AbstentionRecord::class.java to setOf("stage", "reason"),
            ProviderResultEnvelope::class.java to setOf("providerId", "modelId", "inputDigest", "outputDigest", "outcome", "abstentionCode", "duration", "counts", "guard"),
        )
        val forbidden = setOf("path", "url", "text", "message", "transcript", "prompt", "credentials", "payload", "source")
        for ((type, allowed) in fields) {
            assertEquals(allowed, type.declaredFields.filterNot { Modifier.isStatic(it.modifiers) }.map { it.name }.toSet())
            assertFalse(type.methods.any { it.name.removePrefix("get").lowercase() in forbidden })
        }
    }

    // Evidence packets (`openmed.multimodal.apple.v1`) shared with Python and Swift.
    private val packetFixture: JsonObject = javaClass.classLoader!!.getResourceAsStream(
        "multimodal_contract_v1.json",
    )!!.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
    private fun rows(name: String) = packetFixture.getValue(name).jsonArray.map { it.jsonObject }
    private fun id(row: JsonObject) = row.getValue("id").jsonPrimitive.content
    private fun wire(row: JsonObject) = row.getValue("wire").jsonPrimitive.content.toByteArray(Charsets.UTF_8)
    private fun packetRow(id: String) = rows("cases").first { id(it) == id }
    private fun longs(value: JsonElement?) = (value as? JsonArray)?.map { it.jsonPrimitive.long }
    private fun strings(value: JsonElement) = value.jsonArray.map { it.jsonPrimitive.content }

    private fun sorted(value: JsonElement): JsonElement = when (value) {
        is JsonObject -> JsonObject(value.toSortedMap().mapValues { sorted(it.value) })
        is JsonArray -> JsonArray(value.map(::sorted))
        else -> value
    }

    private fun sha256(text: String) = MessageDigest.getInstance("SHA-256")
        .digest(text.toByteArray(Charsets.US_ASCII))
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    /** Re-signs an edited ASCII fixture payload canonically, so only the edit is refused. */
    private fun resigned(id: String, edit: (JsonObject) -> Map<String, JsonElement>): ByteArray {
        val root = Json.parseToJsonElement(String(wire(packetRow(id)), Charsets.UTF_8)).jsonObject
        val payload = JsonObject(edit(root.getValue("payload").jsonObject))
        val body = sorted(JsonObject(root - "evidence_digest" + ("payload" to payload))).jsonObject
        val digest = JsonPrimitive(sha256(body.toString()))
        return sorted(JsonObject(body + ("evidence_digest" to digest))).toString().toByteArray(Charsets.US_ASCII)
    }

    private fun refused(expected: MultimodalContractRefusal, wire: ByteArray, label: String) {
        val error = assertFailsWith<MultimodalPacketException>(label) { MultimodalPacket.fromJson(wire) }
        assertEquals(expected, error.refusal, label)
        assertEquals(expected.code, error.message, label)
        assertNull(error.cause, label)
    }

    @Test
    fun sharedEvidencePacketsAdmitWithPythonAndSwiftDigests() {
        assertEquals(1, packetFixture.getValue("version").jsonPrimitive.int)
        assertTrue(packetFixture.getValue("synthetic_only").jsonPrimitive.boolean)
        val cases = rows("cases")
        val kinds = cases.map { it.getValue("expected").jsonObject.getValue("kind").jsonPrimitive.content }
        assertEquals(MultimodalProjectionKind.values().map { it.code }.toSet(), kinds.toSet())
        for (row in cases) {
            val id = id(row)
            val bytes = wire(row)
            val packet = MultimodalPacket.fromJson(bytes)
            val expected = row.getValue("expected").jsonObject
            assertEquals(row.getValue("digest").jsonPrimitive.content, packet.evidenceDigest, id)
            assertContentEquals(bytes, packet.toJson(), id)
            assertEquals(packet, MultimodalPacket.fromJson(String(bytes, Charsets.UTF_8)), id)
            assertEquals(expected.getValue("kind").jsonPrimitive.content, packet.kind.code, id)
            when (val projection = packet.projection) {
                is MultimodalAsset -> {
                    val manifest = projection.manifest
                    assertEquals(projection.assetRef, manifest.assetId, id)
                    expected["width"]?.let { assertEquals(it.jsonPrimitive.int, manifest.width, id) }
                    expected["height"]?.let { assertEquals(it.jsonPrimitive.int, manifest.height, id) }
                    expected["duration"]?.let {
                        assertEquals(it.jsonPrimitive.content.toDouble(), manifest.durationSeconds, id)
                    }
                    expected["bytes"]?.let { assertEquals(it.jsonPrimitive.content, manifest.byteSize.toString(), id) }
                }
                is MultimodalVisual -> {
                    val box = (expected["box"] as? JsonArray)?.map { it.jsonPrimitive.content.toDouble() }
                    val display = projection.displayBox?.let { listOf(it.left, it.top, it.right, it.bottom) }
                    assertEquals(box, display, id)
                    expected["synthetic_text"]?.let { text ->
                        val span = projection.scalarSpan!!
                        assertEquals(longs(expected["scalar_span"]), listOf(span.start, span.end), id)
                        val scalars = text.jsonPrimitive.content.codePoints().toArray()
                        val cited = String(scalars, span.start.toInt(), (span.end - span.start).toInt())
                        assertEquals("😀é", cited, id)
                    }
                }
                is MultimodalTranscript -> {
                    assertEquals(expected.getValue("finalized").jsonPrimitive.boolean, projection.finalized, id)
                    val samples = longs(expected["sample_bounds"])
                    assertEquals(samples, projection.sampleBounds?.let { listOf(it.start, it.end) }, id)
                    assertEquals(
                        longs(expected["current_span"]),
                        projection.currentScalarSpan?.let { listOf(it.start, it.end) },
                        id,
                    )
                    longs(expected["time_ms"])?.let {
                        assertEquals(it, projection.relativeMilliseconds?.let { time -> listOf(time.start, time.end) }, id)
                    }
                    if (samples == null) assertNull(projection.relativeMilliseconds, id)
                    expected["invalidation"]?.let {
                        assertEquals(it.jsonPrimitive.content, projection.invalidation?.code, id)
                    }
                }
                is MultimodalProvider -> {
                    val result = projection.result
                    val reason = (expected["reason"] as? JsonPrimitive)?.takeIf { it != JsonNull }?.content
                    assertEquals(expected.getValue("outcome").jsonPrimitive.content, result.outcome.code, id)
                    assertEquals(reason, result.abstentionCode?.code, id)
                    if (result.outcome != ProviderResultOutcome.SUCCESS) assertNull(result.outputDigest, id)
                    assertEquals(0.1, result.durationMs, id)
                }
                is MultimodalClockAlignment -> {
                    assertEquals(expected.getValue("state").jsonPrimitive.content, projection.state.code, id)
                    val overlap = projection.guaranteedOverlap?.let { listOf(it.start, it.end) }
                    val large = expected["overlap_strings"]?.let(::strings)
                    if (large != null) {
                        assertEquals(large, overlap?.map { it.toString() }, id)
                    } else {
                        assertEquals(longs(expected["overlap"]), overlap, id)
                    }
                    if (overlap == null) {
                        assertNull(projection.targetStart, id)
                        assertNull(projection.targetEnd, id)
                        assertNull(projection.uncertaintyNanoseconds, id)
                        assertTrue(projection.reasons.isNotEmpty(), id)
                    }
                }
            }
        }
    }

    @Test
    fun sharedNegativePacketsRefuseWithTheFixtureReason() {
        val negative = rows("negative_cases")
        assertEquals(
            MultimodalContractRefusal.values().map { it.code }.toSet(),
            negative.map { it.getValue("reason").jsonPrimitive.content }.toSet(),
        )
        for (row in negative) {
            val code = row.getValue("reason").jsonPrimitive.content
            refused(MultimodalContractRefusal.values().single { it.code == code }, wire(row), id(row))
        }
    }

    @Test
    fun packetVocabulariesMatchPythonAndSwift() {
        assertEquals(strings(packetFixture.getValue("provider_outcomes")), ProviderResultOutcome.values().map { it.code })
        assertEquals(
            strings(packetFixture.getValue("abstention_codes")),
            AbstentionReason.values().filter { it != AbstentionReason.PROVIDER_UNAVAILABLE }.map { it.code },
        )
        assertEquals(strings(packetFixture.getValue("clock_reasons")), MultimodalClockReason.values().map { it.code })
        assertEquals(
            listOf(
                "invalid_multimodal_contract", "unsupported_multimodal_schema", "unsupported_multimodal_capability",
                "invalid_multimodal_transform", "incomplete_multimodal_provenance",
            ),
            MultimodalContractRefusal.values().map { it.code },
        )
    }

    @Test
    fun packetBoundsCanonicalFormAndProvenanceFailClosed() {
        val valid = wire(packetRow("asset_image"))
        val text = String(valid, Charsets.UTF_8)
        val digest = packetRow("asset_image").getValue("digest").jsonPrimitive.content
        val bom = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte())
        val invalid = MultimodalContractRefusal.INVALID
        val schema = MultimodalContractRefusal.SCHEMA
        val provenance = MultimodalContractRefusal.PROVENANCE
        refused(invalid, ByteArray(0), "empty")
        refused(invalid, ByteArray(MultimodalPacket.MAX_WIRE_BYTES + 1) { ' '.code.toByte() }, "oversize")
        refused(invalid, valid.copyOf(valid.size - 1) + byteArrayOf(0xC3.toByte()), "malformed UTF-8")
        refused(invalid, bom + valid, "byte order mark")
        refused(invalid, " ".toByteArray() + valid, "leading whitespace")
        refused(invalid, text.replace("\"byte_size\":128", "\"byte_size\":1.28e2").toByteArray(), "exponent")
        // Python also parses -0 as the integer 0, so only the canonical comparison refuses it.
        val visual = String(wire(packetRow("visual_0")), Charsets.UTF_8)
        refused(invalid, visual.replace("\"page\":0", "\"page\":-0").toByteArray(), "negative zero integer")
        refused(invalid, text.replace("image/png", "image/p\\u006eg").toByteArray(), "escaped ASCII")
        refused(provenance, text.replace(digest, "0".repeat(64)).toByteArray(), "wrong digest")
        refused(provenance, text.replace(digest, digest.uppercase()).toByteArray(), "uppercase digest")
        // Canonically re-signed packets isolate one contract rule at a time.
        val cases = listOf(
            Triple("asset_image", invalid, mapOf("width" to JsonPrimitive(0))),
            Triple("asset_image", invalid, mapOf("media_type" to JsonPrimitive("text/plain"))),
            Triple("asset_image", schema, mapOf("source_schema" to JsonPrimitive(2))),
            Triple("asset_image", invalid, mapOf("source_schema" to JsonPrimitive(true))),
            Triple("asset_audio", invalid, mapOf("duration_f64" to JsonPrimitive("0000000000000000"))),
            Triple("visual_0", MultimodalContractRefusal.CAPABILITY, mapOf("kind" to JsonPrimitive("fused_claim"))),
            Triple("visual_0", invalid, mapOf("kind" to JsonArray(emptyList()))),
            Triple("visual_0", invalid, mapOf("span" to JsonNull)),
            Triple("visual_0", invalid, mapOf("width_f64" to JsonPrimitive("4000000000000000"))),
            Triple("visual_0", MultimodalContractRefusal.TRANSFORM, mapOf("rotation" to JsonPrimitive(90))),
            Triple("visual_0", MultimodalContractRefusal.TRANSFORM, mapOf("rotation" to JsonPrimitive(90.0))),
            Triple("transcript_partial", schema, mapOf("source_schema" to JsonPrimitive("openmed.transcript.revisions.v2"))),
            Triple("transcript_partial", provenance, mapOf("token_refs" to JsonArray(listOf(JsonPrimitive("z".repeat(32)))))),
            Triple("transcript_partial", invalid, mapOf("token_refs" to JsonArray(List(2) { JsonPrimitive("3".repeat(32)) }))),
            Triple("transcript_partial", invalid, mapOf("token_refs" to JsonArray(emptyList()))),
            Triple("transcript_partial", invalid, mapOf("original_revision" to JsonPrimitive(1))),
            Triple("transcript_partial", invalid, mapOf("current_span" to JsonArray(listOf(JsonPrimitive(1), JsonPrimitive(5))))),
            Triple("transcript_partial", invalid, mapOf("finalized" to JsonPrimitive(0))),
            Triple("provider_success_none", schema, mapOf("schema_version" to JsonPrimitive("openmed.multimodal.provider_result.v2"))),
            Triple("provider_success_none", invalid, mapOf("model_id" to JsonPrimitive("patient-model"))),
            Triple("provider_success_none", invalid, mapOf("count_metadata" to JsonObject(mapOf("text" to JsonPrimitive(1))))),
            Triple("provider_success_none", invalid, mapOf("duration_f64" to JsonPrimitive("4194997000000001"))),
            Triple("provider_success_none", provenance, mapOf("input_digest" to JsonPrimitive("short"))),
            Triple("provider_abstention_resource_limit", invalid, mapOf("abstention_code" to JsonPrimitive("provider_unavailable"))),
            Triple("clock_aligned", schema, mapOf("source_schema" to JsonPrimitive("openmed.multimodal.clock-alignment.v2"))),
            Triple("clock_aligned", provenance, mapOf("uncertainty_ns" to JsonPrimitive(1))),
            Triple("clock_aligned", invalid, mapOf("reasons" to JsonArray(listOf(JsonPrimitive("source-timing-missing"))))),
            Triple("clock_hold", invalid, mapOf("reasons" to JsonArray(emptyList()))),
            Triple("clock_aligned", invalid, mapOf("guaranteed_overlap_ns" to JsonArray(List(2) { JsonPrimitive(4000000000L) }))),
        )
        for ((index, case) in cases.withIndex()) {
            val (row, refusal, edit) = case
            refused(refusal, resigned(row) { it + edit }, "$row #$index")
        }
    }

    @Test
    fun packetRefusalsAndProjectionsCarryNoSourceContent() {
        val sentinel = "SYNTH_PRIVATE_SENTINEL /private/scan.png https://invalid.test"
        val attempts = listOf(
            resigned("asset_image") { it + ("media_type" to JsonPrimitive(sentinel)) },
            resigned("asset_image") { it + ("path" to JsonPrimitive(sentinel)) },
            resigned("visual_0") { it + ("unit" to JsonPrimitive(sentinel)) },
            resigned("transcript_partial") { it + ("invalidation" to JsonPrimitive(sentinel)) },
            resigned("provider_success_none") { it + ("provider_id" to JsonPrimitive(sentinel)) },
            resigned("clock_hold") { it + ("reasons" to JsonArray(listOf(JsonPrimitive(sentinel)))) },
            String(wire(packetRow("asset_image")), Charsets.UTF_8).replace("image/png", sentinel).toByteArray(),
        )
        for (attempt in attempts) {
            val error = assertFailsWith<MultimodalPacketException> { MultimodalPacket.fromJson(attempt) }
            assertFalse(error.toString().contains("SYNTH_PRIVATE_SENTINEL"))
            assertFalse(error.toString().contains("/private"))
            assertNull(error.cause)
        }
        val packet = MultimodalPacket.fromJson(wire(packetRow("visual_0")))
        packet.toJson()[0] = 0
        assertContentEquals(wire(packetRow("visual_0")), packet.toJson())
        assertEquals("MultimodalPacket(kind=visual)", packet.toString())
        assertNotEquals(packet, MultimodalPacket.fromJson(wire(packetRow("visual_90"))))
        val forbidden = setOf("path", "url", "text", "message", "transcript", "prompt", "credentials", "payload", "filename")
        val types = listOf(
            MultimodalPacket::class.java, MultimodalAsset::class.java, MultimodalVisual::class.java,
            MultimodalTranscript::class.java, MultimodalProvider::class.java, MultimodalClockAlignment::class.java,
        )
        for (type in types) {
            assertFalse(type.declaredFields.any { it.name.lowercase() in forbidden }, type.simpleName)
            assertFalse(type.methods.any { it.name.removePrefix("get").lowercase() in forbidden }, type.simpleName)
        }
    }
}
