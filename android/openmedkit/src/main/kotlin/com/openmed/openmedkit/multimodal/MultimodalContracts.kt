package com.openmed.openmedkit.multimodal

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** Experimental v1 asset metadata; no source handle, path, URL, or payload is retained. */
class AssetManifest private constructor(
    val assetId: String,
    val mediaType: String,
    val sha256: String,
    val byteSize: Long,
    private val metadata: Map<String, JsonPrimitive>,
) {
    val version: Int get() = 1
    val pages: Int? get() = metadata["pages"]?.content?.toInt()
    val width: Int? get() = metadata["width"]?.content?.toInt()
    val height: Int? get() = metadata["height"]?.content?.toInt()
    val frames: Int? get() = metadata["frames"]?.content?.toInt()
    val durationSeconds: Double? get() = metadata["duration_seconds"]?.content?.toDouble()
    val channelCount: Int? get() = metadata["channel_count"]?.content?.toInt()
    val sampleRateHz: Double? get() = metadata["sample_rate_hz"]?.content?.toDouble()
    val samplesPerChannel: Int? get() = metadata["samples_per_channel"]?.content?.toInt()
    val dicomSubprofile: String? get() = metadata["dicom_subprofile"]?.content

    /** Compact sorted JSON, matching Python AssetManifest.to_json(). */
    fun toJson(): String = ContractJson.encode(JsonObject(linkedMapOf(
        "version" to JsonPrimitive(version), "asset_id" to JsonPrimitive(assetId),
        "media_type" to JsonPrimitive(mediaType), "sha256" to JsonPrimitive(sha256),
        "byte_size" to JsonPrimitive(byteSize),
    ) + metadata), sorted = true)

    companion object {
        /** Parse strict metadata only. Unknown fields and versions fail closed. */
        fun fromJson(payload: String): AssetManifest {
            val fields = ContractJson.parse(payload)
            val numeric = MetadataField.values().map { it.code }.toSet()
            ContractJson.fields(fields, setOf("asset_id", "media_type", "sha256", "byte_size"), numeric + setOf("version", "dicom_subprofile"))
            if ("version" in fields) requireContract(ContractJson.int(fields["version"], COUNT_MAX).toInt() == 1)
            val id = ContractJson.string(fields["asset_id"])
            opaque(id, Regex("[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"))
            val media = ContractJson.string(fields["media_type"])
            opaque(media, Regex("[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*"))
            requireContract(media in setOf("application/pdf", "application/dicom", "application/dicom+json", "video/mp4", "application/edf", "application/wfdb") ||
                media.startsWith("image/") || media.startsWith("audio/"))
            val digest = ContractJson.string(fields["sha256"])
            opaque(digest, DIGEST)
            val bytes = ContractJson.int(fields["byte_size"], BYTE_MAX, java.math.BigInteger.ONE).toLong()
            val values = numeric.filter { !absent(fields[it]) }.associateWith {
                if (it == "duration_seconds" || it == "sample_rate_hz") {
                    ContractJson.numeric(fields[it], COUNT_MAX, positive = true)
                } else {
                    JsonPrimitive(ContractJson.int(fields[it], COUNT_MAX, java.math.BigInteger.ONE).toInt())
                }
            }.toMutableMap()
            if (!absent(fields["dicom_subprofile"])) {
                val subprofile = ContractJson.string(fields["dicom_subprofile"])
                requireContract(media == "application/dicom" && subprofile in setOf("waveform", "non_pixel"))
                values["dicom_subprofile"] = JsonPrimitive(subprofile)
            }
            return AssetManifest(id, media, digest, bytes, values)
        }
    }
}

/** Closed set of manifest metadata fields. */
enum class MetadataField(val code: String) {
    DURATION_SECONDS("duration_seconds"), FRAMES("frames"), HEIGHT("height"), PAGES("pages"), WIDTH("width"),
    CHANNEL_COUNT("channel_count"), SAMPLE_RATE_HZ("sample_rate_hz"), SAMPLES_PER_CHANNEL("samples_per_channel");
}

private val waveformFields = setOf(
    MetadataField.CHANNEL_COUNT, MetadataField.SAMPLE_RATE_HZ, MetadataField.SAMPLES_PER_CHANNEL,
)
private val legacyMetadataFields = MetadataField.values().toSet() - waveformFields

/** Closed set of supported modalities and their preflight profile versions. */
enum class Modality(val code: String, internal val profileVersion: String = "1.0") {
    IMAGE("image"), PDF("pdf"), DICOM("dicom"), AUDIO("audio"), VIDEO("video"),
    WAVEFORM("waveform", "1.1"), DICOM_NON_PIXEL("dicom_non_pixel", "1.1"),
}

/** A v1.0 or v1.1 profile reference from Python PreflightReport.metadata_profile. */
class ManifestProfile private constructor(val modality: Modality) {
    val version: String get() = modality.profileVersion
    val requiredFields: Set<MetadataField> get() = when (modality) {
        Modality.IMAGE -> setOf(MetadataField.WIDTH, MetadataField.HEIGHT)
        Modality.PDF -> setOf(MetadataField.PAGES)
        Modality.DICOM -> setOf(MetadataField.FRAMES, MetadataField.WIDTH, MetadataField.HEIGHT)
        Modality.AUDIO -> setOf(MetadataField.DURATION_SECONDS)
        Modality.VIDEO -> setOf(MetadataField.DURATION_SECONDS, MetadataField.FRAMES, MetadataField.HEIGHT, MetadataField.WIDTH)
        Modality.WAVEFORM -> waveformFields + MetadataField.DURATION_SECONDS
        Modality.DICOM_NON_PIXEL -> emptySet()
    }
    val optionalFields: Set<MetadataField> get() = emptySet()
    val inapplicableFields: Set<MetadataField>
        get() = (if (version == "1.0") legacyMetadataFields else MetadataField.values().toSet()) - requiredFields

    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = ContractJson.encode(obj("modality" to text(modality.code), "version" to text(version)))

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): ManifestProfile {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("modality", "version"))
            val modality = closed(fields["modality"], Modality.values()) { it.code }
            requireContract(ContractJson.string(fields["version"]) == modality.profileVersion)
            return ManifestProfile(modality)
        }
    }
}

/** Ordered checks used by generic and structural PDF preflight. */
enum class PreflightCheck(val code: String) {
    MANIFEST("manifest"), MEDIA_TYPE("media_type"), METADATA("metadata"), LIMITS("limits"), DIGEST("digest"),
    PDF_CONTENT("pdf_content");
}

/** Controlled finding codes; check/field/value combinations are validated too. */
enum class PreflightReason(val code: String, internal val pdfContent: Boolean = false) {
    MALFORMED_MANIFEST("malformed_manifest"), MISMATCH("mismatch"), UNKNOWN("unknown"),
    UNSUPPORTED_MODALITY("unsupported_modality"), BYTE_COUNT_MISMATCH("byte_count_mismatch"),
    SHA256_MISMATCH("sha256_mismatch"), NOT_EVALUATED("not_evaluated"),
    INAPPLICABLE_PRESENT("inapplicable_present"), INVALID_BOOLEAN("invalid_boolean"), INVALID_TYPE("invalid_type"),
    INVALID_ZERO("invalid_zero"), MISSING_REQUIRED("missing_required"), NON_FINITE_NUMERIC("non_finite_numeric"),
    OUT_OF_RANGE("out_of_range"), LIMIT_EXCEEDED("limit_exceeded"), INSUFFICIENT_METADATA("insufficient_metadata"),
    SAMPLE_COUNT_MISMATCH("sample_count_mismatch"), ARITHMETIC_SATURATED("arithmetic_saturated"),
    PDF_SIZE_LIMIT("pdf_size_limit", pdfContent = true),
    PDF_HEADER_MISSING("pdf_header_missing", pdfContent = true),
    PDF_ENCRYPTED("pdf_encrypted", pdfContent = true),
    PDF_OBJECT_LIMIT("pdf_object_limit", pdfContent = true),
    PDF_DECOMPRESSION_LIMIT("pdf_decompression_limit", pdfContent = true),
    PDF_OBJECT_STREAM_UNSUPPORTED("pdf_object_stream_unsupported", pdfContent = true),
    PDF_CATALOG_MISSING("pdf_catalog_missing", pdfContent = true),
    PDF_PAGE_TREE_INVALID("pdf_page_tree_invalid", pdfContent = true),
    PDF_PAGE_LIMIT("pdf_page_limit", pdfContent = true),
    PAGE_COUNT_MISMATCH("page_count_mismatch", pdfContent = true),
    MEDIA_BOX_MISSING("media_box_missing", pdfContent = true),
    MEDIA_BOX_INVALID("media_box_invalid", pdfContent = true),
    CROP_BOX_INVALID("crop_box_invalid", pdfContent = true),
    CROP_BOX_OUTSIDE_MEDIA_BOX("crop_box_outside_media_box", pdfContent = true),
    ROTATION_INVALID("rotation_invalid", pdfContent = true),
    PDF_INVENTORY_INVALID("pdf_inventory_invalid", pdfContent = true),
    PDF_ANNOTATIONS("pdf_annotations", pdfContent = true),
    PDF_FORM_VALUES("pdf_form_values", pdfContent = true),
    PDF_EMBEDDED_FILES("pdf_embedded_files", pdfContent = true),
    PDF_XFA("pdf_xfa", pdfContent = true),
    PDF_OPTIONAL_CONTENT("pdf_optional_content", pdfContent = true),
    PDF_JAVASCRIPT("pdf_javascript", pdfContent = true),
    PDF_OPEN_ACTION("pdf_open_action", pdfContent = true),
    PDF_LAUNCH_ACTION("pdf_launch_action", pdfContent = true),
    PDF_INCREMENTAL_REVISIONS("pdf_incremental_revisions", pdfContent = true),
    PDF_MEDIA_TYPE_REQUIRED("pdf_media_type_required", pdfContent = true);
}

/** Closed field names used by metadata, limit, and digest findings. */
enum class FindingField(val code: String) {
    BYTE_SIZE("byte_size"), PAGES("pages"), WIDTH("width"), HEIGHT("height"), FRAMES("frames"),
    DURATION_SECONDS("duration_seconds"), PIXELS("pixels"), TOTAL_PIXELS("total_pixels"), SHA256("sha256"),
    CHANNEL_COUNT("channel_count"), SAMPLE_RATE_HZ("sample_rate_hz"),
    SAMPLES_PER_CHANNEL("samples_per_channel"), TOTAL_SAMPLES("total_samples");
}

/** A bounded number retains integer/float spelling for byte-identical Python JSON. */
class FindingNumber internal constructor(private val value: JsonPrimitive) {
    init {
        // Enforce the content-free numeric boundary even for same-module callers.
        ContractJson.numeric(value, JSON_INTEGER_MAX)
    }
    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = value.toString()
    internal fun element(): JsonPrimitive = value
}

/** Content-free finding, matching Python PreflightFinding.to_dict() key order. */
class PreflightFinding private constructor(
    val check: PreflightCheck,
    val reasonCode: PreflightReason,
    val fieldName: FindingField?,
    val limit: FindingNumber?,
    val observed: FindingNumber?,
) {
    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = ContractJson.encode(obj(
        "check" to text(check.code), "reason_code" to text(reasonCode.code),
        "field_name" to text(fieldName?.code), "limit" to (limit?.element() ?: JsonNull),
        "observed" to (observed?.element() ?: JsonNull),
    ))

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): PreflightFinding {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("check", "reason_code"), setOf("field_name", "limit", "observed"))
            val check = closed(fields["check"], PreflightCheck.values()) { it.code }
            val reason = closed(fields["reason_code"], PreflightReason.values()) { it.code }
            val field = if (absent(fields["field_name"])) null else closed(fields["field_name"], FindingField.values()) { it.code }
            var limit: FindingNumber? = null
            var observed: FindingNumber? = null
            when (check) {
                PreflightCheck.LIMITS -> {
                    requireContract(field?.code in setOf(
                        "byte_size", "pages", "pixels", "total_pixels", "frames", "duration_seconds",
                        "channel_count", "sample_rate_hz", "samples_per_channel", "total_samples",
                    ))
                    requireContract(reason in setOf(
                        PreflightReason.LIMIT_EXCEEDED, PreflightReason.INSUFFICIENT_METADATA,
                        PreflightReason.ARITHMETIC_SATURATED,
                    ))
                    limit = FindingNumber(ContractJson.numeric(fields["limit"], PIXEL_MAX, positive = true))
                    if (!absent(fields["observed"])) observed = FindingNumber(ContractJson.numeric(fields["observed"], PIXEL_MAX))
                    requireContract(reason != PreflightReason.INSUFFICIENT_METADATA || observed == null)
                }
                PreflightCheck.METADATA -> {
                    if (reason == PreflightReason.UNSUPPORTED_MODALITY) requireContract(field == null)
                    else {
                        requireContract(field?.code in MetadataField.values().map { it.code })
                        requireContract(reason in setOf(
                            PreflightReason.INAPPLICABLE_PRESENT, PreflightReason.INVALID_BOOLEAN, PreflightReason.INVALID_TYPE,
                            PreflightReason.INVALID_ZERO, PreflightReason.MISSING_REQUIRED, PreflightReason.NON_FINITE_NUMERIC,
                            PreflightReason.OUT_OF_RANGE, PreflightReason.SAMPLE_COUNT_MISMATCH,
                        ))
                    }
                    requireContract(absent(fields["limit"]) && absent(fields["observed"]))
                }
                PreflightCheck.DIGEST -> {
                    when (reason) {
                        PreflightReason.BYTE_COUNT_MISMATCH -> {
                            requireContract(field == FindingField.BYTE_SIZE)
                            // Python permits arbitrary positive integers for this finding.
                            val max = JSON_INTEGER_MAX
                            limit = FindingNumber(JsonPrimitive(ContractJson.int(fields["limit"], max, java.math.BigInteger.ONE)))
                            if (!absent(fields["observed"])) observed = FindingNumber(JsonPrimitive(ContractJson.int(fields["observed"], max)))
                        }
                        PreflightReason.SHA256_MISMATCH -> requireContract(field == FindingField.SHA256)
                        PreflightReason.NOT_EVALUATED -> requireContract(field == null)
                        else -> requireContract(false)
                    }
                    if (reason != PreflightReason.BYTE_COUNT_MISMATCH) requireContract(absent(fields["limit"]) && absent(fields["observed"]))
                }
                PreflightCheck.PDF_CONTENT -> {
                    requireContract(reason.pdfContent && field == null)
                    requireContract(absent(fields["limit"]) && absent(fields["observed"]))
                }
                PreflightCheck.MANIFEST -> {
                    requireContract(reason == PreflightReason.MALFORMED_MANIFEST && field == null)
                    requireContract(absent(fields["limit"]) && absent(fields["observed"]))
                }
                PreflightCheck.MEDIA_TYPE -> {
                    requireContract(reason in setOf(PreflightReason.MISMATCH, PreflightReason.UNKNOWN) && field == null)
                    requireContract(absent(fields["limit"]) && absent(fields["observed"]))
                }
            }
            return PreflightFinding(check, reason, field, limit, observed)
        }
    }
}

/** Stable pipeline stages; no stage accepts arbitrary explanations. */
enum class AbstentionStage(val code: String) {
    PREFLIGHT("preflight"), DECODE("decode"), INFERENCE("inference"), POST_PROCESS("post_process");
}

/** Stable metadata-only abstention reasons. */
enum class AbstentionReason(val code: String, val schemaVersion: Int = 1) {
    UNSUPPORTED_MEDIA("unsupported_media"), MALFORMED_MEDIA("malformed_media"), RESOURCE_LIMIT("resource_limit"),
    LOW_QUALITY("low_quality"), PHI_UNCERTAINTY("phi_uncertainty"), SPEAKER_UNCERTAINTY("speaker_uncertainty"),
    TEMPORAL_INSTABILITY("temporal_instability"), PROVIDER_UNAVAILABLE("provider_unavailable"),
    CONSENT_UNAVAILABLE("consent_unavailable", 2),
    CONSENT_WITHDRAWN("consent_withdrawn", 2),
    CAPTURE_INTERRUPTED("capture_interrupted", 2),
    LANGUAGE_UNSUPPORTED("language_unsupported", 2),
    LANGUAGE_UNCERTAIN("language_uncertain", 2),
    POLICY_REFUSED("policy_refused", 2);
}

/** A versioned abstention record with the Python stage/reason invariants. */
class AbstentionRecord private constructor(val stage: AbstentionStage, val reason: AbstentionReason) {
    val schemaVersion: Int get() = reason.schemaVersion
    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = ContractJson.encode(obj(
        "schema_version" to JsonPrimitive(schemaVersion), "stage" to text(stage.code), "reason" to text(reason.code),
    ))

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): AbstentionRecord {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("schema_version", "stage", "reason"))
            val stage = closed(fields["stage"], AbstentionStage.values()) { it.code }
            val reason = closed(fields["reason"], AbstentionReason.values()) { it.code }
            requireContract(ContractJson.int(fields["schema_version"], COUNT_MAX).toInt() == reason.schemaVersion)
            val allowed = when (stage) {
                AbstentionStage.PREFLIGHT -> setOf("unsupported_media", "resource_limit", "provider_unavailable", "consent_unavailable", "consent_withdrawn", "language_unsupported", "policy_refused")
                AbstentionStage.DECODE -> setOf("malformed_media", "resource_limit", "low_quality", "consent_withdrawn", "capture_interrupted")
                AbstentionStage.INFERENCE -> setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability", "provider_unavailable", "consent_withdrawn", "capture_interrupted", "language_unsupported", "language_uncertain", "policy_refused")
                AbstentionStage.POST_PROCESS -> setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability", "consent_withdrawn", "capture_interrupted", "language_unsupported", "language_uncertain", "policy_refused")
            }
            requireContract(reason.code in allowed)
            return AbstentionRecord(stage, reason)
        }
    }
}

/** Closed terminal provider outcomes. */
enum class ProviderResultOutcome(val code: String) {
    SUCCESS("success"), ABSTENTION("abstention"), PROVIDER_UNAVAILABLE("provider_unavailable"), VALIDATION_FAILURE("validation_failure");
}

/** Supported aggregate counts. Arbitrary metadata keys are not accepted. */
enum class ProviderCount(val code: String) {
    DETECTION_COUNT("detection_count"), FRAME_COUNT("frame_count"), INPUT_BYTES("input_bytes"), INPUT_ITEMS("input_items"),
    OUTPUT_ITEMS("output_items"), PAGE_COUNT("page_count"), SAMPLE_COUNT("sample_count"), SEGMENT_COUNT("segment_count"), TOKEN_COUNT("token_count");
}

/** Experimental versioned provider evidence with no clinical output. */
class ProviderResultEnvelope private constructor(
    val providerId: String,
    val modelId: String,
    val inputDigest: String,
    val outputDigest: String?,
    val outcome: ProviderResultOutcome,
    val abstentionCode: AbstentionReason?,
    private val duration: JsonPrimitive,
    private val counts: Map<ProviderCount, Long>,
    private val guard: Map<String, String>,
) {
    val schemaVersion: String get() = "openmed.multimodal.provider_result.v${abstentionCode?.schemaVersion ?: 1}"
    val guardMetadata: Map<String, String> get() = guard.toMap()
    val durationMs: Double get() = duration.content.toDouble()
    val countMetadata: Map<ProviderCount, Long> get() = counts.toMap()

    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String {
        val fields = obj(
            "schema_version" to text(schemaVersion), "provider_id" to text(providerId), "model_id" to text(modelId),
            "input_digest" to text(inputDigest), "output_digest" to text(outputDigest), "outcome" to text(outcome.code),
            "abstention_code" to text(abstentionCode?.code), "duration_ms" to duration,
            "count_metadata" to JsonObject(counts.mapKeys { it.key.code }.mapValues { JsonPrimitive(it.value) }),
        ).toMutableMap()
        if (guard.isNotEmpty()) fields["guard_metadata"] = JsonObject(guard.mapValues { JsonPrimitive(it.value) })
        return ContractJson.encode(JsonObject(fields), sorted = true)
    }

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): ProviderResultEnvelope {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("schema_version", "provider_id", "model_id", "input_digest", "outcome", "duration_ms"),
                setOf("output_digest", "abstention_code", "count_metadata", "guard_metadata"))
            fun identifier(name: String): String {
                val value = ContractJson.string(fields[name])
                opaque(value, Regex("[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?"))
                requireContract(value.split(Regex("[._-]+")).none {
                    it in setOf("bearer", "credential", "mrn", "password", "patient", "prompt", "secret", "token")
                })
                return value
            }
            fun digest(name: String): String = ContractJson.string(fields[name]).also { opaque(it, DIGEST) }
            val provider = identifier("provider_id")
            val model = identifier("model_id")
            val input = digest("input_digest")
            val output = if (absent(fields["output_digest"])) null else digest("output_digest")
            val outcome = closed(fields["outcome"], ProviderResultOutcome.values()) { it.code }
            val reason = if (absent(fields["abstention_code"])) null else closed(fields["abstention_code"], AbstentionReason.values()) { it.code }
            requireContract((outcome == ProviderResultOutcome.SUCCESS) == (output != null))
            requireContract((outcome == ProviderResultOutcome.ABSTENTION) == (reason != null))
            requireContract(reason != AbstentionReason.PROVIDER_UNAVAILABLE)
            val duration = ContractJson.float(fields["duration_ms"], 86400000.0)
            val countFields = fields["count_metadata"] ?: JsonObject(emptyMap())
            requireContract(countFields is JsonObject)
            val counts = (countFields as JsonObject).entries.associate {
                val key = closed(JsonPrimitive(it.key), ProviderCount.values()) { item -> item.code }
                key to ContractJson.int(it.value, BYTE_MAX).toLong()
            }
            val guardField = fields["guard_metadata"] ?: JsonObject(emptyMap())
            requireContract(guardField is JsonObject)
            val guard = (guardField as JsonObject).mapValues { ContractJson.string(it.value) }
            if (guard.isNotEmpty()) {
                val digestKeys = setOf("detector_version_digest", "policy_digest", "report_digest")
                requireContract(guard.keys == digestKeys + "decision")
                for (key in digestKeys) requireContract(DIGEST.matches(guard.getValue(key)))
                when (guard.getValue("decision")) {
                    "pass", "redact" -> requireContract(outcome == ProviderResultOutcome.SUCCESS)
                    "withhold", "abstain" -> requireContract(
                        outcome == ProviderResultOutcome.ABSTENTION && reason == AbstentionReason.PHI_UNCERTAINTY,
                    )
                    else -> requireContract(false)
                }
            }
            val envelope = ProviderResultEnvelope(provider, model, input, output, outcome, reason, duration, counts, guard)
            requireContract(ContractJson.string(fields["schema_version"]) == envelope.schemaVersion)
            return envelope
        }
    }
}

internal fun <T> closed(value: JsonElement?, values: Array<T>, code: (T) -> String): T {
    val string = ContractJson.string(value)
    return values.firstOrNull { code(it) == string } ?: throw MultimodalContractException()
}
