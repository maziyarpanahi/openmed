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
            val numeric = setOf("pages", "width", "height", "frames", "duration_seconds")
            ContractJson.fields(fields, setOf("asset_id", "media_type", "sha256", "byte_size"), numeric + "version")
            if ("version" in fields) requireContract(ContractJson.int(fields["version"], COUNT_MAX).toInt() == 1)
            val id = ContractJson.string(fields["asset_id"])
            opaque(id, Regex("[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"))
            val media = ContractJson.string(fields["media_type"])
            opaque(media, Regex("[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*"))
            requireContract(media in setOf("application/pdf", "application/dicom", "application/dicom+json") ||
                media.startsWith("image/") || media.startsWith("audio/"))
            val digest = ContractJson.string(fields["sha256"])
            opaque(digest, DIGEST)
            val bytes = ContractJson.int(fields["byte_size"], BYTE_MAX, java.math.BigInteger.ONE).toLong()
            val values = numeric.filter { !absent(fields[it]) }.associateWith {
                if (it == "duration_seconds") ContractJson.numeric(fields[it], COUNT_MAX, positive = true)
                else JsonPrimitive(ContractJson.int(fields[it], COUNT_MAX, java.math.BigInteger.ONE).toInt())
            }
            return AssetManifest(id, media, digest, bytes, values)
        }
    }
}

/** Closed set of manifest metadata fields. */
enum class MetadataField(val code: String) {
    DURATION_SECONDS("duration_seconds"), FRAMES("frames"), HEIGHT("height"), PAGES("pages"), WIDTH("width");
}

/** Closed set of supported modalities. */
enum class Modality(val code: String) { IMAGE("image"), PDF("pdf"), DICOM("dicom"), AUDIO("audio") }

/** A v1.0 profile reference, as serialized in Python PreflightReport.metadata_profile. */
class ManifestProfile private constructor(val modality: Modality) {
    val version: String get() = "1.0"
    val requiredFields: Set<MetadataField> get() = when (modality) {
        Modality.IMAGE -> setOf(MetadataField.WIDTH, MetadataField.HEIGHT)
        Modality.PDF -> setOf(MetadataField.PAGES)
        Modality.DICOM -> setOf(MetadataField.FRAMES, MetadataField.WIDTH, MetadataField.HEIGHT)
        Modality.AUDIO -> setOf(MetadataField.DURATION_SECONDS)
    }
    val optionalFields: Set<MetadataField> get() = emptySet()
    val inapplicableFields: Set<MetadataField> get() = MetadataField.values().toSet() - requiredFields

    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = ContractJson.encode(obj("modality" to text(modality.code), "version" to text(version)))

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): ManifestProfile {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("modality", "version"))
            requireContract(ContractJson.string(fields["version"]) == "1.0")
            val modality = closed(fields["modality"], Modality.values()) { it.code }
            return ManifestProfile(modality)
        }
    }
}

/** The five checks, in the Python preflight order. */
enum class PreflightCheck(val code: String) {
    MANIFEST("manifest"), MEDIA_TYPE("media_type"), METADATA("metadata"), LIMITS("limits"), DIGEST("digest");
}

/** Controlled finding codes; check/field/value combinations are validated too. */
enum class PreflightReason(val code: String) {
    MALFORMED_MANIFEST("malformed_manifest"), MISMATCH("mismatch"), UNKNOWN("unknown"),
    UNSUPPORTED_MODALITY("unsupported_modality"), BYTE_COUNT_MISMATCH("byte_count_mismatch"),
    SHA256_MISMATCH("sha256_mismatch"), NOT_EVALUATED("not_evaluated"),
    INAPPLICABLE_PRESENT("inapplicable_present"), INVALID_BOOLEAN("invalid_boolean"), INVALID_TYPE("invalid_type"),
    INVALID_ZERO("invalid_zero"), MISSING_REQUIRED("missing_required"), NON_FINITE_NUMERIC("non_finite_numeric"),
    OUT_OF_RANGE("out_of_range"), LIMIT_EXCEEDED("limit_exceeded"), INSUFFICIENT_METADATA("insufficient_metadata");
}

/** Closed field names used by metadata, limit, and digest findings. */
enum class FindingField(val code: String) {
    BYTE_SIZE("byte_size"), PAGES("pages"), WIDTH("width"), HEIGHT("height"), FRAMES("frames"),
    DURATION_SECONDS("duration_seconds"), PIXELS("pixels"), TOTAL_PIXELS("total_pixels"), SHA256("sha256");
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
                    requireContract(field?.code in setOf("byte_size", "pages", "pixels", "total_pixels", "frames", "duration_seconds"))
                    requireContract(reason in setOf(PreflightReason.LIMIT_EXCEEDED, PreflightReason.INSUFFICIENT_METADATA))
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
                            PreflightReason.OUT_OF_RANGE,
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
enum class AbstentionReason(val code: String) {
    UNSUPPORTED_MEDIA("unsupported_media"), MALFORMED_MEDIA("malformed_media"), RESOURCE_LIMIT("resource_limit"),
    LOW_QUALITY("low_quality"), PHI_UNCERTAINTY("phi_uncertainty"), SPEAKER_UNCERTAINTY("speaker_uncertainty"),
    TEMPORAL_INSTABILITY("temporal_instability"), PROVIDER_UNAVAILABLE("provider_unavailable");
}

/** A v1 abstention record with the Python stage/reason invariants. */
class AbstentionRecord private constructor(val stage: AbstentionStage, val reason: AbstentionReason) {
    val schemaVersion: Int get() = 1
    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = ContractJson.encode(obj(
        "schema_version" to JsonPrimitive(schemaVersion), "stage" to text(stage.code), "reason" to text(reason.code),
    ))

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): AbstentionRecord {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("schema_version", "stage", "reason"))
            requireContract(ContractJson.int(fields["schema_version"], COUNT_MAX).toInt() == 1)
            val stage = closed(fields["stage"], AbstentionStage.values()) { it.code }
            val reason = closed(fields["reason"], AbstentionReason.values()) { it.code }
            val allowed = when (stage) {
                AbstentionStage.PREFLIGHT -> setOf("unsupported_media", "resource_limit", "provider_unavailable")
                AbstentionStage.DECODE -> setOf("malformed_media", "resource_limit", "low_quality")
                AbstentionStage.INFERENCE -> setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability", "provider_unavailable")
                AbstentionStage.POST_PROCESS -> setOf("resource_limit", "low_quality", "phi_uncertainty", "speaker_uncertainty", "temporal_instability")
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

/** Experimental v1 provider envelope; contains no clinical output or review decision. */
class ProviderResultEnvelope private constructor(
    val providerId: String,
    val modelId: String,
    val inputDigest: String,
    val outputDigest: String?,
    val outcome: ProviderResultOutcome,
    val abstentionCode: AbstentionReason?,
    private val duration: JsonPrimitive,
    private val counts: Map<ProviderCount, Long>,
) {
    val schemaVersion: String get() = "openmed.multimodal.provider_result.v1"
    val durationMs: Double get() = duration.content.toDouble()
    val countMetadata: Map<ProviderCount, Long> get() = counts.toMap()

    /** Return the deterministic, content-free JSON representation. */
    fun toJson(): String = ContractJson.encode(obj(
        "schema_version" to text(schemaVersion), "provider_id" to text(providerId), "model_id" to text(modelId),
        "input_digest" to text(inputDigest), "output_digest" to text(outputDigest), "outcome" to text(outcome.code),
        "abstention_code" to text(abstentionCode?.code), "duration_ms" to duration,
        "count_metadata" to JsonObject(counts.mapKeys { it.key.code }.mapValues { JsonPrimitive(it.value) }),
    ), sorted = true)

    companion object {
        /** Parse bounded strict JSON without retaining rejected content. */
        fun fromJson(payload: String): ProviderResultEnvelope {
            val fields = ContractJson.parse(payload)
            ContractJson.fields(fields, setOf("schema_version", "provider_id", "model_id", "input_digest", "outcome", "duration_ms"),
                setOf("output_digest", "abstention_code", "count_metadata"))
            requireContract(ContractJson.string(fields["schema_version"]) == "openmed.multimodal.provider_result.v1")
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
            return ProviderResultEnvelope(provider, model, input, output, outcome, reason, duration, counts)
        }
    }
}

internal fun <T> closed(value: JsonElement?, values: Array<T>, code: (T) -> String): T {
    val string = ContractJson.string(value)
    return values.firstOrNull { code(it) == string } ?: throw MultimodalContractException()
}
