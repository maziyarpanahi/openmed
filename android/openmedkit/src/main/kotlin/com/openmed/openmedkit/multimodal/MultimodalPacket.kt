package com.openmed.openmedkit.multimodal

import java.math.BigInteger
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.security.MessageDigest
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Controlled packet refusals shared with Python `ContractRefusal` and Swift
 * `MultimodalContractRefusal`. No rejected value is ever attached to a refusal.
 */
enum class MultimodalContractRefusal(val code: String) {
    /** Shape, bounds, canonical encoding, state or provider behavior is invalid. */
    INVALID("invalid_multimodal_contract"),

    /** Envelope or source schema is not supported. */
    SCHEMA("unsupported_multimodal_schema"),

    /** Projection kind or visual source kind is unsupported. */
    CAPABILITY("unsupported_multimodal_capability"),

    /** Rotation or its derived display geometry is inconsistent. */
    TRANSFORM("invalid_multimodal_transform"),

    /** A required digest or reference is absent, malformed or unbound. */
    PROVENANCE("incomplete_multimodal_provenance"),
}

/**
 * A content-free packet refusal.
 *
 * The message is the refusal code only. Filenames, source text, rejected
 * values and parser causes are never retained.
 */
class MultimodalPacketException internal constructor(
    val refusal: MultimodalContractRefusal,
) : IllegalArgumentException(refusal.code)

/** Metadata projections, independent of any inference engine. */
enum class MultimodalProjectionKind(val code: String) {
    ASSET("asset"),
    VISUAL("visual"),
    TRANSCRIPT("transcript"),
    PROVIDER("provider"),
    CLOCK("clock"),
}

/** Source kind of a visual projection. */
enum class MultimodalVisualKind(val code: String) {
    PAGE("page_geometry"),
    PIXELS("pixels"),
    OCR("ocr_span"),
}

/** Source coordinate unit; both use a top-left origin. */
enum class MultimodalSpatialUnit(val code: String) {
    PIXEL("pixel"),
    POINT("point"),
}

/** Why a registered transcript citation can no longer be remapped. */
enum class MultimodalSpanInvalidation(val code: String) {
    TEXT_CHANGED("cited_text_changed"),
    TOKEN_CHANGED("cited_token_changed"),
}

/** Clock alignment state. Alignment never implies clinical approval. */
enum class MultimodalClockState(val code: String) {
    ALIGNED("aligned-for-review"),
    REVIEW("review-required"),
}

/** Controlled reasons for withholding a cross-modal clock translation. */
enum class MultimodalClockReason(val code: String) {
    ANCHORS("insufficient-anchors"),
    INCONSISTENT("inconsistent-clock-anchors"),
    COVERAGE("clock-coverage-missing"),
    DISCONTINUITY("clock-discontinuity"),
    SPARSE("anchor-gap-exceeds-policy"),
    EPOCH("clock-epoch-mismatch"),
    DECLARATION("clock-declaration-mismatch"),
    WALL_DOMAIN("wall-reference-mismatch"),
    TIMING("source-timing-missing"),
    BINDING("source-binding-mismatch"),
    REFERENCE("invalid-source-reference"),
    UNFINALIZED("transcript-not-finalized"),
    EVIDENCE("source-requires-review"),
    UNCERTAINTY("alignment-uncertainty-exceeds-policy"),
    NO_OVERLAP("evidence-does-not-overlap"),
    POSSIBLE_OVERLAP("overlap-is-uncertain"),
    CHANGED("source-changed-during-alignment"),
}

/** Top-left source-space, half-open rectangle, in pixels or PDF points. */
data class MultimodalRectangle(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double,
)

/** Half-open scalar, sample or relative-time interval; its parent defines units. */
data class MultimodalInterval(val start: Long, val end: Long)

/** Inclusive uncertainty envelope in a monotonic clock, never wall time. */
data class MultimodalNanosecondBound(val lower: Long, val upper: Long)

/** A typed, validated packet payload. Only [MultimodalPacket] creates one. */
sealed interface MultimodalProjection {
    /** The projection kind declared by the packet. */
    val kind: MultimodalProjectionKind
}

/**
 * Asset metadata projection.
 *
 * [manifest] is the shared [AssetManifest] whose `assetId` is the opaque
 * [assetRef], never the original asset identifier.
 */
class MultimodalAsset internal constructor(
    val manifestDigest: String,
    val assetRef: String,
    val manifest: AssetManifest,
) : MultimodalProjection {
    override val kind: MultimodalProjectionKind get() = MultimodalProjectionKind.ASSET
}

/** Original source coordinates, kept separate from their checked display rotation. */
class MultimodalVisual internal constructor(
    val nodeRef: String,
    val sourceDigest: String,
    val visualKind: MultimodalVisualKind,
    val page: Int,
    val width: Double,
    val height: Double,
    val unit: MultimodalSpatialUnit,
    val sourceBox: MultimodalRectangle?,
    /** Unicode code-point offsets, never UTF-16 indices or bytes. */
    val scalarSpan: MultimodalInterval?,
    val rotation: Int,
    val displayWidth: Double,
    val displayHeight: Double,
    val displayBox: MultimodalRectangle?,
) : MultimodalProjection {
    override val kind: MultimodalProjectionKind get() = MultimodalProjectionKind.VISUAL
}

/** A registered transcript citation; invalidated citations have no current bounds. */
class MultimodalTranscript internal constructor(
    val snapshotDigest: String,
    val sessionRef: String,
    val segmentRef: String,
    val spanRef: String,
    val originalRevision: Int,
    val revision: Int,
    val originalScalarSpan: MultimodalInterval,
    tokenRefs: List<String>,
    val scalarLength: Int,
    val finalized: Boolean,
    val invalidation: MultimodalSpanInvalidation?,
    val currentScalarSpan: MultimodalInterval?,
    val relativeMilliseconds: MultimodalInterval?,
    val sampleRateHz: Int,
    /** Outward conversion to a declared grid, not acoustic alignment evidence. */
    val sampleBounds: MultimodalInterval?,
) : MultimodalProjection {
    /** Opaque token references, in citation order. */
    val tokenRefs: List<String> = tokenRefs.toList()

    override val kind: MultimodalProjectionKind get() = MultimodalProjectionKind.TRANSCRIPT
}

/** A provider outcome, carried as the shared [ProviderResultEnvelope]. */
class MultimodalProvider internal constructor(
    val result: ProviderResultEnvelope,
) : MultimodalProjection {
    override val kind: MultimodalProjectionKind get() = MultimodalProjectionKind.PROVIDER
}

/** Protected clock bounds; review states carry no translated timestamps. */
class MultimodalClockAlignment internal constructor(
    val sourceResultDigest: String,
    val state: MultimodalClockState,
    reasons: List<MultimodalClockReason>,
    val sourceReferenceDigest: String,
    val targetReferenceDigest: String,
    val sourceClockDigest: String,
    val targetClockDigest: String,
    val policyDigest: String,
    val targetStart: MultimodalNanosecondBound?,
    val targetEnd: MultimodalNanosecondBound?,
    /** Half-open guaranteed overlap in nanoseconds. */
    val guaranteedOverlap: MultimodalInterval?,
    val uncertaintyNanoseconds: Long?,
) : MultimodalProjection {
    /** Review reasons, in packet order; empty when aligned. */
    val reasons: List<MultimodalClockReason> = reasons.toList()

    override val kind: MultimodalProjectionKind get() = MultimodalProjectionKind.CLOCK
}

/**
 * Validated `openmed.multimodal.apple.v1` evidence packet.
 *
 * Admission requires the exact canonical bytes Python and Swift produce: compact,
 * ASCII-escaped JSON sorted by field name, at most 64 KiB, with every float as
 * 16 lowercase hex digits of a big-endian IEEE-754 binary64 value. A valid
 * packet is structural evidence only; it does not authenticate who produced it
 * and grants no custody, disclosure or clinical approval.
 */
class MultimodalPacket private constructor(
    private val wire: ByteArray,
    /** The typed projection; no unchecked projection can be supplied. */
    val projection: MultimodalProjection,
    /** Lowercase SHA-256 shared with Python and Swift, without a prefix. */
    val evidenceDigest: String,
) {
    /** The validated projection kind. */
    val kind: MultimodalProjectionKind get() = projection.kind

    /** Returns a copy of the exact admitted canonical bytes. */
    fun toJson(): ByteArray = wire.copyOf()

    override fun equals(other: Any?): Boolean = other is MultimodalPacket && wire.contentEquals(other.wire)

    override fun hashCode(): Int = wire.contentHashCode()

    override fun toString(): String = "MultimodalPacket(kind=${kind.code})"

    companion object {
        /** The only supported envelope schema. */
        const val SCHEMA: String = "openmed.multimodal.apple.v1"

        /** Largest admitted packet, in bytes. */
        const val MAX_WIRE_BYTES: Int = 65536

        /**
         * Admits a canonical packet.
         *
         * @throws MultimodalPacketException with a controlled refusal code.
         */
        fun fromJson(wire: ByteArray): MultimodalPacket {
            val bytes = wire.copyOf()
            try {
                val (projection, digest) = PacketWire.admit(bytes)
                return MultimodalPacket(bytes, projection, digest)
            } catch (refusal: MultimodalPacketException) {
                throw refusal
            } catch (_: Exception) {
                throw MultimodalPacketException(MultimodalContractRefusal.INVALID)
            }
        }

        /** Admits a canonical packet from its text; see the [ByteArray] overload. */
        fun fromJson(wire: String): MultimodalPacket = fromJson(wire.toByteArray(Charsets.UTF_8))
    }
}

/** Packet admission, mirroring `openmed.multimodal.apple_contract._decode`. */
private object PacketWire {
    private const val FUSED_SCHEMA = "openmed.multimodal.fused-evidence.v1"
    private const val TRANSCRIPT_SCHEMA = "openmed.transcript.revisions.v1"
    private const val CLOCK_SCHEMA = "openmed.multimodal.clock-alignment.v1"
    private const val PROVIDER_V1 = "openmed.multimodal.provider_result.v1"
    private const val PROVIDER_V2 = "openmed.multimodal.provider_result.v2"
    private val INTEGER = Regex("-?(0|[1-9][0-9]*)")
    private val HEX16 = Regex("[0-9a-f]{16}")
    private val HEX32 = Regex("[0-9a-f]{32}")
    private val HEX64 = Regex("[0-9a-f]{64}")
    private val INT64_MAX = BigInteger.valueOf(Long.MAX_VALUE)
    private val COORDINATE_MAX = BigInteger.valueOf(Int.MAX_VALUE.toLong())
    private val CLOCK_LIMIT = BigInteger.valueOf(9_000_000_000_000_000_000L)
    private val RATE_MAX = BigInteger.valueOf(384_000L)
    private val TIME_MAX = BigInteger.valueOf(86_400_000L)
    private val SAMPLE_MAX = TIME_MAX * RATE_MAX / BigInteger.valueOf(1000L)
    private val LIMIT_4096 = BigInteger.valueOf(4096L)
    private val LIMIT_65536 = BigInteger.valueOf(65536L)
    private val ROTATIONS = setOf(0, 90, 180, 270).map { BigInteger.valueOf(it.toLong()) }.toSet()
    private val UTF8_BOM = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte())

    private fun refuse(refusal: MultimodalContractRefusal = MultimodalContractRefusal.INVALID): Nothing =
        throw MultimodalPacketException(refusal)

    private fun check(condition: Boolean, refusal: MultimodalContractRefusal = MultimodalContractRefusal.INVALID) {
        if (!condition) refuse(refusal)
    }

    fun admit(wire: ByteArray): Pair<MultimodalProjection, String> {
        check(wire.isNotEmpty() && wire.size <= MultimodalPacket.MAX_WIRE_BYTES)
        val root = keys(parse(wire), "schema_version kind payload evidence_digest")
        check(text(root["schema_version"]) == MultimodalPacket.SCHEMA, MultimodalContractRefusal.SCHEMA)
        val kind = MultimodalProjectionKind.values().firstOrNull { it.code == text(root["kind"]) }
            ?: refuse(MultimodalContractRefusal.CAPABILITY)
        val digest = hex(root["evidence_digest"])
        val payload = root["payload"] as? JsonObject ?: refuse()
        val projection = when (kind) {
            MultimodalProjectionKind.ASSET -> asset(payload)
            MultimodalProjectionKind.VISUAL -> visual(payload)
            MultimodalProjectionKind.TRANSCRIPT -> transcript(payload)
            MultimodalProjectionKind.PROVIDER -> provider(payload)
            MultimodalProjectionKind.CLOCK -> clock(payload)
        }
        check(hash(JsonObject(root - "evidence_digest")) == digest, MultimodalContractRefusal.PROVENANCE)
        check(canonical(root).toByteArray(Charsets.US_ASCII).contentEquals(wire))
        return projection to digest
    }

    // Python decodes bytes after dropping a UTF-8 signature; the canonical check
    // still refuses the original bytes afterwards.
    private fun parse(wire: ByteArray): JsonElement {
        val body = if (wire.size >= 3 && wire.copyOfRange(0, 3).contentEquals(UTF8_BOM)) {
            wire.copyOfRange(3, wire.size)
        } else {
            wire
        }
        val decoded = Charsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(body))
            .toString()
        return try {
            ContractJson.parse(decoded)
        } catch (_: MultimodalContractException) {
            refuse()
        }
    }

    private fun keys(value: JsonElement?, names: String): JsonObject {
        check(value is JsonObject && value.keys == names.split(' ').toSet())
        return value as JsonObject
    }

    /** A JSON string's content, or null for any other JSON value. */
    private fun text(value: JsonElement?): String? =
        (value as? JsonPrimitive)?.takeIf { it.isString }?.content

    private fun isNull(value: JsonElement?): Boolean = value is JsonNull

    // Python raises TypeError (an invalid contract) for an unhashable member test.
    private fun <T> member(
        value: JsonElement?,
        values: Array<T>,
        code: (T) -> String,
        refusal: MultimodalContractRefusal,
    ): T {
        check(value !is JsonArray && value !is JsonObject)
        return values.firstOrNull { code(it) == text(value) } ?: refuse(refusal)
    }

    private fun integer(
        value: JsonElement?,
        minimum: BigInteger = BigInteger.ZERO,
        maximum: BigInteger = INT64_MAX,
    ): BigInteger {
        check(value is JsonPrimitive && !value.isString && INTEGER.matches(value.content))
        val result = BigInteger((value as JsonPrimitive).content)
        check(result >= minimum && result <= maximum)
        return result
    }

    private fun boolean(value: JsonElement?): Boolean {
        check(value is JsonPrimitive && !value.isString && value.content in setOf("true", "false"))
        return (value as JsonPrimitive).content == "true"
    }

    private fun hex(value: JsonElement?, pattern: Regex = HEX64): String {
        val result = text(value)
        check(result != null && pattern.matches(result), MultimodalContractRefusal.PROVENANCE)
        return result!!
    }

    private fun bits(value: Double): String =
        java.lang.Long.toHexString((if (value == 0.0) 0.0 else value).toRawBits()).padStart(16, '0')

    /** Decodes a float only from its canonical hex form; rejects nonfinite and -0. */
    private fun number(value: JsonElement?, maximum: Double = Int.MAX_VALUE.toDouble()): Double {
        val encoded = text(value)
        check(encoded != null && HEX16.matches(encoded))
        val result = java.lang.Double.longBitsToDouble(java.lang.Long.parseUnsignedLong(encoded!!, 16))
        check(result.isFinite() && result >= 0.0 && result <= maximum)
        check(bits(result) == encoded)
        return result
    }

    private fun pair(value: JsonElement?, maximum: BigInteger, allowEmpty: Boolean = false): MultimodalInterval {
        check(value is JsonArray && value.size == 2)
        val (first, last) = (value as JsonArray).map { integer(it, maximum = maximum) }
        check(if (allowEmpty) first <= last else first < last)
        return MultimodalInterval(first.toLong(), last.toLong())
    }

    // Python compares `source_schema == 1`, which also holds for 1.0 and true;
    // those still fail the later integer check as invalid contracts.
    private fun equalsOne(value: JsonElement?): Boolean {
        if (value !is JsonPrimitive || value.isString || value is JsonNull) return false
        val raw = value.content
        return when {
            raw == "true" -> true
            INTEGER.matches(raw) -> BigInteger(raw) == BigInteger.ONE
            else -> raw.toDoubleOrNull() == 1.0
        }
    }

    private fun asset(p: JsonObject): MultimodalAsset {
        keys(p, "source_schema manifest_digest asset_ref media_type sha256 byte_size pages width height frames duration_f64")
        check(equalsOne(p["source_schema"]), MultimodalContractRefusal.SCHEMA)
        val manifestDigest = hex(p["manifest_digest"])
        val assetRef = hex(p["asset_ref"], HEX32)
        val sha256 = hex(p["sha256"])
        val duration = if (isNull(p["duration_f64"])) null else number(p["duration_f64"])
        // Reuse the shared AssetManifest validation, as Python does.
        val fields = linkedMapOf<String, JsonElement>(
            "asset_id" to JsonPrimitive(assetRef),
            "media_type" to p.getValue("media_type"),
            "sha256" to JsonPrimitive(sha256),
            "byte_size" to p.getValue("byte_size"),
        )
        for (name in listOf("pages", "width", "height", "frames")) fields[name] = p.getValue(name)
        if (duration != null) fields["duration_seconds"] = JsonPrimitive(duration)
        val manifest = try {
            AssetManifest.fromJson(ContractJson.encode(JsonObject(fields)))
        } catch (_: MultimodalContractException) {
            refuse()
        }
        integer(p["source_schema"], BigInteger.ONE, BigInteger.ONE)
        return MultimodalAsset(manifestDigest, assetRef, manifest)
    }

    private fun rotated(box: List<Double>, width: Double, height: Double, rotation: Int): List<Double> {
        fun point(x: Double, y: Double): Pair<Double, Double> = when (rotation) {
            0 -> x to y
            90 -> (height - y) to x
            180 -> (width - x) to (height - y)
            else -> y to (width - x)
        }
        val corners = listOf(
            point(box[0], box[1]), point(box[0], box[3]), point(box[2], box[1]), point(box[2], box[3]),
        )
        return listOf(
            corners.minOf { it.first }, corners.minOf { it.second },
            corners.maxOf { it.first }, corners.maxOf { it.second },
        )
    }

    private fun visual(p: JsonObject): MultimodalVisual {
        keys(
            p,
            "source_schema node_ref source_digest kind page width_f64 height_f64 unit box_f64 span rotation " +
                "display_width_f64 display_height_f64 display_box_f64",
        )
        check(text(p["source_schema"]) == FUSED_SCHEMA, MultimodalContractRefusal.SCHEMA)
        val nodeRef = hex(p["node_ref"], HEX32)
        val sourceDigest = hex(p["source_digest"])
        val kind = member(p["kind"], MultimodalVisualKind.values(), { it.code }, MultimodalContractRefusal.CAPABILITY)
        val width = number(p["width_f64"])
        val height = number(p["height_f64"])
        val boxValue = p["box_f64"]
        val box = if (isNull(boxValue)) {
            null
        } else {
            check(boxValue is JsonArray && boxValue.size == 4)
            (boxValue as JsonArray).map { number(it) }
        }
        val span = if (isNull(p["span"])) null else pair(p["span"], COORDINATE_MAX)
        // EvidenceNode and PageGeometry invariants.
        val page = integer(p["page"], maximum = COORDINATE_MAX).toInt()
        check(width > 0.0 && height > 0.0)
        val unit = member(p["unit"], MultimodalSpatialUnit.values(), { it.code }, MultimodalContractRefusal.INVALID)
        if (kind == MultimodalVisualKind.PAGE) {
            check(box == null)
        } else {
            check(box != null && box[0] < box[2] && box[2] <= width && box[1] < box[3] && box[3] <= height)
        }
        check((kind == MultimodalVisualKind.OCR) == (span != null))
        val rotationValue = p["rotation"]
        val rotation = (rotationValue as? JsonPrimitive)
            ?.takeIf { !it.isString && INTEGER.matches(it.content) }
            ?.let { BigInteger(it.content) }
            ?.takeIf { it in ROTATIONS }
            ?.toInt()
            ?: refuse(MultimodalContractRefusal.TRANSFORM)
        val quarter = rotation == 90 || rotation == 270
        val displayWidth = if (quarter) height else width
        val displayHeight = if (quarter) width else height
        val display = box?.let { rotated(it, width, height, rotation) }
        if (display != null) {
            check(display[0] < display[2] && display[1] < display[3], MultimodalContractRefusal.TRANSFORM)
        }
        val expected: JsonElement = display?.let { values -> JsonArray(values.map { JsonPrimitive(bits(it)) }) }
            ?: JsonNull
        check(
            text(p["display_width_f64"]) == bits(displayWidth) &&
                text(p["display_height_f64"]) == bits(displayHeight) &&
                p["display_box_f64"] == expected,
            MultimodalContractRefusal.TRANSFORM,
        )
        fun rectangle(values: List<Double>) = MultimodalRectangle(values[0], values[1], values[2], values[3])
        return MultimodalVisual(
            nodeRef, sourceDigest, kind, page, width, height, unit, box?.let(::rectangle), span, rotation,
            displayWidth, displayHeight, display?.let(::rectangle),
        )
    }

    private fun transcript(p: JsonObject): MultimodalTranscript {
        keys(
            p,
            "source_schema snapshot_digest session_ref segment_ref span_ref original_revision revision " +
                "original_span token_refs length finalized invalidation current_span time_ms sample_rate_hz " +
                "sample_bounds",
        )
        check(text(p["source_schema"]) == TRANSCRIPT_SCHEMA, MultimodalContractRefusal.SCHEMA)
        val snapshot = hex(p["snapshot_digest"])
        val session = hex(p["session_ref"], HEX32)
        val segment = hex(p["segment_ref"], HEX32)
        val spanRef = hex(p["span_ref"], HEX32)
        val revision = integer(p["revision"], maximum = LIMIT_4096)
        val originalRevision = integer(p["original_revision"], maximum = revision)
        val originalSpan = pair(p["original_span"], LIMIT_65536)
        val refsValue = p["token_refs"]
        check(refsValue is JsonArray && refsValue.size in 1..4096)
        val refs = (refsValue as JsonArray).map { hex(it, HEX32) }
        check(refs.toSet().size == refs.size)
        val length = integer(p["length"], maximum = LIMIT_65536)
        val finalized = boolean(p["finalized"])
        val rate = integer(p["sample_rate_hz"], BigInteger.ONE, RATE_MAX)
        var invalidation: MultimodalSpanInvalidation? = null
        var current: MultimodalInterval? = null
        var times: MultimodalInterval? = null
        var samples: MultimodalInterval? = null
        if (!isNull(p["invalidation"])) {
            invalidation = member(
                p["invalidation"], MultimodalSpanInvalidation.values(), { it.code }, MultimodalContractRefusal.INVALID,
            )
            check(listOf("current_span", "time_ms", "sample_bounds").all { isNull(p[it]) })
        } else {
            current = pair(p["current_span"], length)
            times = pair(p["time_ms"], TIME_MAX, allowEmpty = true)
            // A declared target sample grid, not a new acoustic alignment claim.
            val thousand = BigInteger.valueOf(1000L)
            val start = BigInteger.valueOf(times.start) * rate / thousand
            val end = (BigInteger.valueOf(times.end) * rate + BigInteger.valueOf(999L)) / thousand
            samples = pair(p["sample_bounds"], SAMPLE_MAX, allowEmpty = true)
            check(BigInteger.valueOf(samples.start) == start && BigInteger.valueOf(samples.end) == end)
        }
        return MultimodalTranscript(
            snapshot, session, segment, spanRef, originalRevision.toInt(), revision.toInt(), originalSpan, refs,
            length.toInt(), finalized, invalidation, current, times, rate.toInt(), samples,
        )
    }

    private fun provider(p: JsonObject): MultimodalProvider {
        keys(
            p,
            "schema_version provider_id model_id input_digest output_digest outcome abstention_code " +
                "duration_f64 count_metadata",
        )
        val duration = number(p["duration_f64"], maximum = 86_400_000.0)
        val version = text(p["schema_version"])
        check(version == PROVIDER_V1 || version == PROVIDER_V2, MultimodalContractRefusal.SCHEMA)
        val code = text(p["abstention_code"])
        val expected = if (code == null) {
            PROVIDER_V1
        } else {
            val reason = AbstentionReason.values().firstOrNull {
                it != AbstentionReason.PROVIDER_UNAVAILABLE && it.code == code
            } ?: refuse()
            "openmed.multimodal.provider_result.v${reason.schemaVersion}"
        }
        check(version == expected, MultimodalContractRefusal.SCHEMA)
        hex(p["input_digest"])
        if (!isNull(p["output_digest"])) hex(p["output_digest"])
        // Reuse the shared ProviderResultEnvelope validation, as Python does.
        val fields = JsonObject(p - "duration_f64" + ("duration_ms" to JsonPrimitive(duration)))
        val result = try {
            ProviderResultEnvelope.fromJson(ContractJson.encode(fields))
        } catch (_: MultimodalContractException) {
            refuse()
        }
        return MultimodalProvider(result)
    }

    private fun clock(p: JsonObject): MultimodalClockAlignment {
        keys(
            p,
            "source_schema source_result_digest state reasons source_reference_digest target_reference_digest " +
                "source_clock_digest target_clock_digest policy_digest target_start_ns target_end_ns " +
                "guaranteed_overlap_ns uncertainty_ns",
        )
        check(text(p["source_schema"]) == CLOCK_SCHEMA, MultimodalContractRefusal.SCHEMA)
        val digests = listOf(
            "source_result_digest", "source_reference_digest", "target_reference_digest",
            "source_clock_digest", "target_clock_digest", "policy_digest",
        ).associateWith { hex(p[it]) }
        val rawReasons = p["reasons"]
        check(rawReasons is JsonArray && rawReasons.size <= MultimodalClockReason.values().size)
        val reasons = (rawReasons as JsonArray).map { value ->
            MultimodalClockReason.values().firstOrNull { it.code == text(value) } ?: refuse()
        }
        check(reasons.toSet().size == reasons.size)
        val state = member(p["state"], MultimodalClockState.values(), { it.code }, MultimodalContractRefusal.INVALID)
        val timing = listOf("target_start_ns", "target_end_ns", "guaranteed_overlap_ns", "uncertainty_ns")
        var start: MultimodalNanosecondBound? = null
        var end: MultimodalNanosecondBound? = null
        var overlap: MultimodalInterval? = null
        var uncertainty: Long? = null
        if (state == MultimodalClockState.REVIEW) {
            check(reasons.isNotEmpty() && timing.all { isNull(p[it]) })
        } else {
            check(reasons.isEmpty())
            val bounds = listOf("target_start_ns", "target_end_ns").map { name ->
                val bound = keys(p[name], "lower upper")
                val lower = integer(bound["lower"], CLOCK_LIMIT.negate(), CLOCK_LIMIT)
                lower to integer(bound["upper"], lower, CLOCK_LIMIT)
            }
            val overlapValue = p["guaranteed_overlap_ns"]
            check(overlapValue is JsonArray && overlapValue.size == 2)
            val items = overlapValue as JsonArray
            val first = integer(items[0], bounds[0].second, bounds[1].first)
            val last = integer(items[1], first + BigInteger.ONE, bounds[1].first)
            check(first < last)
            val widest = bounds.maxOf { (lower, upper) -> upper - lower }
            uncertainty = integer(p["uncertainty_ns"], widest, CLOCK_LIMIT).toLong()
            start = MultimodalNanosecondBound(bounds[0].first.toLong(), bounds[0].second.toLong())
            end = MultimodalNanosecondBound(bounds[1].first.toLong(), bounds[1].second.toLong())
            overlap = MultimodalInterval(first.toLong(), last.toLong())
        }
        val original = JsonObject(p - "source_schema" - "source_result_digest")
        val provenance = JsonObject(linkedMapOf("algorithm" to JsonPrimitive(CLOCK_SCHEMA), "result" to original))
        check(hash(provenance) == digests.getValue("source_result_digest"), MultimodalContractRefusal.PROVENANCE)
        return MultimodalClockAlignment(
            digests.getValue("source_result_digest"), state, reasons,
            digests.getValue("source_reference_digest"), digests.getValue("target_reference_digest"),
            digests.getValue("source_clock_digest"), digests.getValue("target_clock_digest"),
            digests.getValue("policy_digest"), start, end, overlap, uncertainty,
        )
    }

    private fun hash(value: JsonElement): String =
        MessageDigest.getInstance("SHA-256")
            .digest(canonical(value).toByteArray(Charsets.US_ASCII))
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    /** Python `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)`. */
    private fun canonical(value: JsonElement): String = StringBuilder().also { render(value, it) }.toString()

    private val codePointOrder = Comparator<String> { left, right ->
        val a = left.codePoints().toArray()
        val b = right.codePoints().toArray()
        (0 until minOf(a.size, b.size)).firstOrNull { a[it] != b[it] }?.let { a[it].compareTo(b[it]) }
            ?: a.size.compareTo(b.size)
    }

    private fun render(value: JsonElement, out: StringBuilder) {
        when (value) {
            is JsonObject -> {
                out.append('{')
                value.keys.sortedWith(codePointOrder).forEachIndexed { index, key ->
                    if (index > 0) out.append(',')
                    quote(key, out)
                    out.append(':')
                    render(value.getValue(key), out)
                }
                out.append('}')
            }
            is JsonArray -> {
                out.append('[')
                value.forEachIndexed { index, item ->
                    if (index > 0) out.append(',')
                    render(item, out)
                }
                out.append(']')
            }
            is JsonNull -> out.append("null")
            is JsonPrimitive -> when {
                value.isString -> quote(value.content, out)
                value.content == "true" || value.content == "false" -> out.append(value.content)
                // Every admitted number is an integer; floats are hex strings.
                INTEGER.matches(value.content) -> out.append(BigInteger(value.content).toString())
                else -> refuse()
            }
        }
    }

    private fun quote(value: String, out: StringBuilder) {
        out.append('"')
        for (char in value) {
            when {
                char == '"' -> out.append("\\\"")
                char == '\\' -> out.append("\\\\")
                char == '\n' -> out.append("\\n")
                char == '\r' -> out.append("\\r")
                char == '\t' -> out.append("\\t")
                char == '\b' -> out.append("\\b")
                char == '\u000c' -> out.append("\\f")
                char in ' '..'~' -> out.append(char)
                else -> out.append("\\u").append(Integer.toHexString(char.code).padStart(4, '0'))
            }
        }
        out.append('"')
    }
}
