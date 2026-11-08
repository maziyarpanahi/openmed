package com.openmed.openmedkit.decision

import java.text.Normalizer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Constants of the OpenMed fixed-option decision contract (request and result
 * schema 1.2.0) shared with Python, REST, Swift and the web package.
 *
 * Decision results are assistive review artifacts. They never authorize a
 * diagnosis, treatment, enrollment, outreach, ordering or any other autonomous
 * clinical action.
 */
public object OpenMedDecision {
    public const val SCHEMA_VERSION: String = "1.2.0"
    public val SUPPORTED_SCHEMA_VERSIONS: List<String> = listOf("1.0.0", "1.1.0", "1.2.0")
    public const val COMPATIBILITY_POLICY: String = "same_major"
    public const val POLICY_VERSION: String = "1.0.0"
    public const val DEFAULT_CALIBRATION_ID: String = "openmed.synthetic.fixed_option.v1"
    public const val DEFAULT_QUESTION_ID: String = "decision"
    public const val ADVISORY: String =
        "Decision scores are assistive review evidence, not a diagnosis, treatment " +
            "recommendation, enrollment decision, or autonomous clinical action."
    public const val MAX_INPUT_CHARS: Int = 8192
    public const val MAX_CONFIGURABLE_INPUT_CHARS: Int = 131_072
    public const val MAX_OPTIONS: Int = 64
    public const val MAX_QUESTIONS: Int = 32
    public const val MIN_TIMEOUT_MS: Int = 1
    public const val MAX_TIMEOUT_MS: Int = 30_000
    public const val MAX_MEDIA_ITEMS: Int = 4
    public const val MAX_MEDIA_BYTES: Long = 8L * 1024 * 1024

    internal const val BASE_VERSION = "1.0.0"
    internal const val QUESTION_VERSION = "1.1.0"
    internal const val MEDIA_VERSION = "1.2.0"
    internal const val INPUT_BYTES_PER_CHAR = 4
    internal const val MAX_OPTION_CHARS = 256
    internal const val MAX_OPTION_BYTES = 1024
    internal const val MAX_TOTAL_OPTION_BYTES = 16_384
    internal const val MAX_INSTRUCTIONS_CHARS = 2048
    internal const val MAX_DESCRIPTION_CHARS = 1024
    internal const val MAX_TOTAL_QUESTION_BYTES = 65_536
    internal const val MAX_TOTAL_MEDIA_BYTES = 16L * 1024 * 1024
    internal val MEDIA_TYPES = setOf("image/png", "image/jpeg", "video/mp4", "audio/wav")
    internal val CONTROLLED = Regex("[a-z][a-z0-9_.:/-]{0,127}")
    internal val QUESTION_ID = Regex("[A-Za-z_][A-Za-z0-9_.:-]{0,63}")
    internal val VERSION = Regex("(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)")
    internal val DIGEST = Regex("sha256:[0-9a-f]{64}")
}

/**
 * Value-free decision errors. [reason] is a fixed code; no request text,
 * option, prompt, identifier or path is ever included.
 */
public sealed class DecisionRuntimeError(
    /** `invalid_request`, `unsupported_contract` or `invalid_configuration`. */
    public val kind: String,
    /** The fixed reason code. */
    public val reason: String,
) : Exception("$kind:$reason") {
    /** The request violates the decision contract. */
    public class InvalidRequest(reason: String) : DecisionRuntimeError("invalid_request", reason)

    /** The request declares a contract version or policy this runtime cannot serve. */
    public class UnsupportedContract(reason: String) :
        DecisionRuntimeError("unsupported_contract", reason)

    /** A runtime, profile, policy or identity setting is invalid. */
    public class InvalidConfiguration(reason: String) :
        DecisionRuntimeError("invalid_configuration", reason)
}

/** Supported fixed-output decision shapes. */
public enum class DecisionMode(public val wire: String) {
    FIXED_CHOICE("fixed_choice"),
    BOOLEAN_CHOICE("boolean_choice"),
    ORDERED_PREFERENCE("ordered_preference"),
    SCALAR_SCORE("scalar_score"),
    MULTI_LABEL("multi_label"),
    QUESTION_SET("question_set"),
    ;

    public companion object {
        public fun fromWire(value: String): DecisionMode? = values().firstOrNull { it.wire == value }
    }
}

/** Lossless public decision states. */
public enum class DecisionState(public val wire: String) {
    SUCCESS("success"),
    ABSTAINED("abstained"),
    PARTIAL("partial"),
    UNKNOWN("unknown"),
    CONFLICT("conflict"),
    UNSUPPORTED("unsupported"),
    DENIED("denied"),
    FAILURE("failure"),
}

/**
 * Versioned thresholds pinned to frozen calibration slices.
 *
 * A profile is a plain object today; reading it from a signed model bundle is
 * a separate step. Construction validates it exactly as the Python
 * `DecisionCalibrationProfile` does.
 */
public data class DecisionCalibrationProfile(
    val calibrationId: String,
    val version: String,
    val method: String,
    val minimumConfidence: Double,
    val minimumMargin: Double,
    val multiLabelThreshold: Double,
    val inDomainDigest: String,
    val outOfDomainDigest: String,
) {
    init {
        val probabilities = listOf(minimumConfidence, minimumMargin, multiLabelThreshold)
        if (!OpenMedDecision.CONTROLLED.matches(calibrationId) ||
            !OpenMedDecision.VERSION.matches(version) ||
            !OpenMedDecision.CONTROLLED.matches(method) ||
            !probabilities.all { it.isFinite() && it in 0.0..1.0 } ||
            minimumMargin > minimumConfidence ||
            !OpenMedDecision.DIGEST.matches(inDomainDigest) ||
            !OpenMedDecision.DIGEST.matches(outOfDomainDigest)
        ) {
            throw DecisionRuntimeError.InvalidConfiguration("calibration_profile_invalid")
        }
    }

    internal fun toJson(): JsonObject = JsonObject(
        mapOf(
            "calibration_id" to JsonPrimitive(calibrationId),
            "version" to JsonPrimitive(version),
            "method" to JsonPrimitive(method),
            "minimum_confidence" to JsonPrimitive(minimumConfidence),
            "minimum_margin" to JsonPrimitive(minimumMargin),
            "multi_label_threshold" to JsonPrimitive(multiLabelThreshold),
            "in_domain_digest" to JsonPrimitive(inDomainDigest),
            "out_of_domain_digest" to JsonPrimitive(outOfDomainDigest),
        ),
    )

    public companion object {
        /** The profile every kit and the Python runtime use when none is supplied. */
        public val STANDARD: DecisionCalibrationProfile = DecisionCalibrationProfile(
            calibrationId = OpenMedDecision.DEFAULT_CALIBRATION_ID,
            version = "1.0.0",
            method = "deterministic_margin",
            minimumConfidence = 0.70,
            minimumMargin = 0.15,
            multiLabelThreshold = 0.70,
            inDomainDigest =
                "sha256:a6a2e0596c95b0568fe8871c99774fb1edc5b1db6de692e86172df952642dd6c",
            outOfDomainDigest =
                "sha256:d20c55e3877710223c6c37f87f6a8aaf9dea1b326087b37c60a81522ac0c9289",
        )

        private val FIELDS = setOf(
            "calibration_id", "version", "method", "minimum_confidence", "minimum_margin",
            "multi_label_threshold", "in_domain_digest", "out_of_domain_digest",
        )

        /** Parses a plain profile object with the schema's snake_case fields. */
        public fun fromJson(json: String): DecisionCalibrationProfile {
            val element = try {
                Json.parseToJsonElement(json)
            } catch (error: Exception) {
                throw DecisionRuntimeError.InvalidConfiguration("calibration_profile_invalid")
            }
            return fromJson(element)
        }

        /** Parses a plain profile object that is already a JSON tree. */
        public fun fromJson(element: JsonElement): DecisionCalibrationProfile {
            val invalid = DecisionRuntimeError.InvalidConfiguration("calibration_profile_invalid")
            val payload = element as? JsonObject ?: throw invalid
            if (payload.keys != FIELDS) throw invalid
            return DecisionCalibrationProfile(
                calibrationId = payload["calibration_id"].stringValue() ?: throw invalid,
                version = payload["version"].stringValue() ?: throw invalid,
                method = payload["method"].stringValue() ?: throw invalid,
                minimumConfidence = payload["minimum_confidence"].numberValue() ?: throw invalid,
                minimumMargin = payload["minimum_margin"].numberValue() ?: throw invalid,
                multiLabelThreshold = payload["multi_label_threshold"].numberValue() ?: throw invalid,
                inDomainDigest = payload["in_domain_digest"].stringValue() ?: throw invalid,
                outOfDomainDigest = payload["out_of_domain_digest"].stringValue() ?: throw invalid,
            )
        }
    }
}

/** Namespace, purpose and backend-family allowlist. */
public class DecisionAccessPolicy(
    val allowedNamespaces: Set<String>,
    val allowedPurposes: Set<String>,
    val allowedBackendKinds: Set<String> = BACKEND_KINDS,
    val policyVersion: String = OpenMedDecision.POLICY_VERSION,
) {
    init {
        if (allowedNamespaces.isEmpty() || allowedPurposes.isEmpty() ||
            allowedBackendKinds.isEmpty() ||
            !(allowedNamespaces + allowedPurposes).all { OpenMedDecision.CONTROLLED.matches(it) } ||
            !BACKEND_KINDS.containsAll(allowedBackendKinds) ||
            !OpenMedDecision.VERSION.matches(policyVersion)
        ) {
            throw DecisionRuntimeError.InvalidConfiguration("access_policy_invalid")
        }
    }

    internal fun decision(namespace: String, purpose: String, kind: String): DecisionAccess {
        val code = when {
            namespace !in allowedNamespaces -> "namespace_denied"
            purpose !in allowedPurposes -> "purpose_denied"
            kind !in allowedBackendKinds -> "backend_denied"
            else -> null
        }
        return DecisionAccess(code == null, code, namespace, purpose, policyVersion)
    }

    public companion object {
        public val BACKEND_KINDS: Set<String> =
            setOf("deterministic", "encoder", "cross_encoder", "specialist")

        /**
         * The Python default: namespace `default`; purposes `analytics`,
         * `care_review` and `quality`; every backend family.
         */
        public val STANDARD: DecisionAccessPolicy = DecisionAccessPolicy(
            allowedNamespaces = setOf("default"),
            allowedPurposes = setOf("analytics", "care_review", "quality"),
        )
    }
}

/** The access decision recorded in every result. */
public class DecisionAccess(
    val allowed: Boolean,
    val code: String?,
    val namespace: String,
    val purpose: String,
    val policyVersion: String,
) {
    internal fun toJson(): JsonObject = JsonObject(
        mapOf(
            "allowed" to JsonPrimitive(allowed),
            "code" to jsonString(code),
            "namespace" to JsonPrimitive(namespace),
            "purpose" to JsonPrimitive(purpose),
            "policy_version" to JsonPrimitive(policyVersion),
        ),
    )
}

/**
 * Value-free provenance of an on-device encoder decision backend. The family
 * is always `encoder`, local and non-deterministic. The license is provenance
 * for separately acquired weights, not a redistribution grant.
 */
public class DecisionBackendIdentity(
    val backendId: String,
    val runtime: String,
    val modelId: String,
    val revision: String,
    val licenseId: String,
    val licenseSha256: String? = null,
) {
    val kind: String get() = "encoder"

    init {
        val invalid = DecisionRuntimeError.InvalidConfiguration("backend_identity_invalid")
        if (!listOf(backendId, runtime, modelId, revision).all { DecisionText.isBoundedLine(it, 256) } ||
            !LICENSE.matches(licenseId) || licenseId != licenseId.trim()
        ) {
            throw invalid
        }
        if (licenseSha256 != null &&
            (!SHA256.matches(licenseSha256) || !IMMUTABLE_REVISION.matches(revision))
        ) {
            throw invalid
        }
    }

    internal fun toJson(): JsonObject {
        val payload = linkedMapOf<String, JsonElement>(
            "backend_id" to JsonPrimitive(backendId),
            "kind" to JsonPrimitive(kind),
            "runtime" to JsonPrimitive(runtime),
            "model_id" to JsonPrimitive(modelId),
            "revision" to JsonPrimitive(revision),
            "license_id" to JsonPrimitive(licenseId),
        )
        if (licenseSha256 != null) payload["license_sha256"] = JsonPrimitive(licenseSha256)
        payload["local"] = JsonPrimitive(true)
        payload["deterministic"] = JsonPrimitive(false)
        return JsonObject(payload)
    }

    private companion object {
        val LICENSE = Regex("[A-Za-z0-9][A-Za-z0-9 .()+:/_-]{0,127}")
        val SHA256 = Regex("[0-9a-f]{64}")
        val IMMUTABLE_REVISION = Regex("[0-9a-f]{40,64}")
    }
}

/** One named question scored against a request's shared input. */
internal class DecisionQuestion(
    val questionId: String,
    val mode: DecisionMode,
    val options: List<String>,
    val optionDescriptions: List<String?>?,
    val instructions: String?,
    val levels: List<String>,
) {
    val textBytes: Int
        get() = (options + (optionDescriptions ?: emptyList()).filterNotNull() + levels)
            .sumOf { it.utf8Size() } + (instructions?.utf8Size() ?: 0)
}

/**
 * One validated decision request (schema 1.0.0, 1.1.0 or 1.2.0).
 *
 * The input text and question wording are transient: they are never part of a
 * result, an error or [toString].
 */
public class DecisionRequest internal constructor(
    public val mode: DecisionMode,
    public val namespace: String,
    public val purpose: String,
    public val calibrationId: String,
    public val timeoutMs: Int,
    /** The declared (or resolved) schema version. */
    public val schemaVersion: String,
    /** The newest supported version not newer than [schemaVersion]. */
    public val contractVersion: String,
    public val inputFormat: String,
    public val maxInputChars: Int?,
    /** Number of media items. On-device runtimes are text-only and refuse media. */
    public val mediaCount: Int,
    internal val inputText: String,
    internal val questions: List<DecisionQuestion>,
) {
    override fun toString(): String =
        "DecisionRequest(mode=${mode.wire}, schema=$contractVersion, " +
            "questions=${questions.size}, media=$mediaCount)"

    public companion object {
        /** Parses and validates a request mapping with the schema's field names. */
        public fun parse(json: String): DecisionRequest {
            val element = try {
                Json.parseToJsonElement(json)
            } catch (error: Exception) {
                throw DecisionRuntimeError.InvalidRequest("request_json_invalid")
            }
            return parse(element)
        }

        /** Validates a request mapping that is already a JSON tree. */
        public fun parse(element: JsonElement): DecisionRequest = DecisionRequestParser.parse(element)

        /** Builds a single-question text request through the same validation. */
        public fun of(
            mode: DecisionMode,
            inputText: String,
            options: List<String>,
            namespace: String = "default",
            purpose: String = "care_review",
            calibrationId: String = OpenMedDecision.DEFAULT_CALIBRATION_ID,
            timeoutMs: Int = 5000,
            schemaVersion: String? = OpenMedDecision.SCHEMA_VERSION,
        ): DecisionRequest {
            val payload = linkedMapOf<String, JsonElement>(
                "mode" to JsonPrimitive(mode.wire),
                "input_text" to JsonPrimitive(inputText),
                "options" to JsonArray(options.map(::JsonPrimitive)),
                "namespace" to JsonPrimitive(namespace),
                "purpose" to JsonPrimitive(purpose),
                "calibration_id" to JsonPrimitive(calibrationId),
                "timeout_ms" to JsonPrimitive(timeoutMs),
            )
            if (schemaVersion != null) payload["schema_version"] = JsonPrimitive(schemaVersion)
            return parse(JsonObject(payload))
        }
    }
}

/** One calibrated option score, in request order. */
public data class DecisionOptionScore(val index: Int, val option: String, val score: Double)

/** Calibrated outcome for one named question of a question set. */
public class DecisionQuestionResult(
    val questionId: String,
    val mode: DecisionMode,
    val state: DecisionState,
    val code: String?,
    val optionScores: List<DecisionOptionScore>,
    val choice: String?,
    val choices: List<String>,
    val ranking: List<String>,
    val scalarScore: Double?,
    val confidence: Double?,
    val margin: Double?,
    val warnings: List<String>,
    val reviewReasons: List<String>,
) {
    val abstained: Boolean get() = state == DecisionState.ABSTAINED

    internal fun toJson(): JsonObject {
        val payload = linkedMapOf<String, JsonElement>("question_id" to JsonPrimitive(questionId))
        payload.putAll(
            answerFields(
                mode, state, code, optionScores, choice, choices, ranking, scalarScore, confidence,
                margin,
            ),
        )
        payload["level_scores"] = JsonArray(emptyList())
        payload["expected_level"] = JsonNull
        payload["warnings"] = strings(warnings)
        payload["review"] = review(reviewReasons)
        return JsonObject(payload)
    }
}

/**
 * Canonical decision result; it validates against the 1.2.0 result schema and
 * renders exactly the fields of the request's contract version.
 */
public class DecisionResult(
    val mode: DecisionMode,
    val state: DecisionState,
    val code: String?,
    val optionScores: List<DecisionOptionScore>,
    val choice: String?,
    val choices: List<String>,
    val ranking: List<String>,
    val scalarScore: Double?,
    val confidence: Double?,
    val margin: Double?,
    /** Per-question results of a `question_set` request; empty otherwise. */
    val questions: List<DecisionQuestionResult>,
    val calibration: DecisionCalibrationProfile,
    val backend: DecisionBackendIdentity,
    val access: DecisionAccess,
    val warnings: List<String>,
    val reviewReasons: List<String>,
    val schemaVersion: String,
) {
    val advisory: String get() = OpenMedDecision.ADVISORY

    /** Always `false`: results never authorize an autonomous clinical action. */
    val autonomousAction: Boolean get() = false

    /** Always `true`: every result requires human review. */
    val reviewRequired: Boolean get() = true
    val abstained: Boolean get() = state == DecisionState.ABSTAINED

    /** The result with the schema's field names. */
    public fun toJson(): JsonObject {
        val payload = linkedMapOf<String, JsonElement>()
        payload.putAll(
            answerFields(
                mode, state, code, optionScores, choice, choices, ranking, scalarScore, confidence,
                margin,
            ),
        )
        if (schemaVersion != OpenMedDecision.BASE_VERSION) {
            payload["level_scores"] = JsonArray(emptyList())
            payload["expected_level"] = JsonNull
            payload["questions"] = JsonArray(questions.map { it.toJson() })
        }
        payload["calibration"] = calibration.toJson()
        payload["backend"] = backend.toJson()
        payload["access"] = access.toJson()
        payload["warnings"] = strings(warnings)
        payload["review"] = review(reviewReasons)
        payload["advisory"] = JsonPrimitive(advisory)
        payload["autonomous_action"] = JsonPrimitive(false)
        payload["schema_version"] = JsonPrimitive(schemaVersion)
        payload["compatibility_policy"] = JsonPrimitive(OpenMedDecision.COMPATIBILITY_POLICY)
        payload["extensions"] = JsonObject(emptyMap())
        return JsonObject(payload)
    }

    /** Compact JSON text of [toJson]. */
    public fun toJsonString(): String = toJson().toString()
}

private fun answerFields(
    mode: DecisionMode,
    state: DecisionState,
    code: String?,
    optionScores: List<DecisionOptionScore>,
    choice: String?,
    choices: List<String>,
    ranking: List<String>,
    scalarScore: Double?,
    confidence: Double?,
    margin: Double?,
): Map<String, JsonElement> = linkedMapOf(
    "mode" to JsonPrimitive(mode.wire),
    "state" to JsonPrimitive(state.wire),
    "code" to jsonString(code),
    "option_scores" to JsonArray(
        optionScores.map {
            JsonObject(
                mapOf(
                    "index" to JsonPrimitive(it.index),
                    "option" to JsonPrimitive(it.option),
                    "score" to JsonPrimitive(it.score),
                ),
            )
        },
    ),
    "choice" to jsonString(choice),
    "choices" to strings(choices),
    "ranking" to strings(ranking),
    "scalar_score" to jsonNumber(scalarScore),
    "confidence" to jsonNumber(confidence),
    "margin" to jsonNumber(margin),
)

private fun jsonString(value: String?): JsonElement = value?.let(::JsonPrimitive) ?: JsonNull

private fun jsonNumber(value: Double?): JsonElement = value?.let(::JsonPrimitive) ?: JsonNull

private fun strings(values: List<String>): JsonArray = JsonArray(values.map(::JsonPrimitive))

private fun review(reasons: List<String>): JsonObject = JsonObject(
    mapOf("required" to JsonPrimitive(true), "reasons" to strings(reasons)),
)

// Request validation, in the order of the Python contract: structure, values,
// contract support, then the timeout.
internal object DecisionRequestParser {
    private val BASE_FIELDS = setOf(
        "mode", "input_text", "options", "namespace", "purpose", "calibration_id", "timeout_ms",
        "schema_version", "compatibility_policy",
    )
    private val EXTENSION_FIELDS = setOf(
        "question_id", "instructions", "option_descriptions", "levels", "questions",
        "input_format", "max_input_chars", "media",
    )
    private val QUESTION_FIELDS = setOf(
        "question_id", "mode", "options", "option_descriptions", "instructions", "levels",
    )
    private val MEDIA_FIELDS = setOf("data", "byte_size", "media_type", "sha256")
    private val MEDIA_DIGEST = Regex("[0-9a-f]{64}")

    private fun invalid(reason: String) = DecisionRuntimeError.InvalidRequest(reason)

    private fun unsupported(reason: String) = DecisionRuntimeError.UnsupportedContract(reason)

    fun parse(element: JsonElement): DecisionRequest {
        val payload = element as? JsonObject ?: throw invalid("request_not_object")
        if (!(BASE_FIELDS + EXTENSION_FIELDS).containsAll(payload.keys)) {
            throw invalid("request_field_unsupported")
        }
        val modeValue = payload["mode"] ?: throw invalid("request_field_missing")
        val inputValue = payload["input_text"] ?: throw invalid("request_field_missing")
        val mode = modeValue.stringValue()?.let(DecisionMode::fromWire)
            ?: throw invalid("mode_invalid")
        val options = stringList(payload["options"] ?: JsonArray(emptyList()), "options_invalid")
        val descriptions = optionalProseList(payload["option_descriptions"])
        val levels = stringList(nullable(payload["levels"]) ?: JsonArray(emptyList()), "levels_invalid")
        val rawQuestions = when (val value = nullable(payload["questions"])) {
            null -> JsonArray(emptyList())
            is JsonArray -> value
            else -> throw invalid("questions_invalid")
        }
        val inputFormat = when (val format = nullable(payload["input_format"])) {
            null -> "text"
            else -> format.stringValue()?.takeIf { it == "text" || it == "json" }
                ?: throw invalid("input_format_invalid")
        }
        val mediaCount = mediaItems(payload["media"])

        val maxInputChars = nullable(payload["max_input_chars"])?.let {
            it.integerValue()?.takeIf { bound ->
                bound in 1..OpenMedDecision.MAX_CONFIGURABLE_INPUT_CHARS.toLong()
            }?.toInt() ?: throw invalid("max_input_chars_invalid")
        }
        val inputText = inputValue.stringValue() ?: throw invalid("input_invalid")
        checkInput(inputText, maxInputChars ?: OpenMedDecision.MAX_INPUT_CHARS, inputFormat)

        val instructions = prose(payload["instructions"], OpenMedDecision.MAX_INSTRUCTIONS_CHARS)
        var questionId: String? = null
        val questions: List<DecisionQuestion>
        if (mode == DecisionMode.QUESTION_SET) {
            if (options.isNotEmpty() || descriptions != null || instructions != null ||
                levels.isNotEmpty() || nullable(payload["question_id"]) != null
            ) {
                throw invalid("question_set_fields_invalid")
            }
            questions = validatedQuestions(rawQuestions)
        } else {
            if (rawQuestions.isNotEmpty()) throw invalid("questions_require_question_set")
            nullable(payload["question_id"])?.let {
                questionId = it.stringValue()?.takeIf(OpenMedDecision.QUESTION_ID::matches)
                    ?: throw invalid("question_id_invalid")
            }
            val question = validatedQuestion(
                questionId ?: OpenMedDecision.DEFAULT_QUESTION_ID,
                mode,
                options,
                descriptions,
                instructions,
                levels,
            )
            if (question.textBytes > OpenMedDecision.MAX_TOTAL_QUESTION_BYTES) {
                throw invalid("questions_too_large")
            }
            questions = listOf(question)
        }
        val namespace = controlled(payload["namespace"] ?: JsonPrimitive("default"), "namespace_invalid")
        val purpose = controlled(payload["purpose"] ?: JsonPrimitive("care_review"), "purpose_invalid")
        val calibrationId = controlled(
            payload["calibration_id"] ?: JsonPrimitive(OpenMedDecision.DEFAULT_CALIBRATION_ID),
            "calibration_id_invalid",
        )

        val usesQuestionFields = mode == DecisionMode.QUESTION_SET || questionId != null ||
            descriptions != null || instructions != null || levels.isNotEmpty() ||
            inputFormat != "text" || maxInputChars != null
        val usesMedia = mediaCount > 0
        val declared = when (val version = nullable(payload["schema_version"])) {
            null -> when {
                usesMedia -> OpenMedDecision.MEDIA_VERSION
                usesQuestionFields -> OpenMedDecision.QUESTION_VERSION
                else -> OpenMedDecision.BASE_VERSION
            }
            else -> version.stringValue()?.takeIf(OpenMedDecision.VERSION::matches)
                ?: throw invalid("schema_version_invalid")
        }

        if (declared.substringBefore('.') != "1") throw unsupported("schema_major_unsupported")
        val contract = contractVersion(declared)
        if (usesQuestionFields && contract == OpenMedDecision.BASE_VERSION) {
            throw unsupported("question_fields_require_1_1_0")
        }
        if (usesMedia && versionPrecedes(contract, OpenMedDecision.MEDIA_VERSION)) {
            throw unsupported("media_requires_1_2_0")
        }
        val policy = payload["compatibility_policy"] ?: JsonPrimitive(OpenMedDecision.COMPATIBILITY_POLICY)
        if (policy.stringValue() != OpenMedDecision.COMPATIBILITY_POLICY) {
            throw unsupported("compatibility_policy_unsupported")
        }

        val timeout = (payload["timeout_ms"] ?: JsonPrimitive(5000)).integerValue()
            ?.takeIf { it in OpenMedDecision.MIN_TIMEOUT_MS..OpenMedDecision.MAX_TIMEOUT_MS }
            ?.toInt() ?: throw invalid("timeout_invalid")
        return DecisionRequest(
            mode = mode,
            namespace = namespace,
            purpose = purpose,
            calibrationId = calibrationId,
            timeoutMs = timeout,
            schemaVersion = declared,
            contractVersion = contract,
            inputFormat = inputFormat,
            maxInputChars = maxInputChars,
            mediaCount = mediaCount,
            inputText = inputText,
            questions = questions,
        )
    }

    fun contractVersion(declared: String): String =
        OpenMedDecision.SUPPORTED_SCHEMA_VERSIONS.lastOrNull { !versionPrecedes(declared, it) }
            ?: OpenMedDecision.BASE_VERSION

    /** Whether [left] is an older semantic version than [right]. */
    fun versionPrecedes(left: String, right: String): Boolean {
        val lhs = left.split('.').map { it.toLongOrNull() ?: Long.MAX_VALUE }
        val rhs = right.split('.').map { it.toLongOrNull() ?: Long.MAX_VALUE }
        for (index in 0 until minOf(lhs.size, rhs.size)) {
            if (lhs[index] != rhs[index]) return lhs[index] < rhs[index]
        }
        return lhs.size < rhs.size
    }

    private fun nullable(value: JsonElement?): JsonElement? = if (value is JsonNull) null else value

    private fun stringList(value: JsonElement, reason: String): List<String> {
        val items = value as? JsonArray ?: throw invalid(reason)
        return items.map { it.stringValue() ?: throw invalid(reason) }
    }

    private fun optionalProseList(value: JsonElement?): List<String?>? {
        val items = nullable(value) ?: return null
        val list = items as? JsonArray ?: throw invalid("option_descriptions_invalid")
        return list.map {
            prose(it, OpenMedDecision.MAX_DESCRIPTION_CHARS, "option_descriptions_invalid")
        }
    }

    private fun controlled(value: JsonElement, reason: String): String =
        value.stringValue()?.takeIf(OpenMedDecision.CONTROLLED::matches) ?: throw invalid(reason)

    private fun mediaItems(value: JsonElement?): Int {
        if (value == null) return 0
        val reason = "media_invalid"
        val items = value as? JsonArray ?: throw invalid(reason)
        if (items.size > OpenMedDecision.MAX_MEDIA_ITEMS) throw invalid(reason)
        var total = 0L
        for (item in items) {
            val fields = item as? JsonObject ?: throw invalid(reason)
            val size = fields["byte_size"]?.integerValue()
            val data = fields["data"]?.stringValue()
            if (fields.keys != MEDIA_FIELDS || size == null || size <= 0 ||
                size > OpenMedDecision.MAX_MEDIA_BYTES || data == null ||
                data.codePointCount(0, data.length).toLong() != 4 * ((size + 2) / 3) ||
                fields["media_type"]?.stringValue() !in OpenMedDecision.MEDIA_TYPES ||
                fields["sha256"]?.stringValue()?.let(MEDIA_DIGEST::matches) != true
            ) {
                throw invalid(reason)
            }
            total += size
        }
        if (total > OpenMedDecision.MAX_TOTAL_MEDIA_BYTES) throw invalid(reason)
        // Media is refused on device and is never decoded here.
        return items.size
    }

    private fun checkInput(text: String, maximum: Int, format: String) {
        if (DecisionText.isBlank(text) || text.contains('\u0000') || DecisionText.hasSurrogate(text) ||
            DecisionText.codePoints(text) > maximum ||
            text.utf8Size() > maximum * OpenMedDecision.INPUT_BYTES_PER_CHAR
        ) {
            throw invalid("input_invalid")
        }
        if (format == "json") {
            val parsed = try {
                Json.parseToJsonElement(text)
            } catch (error: Exception) {
                null
            }
            if (parsed !is JsonObject && parsed !is JsonArray) throw invalid("input_invalid")
        }
    }

    fun prose(value: JsonElement?, maximum: Int, reason: String = "instructions_invalid"): String? {
        val element = nullable(value) ?: return null
        val text = element.stringValue() ?: throw invalid(reason)
        if (DecisionText.isBlank(text) || DecisionText.codePoints(text) > maximum ||
            DecisionText.hasSurrogate(text) ||
            text.any { (it.code < 32 && it !in "\n\r\t") || it.code == 127 } ||
            text.utf8Size() > maximum * OpenMedDecision.INPUT_BYTES_PER_CHAR
        ) {
            throw invalid(reason)
        }
        return text
    }

    private fun option(value: String, reason: String): String {
        val stripped = value.trim()
        if (!DecisionText.isBoundedLine(stripped, OpenMedDecision.MAX_OPTION_CHARS) ||
            stripped.utf8Size() > OpenMedDecision.MAX_OPTION_BYTES
        ) {
            throw invalid(reason)
        }
        return stripped
    }

    private fun distinct(values: List<String>, reason: String) {
        val normalized = values.map(DecisionText::normalizedOption)
        if (normalized.toSet().size != normalized.size ||
            values.sumOf { it.utf8Size() } > OpenMedDecision.MAX_TOTAL_OPTION_BYTES
        ) {
            throw invalid(reason)
        }
    }

    private fun validatedQuestion(
        questionId: String,
        mode: DecisionMode,
        rawOptions: List<String>,
        descriptions: List<String?>?,
        instructions: String?,
        rawLevels: List<String>,
    ): DecisionQuestion {
        val options = rawOptions.map { option(it, "options_invalid") }
        val countValid = when (mode) {
            DecisionMode.SCALAR_SCORE -> options.isEmpty()
            DecisionMode.BOOLEAN_CHOICE -> options.size == 2
            else -> options.size in 2..OpenMedDecision.MAX_OPTIONS
        }
        if (!countValid) throw invalid("options_invalid")
        if (options.isNotEmpty()) distinct(options, "options_invalid")
        if (descriptions != null && (options.isEmpty() || descriptions.size != options.size)) {
            throw invalid("option_descriptions_invalid")
        }
        var levels = emptyList<String>()
        if (rawLevels.isNotEmpty()) {
            if (mode != DecisionMode.SCALAR_SCORE) throw invalid("levels_invalid")
            levels = rawLevels.map { option(it, "levels_invalid") }
            if (levels.size !in 2..OpenMedDecision.MAX_OPTIONS) throw invalid("levels_invalid")
            distinct(levels, "levels_invalid")
        }
        return DecisionQuestion(questionId, mode, options, descriptions, instructions, levels)
    }

    private fun validatedQuestions(values: JsonArray): List<DecisionQuestion> {
        val questions = values.map { value ->
            val payload = value as? JsonObject ?: throw invalid("question_invalid")
            if (!QUESTION_FIELDS.containsAll(payload.keys)) throw invalid("question_invalid")
            val identifier = payload["question_id"]?.stringValue()
                ?.takeIf(OpenMedDecision.QUESTION_ID::matches)
                ?: throw invalid("question_id_invalid")
            val mode = payload["mode"]?.stringValue()?.let(DecisionMode::fromWire)
                ?.takeIf { it != DecisionMode.QUESTION_SET }
                ?: throw invalid("question_mode_invalid")
            validatedQuestion(
                identifier,
                mode,
                stringList(nullable(payload["options"]) ?: JsonArray(emptyList()), "options_invalid"),
                optionalProseList(payload["option_descriptions"]),
                prose(payload["instructions"], OpenMedDecision.MAX_INSTRUCTIONS_CHARS),
                stringList(nullable(payload["levels"]) ?: JsonArray(emptyList()), "levels_invalid"),
            )
        }
        if (questions.size !in 1..OpenMedDecision.MAX_QUESTIONS ||
            questions.map { it.questionId }.toSet().size != questions.size ||
            questions.sumOf { it.textBytes } > OpenMedDecision.MAX_TOTAL_QUESTION_BYTES
        ) {
            throw invalid("questions_invalid")
        }
        return questions
    }
}

internal object DecisionText {
    fun isBlank(value: String): Boolean = value.isBlank()

    fun codePoints(value: String): Int = value.codePointCount(0, value.length)

    fun hasSurrogate(value: String): Boolean =
        value.codePoints().anyMatch { it in 0xD800..0xDFFF }

    /** Non-blank, bounded in code points, single-line text without control characters. */
    fun isBoundedLine(value: String, maximum: Int): Boolean =
        !value.isBlank() && codePoints(value) <= maximum && !hasSurrogate(value) &&
            value.none { it.code < 32 || it.code == 127 }

    /**
     * NFKC, case folding (approximated by upper- then lower-casing) and
     * whitespace collapsing, matching the Python option normalization.
     */
    fun normalizedOption(value: String): String =
        collapseWhitespace(
            Normalizer.normalize(value, Normalizer.Form.NFKC).uppercase().lowercase(),
        )

    private fun collapseWhitespace(value: String): String {
        val words = mutableListOf<String>()
        val current = StringBuilder()
        for (char in value) {
            if (char.isWhitespace()) {
                if (current.isNotEmpty()) words.add(current.toString())
                current.setLength(0)
            } else {
                current.append(char)
            }
        }
        if (current.isNotEmpty()) words.add(current.toString())
        return words.joinToString(" ")
    }
}

internal fun String.utf8Size(): Int = toByteArray(Charsets.UTF_8).size

internal fun JsonElement?.stringValue(): String? =
    (this as? JsonPrimitive)?.takeIf { it.isString }?.content

private val JSON_INTEGER = Regex("-?(0|[1-9][0-9]*)")

internal fun JsonElement?.integerValue(): Long? {
    val primitive = this as? JsonPrimitive ?: return null
    if (primitive.isString || primitive is JsonNull || !JSON_INTEGER.matches(primitive.content)) {
        return null
    }
    return primitive.content.toLongOrNull()
}

internal fun JsonElement?.numberValue(): Double? {
    val primitive = this as? JsonPrimitive ?: return null
    if (primitive.isString || primitive is JsonNull ||
        primitive.content == "true" || primitive.content == "false"
    ) {
        return null
    }
    return primitive.content.toDoubleOrNull()?.takeIf { it.isFinite() }
}
