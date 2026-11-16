package com.openmed.openmedkit

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Portable SDK configuration, version 1 (`sdk_config.schema.json`): the same six
 * keys, bounds and defaults that Python and the other OpenMedKit ports read.
 *
 * Parse it once with [parse] and pass it to `OpenMed` and `ModelDownloader`.
 * Unknown keys fail closed. [DEFAULT] is the empty document, which is
 * local-only: `ModelDownloader` then serves only models already in the cache.
 * See `docs/configuration/sdk-config.md`.
 *
 * @property confidenceThreshold Minimum detection confidence, in `[0, 1]`.
 * @property localOnly Whether network downloads are refused.
 * @property modelId Hub repository id of the model, or `null` for the kit default.
 * @property modelRevision Pinned revision of [modelId], or `null`.
 * @property onnxVariant `auto`, `int8`, `fp16` or `fp32`.
 * @property policyProfile Portable policy profile name.
 */
class OpenMedSdkConfig private constructor(
    val confidenceThreshold: Double,
    val localOnly: Boolean,
    val modelId: String?,
    val modelRevision: String?,
    val onnxVariant: String,
    val policyProfile: String,
) {
    companion object {
        /** Configuration schema version this type reads. */
        const val VERSION: Int = 1

        /** The resolved empty document. */
        @JvmField
        val DEFAULT: OpenMedSdkConfig = OpenMedSdkConfig(0.5, true, null, null, "auto", "hipaa_safe_harbor")

        private val FIELDS = listOf(
            "confidence_threshold",
            "local_only",
            "model_id",
            "model_revision",
            "onnx_variant",
            "policy_profile",
        )
        private val ONNX_VARIANTS = setOf("auto", "int8", "fp16", "fp32")
        private val POLICY_PROFILES = setOf(
            "australia_privacy_act",
            "canada_pipeda",
            "clinical_minimal_redaction",
            "gdpr_art9_health",
            "gdpr_pseudonymization",
            "hipaa_expert_review_assist",
            "hipaa_safe_harbor",
            "research_limited_dataset",
            "strict_no_leak",
        )
        private val MODEL_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?")
        private val REVISION = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

        /**
         * Parses a configuration document, filling absent keys with the defaults.
         *
         * @throws OpenMedSdkConfigException with the ordered issues when the
         *   document is invalid.
         */
        @JvmStatic
        fun parse(json: String): OpenMedSdkConfig {
            val document = decode(json)
            val issues = issues(document)
            if (issues.isNotEmpty()) throw OpenMedSdkConfigException(issues)
            document!!
            val threshold = document["confidence_threshold"] as JsonPrimitive?
            return OpenMedSdkConfig(
                confidenceThreshold = threshold?.content?.toDouble() ?: DEFAULT.confidenceThreshold,
                localOnly = (document["local_only"] as JsonPrimitive?)?.content?.toBoolean() ?: DEFAULT.localOnly,
                modelId = text(document, "model_id"),
                modelRevision = text(document, "model_revision"),
                onnxVariant = text(document, "onnx_variant") ?: DEFAULT.onnxVariant,
                policyProfile = text(document, "policy_profile") ?: DEFAULT.policyProfile,
            )
        }

        /**
         * Returns every issue in a configuration document, empty when it is
         * valid: `config_not_object` alone when the text is not a JSON object;
         * otherwise at most one `unknown_field`, then at most one issue per
         * key in schema order.
         */
        @JvmStatic
        fun validate(json: String): List<OpenMedSdkConfigIssue> = issues(decode(json))

        private fun decode(json: String): JsonObject? = try {
            Json.parseToJsonElement(json) as? JsonObject
        } catch (error: Exception) {
            null
        }

        private fun issues(document: JsonObject?): List<OpenMedSdkConfigIssue> {
            if (document == null) return listOf(OpenMedSdkConfigIssue("config_not_object", null))
            val issues = ArrayList<OpenMedSdkConfigIssue>()
            if (!FIELDS.containsAll(document.keys)) issues += OpenMedSdkConfigIssue("unknown_field", null)
            for (field in FIELDS) {
                val value = document[field] ?: continue
                var code = fieldIssue(field, value)
                val model = document["model_id"]
                if (code == null && field == "model_revision" && value != JsonNull && (model == null || model == JsonNull)) {
                    code = "missing_dependency"
                }
                if (code != null) issues += OpenMedSdkConfigIssue(code, field)
            }
            return issues
        }

        private fun fieldIssue(field: String, value: Any): String? {
            val primitive = value as? JsonPrimitive ?: return "wrong_type"
            val literal = !primitive.isString && primitive != JsonNull
            val boolean = literal && (primitive.content == "true" || primitive.content == "false")
            return when (field) {
                "confidence_threshold" -> {
                    if (!literal || boolean) return "wrong_type"
                    val number = primitive.content.toDoubleOrNull() ?: return "wrong_type"
                    if (number >= 0.0 && number <= 1.0) null else "out_of_range"
                }
                "local_only" -> if (boolean) null else "wrong_type"
                "model_id", "model_revision" -> when {
                    primitive == JsonNull -> null
                    !primitive.isString -> "wrong_type"
                    (if (field == "model_id") MODEL_ID else REVISION).matches(primitive.content) &&
                        ".." !in primitive.content -> null
                    else -> "invalid_format"
                }
                else -> when {
                    !primitive.isString -> "wrong_type"
                    primitive.content in (if (field == "onnx_variant") ONNX_VARIANTS else POLICY_PROFILES) -> null
                    else -> "not_allowed_value"
                }
            }
        }

        private fun text(document: JsonObject, key: String): String? =
            (document[key] as? JsonPrimitive)?.takeIf { it.isString }?.content
    }
}

/**
 * One value-free configuration problem.
 *
 * @property code `config_not_object`, `unknown_field`, `wrong_type`,
 *   `out_of_range`, `not_allowed_value`, `invalid_format` or `missing_dependency`.
 * @property field The configuration key, or `null` for a document-level issue.
 */
class OpenMedSdkConfigIssue internal constructor(val code: String, val field: String?) {
    override fun toString(): String = if (field == null) code else "$field:$code"
}

/** Thrown by [OpenMedSdkConfig.parse]; the message lists codes and key names only. */
class OpenMedSdkConfigException internal constructor(
    val issues: List<OpenMedSdkConfigIssue>,
) : IllegalArgumentException("invalid SDK configuration: " + issues.joinToString())
