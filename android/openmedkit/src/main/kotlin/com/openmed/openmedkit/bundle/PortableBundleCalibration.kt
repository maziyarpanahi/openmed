package com.openmed.openmedkit.bundle

/**
 * Value-free reason codes for decision calibration profiles in signed bundles.
 * Mirrors `DecisionCalibrationReason` in `openmed/core/portable_bundle_calibration.py`.
 */
enum class DecisionCalibrationReason(val code: String) {
    BUNDLE_UNVERIFIED("calibration_bundle_unverified"),
    EXTENSION_MISSING("calibration_extension_missing"),
    MALFORMED("calibration_malformed"),
    VERSION_UNSUPPORTED("calibration_version_unsupported"),
    DIGEST_MISMATCH("calibration_digest_mismatch"),
    MODEL_MISMATCH("calibration_model_mismatch"),
    RUNTIME_MISMATCH("calibration_runtime_mismatch"),
    SCOPE_MISMATCH("calibration_scope_mismatch"),
    EVIDENCE_MISMATCH("calibration_evidence_mismatch"),
    PROFILE_MISSING("calibration_profile_missing"),
    RUNTIME_UNBOUND("calibration_runtime_unbound"),
    SCHEMA_INCOMPATIBLE("calibration_schema_incompatible"),
    PIN_MISMATCH("calibration_pin_mismatch"),
    REQUEST_INVALID("calibration_request_invalid"),
}

/** Error that carries only a controlled calibration reason code. */
class DecisionCalibrationException(val reason: DecisionCalibrationReason) : Exception(reason.code)

/** One isotonic-regression breakpoint, in parts per million. */
data class DecisionCalibrationPoint(val inputPpm: Long, val outputPpm: Long)

/** Calibration method with its integer parameters. */
sealed class DecisionCalibrationMethod(val methodName: String) {
    object DeterministicMargin : DecisionCalibrationMethod("deterministic_margin")
    data class TemperatureScaling(val temperaturePpm: Long) : DecisionCalibrationMethod("temperature_scaling")
    data class PlattScaling(val slopePpm: Long, val interceptPpm: Long) : DecisionCalibrationMethod("platt_scaling")
    data class IsotonicRegression(val points: List<DecisionCalibrationPoint>) :
        DecisionCalibrationMethod("isotonic_regression")
}

/** One runtime variant a profile was calibrated on. */
data class DecisionCalibrationRuntimeBinding(val runtimeId: String, val digest: String)

/**
 * A validated calibration profile read from a signed bundle. Probabilities are
 * integers in parts per million; [canonicalJson] rebuilds the exact extension
 * profile object from these typed fields.
 */
class BundleCalibrationProfile(
    val calibrationId: String,
    val version: String,
    val method: DecisionCalibrationMethod,
    val minimumConfidencePpm: Long,
    val minimumMarginPpm: Long,
    val multiLabelThresholdPpm: Long,
    val scopeKind: String,
    val scopeDigest: String,
    val modelId: String,
    val modelRevision: String?,
    val runtimes: List<DecisionCalibrationRuntimeBinding>,
    val decisionSchemaMin: String,
    val decisionSchemaMaxExclusive: String,
    val datasetDigest: String,
    val inDomainDigest: String,
    val outOfDomainDigest: String,
    val evaluationPath: String?,
    val evaluationSha256: String?,
    val profileDigest: String,
) {
    val minimumConfidence: Double get() = minimumConfidencePpm / PPM
    val minimumMargin: Double get() = minimumMarginPpm / PPM
    val multiLabelThreshold: Double get() = multiLabelThresholdPpm / PPM
    val runtimeIds: List<String> get() = runtimes.map { it.runtimeId }

    /** Returns the bound digest for [runtimeId], or `null`. */
    fun runtimeDigest(runtimeId: String): String? = runtimes.firstOrNull { it.runtimeId == runtimeId }?.digest

    /** Returns the canonical JSON of the extension profile object. */
    fun canonicalJson(): ByteArray = Json.encode(toJson())

    internal fun toJson(): Json {
        val parameters = when (val value = method) {
            DecisionCalibrationMethod.DeterministicMargin -> emptyList()
            is DecisionCalibrationMethod.TemperatureScaling -> listOf("temperature_ppm" to Json.Num(value.temperaturePpm))
            is DecisionCalibrationMethod.PlattScaling ->
                listOf("intercept_ppm" to Json.Num(value.interceptPpm), "slope_ppm" to Json.Num(value.slopePpm))
            is DecisionCalibrationMethod.IsotonicRegression -> listOf(
                "points" to Json.Arr(
                    value.points.map {
                        Json.Obj(listOf("input_ppm" to Json.Num(it.inputPpm), "output_ppm" to Json.Num(it.outputPpm)))
                    },
                ),
            )
        }
        val evaluation = evaluationPath?.let {
            Json.Obj(listOf("path" to Json.Str(it), "sha256" to Json.Str(evaluationSha256 ?: "")))
        } ?: Json.Null
        return Json.Obj(
            listOf(
                "calibration_id" to Json.Str(calibrationId),
                "decision_schema" to Json.Obj(
                    listOf("max_exclusive" to Json.Str(decisionSchemaMaxExclusive), "min" to Json.Str(decisionSchemaMin)),
                ),
                "method" to Json.Str(method.methodName),
                "model" to Json.Obj(
                    listOf(
                        "id" to Json.Str(modelId),
                        "revision" to (modelRevision?.let { Json.Str(it) } ?: Json.Null),
                        "runtimes" to Json.Arr(
                            runtimes.map { Json.Obj(listOf("digest" to Json.Str(it.digest), "id" to Json.Str(it.runtimeId))) },
                        ),
                    ),
                ),
                "parameters" to Json.Obj(parameters),
                "profile_digest" to Json.Str(profileDigest),
                "provenance" to Json.Obj(
                    listOf(
                        "dataset_digest" to Json.Str(datasetDigest),
                        "evaluation" to evaluation,
                        "in_domain_digest" to Json.Str(inDomainDigest),
                        "out_of_domain_digest" to Json.Str(outOfDomainDigest),
                    ),
                ),
                "scope" to Json.Obj(listOf("digest" to Json.Str(scopeDigest), "kind" to Json.Str(scopeKind))),
                "thresholds" to Json.Obj(
                    listOf(
                        "minimum_confidence_ppm" to Json.Num(minimumConfidencePpm),
                        "minimum_margin_ppm" to Json.Num(minimumMarginPpm),
                        "multi_label_threshold_ppm" to Json.Num(multiLabelThresholdPpm),
                    ),
                ),
                "version" to Json.Str(version),
            ),
        )
    }

    private companion object {
        const val PPM = 1_000_000.0
    }
}

/** Value-free result of reading calibration profiles from a bundle. */
class DecisionCalibrationRead(
    val ok: Boolean,
    val reason: DecisionCalibrationReason?,
    val contentId: String,
    val profiles: List<BundleCalibrationProfile>,
)

/** What the caller loaded and needs a calibration profile for. */
class DecisionCalibrationRequest(
    val calibrationId: String,
    val runtimeId: String,
    val decisionSchemaVersion: String,
    val modelId: String? = null,
    val modelRevision: String? = null,
    val runtimeDigest: String? = null,
    val scopeKind: String? = null,
    val scopeDigest: String? = null,
    val profileDigest: String? = null,
) {
    init {
        fun optional(value: String?, pattern: Regex) = value == null || pattern.matches(value)
        if (!PortableBundleCalibration.CONTROLLED.matches(calibrationId) || !Patterns.TOKEN.matches(runtimeId) ||
            !Patterns.SEMVER.matches(decisionSchemaVersion) || !optional(modelId, Patterns.MODEL_ID) ||
            !optional(modelRevision, Patterns.REVISION) || !optional(runtimeDigest, Patterns.SHA256) ||
            !optional(profileDigest, Patterns.SHA256) || (scopeKind == null) != (scopeDigest == null) ||
            (scopeKind != null && scopeKind !in PortableBundleCalibration.SCOPE_KINDS) ||
            !optional(scopeDigest, Patterns.SHA256)
        ) {
            throw DecisionCalibrationException(DecisionCalibrationReason.REQUEST_INVALID)
        }
    }
}

/** Value-free result of resolving the profile for a loaded runtime. */
class DecisionCalibrationSelection(
    val ok: Boolean,
    val reason: DecisionCalibrationReason?,
    val contentId: String,
    val profile: BundleCalibrationProfile?,
)

/**
 * Reader for the `org.openmed.decision_calibration` portable-bundle extension.
 *
 * Mirrors `openmed/core/portable_bundle_calibration.py`: the same checks run in
 * the same order and produce the same result for
 * `tests/fixtures/portable_bundle_calibration`. It performs no network access
 * and no logging, never runs a model, and reports only reason codes,
 * identifiers, and digests.
 */
object PortableBundleCalibration {
    const val EXTENSION = "org.openmed.decision_calibration"
    const val EXTENSION_VERSION = 1L
    const val PPM_SCALE = 1_000_000L
    private const val MAX_PROFILES = 16
    private const val MAX_RUNTIMES = 32
    private const val MAX_POINTS = 64
    private const val MAX_TEMPERATURE = 100_000_000L
    private const val MAX_PLATT = 100_000_000L
    internal val CONTROLLED = Regex("[a-z][a-z0-9_.:/-]{0,127}")
    private val EVALUATION_PATH = Regex("[A-Za-z0-9_][A-Za-z0-9._+/-]{0,511}")
    private val METHODS = setOf("deterministic_margin", "isotonic_regression", "platt_scaling", "temperature_scaling")
    internal val SCOPE_KINDS = setOf("labels", "options", "questions")
    private val PROFILE_KEYS = arrayOf(
        "calibration_id", "decision_schema", "method", "model", "parameters", "profile_digest", "provenance", "scope",
        "thresholds", "version",
    )

    private class Malformed : Exception("calibration_malformed")

    /** Reads and validates every calibration profile of a verified bundle. */
    fun read(manifest: ByteArray, verification: PortableBundleVerification): DecisionCalibrationRead =
        read(manifest, verification, requireFiles = false)

    /**
     * Selects the profile for the runtime a caller loaded, failing closed. The
     * bundle must have verified with its file tree; any loaded model identity,
     * decision schema version, scope, and pinned profile digest the caller
     * supplies must then match the selected profile.
     */
    fun resolve(
        manifest: ByteArray,
        verification: PortableBundleVerification,
        request: DecisionCalibrationRequest,
    ): DecisionCalibrationSelection {
        val read = read(manifest, verification, requireFiles = true)
        fun refused(reason: DecisionCalibrationReason?) = DecisionCalibrationSelection(false, reason, read.contentId, null)
        if (!read.ok) return refused(read.reason)
        val candidates = read.profiles.filter { it.calibrationId == request.calibrationId }
        if (candidates.isEmpty()) return refused(DecisionCalibrationReason.PROFILE_MISSING)
        val profile = candidates.firstOrNull { it.runtimeDigest(request.runtimeId) != null }
            ?: return refused(DecisionCalibrationReason.RUNTIME_UNBOUND)
        if ((request.modelId != null && request.modelId != profile.modelId) ||
            (request.modelRevision != null && request.modelRevision != profile.modelRevision) ||
            (request.runtimeDigest != null && request.runtimeDigest != profile.runtimeDigest(request.runtimeId))
        ) {
            return refused(DecisionCalibrationReason.MODEL_MISMATCH)
        }
        val version = semverKey(request.decisionSchemaVersion)
        if (compare(version, semverKey(profile.decisionSchemaMin)) < 0 ||
            compare(version, semverKey(profile.decisionSchemaMaxExclusive)) >= 0
        ) {
            return refused(DecisionCalibrationReason.SCHEMA_INCOMPATIBLE)
        }
        if (request.scopeKind != null &&
            (request.scopeKind != profile.scopeKind || request.scopeDigest != profile.scopeDigest)
        ) {
            return refused(DecisionCalibrationReason.SCOPE_MISMATCH)
        }
        if (request.profileDigest != null && request.profileDigest != profile.profileDigest) {
            return refused(DecisionCalibrationReason.PIN_MISMATCH)
        }
        return DecisionCalibrationSelection(true, null, read.contentId, profile)
    }

    /** Returns the digest that binds a profile to one runtime, or `null`. */
    fun runtimeDigest(manifest: ByteArray, runtimeId: String): String? {
        val root = PortableBundle.structuralRoot(manifest) ?: return null
        return runtimeDigest(root, runtimeId)
    }

    private fun read(
        manifest: ByteArray,
        verification: PortableBundleVerification,
        requireFiles: Boolean,
    ): DecisionCalibrationRead {
        val identity = PortableBundle.contentId(manifest)
        fun refused(reason: DecisionCalibrationReason) = DecisionCalibrationRead(false, reason, identity, emptyList())
        if (!verification.ok || verification.contentId != identity || (requireFiles && !verification.filesVerified)) {
            return refused(DecisionCalibrationReason.BUNDLE_UNVERIFIED)
        }
        val root = PortableBundle.structuralRoot(manifest) ?: return refused(DecisionCalibrationReason.BUNDLE_UNVERIFIED)
        if ((root["extensions"] as? Json.Obj)?.get(EXTENSION) == null) {
            return refused(DecisionCalibrationReason.EXTENSION_MISSING)
        }
        return try {
            DecisionCalibrationRead(true, null, identity, validateExtension(root))
        } catch (error: DecisionCalibrationException) {
            refused(error.reason)
        }
    }

    /** Validates the extension of a structurally valid manifest in the normative order. */
    internal fun validateExtension(root: Json.Obj): List<BundleCalibrationProfile> {
        val raw = (root["extensions"] as? Json.Obj)?.get(EXTENSION)
        val (items, profiles) = try {
            if (raw !is Json.Obj || raw["schema_version"] !is Json.Num) throw Malformed()
            if ((raw["schema_version"] as Json.Num).value != EXTENSION_VERSION) {
                throw DecisionCalibrationException(DecisionCalibrationReason.VERSION_UNSUPPORTED)
            }
            if (!raw.hasExactKeys("profiles", "schema_version")) throw Malformed()
            val list = (raw["profiles"] as? Json.Arr)?.items ?: throw Malformed()
            if (list.size !in 1..MAX_PROFILES) throw Malformed()
            val parsed = list.map { parseProfile(it) }
            val bindings = parsed.flatMap { profile -> profile.runtimeIds.map { profile.calibrationId to it } }
            if (bindings.toSet().size != bindings.size) throw Malformed()
            list to parsed
        } catch (error: Malformed) {
            throw DecisionCalibrationException(DecisionCalibrationReason.MALFORMED)
        }

        val model = root["model"] as? Json.Obj ?: throw DecisionCalibrationException(DecisionCalibrationReason.MALFORMED)
        val manifestModelId = (model["id"] as? Json.Str)?.value
        val manifestRevision = (model["revision"] as? Json.Str)?.value
        val digests = fileDigests(root)
        val runtimeIds = ((root["runtimes"] as? Json.Arr)?.items ?: emptyList())
            .mapNotNull { ((it as? Json.Obj)?.get("id") as? Json.Str)?.value }.toSet()
        val labelsPath = ((root["labels"] as? Json.Obj)?.get("path") as? Json.Str)?.value
        val labelsDigest = labelsPath?.let { digests[it] }
        profiles.zip(items).forEach { (profile, item) ->
            if (profileDigest(item as Json.Obj) != profile.profileDigest) {
                throw DecisionCalibrationException(DecisionCalibrationReason.DIGEST_MISMATCH)
            }
            if (profile.modelId != manifestModelId || profile.modelRevision != manifestRevision) {
                throw DecisionCalibrationException(DecisionCalibrationReason.MODEL_MISMATCH)
            }
            for (binding in profile.runtimes) {
                if (binding.runtimeId !in runtimeIds || runtimeDigest(root, binding.runtimeId) != binding.digest) {
                    throw DecisionCalibrationException(DecisionCalibrationReason.RUNTIME_MISMATCH)
                }
            }
            if (profile.scopeKind == "labels" && (labelsDigest == null || labelsDigest != profile.scopeDigest)) {
                throw DecisionCalibrationException(DecisionCalibrationReason.SCOPE_MISMATCH)
            }
            if (profile.evaluationPath != null && digests[profile.evaluationPath] != profile.evaluationSha256) {
                throw DecisionCalibrationException(DecisionCalibrationReason.EVIDENCE_MISMATCH)
            }
        }
        return profiles
    }

    private fun sha256(json: Json): String = "sha256:" + PortableBundle.sha256Hex(Json.encode(json))

    private fun profileDigest(profile: Json.Obj): String =
        sha256(Json.Obj(profile.members.filter { it.first != "profile_digest" }))

    private fun runtimeDigest(root: Json.Obj, runtimeId: String): String? {
        val runtime = ((root["runtimes"] as? Json.Arr)?.items ?: emptyList())
            .mapNotNull { it as? Json.Obj }
            .firstOrNull { (it["id"] as? Json.Str)?.value == runtimeId } ?: return null
        val digests = fileDigests(root)
        val files = ((runtime["files"] as? Json.Arr)?.items ?: return null).map { entry ->
            val path = (entry as? Json.Str)?.value ?: return null
            Json.Obj(listOf("path" to Json.Str(path), "sha256" to Json.Str(digests[path] ?: return null)))
        }
        return sha256(
            Json.Obj(
                listOf(
                    "entrypoint" to (runtime["entrypoint"] ?: Json.Null),
                    "files" to Json.Arr(files),
                    "format" to (runtime["format"] ?: Json.Null),
                    "id" to Json.Str(runtimeId),
                    "precision" to (runtime["precision"] ?: Json.Null),
                ),
            ),
        )
    }

    private fun fileDigests(root: Json.Obj): Map<String, String> =
        ((root["files"] as? Json.Arr)?.items ?: emptyList()).mapNotNull { entry ->
            val file = entry as? Json.Obj ?: return@mapNotNull null
            val path = (file["path"] as? Json.Str)?.value ?: return@mapNotNull null
            val digest = (file["sha256"] as? Json.Str)?.value ?: return@mapNotNull null
            path to digest
        }.toMap()

    private fun semverKey(value: String): List<Long> = value.split(".").map { it.toLong() }

    private fun compare(left: List<Long>, right: List<Long>): Int {
        for (index in left.indices) {
            if (left[index] != right[index]) return left[index].compareTo(right[index])
        }
        return 0
    }

    private fun obj(value: Json?, vararg keys: String): Json.Obj {
        if (value !is Json.Obj || !value.hasExactKeys(*keys)) throw Malformed()
        return value
    }

    private fun text(value: Json?, pattern: Regex): String {
        val string = (value as? Json.Str)?.value ?: throw Malformed()
        if (!pattern.matches(string)) throw Malformed()
        return string
    }

    private fun integer(value: Json?, minimum: Long, maximum: Long): Long {
        val number = (value as? Json.Num)?.value ?: throw Malformed()
        if (number < minimum || number > maximum) throw Malformed()
        return number
    }

    private fun parseMethod(name: Json?, value: Json?): DecisionCalibrationMethod {
        val method = (name as? Json.Str)?.value ?: throw Malformed()
        if (method !in METHODS) throw Malformed()
        return when (method) {
            "temperature_scaling" -> {
                val parameters = obj(value, "temperature_ppm")
                DecisionCalibrationMethod.TemperatureScaling(integer(parameters["temperature_ppm"], 1, MAX_TEMPERATURE))
            }
            "platt_scaling" -> {
                val parameters = obj(value, "intercept_ppm", "slope_ppm")
                val slope = integer(parameters["slope_ppm"], -MAX_PLATT, MAX_PLATT)
                if (slope == 0L) throw Malformed()
                DecisionCalibrationMethod.PlattScaling(slope, integer(parameters["intercept_ppm"], -MAX_PLATT, MAX_PLATT))
            }
            "isotonic_regression" -> {
                val raw = (obj(value, "points")["points"] as? Json.Arr)?.items ?: throw Malformed()
                if (raw.size !in 2..MAX_POINTS) throw Malformed()
                val points = mutableListOf<DecisionCalibrationPoint>()
                for (item in raw) {
                    val point = obj(item, "input_ppm", "output_ppm")
                    val next = DecisionCalibrationPoint(
                        integer(point["input_ppm"], 0, PPM_SCALE),
                        integer(point["output_ppm"], 0, PPM_SCALE),
                    )
                    val last = points.lastOrNull()
                    if (last != null && (next.inputPpm <= last.inputPpm || next.outputPpm < last.outputPpm)) {
                        throw Malformed()
                    }
                    points += next
                }
                DecisionCalibrationMethod.IsotonicRegression(points)
            }
            else -> {
                obj(value)
                DecisionCalibrationMethod.DeterministicMargin
            }
        }
    }

    private fun parseProfile(value: Json): BundleCalibrationProfile {
        val profile = obj(value, *PROFILE_KEYS)
        val calibrationId = text(profile["calibration_id"], CONTROLLED)
        val version = text(profile["version"], Patterns.SEMVER)
        val method = parseMethod(profile["method"], profile["parameters"])

        val thresholds = obj(
            profile["thresholds"], "minimum_confidence_ppm", "minimum_margin_ppm", "multi_label_threshold_ppm",
        )
        val confidence = integer(thresholds["minimum_confidence_ppm"], 0, PPM_SCALE)
        val margin = integer(thresholds["minimum_margin_ppm"], 0, PPM_SCALE)
        val multiLabel = integer(thresholds["multi_label_threshold_ppm"], 0, PPM_SCALE)
        if (margin > confidence) throw Malformed()

        val scope = obj(profile["scope"], "digest", "kind")
        val kind = (scope["kind"] as? Json.Str)?.value ?: throw Malformed()
        if (kind !in SCOPE_KINDS) throw Malformed()
        val scopeDigest = text(scope["digest"], Patterns.SHA256)

        val model = obj(profile["model"], "id", "revision", "runtimes")
        val modelId = text(model["id"], Patterns.MODEL_ID)
        val revision = if (model["revision"] == Json.Null) null else text(model["revision"], Patterns.REVISION)
        val rawRuntimes = (model["runtimes"] as? Json.Arr)?.items ?: throw Malformed()
        if (rawRuntimes.size !in 1..MAX_RUNTIMES) throw Malformed()
        val bindings = rawRuntimes.map {
            val binding = obj(it, "digest", "id")
            DecisionCalibrationRuntimeBinding(text(binding["id"], Patterns.TOKEN), text(binding["digest"], Patterns.SHA256))
        }
        if (bindings.map { it.runtimeId }.toSet().size != bindings.size) throw Malformed()

        val range = obj(profile["decision_schema"], "max_exclusive", "min")
        val schemaMin = text(range["min"], Patterns.SEMVER)
        val schemaMax = text(range["max_exclusive"], Patterns.SEMVER)
        if (compare(semverKey(schemaMin), semverKey(schemaMax)) >= 0) throw Malformed()

        val provenance = obj(
            profile["provenance"], "dataset_digest", "evaluation", "in_domain_digest", "out_of_domain_digest",
        )
        val datasetDigest = text(provenance["dataset_digest"], Patterns.SHA256)
        val inDomain = text(provenance["in_domain_digest"], Patterns.SHA256)
        val outOfDomain = text(provenance["out_of_domain_digest"], Patterns.SHA256)
        var evaluationPath: String? = null
        var evaluationSha256: String? = null
        if (provenance["evaluation"] != Json.Null) {
            val evaluation = obj(provenance["evaluation"], "path", "sha256")
            evaluationPath = text(evaluation["path"], EVALUATION_PATH)
            evaluationSha256 = text(evaluation["sha256"], Patterns.SHA256)
        }
        return BundleCalibrationProfile(
            calibrationId = calibrationId,
            version = version,
            method = method,
            minimumConfidencePpm = confidence,
            minimumMarginPpm = margin,
            multiLabelThresholdPpm = multiLabel,
            scopeKind = kind,
            scopeDigest = scopeDigest,
            modelId = modelId,
            modelRevision = revision,
            runtimes = bindings,
            decisionSchemaMin = schemaMin,
            decisionSchemaMaxExclusive = schemaMax,
            datasetDigest = datasetDigest,
            inDomainDigest = inDomain,
            outOfDomainDigest = outOfDomain,
            evaluationPath = evaluationPath,
            evaluationSha256 = evaluationSha256,
            profileDigest = text(profile["profile_digest"], Patterns.SHA256),
        )
    }
}
