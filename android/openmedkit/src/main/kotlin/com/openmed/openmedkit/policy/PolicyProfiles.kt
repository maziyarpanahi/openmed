package com.openmed.openmedkit.policy

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.intOrNull
import java.security.MessageDigest

/**
 * Supported de-identification policy actions.
 */
enum class PolicyAction(val wireName: String) {
    KEEP("keep"),
    REDACT("redact"),
    REPLACE("replace"),
    MASK("mask"),
    REMOVE("remove"),
    HASH("hash"),
    ;

    val redactionEquivalent: PolicyAction
        get() = if (this == REDACT) MASK else this

    /** Redaction strength used by `high_recall_union` arbitration. */
    val strength: Int
        get() = when (this) {
            KEEP -> 0
            REPLACE -> 1
            HASH -> 2
            MASK -> 3
            REDACT -> 4
            REMOVE -> 5
        }

    companion object {
        fun fromWireName(value: String): PolicyAction =
            values().firstOrNull { it.wireName == value }
                ?: throw PolicyProfileException("action_unknown", "$.action")

        fun fromWireNameOrNull(value: String?): PolicyAction? =
            values().firstOrNull { it.wireName == value }
    }
}

/**
 * Value-free policy failure: a stable reason code and a JSON field path.
 *
 * Loading, verification, and decision errors never echo profile content,
 * labels, source text, or surrogate keys.
 */
class PolicyProfileException(
    val code: String,
    val fieldPath: String = "$",
) : IllegalArgumentException("$code at $fieldPath")

/** Threshold entry of a portable policy profile, in parts per million. */
data class PolicyThreshold(
    val keepFloorPpm: Int,
    val escalateBelowPpm: Int,
)

/**
 * Portable label taxonomy shared by every bundled profile, generated from
 * `openmed/core/labels.py`.
 */
data class PolicyTaxonomy(
    val labels: List<String>,
    val policyLabels: Map<String, String>,
    val specificity: Map<String, Int>,
    val aliases: Map<String, String>,
    val unknownLabel: String,
    val digest: String,
) {
    /** Normalize a detector label; unknown labels map to [unknownLabel]. */
    fun normalize(label: String): Pair<String, Boolean> {
        val canonical = aliases[labelKey(label)]
        return if (canonical == null) unknownLabel to false else canonical to true
    }

    companion object {
        const val SCHEMA: String = "openmed.portable_taxonomy"
        const val VERSION: Int = 1

        /** Strip whitespace and one BIOES prefix, lowercase, keep `a-z0-9`. */
        fun labelKey(label: String): String {
            val trimmed = label.trim()
            val withoutBioes = if (trimmed.length >= 2 && trimmed[0] in "BIES" && trimmed[1] == '-') {
                trimmed.substring(2)
            } else {
                trimmed
            }
            return withoutBioes.lowercase().filter { it in 'a'..'z' || it in '0'..'9' }
        }
    }
}

/**
 * Loaded, digest-verified OpenMed de-identification policy profile.
 */
data class PolicyProfile(
    val schemaVersion: Int,
    val name: String,
    val posture: String,
    val thresholdProfile: String,
    val defaultAction: PolicyAction,
    val defaultActionBias: String,
    val arbitrationMode: String,
    val strictNoLeak: Boolean,
    val safetySweepMandatory: Boolean,
    val keepMapping: Boolean,
    val reversibleId: Boolean,
    val forcedCascadeTiers: List<String>,
    val policyLabelActions: Map<String, PolicyAction>,
    val actions: Map<String, PolicyAction>,
    val digest: String = "",
    val taxonomy: PolicyTaxonomy = PolicyProfiles.bundledTaxonomy(),
    val defaultThreshold: PolicyThreshold = PolicyThreshold(0, 0),
    val labelThresholds: Map<String, PolicyThreshold> = emptyMap(),
    val requiredCapabilities: List<String> = emptyList(),
    /** Python-only features of the source profile the portable subset omits. */
    val excludedFeatures: List<String> = emptyList(),
) {
    fun actionFor(label: String): PolicyAction {
        val action = configuredAction(canonicalLabel(label))
        return if (strictNoLeak && action == PolicyAction.KEEP) {
            PolicyAction.MASK
        } else {
            action
        }
    }

    fun canonicalLabel(label: String): String = taxonomy.normalize(label).first

    fun thresholdFor(canonicalLabel: String): PolicyThreshold =
        labelThresholds[canonicalLabel] ?: defaultThreshold

    internal fun configuredAction(canonicalLabel: String): PolicyAction =
        actions[canonicalLabel]
            ?: taxonomy.policyLabels[canonicalLabel]?.let { policyLabelActions[it] }
            ?: defaultAction
}

/** Decoded portable policy manifest. */
data class PolicyManifest(
    val defaultProfile: String,
    val profiles: Map<String, String>,
    val excludedProfiles: Map<String, String>,
    val profileAliases: Map<String, String>,
    val taxonomyDigest: String,
)

/**
 * Loader for the bundled portable policy profiles.
 *
 * Profiles are generated from `openmed/core/policies` by
 * `scripts/parity/generate_portable_policies.py` and shipped as byte-identical
 * assets. Loading verifies the manifest, the profile and taxonomy digests,
 * the IR version, and the required capabilities, and fails closed with a
 * value-free [PolicyProfileException].
 */
object PolicyProfiles {
    const val DEFAULT_PROFILE = "hipaa_safe_harbor"
    const val MANIFEST_FILE = "portable_policy_manifest.json"
    const val TAXONOMY_FILE = "portable_policy_taxonomy.json"
    const val MANIFEST_SCHEMA = "openmed.portable_policy_manifest"
    const val MANIFEST_VERSION = 1
    const val IR_SCHEMA = "openmed.portable_policy"
    const val IR_VERSION = 1
    const val MERGE_ALGORITHM = "openmed.portable_merge"
    const val MERGE_VERSION = 1
    const val SURROGATE_ALGORITHM = "openmed.portable_surrogate.hmac_sha256"
    const val SURROGATE_VERSION = 1

    /** Capabilities this runtime implements. */
    val SUPPORTED_CAPABILITIES: List<String> = listOf(
        "actions.v1",
        "labels.portable_taxonomy.v1",
        "merge.cluster_union.v1",
        "merge.semantic_units.v1",
        "strict_no_leak.v1",
        "surrogate.hmac_sha256.v1",
        "thresholds.keep_floor.v1",
    )

    private val json = Json { ignoreUnknownKeys = true }

    private val bundledManifestValue: PolicyManifest? by lazy {
        bundledResource(MANIFEST_FILE)?.let { runCatching { decodeManifest(it) }.getOrNull() }
    }

    private val bundledTaxonomyValue: PolicyTaxonomy? by lazy {
        runCatching { loadBundledTaxonomy() }.getOrNull()
    }

    /** Portable profile names bundled with OpenMedKit. */
    val bundledProfileNames: List<String>
        get() = bundledManifestValue?.profiles?.keys?.sorted().orEmpty()

    /**
     * Bundled taxonomy. Throws `resource_missing` or
     * `taxonomy_digest_mismatch` when it cannot be verified.
     */
    fun bundledTaxonomy(): PolicyTaxonomy =
        bundledTaxonomyValue ?: loadBundledTaxonomy()

    /** Bundled taxonomy labels, or an empty list when it cannot be verified. */
    fun bundledTaxonomyLabels(): List<String> = bundledTaxonomyValue?.labels.orEmpty()

    /** Return a bundled resource by file name. */
    fun bundledResource(file: String): ByteArray? =
        javaClass.classLoader?.getResourceAsStream("policies/$file")?.use { it.readBytes() }

    /**
     * Load and verify a profile.
     *
     * Verification order: manifest, profile name, caller-pinned digest,
     * profile resource digest, taxonomy resource digest, profile document.
     */
    fun load(
        name: String = DEFAULT_PROFILE,
        expectedDigest: String? = null,
        resources: (String) -> ByteArray? = ::bundledResource,
        supportedCapabilities: List<String> = SUPPORTED_CAPABILITIES,
    ): PolicyProfile {
        val manifestBytes = resources(MANIFEST_FILE)
            ?: throw PolicyProfileException("resource_missing", "$.manifest")
        val manifest = decodeManifest(manifestBytes)
        val canonical = resolveProfileName(name, manifest)
        val digest = manifest.profiles.getValue(canonical)
        if (expectedDigest != null && expectedDigest != digest) {
            throw PolicyProfileException("digest_mismatch", "$.digest")
        }
        val bytes = resources("$canonical.json")
            ?: throw PolicyProfileException("resource_missing", "$.profiles")
        if (sha256Digest(bytes) != digest) {
            throw PolicyProfileException("resource_digest_mismatch", "$.profiles")
        }
        val taxonomyBytes = resources(TAXONOMY_FILE)
            ?: throw PolicyProfileException("resource_missing", "$.taxonomy_digest")
        val taxonomyDigest = sha256Digest(taxonomyBytes)
        if (taxonomyDigest != manifest.taxonomyDigest) {
            throw PolicyProfileException("taxonomy_digest_mismatch", "$.taxonomy_digest")
        }
        val taxonomy = decodeTaxonomy(taxonomyBytes, taxonomyDigest)
        return decodeProfile(bytes, canonical, digest, taxonomy, supportedCapabilities)
    }

    fun canonicalProfileName(name: String): String {
        val manifest = bundledManifestValue
            ?: throw PolicyProfileException("resource_missing", "$.manifest")
        return resolveProfileName(name, manifest)
    }

    fun canonicalLabel(label: String): String =
        bundledTaxonomyValue?.normalize(label)?.first ?: "OTHER"

    /** `sha256:` digest of resource bytes. */
    fun sha256Digest(bytes: ByteArray): String =
        "sha256:" + MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    fun decodeManifest(bytes: ByteArray): PolicyManifest {
        val root = parseObject(bytes, "manifest_invalid")
        if (root.string("schema") != MANIFEST_SCHEMA) {
            throw PolicyProfileException("manifest_invalid", "$.schema")
        }
        if (root.int("schema_version") != MANIFEST_VERSION) {
            throw PolicyProfileException("manifest_version_unsupported", "$.schema_version")
        }
        if (root.string("ir_schema") != IR_SCHEMA) {
            throw PolicyProfileException("manifest_invalid", "$.ir_schema")
        }
        if (root.int("ir_version") != IR_VERSION) {
            throw PolicyProfileException("ir_version_unsupported", "$.ir_version")
        }
        val profiles = root.stringMap("profiles")
            ?: throw PolicyProfileException("manifest_invalid", "$.profiles")
        val excluded = root.stringMap("excluded_profiles")
            ?: throw PolicyProfileException("manifest_invalid", "$.excluded_profiles")
        val aliases = root.stringMap("profile_aliases")
            ?: throw PolicyProfileException("manifest_invalid", "$.profile_aliases")
        if (!profiles.values.all(::isDigest)) {
            throw PolicyProfileException("manifest_invalid", "$.profiles")
        }
        val taxonomyDigest = root.string("taxonomy_digest")
        if (taxonomyDigest == null || !isDigest(taxonomyDigest)) {
            throw PolicyProfileException("manifest_invalid", "$.taxonomy_digest")
        }
        val defaultProfile = root.string("default_profile")
        if (defaultProfile == null || defaultProfile !in profiles) {
            throw PolicyProfileException("manifest_invalid", "$.default_profile")
        }
        return PolicyManifest(defaultProfile, profiles, excluded, aliases, taxonomyDigest)
    }

    fun resolveProfileName(name: String, manifest: PolicyManifest): String {
        val trimmed = name.trim()
        if (trimmed.isEmpty()) {
            throw PolicyProfileException("profile_name_invalid", "$.name")
        }
        val normalized = trimmed.lowercase().replace('-', '_')
        val canonical = manifest.profileAliases[normalized] ?: normalized
        if (canonical in manifest.profiles) {
            return canonical
        }
        if (canonical in manifest.excludedProfiles) {
            throw PolicyProfileException("profile_not_portable", "$.name")
        }
        throw PolicyProfileException("profile_unknown", "$.name")
    }

    fun decodeTaxonomy(bytes: ByteArray, digest: String): PolicyTaxonomy {
        val root = parseObject(bytes, "taxonomy_invalid")
        if (root.string("name") != PolicyTaxonomy.SCHEMA) {
            throw PolicyProfileException("taxonomy_invalid", "$.name")
        }
        if (root.int("version") != PolicyTaxonomy.VERSION) {
            throw PolicyProfileException("taxonomy_invalid", "$.version")
        }
        val rawLabels = root["labels"] as? JsonObject
        if (rawLabels == null || rawLabels.isEmpty()) {
            throw PolicyProfileException("taxonomy_invalid", "$.labels")
        }
        val policyLabels = mutableMapOf<String, String>()
        val specificity = mutableMapOf<String, Int>()
        for ((label, entry) in rawLabels) {
            val fields = entry as? JsonObject
            val policyLabel = fields?.string("policy_label")
            val rank = fields?.int("specificity")
            if (policyLabel == null || rank == null) {
                throw PolicyProfileException("taxonomy_invalid", "$.labels")
            }
            policyLabels[label] = policyLabel
            specificity[label] = rank
        }
        val aliases = root.stringMap("aliases")
        if (aliases == null || !aliases.values.all { it in policyLabels }) {
            throw PolicyProfileException("taxonomy_invalid", "$.aliases")
        }
        val unknown = root.string("unknown_label")
        if (unknown == null || unknown !in policyLabels) {
            throw PolicyProfileException("taxonomy_invalid", "$.unknown_label")
        }
        return PolicyTaxonomy(
            labels = policyLabels.keys.sorted(),
            policyLabels = policyLabels,
            specificity = specificity,
            aliases = aliases,
            unknownLabel = unknown,
            digest = digest,
        )
    }

    /** Validate profile document bytes in the order every kit shares. */
    fun decodeProfile(
        bytes: ByteArray,
        name: String,
        digest: String,
        taxonomy: PolicyTaxonomy,
        supportedCapabilities: List<String> = SUPPORTED_CAPABILITIES,
    ): PolicyProfile {
        val root = parseObject(bytes, "profile_invalid")
        if (root.string("ir_schema") != IR_SCHEMA) {
            throw PolicyProfileException("profile_invalid", "$.ir_schema")
        }
        if (root.int("ir_version") != IR_VERSION) {
            throw PolicyProfileException("ir_version_unsupported", "$.ir_version")
        }
        if (root.string("name") != name) {
            throw PolicyProfileException("profile_invalid", "$.name")
        }
        if (root.string("taxonomy_digest") != taxonomy.digest) {
            throw PolicyProfileException("taxonomy_digest_mismatch", "$.taxonomy_digest")
        }
        val required = root.stringList("required_capabilities")
            ?: throw PolicyProfileException("profile_invalid", "$.required_capabilities")
        if (!required.all { it in supportedCapabilities }) {
            throw PolicyProfileException("capability_missing", "$.required_capabilities")
        }
        val merge = root["merge"] as? JsonObject
        if (merge?.string("algorithm") != MERGE_ALGORITHM || merge.int("version") != MERGE_VERSION) {
            throw PolicyProfileException("merge_algorithm_unsupported", "$.merge")
        }
        val surrogate = root["surrogate"] as? JsonObject
        if (surrogate?.string("algorithm") != SURROGATE_ALGORITHM ||
            surrogate.int("version") != SURROGATE_VERSION
        ) {
            throw PolicyProfileException("surrogate_algorithm_unsupported", "$.surrogate")
        }
        val mode = root.string("arbitration_mode")
        if (mode != "balanced" && mode != "high_recall_union") {
            throw PolicyProfileException("profile_invalid", "$.arbitration_mode")
        }
        val strict = root.boolean("strict_no_leak")
            ?: throw PolicyProfileException("profile_invalid", "$.strict_no_leak")
        val rawActions = root.stringMap("actions")
        if (rawActions == null || rawActions.keys != taxonomy.labels.toSet()) {
            throw PolicyProfileException("profile_invalid", "$.actions")
        }
        val actions = rawActions.mapValues { (_, value) ->
            PolicyAction.fromWireNameOrNull(value)
                ?: throw PolicyProfileException("profile_invalid", "$.actions")
        }
        val defaultAction = PolicyAction.fromWireNameOrNull(root.string("default_action"))
            ?: throw PolicyProfileException("profile_invalid", "$.default_action")
        val thresholds = root["thresholds"] as? JsonObject
            ?: throw PolicyProfileException("profile_invalid", "$.thresholds")
        val defaultThreshold = threshold(thresholds["default"])
        val rawLabelThresholds = thresholds["labels"] as? JsonObject
        if (rawLabelThresholds == null || !rawLabelThresholds.keys.all { it in taxonomy.policyLabels }) {
            throw PolicyProfileException("profile_invalid", "$.thresholds")
        }
        val labelThresholds = rawLabelThresholds.mapValues { (_, entry) -> threshold(entry) }
        val policyLabelActions = root.stringMap("policy_label_actions").orEmpty()
            .mapNotNull { (label, value) -> PolicyAction.fromWireNameOrNull(value)?.let { label to it } }
            .toMap()
        return PolicyProfile(
            schemaVersion = root.int("schema_version") ?: 1,
            name = name,
            posture = root.string("posture").orEmpty(),
            thresholdProfile = root.string("threshold_profile").orEmpty(),
            defaultAction = defaultAction,
            defaultActionBias = root.string("default_action_bias") ?: defaultAction.wireName,
            arbitrationMode = mode,
            strictNoLeak = strict,
            safetySweepMandatory = root.boolean("safety_sweep_mandatory") ?: false,
            keepMapping = root.boolean("keep_mapping") ?: false,
            reversibleId = root.boolean("reversible_id") ?: false,
            forcedCascadeTiers = root.stringList("forced_cascade_tiers").orEmpty(),
            policyLabelActions = policyLabelActions,
            actions = actions,
            digest = digest,
            taxonomy = taxonomy,
            defaultThreshold = defaultThreshold,
            labelThresholds = labelThresholds,
            requiredCapabilities = required,
            excludedFeatures = root.stringList("excluded_features").orEmpty(),
        )
    }

    private fun loadBundledTaxonomy(): PolicyTaxonomy {
        val manifest = bundledManifestValue
            ?: throw PolicyProfileException("resource_missing", "$.manifest")
        val bytes = bundledResource(TAXONOMY_FILE)
            ?: throw PolicyProfileException("resource_missing", "$.taxonomy_digest")
        val digest = sha256Digest(bytes)
        if (digest != manifest.taxonomyDigest) {
            throw PolicyProfileException("taxonomy_digest_mismatch", "$.taxonomy_digest")
        }
        return decodeTaxonomy(bytes, digest)
    }

    private fun threshold(entry: JsonElement?): PolicyThreshold {
        val fields = entry as? JsonObject
        val keep = fields?.int("keep_floor_ppm")
        val escalate = fields?.int("escalate_below_ppm")
        if (keep == null || escalate == null || keep !in 0..CONFIDENCE_SCALE || escalate !in 0..CONFIDENCE_SCALE) {
            throw PolicyProfileException("profile_invalid", "$.thresholds")
        }
        return PolicyThreshold(keep, escalate)
    }

    private fun isDigest(value: String): Boolean =
        value.length == 71 && value.startsWith("sha256:") &&
            value.substring(7).all { it in '0'..'9' || it in 'a'..'f' }

    private fun parseObject(bytes: ByteArray, code: String): JsonObject =
        try {
            json.parseToJsonElement(bytes.toString(Charsets.UTF_8)) as? JsonObject
        } catch (_: Exception) {
            null
        } ?: throw PolicyProfileException(code)

    private fun JsonObject.string(key: String): String? =
        (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content

    private fun JsonObject.int(key: String): Int? =
        (this[key] as? JsonPrimitive)?.takeIf { !it.isString }?.intOrNull

    private fun JsonObject.boolean(key: String): Boolean? =
        (this[key] as? JsonPrimitive)?.takeIf { !it.isString }?.booleanOrNull

    private fun JsonObject.stringMap(key: String): Map<String, String>? {
        val fields = this[key] as? JsonObject ?: return null
        val result = mutableMapOf<String, String>()
        for ((name, value) in fields) {
            val primitive = value as? JsonPrimitive
            if (primitive == null || !primitive.isString) {
                return null
            }
            result[name] = primitive.content
        }
        return result
    }

    private fun JsonObject.stringList(key: String): List<String>? {
        val items = this[key] as? JsonArray ?: return null
        return items.map { item ->
            val primitive = item as? JsonPrimitive
            if (primitive == null || !primitive.isString) {
                return null
            }
            primitive.content
        }
    }

    private const val CONFIDENCE_SCALE = 1_000_000
}
