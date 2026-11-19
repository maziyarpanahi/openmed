package com.openmed.openmedkit.bundle

/** Stable, value-free execution-plan reason codes, in evaluation order. */
enum class ExecutionPlanReason(val code: String) {
    REMOTE_EXECUTION_UNSUPPORTED("remote_execution_unsupported"),
    BUNDLE_REVOKED("bundle_revoked"),
    BUNDLE_UNVERIFIED("bundle_unverified"),
    BUNDLE_FILES_UNVERIFIED("bundle_files_unverified"),
    SDK_INCOMPATIBLE("sdk_incompatible"),
    EXECUTION_METADATA_INVALID("execution_metadata_invalid"),
    NO_ELIGIBLE_RUNTIME("no_eligible_runtime"),
    RUNTIME_NOT_ELIGIBLE("runtime_not_eligible"),
    ARTIFACT_REVOKED("artifact_revoked"),
    FORMAT_UNSUPPORTED("format_unsupported"),
    PLATFORM_UNSUPPORTED("platform_unsupported"),
    OS_VERSION_UNKNOWN("os_version_unknown"),
    OS_VERSION_UNSUPPORTED("os_version_unsupported"),
    PRECISION_UNSUPPORTED("precision_unsupported"),
    PRECISION_DISALLOWED("precision_disallowed"),
    MEMORY_INSUFFICIENT("memory_insufficient"),
    STORAGE_INSUFFICIENT("storage_insufficient"),
    MODEL_TIER_EXCEEDED("model_tier_exceeded"),
    SEQUENCE_LIMIT_EXCEEDED("sequence_limit_exceeded"),
    SEQUENCE_LIMIT_UNDECLARED("sequence_limit_undeclared"),
    OPERATOR_SET_UNKNOWN("operator_set_unknown"),
    OPERATOR_SET_UNSUPPORTED("operator_set_unsupported"),
    FEATURE_UNAVAILABLE("feature_unavailable"),
    EVIDENCE_MISSING("evidence_missing"),
    PROVIDER_UNAVAILABLE("provider_unavailable"),
    PROVIDER_DISALLOWED("provider_disallowed"),
    ENERGY_POLICY_EXCLUDED("energy_policy_excluded"),
    NOT_DECLARED_FALLBACK("not_declared_fallback"),
    FALLBACK_BOUND_REACHED("fallback_bound_reached"),
}

/**
 * Caller policy applied on top of the bundle and the device profile.
 *
 * [localOnly] must be `true`: version 1 plans never contain a remote or cloud
 * step, so `false` resolves to `remote_execution_unsupported`.
 */
class ExecutionConstraints(
    val localOnly: Boolean = true,
    allowedPrecisions: Collection<String>? = null,
    disallowedProviders: Collection<String> = emptyList(),
    val maxMemoryBytes: Long? = null,
    val minSequenceLength: Long? = null,
    /** `balanced` or `low_power`; `low_power` excludes GPU providers. */
    val energy: String = "balanced",
    requiredEvidence: Collection<String> = emptyList(),
    /** `sha256:` file digests no step may use. */
    revokedArtifacts: Collection<String> = emptyList(),
    val maxSteps: Long = DEFAULT_MAX_STEPS,
    /** Wall-clock budget the executor applies across steps; never benchmarked. */
    val deadlineMs: Long? = null,
    val requireVerifiedFiles: Boolean = true,
) {
    val allowedPrecisions: List<String>?
    val disallowedProviders: List<String>
    val requiredEvidence: List<String>
    val revokedArtifacts: List<String>

    init {
        val valid = energy in ENERGY_POSTURES &&
            (maxMemoryBytes == null || maxMemoryBytes in 0..PortableBundle.MAX_SAFE_INTEGER) &&
            (minSequenceLength == null || minSequenceLength in 1..DeviceCapabilityVocabulary.MAX_SEQUENCE_LENGTH) &&
            maxSteps in 1..MAX_STEPS_LIMIT && (deadlineMs == null || deadlineMs in 1..MAX_DEADLINE_MS) &&
            (allowedPrecisions ?: emptyList()).all { it in DeviceCapabilityVocabulary.PRECISIONS } &&
            disallowedProviders.all { it in DeviceCapabilityVocabulary.PROVIDER_ACCELERATORS } &&
            requiredEvidence.all { Patterns.TOKEN.matches(it) } && revokedArtifacts.all { Patterns.SHA256.matches(it) }
        if (!valid) throw DeviceCapabilityException(DeviceCapabilityReason.CONSTRAINTS_INVALID)
        this.allowedPrecisions = allowedPrecisions?.distinct()?.sorted()
        this.disallowedProviders = disallowedProviders.distinct().sorted()
        this.requiredEvidence = requiredEvidence.distinct().sorted()
        this.revokedArtifacts = revokedArtifacts.distinct().sorted()
    }

    internal fun document(): Json = Json.Obj(
        listOf(
            "allowed_precisions" to (allowedPrecisions?.let { strings(it) } ?: Json.Null),
            "deadline_ms" to optional(deadlineMs),
            "disallowed_providers" to strings(disallowedProviders),
            "energy" to Json.Str(energy),
            "local_only" to Json.Bool(localOnly),
            "max_memory_bytes" to optional(maxMemoryBytes),
            "max_steps" to Json.Num(maxSteps),
            "min_sequence_length" to optional(minSequenceLength),
            "require_verified_files" to Json.Bool(requireVerifiedFiles),
            "required_evidence" to strings(requiredEvidence),
            "revoked_artifacts" to strings(revokedArtifacts),
            "schema" to Json.Str(SCHEMA),
            "schema_version" to Json.Num(1),
        ),
    )

    /** The canonical constraints document bytes. */
    fun canonicalJson(): ByteArray = Json.encode(document())

    /** `sha256:` digest of the canonical constraints document. */
    val digest: String get() = ExecutionPlanResolver.digest(document())

    override fun equals(other: Any?): Boolean =
        other is ExecutionConstraints && canonicalJson().contentEquals(other.canonicalJson())

    override fun hashCode(): Int = canonicalJson().contentHashCode()

    companion object {
        const val SCHEMA = "openmed.execution_constraints"
        const val DEFAULT_MAX_STEPS = 4L
        const val MAX_STEPS_LIMIT = 16L
        const val MAX_DEADLINE_MS = 86_400_000L
        val ENERGY_POSTURES: Set<String> = setOf("balanced", "low_power")
        private val DOCUMENT_KEYS = arrayOf(
            "allowed_precisions", "deadline_ms", "disallowed_providers", "energy", "local_only", "max_memory_bytes",
            "max_steps", "min_sequence_length", "require_verified_files", "required_evidence", "revoked_artifacts",
            "schema", "schema_version",
        )

        /** Parse a constraints document strictly; unknown or missing members fail. */
        fun fromJson(data: ByteArray): ExecutionConstraints {
            val fail = { DeviceCapabilityException(DeviceCapabilityReason.CONSTRAINTS_INVALID) }
            val value = try {
                Json.parse(data)
            } catch (error: Json.Invalid) {
                throw fail()
            }
            if (value !is Json.Obj || !value.hasExactKeys(*DOCUMENT_KEYS) || value["schema"] != Json.Str(SCHEMA) ||
                value["schema_version"] != Json.Num(1)
            ) {
                throw fail()
            }
            return try {
                ExecutionConstraints(
                    localOnly = value.bool("local_only") ?: throw fail(),
                    allowedPrecisions = if (value["allowed_precisions"] == Json.Null) {
                        null
                    } else {
                        value.strings("allowed_precisions") ?: throw fail()
                    },
                    disallowedProviders = value.strings("disallowed_providers") ?: throw fail(),
                    maxMemoryBytes = value.optionalLong("max_memory_bytes"),
                    minSequenceLength = value.optionalLong("min_sequence_length"),
                    energy = value.string("energy") ?: throw fail(),
                    requiredEvidence = value.strings("required_evidence") ?: throw fail(),
                    revokedArtifacts = value.strings("revoked_artifacts") ?: throw fail(),
                    maxSteps = value.long("max_steps") ?: throw fail(),
                    deadlineMs = value.optionalLong("deadline_ms"),
                    requireVerifiedFiles = value.bool("require_verified_files") ?: throw fail(),
                )
            } catch (error: Json.Invalid) {
                throw fail()
            }
        }
    }
}

/** One `(runtime, provider)` attempt in an execution plan. */
data class ExecutionPlanStep(
    val runtimeId: String,
    val runtimeIndex: Int,
    val format: String,
    val precision: String,
    val provider: String,
    val accelerator: String,
    /** Digest of the runtime entrypoint file. */
    val artifactSha256: String,
    val minMemoryBytes: Long,
    val maxSequenceLength: Long?,
    val tier: String?,
    /** Evidence kinds that qualify the runtime. */
    val evidence: List<String>,
    /** `selected`, `provider_fallback`, or `artifact_fallback`. */
    val role: String,
)

/** A runtime, or `(runtime, provider)` pair, the plan does not use. */
data class ExecutionPlanSkip(val runtimeId: String, val provider: String?, val reason: ExecutionPlanReason)

/** Value-free resolver result (`execution_plan`). */
data class ExecutionPlan(
    /** `planned` or `unsupported`. */
    val status: String,
    val reason: ExecutionPlanReason?,
    val bundleContentId: String,
    val profileDigest: String,
    val constraintsDigest: String,
    val verificationDigest: String,
    /** Invalidation key over every input digest and the resolver version. */
    val planKey: String,
    val rootRuntime: String?,
    val steps: List<ExecutionPlanStep>,
    val skipped: List<ExecutionPlanSkip>,
    /** Limits that were unknown and therefore not enforced. */
    val assumptions: List<String>,
    val maxSteps: Long,
    val deadlineMs: Long?,
    val maxMemoryBytes: Long?,
) {
    val isSupported: Boolean get() = status == "planned"

    internal fun document(): Json = Json.Obj(
        listOf(
            "assumptions" to strings(assumptions),
            "bounds" to Json.Obj(
                listOf(
                    "deadline_ms" to optional(deadlineMs),
                    "max_memory_bytes" to optional(maxMemoryBytes),
                    "max_steps" to Json.Num(maxSteps),
                ),
            ),
            "bundle_content_id" to Json.Str(bundleContentId),
            "constraints_digest" to Json.Str(constraintsDigest),
            "local_only" to Json.Bool(true),
            "plan_key" to Json.Str(planKey),
            "profile_digest" to Json.Str(profileDigest),
            "reason" to optional(reason?.code),
            "resolver_version" to Json.Num(ExecutionPlanResolver.RESOLVER_VERSION),
            "root_runtime" to optional(rootRuntime),
            "schema" to Json.Str(ExecutionPlanResolver.SCHEMA),
            "schema_version" to Json.Num(1),
            "skipped" to Json.Arr(
                skipped.map {
                    Json.Obj(
                        listOf(
                            "provider" to optional(it.provider),
                            "reason" to Json.Str(it.reason.code),
                            "runtime_id" to Json.Str(it.runtimeId),
                        ),
                    )
                },
            ),
            "status" to Json.Str(status),
            "steps" to Json.Arr(
                steps.map {
                    Json.Obj(
                        listOf(
                            "accelerator" to Json.Str(it.accelerator),
                            "artifact_sha256" to Json.Str(it.artifactSha256),
                            "evidence" to strings(it.evidence),
                            "format" to Json.Str(it.format),
                            "max_sequence_length" to optional(it.maxSequenceLength),
                            "min_memory_bytes" to Json.Num(it.minMemoryBytes),
                            "precision" to Json.Str(it.precision),
                            "provider" to Json.Str(it.provider),
                            "role" to Json.Str(it.role),
                            "runtime_id" to Json.Str(it.runtimeId),
                            "runtime_index" to Json.Num(it.runtimeIndex.toLong()),
                            "tier" to optional(it.tier),
                        ),
                    )
                },
            ),
            "verification_digest" to Json.Str(verificationDigest),
        ),
    )

    /** The canonical plan document bytes. */
    fun canonicalJson(): ByteArray = Json.encode(document())
}

/**
 * Resolves deterministic execution plans for signed portable bundles.
 *
 * Mirrors `openmed/core/execution_plan.py`: the same checks in the same order
 * produce byte-identical plans for `tests/fixtures/device_capabilities`.
 * Resolution is pure: no network access, no logging, no model loading.
 */
object ExecutionPlanResolver {
    const val SCHEMA = "openmed.execution_plan"
    const val RESOLVER_VERSION = 1L

    /** Manifest extension that declares providers and qualifications per runtime. */
    const val EXTENSION_NAME = "org.openmed.execution"

    /** Providers that can execute each bundle runtime format. */
    val FORMAT_PROVIDERS: Map<String, Set<String>> = sortedMapOf(
        "coreml" to setOf("coreml", "cpu"),
        "mlx" to setOf("cpu", "metal"),
        "onnx" to setOf("coreml", "cpu", "nnapi", "qnn", "wasm", "webgpu"),
        "ort" to setOf("coreml", "cpu", "nnapi", "qnn"),
        "pytorch" to setOf("cpu"),
        "transformersjs" to setOf("wasm", "webgpu"),
    )

    /** Baseline providers for a runtime the execution extension does not describe. */
    val DEFAULT_PROVIDERS: Map<String, List<String>> = sortedMapOf(
        "coreml" to listOf("coreml"),
        "mlx" to listOf("metal"),
        "onnx" to listOf("cpu", "wasm"),
        "ort" to listOf("cpu"),
        "pytorch" to listOf("cpu"),
        "transformersjs" to listOf("wasm"),
    )
    val STEP_ROLES: List<String> = listOf("selected", "provider_fallback", "artifact_fallback")
    val ASSUMPTIONS: List<String> = listOf(
        "device_sequence_limit_unknown", "memory_budget_unknown", "model_tier_limit_unknown", "storage_budget_unknown",
    )
    val INVALIDATION_COMPONENTS: List<String> = listOf("bundle", "constraints", "profile", "resolver", "verification")
    private val META_KEYS = arrayOf("evidence", "features", "max_sequence_length", "min_operator_set", "providers", "tier")

    internal fun digest(value: Json): String = "sha256:" + PortableBundle.sha256Hex(Json.encode(value))

    /** Digest of the verification facts a plan depends on. */
    fun verificationDigest(verification: PortableBundleVerification): String = digest(
        Json.Obj(
            listOf(
                "content_id" to optional(verification.contentId),
                "eligible_runtimes" to Json.Arr(verification.eligibleRuntimes.map { Json.Num(it.toLong()) }),
                "files_verified" to Json.Bool(verification.filesVerified),
                "ok" to Json.Bool(verification.ok),
                "reason" to optional(verification.reason?.code),
            ),
        ),
    )

    /** Plan invalidation key over the input digests and resolver version. */
    fun planKey(
        bundleContentId: String,
        profileDigest: String,
        constraintsDigest: String,
        verificationDigest: String,
        resolverVersion: Long = RESOLVER_VERSION,
    ): String = digest(
        Json.Obj(
            listOf(
                "bundle_content_id" to Json.Str(bundleContentId),
                "constraints_digest" to Json.Str(constraintsDigest),
                "profile_digest" to Json.Str(profileDigest),
                "resolver_version" to Json.Num(resolverVersion),
                "verification_digest" to Json.Str(verificationDigest),
            ),
        ),
    )

    private class MetadataInvalid : Exception("execution_metadata_invalid")

    private class RuntimeMeta(
        val providers: List<String>,
        val features: List<String>,
        val evidence: List<String>,
        val maxSequenceLength: Long?,
        val minOperatorSet: Long?,
        val tier: String?,
        val declared: Boolean,
    ) {
        fun providersOn(platform: String): List<String> = if (declared) {
            providers
        } else {
            val valid = DeviceCapabilityVocabulary.PLATFORM_PROVIDERS[platform].orEmpty()
            providers.filter { it in valid }
        }
    }

    private class Runtime(
        val id: String,
        val format: String,
        val precision: String,
        val entrypoint: String,
        val files: List<String>,
        val platforms: List<Triple<String, String?, String?>>,
        val minMemory: Long,
        val minStorage: Long,
        val fallback: String?,
    )

    /**
     * Resolve an ordered, value-free plan or an explicit refusal. The plan
     * follows exactly one declared fallback chain: the first root runtime (in
     * manifest order) whose chain contains a qualified step.
     */
    fun resolve(
        manifest: ByteArray,
        verification: PortableBundleVerification,
        profile: DeviceCapabilityProfile,
        constraints: ExecutionConstraints = ExecutionConstraints(),
    ): ExecutionPlan {
        val bundleId = PortableBundle.contentId(manifest)
        val profileDigest = profile.digest
        val constraintsDigest = constraints.digest
        val verificationValue = verificationDigest(verification)
        val key = planKey(bundleId, profileDigest, constraintsDigest, verificationValue)
        val memoryLimit = listOfNotNull(profile.memoryBudgetBytes, constraints.maxMemoryBytes).minOrNull()

        fun result(
            reason: ExecutionPlanReason?,
            root: String? = null,
            steps: List<ExecutionPlanStep> = emptyList(),
            skipped: List<ExecutionPlanSkip> = emptyList(),
            assumptions: Set<String> = emptySet(),
        ) = ExecutionPlan(
            status = if (reason == null) "planned" else "unsupported",
            reason = reason,
            bundleContentId = bundleId,
            profileDigest = profileDigest,
            constraintsDigest = constraintsDigest,
            verificationDigest = verificationValue,
            planKey = key,
            rootRuntime = root,
            steps = steps,
            skipped = skipped,
            assumptions = assumptions.sorted(),
            maxSteps = constraints.maxSteps,
            deadlineMs = constraints.deadlineMs,
            maxMemoryBytes = memoryLimit,
        )

        if (!constraints.localOnly) return result(ExecutionPlanReason.REMOTE_EXECUTION_UNSUPPORTED)
        if (!verification.ok) {
            return result(
                if (verification.reason == PortableBundleReason.BUNDLE_REVOKED) {
                    ExecutionPlanReason.BUNDLE_REVOKED
                } else {
                    ExecutionPlanReason.BUNDLE_UNVERIFIED
                },
            )
        }
        val value = parseManifest(manifest)
        if (verification.contentId != bundleId || value == null) return result(ExecutionPlanReason.BUNDLE_UNVERIFIED)
        if (constraints.requireVerifiedFiles && !verification.filesVerified) {
            return result(ExecutionPlanReason.BUNDLE_FILES_UNVERIFIED)
        }
        if (!sdkCompatible(value, profile)) return result(ExecutionPlanReason.SDK_INCOMPATIBLE)
        val runtimes = readRuntimes(value)
        val metas = try {
            parseExtension(value, runtimes)
        } catch (error: MetadataInvalid) {
            return result(ExecutionPlanReason.EXECUTION_METADATA_INVALID)
        }
        val digests = (value["files"] as Json.Arr).items.associate {
            val item = it as Json.Obj
            item.string("path").orEmpty() to item.string("sha256").orEmpty()
        }

        val exclusions = mutableMapOf<String, ExecutionPlanReason>()
        val pairReasons = mutableMapOf<Pair<String, String>, ExecutionPlanReason>()
        val qualified = linkedMapOf<String, List<String>>()
        runtimes.forEachIndexed { index, runtime ->
            val meta = metas.getValue(runtime.id)
            val reason = runtimeReason(index, runtime, meta, digests, verification, profile, constraints, memoryLimit)
            if (reason != null) {
                exclusions[runtime.id] = reason
                return@forEachIndexed
            }
            val providers = mutableListOf<String>()
            for (provider in meta.providersOn(profile.platform)) {
                val failure = providerReason(provider, profile, constraints)
                if (failure == null) providers += provider else pairReasons[runtime.id to provider] = failure
            }
            qualified[runtime.id] = providers
        }

        val byId = runtimes.associateBy { it.id }
        val targets = runtimes.mapNotNull { it.fallback }.toSet()
        var root: String? = null
        var candidates: List<Pair<String, String>> = emptyList()
        for (runtime in runtimes) {
            if (runtime.id in targets) continue
            val chain = mutableListOf(runtime.id)
            while (true) chain += byId.getValue(chain.last()).fallback ?: break
            val pairs = chain.flatMap { id -> qualified[id].orEmpty().map { id to it } }
            if (pairs.isNotEmpty()) {
                root = runtime.id
                candidates = pairs
                break
            }
        }
        val bound = constraints.maxSteps.toInt()
        val used = candidates.take(bound)
        candidates.drop(bound).forEach { pairReasons[it] = ExecutionPlanReason.FALLBACK_BOUND_REACHED }
        for ((id, providers) in qualified) {
            for (provider in providers) {
                val pair = id to provider
                if (pair !in used && pair !in pairReasons) pairReasons[pair] = ExecutionPlanReason.NOT_DECLARED_FALLBACK
            }
        }

        val skipped = mutableListOf<ExecutionPlanSkip>()
        for (runtime in runtimes) {
            val exclusion = exclusions[runtime.id]
            if (exclusion != null) {
                skipped += ExecutionPlanSkip(runtime.id, null, exclusion)
                continue
            }
            for (provider in metas.getValue(runtime.id).providersOn(profile.platform)) {
                pairReasons[runtime.id to provider]?.let { skipped += ExecutionPlanSkip(runtime.id, provider, it) }
            }
        }
        if (used.isEmpty()) return result(ExecutionPlanReason.NO_ELIGIBLE_RUNTIME, skipped = skipped)

        val assumed = mutableSetOf<String>()
        val steps = used.mapIndexed { position, (id, provider) ->
            val runtime = byId.getValue(id)
            val meta = metas.getValue(id)
            val role = when {
                position == 0 -> "selected"
                used[position - 1].first == id -> "provider_fallback"
                else -> "artifact_fallback"
            }
            if (memoryLimit == null && runtime.minMemory > 0) assumed += "memory_budget_unknown"
            if (profile.storageBudgetBytes == null && runtime.minStorage > 0) assumed += "storage_budget_unknown"
            if (meta.tier != null && profile.maxModelTier == null) assumed += "model_tier_limit_unknown"
            if (constraints.minSequenceLength != null && profile.maxSequenceLength == null) {
                assumed += "device_sequence_limit_unknown"
            }
            ExecutionPlanStep(
                runtimeId = id,
                runtimeIndex = runtimes.indexOf(runtime),
                format = runtime.format,
                precision = runtime.precision,
                provider = provider,
                accelerator = DeviceCapabilityVocabulary.PROVIDER_ACCELERATORS.getValue(provider),
                artifactSha256 = digests.getValue(runtime.entrypoint),
                minMemoryBytes = runtime.minMemory,
                maxSequenceLength = meta.maxSequenceLength,
                tier = meta.tier,
                evidence = meta.evidence,
                role = role,
            )
        }
        return result(null, root, steps, skipped, assumed)
    }

    /**
     * Return which inputs changed since [plan] was resolved, sorted. Re-probe
     * the device and re-verify the bundle first; a cached plan may be reused
     * only when the result is empty.
     */
    fun invalidation(
        plan: ExecutionPlan,
        manifest: ByteArray,
        verification: PortableBundleVerification,
        profile: DeviceCapabilityProfile,
        constraints: ExecutionConstraints = ExecutionConstraints(),
    ): List<String> = buildList {
        if (plan.bundleContentId != PortableBundle.contentId(manifest)) add("bundle")
        if (plan.constraintsDigest != constraints.digest) add("constraints")
        if (plan.profileDigest != profile.digest) add("profile")
        val expected = planKey(plan.bundleContentId, plan.profileDigest, plan.constraintsDigest, plan.verificationDigest)
        if (plan.planKey != expected) add("resolver")
        if (plan.verificationDigest != verificationDigest(verification)) add("verification")
    }

    private fun parseManifest(manifest: ByteArray): Json.Obj? {
        val value = try {
            Json.parse(manifest)
        } catch (error: Json.Invalid) {
            return null
        }
        if (value !is Json.Obj) return null
        val canonical = try {
            Json.encode(value)
        } catch (error: Json.Invalid) {
            return null
        }
        if (!canonical.contentEquals(manifest)) return null
        val critical = value.strings("critical_extensions")?.toSet() ?: emptySet()
        return try {
            PortableBundle.validateManifest(manifest, critical)
            value
        } catch (error: PortableBundleException) {
            null
        }
    }

    private fun readRuntimes(value: Json.Obj): List<Runtime> = (value["runtimes"] as Json.Arr).items.map {
        val item = it as Json.Obj
        Runtime(
            id = item.string("id").orEmpty(),
            format = item.string("format").orEmpty(),
            precision = item.string("precision").orEmpty(),
            entrypoint = item.string("entrypoint").orEmpty(),
            files = item.strings("files").orEmpty(),
            platforms = (item["platforms"] as Json.Arr).items.map { entry ->
                val platform = entry as Json.Obj
                Triple(
                    platform.string("platform").orEmpty(),
                    platform.string("min_os"),
                    platform.string("max_os_exclusive"),
                )
            },
            minMemory = item.long("min_memory_bytes") ?: 0,
            minStorage = item.long("min_storage_bytes") ?: 0,
            fallback = item.string("fallback"),
        )
    }

    private fun tokens(value: Json?, maximum: Int): List<String> {
        val items = (value as? Json.Arr)?.items ?: throw MetadataInvalid()
        val strings = items.mapNotNull { (it as? Json.Str)?.value }
        if (strings.size != items.size || strings.size > maximum || strings.distinct().size != strings.size ||
            !strings.all { Patterns.TOKEN.matches(it) }
        ) {
            throw MetadataInvalid()
        }
        return strings
    }

    private fun bounded(value: Json?, maximum: Long): Long? = when (value) {
        Json.Null -> null
        is Json.Num -> if (value.value in 1..maximum) value.value else throw MetadataInvalid()
        else -> throw MetadataInvalid()
    }

    private fun parseExtension(value: Json.Obj, runtimes: List<Runtime>): Map<String, RuntimeMeta> {
        val evidenceKinds = (value["evidence"] as Json.Arr).items.mapNotNull { (it as? Json.Obj)?.string("kind") }.toSet()
        val ids = runtimes.map { it.id }.toSet()
        var declared: Json.Obj? = null
        val raw = (value["extensions"] as Json.Obj)[EXTENSION_NAME]
        if (raw != null) {
            val entries = (raw as? Json.Obj)?.get("runtimes") as? Json.Obj
            if (raw !is Json.Obj || !raw.hasExactKeys("runtimes", "schema_version") ||
                raw["schema_version"] != Json.Num(1) || entries == null || !entries.members.all { it.first in ids } ||
                entries.members.map { it.first }.distinct().size != entries.members.size
            ) {
                throw MetadataInvalid()
            }
            declared = entries
        }
        return runtimes.associate { runtime ->
            val entry = declared?.get(runtime.id)
            if (entry == null) {
                val defaults = DEFAULT_PROVIDERS[runtime.format].orEmpty()
                return@associate runtime.id to RuntimeMeta(defaults, emptyList(), emptyList(), null, null, null, false)
            }
            if (entry !is Json.Obj || !entry.hasExactKeys(*META_KEYS)) throw MetadataInvalid()
            val providers = tokens(entry["providers"], 8)
            val allowed = FORMAT_PROVIDERS[runtime.format].orEmpty()
            if (providers.isEmpty() || !providers.all { it in allowed }) throw MetadataInvalid()
            val features = tokens(entry["features"], 16)
            val evidence = tokens(entry["evidence"], 16)
            if (!evidence.all { it in evidenceKinds }) throw MetadataInvalid()
            val tier = when (val item = entry["tier"]) {
                Json.Null -> null
                is Json.Str -> item.value.takeIf { DeviceCapabilityVocabulary.tierRank(it) >= 0 } ?: throw MetadataInvalid()
                else -> throw MetadataInvalid()
            }
            runtime.id to RuntimeMeta(
                providers = providers,
                features = features,
                evidence = evidence.sorted(),
                maxSequenceLength = bounded(entry["max_sequence_length"], DeviceCapabilityVocabulary.MAX_SEQUENCE_LENGTH),
                minOperatorSet = bounded(entry["min_operator_set"], DeviceCapabilityVocabulary.MAX_OPERATOR_SET),
                tier = tier,
                declared = true,
            )
        }
    }

    private fun versionKey(value: String): List<Long> {
        val parts = value.split(".").map { it.toLong() }
        return parts + List(4 - parts.size) { 0L }
    }

    private fun compareVersions(left: String, right: String): Int {
        for ((a, b) in versionKey(left).zip(versionKey(right))) {
            if (a != b) return if (a < b) -1 else 1
        }
        return 0
    }

    private fun sdkCompatible(value: Json.Obj, profile: DeviceCapabilityProfile): Boolean {
        val sdks = ((value["compatibility"] as Json.Obj)["sdks"] as Json.Arr).items.map { it as Json.Obj }
        val sdk = sdks.firstOrNull { it.string("name") == profile.sdkName } ?: return false
        val maximum = sdk.string("max_exclusive")
        return compareVersions(profile.sdkVersion, sdk.string("min").orEmpty()) >= 0 &&
            (maximum == null || compareVersions(profile.sdkVersion, maximum) < 0)
    }

    private fun runtimeReason(
        index: Int,
        runtime: Runtime,
        meta: RuntimeMeta,
        digests: Map<String, String>,
        verification: PortableBundleVerification,
        profile: DeviceCapabilityProfile,
        constraints: ExecutionConstraints,
        memoryLimit: Long?,
    ): ExecutionPlanReason? {
        if (index !in verification.eligibleRuntimes) return ExecutionPlanReason.RUNTIME_NOT_ELIGIBLE
        if ((listOf(runtime.entrypoint) + runtime.files).any { digests[it] in constraints.revokedArtifacts }) {
            return ExecutionPlanReason.ARTIFACT_REVOKED
        }
        if (runtime.format !in profile.runtimes) return ExecutionPlanReason.FORMAT_UNSUPPORTED
        val platform = runtime.platforms.firstOrNull { it.first == profile.platform }
            ?: return ExecutionPlanReason.PLATFORM_UNSUPPORTED
        val (_, low, high) = platform
        if (low != null || high != null) {
            val os = profile.osVersion ?: return ExecutionPlanReason.OS_VERSION_UNKNOWN
            if ((low != null && compareVersions(os, low) < 0) || (high != null && compareVersions(os, high) >= 0)) {
                return ExecutionPlanReason.OS_VERSION_UNSUPPORTED
            }
        }
        if (runtime.precision !in profile.precisions) return ExecutionPlanReason.PRECISION_UNSUPPORTED
        val allowed = constraints.allowedPrecisions
        if (allowed != null && runtime.precision !in allowed) return ExecutionPlanReason.PRECISION_DISALLOWED
        if (memoryLimit != null && runtime.minMemory > memoryLimit) return ExecutionPlanReason.MEMORY_INSUFFICIENT
        val storage = profile.storageBudgetBytes
        if (storage != null && runtime.minStorage > storage) return ExecutionPlanReason.STORAGE_INSUFFICIENT
        val ceiling = profile.maxModelTier
        if (meta.tier != null && ceiling != null &&
            DeviceCapabilityVocabulary.tierRank(meta.tier) > DeviceCapabilityVocabulary.tierRank(ceiling)
        ) {
            return ExecutionPlanReason.MODEL_TIER_EXCEEDED
        }
        val required = constraints.minSequenceLength
        if (required != null) {
            val device = profile.maxSequenceLength
            if (device != null && device < required) return ExecutionPlanReason.SEQUENCE_LIMIT_EXCEEDED
            val declared = meta.maxSequenceLength ?: return ExecutionPlanReason.SEQUENCE_LIMIT_UNDECLARED
            if (declared < required) return ExecutionPlanReason.SEQUENCE_LIMIT_EXCEEDED
        }
        val minimum = meta.minOperatorSet
        if (minimum != null) {
            val supported = profile.operatorSets[runtime.format] ?: return ExecutionPlanReason.OPERATOR_SET_UNKNOWN
            if (supported < minimum) return ExecutionPlanReason.OPERATOR_SET_UNSUPPORTED
        }
        if (!profile.features.containsAll(meta.features)) return ExecutionPlanReason.FEATURE_UNAVAILABLE
        if (!meta.evidence.containsAll(constraints.requiredEvidence)) return ExecutionPlanReason.EVIDENCE_MISSING
        return null
    }

    private fun providerReason(
        provider: String,
        profile: DeviceCapabilityProfile,
        constraints: ExecutionConstraints,
    ): ExecutionPlanReason? = when {
        provider !in profile.providers -> ExecutionPlanReason.PROVIDER_UNAVAILABLE
        provider in constraints.disallowedProviders -> ExecutionPlanReason.PROVIDER_DISALLOWED
        constraints.energy == "low_power" && DeviceCapabilityVocabulary.PROVIDER_ACCELERATORS[provider] == "gpu" ->
            ExecutionPlanReason.ENERGY_POLICY_EXCLUDED
        else -> null
    }
}
