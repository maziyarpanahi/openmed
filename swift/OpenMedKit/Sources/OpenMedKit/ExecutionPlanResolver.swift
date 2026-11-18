import Foundation

/// Stable, value-free execution-plan reason codes, in evaluation order.
public enum ExecutionPlanReason: String, CaseIterable, Sendable {
    case remoteExecutionUnsupported = "remote_execution_unsupported"
    case bundleRevoked = "bundle_revoked"
    case bundleUnverified = "bundle_unverified"
    case bundleFilesUnverified = "bundle_files_unverified"
    case sdkIncompatible = "sdk_incompatible"
    case executionMetadataInvalid = "execution_metadata_invalid"
    case noEligibleRuntime = "no_eligible_runtime"
    case runtimeNotEligible = "runtime_not_eligible"
    case artifactRevoked = "artifact_revoked"
    case formatUnsupported = "format_unsupported"
    case platformUnsupported = "platform_unsupported"
    case osVersionUnknown = "os_version_unknown"
    case osVersionUnsupported = "os_version_unsupported"
    case precisionUnsupported = "precision_unsupported"
    case precisionDisallowed = "precision_disallowed"
    case memoryInsufficient = "memory_insufficient"
    case storageInsufficient = "storage_insufficient"
    case modelTierExceeded = "model_tier_exceeded"
    case sequenceLimitExceeded = "sequence_limit_exceeded"
    case sequenceLimitUndeclared = "sequence_limit_undeclared"
    case operatorSetUnknown = "operator_set_unknown"
    case operatorSetUnsupported = "operator_set_unsupported"
    case featureUnavailable = "feature_unavailable"
    case evidenceMissing = "evidence_missing"
    case providerUnavailable = "provider_unavailable"
    case providerDisallowed = "provider_disallowed"
    case energyPolicyExcluded = "energy_policy_excluded"
    case notDeclaredFallback = "not_declared_fallback"
    case fallbackBoundReached = "fallback_bound_reached"
}

/// Caller policy applied on top of the bundle and the device profile.
///
/// `localOnly` must be `true`: version 1 plans never contain a remote or cloud
/// step, so `false` resolves to `remote_execution_unsupported`.
public struct ExecutionConstraints: Equatable, Sendable {
    public static let schema = "openmed.execution_constraints"
    public static let defaultMaxSteps: Int64 = 4
    public static let maxStepsLimit: Int64 = 16
    public static let maxDeadlineMilliseconds: Int64 = 86_400_000

    public let localOnly: Bool
    public let allowedPrecisions: [String]?
    public let disallowedProviders: [String]
    public let maxMemoryBytes: Int64?
    public let minSequenceLength: Int64?
    /// `balanced` or `low_power`; `low_power` excludes GPU providers.
    public let energy: String
    public let requiredEvidence: [String]
    /// `sha256:` file digests no step may use.
    public let revokedArtifacts: [String]
    public let maxSteps: Int64
    /// Wall-clock budget the executor applies across steps; never benchmarked.
    public let deadlineMilliseconds: Int64?
    public let requireVerifiedFiles: Bool

    private static let documentKeys = [
        "allowed_precisions", "deadline_ms", "disallowed_providers", "energy", "local_only", "max_memory_bytes",
        "max_steps", "min_sequence_length", "require_verified_files", "required_evidence", "revoked_artifacts",
        "schema", "schema_version",
    ]
    private static let sha256 = PortableBundlePattern("sha256:[0-9a-f]{64}")

    public init(
        localOnly: Bool = true, allowedPrecisions: [String]? = nil, disallowedProviders: [String] = [],
        maxMemoryBytes: Int64? = nil, minSequenceLength: Int64? = nil, energy: String = "balanced",
        requiredEvidence: [String] = [], revokedArtifacts: [String] = [], maxSteps: Int64 = 4,
        deadlineMilliseconds: Int64? = nil, requireVerifiedFiles: Bool = true
    ) throws {
        let fail = DeviceCapabilityError.constraintsInvalid
        guard ["balanced", "low_power"].contains(energy),
            maxMemoryBytes.map({ $0 >= 0 && $0 <= PortableBundle.maxSafeInteger }) ?? true,
            minSequenceLength.map({ $0 > 0 && $0 <= DeviceCapabilityVocabulary.maxSequenceLength }) ?? true,
            maxSteps >= 1, maxSteps <= Self.maxStepsLimit,
            deadlineMilliseconds.map({ $0 > 0 && $0 <= Self.maxDeadlineMilliseconds }) ?? true,
            (allowedPrecisions ?? []).allSatisfy({ PortableBundle.precisions.contains($0) }),
            disallowedProviders.allSatisfy({ DeviceCapabilityVocabulary.providerAccelerators[$0] != nil }),
            requiredEvidence.allSatisfy({ PortableBundlePattern.token.matches($0) }),
            revokedArtifacts.allSatisfy({ Self.sha256.matches($0) })
        else { throw fail }
        let set = { (items: [String]) in Set(items).sorted(by: PortableBundle.utf8Less) }
        self.localOnly = localOnly
        self.allowedPrecisions = allowedPrecisions.map(set)
        self.disallowedProviders = set(disallowedProviders)
        self.maxMemoryBytes = maxMemoryBytes
        self.minSequenceLength = minSequenceLength
        self.energy = energy
        self.requiredEvidence = set(requiredEvidence)
        self.revokedArtifacts = set(revokedArtifacts)
        self.maxSteps = maxSteps
        self.deadlineMilliseconds = deadlineMilliseconds
        self.requireVerifiedFiles = requireVerifiedFiles
    }

    /// Default constraints: local only, verified files, four steps.
    public static let defaults = ExecutionConstraints(defaultsOnly: ())

    private init(defaultsOnly: Void) {
        localOnly = true
        allowedPrecisions = nil
        disallowedProviders = []
        maxMemoryBytes = nil
        minSequenceLength = nil
        energy = "balanced"
        requiredEvidence = []
        revokedArtifacts = []
        maxSteps = Self.defaultMaxSteps
        deadlineMilliseconds = nil
        requireVerifiedFiles = true
    }

    /// Parses a constraints document strictly; unknown or missing members fail.
    public init(json: Data) throws {
        let fail = DeviceCapabilityError.constraintsInvalid
        guard let value = try? PortableBundleJSON.parse(json), case .object(let members) = value,
            PortableBundleJSON.hasExactKeys(members, Self.documentKeys),
            value.plan("schema")?.planString == Self.schema, value.plan("schema_version")?.planInt == 1,
            let localOnly = value.plan("local_only")?.planBool,
            let requireFiles = value.plan("require_verified_files")?.planBool,
            let energy = value.plan("energy")?.planString,
            let disallowed = value.plan("disallowed_providers")?.planStrings,
            let evidence = value.plan("required_evidence")?.planStrings,
            let revoked = value.plan("revoked_artifacts")?.planStrings,
            let maxSteps = value.plan("max_steps")?.planInt
        else { throw fail }
        let optionalInt = { (key: String) throws -> Int64? in
            guard let item = value.plan(key) else { throw fail }
            if item.planIsNull { return nil }
            guard let number = item.planInt else { throw fail }
            return number
        }
        var allowed: [String]?
        if let item = value.plan("allowed_precisions"), !item.planIsNull {
            guard let strings = item.planStrings else { throw fail }
            allowed = strings
        }
        try self.init(
            localOnly: localOnly, allowedPrecisions: allowed, disallowedProviders: disallowed,
            maxMemoryBytes: try optionalInt("max_memory_bytes"), minSequenceLength: try optionalInt("min_sequence_length"),
            energy: energy, requiredEvidence: evidence, revokedArtifacts: revoked, maxSteps: maxSteps,
            deadlineMilliseconds: try optionalInt("deadline_ms"), requireVerifiedFiles: requireFiles)
    }

    func document() -> PortableBundleJSON {
        let strings = { (items: [String]) in PortableBundleJSON.array(items.map { .string($0) }) }
        let optionalInt = { (item: Int64?) in item.map { PortableBundleJSON.int($0) } ?? .null }
        return .object([
            ("allowed_precisions", allowedPrecisions.map(strings) ?? .null),
            ("deadline_ms", optionalInt(deadlineMilliseconds)),
            ("disallowed_providers", strings(disallowedProviders)),
            ("energy", .string(energy)),
            ("local_only", .bool(localOnly)),
            ("max_memory_bytes", optionalInt(maxMemoryBytes)),
            ("max_steps", .int(maxSteps)),
            ("min_sequence_length", optionalInt(minSequenceLength)),
            ("require_verified_files", .bool(requireVerifiedFiles)),
            ("required_evidence", strings(requiredEvidence)),
            ("revoked_artifacts", strings(revokedArtifacts)),
            ("schema", .string(Self.schema)),
            ("schema_version", .int(1)),
        ])
    }

    /// The canonical constraints document bytes.
    public func canonicalJSON() -> Data { ExecutionPlanResolver.canonical(document()) }

    /// `sha256:` digest of the canonical constraints document.
    public var digest: String { ExecutionPlanResolver.digest(document()) }
}

/// One `(runtime, provider)` attempt in an execution plan.
public struct ExecutionPlanStep: Equatable, Sendable {
    public let runtimeID: String
    public let runtimeIndex: Int
    public let format: String
    public let precision: String
    public let provider: String
    public let accelerator: String
    /// Digest of the runtime entrypoint file.
    public let artifactSHA256: String
    public let minMemoryBytes: Int64
    public let maxSequenceLength: Int64?
    public let tier: String?
    /// Evidence kinds that qualify the runtime.
    public let evidence: [String]
    /// `selected`, `provider_fallback`, or `artifact_fallback`.
    public let role: String
}

/// A runtime, or `(runtime, provider)` pair, the plan does not use.
public struct ExecutionPlanSkip: Equatable, Sendable {
    public let runtimeID: String
    public let provider: String?
    public let reason: ExecutionPlanReason
}

/// Value-free resolver result (`execution_plan`).
public struct ExecutionPlan: Equatable, Sendable {
    /// `planned` or `unsupported`.
    public let status: String
    public let reason: ExecutionPlanReason?
    public let bundleContentID: String
    public let profileDigest: String
    public let constraintsDigest: String
    public let verificationDigest: String
    /// Invalidation key over every input digest and the resolver version.
    public let planKey: String
    public let rootRuntime: String?
    public let steps: [ExecutionPlanStep]
    public let skipped: [ExecutionPlanSkip]
    /// Limits that were unknown and therefore not enforced.
    public let assumptions: [String]
    public let maxSteps: Int64
    public let deadlineMilliseconds: Int64?
    public let maxMemoryBytes: Int64?

    public var isSupported: Bool { status == "planned" }

    func document() -> PortableBundleJSON {
        let strings = { (items: [String]) in PortableBundleJSON.array(items.map { .string($0) }) }
        let optionalString = { (item: String?) in item.map { PortableBundleJSON.string($0) } ?? .null }
        let optionalInt = { (item: Int64?) in item.map { PortableBundleJSON.int($0) } ?? .null }
        return .object([
            ("assumptions", strings(assumptions)),
            (
                "bounds",
                .object([
                    ("deadline_ms", optionalInt(deadlineMilliseconds)),
                    ("max_memory_bytes", optionalInt(maxMemoryBytes)),
                    ("max_steps", .int(maxSteps)),
                ])
            ),
            ("bundle_content_id", .string(bundleContentID)),
            ("constraints_digest", .string(constraintsDigest)),
            ("local_only", .bool(true)),
            ("plan_key", .string(planKey)),
            ("profile_digest", .string(profileDigest)),
            ("reason", optionalString(reason?.rawValue)),
            ("resolver_version", .int(ExecutionPlanResolver.resolverVersion)),
            ("root_runtime", optionalString(rootRuntime)),
            ("schema", .string(ExecutionPlanResolver.schema)),
            ("schema_version", .int(1)),
            (
                "skipped",
                .array(
                    skipped.map {
                        .object([
                            ("provider", optionalString($0.provider)), ("reason", .string($0.reason.rawValue)),
                            ("runtime_id", .string($0.runtimeID)),
                        ])
                    })
            ),
            ("status", .string(status)),
            (
                "steps",
                .array(
                    steps.map {
                        .object([
                            ("accelerator", .string($0.accelerator)),
                            ("artifact_sha256", .string($0.artifactSHA256)),
                            ("evidence", strings($0.evidence)),
                            ("format", .string($0.format)),
                            ("max_sequence_length", optionalInt($0.maxSequenceLength)),
                            ("min_memory_bytes", .int($0.minMemoryBytes)),
                            ("precision", .string($0.precision)),
                            ("provider", .string($0.provider)),
                            ("role", .string($0.role)),
                            ("runtime_id", .string($0.runtimeID)),
                            ("runtime_index", .int(Int64($0.runtimeIndex))),
                            ("tier", optionalString($0.tier)),
                        ])
                    })
            ),
            ("verification_digest", .string(verificationDigest)),
        ])
    }

    /// The canonical plan document bytes.
    public func canonicalJSON() -> Data { ExecutionPlanResolver.canonical(document()) }
}

/// Resolves deterministic execution plans for signed portable bundles.
///
/// Mirrors `openmed/core/execution_plan.py`: the same checks in the same order
/// produce byte-identical plans for `tests/fixtures/device_capabilities`.
/// Resolution is pure: it opens no network connection and loads no model.
public enum ExecutionPlanResolver {
    public static let schema = "openmed.execution_plan"
    public static let resolverVersion: Int64 = 1
    /// Manifest extension that declares providers and qualifications per runtime.
    public static let extensionName = "org.openmed.execution"
    /// Providers that can execute each bundle runtime format.
    public static let formatProviders: [String: Set<String>] = [
        "coreml": ["coreml", "cpu"],
        "mlx": ["cpu", "metal"],
        "onnx": ["coreml", "cpu", "nnapi", "qnn", "wasm", "webgpu"],
        "ort": ["coreml", "cpu", "nnapi", "qnn"],
        "pytorch": ["cpu"],
        "transformersjs": ["wasm", "webgpu"],
    ]
    /// Baseline providers for a runtime the execution extension does not describe.
    public static let defaultProviders: [String: [String]] = [
        "coreml": ["coreml"], "mlx": ["metal"], "onnx": ["cpu", "wasm"], "ort": ["cpu"], "pytorch": ["cpu"],
        "transformersjs": ["wasm"],
    ]
    public static let stepRoles = ["selected", "provider_fallback", "artifact_fallback"]
    public static let assumptions = [
        "device_sequence_limit_unknown", "memory_budget_unknown", "model_tier_limit_unknown",
        "storage_budget_unknown",
    ]
    public static let invalidationComponents = ["bundle", "constraints", "profile", "resolver", "verification"]

    static func canonical(_ value: PortableBundleJSON) -> Data {
        Data((try? PortableBundleJSON.encode(value)) ?? [])
    }

    static func digest(_ value: PortableBundleJSON) -> String {
        "sha256:" + PortableBundle.sha256Hex(canonical(value))
    }

    /// Digest of the verification facts a plan depends on.
    public static func verificationDigest(_ verification: PortableBundleVerification) -> String {
        digest(
            .object([
                ("content_id", verification.contentID.map { .string($0) } ?? .null),
                ("eligible_runtimes", .array(verification.eligibleRuntimes.map { .int(Int64($0)) })),
                ("files_verified", .bool(verification.filesVerified)),
                ("ok", .bool(verification.ok)),
                ("reason", verification.reason.map { .string($0.rawValue) } ?? .null),
            ]))
    }

    /// Plan invalidation key over the input digests and resolver version.
    public static func planKey(
        bundleContentID: String, profileDigest: String, constraintsDigest: String, verificationDigest: String,
        resolverVersion: Int64 = resolverVersion
    ) -> String {
        digest(
            .object([
                ("bundle_content_id", .string(bundleContentID)),
                ("constraints_digest", .string(constraintsDigest)),
                ("profile_digest", .string(profileDigest)),
                ("resolver_version", .int(resolverVersion)),
                ("verification_digest", .string(verificationDigest)),
            ]))
    }

    private struct RuntimeMeta {
        let providers: [String]
        let features: [String]
        let evidence: [String]
        let maxSequenceLength: Int64?
        let minOperatorSet: Int64?
        let tier: String?
        let declared: Bool

        func providers(on platform: String) -> [String] {
            if declared { return providers }
            let valid = DeviceCapabilityVocabulary.platformProviders[platform] ?? []
            return providers.filter { valid.contains($0) }
        }
    }

    private struct MetadataInvalid: Error {}

    private struct Runtime {
        let id: String
        let format: String
        let precision: String
        let entrypoint: String
        let files: [String]
        let platforms: [(name: String, minOS: String?, maxOS: String?)]
        let minMemory: Int64
        let minStorage: Int64
        let fallback: String?
    }

    /// Resolves an ordered, value-free plan or an explicit refusal.
    ///
    /// The plan follows exactly one declared fallback chain: the first root
    /// runtime (in manifest order) whose chain contains a qualified step.
    /// Within a runtime, providers keep the signed extension's order.
    public static func resolve(
        manifest: Data, verification: PortableBundleVerification, profile: DeviceCapabilityProfile,
        constraints: ExecutionConstraints = .defaults
    ) -> ExecutionPlan {
        let bundleID = PortableBundle.contentID(of: manifest)
        let profileDigest = profile.digest
        let constraintsDigest = constraints.digest
        let verificationValue = verificationDigest(verification)
        let key = planKey(
            bundleContentID: bundleID, profileDigest: profileDigest, constraintsDigest: constraintsDigest,
            verificationDigest: verificationValue)
        let memoryLimit = [profile.memoryBudgetBytes, constraints.maxMemoryBytes].compactMap { $0 }.min()

        func result(
            _ reason: ExecutionPlanReason?, root: String? = nil, steps: [ExecutionPlanStep] = [],
            skipped: [ExecutionPlanSkip] = [], assumptions: Set<String> = []
        ) -> ExecutionPlan {
            ExecutionPlan(
                status: reason == nil ? "planned" : "unsupported", reason: reason, bundleContentID: bundleID,
                profileDigest: profileDigest, constraintsDigest: constraintsDigest,
                verificationDigest: verificationValue, planKey: key, rootRuntime: root, steps: steps,
                skipped: skipped, assumptions: assumptions.sorted(by: PortableBundle.utf8Less),
                maxSteps: constraints.maxSteps, deadlineMilliseconds: constraints.deadlineMilliseconds,
                maxMemoryBytes: memoryLimit)
        }

        guard constraints.localOnly else { return result(.remoteExecutionUnsupported) }
        guard verification.ok else {
            return result(verification.reason == .bundleRevoked ? .bundleRevoked : .bundleUnverified)
        }
        guard verification.contentID == bundleID, let value = parseManifest(manifest) else {
            return result(.bundleUnverified)
        }
        if constraints.requireVerifiedFiles && !verification.filesVerified { return result(.bundleFilesUnverified) }
        guard sdkCompatible(value, profile: profile) else { return result(.sdkIncompatible) }
        guard let runtimes = readRuntimes(value), let metas = try? parseExtension(value, runtimes: runtimes) else {
            return result(.executionMetadataInvalid)
        }
        var digests: [String: String] = [:]
        for item in value.plan("files")?.planArray ?? [] {
            if let path = item.plan("path")?.planString, let sha = item.plan("sha256")?.planString {
                digests[path] = sha
            }
        }

        var exclusions: [String: ExecutionPlanReason] = [:]
        var pairReasons: [String: ExecutionPlanReason] = [:]
        var qualified: [String: [String]] = [:]
        let pairKey = { (runtime: String, provider: String) in runtime + "\u{0}" + provider }
        for (index, runtime) in runtimes.enumerated() {
            let meta = metas[runtime.id]!
            if let reason = runtimeReason(
                index: index, runtime: runtime, meta: meta, digests: digests, verification: verification,
                profile: profile, constraints: constraints, memoryLimit: memoryLimit)
            {
                exclusions[runtime.id] = reason
                continue
            }
            var providers: [String] = []
            for provider in meta.providers(on: profile.platform) {
                if let reason = providerReason(provider, profile: profile, constraints: constraints) {
                    pairReasons[pairKey(runtime.id, provider)] = reason
                } else {
                    providers.append(provider)
                }
            }
            qualified[runtime.id] = providers
        }

        let byID = Dictionary(uniqueKeysWithValues: runtimes.map { ($0.id, $0) })
        let targets = Set(runtimes.compactMap(\.fallback))
        var root: String?
        var candidates: [(String, String)] = []
        for runtime in runtimes where !targets.contains(runtime.id) {
            var chain = [runtime.id]
            while let next = byID[chain[chain.count - 1]]?.fallback { chain.append(next) }
            let pairs = chain.flatMap { id in (qualified[id] ?? []).map { (id, $0) } }
            if !pairs.isEmpty {
                root = runtime.id
                candidates = pairs
                break
            }
        }
        let bound = Int(constraints.maxSteps)
        let used = Array(candidates.prefix(bound))
        let usedKeys = Set(used.map { pairKey($0.0, $0.1) })
        for pair in candidates.dropFirst(bound) { pairReasons[pairKey(pair.0, pair.1)] = .fallbackBoundReached }
        for (id, providers) in qualified {
            for provider in providers where !usedKeys.contains(pairKey(id, provider)) {
                if pairReasons[pairKey(id, provider)] == nil {
                    pairReasons[pairKey(id, provider)] = .notDeclaredFallback
                }
            }
        }

        var skipped: [ExecutionPlanSkip] = []
        for runtime in runtimes {
            if let exclusion = exclusions[runtime.id] {
                skipped.append(ExecutionPlanSkip(runtimeID: runtime.id, provider: nil, reason: exclusion))
                continue
            }
            for provider in metas[runtime.id]!.providers(on: profile.platform) {
                if let reason = pairReasons[pairKey(runtime.id, provider)] {
                    skipped.append(ExecutionPlanSkip(runtimeID: runtime.id, provider: provider, reason: reason))
                }
            }
        }
        guard !used.isEmpty else { return result(.noEligibleRuntime, skipped: skipped) }

        var steps: [ExecutionPlanStep] = []
        var assumed = Set<String>()
        for (position, pair) in used.enumerated() {
            let runtime = byID[pair.0]!
            let meta = metas[pair.0]!
            let role =
                position == 0 ? "selected" : (used[position - 1].0 == pair.0 ? "provider_fallback" : "artifact_fallback")
            if memoryLimit == nil && runtime.minMemory > 0 { assumed.insert("memory_budget_unknown") }
            if profile.storageBudgetBytes == nil && runtime.minStorage > 0 { assumed.insert("storage_budget_unknown") }
            if meta.tier != nil && profile.maxModelTier == nil { assumed.insert("model_tier_limit_unknown") }
            if constraints.minSequenceLength != nil && profile.maxSequenceLength == nil {
                assumed.insert("device_sequence_limit_unknown")
            }
            steps.append(
                ExecutionPlanStep(
                    runtimeID: runtime.id, runtimeIndex: runtimes.firstIndex(where: { $0.id == runtime.id }) ?? 0,
                    format: runtime.format, precision: runtime.precision, provider: pair.1,
                    accelerator: DeviceCapabilityVocabulary.providerAccelerators[pair.1] ?? "cpu",
                    artifactSHA256: digests[runtime.entrypoint] ?? "", minMemoryBytes: runtime.minMemory,
                    maxSequenceLength: meta.maxSequenceLength, tier: meta.tier, evidence: meta.evidence, role: role))
        }
        return result(nil, root: root, steps: steps, skipped: skipped, assumptions: assumed)
    }

    /// Returns which inputs changed since `plan` was resolved, sorted.
    ///
    /// Re-probe the device and re-verify the bundle before calling this; a
    /// cached plan may be reused only when the result is empty.
    public static func invalidation(
        of plan: ExecutionPlan, manifest: Data, verification: PortableBundleVerification,
        profile: DeviceCapabilityProfile, constraints: ExecutionConstraints = .defaults
    ) -> [String] {
        var changed: [String] = []
        if plan.bundleContentID != PortableBundle.contentID(of: manifest) { changed.append("bundle") }
        if plan.constraintsDigest != constraints.digest { changed.append("constraints") }
        if plan.profileDigest != profile.digest { changed.append("profile") }
        let expected = planKey(
            bundleContentID: plan.bundleContentID, profileDigest: plan.profileDigest,
            constraintsDigest: plan.constraintsDigest, verificationDigest: plan.verificationDigest)
        if plan.planKey != expected { changed.append("resolver") }
        if plan.verificationDigest != verificationDigest(verification) { changed.append("verification") }
        return changed
    }

    private static func parseManifest(_ manifest: Data) -> PortableBundleJSON? {
        guard let value = try? PortableBundle.parseCanonical(manifest) else { return nil }
        let critical = Set(value.plan("critical_extensions")?.planStrings ?? [])
        guard (try? PortableBundle.validate(value, understood: critical)) != nil else { return nil }
        return value
    }

    private static func readRuntimes(_ value: PortableBundleJSON) -> [Runtime]? {
        var runtimes: [Runtime] = []
        for item in value.plan("runtimes")?.planArray ?? [] {
            guard let id = item.plan("id")?.planString, let format = item.plan("format")?.planString,
                let precision = item.plan("precision")?.planString,
                let entrypoint = item.plan("entrypoint")?.planString,
                let files = item.plan("files")?.planStrings,
                let minMemory = item.plan("min_memory_bytes")?.planInt,
                let minStorage = item.plan("min_storage_bytes")?.planInt
            else { return nil }
            let platforms = (item.plan("platforms")?.planArray ?? []).compactMap {
                entry -> (name: String, minOS: String?, maxOS: String?)? in
                guard let name = entry.plan("platform")?.planString else { return nil }
                return (name, entry.plan("min_os")?.planString, entry.plan("max_os_exclusive")?.planString)
            }
            runtimes.append(
                Runtime(
                    id: id, format: format, precision: precision, entrypoint: entrypoint, files: files,
                    platforms: platforms, minMemory: minMemory, minStorage: minStorage,
                    fallback: item.plan("fallback")?.planString))
        }
        return runtimes
    }

    private static func tokens(_ value: PortableBundleJSON?, maximum: Int) throws -> [String] {
        guard let items = value?.planStrings, items.count <= maximum,
            items.allSatisfy({ PortableBundlePattern.token.matches($0) }), Set(items).count == items.count
        else { throw MetadataInvalid() }
        return items
    }

    private static func bounded(_ value: PortableBundleJSON?, maximum: Int64) throws -> Int64? {
        guard let value else { throw MetadataInvalid() }
        if value.planIsNull { return nil }
        guard let number = value.planInt, number > 0, number <= maximum else { throw MetadataInvalid() }
        return number
    }

    private static func parseExtension(_ value: PortableBundleJSON, runtimes: [Runtime]) throws -> [String: RuntimeMeta] {
        let evidenceKinds = Set((value.plan("evidence")?.planArray ?? []).compactMap { $0.plan("kind")?.planString })
        let ids = Set(runtimes.map(\.id))
        var declared: [(String, PortableBundleJSON)] = []
        if let raw = value.plan("extensions")?.plan(extensionName) {
            guard case .object(let members) = raw, PortableBundleJSON.hasExactKeys(members, ["runtimes", "schema_version"]),
                raw.plan("schema_version")?.planInt == 1, case .object(let entries)? = raw.plan("runtimes"),
                entries.allSatisfy({ ids.contains($0.0) })
            else { throw MetadataInvalid() }
            declared = entries
        }
        var metas: [String: RuntimeMeta] = [:]
        for runtime in runtimes {
            guard let entry = PortableBundleJSON.member(declared, runtime.id) else {
                metas[runtime.id] = RuntimeMeta(
                    providers: defaultProviders[runtime.format] ?? [], features: [], evidence: [],
                    maxSequenceLength: nil, minOperatorSet: nil, tier: nil, declared: false)
                continue
            }
            guard case .object(let members) = entry,
                PortableBundleJSON.hasExactKeys(
                    members, ["evidence", "features", "max_sequence_length", "min_operator_set", "providers", "tier"])
            else { throw MetadataInvalid() }
            let providers = try tokens(entry.plan("providers"), maximum: 8)
            let allowed = formatProviders[runtime.format] ?? []
            guard !providers.isEmpty, providers.allSatisfy({ allowed.contains($0) }) else { throw MetadataInvalid() }
            let features = try tokens(entry.plan("features"), maximum: 16)
            let evidence = try tokens(entry.plan("evidence"), maximum: 16)
            guard evidence.allSatisfy({ evidenceKinds.contains($0) }) else { throw MetadataInvalid() }
            var tier: String?
            if let item = entry.plan("tier"), !item.planIsNull {
                guard let text = item.planString, DeviceCapabilityVocabulary.tierRank(text) != nil else {
                    throw MetadataInvalid()
                }
                tier = text
            }
            metas[runtime.id] = RuntimeMeta(
                providers: providers, features: features, evidence: evidence.sorted(by: PortableBundle.utf8Less),
                maxSequenceLength: try bounded(
                    entry.plan("max_sequence_length"), maximum: DeviceCapabilityVocabulary.maxSequenceLength),
                minOperatorSet: try bounded(entry.plan("min_operator_set"), maximum: DeviceCapabilityVocabulary.maxOperatorSet),
                tier: tier, declared: true)
        }
        return metas
    }

    private static func sdkCompatible(_ value: PortableBundleJSON, profile: DeviceCapabilityProfile) -> Bool {
        for sdk in value.plan("compatibility")?.plan("sdks")?.planArray ?? [] {
            guard sdk.plan("name")?.planString == profile.sdkName, let minimum = sdk.plan("min")?.planString else {
                continue
            }
            let maximum = sdk.plan("max_exclusive")?.planString
            return PortableBundle.compareVersions(profile.sdkVersion, minimum) >= 0
                && (maximum.map { PortableBundle.compareVersions(profile.sdkVersion, $0) < 0 } ?? true)
        }
        return false
    }

    private static func runtimeReason(
        index: Int, runtime: Runtime, meta: RuntimeMeta, digests: [String: String],
        verification: PortableBundleVerification, profile: DeviceCapabilityProfile, constraints: ExecutionConstraints,
        memoryLimit: Int64?
    ) -> ExecutionPlanReason? {
        guard verification.eligibleRuntimes.contains(index) else { return .runtimeNotEligible }
        let paths = [runtime.entrypoint] + runtime.files
        if paths.contains(where: { constraints.revokedArtifacts.contains(digests[$0] ?? "") }) {
            return .artifactRevoked
        }
        guard profile.runtimes.contains(runtime.format) else { return .formatUnsupported }
        guard let platform = runtime.platforms.first(where: { $0.name == profile.platform }) else {
            return .platformUnsupported
        }
        if platform.minOS != nil || platform.maxOS != nil {
            guard let os = profile.osVersion else { return .osVersionUnknown }
            if (platform.minOS.map { PortableBundle.compareVersions(os, $0) < 0 } ?? false)
                || (platform.maxOS.map { PortableBundle.compareVersions(os, $0) >= 0 } ?? false)
            {
                return .osVersionUnsupported
            }
        }
        guard profile.precisions.contains(runtime.precision) else { return .precisionUnsupported }
        if let allowed = constraints.allowedPrecisions, !allowed.contains(runtime.precision) {
            return .precisionDisallowed
        }
        if let limit = memoryLimit, runtime.minMemory > limit { return .memoryInsufficient }
        if let storage = profile.storageBudgetBytes, runtime.minStorage > storage { return .storageInsufficient }
        if let tier = meta.tier, let ceiling = profile.maxModelTier,
            let rank = DeviceCapabilityVocabulary.tierRank(tier), let maximum = DeviceCapabilityVocabulary.tierRank(ceiling),
            rank > maximum
        {
            return .modelTierExceeded
        }
        if let required = constraints.minSequenceLength {
            if let device = profile.maxSequenceLength, device < required { return .sequenceLimitExceeded }
            guard let declared = meta.maxSequenceLength else { return .sequenceLimitUndeclared }
            if declared < required { return .sequenceLimitExceeded }
        }
        if let minimum = meta.minOperatorSet {
            guard let supported = profile.operatorSets[runtime.format] else { return .operatorSetUnknown }
            if supported < minimum { return .operatorSetUnsupported }
        }
        guard meta.features.allSatisfy({ profile.features.contains($0) }) else { return .featureUnavailable }
        guard constraints.requiredEvidence.allSatisfy({ meta.evidence.contains($0) }) else { return .evidenceMissing }
        return nil
    }

    private static func providerReason(
        _ provider: String, profile: DeviceCapabilityProfile, constraints: ExecutionConstraints
    ) -> ExecutionPlanReason? {
        guard profile.providers.contains(provider) else { return .providerUnavailable }
        if constraints.disallowedProviders.contains(provider) { return .providerDisallowed }
        if constraints.energy == "low_power" && DeviceCapabilityVocabulary.providerAccelerators[provider] == "gpu" {
            return .energyPolicyExcluded
        }
        return nil
    }
}
